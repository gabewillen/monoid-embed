#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig, MonoidCpuConfig, MonoidCpuKernel


def _get_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if "model name" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return platform.processor() or "unknown"


def _get_git_info() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        state = "dirty" if dirty else "clean"
        return commit, state
    except Exception:
        return "unknown", "unknown"


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _make_random_checkpoint(name: str, config: MonoidEmbedConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MonoidEmbed(config)
    path = out_dir / f"{_sanitize_name(name)}.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, path)
    return path


def _estimate_param_count(config: MonoidEmbedConfig) -> int:
    vocab = int(config.vocab_size)
    d_state = int(config.d_state)
    n_layers = int(config.n_layers)
    exchange_dim = int(config.exchange_dim) if config.use_exchange else 0
    embed_dim = max(config.matryoshka_dims) if config.matryoshka_dims else d_state

    layer_params = 2 * vocab * d_state
    if exchange_dim > 0:
        layer_params += exchange_dim * exchange_dim
    total = n_layers * layer_params
    if n_layers > 1:
        total += n_layers * 2 * d_state
    if d_state != embed_dim:
        total += d_state * embed_dim
    return int(total)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(max(0, min(len(sorted_vals) - 1, pct * len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _run_benchmark(
    run_once,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            run_once()
        times = []
        for _ in range(max(1, repeat)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    times_sorted = sorted(times)
    avg = sum(times) / len(times)
    return {
        "avg": avg,
        "p50": _percentile(times_sorted, 0.50),
        "p95": _percentile(times_sorted, 0.95),
    }


def _read_rss_kb() -> int:
    status = Path("/proc/self/status")
    if not status.exists():
        return 0
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return 0


def _evict_cache(bytes_to_touch: int) -> int:
    if bytes_to_touch <= 0:
        return 0
    buf = bytearray(bytes_to_touch)
    step = 4096
    total = 0
    for idx in range(0, len(buf), step):
        buf[idx] = (buf[idx] + 1) & 0xFF
        total += buf[idx]
    return total


def _profile_memory(
    run_once,
    warm_reps: int,
    cache_evict_bytes: int,
    device: torch.device,
) -> dict:
    with torch.inference_mode():
        _evict_cache(cache_evict_bytes)
        rss_before = _read_rss_kb()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - start) * 1000.0
        rss_after = _read_rss_kb()

        warm_times = []
        rss_samples = []
        for _ in range(max(1, warm_reps)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - start) * 1000.0)
            rss_samples.append(_read_rss_kb())

    warm_avg = sum(warm_times) / len(warm_times)
    warm_min = min(rss_samples) if rss_samples else rss_after
    warm_max = max(rss_samples) if rss_samples else rss_after
    return {
        "rss_before_kb": rss_before,
        "rss_after_kb": rss_after,
        "rss_delta_kb": rss_after - rss_before,
        "rss_warm_min_kb": warm_min,
        "rss_warm_max_kb": warm_max,
        "rss_warm_spread_kb": warm_max - warm_min,
        "cold_ms": cold_ms,
        "warm_avg_ms": warm_avg,
    }


def _run_chunked(
    run_chunk,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    aggregate: str,
) -> torch.Tensor:
    if not chunks:
        raise RuntimeError("No chunks to run.")
    acc = None
    count = 0
    for tokens, lengths in chunks:
        out = run_chunk(tokens, lengths)
        if aggregate == "last":
            acc = out
            count = 1
            continue
        if acc is None:
            acc = out
        elif aggregate == "mean":
            acc = acc + out
        else:
            acc = torch.maximum(acc, out)
        count += 1
    if aggregate == "mean" and acc is not None and count > 1:
        acc = acc / count
    return acc


def _format_markdown(
    results: list[dict],
    metadata: dict,
    params: dict,
    breakdowns: list[dict] | None = None,
    memory_rows: list[dict] | None = None,
) -> str:
    lines = []
    timestamp = metadata["timestamp"]
    lines.append(f"## {timestamp}: Embed benchmark snapshot")
    lines.append("")
    lines.append("### Setup")
    lines.append(f"- host: {metadata['host']}")
    lines.append(f"- cpu: {metadata['cpu']}")
    lines.append(f"- os: {metadata['os']}")
    lines.append(f"- python: {metadata['python']}")
    lines.append(f"- torch: {metadata['torch']}")
    lines.append(f"- git: {metadata['git_commit']} ({metadata['git_state']})")
    lines.append(f"- command: `{metadata['command']}`")
    lines.append("")
    lines.append("### Parameters")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key, value in params.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("### Results")
    lines.append("| Preset | Params | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        lines.append(
            "| {preset} | {params} | {status} | {avg} | {p50} | {p95} | {kb_s} | {emb_s} |".format(**row)
        )
    lines.append("")
    if breakdowns:
        lines.append("### Breakdown (ms)")
        lines.append(
            "| Preset | Status | setup | recurrence | butterfly | activation | exchange | pooling | layer_norm | proj | total |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        fields = [
            "setup_ms",
            "recurrence_ms",
            "butterfly_ms",
            "activation_ms",
            "exchange_ms",
            "pooling_ms",
            "layer_norm_ms",
            "proj_ms",
            "total_ms",
        ]
        for row in breakdowns:
            if row.get("status") != "OK":
                lines.append(f"| {row['preset']} | {row['status']} | - | - | - | - | - | - | - | - | - |")
                continue
            values = [f"{row.get(field, 0.0):.2f}" for field in fields]
            lines.append(
                "| {preset} | {status} | {vals} |".format(
                    preset=row["preset"],
                    status=row["status"],
                    vals=" | ".join(values),
                )
            )
        lines.append("")
    if memory_rows:
        lines.append("### Memory")
        lines.append(
            "| Preset | Status | RSS before (KB) | RSS after (KB) | RSS delta (KB) | Warm RSS min (KB) | Warm RSS max (KB) | Warm RSS spread (KB) | Cold (ms) | Warm avg (ms) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in memory_rows:
            if row.get("status") != "OK":
                lines.append(f"| {row['preset']} | {row['status']} | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                "| {preset} | {status} | {rss_before_kb} | {rss_after_kb} | {rss_delta_kb} | {rss_warm_min_kb} | {rss_warm_max_kb} | {rss_warm_spread_kb} | {cold_ms:.2f} | {warm_avg_ms:.2f} |".format(
                    **row
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monoid embedding benchmarks and emit markdown snapshots.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--presets", type=str, nargs="*", default=None)
    parser.add_argument("--shape", action="append", default=None, help="name:n_layers:d_state:microblock_size")
    parser.add_argument("--checkpoint-dir", type=str, default="tmp/bench_ckpts")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--doc-max-bytes", type=int, default=0)
    parser.add_argument("--chunk-stride", type=int, default=0)
    parser.add_argument("--chunk-aggregate", type=str, default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--engine", type=str, default="kernel", choices=["kernel", "torch"])
    parser.add_argument("--mode", type=str, default="float", choices=["float", "quant", "quant_int8"])
    parser.add_argument("--pool-strategy", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--normalize-output", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-exchange", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-second-activation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fast-math", type=int, default=None, choices=[0, 1])
    parser.add_argument("--fast-tanh", type=int, default=None, choices=[0, 1])
    parser.add_argument("--profile-breakdown", action="store_true")
    parser.add_argument("--memory-profile", action="store_true")
    parser.add_argument("--memory-warm-reps", type=int, default=5)
    parser.add_argument("--cache-evict-bytes", type=int, default=0)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip writing markdown snapshot.")
    args = parser.parse_args()

    if args.fast_math is not None:
        os.environ["MONOID_CPU_FAST_MATH"] = "1" if args.fast_math else "0"
    if args.fast_tanh is not None:
        os.environ["MONOID_CPU_FAST_TANH"] = "1" if args.fast_tanh else "0"

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, args.threads))

    if args.engine == "kernel" and device.type != "cpu":
        raise SystemExit("--engine kernel requires --device cpu")
    if args.chunk_stride < 0:
        raise SystemExit("--chunk-stride must be >= 0")
    if args.doc_max_bytes < 0:
        raise SystemExit("--doc-max-bytes must be >= 0")

    preset_names = args.presets
    if preset_names is None and not args.shape:
        preset_names = sorted(MonoidEmbedConfig._PRESET_SPECS.keys())
    if preset_names is None:
        preset_names = []
    checkpoint_dir = Path(args.checkpoint_dir)

    data = Path(args.input).read_bytes()
    if args.doc_max_bytes > 0:
        data = data[: args.doc_max_bytes]
    doc_bytes = len(data)
    if doc_bytes == 0:
        raise SystemExit("Input file is empty after max-bytes truncation.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    chunked = args.chunk_stride > 0
    if chunked and args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be > 0 when --chunk-stride is set.")
    if chunked:
        chunk_size = min(args.max_bytes, doc_bytes)
        stride = args.chunk_stride
        if stride <= 0:
            raise SystemExit("--chunk-stride must be > 0 when chunking is enabled.")
        chunks_raw = []
        start = 0
        while True:
            end = min(doc_bytes, start + chunk_size)
            chunk = data[start:end]
            if not chunk:
                break
            chunks_raw.append(chunk)
            if end >= doc_bytes:
                break
            start += stride
    else:
        if args.max_bytes > 0:
            data = data[: args.max_bytes]
        chunk_size = len(data)
        chunks_raw = [data]

    seq_len = len(chunks_raw[0])
    chunks = []
    for chunk in chunks_raw:
        tokens_single = torch.tensor(list(chunk), dtype=torch.long, device=device).unsqueeze(0)
        tokens = tokens_single.repeat(args.batch_size, 1)
        lengths = torch.full((args.batch_size,), len(chunk), device=device, dtype=torch.long)
        chunks.append((tokens, lengths))

    tokens, lengths = chunks[0]
    chunk_count = len(chunks_raw)
    chunk_bytes_total = sum(len(chunk) for chunk in chunks_raw)

    entries = []
    shape_specs = []
    for preset in preset_names:
        config = MonoidEmbedConfig.from_preset(preset)
        entries.append(
            {
                "name": preset,
                "display": preset,
                "config": config,
                "kind": "preset",
            }
        )

    if args.shape:
        for shape in args.shape:
            parts = shape.split(":")
            if len(parts) != 4:
                raise SystemExit(f"--shape must be name:n_layers:d_state:microblock_size (got {shape})")
            name = parts[0].strip()
            if not name:
                raise SystemExit("--shape name cannot be empty.")
            n_layers = int(parts[1])
            d_state = int(parts[2])
            microblock_size = int(parts[3])
            config = MonoidEmbedConfig(
                n_layers=n_layers,
                d_state=d_state,
                microblock_size=microblock_size,
                exchange_dim=d_state // 16,
            )
            shape_specs.append(shape)
            entries.append(
                {
                    "name": f"shape_{_sanitize_name(name)}",
                    "display": f"shape:{name}",
                    "config": config,
                    "kind": "shape",
                }
            )

    if not entries:
        raise SystemExit("No presets or shapes to benchmark.")

    results = []
    results_numeric = []
    breakdowns = []
    memory_rows = []
    for entry in entries:
        config = entry["config"]
        display_name = entry["display"]
        param_count = _estimate_param_count(config)
        if args.mode != "float" and config.n_layers > 1 and args.engine != "kernel":
            results.append(
                {
                    "preset": display_name,
                    "params": str(param_count),
                    "status": "SKIP (torch quant needs 1 layer)",
                    "avg": "-",
                    "p50": "-",
                    "p95": "-",
                    "kb_s": "-",
                    "emb_s": "-",
                }
            )
            continue

        ckpt_path = checkpoint_dir / f"{entry['name']}.pt"
        if not ckpt_path.exists():
            ckpt_path = _make_random_checkpoint(entry["name"], config, checkpoint_dir)

        normalize_output = bool(args.normalize_output)
        use_exchange = bool(args.use_exchange)
        use_second_activation = bool(args.use_second_activation)
        emit_int8 = args.mode == "quant_int8"

        if args.engine == "kernel":
            kernel_cfg = MonoidCpuConfig(
                normalize_output=normalize_output,
                pool_strategy=args.pool_strategy,
                use_exchange=use_exchange,
                use_second_activation=use_second_activation,
                emit_int8=emit_int8,
                threads=args.threads,
            )
            kernel = MonoidCpuKernel.from_checkpoint(str(ckpt_path), config=kernel_cfg)

            if args.mode == "float":
                if chunked:
                    run_chunk = lambda toks, lens: kernel.forward_full_precision(toks, lengths=lens)
                    run_once = lambda: _run_chunked(run_chunk, chunks, args.chunk_aggregate)
                else:
                    run_once = lambda: kernel.forward_full_precision(tokens, lengths=lengths)
            else:
                if chunked:
                    run_chunk = lambda toks, lens: kernel.forward(toks, lengths=lens)
                    run_once = lambda: _run_chunked(run_chunk, chunks, args.chunk_aggregate)
                else:
                    run_once = lambda: kernel.forward(tokens, lengths=lengths)
        else:
            config.normalize_output = normalize_output
            config.pool_strategy = args.pool_strategy
            config.use_exchange = use_exchange
            config.use_second_activation = use_second_activation
            config.emit_int8 = emit_int8
            model = MonoidEmbed(config).to(device)
            state = _load_checkpoint(ckpt_path, device)
            model.load_state_dict(state, strict=False)
            model.eval()

            if args.mode == "float":
                if chunked:
                    run_chunk = lambda toks, lens: model(
                        toks,
                        lengths=lens,
                        quantized=False,
                        normalize_output=normalize_output,
                        pool_strategy=args.pool_strategy,
                    )
                    run_once = lambda: _run_chunked(run_chunk, chunks, args.chunk_aggregate)
                else:
                    run_once = lambda: model(
                        tokens,
                        lengths=lengths,
                        quantized=False,
                        normalize_output=normalize_output,
                        pool_strategy=args.pool_strategy,
                    )
            else:
                if chunked:
                    run_chunk = lambda toks, lens: model(
                        toks,
                        lengths=lens,
                        quantized=True,
                        normalize_output=normalize_output,
                        pool_strategy=args.pool_strategy,
                    )
                    run_once = lambda: _run_chunked(run_chunk, chunks, args.chunk_aggregate)
                else:
                    run_once = lambda: model(
                        tokens,
                        lengths=lengths,
                        quantized=True,
                        normalize_output=normalize_output,
                        pool_strategy=args.pool_strategy,
                    )

        timing = _run_benchmark(run_once, device=device, warmup=args.warmup, repeat=args.repeat)
        total_bytes = int((doc_bytes if chunked else seq_len) * args.batch_size)
        kb_per_s = (total_bytes / 1024.0) / max(timing["avg"], 1e-9)
        emb_per_s = args.batch_size / max(timing["avg"], 1e-9)

        results.append(
            {
                "preset": display_name,
                "params": str(param_count),
                "status": "OK",
                "avg": f"{timing['avg'] * 1000:.2f}",
                "p50": f"{timing['p50'] * 1000:.2f}",
                "p95": f"{timing['p95'] * 1000:.2f}",
                "kb_s": f"{kb_per_s:.2f}",
                "emb_s": f"{emb_per_s:.2f}",
            }
        )
        results_numeric.append(
            {
                "preset": display_name,
                "params": param_count,
                "status": "OK",
                "avg_ms": timing["avg"] * 1000.0,
                "p50_ms": timing["p50"] * 1000.0,
                "p95_ms": timing["p95"] * 1000.0,
                "kb_s": kb_per_s,
                "emb_s": emb_per_s,
            }
        )
        if args.profile_breakdown:
            breakdown_row = {"preset": display_name, "status": "OK"}
            if args.engine != "kernel":
                breakdown_row["status"] = "SKIP (kernel only)"
            elif args.mode != "float":
                breakdown_row["status"] = "SKIP (float only)"
            else:
                timing_total: dict[str, float] = {}
                if chunked:
                    for tokens_chunk, lengths_chunk in chunks:
                        _, timing_map = kernel.forward_full_precision_profile(
                            tokens_chunk,
                            lengths=lengths_chunk,
                        )
                        for key, value in timing_map.items():
                            timing_total[key] = timing_total.get(key, 0.0) + float(value)
                else:
                    _, timing_map = kernel.forward_full_precision_profile(tokens, lengths=lengths)
                    timing_total = {key: float(value) for key, value in timing_map.items()}
                breakdown_row.update(timing_total)
            breakdowns.append(breakdown_row)
        if args.memory_profile:
            mem_row = {"preset": display_name, "status": "OK"}
            if args.engine != "kernel":
                mem_row["status"] = "SKIP (kernel only)"
            else:
                mem_row.update(
                    _profile_memory(
                        run_once,
                        warm_reps=args.memory_warm_reps,
                        cache_evict_bytes=args.cache_evict_bytes,
                        device=device,
                    )
                )
            memory_rows.append(mem_row)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    commit, git_state = _get_git_info()
    metadata = {
        "timestamp": timestamp,
        "host": platform.node() or "unknown",
        "cpu": _get_cpu_model(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "git_commit": commit,
        "git_state": git_state,
        "command": " ".join(sys.argv),
    }
    params = {
        "engine": args.engine,
        "mode": args.mode,
        "device": str(device),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "max_bytes": args.max_bytes,
        "doc_max_bytes": args.doc_max_bytes,
        "chunk_stride": args.chunk_stride,
        "chunk_aggregate": args.chunk_aggregate,
        "chunk_count": chunk_count,
        "chunk_bytes_total": chunk_bytes_total,
        "doc_bytes": doc_bytes,
        "seq_len": seq_len,
        "input": args.input,
        "pool_strategy": args.pool_strategy,
        "normalize_output": int(args.normalize_output),
        "use_exchange": int(args.use_exchange),
        "use_second_activation": int(args.use_second_activation),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "checkpoint_dir": str(checkpoint_dir),
        "presets": ", ".join(preset_names),
        "shapes": ", ".join(shape_specs) if shape_specs else "none",
        "fast_math": os.getenv("MONOID_CPU_FAST_MATH", "unset"),
        "fast_tanh": os.getenv("MONOID_CPU_FAST_TANH", "unset"),
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "unset"),
        "MONOID_CPU_THREADS": os.getenv("MONOID_CPU_THREADS", "unset"),
        "profile_breakdown": int(args.profile_breakdown),
        "memory_profile": int(args.memory_profile),
        "memory_warm_reps": args.memory_warm_reps,
        "cache_evict_bytes": args.cache_evict_bytes,
    }

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "params": params,
            "results": results_numeric,
            "breakdowns": breakdowns,
            "memory": memory_rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.no_snapshot:
        return
    markdown = _format_markdown(
        results,
        metadata,
        params,
        breakdowns=breakdowns if breakdowns else None,
        memory_rows=memory_rows if memory_rows else None,
    )
    out_path: Path
    if args.output:
        out_path = Path(args.output)
    else:
        snap_dir = Path(".snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        safe_ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = snap_dir / f"embed_bench_{safe_ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote snapshot: {out_path}")


if __name__ == "__main__":
    main()
