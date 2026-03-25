#!/usr/bin/env python3
import argparse
import datetime
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _parse_int_list(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(int(part))
    return parsed


def _parse_str_list(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(part)
    return parsed


def _valid_d_state(d_state: int) -> bool:
    if d_state % 8 != 0:
        return False
    tile_dim = d_state // 8
    if tile_dim <= 0 or (tile_dim & (tile_dim - 1)) != 0:
        return False
    if tile_dim % 4 != 0:
        return False
    return True


def _valid_microblock(microblock: int) -> bool:
    if microblock < 64:
        return False
    return (microblock & (microblock - 1)) == 0


def _unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _default_inputs() -> list[tuple[str, str]]:
    candidates = [
        ("ascii_english_4096", "tmp/bench_inputs/ascii_english_4096.txt"),
        ("utf8_heavy_4096", "tmp/bench_inputs/utf8_heavy_4096.txt"),
    ]
    out = []
    for label, path in candidates:
        if Path(path).exists():
            out.append((label, path))
    if not out:
        raise SystemExit("No default inputs found in tmp/bench_inputs; pass --inputs.")
    return out


def _parse_inputs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return _default_inputs()
    out = []
    for value in values:
        if ":" in value:
            label, path = value.split(":", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = value.strip()
            label = Path(path).stem
        if not label:
            raise SystemExit(f"Invalid input label for {value}")
        if not Path(path).exists():
            raise SystemExit(f"Input not found: {path}")
        out.append((label, path))
    return out


def _progress_iter(items, total: int):
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    if tqdm is not None:
        for item in tqdm(items, total=total, unit="run"):
            yield item
        return

    for idx, item in enumerate(items, 1):
        label = ""
        if isinstance(item, tuple) and len(item) >= 6:
            input_label, _, mode, batch_size, threads, fast_math = item[:6]
            fast_label = "-" if fast_math is None else str(fast_math)
            label = f"{input_label} {mode} b{batch_size} t{threads} fm{fast_label}"
        if label:
            print(f"Progress {idx}/{total}: {label}", flush=True)
        else:
            print(f"Progress {idx}/{total}", flush=True)
        yield item


def _estimate_param_count(
    n_layers: int,
    d_state: int,
    vocab_size: int = 256,
    matryoshka_dims: tuple[int, ...] = (512, 256, 128),
    use_exchange: bool = True,
) -> int:
    exchange_dim = d_state // 16 if use_exchange else 0
    embed_dim = max(matryoshka_dims) if matryoshka_dims else d_state
    layer_params = 2 * vocab_size * d_state
    if exchange_dim:
        layer_params += exchange_dim * exchange_dim
    if n_layers > 1:
        layer_params += 2 * d_state
    total = n_layers * layer_params
    if d_state != embed_dim:
        total += d_state * embed_dim
    return int(total)


def _format_markdown(
    metadata: dict,
    grid: dict,
    rows: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Kitchen Sink Benchmark")
    lines.append("")
    lines.append(f"- timestamp: {metadata['timestamp']}")
    lines.append(f"- host: {metadata['host']}")
    lines.append(f"- cpu: {metadata['cpu']}")
    lines.append(f"- os: {metadata['os']}")
    lines.append(f"- python: {metadata['python']}")
    lines.append(f"- command: `{metadata['command']}`")
    lines.append("")
    lines.append("## Grid")
    lines.append(f"- inputs: {', '.join(grid['inputs'])}")
    lines.append(f"- modes: {', '.join(grid['modes'])}")
    lines.append(f"- layers: {', '.join(str(v) for v in grid['layers'])}")
    lines.append(f"- d_state: {', '.join(str(v) for v in grid['d_states'])}")
    lines.append(f"- microblocks: {', '.join(str(v) for v in grid['microblocks'])}")
    lines.append(f"- batch_sizes: {', '.join(str(v) for v in grid['batch_sizes'])}")
    lines.append(f"- threads: {', '.join(str(v) for v in grid['threads'])}")
    lines.append(f"- fast_math (float only): {', '.join(str(v) for v in grid['fast_math'])}")
    lines.append(f"- max_bytes: {grid['max_bytes']}")
    if grid.get("max_params") is not None:
        lines.append(f"- max_params: {grid['max_params']}")
    if grid.get("min_params") is not None:
        lines.append(f"- min_params: {grid['min_params']}")
    lines.append(f"- pin: {grid['pin'] or 'unset'}")
    lines.append("")
    lines.append("## Results")
    lines.append(
        "| Input | Mode | FastMath | Layers | d_state | microblock | Params | Batch | Threads | Status | Avg (ms) | p50 (ms) | p95 (ms) | Embeddings/s | KB/s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {input} | {mode} | {fast_math} | {layers} | {d_state} | {microblock} | {params} | {batch} | {threads} | {status} | {avg_ms} | {p50_ms} | {p95_ms} | {emb_s} | {kb_s} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Quant modes use the kernel implementation; torch quant still supports n_layers=1 only.")
    lines.append("- Invalid d_state values (tile_dim must be power-of-two and divisible by 4) are filtered.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full kitchen-sink benchmark and emit a single markdown table.")
    parser.add_argument("--inputs", nargs="*", default=None, help="label:path or path (defaults to tmp/bench_inputs).")
    parser.add_argument("--modes", nargs="*", default=["float", "quant", "quant_int8"])
    parser.add_argument("--layers", nargs="*", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--d-states", nargs="*", default=["256", "512", "1024"])
    parser.add_argument("--microblocks", nargs="*", default=["64", "128", "256", "512"])
    parser.add_argument("--batch-sizes", nargs="*", default=["1", "64"])
    parser.add_argument("--threads", nargs="*", default=["1", "2", "4", "8", "16"])
    parser.add_argument("--fast-math-values", nargs="*", default=["0", "1"])
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--min-params", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pin", type=str, default="0-15")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    inputs = _parse_inputs(args.inputs)
    modes = _unique(_parse_str_list(args.modes))
    layers = _unique(_parse_int_list(args.layers))
    d_states = _unique([d for d in _parse_int_list(args.d_states) if _valid_d_state(d)])
    microblocks = _unique([m for m in _parse_int_list(args.microblocks) if _valid_microblock(m)])
    batch_sizes = _unique(_parse_int_list(args.batch_sizes))
    threads_list = _unique(_parse_int_list(args.threads))
    fast_math_values = _unique(_parse_int_list(args.fast_math_values))
    fast_math_values = [v for v in fast_math_values if v in (0, 1)]
    if not fast_math_values:
        fast_math_values = [0, 1]

    if not d_states:
        raise SystemExit("No valid d_state values; must be divisible by 8 and have power-of-two tile_dim.")
    if not microblocks:
        raise SystemExit("No valid microblock sizes; must be power-of-two and >= 64.")

    shapes = []
    for n_layers, d_state, microblock in itertools.product(layers, d_states, microblocks):
        name = f"L{n_layers}_D{d_state}_MB{microblock}"
        param_count = _estimate_param_count(n_layers, d_state)
        if args.max_params is not None and param_count > args.max_params:
            continue
        if args.min_params is not None and param_count < args.min_params:
            continue
        shapes.append(
            {
                "name": name,
                "display": f"shape:{name}",
                "spec": f"{name}:{n_layers}:{d_state}:{microblock}",
                "n_layers": n_layers,
                "d_state": d_state,
                "microblock": microblock,
                "params": param_count,
            }
        )

    rows: list[dict] = []
    metadata = {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": platform.node() or "unknown",
        "cpu": "unknown",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
    }

    runs = []
    for (input_label, input_path), mode, batch_size, threads in itertools.product(
        inputs, modes, batch_sizes, threads_list
    ):
        mode_fast_values = fast_math_values if mode == "float" else [None]
        for fast_math in mode_fast_values:
            runs.append((input_label, input_path, mode, batch_size, threads, fast_math))

    total_runs = len(runs)
    for input_label, input_path, mode, batch_size, threads, fast_math in _progress_iter(runs, total_runs):
        suffix = f"fm{fast_math}" if fast_math is not None else "fmna"
        json_path = (
            Path("tmp")
            / "kitchen_sink"
            / f"{input_label}_{mode}_{suffix}_b{batch_size}_t{threads}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/run_embed_bench.py",
            "--input",
            input_path,
            "--engine",
            "kernel",
            "--mode",
            mode,
            "--threads",
            str(threads),
            "--batch-size",
            str(batch_size),
            "--max-bytes",
            str(args.max_bytes),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--json-output",
            str(json_path),
            "--no-snapshot",
        ]
        if fast_math is not None:
            cmd.extend(["--fast-math", str(fast_math), "--fast-tanh", str(fast_math)])
        for shape in shapes:
            cmd.extend(["--shape", shape["spec"]])
        env = os.environ.copy()
        env["MONOID_CPU_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = "1"
        if args.pin:
            cmd = ["taskset", "-c", args.pin, *cmd]
        subprocess.run(cmd, check=True, env=env)
        payload = json.loads(json_path.read_text())
        if metadata["cpu"] == "unknown" and payload.get("metadata"):
            meta = payload["metadata"]
            metadata["cpu"] = meta.get("cpu", "unknown")
        results = payload.get("results", [])
        index = {row["preset"]: row for row in results}
        fast_label = "-" if fast_math is None else str(fast_math)
        for shape in shapes:
            result = index.get(shape["display"])
            if result:
                rows.append(
                    {
                        "input": input_label,
                        "mode": mode,
                        "fast_math": fast_label,
                        "layers": shape["n_layers"],
                        "d_state": shape["d_state"],
                        "microblock": shape["microblock"],
                        "params": str(result.get("params", shape["params"])),
                        "batch": batch_size,
                        "threads": threads,
                        "status": "OK",
                        "avg_ms": f"{result.get('avg_ms', 0.0):.2f}",
                        "p50_ms": f"{result.get('p50_ms', 0.0):.2f}",
                        "p95_ms": f"{result.get('p95_ms', 0.0):.2f}",
                        "emb_s": f"{result.get('emb_s', 0.0):.2f}",
                        "kb_s": f"{result.get('kb_s', 0.0):.2f}",
                    }
                )
            else:
                status = "SKIP (missing result)"
                rows.append(
                    {
                        "input": input_label,
                        "mode": mode,
                        "fast_math": fast_label,
                        "layers": shape["n_layers"],
                        "d_state": shape["d_state"],
                        "microblock": shape["microblock"],
                        "params": str(shape["params"]),
                        "batch": batch_size,
                        "threads": threads,
                        "status": status,
                        "avg_ms": "-",
                        "p50_ms": "-",
                        "p95_ms": "-",
                        "emb_s": "-",
                        "kb_s": "-",
                    }
                )

    grid = {
        "inputs": [label for label, _ in inputs],
        "modes": modes,
        "layers": layers,
        "d_states": d_states,
        "microblocks": microblocks,
        "batch_sizes": batch_sizes,
        "threads": threads_list,
        "fast_math": fast_math_values,
        "max_bytes": args.max_bytes,
        "max_params": args.max_params,
        "min_params": args.min_params,
        "pin": args.pin,
    }
    markdown = _format_markdown(metadata, grid, rows)

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.output:
        out_path = Path(args.output)
    else:
        snap_dir = Path(".snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        safe_ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = snap_dir / f"kitchen_sink_{safe_ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote kitchen sink snapshot: {out_path}")


if __name__ == "__main__":
    main()
