#!/usr/bin/env python3
import argparse
import datetime
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


def _unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dynamic_batch(length: int, target_bytes: int, min_batch: int, max_batch: int) -> int:
    if length <= 0:
        return min_batch
    batch = max(1, target_bytes // length)
    if min_batch:
        batch = max(batch, min_batch)
    if max_batch:
        batch = min(batch, max_batch)
    return int(batch)


def _format_markdown(metadata: dict, grid: dict, rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Length Sweep Benchmark")
    lines.append("")
    lines.append(f"- timestamp: {metadata['timestamp']}")
    lines.append(f"- host: {metadata['host']}")
    lines.append(f"- cpu: {metadata['cpu']}")
    lines.append(f"- os: {metadata['os']}")
    lines.append(f"- python: {metadata['python']}")
    lines.append(f"- command: `{metadata['command']}`")
    lines.append("")
    lines.append("## Grid")
    lines.append(f"- input: {grid['input']}")
    lines.append(f"- modes: {', '.join(grid['modes'])}")
    lines.append(f"- presets: {', '.join(grid['presets'])}")
    lines.append(f"- lengths: {', '.join(str(v) for v in grid['lengths'])}")
    lines.append(f"- threads: {', '.join(str(v) for v in grid['threads'])}")
    lines.append(f"- fast_math (float only): {', '.join(str(v) for v in grid['fast_math'])}")
    lines.append(f"- target_bytes_per_batch: {grid['target_bytes']}")
    lines.append(f"- min_batch: {grid['min_batch']}, max_batch: {grid['max_batch']}")
    lines.append(f"- pin: {grid['pin'] or 'unset'}")
    lines.append("")
    lines.append(
        "| Mode | FastMath | Preset | Params | Length | Batch | Total bytes | Threads | Status | Avg (ms) | p50 (ms) | p95 (ms) | Embeddings/s | KB/s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {mode} | {fast_math} | {preset} | {params} | {length} | {batch} | {total_bytes} | {threads} | {status} | {avg_ms} | {p50_ms} | {p95_ms} | {emb_s} | {kb_s} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Batch size is computed as floor(target_bytes / length) and clamped to [min_batch, max_batch].")
    lines.append("- Quant modes run on the kernel path; torch quant still supports n_layers=1 only.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark variable-length inputs with dynamic batch sizing."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--presets", nargs="*", default=["small", "small_l2"])
    parser.add_argument("--modes", nargs="*", default=["float", "quant", "quant_int8"])
    parser.add_argument("--lengths", nargs="*", default=["64", "128", "256", "512", "1024", "2048", "4096"])
    parser.add_argument("--threads", nargs="*", default=["1", "2", "4", "8", "16", "32"])
    parser.add_argument("--fast-math-values", nargs="*", default=["0", "1"])
    parser.add_argument("--target-bytes", type=int, default=262144)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pin", type=str, default="0-15")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    presets = _unique(_parse_str_list(args.presets))
    modes = _unique(_parse_str_list(args.modes))
    lengths = _unique(_parse_int_list(args.lengths))
    threads_list = _unique(_parse_int_list(args.threads))
    fast_math_values = _unique(_parse_int_list(args.fast_math_values))
    fast_math_values = [v for v in fast_math_values if v in (0, 1)]
    if not fast_math_values:
        fast_math_values = [0, 1]

    rows: list[dict] = []
    metadata = {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": platform.node() or "unknown",
        "cpu": "unknown",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "command": " ".join(sys.argv),
    }

    for mode in modes:
        for length in lengths:
            for threads in threads_list:
                mode_fast_values = fast_math_values if mode == "float" else [None]
                for fast_math in mode_fast_values:
                    batch_size = _dynamic_batch(length, args.target_bytes, args.min_batch, args.max_batch)
                    suffix = f"fm{fast_math}" if fast_math is not None else "fmna"
                    json_path = (
                        Path("tmp") / "length_sweep" / f"{mode}_{suffix}_len{length}_t{threads}.json"
                    )
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        "scripts/run_embed_bench.py",
                        "--input",
                        str(input_path),
                        "--engine",
                        "kernel",
                        "--mode",
                        mode,
                        "--threads",
                        str(threads),
                        "--batch-size",
                        str(batch_size),
                        "--max-bytes",
                        str(length),
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
                    cmd.extend(["--presets", *presets])
                    env = os.environ.copy()
                    env["MONOID_CPU_THREADS"] = str(threads)
                    env["OMP_NUM_THREADS"] = str(threads)
                    env["MKL_NUM_THREADS"] = "1"
                    if args.pin:
                        cmd = ["taskset", "-c", args.pin, *cmd]
                    subprocess.run(cmd, check=True, env=env)

                    payload = json.loads(json_path.read_text())
                    if metadata["cpu"] == "unknown" and payload.get("metadata"):
                        metadata["cpu"] = payload["metadata"].get("cpu", "unknown")
                    params = payload.get("params", {})
                    seq_len = int(params.get("seq_len", length) or length)
                    results = payload.get("results", [])
                    index = {row.get("preset"): row for row in results}
                    fast_label = "-" if fast_math is None else str(fast_math)
                    for preset in presets:
                        result = index.get(preset)
                        if result:
                            rows.append(
                                {
                                    "mode": mode,
                                    "fast_math": fast_label,
                                    "preset": preset,
                                    "params": str(result.get("params", "-")),
                                    "length": seq_len,
                                    "batch": int(params.get("batch_size", batch_size)),
                                    "total_bytes": int(seq_len) * int(params.get("batch_size", batch_size)),
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
                            rows.append(
                                {
                                    "mode": mode,
                                    "fast_math": fast_label,
                                    "preset": preset,
                                    "params": "-",
                                    "length": seq_len,
                                    "batch": int(params.get("batch_size", batch_size)),
                                    "total_bytes": int(seq_len) * int(params.get("batch_size", batch_size)),
                                    "threads": threads,
                                    "status": "SKIP",
                                    "avg_ms": "-",
                                    "p50_ms": "-",
                                    "p95_ms": "-",
                                    "emb_s": "-",
                                    "kb_s": "-",
                                }
                            )

    grid = {
        "input": str(input_path),
        "modes": modes,
        "presets": presets,
        "lengths": lengths,
        "threads": threads_list,
        "fast_math": fast_math_values,
        "target_bytes": args.target_bytes,
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
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
        out_path = snap_dir / f"length_sweep_{safe_ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote length sweep snapshot: {out_path}")


if __name__ == "__main__":
    main()
