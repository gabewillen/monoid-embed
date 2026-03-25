#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunSpec:
    mode: str
    batch_size: int
    threads: int
    presets: list[str]


def _build_runs(suite: str) -> list[RunSpec]:
    if suite == "smoke":
        presets = ["small", "small_l2"]
        modes = ["float"]
        batch_sizes = [1, 64]
        threads = [1, 16]
    else:
        presets = ["small", "small_l2", "medium", "base"]
        modes = ["float", "quant", "quant_int8"]
        batch_sizes = [1, 64]
        threads = [1, 16, 32]

    runs = []
    for mode in modes:
        for batch_size in batch_sizes:
            for threads_count in threads:
                runs.append(
                    RunSpec(
                        mode=mode,
                        batch_size=batch_size,
                        threads=threads_count,
                        presets=presets,
                    )
                )
    return runs


def _run_bench(spec: RunSpec, args: argparse.Namespace, json_path: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_embed_bench.py",
        "--input",
        args.input,
        "--engine",
        "kernel",
        "--mode",
        spec.mode,
        "--threads",
        str(spec.threads),
        "--batch-size",
        str(spec.batch_size),
        "--max-bytes",
        str(args.max_bytes),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--json-output",
        str(json_path),
    ]
    cmd.extend(["--presets", *spec.presets])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MONOID_CPU_THREADS"] = str(spec.threads)
    env["MKL_NUM_THREADS"] = "1"

    if args.pin:
        cmd = ["taskset", "-c", args.pin, *cmd]

    subprocess.run(cmd, check=True, env=env)
    return json.loads(json_path.read_text())


def _flatten_results(payload: dict) -> list[dict]:
    params = payload.get("params", {})
    mode = params.get("mode", "unknown")
    batch_size = int(params.get("batch_size", 0))
    threads = int(params.get("threads", 0))
    results = []
    for row in payload.get("results", []):
        results.append(
            {
                "preset": row.get("preset"),
                "mode": mode,
                "batch_size": batch_size,
                "threads": threads,
                "status": row.get("status"),
                "avg_ms": row.get("avg_ms"),
                "p50_ms": row.get("p50_ms"),
                "p95_ms": row.get("p95_ms"),
                "emb_s": row.get("emb_s"),
                "kb_s": row.get("kb_s"),
            }
        )
    return results


def _build_index(rows: list[dict]) -> dict:
    index = {}
    for row in rows:
        key = (row["preset"], row["mode"], row["batch_size"], row["threads"])
        index[key] = row
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regression benchmarks for Monoid CPU embed kernel.")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--input", default="docs/embed.md")
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--pin", default="0-15")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--throughput-drop-pct", type=float, default=0.10)
    parser.add_argument("--latency-increase-pct", type=float, default=0.10)
    parser.add_argument(
        "--metric",
        choices=["avg", "p50"],
        default="p50",
        help="Metric for regression gating (avg is sensitive to outliers; p50 is more stable).",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline or f"benchmarks/embed_regression_{args.suite}.json")
    runs = _build_runs(args.suite)

    all_rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in runs:
            json_path = Path(tmpdir) / f"run_{spec.mode}_{spec.batch_size}_{spec.threads}.json"
            payload = _run_bench(spec, args, json_path)
            all_rows.extend(_flatten_results(payload))

    ok_rows = [row for row in all_rows if row.get("status") == "OK"]
    baseline_payload = {
        "suite": args.suite,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {
            "input": args.input,
            "max_bytes": args.max_bytes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "pin": args.pin,
            "metric": args.metric,
        },
        "results": ok_rows,
    }

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
        print(f"Wrote baseline: {baseline_path}")
        return

    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}. Use --update-baseline to create it.")

    baseline = json.loads(baseline_path.read_text())
    baseline_rows = baseline.get("results", [])

    baseline_index = _build_index(baseline_rows)
    current_index = _build_index(ok_rows)

    def _get_latency_ms(row: dict) -> float:
        if args.metric == "p50":
            return float(row.get("p50_ms", 0.0) or 0.0)
        return float(row.get("avg_ms", 0.0) or 0.0)

    def _get_emb_s(row: dict) -> float:
        if args.metric == "p50":
            batch = float(row.get("batch_size", 0.0) or 0.0)
            p50_ms = float(row.get("p50_ms", 0.0) or 0.0)
            if batch > 0 and p50_ms > 0:
                return batch / (p50_ms / 1000.0)
            return 0.0
        return float(row.get("emb_s", 0.0) or 0.0)

    regressions = []
    for key, base in baseline_index.items():
        cur = current_index.get(key)
        if cur is None:
            regressions.append({"key": key, "issue": "missing"})
            continue
        base_emb = _get_emb_s(base)
        cur_emb = _get_emb_s(cur)
        base_ms = _get_latency_ms(base)
        cur_ms = _get_latency_ms(cur)
        if base_emb > 0:
            drop = (base_emb - cur_emb) / base_emb
            if drop > args.throughput_drop_pct:
                regressions.append({"key": key, "issue": f"throughput_drop {drop:.2%}"})
        if base_ms > 0:
            inc = (cur_ms - base_ms) / base_ms
            if inc > args.latency_increase_pct:
                regressions.append({"key": key, "issue": f"latency_increase {inc:.2%}"})

    if regressions:
        print("Regressions detected:")
        for reg in regressions:
            print(f"- {reg['key']}: {reg['issue']}")
        raise SystemExit(1)

    print("Regression check passed.")


if __name__ == "__main__":
    main()
