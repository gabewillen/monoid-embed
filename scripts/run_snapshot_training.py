#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import deque
from subprocess import CalledProcessError, check_output

import wandb


def _read_run_id_from_latest():
    latest = os.path.join("wandb", "latest-run")
    if not os.path.exists(latest):
        return None
    real = os.path.realpath(latest)
    base = os.path.basename(real)
    match = re.match(r"run-\d{8}_\d{6}-([a-z0-9]+)$", base)
    if match:
        return match.group(1)
    return None


def _infer_entity(api):
    entity = getattr(api, "default_entity", None)
    if entity:
        return entity
    try:
        viewer = api.viewer()
        if hasattr(viewer, "entity") and viewer.entity:
            return viewer.entity
        if hasattr(viewer, "username") and viewer.username:
            return viewer.username
        if isinstance(viewer, dict):
            return viewer.get("entity") or viewer.get("username")
    except Exception:
        return None
    return None


def _extract_step(row):
    for key in ("_step", "step", "global_step", "train/step"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _pick_first(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _collect_history(run, keys, every, last):
    rows_iter = run.scan_history(keys=keys, page_size=1000)
    if last:
        tail = deque(maxlen=last)
        for row in rows_iter:
            tail.append(row)
        rows_iter = tail

    selected = []
    last_step = None
    index = 0
    for row in rows_iter:
        index += 1
        if every:
            step_val = _extract_step(row)
            if step_val is None:
                step_val = index - 1
            if last_step is not None and (step_val - last_step) < every:
                continue
            last_step = step_val
        selected.append(row)
    return selected


def _infer_checkpoint(config, args):
    if args.checkpoint:
        return args.checkpoint
    output_dir = config.get("output_dir")
    if not output_dir:
        run_name = config.get("run_name") or config.get("run_name_full")
        if run_name:
            output_dir = os.path.join("checkpoints", run_name)
    if not output_dir or not os.path.exists(output_dir):
        return None
    candidates = [
        os.path.join(output_dir, "monoid_embed_latest_full.pt"),
        os.path.join(output_dir, "monoid_embed_latest.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    pt_files = [f for f in os.listdir(output_dir) if f.endswith(".pt")]
    step_files = []
    for name in pt_files:
        stem = name[:-3]
        if stem.isdigit():
            step_files.append((int(stem), name))
    if step_files:
        step_files.sort(reverse=True)
        return os.path.join(output_dir, step_files[0][1])
    return None


def _run_retrieval_eval(checkpoint, config, args):
    if not checkpoint:
        return {"error": "checkpoint_not_found"}
    env = os.environ.copy()
    preset = config.get("preset")
    if preset:
        env["MONOID_PRESET"] = str(preset)
    cmd = [
        sys.executable,
        os.path.join("scripts", "run_retrieval_eval.py"),
        "--checkpoint",
        checkpoint,
        "--dataset",
        args.retrieval_dataset,
        "--split",
        args.retrieval_split,
        "--docs",
        str(args.retrieval_docs),
        "--queries",
        str(args.retrieval_queries),
        "--batch_size",
        str(args.retrieval_batch_size),
        "--device",
        args.retrieval_device,
    ]
    if args.retrieval_normalize:
        cmd.append("--normalize")
    if args.retrieval_max_bytes is not None:
        cmd.extend(["--max_bytes", str(args.retrieval_max_bytes)])
    if args.retrieval_positives_per_query is not None:
        cmd.extend(["--positives_per_query", str(args.retrieval_positives_per_query)])
    if args.retrieval_full_corpus:
        cmd.append("--full_corpus")

    try:
        output = check_output(cmd, env=env, text=True, stderr=sys.stderr)
    except CalledProcessError as exc:
        return {"error": f"retrieval_eval_failed: {exc}"}

    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("recall@"):
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("retrieval/recall@"):
                metrics[key.split("/", 1)[1]] = float(value)
            elif key.startswith("recall@"):
                metrics[key] = float(value)
    return {
        "checkpoint": checkpoint,
        "dataset": args.retrieval_dataset,
        "split": args.retrieval_split,
        "queries": args.retrieval_queries,
        "docs": args.retrieval_docs,
        "batch_size": args.retrieval_batch_size,
        "metrics": metrics,
        "raw_output": output.strip(),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Snapshot current W&B run with history and retrieval eval.")
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--project", type=str, default=os.getenv("WANDB_PROJECT", "monoid"))
    parser.add_argument("--run-id", type=str, default=os.getenv("WANDB_RUN_ID"))
    parser.add_argument("--output", type=str, default="-", help="Output path for JSON ('-' for stdout).")
    parser.add_argument("--history-keys", type=str, nargs="*", default=None)
    parser.add_argument("--history-last", type=int, default=200)
    parser.add_argument("--history-every", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--retrieval-dataset", type=str, default="mteb/scifact")
    parser.add_argument("--retrieval-split", type=str, default="test")
    parser.add_argument("--retrieval-queries", type=int, default=200)
    parser.add_argument("--retrieval-docs", type=int, default=2000)
    parser.add_argument(
        "--retrieval-positives-per-query",
        type=int,
        default=None,
        help="Limit positives per query (default: all).",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--retrieval-max-bytes", type=int, default=None)
    parser.add_argument("--retrieval-device", type=str, default="cpu")
    parser.add_argument("--retrieval-full-corpus", action="store_true")
    parser.add_argument("--retrieval-normalize", dest="retrieval_normalize", action="store_true", default=True)
    parser.add_argument("--no-retrieval-normalize", dest="retrieval_normalize", action="store_false")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_id = args.run_id or _read_run_id_from_latest()
    if not run_id:
        print("Error: run id not provided and could not infer from wandb/latest-run.", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    entity = args.entity or _infer_entity(api)
    project = args.project
    if not project:
        print("Error: missing project. Set --project or WANDB_PROJECT.", file=sys.stderr)
        sys.exit(2)

    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        run = api.run(run_path)
    except Exception as exc:
        print(f"Error fetching run: {exc}", file=sys.stderr)
        sys.exit(1)

    config = dict(run.config or {})
    config = {k: v for k, v in config.items() if not str(k).startswith("_")}
    summary = dict(run.summary or {})
    summary = {k: v for k, v in summary.items() if not str(k).startswith("_")}

    model_params = _pick_first(
        summary,
        ["MonoidEmbed params", "model.params", "model.num_parameters", "num_parameters", "params"],
    )
    model_preset = _pick_first(summary, ["model.preset", "model_preset"]) or config.get("preset")
    model_info = {
        "preset": model_preset,
        "n_layers": _pick_first(summary, ["model.n_layers", "n_layers"]) or config.get("n_layers"),
        "d_state": _pick_first(summary, ["model.d_state", "d_state"]) or config.get("d_state"),
        "microblock_size": _pick_first(summary, ["model.microblock_size", "microblock_size"])
        or config.get("microblock_size"),
        "exchange_dim": _pick_first(summary, ["model.exchange_dim", "exchange_dim"]) or config.get("exchange_dim"),
        "params": model_params,
    }

    checkpoint = _infer_checkpoint(config, args)
    history = _collect_history(run, args.history_keys, args.history_every, args.history_last)
    retrieval = _run_retrieval_eval(checkpoint, config, args)

    report = {
        "model": model_info,
        "cli_args": config,
        "logs_params": {
            "every": args.history_every,
            "last": args.history_last,
        },
        "logs": history,
        "retrieval_eval": retrieval,
    }

    output_to_stdout = args.output == "-"
    if output_to_stdout:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
