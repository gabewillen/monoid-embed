#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _load_scores(result_dir: Path, split: str) -> dict:
    base = result_dir / "no_model_name_available" / "no_revision_available"
    if not base.exists():
        raise FileNotFoundError(f"Missing MTEB results dir: {base}")
    scores = {}
    for path in base.glob("*.json"):
        data = json.loads(path.read_text())
        task_name = data.get("task_name") or path.stem
        entries = data.get("scores", {}).get(split, [])
        if not entries:
            continue
        main_score = entries[0].get("main_score")
        if main_score is None:
            continue
        scores[task_name] = float(main_score)
    return scores


def _task_type_map(task_names: list[str]) -> dict:
    try:
        from mteb import get_tasks
    except Exception as exc:
        raise RuntimeError(f"Failed to import mteb: {exc}") from exc

    tasks = get_tasks(tasks=task_names)
    mapping = {}
    for task in tasks:
        name = task.metadata.name
        task_type = getattr(task.metadata, "type", None) or getattr(task.metadata, "task_type", None)
        if isinstance(task_type, list):
            task_type = task_type[0] if task_type else "unknown"
        if hasattr(task_type, "value"):
            task_type = task_type.value
        mapping[name] = task_type or "unknown"
    return mapping


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MTEB results into a table.")
    parser.add_argument(
        "--dim",
        action="append",
        required=True,
        help="Dimension mapping like 512=/path/to/mteb_results.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    dim_map = {}
    for item in args.dim:
        if "=" not in item:
            raise SystemExit("Use --dim 512=path syntax.")
        dim_str, path = item.split("=", 1)
        dim = int(dim_str)
        dim_map[dim] = Path(path)

    task_names = None
    results = []
    for dim, path in sorted(dim_map.items()):
        scores = _load_scores(path, args.split)
        if task_names is None:
            task_names = list(scores.keys())
            type_map = _task_type_map(task_names)
        else:
            type_map = _task_type_map(list(scores.keys()))
        type_scores = defaultdict(list)
        for name, score in scores.items():
            ttype = type_map.get(name, "unknown")
            type_scores[ttype].append(score)

        mean_task = _mean(list(scores.values())) * args.scale
        mean_task_type = _mean([_mean(vals) for vals in type_scores.values()]) * args.scale
        results.append(
            {
                "dim": dim,
                "mean_task": mean_task,
                "mean_task_type": mean_task_type,
                "tasks": len(scores),
                "task_types": len(type_scores),
            }
        )

    lines = [
        "Dimensionality\tMean (Task)\tMean (TaskType)",
    ]
    for row in results:
        lines.append(f"{row['dim']}d\t{row['mean_task']:.2f}\t{row['mean_task_type']:.2f}")
    print("\n".join(lines))

    if args.json_out:
        payload = {
            "results": results,
            "scale": args.scale,
            "split": args.split,
        }
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
