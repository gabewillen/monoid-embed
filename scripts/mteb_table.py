#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import numpy as np

from mteb.get_tasks import get_task
from mteb.results import TaskResult


def _load_scores(result_dir: str, split: str, language: str | None) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name in os.listdir(result_dir):
        if not name.endswith(".json") or name == "model_meta.json":
            continue
        path = os.path.join(result_dir, name)
        task_result = TaskResult.from_disk(path)
        if language:
            score = task_result.get_score(splits=[split], languages=[language])
        else:
            score = task_result.get_score(splits=[split])
        scores[task_result.task_name] = float(score)
    return scores


def _summarize(scores: dict[str, float]) -> tuple[float, float]:
    if not scores:
        raise ValueError("No task scores found to summarize.")
    mean_task = float(np.mean(list(scores.values())))
    by_type: dict[str, list[float]] = defaultdict(list)
    for task_name, score in scores.items():
        try:
            task_type = get_task(task_name).metadata.type
        except Exception:
            continue
        by_type[task_type].append(score)
    if not by_type:
        raise ValueError("No task types found to summarize.")
    mean_by_type = float(np.mean([np.mean(vals) for vals in by_type.values()]))
    return mean_task, mean_by_type


def _format_table(rows: list[dict[str, str]]) -> str:
    headers = ["Dimensionality", "Mean (Task)", "Mean (TaskType)"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join([row[h] for h in headers]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MTEB results into an EmbeddingGemma-style table.")
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Mapping of label=path for a result dir (repeatable). Example: --entry 512d=path/to/results",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--language", type=str, default="eng")
    parser.add_argument("--scale", type=float, default=100.0, help="Multiply scores by this factor (default 100).")
    args = parser.parse_args()

    if not args.entry:
        raise SystemExit("No entries provided. Use --entry label=path.")

    rows: list[dict[str, str]] = []
    for entry in args.entry:
        if "=" not in entry:
            raise SystemExit(f"Invalid entry: {entry}. Use label=path.")
        label, path = entry.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"Result directory not found: {path}")
        scores = _load_scores(path, args.split, args.language)
        mean_task, mean_task_type = _summarize(scores)
        rows.append(
            {
                "Dimensionality": label,
                "Mean (Task)": f"{mean_task * args.scale:.2f}",
                "Mean (TaskType)": f"{mean_task_type * args.scale:.2f}",
            }
        )

    print(_format_table(rows))


if __name__ == "__main__":
    main()
