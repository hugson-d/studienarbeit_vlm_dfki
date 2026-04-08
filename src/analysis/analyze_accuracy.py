#!/usr/bin/env python3
"""Aggregate zero-shot result files into a compact CSV summary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("results/zero_shot/raw")
DEFAULT_OUTPUT_CSV = Path("results/zero_shot/summary/model_accuracy_summary.csv")


def is_zero_shot_file(path: Path) -> bool:
    name = path.name
    if path.suffix != ".jsonl":
        return False
    excluded_parts = (
        "CoT-Voting",
        "_n1",
        "_n5",
        "TempSweep",
        "null_prediction",
        "with_prediction",
        "results_2025_non_cot_voting",
    )
    return not any(part in name for part in excluded_parts)


def empty_stats() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "null_prediction": 0,
        "invalid_formats": 0,
        "missing_format_flag": 0,
    }


def update_stats(stats: dict[str, int], entry: dict) -> None:
    stats["total"] += 1

    prediction = entry.get("prediction")
    if prediction is None:
        stats["null_prediction"] += 1
    elif bool(entry.get("is_correct", False)):
        stats["correct"] += 1

    if "format_valid" in entry:
        if not bool(entry.get("format_valid")):
            stats["invalid_formats"] += 1
    else:
        stats["missing_format_flag"] += 1


def difficulty_from_task_id(task_id: str) -> str:
    if not task_id:
        return "unknown"
    final_part = task_id.split("_")[-1]
    if final_part.startswith("A"):
        return "A"
    if final_part.startswith("B"):
        return "B"
    if final_part.startswith("C"):
        return "C"
    return "unknown"


def stats_to_row(model: str, group_kind: str, group_value: str, stats: dict[str, int]) -> dict[str, str | int]:
    total = stats["total"]
    correct = stats["correct"]
    null_prediction = stats["null_prediction"]
    accuracy = (correct / total) if total else 0.0
    coverage = ((total - null_prediction) / total) if total else 0.0
    return {
        "model": model,
        "group_kind": group_kind,
        "group_value": group_value,
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy:.4f}",
        "null_prediction": null_prediction,
        "coverage": f"{coverage:.4f}",
        "invalid_formats": stats["invalid_formats"],
        "missing_format_flag": stats["missing_format_flag"],
    }


def aggregate_results(results_dir: Path) -> list[dict[str, str | int]]:
    files = sorted(p for p in results_dir.glob("*.jsonl") if is_zero_shot_file(p))
    if not files:
        raise FileNotFoundError(f"No zero-shot JSONL files found in {results_dir}")

    model_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {
        "overall": empty_stats(),
        "category": defaultdict(empty_stats),
        "difficulty": defaultdict(empty_stats),
    })

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                model = entry.get("model") or file_path.stem
                category = entry.get("math_category") or "unknown"
                difficulty = difficulty_from_task_id(entry.get("task_id", ""))

                update_stats(model_stats[model]["overall"], entry)
                update_stats(model_stats[model]["category"][category], entry)
                update_stats(model_stats[model]["difficulty"][difficulty], entry)

    rows: list[dict[str, str | int]] = []
    for model in sorted(model_stats):
        rows.append(stats_to_row(model, "overall", "ALL", model_stats[model]["overall"]))
        for category in sorted(model_stats[model]["category"]):
            rows.append(stats_to_row(model, "category", category, model_stats[model]["category"][category]))
        for difficulty in sorted(model_stats[model]["difficulty"]):
            rows.append(stats_to_row(model, "difficulty", difficulty, model_stats[model]["difficulty"][difficulty]))
    return rows


def write_csv(rows: list[dict[str, str | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "group_kind",
        "group_value",
        "total",
        "correct",
        "accuracy",
        "null_prediction",
        "coverage",
        "invalid_formats",
        "missing_format_flag",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate zero-shot JSONL result files into a summary CSV."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory with zero-shot JSONL files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = aggregate_results(args.results_dir)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
