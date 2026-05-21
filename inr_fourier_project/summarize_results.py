"""Summarize standard INR experiment outputs into a comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


METHOD_ORDER = [
    "baseline",
    "frequency_curriculum",
    "edge_sampling",
    "full_method",
    "band_loss",
    "ramped_edge",
    "coupled",
]

TABLE_COLUMNS = [
    "method",
    "final_psnr",
    "final_ssim",
    "best_psnr",
    "best_psnr_step",
    "final_edge_psnr",
    "final_smooth_psnr",
    "iter_to_30db",
    "iter_to_35db",
    "iter_to_38db",
    "iter_to_40db",
    "total_training_time",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a CSV and markdown comparison table from INR experiment summaries."
    )
    parser.add_argument("--experiment_root", type=str, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, object]:
    """Load one summary JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_value(summary: Dict[str, object], key: str) -> Optional[object]:
    """Read a summary value with compatibility for the table column names."""
    if key == "total_training_time":
        return summary.get("total_training_time_seconds")
    return summary.get(key)


def make_row(method: str, summary: Dict[str, object]) -> Dict[str, object]:
    """Convert one summary dictionary into a flat table row."""
    row: Dict[str, object] = {"method": method}
    for column in TABLE_COLUMNS:
        if column == "method":
            continue
        row[column] = get_value(summary, column)
    return row


def load_rows(experiment_root: Path) -> List[Dict[str, object]]:
    """Load summaries for any methods present under ``experiment_root``.

    Methods listed in :data:`METHOD_ORDER` appear first and in that order.
    Any additional method subdirectories with a ``summary.json`` file are
    appended afterwards. Missing methods from ``METHOD_ORDER`` are skipped so
    the script works with partial ablation runs.
    """
    rows: List[Dict[str, object]] = []
    seen = set()
    for method in METHOD_ORDER:
        summary_path = experiment_root / method / "summary.json"
        if not summary_path.exists():
            continue
        rows.append(make_row(method, load_summary(summary_path)))
        seen.add(method)

    if experiment_root.is_dir():
        for entry in sorted(experiment_root.iterdir()):
            if not entry.is_dir() or entry.name in seen:
                continue
            summary_path = entry / "summary.json"
            if not summary_path.exists():
                continue
            rows.append(make_row(entry.name, load_summary(summary_path)))

    if not rows:
        raise FileNotFoundError(
            f"No method subdirectory with summary.json found under {experiment_root}."
        )
    return rows


def save_comparison_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    """Save comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_value(value: object) -> str:
    """Format values compactly for terminal markdown output."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_markdown_table(rows: Sequence[Dict[str, object]]) -> None:
    """Print a clean markdown table to stdout."""
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |"
    print(header)
    print(separator)
    for row in rows:
        values = [format_markdown_value(row.get(column)) for column in TABLE_COLUMNS]
        print("| " + " | ".join(values) + " |")


def main() -> None:
    """Create comparison_table.csv and print a markdown table."""
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    rows = load_rows(experiment_root)

    output_path = experiment_root / "comparison_table.csv"
    save_comparison_csv(rows, output_path)

    print_markdown_table(rows)
    print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
