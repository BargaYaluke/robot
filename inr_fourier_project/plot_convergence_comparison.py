"""Plot PSNR convergence comparisons for standard INR experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "baseline",
    "frequency_curriculum",
    "edge_sampling",
    "full_method",
]

METHOD_LABELS = {
    "baseline": "Baseline",
    "frequency_curriculum": "Frequency Curriculum",
    "edge_sampling": "Edge Sampling",
    "full_method": "Full Method",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from standard INR experiment metrics."
    )
    parser.add_argument("--experiment_root", type=str, required=True)
    return parser.parse_args()


def parse_optional_float(value: str) -> Optional[float]:
    """Parse a CSV float field that may be empty or None-like."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def read_metrics_csv(path: Path) -> List[Dict[str, Optional[float]]]:
    """Read step, PSNR, and edge PSNR values from one metrics CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    records: List[Dict[str, Optional[float]]] = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                {
                    "step": int(row["step"]),
                    "psnr": parse_optional_float(row.get("psnr")),
                    "edge_psnr": parse_optional_float(row.get("edge_psnr")),
                }
            )
    return records


def load_all_metrics(experiment_root: Path) -> Dict[str, List[Dict[str, Optional[float]]]]:
    """Load metrics for all standard methods in fixed order."""
    metrics = {}
    for method in METHOD_ORDER:
        metrics_path = experiment_root / method / "metrics.csv"
        metrics[method] = read_metrics_csv(metrics_path)
    return metrics


def plot_metric(
    all_metrics: Dict[str, List[Dict[str, Optional[float]]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one metric against training steps for all methods."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted_any = False

    for method in METHOD_ORDER:
        records = all_metrics[method]
        points = [
            (record["step"], record[metric_key])
            for record in records
            if record.get(metric_key) is not None
        ]
        if not points:
            continue

        steps = [point[0] for point in points]
        values = [point[1] for point in points]
        ax.plot(steps, values, linewidth=2, label=METHOD_LABELS[method])
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Create PSNR and edge-PSNR convergence comparison figures."""
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    all_metrics = load_all_metrics(experiment_root)

    plot_metric(
        all_metrics=all_metrics,
        metric_key="psnr",
        ylabel="PSNR",
        title="PSNR Convergence Comparison",
        output_path=experiment_root / "psnr_convergence_comparison.png",
    )
    plot_metric(
        all_metrics=all_metrics,
        metric_key="edge_psnr",
        ylabel="Edge PSNR",
        title="Edge PSNR Convergence Comparison",
        output_path=experiment_root / "edge_psnr_convergence_comparison.png",
    )

    print(f"Saved: {experiment_root / 'psnr_convergence_comparison.png'}")
    print(f"Saved: {experiment_root / 'edge_psnr_convergence_comparison.png'}")


if __name__ == "__main__":
    main()
