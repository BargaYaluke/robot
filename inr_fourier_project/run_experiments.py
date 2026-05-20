"""Run standard single-image INR experiments.

This script launches four Fourier MLP settings on one image:
baseline, frequency curriculum only, edge sampling only, and the full method.
It intentionally uses subprocess calls to keep each experiment isolated and to
reuse the same training entry point used by manual runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


EVAL_INTERVAL = 100
SAVE_INTERVAL = 500


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-image experiment runs."""
    parser = argparse.ArgumentParser(
        description="Run standard single-image INR experiments."
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_root", type=str, default="experiment_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid experiment settings."""
    if args.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")


def base_train_command(args: argparse.Namespace, train_script: Path, output_dir: Path) -> List[str]:
    """Build command arguments shared by all experiment settings."""
    return [
        sys.executable,
        str(train_script),
        "--image_path",
        args.image_path,
        "--model_type",
        "fourier_mlp",
        "--image_size",
        str(args.image_size),
        "--num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--eval_interval",
        str(EVAL_INTERVAL),
        "--save_interval",
        str(SAVE_INTERVAL),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
    ]


def experiment_configs() -> Dict[str, Sequence[str]]:
    """Return experiment-specific command-line flags."""
    return {
        "baseline": [],
        "frequency_curriculum": ["--use_frequency_curriculum"],
        "edge_sampling": ["--use_edge_sampling"],
        "full_method": ["--use_frequency_curriculum", "--use_edge_sampling"],
    }


def run_experiment(name: str, command: Sequence[str], cwd: Path) -> None:
    """Run one experiment and raise if training fails."""
    print(f"\n=== Running {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    """Launch all standard experiments and print summary locations."""
    args = parse_args()
    validate_args(args)

    project_dir = Path(__file__).resolve().parent
    train_script = project_dir / "train.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_paths: Dict[str, Path] = {}
    for name, extra_flags in experiment_configs().items():
        output_dir = output_root / name
        command = base_train_command(args, train_script, output_dir)
        command.extend(extra_flags)
        run_experiment(name, command, cwd=project_dir)
        summary_paths[name] = output_dir / "summary.json"

    print("\nSummary files:")
    for name, summary_path in summary_paths.items():
        print(f"{name}: {summary_path}")


if __name__ == "__main__":
    main()
