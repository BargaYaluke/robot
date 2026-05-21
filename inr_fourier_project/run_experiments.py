"""Run standard single-image INR experiments.

This script launches four Fourier MLP settings on one image:
baseline, frequency curriculum only, edge sampling only, and the full method.
It intentionally uses subprocess calls to keep each experiment isolated and to
reuse the same training entry point used by manual runs.

Frequency curriculum experiments use a short blended warm-up by default, and
the full method delays edge-aware sampling until the original-image curriculum
stage.
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
    parser.add_argument("--curriculum_sigmas", type=str, default="4.0,2.0,1.0,0.0")
    parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--curriculum_blend_ratio", type=float, default=0.5)
    parser.add_argument("--curriculum_stage_ratios", type=str, default="")
    parser.add_argument("--edge_start_ratio", type=float, default=0.5)
    parser.add_argument("--edge_ratio", type=float, default=0.5)
    return parser.parse_args()


def parse_float_list(value: str, name: str) -> List[float]:
    """Parse a comma-separated float list for experiment-level validation."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc


def ensure_final_original_sigma(sigmas: Sequence[float]) -> List[float]:
    """Mirror train.py behavior so experiment flags fail early."""
    sigma_list = [float(sigma) for sigma in sigmas]
    if not sigma_list:
        raise ValueError("curriculum_sigmas must contain at least one value.")
    if sigma_list[-1] != 0.0:
        sigma_list.append(0.0)
    return sigma_list


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
    curriculum_sigmas = ensure_final_original_sigma(
        parse_float_list(args.curriculum_sigmas, "curriculum_sigmas")
    )
    if any(sigma < 0.0 for sigma in curriculum_sigmas):
        raise ValueError("curriculum_sigmas must be non-negative.")
    if not 0.0 <= args.curriculum_warmup_ratio < 1.0:
        raise ValueError("curriculum_warmup_ratio must be in [0, 1).")
    if not 0.0 <= args.curriculum_blend_ratio <= 1.0:
        raise ValueError("curriculum_blend_ratio must be in [0, 1].")
    if args.curriculum_stage_ratios.strip():
        stage_ratios = parse_float_list(
            args.curriculum_stage_ratios,
            "curriculum_stage_ratios",
        )
        if len(stage_ratios) != len(curriculum_sigmas):
            raise ValueError(
                "curriculum_stage_ratios length must match curriculum_sigmas "
                f"after appending 0.0 if needed, got {len(stage_ratios)} and "
                f"{len(curriculum_sigmas)}."
            )
        if any(ratio < 0.0 for ratio in stage_ratios):
            raise ValueError("curriculum_stage_ratios must be non-negative.")
        if sum(stage_ratios) <= 0.0:
            raise ValueError(
                "curriculum_stage_ratios must contain positive total duration."
            )
        if stage_ratios[-1] <= 0.0:
            raise ValueError(
                "The final original-image curriculum stage must have positive duration."
            )
    if not 0.0 <= args.edge_start_ratio <= 1.0:
        raise ValueError("edge_start_ratio must be in [0, 1].")
    if not 0.0 <= args.edge_ratio <= 1.0:
        raise ValueError("edge_ratio must be in [0, 1].")


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


def curriculum_flags(args: argparse.Namespace) -> List[str]:
    """Build the recommended short blended curriculum flags."""
    flags = [
        "--use_frequency_curriculum",
        "--curriculum_sigmas",
        args.curriculum_sigmas,
        "--use_blended_curriculum",
        "--curriculum_blend_ratio",
        str(args.curriculum_blend_ratio),
    ]
    if args.curriculum_stage_ratios.strip():
        flags.extend(["--curriculum_stage_ratios", args.curriculum_stage_ratios])
    else:
        flags.extend(["--curriculum_warmup_ratio", str(args.curriculum_warmup_ratio)])
    return flags


def edge_sampling_flags(args: argparse.Namespace) -> List[str]:
    """Build edge-aware sampling flags shared by edge and full experiments."""
    return [
        "--use_edge_sampling",
        "--edge_start_ratio",
        str(args.edge_start_ratio),
        "--edge_ratio",
        str(args.edge_ratio),
    ]


def experiment_configs(args: argparse.Namespace) -> Dict[str, Sequence[str]]:
    """Return experiment-specific command-line flags."""
    return {
        "baseline": [],
        "frequency_curriculum": curriculum_flags(args),
        "edge_sampling": edge_sampling_flags(args),
        "full_method": (
            curriculum_flags(args)
            + edge_sampling_flags(args)
            + ["--align_edge_start_to_original"]
        ),
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
    for name, extra_flags in experiment_configs(args).items():
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
