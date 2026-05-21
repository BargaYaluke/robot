"""Run standard single-image INR ablation experiments.

This script launches a fixed grid of Fourier MLP configurations on one image
and supports running each configuration across multiple seeds. The grid covers
the original four settings used in earlier reports (baseline, blur-target
frequency curriculum, step-flip edge sampling, and the old "full" combination)
plus three new arms introduced after the v1 findings:

- ``band_loss``: frequency-band loss reweighting (no edge sampling).
- ``ramped_edge``: edge-aware sampling with a smooth ramp from step 0
  (no curriculum).
- ``coupled``: band-loss reweighting combined with ramped edge sampling, both
  driven by the same progress window.

Each configuration calls ``train.py`` as a subprocess so runs remain isolated
and the manual entry point keeps working unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


EVAL_INTERVAL = 100
SAVE_INTERVAL = 500

DEFAULT_METHODS = (
    "baseline",
    "frequency_curriculum",
    "edge_sampling",
    "full_method",
    "band_loss",
    "ramped_edge",
    "coupled",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-image ablation runs."""
    parser = argparse.ArgumentParser(
        description="Run standard single-image INR ablation experiments."
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_root", type=str, default="experiment_results")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated list of integer seeds. One subdir per seed when len > 1.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names to run. Use any subset of the default grid.",
    )

    # Blur-target curriculum (legacy).
    parser.add_argument("--curriculum_sigmas", type=str, default="4.0,2.0,1.0,0.0")
    parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--curriculum_blend_ratio", type=float, default=0.5)
    parser.add_argument("--curriculum_stage_ratios", type=str, default="")

    # Step-flip edge sampling (legacy).
    parser.add_argument("--edge_start_ratio", type=float, default=0.5)
    parser.add_argument("--edge_ratio", type=float, default=0.5)

    # Frequency-band loss (new).
    parser.add_argument("--band_sigmas", type=str, default="8.0,4.0,2.0")
    parser.add_argument("--band_w_low_start", type=float, default=1.0)
    parser.add_argument("--band_w_low_end", type=float, default=0.3)
    parser.add_argument("--band_w_high_start", type=float, default=0.3)
    parser.add_argument("--band_w_high_end", type=float, default=1.5)
    parser.add_argument(
        "--band_loss_mode",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
    )

    # Edge ramp (new).
    parser.add_argument("--edge_ramp_start_ratio", type=float, default=0.0)
    parser.add_argument("--edge_ramp_end_ratio", type=float, default=1.0)
    parser.add_argument(
        "--edge_ramp_mode",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
    )

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


def parse_int_list(value: str, name: str) -> List[int]:
    """Parse a comma-separated integer list."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers.") from exc


def parse_method_list(value: str) -> List[str]:
    """Parse and validate a comma-separated method-name list."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("methods must contain at least one value.")
    unknown = [name for name in items if name not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(
            "Unknown method name(s): "
            + ", ".join(unknown)
            + f". Expected any of {list(DEFAULT_METHODS)}."
        )
    # Preserve user-specified order while removing duplicates.
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


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

    band_sigmas = parse_float_list(args.band_sigmas, "band_sigmas")
    if any(sigma <= 0.0 for sigma in band_sigmas):
        raise ValueError("band_sigmas must be strictly positive.")
    for prev, current in zip(band_sigmas[:-1], band_sigmas[1:]):
        if current >= prev:
            raise ValueError(
                "band_sigmas must be strictly decreasing from coarse to fine."
            )
    for name in (
        "band_w_low_start",
        "band_w_low_end",
        "band_w_high_start",
        "band_w_high_end",
    ):
        value = getattr(args, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    if not 0.0 <= args.edge_ramp_start_ratio <= 1.0:
        raise ValueError("edge_ramp_start_ratio must be in [0, 1].")
    if not 0.0 <= args.edge_ramp_end_ratio <= 1.0:
        raise ValueError("edge_ramp_end_ratio must be in [0, 1].")
    if args.edge_ramp_end_ratio < args.edge_ramp_start_ratio:
        raise ValueError("edge_ramp_end_ratio must be >= edge_ramp_start_ratio.")

    parse_int_list(args.seeds, "seeds")
    parse_method_list(args.methods)


def base_train_command(
    args: argparse.Namespace,
    train_script: Path,
    output_dir: Path,
    seed: int,
) -> List[str]:
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
        str(seed),
        "--output_dir",
        str(output_dir),
    ]


def curriculum_flags(args: argparse.Namespace) -> List[str]:
    """Build the legacy short blended blur-target curriculum flags."""
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


def step_flip_edge_flags(args: argparse.Namespace) -> List[str]:
    """Build the legacy step-flip edge-aware sampling flags."""
    return [
        "--use_edge_sampling",
        "--edge_start_ratio",
        str(args.edge_start_ratio),
        "--edge_ratio",
        str(args.edge_ratio),
    ]


def band_loss_flags(args: argparse.Namespace) -> List[str]:
    """Build the frequency-band loss reweighting flags."""
    return [
        "--use_band_loss",
        "--band_sigmas",
        args.band_sigmas,
        "--band_w_low_start",
        str(args.band_w_low_start),
        "--band_w_low_end",
        str(args.band_w_low_end),
        "--band_w_high_start",
        str(args.band_w_high_start),
        "--band_w_high_end",
        str(args.band_w_high_end),
        "--band_loss_mode",
        args.band_loss_mode,
    ]


def ramped_edge_flags(args: argparse.Namespace) -> List[str]:
    """Build the ramped edge-aware sampling flags."""
    return [
        "--use_edge_sampling",
        "--edge_ratio",
        str(args.edge_ratio),
        "--use_ramped_edge",
        "--edge_ramp_start_ratio",
        str(args.edge_ramp_start_ratio),
        "--edge_ramp_end_ratio",
        str(args.edge_ramp_end_ratio),
        "--edge_ramp_mode",
        args.edge_ramp_mode,
    ]


def experiment_configs(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Return experiment-specific command-line flags."""
    return {
        "baseline": [],
        "frequency_curriculum": curriculum_flags(args),
        "edge_sampling": step_flip_edge_flags(args),
        "full_method": (
            curriculum_flags(args)
            + step_flip_edge_flags(args)
            + ["--align_edge_start_to_original"]
        ),
        "band_loss": band_loss_flags(args),
        "ramped_edge": ramped_edge_flags(args),
        "coupled": (
            band_loss_flags(args)
            + ramped_edge_flags(args)
            + ["--coupled_schedule"]
        ),
    }


def run_experiment(name: str, command: Sequence[str], cwd: Path) -> None:
    """Run one experiment and raise if training fails."""
    print(f"\n=== Running {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    """Launch all selected experiments across all seeds."""
    args = parse_args()
    validate_args(args)

    project_dir = Path(__file__).resolve().parent
    train_script = project_dir / "train.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds, "seeds")
    methods = parse_method_list(args.methods)
    configs = experiment_configs(args)

    multi_seed = len(seeds) > 1
    summary_paths: Dict[str, Path] = {}

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}" if multi_seed else output_root
        seed_root.mkdir(parents=True, exist_ok=True)
        for name in methods:
            extra_flags = configs[name]
            method_dir = seed_root / name
            command = base_train_command(args, train_script, method_dir, seed)
            command.extend(extra_flags)
            run_experiment(
                name=f"{name} (seed={seed})" if multi_seed else name,
                command=command,
                cwd=project_dir,
            )
            summary_paths[f"{name}@{seed}"] = method_dir / "summary.json"

    print("\nSummary files:")
    for label, summary_path in summary_paths.items():
        print(f"{label}: {summary_path}")


if __name__ == "__main__":
    main()
