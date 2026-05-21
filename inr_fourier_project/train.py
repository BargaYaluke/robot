"""Training entry point for single-image INR fitting.

This script fits one RGB image with either a vanilla coordinate MLP or a
Fourier Feature MLP. The default path keeps the original uniform-sampling
baseline, while optional flags enable frequency curriculum learning and
edge-aware coordinate sampling for second-stage experiments.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import trange

from data.image_dataset import image_to_coord_rgb, load_image
from methods.edge_sampler import (
    compute_sobel_edge_map,
    make_sampling_prob,
    sample_edge_aware,
    sample_mixed,
    sample_uniform,
)
from methods.frequency_curriculum import (
    get_blended_curriculum_target,
    get_curriculum_stage,
    get_curriculum_target,
    make_blur_pyramid,
)
from models import FourierMLP, VanillaMLP
from utils.convergence import compute_iterations_to_targets, summarize_convergence
from utils.metrics import compute_edge_smooth_psnr, compute_psnr, compute_ssim
from utils.seed import set_seed
from utils.visualization import (
    save_comparison_image,
    save_heatmap,
    save_image_tensor,
    save_psnr_curve,
    save_sampling_points_visualization,
)

METRIC_FIELDNAMES = [
    "step",
    "train_loss",
    "psnr",
    "ssim",
    "edge_psnr",
    "smooth_psnr",
    "current_curriculum_stage",
    "sampling_mode",
    "elapsed_time_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-image fitting."""
    parser = argparse.ArgumentParser(description="Train an INR on a single image.")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="fourier_mlp",
        choices=["vanilla_mlp", "fourier_mlp"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mapping_size", type=int, default=256)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")

    parser.add_argument("--use_frequency_curriculum", action="store_true")
    parser.add_argument("--curriculum_sigmas", type=str, default="4.0,2.0,1.0,0.0")
    parser.add_argument("--curriculum_stage_ratios", type=str, default="")
    parser.add_argument("--curriculum_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--use_blended_curriculum", action="store_true")
    parser.add_argument("--curriculum_blend_ratio", type=float, default=0.5)

    parser.add_argument("--use_edge_sampling", action="store_true")
    parser.add_argument("--edge_start_ratio", type=float, default=0.5)
    parser.add_argument("--align_edge_start_to_original", action="store_true")
    parser.add_argument("--edge_alpha", type=float, default=0.2)
    parser.add_argument("--edge_beta", type=float, default=1.0)
    parser.add_argument("--edge_ratio", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.2)

    parser.add_argument("--target_psnrs", type=str, default="30,35,38,40")

    return parser.parse_args()


def parse_float_list(value: str, name: str) -> List[float]:
    """Parse a comma-separated list of floats from a command-line argument."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")

    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc


def ensure_final_original_sigma(sigmas: Sequence[float]) -> List[float]:
    """Ensure the last curriculum stage is the unblurred original image."""
    sigma_list = [float(sigma) for sigma in sigmas]
    if not sigma_list:
        raise ValueError("curriculum_sigmas must contain at least one value.")
    if sigma_list[-1] != 0.0:
        sigma_list.append(0.0)
    return sigma_list


def resolve_curriculum_stage_ratios(
    sigmas: Sequence[float],
    explicit_ratios: str,
    warmup_ratio: float,
) -> List[float]:
    """Resolve custom or default stage durations for frequency curriculum.

    By default, all blurred stages share a short warm-up window and the final
    original-image stage receives the remaining budget. This keeps curriculum
    supervision from consuming most of a limited training run.
    """
    num_stages = len(sigmas)
    if num_stages == 0:
        raise ValueError("curriculum_sigmas must contain at least one value.")

    if explicit_ratios.strip():
        ratios = parse_float_list(explicit_ratios, "curriculum_stage_ratios")
        if len(ratios) != num_stages:
            raise ValueError(
                "curriculum_stage_ratios length must match curriculum_sigmas "
                f"after appending 0.0 if needed, got {len(ratios)} and {num_stages}."
            )
    elif num_stages == 1:
        ratios = [1.0]
    else:
        blurred_stage_count = num_stages - 1
        blurred_ratio = warmup_ratio / float(blurred_stage_count)
        ratios = [blurred_ratio] * blurred_stage_count + [1.0 - warmup_ratio]

    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("curriculum_stage_ratios must be non-negative.")
    if sum(ratios) <= 0.0:
        raise ValueError("curriculum_stage_ratios must contain positive total duration.")
    if ratios[-1] <= 0.0:
        raise ValueError(
            "The final original-image curriculum stage must have positive duration."
        )

    total = sum(ratios)
    return [ratio / total for ratio in ratios]


def get_original_stage_start_ratio(stage_ratios: Sequence[float]) -> float:
    """Return the progress ratio where the final original-image stage starts."""
    if not stage_ratios:
        raise ValueError("stage_ratios must contain at least one value.")
    total = sum(float(ratio) for ratio in stage_ratios)
    if total <= 0.0:
        raise ValueError("stage_ratios must contain positive total duration.")
    return sum(float(ratio) for ratio in stage_ratios[:-1]) / total


def resolve_effective_edge_start_ratio(
    args: argparse.Namespace,
    curriculum_stage_ratios: Optional[Sequence[float]],
) -> float:
    """Apply optional edge/curriculum alignment for full-method experiments."""
    edge_start_ratio = float(args.edge_start_ratio)
    if (
        args.use_edge_sampling
        and args.use_frequency_curriculum
        and args.align_edge_start_to_original
        and curriculum_stage_ratios is not None
    ):
        edge_start_ratio = max(
            edge_start_ratio,
            get_original_stage_start_ratio(curriculum_stage_ratios),
        )
    return edge_start_ratio


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid training arguments."""
    if args.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if args.mapping_size <= 0:
        raise ValueError("mapping_size must be positive.")
    if args.scale <= 0.0:
        raise ValueError("scale must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if args.save_interval < 0:
        raise ValueError("save_interval must be non-negative.")
    if not 0.0 <= args.curriculum_blend_ratio <= 1.0:
        raise ValueError("curriculum_blend_ratio must be in [0, 1].")
    if not 0.0 <= args.curriculum_warmup_ratio < 1.0:
        raise ValueError("curriculum_warmup_ratio must be in [0, 1).")
    if not 0.0 <= args.edge_start_ratio <= 1.0:
        raise ValueError("edge_start_ratio must be in [0, 1].")
    if args.edge_alpha < 0.0:
        raise ValueError("edge_alpha must be non-negative.")
    if args.edge_beta < 0.0:
        raise ValueError("edge_beta must be non-negative.")
    if args.use_edge_sampling and args.edge_alpha == 0.0 and args.edge_beta == 0.0:
        raise ValueError(
            "edge_alpha and edge_beta cannot both be zero when edge sampling is enabled."
        )
    if not 0.0 <= args.edge_ratio <= 1.0:
        raise ValueError("edge_ratio must be in [0, 1].")
    if args.use_edge_sampling and 0.0 < args.edge_ratio < 1.0 and args.batch_size < 2:
        raise ValueError("mixed edge sampling requires batch_size >= 2.")
    if not 0.0 <= args.edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be in [0, 1].")

    curriculum_sigmas = parse_float_list(args.curriculum_sigmas, "curriculum_sigmas")
    if any(sigma < 0.0 for sigma in curriculum_sigmas):
        raise ValueError("curriculum_sigmas must be non-negative.")
    resolve_curriculum_stage_ratios(
        sigmas=ensure_final_original_sigma(curriculum_sigmas),
        explicit_ratios=args.curriculum_stage_ratios,
        warmup_ratio=args.curriculum_warmup_ratio,
    )

    target_psnrs = parse_float_list(args.target_psnrs, "target_psnrs")
    if any(target <= 0.0 for target in target_psnrs):
        raise ValueError("target_psnrs must be positive.")


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build the requested INR model."""
    if args.model_type == "vanilla_mlp":
        return VanillaMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )

    if args.model_type == "fourier_mlp":
        return FourierMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            mapping_size=args.mapping_size,
            scale=args.scale,
        )

    raise ValueError(f"Unsupported model_type: {args.model_type}")


def make_output_dirs(output_dir: str) -> Dict[str, Path]:
    """Create and return standard output directories."""
    root = Path(output_dir)
    dirs = {
        "root": root,
        "reconstructions": root / "reconstructions",
        "curves": root / "curves",
        "logs": root / "logs",
        "visualizations": root / "visualizations",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    coords: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """Run full-coordinate inference and return an image tensor [H, W, 3]."""
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [H * W, 2], got {tuple(coords.shape)}.")
    if coords.shape[0] != h * w:
        raise ValueError(f"Expected {h * w} coordinates, got {coords.shape[0]}.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    was_training = model.training
    model.eval()

    chunks = []
    for start in range(0, coords.shape[0], chunk_size):
        coord_chunk = coords[start : start + chunk_size].to(device)
        chunks.append(model(coord_chunk).detach().cpu())

    pred_rgb = torch.cat(chunks, dim=0)
    pred_img = pred_rgb.reshape(h, w, 3).clamp(0.0, 1.0)

    if was_training:
        model.train()

    return pred_img


def evaluate_full_image(
    model: torch.nn.Module,
    coords: torch.Tensor,
    target_img: torch.Tensor,
    edge_map_2d: torch.Tensor,
    edge_threshold: float,
    h: int,
    w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float, Optional[float], Optional[float]]:
    """Reconstruct the full image and compute global and region metrics."""
    pred_img = reconstruct(model, coords, h, w, device)
    psnr = compute_psnr(pred_img, target_img)
    ssim = compute_ssim(pred_img, target_img)
    region_psnr = compute_edge_smooth_psnr(
        pred_img=pred_img,
        target_img=target_img,
        edge_map_2d=edge_map_2d,
        threshold=edge_threshold,
    )
    return pred_img, psnr, ssim, region_psnr["edge_psnr"], region_psnr["smooth_psnr"]


def save_metrics_csv(records: List[Dict[str, object]], path: Path) -> None:
    """Save structured metric records as a table-friendly CSV file."""
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def save_json(data: object, path: Path) -> None:
    """Save JSON data with stable formatting for later analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def record_metrics(
    records: List[Dict[str, object]],
    step: int,
    train_loss: float,
    psnr: float,
    ssim: float,
    edge_psnr: Optional[float],
    smooth_psnr: Optional[float],
    current_curriculum_stage: Optional[int],
    sampling_mode: str,
    elapsed_time_seconds: float,
) -> None:
    """Append or update a metric row for a training step."""
    row: Dict[str, object] = {
        "step": int(step),
        "train_loss": float(train_loss),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "edge_psnr": None if edge_psnr is None else float(edge_psnr),
        "smooth_psnr": None if smooth_psnr is None else float(smooth_psnr),
        "current_curriculum_stage": current_curriculum_stage,
        "sampling_mode": sampling_mode,
        "elapsed_time_seconds": float(elapsed_time_seconds),
    }
    if records and int(records[-1]["step"]) == step:
        records[-1] = row
    else:
        records.append(row)


def format_optional_metric(value: Optional[float]) -> str:
    """Format optional metrics for compact console logging."""
    if value is None:
        return "None"
    return f"{value:.4f}"


def flatten_targets_for_training(
    targets: Sequence[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Convert HWC curriculum targets into flattened RGB tensors on device."""
    return [target.reshape(-1, 3).float().contiguous().to(device) for target in targets]


def get_sampling_mode(
    use_edge_sampling: bool,
    step: int,
    edge_start_step: int,
    edge_ratio: float,
) -> str:
    """Return the active sampling mode label for a training step."""
    if not use_edge_sampling or step <= edge_start_step or edge_ratio <= 0.0:
        return "uniform"
    if edge_ratio >= 1.0:
        return "edge_aware"
    return "mixed_edge"


def get_curriculum_progress_step(step: int, total_steps: int) -> int:
    """Map one-based training steps to curriculum progress.

    Non-final steps use ``step - 1`` so equal-length stages keep intuitive
    one-based boundaries. The final step is forced to ``total_steps`` so the
    last curriculum stage always trains on the original image.
    """
    if step >= total_steps:
        return total_steps
    return max(step - 1, 0)


def get_current_curriculum_stage(
    step: int,
    total_steps: int,
    curriculum_rgbs: Optional[Sequence[torch.Tensor]],
    curriculum_stage_ratios: Optional[Sequence[float]],
) -> Optional[int]:
    """Return the active curriculum stage, or None when curriculum is disabled."""
    if curriculum_rgbs is None:
        return None
    progress_step = get_curriculum_progress_step(step, total_steps)
    return get_curriculum_stage(
        step=progress_step,
        total_steps=total_steps,
        num_stages=len(curriculum_rgbs),
        stage_ratios=curriculum_stage_ratios,
    )


def get_sampling_visualization_steps(num_steps: int, edge_start_step: int) -> List[int]:
    """Choose representative steps for edge-sampling point visualizations."""
    first_edge_step = edge_start_step + 1
    if first_edge_step > num_steps:
        return []

    middle_edge_step = (first_edge_step + num_steps) // 2
    return sorted({first_edge_step, middle_edge_step, num_steps})


def build_hyperparameter_summary(
    args: argparse.Namespace,
    curriculum_sigmas: Sequence[float],
    curriculum_stage_ratios: Optional[Sequence[float]],
    target_psnrs: Sequence[float],
    edge_start_step: int,
    effective_edge_start_ratio: float,
    h: int,
    w: int,
    device: torch.device,
) -> Dict[str, object]:
    """Collect run settings in JSON-friendly form for summary tables."""
    hyperparameters = vars(args).copy()
    hyperparameters["curriculum_sigmas"] = list(curriculum_sigmas)
    hyperparameters["curriculum_stage_ratios"] = (
        None if curriculum_stage_ratios is None else list(curriculum_stage_ratios)
    )
    hyperparameters["target_psnrs"] = list(target_psnrs)
    hyperparameters["edge_start_step"] = int(edge_start_step)
    hyperparameters["effective_edge_start_ratio"] = float(effective_edge_start_ratio)
    hyperparameters["image_height"] = int(h)
    hyperparameters["image_width"] = int(w)
    hyperparameters["device"] = str(device)
    return hyperparameters


def build_training_summary(
    psnr_records: Sequence[Tuple[int, float]],
    final_psnr: float,
    final_ssim: float,
    final_edge_psnr: Optional[float],
    final_smooth_psnr: Optional[float],
    total_training_time_seconds: float,
    hyperparameters: Dict[str, object],
    target_psnrs: Sequence[float],
) -> Dict[str, object]:
    """Create the final compact summary used for experiment comparison."""
    convergence = summarize_convergence(psnr_records)
    iterations_to_targets = compute_iterations_to_targets(psnr_records)
    custom_iterations_to_targets = compute_iterations_to_targets(psnr_records, target_psnrs)

    summary: Dict[str, object] = {
        "final_psnr": float(final_psnr),
        "final_ssim": float(final_ssim),
        "best_psnr": convergence["best_psnr"],
        "best_psnr_step": convergence["best_step"],
        "final_edge_psnr": None if final_edge_psnr is None else float(final_edge_psnr),
        "final_smooth_psnr": None if final_smooth_psnr is None else float(final_smooth_psnr),
        "total_training_time_seconds": float(total_training_time_seconds),
        "hyperparameters": hyperparameters,
    }
    summary.update(iterations_to_targets)
    if custom_iterations_to_targets != iterations_to_targets:
        summary["custom_iterations_to_targets"] = custom_iterations_to_targets
    return summary


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    metrics: List[Dict[str, object]],
    path: Path,
) -> None:
    """Save model, optimizer, config, and metric history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def main() -> None:
    """Train a coordinate-based INR model on one image."""
    args = parse_args()
    validate_args(args)
    curriculum_sigmas = ensure_final_original_sigma(
        parse_float_list(args.curriculum_sigmas, "curriculum_sigmas")
    )
    curriculum_stage_ratios = (
        resolve_curriculum_stage_ratios(
            sigmas=curriculum_sigmas,
            explicit_ratios=args.curriculum_stage_ratios,
            warmup_ratio=args.curriculum_warmup_ratio,
        )
        if args.use_frequency_curriculum
        else None
    )
    target_psnrs = parse_float_list(args.target_psnrs, "target_psnrs")
    set_seed(args.seed)

    output_dirs = make_output_dirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_img = load_image(args.image_path, size=args.image_size)
    h, w, _ = target_img.shape
    coords, rgb = image_to_coord_rgb(target_img)
    coords = coords.to(device)
    rgb = rgb.to(device)

    edge_map_2d, edge_map_flat = compute_sobel_edge_map(target_img)
    edge_prob: Optional[torch.Tensor] = None
    if args.use_edge_sampling:
        edge_prob = make_sampling_prob(
            edge_map_flat,
            alpha=args.edge_alpha,
            beta=args.edge_beta,
        ).to(device)
    effective_edge_start_ratio = resolve_effective_edge_start_ratio(
        args=args,
        curriculum_stage_ratios=curriculum_stage_ratios,
    )
    edge_start_step = int(effective_edge_start_ratio * args.num_steps)
    sampling_visualization_steps = set(
        get_sampling_visualization_steps(args.num_steps, edge_start_step)
    )

    if args.use_edge_sampling:
        save_heatmap(
            edge_map_2d,
            str(output_dirs["visualizations"] / "edge_map.png"),
            title="Sobel Edge Map",
        )
        if edge_prob is not None:
            save_heatmap(
                edge_prob.reshape(h, w),
                str(output_dirs["visualizations"] / "sampling_probability.png"),
                title="Sampling Probability",
            )

    curriculum_rgbs: Optional[List[torch.Tensor]] = None
    if args.use_frequency_curriculum:
        curriculum_targets = make_blur_pyramid(target_img, sigmas=curriculum_sigmas)
        curriculum_rgbs = flatten_targets_for_training(curriculum_targets, device)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics: List[Dict[str, object]] = []
    psnr_records: List[Tuple[int, float]] = []
    latest_eval_step = 0
    latest_pred_img: Optional[torch.Tensor] = None
    latest_psnr = 0.0
    latest_ssim = 0.0
    latest_edge_psnr: Optional[float] = None
    latest_smooth_psnr: Optional[float] = None

    print(f"Device: {device}")
    print(f"Image: {args.image_path} -> {h}x{w}")
    print(f"Model: {args.model_type}")
    print(f"Frequency curriculum: {args.use_frequency_curriculum}")
    if curriculum_stage_ratios is not None:
        print(f"Curriculum sigmas: {curriculum_sigmas}")
        print(f"Curriculum stage ratios: {curriculum_stage_ratios}")
    print(f"Edge-aware sampling: {args.use_edge_sampling}")
    if args.use_edge_sampling:
        print(f"Edge start ratio: {effective_edge_start_ratio:.4f}")

    progress = trange(1, args.num_steps + 1, desc="Training", dynamic_ncols=True)
    last_loss = 0.0
    training_start_time = time.perf_counter()

    for step in progress:
        sampling_mode = get_sampling_mode(
            use_edge_sampling=args.use_edge_sampling,
            step=step,
            edge_start_step=edge_start_step,
            edge_ratio=args.edge_ratio,
        )
        if sampling_mode == "mixed_edge":
            if edge_prob is None:
                raise RuntimeError(
                    "edge_prob must be initialized when edge sampling is enabled."
                )
            indices = sample_mixed(
                edge_prob=edge_prob,
                num_pixels=coords.shape[0],
                batch_size=args.batch_size,
                device=device,
                edge_ratio=args.edge_ratio,
            )
        elif sampling_mode == "edge_aware":
            if edge_prob is None:
                raise RuntimeError(
                    "edge_prob must be initialized when edge sampling is enabled."
                )
            indices = sample_edge_aware(
                prob=edge_prob,
                batch_size=args.batch_size,
                device=device,
            )
        else:
            # Uniform random coordinate sampling. This is the baseline sampler.
            indices = sample_uniform(coords.shape[0], args.batch_size, device)

        if (
            args.use_edge_sampling
            and sampling_mode != "uniform"
            and step in sampling_visualization_steps
        ):
            save_sampling_points_visualization(
                image=target_img,
                sampled_indices=indices,
                h=h,
                w=w,
                path=str(
                    output_dirs["visualizations"]
                    / f"sampling_points_step_{step:06d}.png"
                ),
            )

        current_curriculum_stage = get_current_curriculum_stage(
            step=step,
            total_steps=args.num_steps,
            curriculum_rgbs=curriculum_rgbs,
            curriculum_stage_ratios=curriculum_stage_ratios,
        )
        curriculum_progress_step = get_curriculum_progress_step(step, args.num_steps)
        if curriculum_rgbs is None:
            current_rgb = rgb
        elif args.use_blended_curriculum:
            current_rgb = get_blended_curriculum_target(
                step=curriculum_progress_step,
                total_steps=args.num_steps,
                targets=curriculum_rgbs,
                blend_ratio=args.curriculum_blend_ratio,
                stage_ratios=curriculum_stage_ratios,
            )
        else:
            current_rgb = get_curriculum_target(
                step=curriculum_progress_step,
                total_steps=args.num_steps,
                targets=curriculum_rgbs,
                stage_ratios=curriculum_stage_ratios,
            )

        batch_coords = coords[indices]
        batch_rgb = current_rgb[indices]

        pred_rgb = model(batch_coords)
        loss = F.mse_loss(pred_rgb, batch_rgb)
        last_loss = float(loss.detach().cpu().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{last_loss:.6f}")

        should_eval = step % args.eval_interval == 0 or step == args.num_steps
        if should_eval:
            pred_img, psnr, ssim, edge_psnr, smooth_psnr = evaluate_full_image(
                model=model,
                coords=coords,
                target_img=target_img,
                edge_map_2d=edge_map_2d,
                edge_threshold=args.edge_threshold,
                h=h,
                w=w,
                device=device,
            )
            elapsed_time_seconds = time.perf_counter() - training_start_time
            record_metrics(
                records=metrics,
                step=step,
                train_loss=last_loss,
                psnr=psnr,
                ssim=ssim,
                edge_psnr=edge_psnr,
                smooth_psnr=smooth_psnr,
                current_curriculum_stage=current_curriculum_stage,
                sampling_mode=sampling_mode,
                elapsed_time_seconds=elapsed_time_seconds,
            )
            psnr_records.append((step, psnr))
            save_metrics_csv(metrics, output_dirs["root"] / "metrics.csv")
            save_json(metrics, output_dirs["root"] / "metrics.json")
            latest_eval_step = step
            latest_pred_img = pred_img
            latest_psnr = psnr
            latest_ssim = ssim
            latest_edge_psnr = edge_psnr
            latest_smooth_psnr = smooth_psnr

            print(
                f"Step {step:06d} | "
                f"Loss {last_loss:.6f} | "
                f"PSNR {psnr:.4f} | "
                f"SSIM {ssim:.6f} | "
                f"Edge PSNR {format_optional_metric(edge_psnr)} | "
                f"Smooth PSNR {format_optional_metric(smooth_psnr)}"
            )

        should_save = args.save_interval > 0 and step % args.save_interval == 0
        if should_save:
            pred_img = reconstruct(model, coords, h, w, device)
            save_image_tensor(
                pred_img,
                str(output_dirs["reconstructions"] / f"reconstruction_step_{step:06d}.png"),
            )

    if latest_eval_step == args.num_steps and latest_pred_img is not None:
        final_img = latest_pred_img
        final_psnr = latest_psnr
        final_ssim = latest_ssim
        final_edge_psnr = latest_edge_psnr
        final_smooth_psnr = latest_smooth_psnr
    else:
        (
            final_img,
            final_psnr,
            final_ssim,
            final_edge_psnr,
            final_smooth_psnr,
        ) = evaluate_full_image(
            model=model,
            coords=coords,
            target_img=target_img,
            edge_map_2d=edge_map_2d,
            edge_threshold=args.edge_threshold,
            h=h,
            w=w,
            device=device,
        )
    total_training_time_seconds = time.perf_counter() - training_start_time
    final_curriculum_stage = get_current_curriculum_stage(
        step=args.num_steps,
        total_steps=args.num_steps,
        curriculum_rgbs=curriculum_rgbs,
        curriculum_stage_ratios=curriculum_stage_ratios,
    )
    final_sampling_mode = get_sampling_mode(
        use_edge_sampling=args.use_edge_sampling,
        step=args.num_steps,
        edge_start_step=edge_start_step,
        edge_ratio=args.edge_ratio,
    )
    record_metrics(
        records=metrics,
        step=args.num_steps,
        train_loss=last_loss,
        psnr=final_psnr,
        ssim=final_ssim,
        edge_psnr=final_edge_psnr,
        smooth_psnr=final_smooth_psnr,
        current_curriculum_stage=final_curriculum_stage,
        sampling_mode=final_sampling_mode,
        elapsed_time_seconds=total_training_time_seconds,
    )
    if not psnr_records or psnr_records[-1][0] != args.num_steps:
        psnr_records.append((args.num_steps, final_psnr))

    hyperparameters = build_hyperparameter_summary(
        args=args,
        curriculum_sigmas=curriculum_sigmas,
        curriculum_stage_ratios=curriculum_stage_ratios,
        target_psnrs=target_psnrs,
        edge_start_step=edge_start_step,
        effective_edge_start_ratio=effective_edge_start_ratio,
        h=h,
        w=w,
        device=device,
    )
    summary = build_training_summary(
        psnr_records=psnr_records,
        final_psnr=final_psnr,
        final_ssim=final_ssim,
        final_edge_psnr=final_edge_psnr,
        final_smooth_psnr=final_smooth_psnr,
        total_training_time_seconds=total_training_time_seconds,
        hyperparameters=hyperparameters,
        target_psnrs=target_psnrs,
    )

    save_image_tensor(final_img, str(output_dirs["reconstructions"] / "final_reconstruction.png"))
    save_comparison_image(
        gt=target_img,
        pred=final_img,
        path=str(output_dirs["reconstructions"] / "final_comparison.png"),
    )
    save_psnr_curve(psnr_records, str(output_dirs["curves"] / "psnr_curve.png"))
    save_metrics_csv(metrics, output_dirs["root"] / "metrics.csv")
    save_json(metrics, output_dirs["root"] / "metrics.json")
    save_json(summary, output_dirs["root"] / "summary.json")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        args=args,
        metrics=metrics,
        path=output_dirs["root"] / "model_checkpoint.pt",
    )

    print(
        f"Final | Loss {last_loss:.6f} | "
        f"PSNR {final_psnr:.4f} | SSIM {final_ssim:.6f} | "
        f"Edge PSNR {format_optional_metric(final_edge_psnr)} | "
        f"Smooth PSNR {format_optional_metric(final_smooth_psnr)}"
    )
    print(f"Outputs saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
