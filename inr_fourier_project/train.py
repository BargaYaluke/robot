"""Training entry point for single-image INR fitting.

This script fits one RGB image with either a vanilla coordinate MLP or a
Fourier Feature MLP. Coordinates are sampled uniformly at random during
training; no frequency curriculum or edge-aware sampling is implemented here.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import trange

from data.image_dataset import image_to_coord_rgb, load_image
from models import FourierMLP, VanillaMLP
from utils.metrics import compute_psnr, compute_ssim
from utils.seed import set_seed
from utils.visualization import (
    save_comparison_image,
    save_image_tensor,
    save_psnr_curve,
)


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

    return parser.parse_args()


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
    h: int,
    w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float]:
    """Reconstruct the full image and compute PSNR and SSIM."""
    pred_img = reconstruct(model, coords, h, w, device)
    psnr = compute_psnr(pred_img, target_img)
    ssim = compute_ssim(pred_img, target_img)
    return pred_img, psnr, ssim


def save_log_csv(records: List[Dict[str, float]], path: Path) -> None:
    """Save training/evaluation records as CSV."""
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "loss", "psnr", "ssim"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def record_metrics(
    records: List[Dict[str, float]],
    step: int,
    loss: float,
    psnr: float,
    ssim: float,
) -> None:
    """Append or update a metric row for a training step."""
    row = {
        "step": int(step),
        "loss": float(loss),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
    if records and int(records[-1]["step"]) == step:
        records[-1] = row
    else:
        records.append(row)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    metrics: List[Dict[str, float]],
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
    set_seed(args.seed)

    output_dirs = make_output_dirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_img = load_image(args.image_path, size=args.image_size)
    h, w, _ = target_img.shape
    coords, rgb = image_to_coord_rgb(target_img)
    coords = coords.to(device)
    rgb = rgb.to(device)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics: List[Dict[str, float]] = []
    psnr_records: List[Tuple[int, float]] = []
    latest_eval_step = 0
    latest_pred_img: Optional[torch.Tensor] = None
    latest_psnr = 0.0
    latest_ssim = 0.0

    print(f"Device: {device}")
    print(f"Image: {args.image_path} -> {h}x{w}")
    print(f"Model: {args.model_type}")

    progress = trange(1, args.num_steps + 1, desc="Training", dynamic_ncols=True)
    last_loss = 0.0

    for step in progress:
        # Uniform random coordinate sampling. This is the baseline sampler.
        indices = torch.randint(0, coords.shape[0], (args.batch_size,), device=device)
        batch_coords = coords[indices]
        batch_rgb = rgb[indices]

        pred_rgb = model(batch_coords)
        loss = F.mse_loss(pred_rgb, batch_rgb)
        last_loss = float(loss.detach().cpu().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{last_loss:.6f}")

        should_eval = step % args.eval_interval == 0 or step == args.num_steps
        if should_eval:
            pred_img, psnr, ssim = evaluate_full_image(
                model=model,
                coords=coords,
                target_img=target_img,
                h=h,
                w=w,
                device=device,
            )
            record_metrics(metrics, step, last_loss, psnr, ssim)
            psnr_records.append((step, psnr))
            save_log_csv(metrics, output_dirs["logs"] / "train_log.csv")
            latest_eval_step = step
            latest_pred_img = pred_img
            latest_psnr = psnr
            latest_ssim = ssim

            print(
                f"Step {step:06d} | "
                f"Loss {last_loss:.6f} | "
                f"PSNR {psnr:.4f} | "
                f"SSIM {ssim:.6f}"
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
    else:
        final_img, final_psnr, final_ssim = evaluate_full_image(
            model=model,
            coords=coords,
            target_img=target_img,
            h=h,
            w=w,
            device=device,
        )
    record_metrics(metrics, args.num_steps, last_loss, final_psnr, final_ssim)
    if not psnr_records or psnr_records[-1][0] != args.num_steps:
        psnr_records.append((args.num_steps, final_psnr))

    save_image_tensor(final_img, str(output_dirs["reconstructions"] / "final_reconstruction.png"))
    save_comparison_image(
        gt=target_img,
        pred=final_img,
        path=str(output_dirs["reconstructions"] / "final_comparison.png"),
    )
    save_psnr_curve(psnr_records, str(output_dirs["curves"] / "psnr_curve.png"))
    save_log_csv(metrics, output_dirs["logs"] / "train_log.csv")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        args=args,
        metrics=metrics,
        path=output_dirs["root"] / "model_checkpoint.pt",
    )

    print(
        f"Final | Loss {last_loss:.6f} | "
        f"PSNR {final_psnr:.4f} | SSIM {final_ssim:.6f}"
    )
    print(f"Outputs saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
