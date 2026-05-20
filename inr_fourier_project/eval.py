"""Evaluation entry point for trained INR image fitting checkpoints."""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from data.image_dataset import image_to_coord_rgb, load_image
from models import FourierMLP, VanillaMLP
from utils.metrics import compute_psnr, compute_ssim
from utils.visualization import save_comparison_image, save_image_tensor


ARCH_KEYS = ("model_type", "hidden_dim", "num_layers", "mapping_size", "scale")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained INR checkpoint.")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="fourier_mlp",
        choices=["vanilla_mlp", "fourier_mlp"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mapping_size", type=int, default=256)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--output_dir", type=str, default="results/eval")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid evaluation arguments."""
    if args.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if args.mapping_size <= 0:
        raise ValueError("mapping_size must be positive.")
    if args.scale <= 0.0:
        raise ValueError("scale must be positive.")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint dictionary on the requested device."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to be a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    return checkpoint


def resolve_arch_config(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model architecture, preferring training config saved in checkpoint."""
    train_args = checkpoint.get("args", {})
    if train_args is None:
        train_args = {}
    if not isinstance(train_args, dict):
        raise TypeError("Checkpoint field 'args' must be a dictionary when present.")

    arch = {
        "model_type": checkpoint.get("model_type", train_args.get("model_type", args.model_type)),
        "hidden_dim": train_args.get("hidden_dim", args.hidden_dim),
        "num_layers": train_args.get("num_layers", args.num_layers),
        "mapping_size": train_args.get("mapping_size", args.mapping_size),
        "scale": train_args.get("scale", args.scale),
    }

    model_type = arch["model_type"]
    relevant_keys = ("model_type", "hidden_dim", "num_layers")
    if model_type == "fourier_mlp":
        relevant_keys = ARCH_KEYS

    for key in relevant_keys:
        cli_value = getattr(args, key)
        if key in train_args and train_args[key] != cli_value:
            print(
                f"Using checkpoint {key}={train_args[key]} "
                f"instead of CLI value {cli_value} to match training."
            )

    return arch


def build_model(arch: Dict[str, Any]) -> torch.nn.Module:
    """Rebuild the same model architecture used during training."""
    model_type = arch["model_type"]

    if model_type == "vanilla_mlp":
        return VanillaMLP(
            hidden_dim=int(arch["hidden_dim"]),
            num_layers=int(arch["num_layers"]),
        )

    if model_type == "fourier_mlp":
        return FourierMLP(
            hidden_dim=int(arch["hidden_dim"]),
            num_layers=int(arch["num_layers"]),
            mapping_size=int(arch["mapping_size"]),
            scale=float(arch["scale"]),
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


@torch.no_grad()
def reconstruct(
    model: torch.nn.Module,
    coords: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
    chunk_size: int = 65536,
) -> torch.Tensor:
    """Run full-coordinate inference and return a reconstructed [H, W, 3] image."""
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [H * W, 2], got {tuple(coords.shape)}.")
    if coords.shape[0] != h * w:
        raise ValueError(f"Expected {h * w} coordinates, got {coords.shape[0]}.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    model.eval()
    chunks = []
    for start in range(0, coords.shape[0], chunk_size):
        coord_chunk = coords[start : start + chunk_size].to(device)
        chunks.append(model(coord_chunk).detach().cpu())

    pred_rgb = torch.cat(chunks, dim=0)
    return pred_rgb.reshape(h, w, 3).clamp(0.0, 1.0)


def save_metrics_text(psnr: float, ssim: float, path: Path) -> None:
    """Save PSNR and SSIM to a small text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"PSNR: {psnr:.6f}\n")
        file.write(f"SSIM: {ssim:.6f}\n")


def main() -> None:
    """Evaluate a trained coordinate-based INR model."""
    args = parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint_path, device)
    arch = resolve_arch_config(args, checkpoint)

    target_img = load_image(args.image_path, size=args.image_size)
    h, w, _ = target_img.shape
    coords, _ = image_to_coord_rgb(target_img)

    model = build_model(arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_img = reconstruct(model, coords, h, w, device)
    psnr = compute_psnr(pred_img, target_img)
    ssim = compute_ssim(pred_img, target_img)

    reconstruction_path = output_dir / "eval_reconstruction.png"
    comparison_path = output_dir / "eval_comparison.png"
    metrics_path = output_dir / "eval_metrics.txt"

    save_image_tensor(pred_img, str(reconstruction_path))
    save_comparison_image(target_img, pred_img, str(comparison_path))
    save_metrics_text(psnr, ssim, metrics_path)

    print(f"Device: {device}")
    print(f"Image: {args.image_path} -> {h}x{w}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Model: {arch['model_type']}")
    print(f"PSNR: {psnr:.6f}")
    print(f"SSIM: {ssim:.6f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
