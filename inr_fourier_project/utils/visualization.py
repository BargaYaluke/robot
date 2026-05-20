"""Visualization and result-saving helpers for INR image fitting."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for an output file if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_image_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a [H, W, 3] tensor in [0, 1] to a clamped NumPy image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {tuple(image.shape)}.")

    image_np = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    return image_np


def save_image_tensor(image: torch.Tensor, path: str) -> None:
    """Save an RGB image tensor to disk.

    Args:
        image: Tensor with shape [H, W, 3] and values in [0, 1].
        path: Output image path.
    """
    _ensure_parent_dir(path)
    image_np = _to_image_numpy(image)
    plt.imsave(path, image_np)


def save_psnr_curve(psnr_records: Sequence[Tuple[int, float]], path: str) -> None:
    """Save a PSNR convergence curve.

    Args:
        psnr_records: Sequence of (step, psnr) tuples.
        path: Output figure path.
    """
    if len(psnr_records) == 0:
        raise ValueError("psnr_records must contain at least one (step, psnr) tuple.")

    _ensure_parent_dir(path)

    steps = [record[0] for record in psnr_records]
    psnr_values = [record[1] for record in psnr_records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, psnr_values, linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR Convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_comparison_image(gt: torch.Tensor, pred: torch.Tensor, path: str) -> None:
    """Save a side-by-side image comparison.

    The saved figure contains ground truth, reconstruction, and an absolute
    RGB error map averaged over color channels.

    Args:
        gt: Ground truth image tensor with shape [H, W, 3].
        pred: Reconstructed image tensor with shape [H, W, 3].
        path: Output figure path.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)}.")

    _ensure_parent_dir(path)

    gt_np = _to_image_numpy(gt)
    pred_np = _to_image_numpy(pred)
    error_map = np.mean(np.abs(gt_np - pred_np), axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels: List[Tuple[str, np.ndarray, Optional[str]]] = [
        ("Ground Truth", gt_np, None),
        ("Reconstruction", pred_np, None),
        ("Absolute Error", error_map, "magma"),
    ]

    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
