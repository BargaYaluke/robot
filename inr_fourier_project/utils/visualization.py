"""Visualization and result-saving helpers for INR image fitting."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


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


def _to_numpy_array(array: ArrayLike) -> np.ndarray:
    """Convert a tensor or NumPy array to a detached NumPy array."""
    if isinstance(array, torch.Tensor):
        tensor = array.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    if isinstance(array, np.ndarray):
        return array
    raise TypeError(f"array must be a torch.Tensor or np.ndarray, got {type(array).__name__}.")


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


def save_heatmap(array: ArrayLike, path: str, title: Optional[str] = None) -> None:
    """Save a 2D heatmap with a colorbar.

    Args:
        array: Tensor or NumPy array with shape [H, W].
        path: Output figure path.
        title: Optional title shown above the heatmap.
    """
    array_np = _to_numpy_array(array)
    if array_np.ndim != 2:
        raise ValueError(f"Expected heatmap array shape [H, W], got {tuple(array_np.shape)}.")

    _ensure_parent_dir(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(array_np, cmap="magma")
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_sampling_points_visualization(
    image: ArrayLike,
    sampled_indices: ArrayLike,
    h: int,
    w: int,
    path: str,
    max_points: int = 3000,
) -> None:
    """Overlay sampled flattened coordinate indices on the original image.

    Args:
        image: RGB image tensor or NumPy array with shape [H, W, 3].
        sampled_indices: Flattened sampled pixel indices.
        h: Image height.
        w: Image width.
        path: Output figure path.
        max_points: Maximum number of points to draw for readability.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got h={h}, w={w}.")
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}.")

    if isinstance(image, torch.Tensor):
        image_np = _to_image_numpy(image)
    else:
        image_np = np.asarray(image, dtype=np.float32)
        if image_np.ndim != 3 or image_np.shape[-1] != 3:
            raise ValueError(f"Expected image shape [H, W, 3], got {tuple(image_np.shape)}.")
        image_np = np.clip(image_np, 0.0, 1.0)

    if image_np.shape[:2] != (h, w):
        raise ValueError(f"Image shape {image_np.shape[:2]} does not match h={h}, w={w}.")

    indices = _to_numpy_array(sampled_indices).reshape(-1).astype(np.int64)
    if indices.size == 0:
        raise ValueError("sampled_indices must contain at least one index.")
    if np.any(indices < 0) or np.any(indices >= h * w):
        raise ValueError("sampled_indices contains values outside the flattened image range.")

    if indices.size > max_points:
        selected = np.random.choice(indices.size, size=max_points, replace=False)
        indices = indices[selected]

    ys = indices // w
    xs = indices % w

    _ensure_parent_dir(path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.scatter(xs, ys, s=4, c="#00E5FF", alpha=0.65, linewidths=0)
    ax.set_title("Sampled Coordinates")
    ax.axis("off")
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
