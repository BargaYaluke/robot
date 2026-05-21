"""Image reconstruction metrics for INR image fitting."""

<<<<<<< HEAD
from typing import Dict, Optional, Tuple

=======
>>>>>>> f610dac054b21fcc513794ac6426b207636e7b32
import numpy as np
import torch
from skimage.metrics import structural_similarity


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute PSNR between predicted and target RGB values.

    Args:
        pred: Predicted tensor in [0, 1], shaped [N, 3] or [H, W, 3].
        target: Target tensor in [0, 1], with the same shape as pred.
        eps: Small constant to avoid log10(0).

    Returns:
        PSNR value as a Python float.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}.")
    if pred.ndim not in (2, 3) or pred.shape[-1] != 3:
        raise ValueError(
            f"Expected shape [N, 3] or [H, W, 3], got {tuple(pred.shape)}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    # Detach so metric computation never enters the autograd graph.
    pred = pred.detach().float()
    target = target.detach().float().to(pred.device)

    mse = torch.mean((pred - target) ** 2)
    psnr = -10.0 * torch.log10(mse + eps)
    return float(psnr.detach().cpu().item())


def compute_ssim(pred_img: torch.Tensor, target_img: torch.Tensor) -> float:
    """Compute SSIM between reconstructed and target RGB images.

    Args:
        pred_img: Predicted image tensor with shape [H, W, 3] in [0, 1].
        target_img: Target image tensor with shape [H, W, 3] in [0, 1].

    Returns:
        SSIM value as a Python float.
    """
    if pred_img.shape != target_img.shape:
        raise ValueError(
            f"Shape mismatch: pred_img {tuple(pred_img.shape)} vs "
            f"target_img {tuple(target_img.shape)}."
        )
    if pred_img.ndim != 3 or pred_img.shape[-1] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {tuple(pred_img.shape)}.")
    min_side = min(pred_img.shape[0], pred_img.shape[1])
    if min_side < 3:
        raise ValueError("SSIM requires image height and width to be at least 3.")

    # Move tensors to CPU NumPy arrays for skimage. Clipping protects the metric
    # from tiny numerical excursions outside [0, 1].
    pred_np = pred_img.detach().float().cpu().numpy()
    target_np = target_img.detach().float().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    return float(
        structural_similarity(
            target_np,
            pred_np,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
        )
    )
<<<<<<< HEAD


def compute_region_psnr(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> Optional[float]:
    """Compute PSNR over a masked image region.

    Args:
        pred_img: Predicted image tensor with shape [H, W, 3] in [0, 1].
        target_img: Target image tensor with shape [H, W, 3] in [0, 1].
        mask: Boolean or 0/1 tensor with shape [H, W]. Pixels with True/nonzero
            values are included in the region metric.
        eps: Small constant to avoid log10(0).

    Returns:
        Region PSNR as a Python float, or None if the mask contains no pixels.
    """
    if pred_img.shape != target_img.shape:
        raise ValueError(
            f"Shape mismatch: pred_img {tuple(pred_img.shape)} vs "
            f"target_img {tuple(target_img.shape)}."
        )
    if pred_img.ndim != 3 or pred_img.shape[-1] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {tuple(pred_img.shape)}.")
    if mask.shape != pred_img.shape[:2]:
        raise ValueError(
            f"mask must have shape [H, W] matching the images, got {tuple(mask.shape)}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    pred = pred_img.detach().float()
    target = target_img.detach().float().to(pred.device)
    region_mask = mask.detach().to(device=pred.device).bool()

    num_region_pixels = int(region_mask.sum().detach().cpu().item())
    if num_region_pixels == 0:
        return None

    region_error = (pred - target).square()[region_mask]
    mse = region_error.mean()
    psnr = -10.0 * torch.log10(mse + eps)
    return float(psnr.detach().cpu().item())


def make_edge_smooth_masks(
    edge_map_2d: torch.Tensor,
    threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an edge map into edge and smooth-region masks.

    Args:
        edge_map_2d: Edge map tensor with shape [H, W] and values in [0, 1].
        threshold: Pixels above this value are treated as edge pixels.

    Returns:
        edge_mask: Boolean tensor where ``edge_map_2d > threshold``.
        smooth_mask: Boolean tensor where ``edge_map_2d <= threshold``.
    """
    if not isinstance(edge_map_2d, torch.Tensor):
        raise TypeError(
            f"edge_map_2d must be a torch.Tensor, got {type(edge_map_2d).__name__}."
        )
    if edge_map_2d.ndim != 2:
        raise ValueError(f"edge_map_2d must have shape [H, W], got {tuple(edge_map_2d.shape)}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

    edge_map = edge_map_2d.detach()
    edge_mask = edge_map > threshold
    smooth_mask = edge_map <= threshold
    return edge_mask, smooth_mask


def compute_edge_smooth_psnr(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    edge_map_2d: torch.Tensor,
    threshold: float = 0.2,
) -> Dict[str, Optional[float]]:
    """Compute PSNR separately for edge and smooth image regions.

    Args:
        pred_img: Predicted image tensor with shape [H, W, 3] in [0, 1].
        target_img: Target image tensor with shape [H, W, 3] in [0, 1].
        edge_map_2d: Normalized edge map tensor with shape [H, W].
        threshold: Edge/smooth split threshold.

    Returns:
        Dictionary with ``edge_psnr`` and ``smooth_psnr`` entries. Either value
        can be None if the corresponding mask is empty.
    """
    edge_mask, smooth_mask = make_edge_smooth_masks(edge_map_2d, threshold=threshold)
    return {
        "edge_psnr": compute_region_psnr(pred_img, target_img, edge_mask),
        "smooth_psnr": compute_region_psnr(pred_img, target_img, smooth_mask),
    }
=======
>>>>>>> f610dac054b21fcc513794ac6426b207636e7b32
