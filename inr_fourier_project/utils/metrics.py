"""Image reconstruction metrics for INR image fitting."""

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
