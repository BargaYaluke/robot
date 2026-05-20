"""Edge-aware coordinate sampling utilities for INR image fitting.

This module builds a Sobel edge map from an RGB target image and exposes
samplers that can mix uniform coordinate sampling with edge-biased sampling.
The functions are intentionally independent from the training script so they
can be introduced into experiments without changing the baseline pipeline.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _validate_image(image: torch.Tensor) -> None:
    """Validate the expected INR image layout."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}.")
    if image.ndim != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {tuple(image.shape)}.")
    if image.shape[-1] != 3:
        raise ValueError(f"image must have 3 RGB channels, got shape {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise TypeError("image must be a floating point tensor with values in [0, 1].")


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use a convolution-friendly floating dtype for Sobel filtering."""
    if dtype in (torch.float32, torch.float64):
        return dtype
    return torch.float32


def compute_sobel_edge_map(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a normalized Sobel edge map from an RGB image.

    Args:
        image: RGB image tensor with shape ``[H, W, 3]`` and values in
            ``[0, 1]``.

    Returns:
        edge_map_2d: Tensor with shape ``[H, W]`` and values in ``[0, 1]``.
        edge_map_flat: Flattened tensor with shape ``[H * W]``.

    Notes:
        The returned tensors stay on the same device as ``image``. For low
        precision inputs, Sobel filtering is computed in float32 for better
        numerical and operator compatibility.
    """
    _validate_image(image)

    device = image.device
    output_dtype = image.dtype
    work_dtype = _compute_dtype(image.dtype)
    image_work = image.to(dtype=work_dtype)

    rgb_weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=work_dtype,
        device=device,
    )
    gray = (image_work * rgb_weights).sum(dim=-1)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=work_dtype,
        device=device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=work_dtype,
        device=device,
    ).view(1, 1, 3, 3)

    gray_nchw = gray.unsqueeze(0).unsqueeze(0)
    gray_padded = F.pad(gray_nchw, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(gray_padded, sobel_x)
    grad_y = F.conv2d(gray_padded, sobel_y)

    edge = torch.sqrt(grad_x.square() + grad_y.square()).squeeze(0).squeeze(0)
    edge = edge / edge.max().clamp_min(torch.finfo(edge.dtype).eps)

    edge_map_2d = edge.to(dtype=output_dtype).contiguous()
    edge_map_flat = edge_map_2d.reshape(-1).contiguous()
    return edge_map_2d, edge_map_flat


def make_sampling_prob(
    edge_map_flat: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 1.0,
) -> torch.Tensor:
    """Create a normalized edge-aware sampling distribution.

    The unnormalized probability is ``p = alpha + beta * edge_map_flat``.
    ``alpha`` keeps smooth regions sampleable, while ``beta`` controls how
    strongly edge regions are emphasized.

    Args:
        edge_map_flat: Flattened edge map with shape ``[H * W]``.
        alpha: Uniform probability floor. Must be non-negative.
        beta: Edge weighting factor. Must be non-negative.

    Returns:
        A probability tensor with shape ``[H * W]`` whose sum is 1.
    """
    if not isinstance(edge_map_flat, torch.Tensor):
        raise TypeError(
            f"edge_map_flat must be a torch.Tensor, got {type(edge_map_flat).__name__}."
        )
    if edge_map_flat.ndim != 1:
        raise ValueError(f"edge_map_flat must have shape [H * W], got {tuple(edge_map_flat.shape)}.")
    if edge_map_flat.numel() == 0:
        raise ValueError("edge_map_flat must contain at least one value.")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")
    if beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {beta}.")

    prob = float(alpha) + float(beta) * edge_map_flat.float()
    prob = prob.to(device=edge_map_flat.device)
    total = prob.sum()
    if float(total.detach().cpu().item()) <= 0.0:
        raise ValueError("sampling probability has zero mass; increase alpha or beta.")

    return (prob / total).contiguous()


def sample_uniform(num_pixels: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample flattened pixel indices uniformly at random.

    Args:
        num_pixels: Number of pixels in the flattened coordinate grid.
        batch_size: Number of indices to sample.
        device: Device where returned indices should live.

    Returns:
        Long tensor of shape ``[batch_size]`` on ``device``.
    """
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}.")
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}.")

    return torch.randint(0, num_pixels, (batch_size,), device=device, dtype=torch.long)


def sample_edge_aware(
    prob: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample flattened pixel indices using an edge-aware probability vector.

    Args:
        prob: Probability tensor with shape ``[H * W]``.
        batch_size: Number of indices to sample.
        device: Device where sampling and returned indices should live.

    Returns:
        Long tensor of shape ``[batch_size]`` on ``device``.
    """
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be a torch.Tensor, got {type(prob).__name__}.")
    if prob.ndim != 1:
        raise ValueError(f"prob must have shape [H * W], got {tuple(prob.shape)}.")
    if prob.numel() == 0:
        raise ValueError("prob must contain at least one value.")
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}.")

    prob_device = prob.to(device=device, dtype=torch.float32)
    return torch.multinomial(prob_device, batch_size, replacement=True).to(device=device)


def sample_mixed(
    edge_prob: torch.Tensor,
    num_pixels: int,
    batch_size: int,
    device: torch.device,
    edge_ratio: float = 0.5,
) -> torch.Tensor:
    """Sample a batch using both uniform and edge-aware distributions.

    Args:
        edge_prob: Edge-aware probability tensor with shape ``[H * W]``.
        num_pixels: Number of pixels in the flattened coordinate grid.
        batch_size: Total number of indices to sample.
        device: Device where returned indices should live.
        edge_ratio: Fraction of the batch sampled from ``edge_prob``. The rest
            is sampled uniformly. Must be in ``[0, 1]``.

    Returns:
        Long tensor of shape ``[batch_size]`` on ``device``.
    """
    if not 0.0 <= edge_ratio <= 1.0:
        raise ValueError(f"edge_ratio must be in [0, 1], got {edge_ratio}.")
    if not isinstance(edge_prob, torch.Tensor):
        raise TypeError(f"edge_prob must be a torch.Tensor, got {type(edge_prob).__name__}.")
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}.")
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}.")
    if edge_prob.numel() != num_pixels:
        raise ValueError(
            f"edge_prob length must match num_pixels, got {edge_prob.numel()} and {num_pixels}."
        )

    if 0.0 < edge_ratio < 1.0 and batch_size < 2:
        raise ValueError("mixed sampling requires batch_size >= 2.")

    edge_count = int(round(batch_size * edge_ratio))
    if 0.0 < edge_ratio < 1.0:
        edge_count = min(max(edge_count, 1), batch_size - 1)
    uniform_count = batch_size - edge_count

    uniform_indices = sample_uniform(num_pixels, uniform_count, device)
    edge_indices = sample_edge_aware(edge_prob, edge_count, device)

    if uniform_count == 0:
        return edge_indices
    if edge_count == 0:
        return uniform_indices

    return torch.cat([uniform_indices, edge_indices], dim=0).to(device=device)


__all__ = [
    "compute_sobel_edge_map",
    "make_sampling_prob",
    "sample_uniform",
    "sample_edge_aware",
    "sample_mixed",
]
