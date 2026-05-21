"""Frequency-band loss reweighting for INR fitting.

Instead of replacing the supervision target with a blurred version (which
permanently removes information from the signal), this module keeps the
original RGB target intact and re-weights the per-pixel MSE according to the
target's local frequency content. A Laplacian-style pyramid of the target is
computed once, giving each pixel a vector of band magnitudes. A time-varying
scalar per band controls how strongly that band contributes to the loss.

Typical usage during training:

    band_pixel_weights = make_band_pixel_weights(target_img, band_sigmas)
    band_scalars = compute_band_scalars(
        progress=step / total_steps,
        num_bands=band_pixel_weights.shape[0],
        w_low_start=1.0, w_low_end=0.3,
        w_high_start=0.3, w_high_end=1.5,
    )
    pixel_weight = make_pixel_loss_weight(band_pixel_weights, band_scalars)
    loss = band_weighted_mse(pred_rgb, target_rgb, pixel_weight[indices])

The shape conventions match the rest of the project: images are ``[H, W, 3]``
in ``[0, 1]`` and pixel-indexed quantities are flattened to ``[H * W, ...]``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

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


def _validate_band_sigmas(band_sigmas: Sequence[float]) -> List[float]:
    """Validate and return a sorted descending list of positive sigmas.

    Band sigmas describe successive blur scales. ``band_sigmas[0]`` is the
    coarsest blur (lowpass cutoff) and the last entry is the sharpest blur.
    The function enforces a strictly decreasing positive sequence so that the
    pyramid bands are well defined.
    """
    if len(band_sigmas) == 0:
        raise ValueError("band_sigmas must contain at least one value.")
    sigmas = [float(sigma) for sigma in band_sigmas]
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("band_sigmas must be strictly positive.")
    for prev, current in zip(sigmas[:-1], sigmas[1:]):
        if current >= prev:
            raise ValueError(
                "band_sigmas must be strictly decreasing from coarse to fine, "
                f"got {sigmas}."
            )
    return sigmas


def _kernel_size_from_sigma(sigma: float) -> int:
    """Choose a standard odd Gaussian kernel size covering +/- 3 sigma."""
    kernel_size = int(math.ceil(6.0 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return max(kernel_size, 3)


def _gaussian_blur_hwc(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian blur to a [H, W, 3] image tensor."""
    if sigma <= 0.0:
        return image.clone()

    kernel_size = _kernel_size_from_sigma(sigma)
    radius = kernel_size // 2

    coords = torch.arange(
        kernel_size,
        dtype=image.dtype,
        device=image.device,
    ) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = image.shape[-1]
    horizontal_kernel = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
    vertical_kernel = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)

    nchw = image.permute(2, 0, 1).unsqueeze(0)
    blurred = F.pad(nchw, (radius, radius, 0, 0), mode="replicate")
    blurred = F.conv2d(blurred, horizontal_kernel, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
    blurred = F.conv2d(blurred, vertical_kernel, groups=channels)
    return blurred.squeeze(0).permute(1, 2, 0).contiguous()


def make_laplacian_pyramid(
    image: torch.Tensor,
    band_sigmas: Sequence[float],
) -> List[torch.Tensor]:
    """Decompose an image into a Laplacian-style pyramid.

    Given ``K`` strictly-decreasing positive sigmas, returns ``K + 1`` band
    images ordered from low to high frequency:

    - ``bands[0]    = blur(image, band_sigmas[0])``
    - ``bands[k]    = blur(image, band_sigmas[k]) - blur(image, band_sigmas[k - 1])``
      for ``1 <= k <= K - 1``
    - ``bands[K]    = image - blur(image, band_sigmas[K - 1])``

    The sum of all returned bands reconstructs the input image exactly (up to
    floating-point error). Each band has shape ``[H, W, 3]``.
    """
    _validate_image(image)
    sigmas = _validate_band_sigmas(band_sigmas)

    blurred_versions: List[torch.Tensor] = [
        _gaussian_blur_hwc(image, sigma) for sigma in sigmas
    ]

    bands: List[torch.Tensor] = [blurred_versions[0]]
    for index in range(1, len(sigmas)):
        bands.append(blurred_versions[index] - blurred_versions[index - 1])
    bands.append(image - blurred_versions[-1])
    return bands


def make_band_pixel_weights(
    image: torch.Tensor,
    band_sigmas: Sequence[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-pixel per-band magnitudes from a Laplacian pyramid.

    For each pyramid band, the per-pixel magnitude is the L2 norm across the
    RGB channels. Each band is normalized so that its per-pixel mean is 1, so
    the band scalars provided later are directly comparable across bands.

    Args:
        image: RGB image tensor with shape ``[H, W, 3]`` in ``[0, 1]``.
        band_sigmas: Strictly decreasing positive sigmas. See
            :func:`make_laplacian_pyramid`.
        eps: Small constant guarding the per-band normalization.

    Returns:
        Tensor with shape ``[num_bands, H * W]`` whose row ``k`` holds the
        normalized magnitude of band ``k`` at every pixel.
    """
    _validate_image(image)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    bands = make_laplacian_pyramid(image, band_sigmas)
    num_pixels = image.shape[0] * image.shape[1]

    magnitudes = []
    for band in bands:
        magnitude = band.reshape(num_pixels, 3).pow(2.0).sum(dim=-1).sqrt()
        mean_magnitude = magnitude.mean().clamp_min(eps)
        magnitudes.append(magnitude / mean_magnitude)

    return torch.stack(magnitudes, dim=0).contiguous()


def compute_band_scalars(
    progress: float,
    num_bands: int,
    w_low_start: float = 1.0,
    w_low_end: float = 0.3,
    w_high_start: float = 0.3,
    w_high_end: float = 1.5,
) -> torch.Tensor:
    """Compute per-band time-varying scalars ``w_k(progress)``.

    The scalars interpolate the lowest band linearly between ``w_low_start``
    and ``w_low_end``, the highest band between ``w_high_start`` and
    ``w_high_end``, and linearly fill in the intermediate bands. With default
    values, early training emphasizes the lowest band (smooth content) while
    late training emphasizes the highest band (edges and texture).

    The input ``progress`` is expected to already be eased to taste. Callers
    that want cosine easing should apply it via :func:`compute_progress`
    rather than asking this function to re-ease the value, which would
    compound the easing curve.

    Args:
        progress: Progress in ``[0, 1]``. Values outside the range are clamped.
        num_bands: Number of pyramid bands. Must be at least 1.
        w_low_start: Lowest-band scalar at ``progress = 0``. Non-negative.
        w_low_end: Lowest-band scalar at ``progress = 1``. Non-negative.
        w_high_start: Highest-band scalar at ``progress = 0``. Non-negative.
        w_high_end: Highest-band scalar at ``progress = 1``. Non-negative.

    Returns:
        Float tensor with shape ``[num_bands]``.
    """
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}.")
    for name, value in (
        ("w_low_start", w_low_start),
        ("w_low_end", w_low_end),
        ("w_high_start", w_high_start),
        ("w_high_end", w_high_end),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    clamped = min(max(float(progress), 0.0), 1.0)
    low_value = float(w_low_start) + (float(w_low_end) - float(w_low_start)) * clamped
    high_value = float(w_high_start) + (float(w_high_end) - float(w_high_start)) * clamped

    if num_bands == 1:
        return torch.tensor([0.5 * (low_value + high_value)], dtype=torch.float32)

    band_positions = torch.linspace(0.0, 1.0, steps=num_bands)
    scalars = (1.0 - band_positions) * low_value + band_positions * high_value
    return scalars.to(dtype=torch.float32).contiguous()


def make_pixel_loss_weight(
    band_pixel_weights: torch.Tensor,
    band_scalars: torch.Tensor,
    normalize_mean: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Combine per-band magnitudes with per-band scalars into one weight map.

    Args:
        band_pixel_weights: Tensor with shape ``[num_bands, H * W]`` from
            :func:`make_band_pixel_weights`.
        band_scalars: Tensor with shape ``[num_bands]`` from
            :func:`compute_band_scalars`.
        normalize_mean: If True, scale the result so its mean equals 1. This
            keeps the loss magnitude comparable across training steps.
        eps: Small constant guarding mean normalization.

    Returns:
        Float tensor with shape ``[H * W]``.
    """
    if band_pixel_weights.ndim != 2:
        raise ValueError(
            "band_pixel_weights must have shape [num_bands, H * W], "
            f"got {tuple(band_pixel_weights.shape)}."
        )
    if band_scalars.ndim != 1:
        raise ValueError(
            f"band_scalars must have shape [num_bands], got {tuple(band_scalars.shape)}."
        )
    if band_pixel_weights.shape[0] != band_scalars.shape[0]:
        raise ValueError(
            "band count mismatch between band_pixel_weights and band_scalars, "
            f"got {band_pixel_weights.shape[0]} and {band_scalars.shape[0]}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    scalars = band_scalars.to(
        device=band_pixel_weights.device,
        dtype=band_pixel_weights.dtype,
    ).view(-1, 1)
    weighted = (band_pixel_weights * scalars).sum(dim=0)

    weighted = weighted.clamp_min(0.0)
    if normalize_mean:
        mean_value = weighted.mean().clamp_min(eps)
        weighted = weighted / mean_value
    return weighted.contiguous()


def band_weighted_mse(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    pixel_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pixel weighted MSE between predicted and target RGB values.

    Args:
        pred_rgb: Predicted RGB tensor with shape ``[B, 3]``.
        target_rgb: Target RGB tensor with shape ``[B, 3]``.
        pixel_weight: Per-pixel weights with shape ``[B]``.

    Returns:
        Scalar tensor with the weighted mean squared error.
    """
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_rgb.shape)} vs "
            f"target {tuple(target_rgb.shape)}."
        )
    if pred_rgb.ndim != 2 or pred_rgb.shape[-1] != 3:
        raise ValueError(f"pred_rgb must have shape [B, 3], got {tuple(pred_rgb.shape)}.")
    if pixel_weight.ndim != 1 or pixel_weight.shape[0] != pred_rgb.shape[0]:
        raise ValueError(
            "pixel_weight must have shape [B] matching the batch, "
            f"got {tuple(pixel_weight.shape)} for batch {pred_rgb.shape[0]}."
        )

    per_pixel_mse = (pred_rgb - target_rgb).pow(2.0).mean(dim=-1)
    return (pixel_weight * per_pixel_mse).mean()


__all__ = [
    "make_laplacian_pyramid",
    "make_band_pixel_weights",
    "compute_band_scalars",
    "make_pixel_loss_weight",
    "band_weighted_mse",
]
