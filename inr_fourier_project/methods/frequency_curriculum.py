"""Frequency curriculum targets for single-image INR fitting.

The curriculum starts with low-frequency versions of an image and gradually
moves toward the original target. Training code can use these helpers to swap
or blend targets without changing the model itself.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


DEFAULT_SIGMAS = [4.0, 2.0, 1.0, 0.0]


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


def _kernel_size_from_sigma(sigma: float) -> int:
    """Choose a standard odd Gaussian kernel size covering +/- 3 sigma."""
    kernel_size = int(math.ceil(6.0 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return max(kernel_size, 3)


def _gaussian_blur_hwc(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to an image tensor with shape [H, W, 3]."""
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

    # Convert HWC -> NCHW for grouped convolution, then restore HWC.
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    blurred = F.pad(nchw, (radius, radius, 0, 0), mode="replicate")
    blurred = F.conv2d(blurred, horizontal_kernel, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="replicate")
    blurred = F.conv2d(blurred, vertical_kernel, groups=channels)
    return blurred.squeeze(0).permute(1, 2, 0).contiguous()


def make_blur_pyramid(
    image: torch.Tensor,
    sigmas: Optional[Sequence[float]] = None,
) -> List[torch.Tensor]:
    """Create low-to-high frequency target images for curriculum training.

    Args:
        image: RGB image tensor with shape [H, W, 3] and values in [0, 1].
        sigmas: Gaussian blur sigmas. If omitted, uses
            ``[4.0, 2.0, 1.0, 0.0]``.

    Returns:
        A list of image tensors ordered from low-frequency to high-frequency.
        With the default sigmas this is
        ``[blur_sigma_4, blur_sigma_2, blur_sigma_1, original]``.

    Raises:
        TypeError: If ``image`` is not a floating point tensor.
        ValueError: If ``image`` shape or any sigma value is invalid.
    """
    _validate_image(image)

    if sigmas is None:
        sigmas = DEFAULT_SIGMAS
    if len(sigmas) == 0:
        raise ValueError("sigmas must contain at least one value.")

    targets: List[torch.Tensor] = []
    for sigma_value in sigmas:
        sigma = float(sigma_value)
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")
        if sigma == 0.0:
            targets.append(image)
        else:
            targets.append(_gaussian_blur_hwc(image, sigma))

    return targets


def _normalize_stage_ratios(
    num_stages: int,
    stage_ratios: Optional[Sequence[float]],
) -> Optional[List[float]]:
    """Validate and normalize optional curriculum stage durations."""
    if stage_ratios is None:
        return None
    if len(stage_ratios) != num_stages:
        raise ValueError(
            "stage_ratios length must match num_stages, "
            f"got {len(stage_ratios)} and {num_stages}."
        )

    ratios = [float(ratio) for ratio in stage_ratios]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("stage_ratios must be non-negative.")

    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("stage_ratios must contain positive total duration.")

    return [ratio / total for ratio in ratios]


def get_curriculum_stage(
    step: int,
    total_steps: int,
    num_stages: int,
    stage_ratios: Optional[Sequence[float]] = None,
) -> int:
    """Return the active curriculum stage for the current training step.

    Stages are equal length by default. For four stages, progress in
    ``[0%, 25%)`` maps to stage 0, ``[25%, 50%)`` maps to stage 1,
    ``[50%, 75%)`` maps to stage 2, and ``[75%, 100%]`` maps to stage 3.
    Passing ``stage_ratios`` makes stages use custom relative durations.
    Steps outside the valid range are clamped to the nearest valid stage.

    Args:
        step: Current training step, interpreted as progress over
            ``total_steps``.
        total_steps: Total number of training steps.
        num_stages: Number of curriculum stages.
        stage_ratios: Optional non-negative relative duration per stage.

    Returns:
        The current stage index in ``[0, num_stages - 1]``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")

    clamped_step = min(max(int(step), 0), total_steps)
    if clamped_step >= total_steps:
        return num_stages - 1

    ratios = _normalize_stage_ratios(num_stages, stage_ratios)
    if ratios is None:
        stage = int(clamped_step * num_stages // total_steps)
        return min(stage, num_stages - 1)

    progress = float(clamped_step) / float(total_steps)
    cumulative = 0.0
    for stage, ratio in enumerate(ratios):
        cumulative += ratio
        if ratio > 0.0 and progress < cumulative:
            return stage

    return num_stages - 1


def get_curriculum_target(
    step: int,
    total_steps: int,
    targets: Sequence[torch.Tensor],
    stage_ratios: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Return the target image for the current curriculum stage.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        targets: Ordered target images, usually from ``make_blur_pyramid``.
        stage_ratios: Optional non-negative relative duration per target.

    Returns:
        The target image corresponding to the current stage.
    """
    if len(targets) == 0:
        raise ValueError("targets must contain at least one image.")

    stage = get_curriculum_stage(
        step=step,
        total_steps=total_steps,
        num_stages=len(targets),
        stage_ratios=stage_ratios,
    )
    return targets[stage]


def _blend_targets(
    current_target: torch.Tensor,
    next_target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Linearly blend two target tensors."""
    if current_target.shape != next_target.shape:
        raise ValueError(
            "curriculum targets must have matching shapes for blending, "
            f"got {tuple(current_target.shape)} and {tuple(next_target.shape)}."
        )
    return torch.lerp(current_target, next_target, float(alpha))


def get_blended_curriculum_target(
    step: int,
    total_steps: int,
    targets: Sequence[torch.Tensor],
    blend_ratio: float = 0.1,
    stage_ratios: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Return a curriculum target with smooth transitions between stages.

    Near each boundary, this function linearly blends the previous and next
    target image. ``blend_ratio`` controls the transition width as a fraction
    of the shorter neighboring stage. For equal-length stages and
    ``blend_ratio=0.1``, each transition spans 10% of one stage length centered
    on the boundary.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        targets: Ordered target images, usually from ``make_blur_pyramid``.
        blend_ratio: Fraction of each stage used for boundary blending. Must be
            in ``[0, 1]``. A value of ``0`` disables blending.
        stage_ratios: Optional non-negative relative duration per target.

    Returns:
        Either the active stage target or a blended target near a boundary.
    """
    if len(targets) == 0:
        raise ValueError("targets must contain at least one image.")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if not 0.0 <= blend_ratio <= 1.0:
        raise ValueError(f"blend_ratio must be in [0, 1], got {blend_ratio}.")

    num_stages = len(targets)
    ratios = _normalize_stage_ratios(num_stages, stage_ratios)
    stage = get_curriculum_stage(
        step=step,
        total_steps=total_steps,
        num_stages=num_stages,
        stage_ratios=ratios,
    )
    if num_stages == 1 or blend_ratio == 0.0:
        return targets[stage]

    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    stage_lengths = ratios
    if stage_lengths is None:
        stage_lengths = [1.0 / float(num_stages)] * num_stages

    cumulative = 0.0
    for next_stage in range(1, num_stages):
        cumulative += stage_lengths[next_stage - 1]
        boundary = cumulative
        transition_width = blend_ratio * min(
            stage_lengths[next_stage - 1],
            stage_lengths[next_stage],
        )
        if transition_width <= 0.0:
            continue

        half_blend_width = 0.5 * transition_width
        blend_start = boundary - half_blend_width
        blend_end = boundary + half_blend_width

        if blend_start <= progress <= blend_end:
            alpha = (progress - blend_start) / (blend_end - blend_start)
            return _blend_targets(targets[next_stage - 1], targets[next_stage], alpha)

    return targets[stage]


__all__ = [
    "make_blur_pyramid",
    "get_curriculum_stage",
    "get_curriculum_target",
    "get_blended_curriculum_target",
]
