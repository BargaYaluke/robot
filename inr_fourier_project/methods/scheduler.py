"""Shared progress schedules for coupled band-loss and edge-aware sampling.

A coupled experiment moves band-loss weights and edge-sampling bias along the
same time axis. This module gives both modules a single source of truth for
"how far along are we", so they always agree on the current progress value.

Two helpers are exposed:

- :func:`compute_progress` maps an integer step to a clamped progress in
  ``[0, 1]`` over an optional sub-window of training.
- :func:`compute_effective_edge_ratio` ramps the edge-sampling fraction from
  ``0`` up to a target ratio using the same progress value.
"""

from __future__ import annotations

import math


SUPPORTED_MODES = ("linear", "cosine")


def _validate_mode(mode: str) -> None:
    """Reject unsupported interpolation modes early."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported schedule mode: {mode}. Expected one of {SUPPORTED_MODES}."
        )


def _validate_ratio_window(start_ratio: float, end_ratio: float) -> None:
    """Ensure a progress window is a valid sub-range of training."""
    if not 0.0 <= start_ratio <= 1.0:
        raise ValueError(f"start_ratio must be in [0, 1], got {start_ratio}.")
    if not 0.0 <= end_ratio <= 1.0:
        raise ValueError(f"end_ratio must be in [0, 1], got {end_ratio}.")
    if end_ratio < start_ratio:
        raise ValueError(
            f"end_ratio must be >= start_ratio, got end={end_ratio} start={start_ratio}."
        )


def compute_progress(
    step: int,
    total_steps: int,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    mode: str = "linear",
) -> float:
    """Return the active progress value for a training step.

    The raw progress fraction is ``(step / total_steps - start_ratio) /
    (end_ratio - start_ratio)`` clamped to ``[0, 1]``. With ``mode="cosine"``,
    the clamped progress passes through a half-cosine easing curve that starts
    and ends with zero slope.

    Args:
        step: Current training step. Negative values are clamped to 0.
        total_steps: Total number of training steps. Must be positive.
        start_ratio: Fraction of training before which progress stays at 0.
        end_ratio: Fraction of training after which progress stays at 1.
        mode: ``"linear"`` or ``"cosine"`` easing of the clamped fraction.

    Returns:
        Progress value in ``[0, 1]``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    _validate_ratio_window(start_ratio, end_ratio)
    _validate_mode(mode)

    step_fraction = max(float(step), 0.0) / float(total_steps)
    window = float(end_ratio) - float(start_ratio)
    if window <= 0.0:
        return 1.0 if step_fraction >= float(end_ratio) else 0.0

    raw = (step_fraction - float(start_ratio)) / window
    clamped = min(max(raw, 0.0), 1.0)
    if mode == "linear":
        return clamped
    return 0.5 - 0.5 * math.cos(math.pi * clamped)


def compute_effective_edge_ratio(
    progress: float,
    target_edge_ratio: float,
) -> float:
    """Scale a target edge-sampling fraction by a progress value.

    Used together with :func:`compute_progress` to produce a smoothly ramped
    edge-sampling fraction. The result starts at ``0`` when ``progress == 0``
    and reaches ``target_edge_ratio`` when ``progress == 1``.
    """
    if not 0.0 <= target_edge_ratio <= 1.0:
        raise ValueError(
            f"target_edge_ratio must be in [0, 1], got {target_edge_ratio}."
        )
    clamped_progress = min(max(float(progress), 0.0), 1.0)
    return float(target_edge_ratio) * clamped_progress


__all__ = [
    "SUPPORTED_MODES",
    "compute_progress",
    "compute_effective_edge_ratio",
]
