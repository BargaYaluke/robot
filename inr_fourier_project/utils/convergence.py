"""Convergence analysis utilities for INR image fitting.

These helpers summarize PSNR curves in terms of early training progress and
iterations needed to reach target reconstruction quality. They are designed for
comparing baseline training against curriculum or edge-aware strategies.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple


PSNRRecord = Tuple[int, float]
DEFAULT_TARGET_PSNR = [30, 35, 38, 40]


def find_iteration_to_psnr(
    psnr_records: Sequence[PSNRRecord],
    target_psnr: float,
) -> Optional[int]:
    """Find the first iteration where PSNR reaches a target value.

    Args:
        psnr_records: Ordered sequence of ``(step, psnr)`` records.
        target_psnr: Target PSNR value in dB.

    Returns:
        The first step whose PSNR is greater than or equal to ``target_psnr``.
        Returns None if the target is never reached.
    """
    for step, psnr in psnr_records:
        if float(psnr) >= float(target_psnr):
            return int(step)
    return None


def _target_key(target_psnr: float) -> str:
    """Create a stable dictionary key for a target PSNR value."""
    target = float(target_psnr)
    if target.is_integer():
        target_text = str(int(target))
    else:
        target_text = str(target).replace(".", "_")
    return f"iter_to_{target_text}db"


def compute_iterations_to_targets(
    psnr_records: Sequence[PSNRRecord],
    targets: Optional[Sequence[float]] = None,
) -> Dict[str, Optional[int]]:
    """Compute iterations needed to reach several PSNR targets.

    Args:
        psnr_records: Ordered sequence of ``(step, psnr)`` records.
        targets: Target PSNR values in dB. If omitted, uses
            ``[30, 35, 38, 40]``.

    Returns:
        Dictionary mapping target names such as ``iter_to_30db`` to the first
        reached step. Values are None for targets that are not reached.
    """
    if targets is None:
        targets = DEFAULT_TARGET_PSNR

    return {
        _target_key(target): find_iteration_to_psnr(psnr_records, target)
        for target in targets
    }


def summarize_convergence(psnr_records: Sequence[PSNRRecord]) -> Dict[str, object]:
    """Summarize convergence behavior from a PSNR curve.

    Args:
        psnr_records: Ordered sequence of ``(step, psnr)`` records.

    Returns:
        Dictionary containing:
            ``best_psnr``: Highest recorded PSNR, or None if no records exist.
            ``final_psnr``: Last recorded PSNR, or None if no records exist.
            ``best_step``: Step where the highest PSNR occurs, or None.
            ``iter_to_*db``: Iterations needed to reach default PSNR targets.
    """
    if not psnr_records:
        summary: Dict[str, object] = {
            "best_psnr": None,
            "final_psnr": None,
            "best_step": None,
        }
        summary.update(compute_iterations_to_targets(psnr_records))
        return summary

    # Keep the first occurrence when multiple records share the same best PSNR.
    best_step, best_psnr = max(psnr_records, key=lambda record: float(record[1]))
    _, final_psnr = psnr_records[-1]

    summary = {
        "best_psnr": float(best_psnr),
        "final_psnr": float(final_psnr),
        "best_step": int(best_step),
    }
    summary.update(compute_iterations_to_targets(psnr_records))
    return summary


__all__ = [
    "find_iteration_to_psnr",
    "compute_iterations_to_targets",
    "summarize_convergence",
]
