import math
from typing import Union

import numpy as np
import torch
from skimage.metrics import structural_similarity


ArrayLike = Union[np.ndarray, torch.Tensor]


def mse_to_psnr(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    return mse_to_psnr(mse, data_range=data_range)


def _to_numpy_image(image: ArrayLike) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()
    return np.asarray(image, dtype=np.float32)


def compute_ssim(pred: ArrayLike, target: ArrayLike, data_range: float = 1.0) -> float:
    pred_np = np.clip(_to_numpy_image(pred), 0.0, data_range)
    target_np = np.clip(_to_numpy_image(target), 0.0, data_range)

    if pred_np.shape != target_np.shape:
        raise ValueError(f"SSIM shape mismatch: {pred_np.shape} vs {target_np.shape}")
    if pred_np.ndim != 3 or pred_np.shape[-1] != 3:
        raise ValueError("SSIM expects RGB images with shape [H, W, 3].")

    min_side = min(pred_np.shape[0], pred_np.shape[1])
    if min_side < 3:
        return float("nan")

    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    return float(
        structural_similarity(
            target_np,
            pred_np,
            data_range=data_range,
            channel_axis=-1,
            win_size=win_size,
        )
    )
