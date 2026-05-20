"""Image loading and coordinate conversion utilities for INR fitting."""

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load an RGB image as a float tensor with shape [H, W, 3].

    Pixel values are normalized to [0, 1]. The image is resized to
    (size, size), which keeps the baseline single-image fitting pipeline
    simple and batch-friendly.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        resample = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize((size, size), resample=resample)

    # TF.to_tensor returns [C, H, W] in [0, 1]; INR fitting is more convenient
    # with image tensors in [H, W, C] so coordinates and colors share layout.
    tensor = TF.to_tensor(image).permute(1, 2, 0).contiguous()

    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise RuntimeError(f"Expected image tensor shape [H, W, 3], got {tuple(tensor.shape)}.")

    return tensor.float()


def make_coord_grid(h: int, w: int) -> torch.Tensor:
    """Create a flattened 2D coordinate grid in [-1, 1].

    Returns:
        Tensor of shape [H * W, 2], where each row is (x, y).
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got h={h}, w={w}.")

    y = torch.linspace(-1.0, 1.0, steps=h)
    x = torch.linspace(-1.0, 1.0, steps=w)

    try:
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(y, x)

    # Stack as (x, y), then flatten in row-major order to match image.reshape.
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(h * w, 2)

    if coords.shape != (h * w, 2):
        raise RuntimeError(f"Expected coords shape [{h * w}, 2], got {tuple(coords.shape)}.")

    return coords.float()


def image_to_coord_rgb(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert an image tensor into coordinate and RGB training pairs.

    Args:
        image: Tensor with shape [H, W, 3] and values in [0, 1].

    Returns:
        coords: Tensor with shape [H * W, 2].
        rgb: Tensor with shape [H * W, 3].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}.")
    if image.ndim != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {tuple(image.shape)}.")
    if image.shape[-1] != 3:
        raise ValueError(f"image must have 3 RGB channels, got shape {tuple(image.shape)}.")

    h, w, _ = image.shape
    coords = make_coord_grid(h, w)
    rgb = image.reshape(h * w, 3).float().contiguous()

    if rgb.shape != (h * w, 3):
        raise RuntimeError(f"Expected rgb shape [{h * w}, 3], got {tuple(rgb.shape)}.")

    return coords, rgb
