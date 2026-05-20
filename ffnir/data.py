from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def load_image_tensor(
    image_path: str,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> torch.Tensor:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    if image_height is not None or image_width is not None:
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width must be set together.")
        image = TF.resize(
            image,
            [image_height, image_width],
            interpolation=InterpolationMode.BICUBIC,
        )

    tensor = TF.to_tensor(image).permute(1, 2, 0).contiguous()
    return tensor.float()


def make_coordinate_grid(
    height: int,
    width: int,
    coordinate_range: str = "minus_one_one",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if coordinate_range == "minus_one_one":
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
    elif coordinate_range == "zero_one":
        ys = torch.linspace(0.0, 1.0, height, device=device)
        xs = torch.linspace(0.0, 1.0, width, device=device)
    else:
        raise ValueError(f"Unsupported coordinate_range: {coordinate_range}")

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    return coords.reshape(-1, 2).float()


class ImageFittingDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        coordinate_range: str = "minus_one_one",
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.image = load_image_tensor(image_path, image_height, image_width)
        self.height, self.width = self.image.shape[:2]
        self.coords = make_coordinate_grid(self.height, self.width, coordinate_range)
        self.rgb = self.image.reshape(-1, 3).float()

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[index], self.rgb[index]

    def full_image(self) -> torch.Tensor:
        return self.image

    def full_coords(self) -> torch.Tensor:
        return self.coords
