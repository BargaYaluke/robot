import math

import torch
from torch import nn


class FourierFeatureMapping(nn.Module):
    """Random Gaussian Fourier feature mapping for low-dimensional coordinates."""

    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 256,
        scale: float = 10.0,
        include_input: bool = False,
    ) -> None:
        super().__init__()
        if mapping_size <= 0:
            raise ValueError("mapping_size must be positive.")

        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.include_input = include_input

        basis = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("basis", basis)

        self.output_dim = 2 * mapping_size
        if include_input:
            self.output_dim += input_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = (2.0 * math.pi) * coords @ self.basis
        encoded = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded
