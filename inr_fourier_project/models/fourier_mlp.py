"""Fourier Feature MLP for implicit neural image representation."""

import math

import torch
from torch import nn


class FourierFeatureMapping(nn.Module):
    """Random Gaussian Fourier feature mapping for coordinate inputs.

    This module maps low-dimensional pixel coordinates into a higher-dimensional
    sinusoidal feature space. It is used as the positional encoding layer for
    the base INR model in image fitting experiments.

    Args:
        in_dim: Coordinate input dimension. Defaults to 2 for (x, y).
        mapping_size: Number of random Fourier frequencies.
        scale: Standard deviation multiplier for the Gaussian matrix.
    """

    def __init__(
        self,
        in_dim: int = 2,
        mapping_size: int = 256,
        scale: float = 10.0,
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        if mapping_size <= 0:
            raise ValueError(f"mapping_size must be positive, got {mapping_size}.")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}.")

        self.in_dim = in_dim
        self.mapping_size = mapping_size
        self.out_dim = 2 * mapping_size

        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with sinusoidal Fourier features.

        Args:
            x: Tensor of shape [N, in_dim], usually normalized to [-1, 1].

        Returns:
            Tensor of shape [N, 2 * mapping_size].
        """
        if x.ndim != 2:
            raise ValueError(f"x must have shape [N, C], got {tuple(x.shape)}.")
        if x.shape[-1] != self.B.shape[0]:
            raise ValueError(
                f"Expected input dimension {self.B.shape[0]}, got {x.shape[-1]}."
            )

        x_proj = 2.0 * math.pi * x @ self.B
        encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if encoded.shape[-1] != self.out_dim:
            raise RuntimeError(
                f"Expected encoded dimension {self.out_dim}, got {encoded.shape[-1]}."
            )
        return encoded


class FourierMLP(nn.Module):
    """Fourier Feature INR model for single-image fitting.

    The model first encodes 2D coordinates using random Gaussian Fourier
    features, then predicts RGB values with a small ReLU MLP. This is the base
    Fourier Feature Network used for image fitting experiments.

    Args:
        in_dim: Coordinate input dimension. Defaults to 2 for (x, y).
        out_dim: Output dimension. Defaults to 3 for RGB.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden Linear + ReLU layers.
        mapping_size: Number of random Fourier frequencies.
        scale: Standard deviation multiplier for the Gaussian matrix.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        mapping_size: int = 256,
        scale: float = 10.0,
    ) -> None:
        super().__init__()

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.mapping = FourierFeatureMapping(
            in_dim=in_dim,
            mapping_size=mapping_size,
            scale=scale,
        )

        layers = []
        current_dim = self.mapping.out_dim

        # Hidden layers operate on Fourier-encoded coordinates.
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim

        # RGB output is normalized to [0, 1] for direct image reconstruction.
        layers.append(nn.Linear(current_dim, out_dim))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict RGB values from 2D coordinates.

        Args:
            coords: Tensor of shape [N, in_dim].

        Returns:
            Tensor of shape [N, out_dim] with RGB values in [0, 1].
        """
        encoded = self.mapping(coords)
        return self.mlp(encoded)
