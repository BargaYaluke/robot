"""Vanilla coordinate MLP for implicit neural image representation."""

import torch
from torch import nn


class VanillaMLP(nn.Module):
    """A basic coordinate-based MLP for image fitting.

    The model maps 2D pixel coordinates directly to RGB values. It intentionally
    uses a simple architecture so it can serve as the weakest baseline against
    Fourier Feature Networks.

    Args:
        in_dim: Coordinate input dimension. Defaults to 2 for (x, y).
        out_dim: Output dimension. Defaults to 3 for RGB.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden Linear + ReLU layers.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        current_dim = in_dim

        # Hidden layers: plain Linear + ReLU blocks.
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim

        # Final RGB prediction layer followed by sigmoid normalization.
        layers.append(nn.Linear(current_dim, out_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict RGB values from 2D coordinates.

        Args:
            coords: Tensor of shape [N, in_dim], usually normalized to [-1, 1].

        Returns:
            Tensor of shape [N, out_dim] with values in [0, 1].
        """
        if coords.ndim != 2:
            raise ValueError(f"coords must have shape [N, C], got {tuple(coords.shape)}.")
        if coords.shape[-1] != self.in_dim:
            raise ValueError(f"Expected coordinate dimension {self.in_dim}, got {coords.shape[-1]}.")

        return self.net(coords)
