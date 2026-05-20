from typing import Optional

import torch
from torch import nn

from .encoding import FourierFeatureMapping


def get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def get_output_activation(name: str) -> Optional[nn.Module]:
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "none":
        return None
    raise ValueError(f"Unsupported output activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(get_activation(activation))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        final_activation = get_output_activation(output_activation)
        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordinateMLP(nn.Module):
    """Vanilla coordinate MLP that maps 2D coordinates directly to RGB."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=2,
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class FourierFeatureMLP(nn.Module):
    """Coordinate MLP with random Gaussian Fourier feature encoding."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        mapping_size: int = 256,
        fourier_scale: float = 10.0,
        include_input: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = FourierFeatureMapping(
            input_dim=2,
            mapping_size=mapping_size,
            scale=fourier_scale,
            include_input=include_input,
        )
        self.mlp = MLP(
            input_dim=self.encoder.output_dim,
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(coords))


def build_model(config) -> nn.Module:
    if config.model == "vanilla":
        return CoordinateMLP(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            output_activation=config.output_activation,
        )

    if config.model == "fourier":
        return FourierFeatureMLP(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            output_activation=config.output_activation,
            mapping_size=config.mapping_size,
            fourier_scale=config.fourier_scale,
            include_input=config.include_input,
        )

    raise ValueError(f"Unsupported model type: {config.model}")
