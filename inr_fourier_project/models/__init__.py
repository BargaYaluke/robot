"""Model definitions for INR image fitting."""

from .fourier_mlp import FourierFeatureMapping, FourierMLP
from .vanilla_mlp import VanillaMLP

__all__ = ["FourierFeatureMapping", "FourierMLP", "VanillaMLP"]
