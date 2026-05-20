"""Fourier Feature Neural Image Representation prototype."""

from .config import TrainConfig
from .models import CoordinateMLP, FourierFeatureMLP

__all__ = ["TrainConfig", "CoordinateMLP", "FourierFeatureMLP"]
