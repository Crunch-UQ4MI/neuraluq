"""Package for surrogates module."""

__all__ = [
    "Surrogate",
    "FNN",
    "Identity",
    "Generator",
    "DeepONet",
    "DeepONet_pretrained",
]

from .surrogate import Surrogate, Identity
from .fnn import FNN
from .generator import Generator
from .deeponet import DeepONet, DeepONet_pretrained
