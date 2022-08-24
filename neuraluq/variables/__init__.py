"""Package for variables module."""

__all__ = [
    "Variable",
    "_Trainable",
    "_Samplable",
    "_Variational",
    "fnn",
    "const",
]

from .variables import Variable, _Trainable, _Samplable, _Variational

from . import fnn
from . import const
