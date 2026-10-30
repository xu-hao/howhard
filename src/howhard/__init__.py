"""Bayesian model and effort selection, with a zero-configuration starting point."""

__version__ = "0.6.1"

from .config import load_config
from .engine import Engine
from .outcomes import fit

__all__ = ["Engine", "fit", "load_config", "__version__"]
