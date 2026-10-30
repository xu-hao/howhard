"""LLM model and effort selection, with a zero-configuration starting point."""

__version__ = "0.6.0"

from .config import load_config
from .engine import Engine

__all__ = ["Engine", "load_config", "__version__"]
