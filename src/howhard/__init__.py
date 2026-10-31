"""LLM model and effort selection, with a zero-configuration starting point."""

__version__ = "1.0.0"

from .config import load_config
from .engine import Engine
from .retrieval import RetrievalPipeline

__all__ = ["Engine", "RetrievalPipeline", "load_config", "__version__"]
