"""LLM model and effort selection, with a zero-configuration starting point."""

__version__ = "3.0.0"

from .config import load_config
from .engine import Engine
from .retrieval import RetrievalPipeline
from .writes import WriteError, WriteTimeout, WriteCancelled, WriteDependencyError

__all__ = ["Engine", "RetrievalPipeline", "load_config", "__version__",
           "WriteError", "WriteTimeout", "WriteCancelled", "WriteDependencyError"]
