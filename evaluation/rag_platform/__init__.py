"""Modular RAG Platform - A no-code RAG deployment and evaluation system."""

__version__ = "0.1.0"

# Import all modules to register components
from . import loaders
from . import chunkers
from . import embeddings
from . import vectorstores
from . import chains
from . import evaluation