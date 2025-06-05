"""Embedding models."""

# Import modules to trigger registration
from . import openai_embeddings

from .openai_embeddings import OpenAIEmbeddings

__all__ = ["OpenAIEmbeddings"]