"""OpenAI embedding models."""

from typing import List

from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

from rag_platform.core.base import BaseEmbedding
from rag_platform.core.config import Settings
from rag_platform.core.registry import registry


class OpenAIEmbeddings(BaseEmbedding):
    """OpenAI embedding wrapper."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        settings = Settings()
        self.embeddings = LangchainOpenAIEmbeddings(
            model=model,
            openai_api_key=settings.openai_api_key,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(text)


registry.register_embedding("openai", OpenAIEmbeddings)