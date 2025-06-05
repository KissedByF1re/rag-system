"""Base classes for RAG components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.schema import Document


class BaseLoader(ABC):
    """Base class for document loaders."""
    
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Load documents from the given path."""
        pass


class BaseChunker(ABC):
    """Base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        pass


class BaseEmbedding(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass


class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass



class BaseRAGChain(ABC):
    """Base class for RAG chains."""
    
    @abstractmethod
    def run(self, query: str, context: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run the RAG chain."""
        pass


class BaseEvaluator(ABC):
    """Base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate RAG results."""
        pass