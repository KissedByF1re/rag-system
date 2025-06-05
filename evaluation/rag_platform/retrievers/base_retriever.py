"""Base retriever class for all retriever implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.schema import Document


class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    def __init__(self, **kwargs):
        """Initialize retriever with configuration."""
        self.config = kwargs
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for the given query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get retriever configuration."""
        return self.config.copy()
    
    def update_config(self, **kwargs) -> None:
        """Update retriever configuration."""
        self.config.update(kwargs)