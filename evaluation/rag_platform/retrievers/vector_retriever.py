"""Vector similarity-based retriever."""

from typing import List, Optional, Union

from langchain.schema import Document

from rag_platform.core.registry import registry
from rag_platform.core.base import BaseVectorStore
from .base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Vector similarity-based document retriever."""
    
    def __init__(
        self,
        vectorstore: Union[BaseVectorStore, str],
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
        **kwargs
    ):
        """Initialize vector retriever.
        
        Args:
            vectorstore: Vector store instance or name
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters
        """
        super().__init__(**kwargs)
        
        if isinstance(vectorstore, str):
            vectorstore_class = registry.get_vectorstore(vectorstore)
            self.vectorstore = vectorstore_class()
        else:
            self.vectorstore = vectorstore
            
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using vector similarity search.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        search_kwargs = self.search_kwargs.copy()
        search_kwargs["k"] = k
        
        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, k=k)
        elif self.search_type == "mmr":
            return self._mmr_search(query, **search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            return self._similarity_score_threshold_search(query, **search_kwargs)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
    
    def _mmr_search(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Document]:
        """Maximal Marginal Relevance search for diverse results."""
        # Basic MMR implementation - would need more sophisticated implementation
        # For now, fall back to similarity search
        return self.vectorstore.similarity_search(query, k=k)
    
    def _similarity_score_threshold_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.5
    ) -> List[Document]:
        """Search with similarity score threshold."""
        # Basic implementation - would need score-aware vectorstore
        # For now, fall back to similarity search
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_vectorstore(self) -> BaseVectorStore:
        """Get the underlying vector store."""
        return self.vectorstore


# Register the retriever
registry.register_retriever("vector", VectorRetriever)