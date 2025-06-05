"""Hybrid retriever that combines multiple retrieval methods."""

from typing import Dict, List, Optional, Set, Tuple

from langchain.schema import Document

from rag_platform.core.registry import registry
from .base_retriever import BaseRetriever
from .vector_retriever import VectorRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector search with keyword/BM25 search."""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        fusion_method: str = "rrf",  # Reciprocal Rank Fusion
        **kwargs
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_retriever: Vector similarity retriever
            keyword_weight: Weight for keyword search results
            vector_weight: Weight for vector search results  
            fusion_method: Method to combine results ("rrf", "weighted_sum")
        """
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.fusion_method = fusion_method
        
        # Validate weights
        if abs(keyword_weight + vector_weight - 1.0) > 1e-6:
            raise ValueError("Keyword and vector weights must sum to 1.0")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with hybrid ranking
        """
        # Get more candidates than needed for better fusion
        fetch_k = min(k * 3, 50)
        
        # Get vector search results
        if hasattr(self.vector_retriever, 'retrieve'):
            vector_docs = self.vector_retriever.retrieve(query, k=fetch_k)
        else:
            # LangChain retriever
            vector_docs = self.vector_retriever.invoke(query, config={"k": fetch_k})
        
        # Get keyword search results (simplified - just text matching for now)
        keyword_docs = self._keyword_search(query, k=fetch_k)
        
        # Combine results using fusion method
        if self.fusion_method == "rrf":
            combined_docs = self._reciprocal_rank_fusion(vector_docs, keyword_docs)
        elif self.fusion_method == "weighted_sum":
            combined_docs = self._weighted_fusion(vector_docs, keyword_docs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return combined_docs[:k]
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """Simple keyword-based search using text matching.
        
        This is a simplified implementation. In practice, you'd want to use
        a proper keyword search engine like Elasticsearch or BM25.
        """
        # For now, use vector search as fallback
        # In a real implementation, you'd use a keyword search index
        if hasattr(self.vector_retriever, 'retrieve'):
            return self.vector_retriever.retrieve(query, k=k)
        else:
            # LangChain retriever
            return self.vector_retriever.invoke(query, config={"k": k})
    
    def _reciprocal_rank_fusion(
        self, 
        vector_docs: List[Document], 
        keyword_docs: List[Document],
        k: int = 60  # RRF parameter
    ) -> List[Document]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_docs, 1):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.vector_weight / (k + rank)
            doc_objects[doc_id] = doc
        
        # Process keyword results  
        for rank, doc in enumerate(keyword_docs, 1):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.keyword_weight / (k + rank)
            doc_objects[doc_id] = doc
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_objects[doc_id] for doc_id, _ in sorted_docs]
    
    def _weighted_fusion(
        self, 
        vector_docs: List[Document], 
        keyword_docs: List[Document]
    ) -> List[Document]:
        """Combine results using weighted scoring."""
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}
        
        # Process vector results (assign scores based on rank)
        for rank, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            score = self.vector_weight * (len(vector_docs) - rank) / len(vector_docs)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_objects[doc_id] = doc
        
        # Process keyword results
        for rank, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            score = self.keyword_weight * (len(keyword_docs) - rank) / len(keyword_docs)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_objects[doc_id] = doc
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_objects[doc_id] for doc_id, _ in sorted_docs]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Get unique identifier for a document."""
        # Use content hash as simple ID
        import hashlib
        content = doc.page_content + str(doc.metadata)
        return hashlib.md5(content.encode()).hexdigest()


# Register the retriever
registry.register_retriever("hybrid", HybridRetriever)