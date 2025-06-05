"""Retriever with reranking capability."""

from typing import List, Optional, Union

from langchain.schema import Document

from rag_platform.core.registry import registry
from .base_retriever import BaseRetriever


class RerankerRetriever(BaseRetriever):
    """Retriever that adds reranking to improve result relevance."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker_type: str = "cross_encoder",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_rerank: Optional[int] = None,
        **kwargs
    ):
        """Initialize reranker retriever.
        
        Args:
            base_retriever: Base retriever to get initial candidates
            reranker_type: Type of reranker ("cross_encoder", "sentence_transformer")
            reranker_model: Model name for reranking
            top_k_rerank: Number of top candidates to rerank (None = rerank all)
        """
        super().__init__(**kwargs)
        self.base_retriever = base_retriever
        self.reranker_type = reranker_type
        self.reranker_model = reranker_model
        self.top_k_rerank = top_k_rerank
        self._reranker = None
    
    def _load_reranker(self):
        """Lazy load the reranker model."""
        if self._reranker is None:
            if self.reranker_type == "cross_encoder":
                self._reranker = self._load_cross_encoder()
            elif self.reranker_type == "sentence_transformer":
                self._reranker = self._load_sentence_transformer()
            else:
                raise ValueError(f"Unknown reranker type: {self.reranker_type}")
    
    def _load_cross_encoder(self):
        """Load cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(self.reranker_model)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
    
    def _load_sentence_transformer(self):
        """Load sentence transformer for reranking."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.reranker_model)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for sentence transformer reranking. "
                "Install with: pip install sentence-transformers"
            )
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve and rerank documents.
        
        Args:
            query: Search query
            k: Number of final documents to return
            
        Returns:
            List of reranked relevant documents
        """
        # Get more candidates for reranking
        fetch_k = self.top_k_rerank or min(k * 4, 100)
        if hasattr(self.base_retriever, 'retrieve'):
            candidates = self.base_retriever.retrieve(query, k=fetch_k)
        else:
            # LangChain retriever
            candidates = self.base_retriever.invoke(query, config={"k": fetch_k})
        
        if not candidates:
            return []
        
        # If we don't need reranking or have fewer docs than requested
        if len(candidates) <= k and self.top_k_rerank is None:
            return candidates[:k]
        
        # Load reranker and rerank documents
        self._load_reranker()
        reranked_docs = self._rerank_documents(query, candidates)
        
        return reranked_docs[:k]
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if self.reranker_type == "cross_encoder":
            return self._rerank_with_cross_encoder(query, documents)
        elif self.reranker_type == "sentence_transformer":
            return self._rerank_with_sentence_transformer(query, documents)
        else:
            raise ValueError(f"Unknown reranker type: {self.reranker_type}")
    
    def _rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using cross-encoder model."""
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self._reranker.predict(pairs)
        
        # Sort documents by relevance score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores]
    
    def _rerank_with_sentence_transformer(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using sentence transformer similarity."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get embeddings
        query_embedding = self._reranker.encode([query])
        doc_embeddings = self._reranker.encode([doc.page_content for doc in documents])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Sort documents by similarity
        doc_scores = list(zip(documents, similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_scores]
    
    def get_base_retriever(self) -> BaseRetriever:
        """Get the base retriever."""
        return self.base_retriever


# Register the retriever
registry.register_retriever("reranker", RerankerRetriever)