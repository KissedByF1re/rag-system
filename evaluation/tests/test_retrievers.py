"""Tests for retriever implementations."""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document

from rag_platform.retrievers import (
    BaseRetriever,
    VectorRetriever, 
    HybridRetriever,
    RerankerRetriever
)
from rag_platform.core.registry import registry


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self.docs = [
            Document(page_content="Python is a programming language", metadata={"id": 1}),
            Document(page_content="Machine learning with Python", metadata={"id": 2}),
            Document(page_content="Data science and analytics", metadata={"id": 3}),
            Document(page_content="Natural language processing", metadata={"id": 4}),
        ]
    
    def similarity_search(self, query: str, k: int = 4) -> list:
        """Mock similarity search."""
        # Simple mock: return first k documents
        return self.docs[:k]


class TestBaseRetriever:
    """Test base retriever functionality."""
    
    def test_base_retriever_is_abstract(self):
        """Test that BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever()
    
    def test_base_retriever_config(self):
        """Test base retriever configuration methods."""
        class ConcreteRetriever(BaseRetriever):
            def retrieve(self, query: str, k: int = 5):
                return []
        
        retriever = ConcreteRetriever(test_param="value")
        
        # Test config methods
        config = retriever.get_config()
        assert config["test_param"] == "value"
        
        retriever.update_config(new_param="new_value")
        config = retriever.get_config()
        assert config["new_param"] == "new_value"


class TestVectorRetriever:
    """Test vector retriever implementation."""
    
    def test_vector_retriever_init_with_instance(self):
        """Test initialization with vector store instance."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(vectorstore=mock_store)
        
        assert retriever.vectorstore == mock_store
        assert retriever.search_type == "similarity"
        assert retriever.search_kwargs == {}
    
    def test_vector_retriever_init_with_config(self):
        """Test initialization with configuration."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(
            vectorstore=mock_store,
            search_type="mmr",
            search_kwargs={"lambda_mult": 0.7}
        )
        
        assert retriever.search_type == "mmr"
        assert retriever.search_kwargs["lambda_mult"] == 0.7
    
    def test_vector_retriever_similarity_search(self):
        """Test basic similarity search."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(vectorstore=mock_store)
        
        results = retriever.retrieve("Python programming", k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert "Python" in results[0].page_content
    
    def test_vector_retriever_mmr_search(self):
        """Test MMR search (falls back to similarity for now)."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(
            vectorstore=mock_store,
            search_type="mmr"
        )
        
        results = retriever.retrieve("Python programming", k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
    
    def test_vector_retriever_invalid_search_type(self):
        """Test invalid search type raises error."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(
            vectorstore=mock_store,
            search_type="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown search type"):
            retriever.retrieve("test query")
    
    def test_vector_retriever_get_vectorstore(self):
        """Test getting underlying vector store."""
        mock_store = MockVectorStore()
        retriever = VectorRetriever(vectorstore=mock_store)
        
        assert retriever.get_vectorstore() == mock_store


class TestHybridRetriever:
    """Test hybrid retriever implementation."""
    
    def test_hybrid_retriever_init(self):
        """Test hybrid retriever initialization."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_weight=0.6,
            vector_weight=0.4
        )
        
        assert hybrid.vector_retriever == vector_retriever
        assert hybrid.keyword_weight == 0.6
        assert hybrid.vector_weight == 0.4
        assert hybrid.fusion_method == "rrf"
    
    def test_hybrid_retriever_invalid_weights(self):
        """Test that invalid weights raise error."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HybridRetriever(
                vector_retriever=vector_retriever,
                keyword_weight=0.6,
                vector_weight=0.6  # Sum > 1.0
            )
    
    def test_hybrid_retriever_retrieve(self):
        """Test hybrid retrieval."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        hybrid = HybridRetriever(vector_retriever=vector_retriever)
        
        results = hybrid.retrieve("Python programming", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_hybrid_retriever_rrf_fusion(self):
        """Test RRF fusion method."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            fusion_method="rrf"
        )
        
        # Test internal RRF method
        docs1 = mock_store.docs[:2]
        docs2 = mock_store.docs[1:3]
        
        result = hybrid._reciprocal_rank_fusion(docs1, docs2)
        assert len(result) >= 2
        assert all(isinstance(doc, Document) for doc in result)
    
    def test_hybrid_retriever_weighted_fusion(self):
        """Test weighted fusion method."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            fusion_method="weighted_sum"
        )
        
        results = hybrid.retrieve("Python programming", k=2)
        assert len(results) <= 2
    
    def test_hybrid_retriever_invalid_fusion_method(self):
        """Test invalid fusion method raises error."""
        mock_store = MockVectorStore()
        vector_retriever = VectorRetriever(vectorstore=mock_store)
        
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            fusion_method="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown fusion method"):
            hybrid.retrieve("test query")


class TestRerankerRetriever:
    """Test reranker retriever implementation."""
    
    def test_reranker_retriever_init(self):
        """Test reranker retriever initialization."""
        mock_store = MockVectorStore()
        base_retriever = VectorRetriever(vectorstore=mock_store)
        
        reranker = RerankerRetriever(
            base_retriever=base_retriever,
            reranker_type="cross_encoder",
            reranker_model="test-model"
        )
        
        assert reranker.base_retriever == base_retriever
        assert reranker.reranker_type == "cross_encoder"
        assert reranker.reranker_model == "test-model"
        assert reranker._reranker is None  # Lazy loading
    
    def test_reranker_retriever_get_base_retriever(self):
        """Test getting base retriever."""
        mock_store = MockVectorStore()
        base_retriever = VectorRetriever(vectorstore=mock_store)
        reranker = RerankerRetriever(base_retriever=base_retriever)
        
        assert reranker.get_base_retriever() == base_retriever
    
    @patch('sentence_transformers.CrossEncoder')
    def test_reranker_retriever_cross_encoder_loading(self, mock_cross_encoder):
        """Test cross encoder loading."""
        mock_store = MockVectorStore()
        base_retriever = VectorRetriever(vectorstore=mock_store)
        
        reranker = RerankerRetriever(
            base_retriever=base_retriever,
            reranker_type="cross_encoder"
        )
        
        # Mock the cross encoder
        mock_model = Mock()
        mock_model.predict.return_value = [0.8, 0.6, 0.4, 0.2]
        mock_cross_encoder.return_value = mock_model
        
        results = reranker.retrieve("Python programming", k=2)
        
        # Should call CrossEncoder
        mock_cross_encoder.assert_called_once()
        assert len(results) <= 2
    
    def test_reranker_retriever_invalid_type(self):
        """Test invalid reranker type."""
        mock_store = MockVectorStore()
        base_retriever = VectorRetriever(vectorstore=mock_store)
        
        reranker = RerankerRetriever(
            base_retriever=base_retriever,
            reranker_type="invalid_type"
        )
        
        # Invalid type should raise error when there are documents to rerank
        with pytest.raises(ValueError, match="Unknown reranker type"):
            reranker.retrieve("test query", k=2)  # Will have documents to rerank
    
    def test_reranker_retriever_no_candidates(self):
        """Test behavior when base retriever returns no results."""
        mock_store = Mock()
        mock_store.similarity_search.return_value = []
        
        base_retriever = VectorRetriever(vectorstore=mock_store)
        reranker = RerankerRetriever(base_retriever=base_retriever)
        
        results = reranker.retrieve("test query")
        assert results == []


class TestRetrieverRegistry:
    """Test retriever registration."""
    
    def test_retrievers_are_registered(self):
        """Test that all retrievers are properly registered."""
        # Check if retrievers are registered
        components = registry.list_components()
        
        assert "vector" in components["retrievers"]
        assert "hybrid" in components["retrievers"] 
        assert "reranker" in components["retrievers"]
    
    def test_get_registered_retrievers(self):
        """Test getting registered retrievers from registry."""
        vector_class = registry.get_retriever("vector")
        hybrid_class = registry.get_retriever("hybrid")
        reranker_class = registry.get_retriever("reranker")
        
        assert vector_class == VectorRetriever
        assert hybrid_class == HybridRetriever
        assert reranker_class == RerankerRetriever
    
    def test_get_unknown_retriever(self):
        """Test that unknown retriever raises error."""
        with pytest.raises(ValueError, match="Unknown retriever"):
            registry.get_retriever("nonexistent_retriever")