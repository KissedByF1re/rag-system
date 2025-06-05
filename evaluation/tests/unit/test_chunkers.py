"""Test text chunkers."""

import pytest
from langchain.schema import Document

from rag_platform.chunkers.recursive_chunker import RecursiveChunker


class TestRecursiveChunker:
    """Test RecursiveChunker class."""
    
    def test_chunk_documents(self):
        """Test chunking documents."""
        documents = [
            Document(
                page_content="This is a long text. " * 50,  # ~350 characters
                metadata={"source": "test1"}
            ),
            Document(
                page_content="Another document. " * 30,  # ~210 characters
                metadata={"source": "test2"}
            )
        ]
        
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(documents)
        
        # Should create multiple chunks from the first document
        assert len(chunks) > len(documents)
        
        # Check chunk properties
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20  # Allow for overlap
            assert "source" in chunk.metadata
    
    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        documents = [
            Document(
                page_content="This is a test document with metadata. " * 20,
                metadata={"source": "test", "page": 1, "custom": "value"}
            )
        ]
        
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(documents)
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["custom"] == "value"
    
    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        documents = [Document(page_content=text)]
        
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=10)
        chunks = chunker.chunk(documents)
        
        # Check that chunks have overlap
        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            # Check if there's any common text between chunks
            common_words = set(chunks[0].page_content.split()) & set(chunks[1].page_content.split())
            assert len(common_words) > 0
    
    def test_custom_chunk_parameters(self):
        """Test custom chunk size and overlap."""
        documents = [Document(page_content="Test text. " * 100)]
        
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(documents)
        
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 50
        
        for chunk in chunks:
            assert len(chunk.page_content) <= 250  # chunk_size + overlap