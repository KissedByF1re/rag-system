"""Sentence-based text chunker."""

from typing import List

from langchain.schema import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from rag_platform.core.base import BaseChunker
from rag_platform.core.registry import registry


class SentenceChunker(BaseChunker):
    """Sentence-based text splitter using sentence transformers."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # This splitter respects sentence boundaries
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size,
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents by sentences."""
        chunked_docs = []
        
        for doc in documents:
            # Split text into chunks
            chunks = self.splitter.split_text(doc.page_content)
            
            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                
                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                )
        
        return chunked_docs


registry.register_chunker("sentence", SentenceChunker)