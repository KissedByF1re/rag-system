"""Recursive text splitter for hierarchical chunking."""

from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_platform.core.base import BaseChunker
from rag_platform.core.registry import registry


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        return self.splitter.split_documents(documents)


registry.register_chunker("recursive", RecursiveChunker)