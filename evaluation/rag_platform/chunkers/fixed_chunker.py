"""Fixed-size text chunker."""

from typing import List

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from rag_platform.core.base import BaseChunker
from rag_platform.core.registry import registry


class FixedChunker(BaseChunker):
    """Fixed-size character text splitter."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separator="",
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into fixed-size pieces."""
        return self.splitter.split_documents(documents)


registry.register_chunker("fixed", FixedChunker)