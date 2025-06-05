"""ChromaDB vector store implementation."""

from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from rag_platform.core.base import BaseVectorStore
from rag_platform.core.registry import registry


class ChromaStore(BaseVectorStore):
    """ChromaDB vector store wrapper."""
    
    def __init__(
        self,
        embedding_function,
        collection_name: str = "default",
        persist_directory: str = "./chroma_db",
    ):
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )
        else:
            self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        if self.vectorstore is None:
            raise ValueError("No documents in vector store")
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, **kwargs):
        """Get the vector store as a retriever."""
        if self.vectorstore is None:
            raise ValueError("No documents in vector store")
        return self.vectorstore.as_retriever(**kwargs)


registry.register_vectorstore("chroma", ChromaStore)