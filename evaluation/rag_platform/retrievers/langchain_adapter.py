"""Adapter to make custom retrievers compatible with LangChain."""

from typing import List, Optional

from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class LangChainRetrieverAdapter(BaseRetriever):
    """Adapter to make custom retrievers compatible with LangChain chains."""
    
    def __init__(self, custom_retriever, k: int = 5):
        """Initialize adapter with custom retriever.
        
        Args:
            custom_retriever: Custom retriever instance
            k: Number of documents to retrieve
        """
        super().__init__()
        self.custom_retriever = custom_retriever
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents using custom retriever."""
        return self.custom_retriever.retrieve(query, k=self.k)