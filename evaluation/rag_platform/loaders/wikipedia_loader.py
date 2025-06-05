"""Wikipedia loader for fetching articles."""

import re
from typing import List
from urllib.parse import quote

from langchain.schema import Document
from langchain_community.document_loaders import WikipediaLoader as LangchainWikipediaLoader

from rag_platform.core.base import BaseLoader
from rag_platform.core.registry import registry


class WikipediaLoader(BaseLoader):
    """Loader for Wikipedia articles."""
    
    def __init__(self, lang: str = "ru"):
        self.lang = lang
    
    def load(self, path: str) -> List[Document]:
        """Load Wikipedia articles.
        
        Path can be:
        - Wikipedia URL (https://ru.wikipedia.org/?curid=123)
        - Page ID (123)
        - Article title ("Москва")
        """
        documents = []
        
        # Extract page ID from URL
        page_id_match = re.search(r'curid=(\d+)', path)
        if page_id_match:
            page_id = page_id_match.group(1)
            # Convert page ID to title (simplified approach)
            # In production, you'd use Wikipedia API to get the title
            documents = self._load_by_query(f"pageid:{page_id}")
        elif path.isdigit():
            # Direct page ID
            documents = self._load_by_query(f"pageid:{path}")
        else:
            # Assume it's a title
            documents = self._load_by_query(path)
        
        return documents
    
    def _load_by_query(self, query: str) -> List[Document]:
        """Load Wikipedia article by query."""
        try:
            loader = LangchainWikipediaLoader(
                query=query,
                lang=self.lang,
                load_max_docs=1,
            )
            return loader.load()
        except Exception as e:
            print(f"Warning: Failed to load Wikipedia article '{query}': {e}")
            return []


registry.register_loader("wikipedia_loader", WikipediaLoader)