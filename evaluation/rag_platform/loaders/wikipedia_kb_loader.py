"""Loader for full Wikipedia knowledge base."""

import json
from pathlib import Path
from typing import List

from langchain.schema import Document

from rag_platform.core.base import BaseLoader
from rag_platform.core.registry import registry


class WikipediaKBLoader(BaseLoader):
    """Loader for full Wikipedia knowledge base articles."""
    
    def load(self, path: str) -> List[Document]:
        """Load documents from Wikipedia knowledge base JSON."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        if 'articles' not in kb_data:
            raise ValueError("Invalid knowledge base format: missing 'articles' key")
        
        documents = []
        articles = kb_data['articles']
        
        print(f"Loading {len(articles)} full Wikipedia articles...")
        
        for article in articles:
            # Skip articles without content
            content = article.get('content', '').strip()
            if not content or len(content) < 100:
                continue
            
            metadata = {
                'source': f"wikipedia_{article.get('page_id', 'unknown')}",
                'title': article.get('title', 'Unknown'),
                'page_id': article.get('page_id', 'unknown'),
                'url': article.get('url', ''),
                'article_length': len(content),
                'source_type': 'full_wikipedia_article'
            }
            
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        print(f"Loaded {len(documents)} full Wikipedia articles")
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"Total content: {total_chars:,} characters (~{total_chars//4:,} tokens)")
        
        return documents


registry.register_loader("wikipedia_kb_loader", WikipediaKBLoader)