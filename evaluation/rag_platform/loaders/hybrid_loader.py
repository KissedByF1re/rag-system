"""Hybrid loader that combines full Wikipedia articles with QA evaluation data."""

import json
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document

from rag_platform.core.base import BaseLoader
from rag_platform.core.registry import registry


class HybridLoader(BaseLoader):
    """Loader that uses full Wikipedia for knowledge base but original QA for evaluation."""
    
    def __init__(self, 
                 wikipedia_kb_path: str = "./data/wikipedia_articles/wikipedia_knowledge_base.json",
                 qa_dataset_path: str = "./data/ru_rag_test_dataset.pkl"):
        self.wikipedia_kb_path = wikipedia_kb_path
        self.qa_dataset_path = qa_dataset_path
    
    def load(self, path: str) -> List[Document]:
        """Load both full Wikipedia articles and QA evaluation data."""
        # For hybrid loader, path is ignored - we use configured paths
        
        # Load full Wikipedia articles for knowledge base
        kb_documents = self._load_wikipedia_kb()
        
        # Load QA dataset for evaluation questions
        qa_documents = self._load_qa_dataset()
        
        # Combine but mark them differently
        all_documents = kb_documents + qa_documents
        
        print(f"Hybrid loader: {len(kb_documents)} Wikipedia articles + {len(qa_documents)} QA pairs")
        
        return all_documents
    
    def _load_wikipedia_kb(self) -> List[Document]:
        """Load full Wikipedia articles."""
        kb_path = Path(self.wikipedia_kb_path)
        if not kb_path.exists():
            print(f"WARNING: Wikipedia KB not found: {kb_path}")
            return []
        
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        documents = []
        articles = kb_data.get('articles', [])
        
        for article in articles:
            content = article.get('content', '').strip()
            if not content or len(content) < 100:
                continue
            
            metadata = {
                'source': f"wikipedia_{article.get('page_id', 'unknown')}",
                'title': article.get('title', 'Unknown'),
                'page_id': article.get('page_id', 'unknown'),
                'url': article.get('url', ''),
                'article_length': len(content),
                'source_type': 'knowledge_base',  # Mark as KB content
                'document_type': 'wikipedia_article'
            }
            
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return documents
    
    def _load_qa_dataset(self) -> List[Document]:
        """Load QA dataset for evaluation questions."""
        qa_path = Path(self.qa_dataset_path)
        if not qa_path.exists():
            print(f"WARNING: QA dataset not found: {qa_path}")
            return []
        
        with open(qa_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            return []
        
        documents = []
        
        # Column mapping for Russian dataset
        column_mapping = {
            "Вопрос": "question",
            "Правильный ответ": "answer",
            "Контекст": "content", 
            "Файл": "source"
        }
        
        for idx, row in data.iterrows():
            # Extract data using column mapping
            question = ""
            answer = ""
            content = ""
            source = "unknown"
            
            for col, mapped_name in column_mapping.items():
                if col in row:
                    value = row[col] if pd.notna(row[col]) else ""
                    if mapped_name == "question":
                        question = value
                    elif mapped_name == "answer":
                        answer = value
                    elif mapped_name == "content":
                        content = value
                    elif mapped_name == "source":
                        source = value
            
            # Use answer as content if context is empty
            if not content and answer:
                content = answer
                
            metadata = {
                'source': source,
                'question': question,
                'answer': answer,
                'row_index': idx,
                'source_type': 'evaluation',  # Mark as evaluation content
                'document_type': 'qa_pair'
            }
            
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return documents


registry.register_loader("hybrid_loader", HybridLoader)