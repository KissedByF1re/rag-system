"""Loader for pickle files (specifically for the Russian RAG dataset)."""

import pickle
from pathlib import Path
from typing import List

import pandas as pd
from langchain.schema import Document

from rag_platform.core.base import BaseLoader
from rag_platform.core.registry import registry


class PickleLoader(BaseLoader):
    """Loader for pickle files containing pandas DataFrames."""
    
    def load(self, path: str) -> List[Document]:
        """Load documents from a pickle file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            return self._load_from_dataframe(data)
        else:
            raise ValueError(f"Unsupported pickle content type: {type(data)}")
    
    def _load_from_dataframe(self, df: pd.DataFrame) -> List[Document]:
        """Convert DataFrame rows to Documents."""
        documents = []
        
        # Handle both English and Russian column names
        column_mapping = {
            # English columns
            "Question": "question",
            "Correct Answer": "answer", 
            "Context": "content",
            "Filename": "source",
            # Russian columns
            "Вопрос": "question",
            "Правильный ответ": "answer",
            "Контекст": "content", 
            "Файл": "source"
        }
        
        for idx, row in df.iterrows():
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
                "source": source,
                "question": question,
                "answer": answer,
                "row_index": idx,
            }
            
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )
        
        return documents


registry.register_loader("pickle_loader", PickleLoader)