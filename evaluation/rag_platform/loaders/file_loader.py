"""General file loader for various formats."""

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)

from rag_platform.core.base import BaseLoader
from rag_platform.core.registry import registry


class FileLoader(BaseLoader):
    """Loader for various file formats."""
    
    def __init__(self):
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".markdown": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
        }
    
    def load(self, path: str) -> List[Document]:
        """Load documents from file."""
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if file_path.is_dir():
            return self._load_directory(file_path)
        
        suffix = file_path.suffix.lower()
        if suffix not in self.loader_map:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        loader_class = self.loader_map[suffix]
        loader = loader_class(str(file_path))
        
        try:
            return loader.load()
        except Exception as e:
            raise RuntimeError(f"Error loading file {path}: {str(e)}")
    
    def _load_directory(self, dir_path: Path) -> List[Document]:
        """Load all supported files from a directory."""
        documents = []
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.loader_map:
                try:
                    docs = self.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return documents


registry.register_loader("file_loader", FileLoader)