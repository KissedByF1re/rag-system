"""Test document loaders."""

import pickle
import pytest
from pathlib import Path
import pandas as pd

from rag_platform.loaders.pickle_loader import PickleLoader


class TestPickleLoader:
    """Test PickleLoader class."""
    
    def test_load_dataframe(self, tmp_path):
        """Test loading DataFrame from pickle."""
        # Create test data
        df = pd.DataFrame({
            "Question": ["What is the capital?", "When was it founded?"],
            "Correct Answer": ["Moscow", "1147"],
            "Context": ["Moscow is the capital of Russia.", "Moscow was founded in 1147."],
            "Filename": ["file1.txt", "file2.txt"]
        })
        
        # Save to pickle
        pickle_path = tmp_path / "test_data.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(df, f)
        
        # Load with PickleLoader
        loader = PickleLoader()
        documents = loader.load(str(pickle_path))
        
        assert len(documents) == 2
        assert documents[0].page_content == "Moscow is the capital of Russia."
        assert documents[0].metadata["question"] == "What is the capital?"
        assert documents[0].metadata["answer"] == "Moscow"
        assert documents[0].metadata["source"] == "file1.txt"
        assert documents[1].metadata["row_index"] == 1
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = PickleLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.pkl")
    
    def test_load_non_dataframe(self, tmp_path):
        """Test loading non-DataFrame pickle raises error."""
        # Save a list instead of DataFrame
        pickle_path = tmp_path / "test_list.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump([1, 2, 3], f)
        
        loader = PickleLoader()
        
        with pytest.raises(ValueError, match="Unsupported pickle content type"):
            loader.load(str(pickle_path))
    
    def test_load_empty_context(self, tmp_path):
        """Test handling empty context field."""
        # Create test data with empty context
        df = pd.DataFrame({
            "Question": ["What is the answer?"],
            "Correct Answer": ["42"],
            "Context": [""],  # Empty context
            "Filename": ["file1.txt"]
        })
        
        pickle_path = tmp_path / "test_empty.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(df, f)
        
        loader = PickleLoader()
        documents = loader.load(str(pickle_path))
        
        # Should use answer as content when context is empty
        assert documents[0].page_content == "42"