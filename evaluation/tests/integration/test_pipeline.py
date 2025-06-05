"""Integration tests for the complete RAG pipeline."""

import json
import os
import pickle
from pathlib import Path

import pandas as pd
import pytest

from rag_platform.core.config import ExperimentConfig
from rag_platform.core.experiment import ExperimentRunner


class TestRAGPipeline:
    """Test complete RAG pipeline."""
    
    @pytest.fixture
    def test_dataset(self, tmp_path):
        """Create a test dataset."""
        df = pd.DataFrame({
            "Question": [
                "Какая столица России?",
                "Когда был основан Санкт-Петербург?",
                "Какое самое глубокое озеро в мире?",
            ],
            "Correct Answer": [
                "Москва",
                "1703 год",
                "Байкал",
            ],
            "Context": [
                "Москва - столица России. Население составляет 12 миллионов.",
                "Санкт-Петербург основан Петром I в 1703 году.",
                "Байкал - самое глубокое озеро в мире. Глубина 1642 метра.",
            ],
            "Filename": ["wiki_1", "wiki_2", "wiki_3"],
        })
        
        data_path = tmp_path / "test_data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(df, f)
        
        return str(data_path)
    
    @pytest.fixture
    def pipeline_config(self, test_dataset, tmp_path):
        """Create pipeline configuration."""
        config_dict = {
            "name": "integration_test",
            "description": "Integration test pipeline",
            "data": {
                "source": "local",
                "path": test_dataset,
                "loader": "pickle_loader"
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 100,
                "chunk_overlap": 20
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            },
            "vectorstore": {
                "type": "chroma",
                "collection_name": "test_integration",
                "persist_directory": str(tmp_path / "chroma_test")
            },
            "retriever": {
                "type": "vector",
                "k": 2
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 100
            },
            "rag_chain": {
                "type": "vanilla"
            },
            "evaluation": {
                "metrics": ["faithfulness", "answer_relevancy"],
                "sample_size": 2
            },
            "output": {
                "results_dir": str(tmp_path / "results"),
                "format": "json"
            }
        }
        
        return ExperimentConfig(**config_dict)
    
    @pytest.mark.integration
    def test_full_pipeline(self, pipeline_config):
        """Test the complete RAG pipeline."""
        # Load real API key from .env if available
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if we have a real API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("test-"):
            pytest.skip("Integration test requires real OpenAI API key")
            
        runner = ExperimentRunner(pipeline_config)
        results = runner.run()
        
        assert "experiment_name" in results
        assert results["experiment_name"] == "integration_test"
        assert "evaluation_results" in results
        assert "output_path" in results
        
        # Check output file exists
        output_path = Path(results["output_path"])
        assert output_path.exists()
        
        # Load and verify results
        with open(output_path) as f:
            saved_results = json.load(f)
        
        assert "evaluation_metrics" in saved_results
        assert "complete_results" in saved_results
        assert len(saved_results["complete_results"]) >= 2
    
    def test_pipeline_components_initialization(self, pipeline_config, mock_openai_key):
        """Test that all pipeline components initialize correctly."""
        runner = ExperimentRunner(pipeline_config)
        
        assert runner.loader is not None
        assert runner.chunker is not None
        assert runner.embedding is not None
        assert runner.vectorstore is not None
        assert runner.evaluator is not None