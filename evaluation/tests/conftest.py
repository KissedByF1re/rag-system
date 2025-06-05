"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import yaml
from langchain.schema import Document

from rag_platform.core.config import ExperimentConfig


@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(
            page_content="Москва - столица России. Население Москвы составляет около 12 миллионов человек.",
            metadata={
                "source": "wikipedia_1",
                "question": "Какая столица России?",
                "answer": "Москва",
            }
        ),
        Document(
            page_content="Санкт-Петербург был основан Петром I в 1703 году. Это второй по величине город России.",
            metadata={
                "source": "wikipedia_2",
                "question": "Когда был основан Санкт-Петербург?",
                "answer": "1703 год",
            }
        ),
        Document(
            page_content="Байкал - самое глубокое озеро в мире. Его максимальная глубина составляет 1642 метра.",
            metadata={
                "source": "wikipedia_3",
                "question": "Какое самое глубокое озеро в мире?",
                "answer": "Байкал",
            }
        ),
    ]


@pytest.fixture
def sample_config_dict() -> Dict:
    """Sample configuration dictionary."""
    return {
        "experiment": {
            "name": "test_experiment",
            "description": "Test experiment configuration",
            "data": {
                "source": "local",
                "path": "./test_data.pkl",
                "loader": "pickle_loader"
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 100,
                "chunk_overlap": 20
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 1536
            },
            "vectorstore": {
                "type": "chroma",
                "collection_name": "test_collection",
                "persist_directory": "./test_chroma_db"
            },
            "retriever": {
                "type": "vector",
                "k": 3,
                "rerank": False
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 100
            },
            "rag_chain": {
                "type": "vanilla",
                "prompt_template": "default"
            },
            "evaluation": {
                "metrics": ["faithfulness", "answer_relevancy"],
                "sample_size": 10
            },
            "output": {
                "results_dir": "./test_results",
                "format": "json"
            }
        }
    }


@pytest.fixture
def sample_config(sample_config_dict) -> ExperimentConfig:
    """Sample experiment configuration."""
    return ExperimentConfig(**sample_config_dict["experiment"])


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")


@pytest.fixture
def temp_directories():
    """Create temporary directories for testing."""
    dirs = {
        "data": tempfile.mkdtemp(),
        "results": tempfile.mkdtemp(),
        "chroma": tempfile.mkdtemp(),
    }
    
    yield dirs
    
    # Cleanup
    import shutil
    for dir_path in dirs.values():
        shutil.rmtree(dir_path, ignore_errors=True)