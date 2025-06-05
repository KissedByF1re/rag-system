"""Test configuration management."""

import pytest
from pathlib import Path

from rag_platform.core.config import (
    ExperimentConfig,
    DataConfig,
    ChunkingConfig,
    EmbeddingConfig,
    Settings,
)


class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_from_dict(self, sample_config_dict):
        """Test creating config from dictionary."""
        config = ExperimentConfig.from_dict(sample_config_dict["experiment"])
        
        assert config.name == "test_experiment"
        assert config.description == "Test experiment configuration"
        assert config.data.loader == "pickle_loader"
        assert config.chunking.chunk_size == 100
        assert config.embedding.model == "text-embedding-3-small"
    
    def test_from_yaml(self, temp_config_file):
        """Test loading config from YAML file."""
        config = ExperimentConfig.from_yaml(temp_config_file)
        
        assert config.name == "test_experiment"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
    
    def test_to_yaml(self, sample_config, tmp_path):
        """Test saving config to YAML file."""
        output_path = tmp_path / "test_config.yaml"
        sample_config.to_yaml(str(output_path))
        
        assert output_path.exists()
        
        # Load it back and verify
        loaded_config = ExperimentConfig.from_yaml(str(output_path))
        assert loaded_config.name == sample_config.name
    
    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            # Missing required fields
            ExperimentConfig(name="test")


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_defaults(self):
        """Test default values."""
        config = DataConfig(path="/test/path")
        
        assert config.source == "local"
        assert config.loader == "auto"
        assert config.path == "/test/path"
    
    def test_custom_values(self):
        """Test custom values."""
        config = DataConfig(
            source="wikipedia",
            path="/custom/path",
            loader="custom_loader"
        )
        
        assert config.source == "wikipedia"
        assert config.path == "/custom/path"
        assert config.loader == "custom_loader"


class TestChunkingConfig:
    """Test ChunkingConfig class."""
    
    def test_defaults(self):
        """Test default values."""
        config = ChunkingConfig()
        
        assert config.strategy == "recursive"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
    
    def test_custom_values(self):
        """Test custom values."""
        config = ChunkingConfig(
            strategy="sentence",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        assert config.strategy == "sentence"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100


class TestSettings:
    """Test Settings class."""
    
    def test_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("BATCH_SIZE", "50")
        
        settings = Settings()
        
        assert settings.openai_api_key == "test-key"
        assert settings.default_embedding_model == "custom-model"
        assert settings.batch_size == 50
    
    def test_defaults(self):
        """Test default settings."""
        settings = Settings()
        
        assert settings.default_llm_model == "gpt-4.1-nano"
        assert settings.default_temperature == 0.0
        assert settings.chroma_persist_dir == "./chroma_db"
        assert settings.batch_size == 100