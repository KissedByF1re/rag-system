"""Configuration management for RAG experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data source configuration."""
    source: str = Field(default="local", description="Data source type")
    path: str = Field(description="Path to data file or directory")
    loader: str = Field(default="auto", description="Loader type to use")


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    strategy: str = Field(default="recursive", description="Chunking strategy")
    chunk_size: int = Field(default=512, description="Size of text chunks")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Model name")
    dimension: Optional[int] = Field(default=None, description="Embedding dimension")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = Field(default="chroma", description="Vector store type")
    collection_name: str = Field(default="default", description="Collection name")
    persist_directory: str = Field(default="./chroma_db", description="Persistence directory")


class RetrieverConfig(BaseModel):
    """Retriever configuration."""
    type: str = Field(default="vector", description="Retriever type")
    k: int = Field(default=5, description="Number of documents to retrieve")
    rerank: bool = Field(default=False, description="Whether to use reranking")
    reranker: Optional[str] = Field(default=None, description="Reranker to use")


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    max_tokens: int = Field(default=500, description="Maximum tokens to generate")


class RAGChainConfig(BaseModel):
    """RAG chain configuration."""
    type: str = Field(default="vanilla", description="RAG chain type")
    prompt_template: str = Field(default="default", description="Prompt template to use")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: List[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="RAGAS metrics to compute"
    )
    sample_size: Optional[int] = Field(default=None, description="Number of samples to evaluate")
    evaluator: str = Field(default="ragas", description="Evaluator type to use")


class OutputConfig(BaseModel):
    """Output configuration."""
    results_dir: str = Field(default="./results", description="Results directory")
    format: str = Field(default="json", description="Output format")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    name: str = Field(description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retriever: RetrieverConfig
    llm: LLMConfig
    rag_chain: RAGChainConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict["experiment"])

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {"experiment": self.dict()}
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    default_embedding_model: str = Field(
        default="text-embedding-3-small", env="DEFAULT_EMBEDDING_MODEL"
    )
    default_llm_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    default_temperature: float = Field(default=0.0, env="DEFAULT_TEMPERATURE")
    
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    faiss_index_dir: str = Field(default="./faiss_indexes", env="FAISS_INDEX_DIR")
    
    mlflow_tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    experiment_results_dir: str = Field(default="./results", env="EXPERIMENT_RESULTS_DIR")
    
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"