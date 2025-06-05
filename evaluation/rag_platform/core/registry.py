"""Component registry for dynamic loading of RAG modules."""

from typing import Any, Dict, List, Type

from rag_platform.core.base import (
    BaseChunker,
    BaseEmbedding,
    BaseEvaluator,
    BaseLoader,
    BaseRAGChain,
    BaseVectorStore,
)


class ComponentRegistry:
    """Registry for RAG components."""
    
    def __init__(self):
        self._loaders: Dict[str, Type[BaseLoader]] = {}
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._embeddings: Dict[str, Type[BaseEmbedding]] = {}
        self._vectorstores: Dict[str, Type[BaseVectorStore]] = {}
        self._retrievers: Dict[str, Any] = {}  # Will hold retriever classes when implemented
        self._chains: Dict[str, Type[BaseRAGChain]] = {}
        self._evaluators: Dict[str, Type[BaseEvaluator]] = {}
    
    def register_loader(self, name: str, loader_class: Type[BaseLoader]) -> None:
        """Register a document loader."""
        self._loaders[name] = loader_class
    
    def register_chunker(self, name: str, chunker_class: Type[BaseChunker]) -> None:
        """Register a text chunker."""
        self._chunkers[name] = chunker_class
    
    def register_embedding(self, name: str, embedding_class: Type[BaseEmbedding]) -> None:
        """Register an embedding model."""
        self._embeddings[name] = embedding_class
    
    def register_vectorstore(self, name: str, vectorstore_class: Type[BaseVectorStore]) -> None:
        """Register a vector store."""
        self._vectorstores[name] = vectorstore_class
    
    def register_retriever(self, name: str, retriever_class: Any) -> None:
        """Register a retriever."""
        self._retrievers[name] = retriever_class
    
    def register_chain(self, name: str, chain_class: Type[BaseRAGChain]) -> None:
        """Register a RAG chain."""
        self._chains[name] = chain_class
    
    def register_evaluator(self, name: str, evaluator_class: Type[BaseEvaluator]) -> None:
        """Register an evaluator."""
        self._evaluators[name] = evaluator_class
    
    def get_loader(self, name: str) -> Type[BaseLoader]:
        """Get a registered loader."""
        # Auto-detection logic
        if name == "auto":
            # Default to file_loader for general file loading
            if "file_loader" in self._loaders:
                return self._loaders["file_loader"]
            # Fallback to first available loader
            if self._loaders:
                return next(iter(self._loaders.values()))
            raise ValueError("No loaders registered for auto-detection")
        
        if name not in self._loaders:
            raise ValueError(f"Unknown loader: {name}")
        return self._loaders[name]
    
    def get_chunker(self, name: str) -> Type[BaseChunker]:
        """Get a registered chunker."""
        if name not in self._chunkers:
            raise ValueError(f"Unknown chunker: {name}")
        return self._chunkers[name]
    
    def get_embedding(self, name: str) -> Type[BaseEmbedding]:
        """Get a registered embedding."""
        if name not in self._embeddings:
            raise ValueError(f"Unknown embedding: {name}")
        return self._embeddings[name]
    
    def get_vectorstore(self, name: str) -> Type[BaseVectorStore]:
        """Get a registered vector store."""
        if name not in self._vectorstores:
            raise ValueError(f"Unknown vector store: {name}")
        return self._vectorstores[name]
    
    def get_retriever(self, name: str) -> Any:
        """Get a registered retriever."""
        if name not in self._retrievers:
            raise ValueError(f"Unknown retriever: {name}")
        return self._retrievers[name]
    
    def get_chain(self, name: str) -> Type[BaseRAGChain]:
        """Get a registered chain."""
        if name not in self._chains:
            raise ValueError(f"Unknown chain: {name}")
        return self._chains[name]
    
    def get_evaluator(self, name: str) -> Type[BaseEvaluator]:
        """Get a registered evaluator."""
        if name not in self._evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        return self._evaluators[name]
    
    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components."""
        return {
            "loaders": list(self._loaders.keys()),
            "chunkers": list(self._chunkers.keys()),
            "embeddings": list(self._embeddings.keys()),
            "vectorstores": list(self._vectorstores.keys()),
            "retrievers": list(self._retrievers.keys()),
            "chains": list(self._chains.keys()),
            "evaluators": list(self._evaluators.keys()),
        }


registry = ComponentRegistry()