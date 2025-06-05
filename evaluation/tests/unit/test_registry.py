"""Test component registry."""

import pytest

from rag_platform.core.base import BaseLoader, BaseChunker
from rag_platform.core.registry import ComponentRegistry


class MockLoader(BaseLoader):
    """Mock loader for testing."""
    def load(self, path: str):
        return []


class MockChunker(BaseChunker):
    """Mock chunker for testing."""
    def chunk(self, documents):
        return documents


class TestComponentRegistry:
    """Test ComponentRegistry class."""
    
    def test_register_and_get_loader(self):
        """Test registering and retrieving a loader."""
        registry = ComponentRegistry()
        registry.register_loader("mock", MockLoader)
        
        loader_class = registry.get_loader("mock")
        assert loader_class == MockLoader
        
        # Test instance creation
        loader = loader_class()
        assert isinstance(loader, MockLoader)
    
    def test_register_and_get_chunker(self):
        """Test registering and retrieving a chunker."""
        registry = ComponentRegistry()
        registry.register_chunker("mock", MockChunker)
        
        chunker_class = registry.get_chunker("mock")
        assert chunker_class == MockChunker
    
    def test_get_unknown_component(self):
        """Test retrieving unknown component raises error."""
        registry = ComponentRegistry()
        
        with pytest.raises(ValueError, match="Unknown loader: unknown"):
            registry.get_loader("unknown")
        
        with pytest.raises(ValueError, match="Unknown chunker: unknown"):
            registry.get_chunker("unknown")
    
    def test_list_components(self):
        """Test listing all registered components."""
        registry = ComponentRegistry()
        registry.register_loader("mock_loader", MockLoader)
        registry.register_chunker("mock_chunker", MockChunker)
        
        components = registry.list_components()
        
        assert "mock_loader" in components["loaders"]
        assert "mock_chunker" in components["chunkers"]
        assert isinstance(components["embeddings"], list)
        assert isinstance(components["vectorstores"], list)