import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.invoke.return_value = {"type": "ai", "content": "This is a test response"}
    agent.stream.return_value = []
    agent.astream.return_value = []
    return agent 