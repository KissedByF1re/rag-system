from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from service.utils import langchain_to_chat_message

def test_messages_conversion():
    """Test conversion between LangChain messages and our ChatMessage format."""
    # Test Human Message conversion
    human_message = HumanMessage(content="Hello")
    chat_message = langchain_to_chat_message(human_message)
    assert chat_message.type == "human"
    assert chat_message.content == "Hello"
    
    # Test AI Message conversion
    ai_message = AIMessage(content="Hi there!")
    chat_message = langchain_to_chat_message(ai_message)
    assert chat_message.type == "ai"
    assert chat_message.content == "Hi there!"
    
    # Test Tool Message conversion
    tool_message = ToolMessage(content="result", tool_call_id="123")
    chat_message = langchain_to_chat_message(tool_message)
    assert chat_message.type == "tool"
    assert chat_message.content == "result"
    assert chat_message.tool_call_id == "123" 