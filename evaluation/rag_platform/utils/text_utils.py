"""Text processing utilities."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex patterns."""
    # Simple sentence splitting on periods, exclamation marks, and question marks
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter out empty sentences
    sentences = [clean_text(s) for s in sentences if clean_text(s)]
    return sentences


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def count_tokens_approx(text: str) -> int:
    """Approximate token count by splitting on whitespace."""
    return len(text.split())