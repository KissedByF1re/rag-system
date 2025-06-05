"""File handling utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    ensure_directory(Path(file_path).parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """Save data to pickle file."""
    ensure_directory(Path(file_path).parent)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension in lowercase."""
    return Path(file_path).suffix.lower()


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """List files in directory matching pattern."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return list(directory.glob(pattern))