"""Document loaders for various file formats."""

# Import modules to trigger registration
from . import file_loader
from . import pickle_loader
from . import wikipedia_loader
from . import wikipedia_kb_loader
from . import hybrid_loader

from .file_loader import FileLoader
from .pickle_loader import PickleLoader
from .wikipedia_loader import WikipediaLoader
from .wikipedia_kb_loader import WikipediaKBLoader
from .hybrid_loader import HybridLoader

__all__ = ["FileLoader", "PickleLoader", "WikipediaLoader", "WikipediaKBLoader", "HybridLoader"]