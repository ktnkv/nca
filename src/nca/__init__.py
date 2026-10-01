from importlib.metadata import version

from .core import NCA
from .neighborhoods import NEIGHBORHOODS

__version__ = version("nca")
__all__ = ["NCA", "NEIGHBORHOODS"]
