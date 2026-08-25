"""Strategy library.

Importing this package registers every built-in strategy. Third-party
strategies register themselves the same way, by subclassing ``Strategy`` and
applying the ``@register`` decorator.
"""

from .base import MEMBRANE, REGISTRY, SOLUBLE, Strategy, register
from . import soluble as _soluble        # noqa: F401  (import registers)
from . import membrane as _membrane      # noqa: F401
from . import liabilities as _liabilities  # noqa: F401
from . import thermophile as _thermophile  # noqa: F401
from . import packing as _packing        # noqa: F401

__all__ = ["Strategy", "REGISTRY", "register", "SOLUBLE", "MEMBRANE"]
