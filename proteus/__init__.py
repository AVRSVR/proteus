"""Proteus -- a strategy library for stabilizing designed proteins.

Generative models produce backbones and sequences that look right. Proteus is
the layer after that: given a structure, it diagnoses which stabilization
mechanisms the fold actually admits, applies them, and keeps a record of which
mechanisms earned their place.

Two environments are first-class. Soluble proteins bury hydrophobics and expose
polars; membrane proteins do the reverse inside the bilayer, so membrane depth
is carried alongside burial everywhere rather than bolted on.
"""

from .context import DesignContext
from .fingerprint import Fingerprint
from .knowledge import KnowledgeBase
from .membrane import MembraneModel
from .proposals import Proposal, Resolution, resolve, to_resfile
from .strategies import REGISTRY, Strategy, register
from .structure import Structure, from_arrays, from_pdb
from .validate import NullGate, RefoldGate, RefoldResult, rmsd, superpose

__version__ = "0.1.0"

__all__ = [
    "DesignContext",
    "Fingerprint",
    "KnowledgeBase",
    "MembraneModel",
    "Proposal",
    "Resolution",
    "REGISTRY",
    "Strategy",
    "Structure",
    "RefoldGate",
    "RefoldResult",
    "NullGate",
    "rmsd",
    "superpose",
    "from_arrays",
    "from_pdb",
    "register",
    "resolve",
    "to_resfile",
    "__version__",
]
