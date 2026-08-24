"""The analysis a strategy reasons over.

A ``DesignContext`` is computed once per structure and handed to every
strategy. Strategies never recompute geometry themselves -- they ask the
context questions. That keeps burial, secondary structure and membrane depth
consistent across a whole generation, and it means adding a strategy requires
no geometry code at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import geometry as geom
from .membrane import MembraneModel, Zone
from .structure import Structure


@dataclass
class DesignContext:
    structure: Structure
    frozen: frozenset[int] = frozenset()
    membrane: MembraneModel | None = None
    dssp: str | None = None
    core_cutoff: float = geom.CORE_CUTOFF
    surface_cutoff: float = geom.SURFACE_CUTOFF
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- basics

    def __len__(self) -> int:
        return len(self.structure)

    @property
    def positions(self) -> list[int]:
        return [r.resi for r in self.structure]

    @property
    def designable(self) -> list[int]:
        """Positions a strategy is permitted to touch."""
        return [p for p in self.positions if p not in self.frozen]

    @cached_property
    def neighbors(self) -> np.ndarray:
        return geom.sidechain_neighbors(self.structure)

    @cached_property
    def layers(self) -> list[str]:
        return geom.layers(self.structure, self.core_cutoff, self.surface_cutoff)

    @cached_property
    def ss(self) -> str:
        return self.dssp if self.dssp else geom.secondary_structure(self.structure)

    @cached_property
    def cb_dist(self) -> np.ndarray:
        return geom.cb_distances(self.structure)

    @cached_property
    def ca_dist(self) -> np.ndarray:
        return geom.ca_distances(self.structure)

    @cached_property
    def chain_breaks(self) -> frozenset[int]:
        return frozenset(self.structure.chain_breaks())

    # ------------------------------------------------------------ per-residue

    def layer(self, resi: int) -> str:
        return self.layers[resi - 1]

    def ss_at(self, resi: int) -> str:
        return self.ss[resi - 1]

    def burial(self, resi: int) -> float:
        return float(self.neighbors[resi - 1])

    def aa(self, resi: int) -> str:
        return self.structure[resi].aa

    def label(self, resi: int) -> str:
        return self.structure.label(resi)

    # -------------------------------------------------------------- membrane

    @property
    def is_membrane(self) -> bool:
        return self.membrane is not None

    @cached_property
    def zones(self) -> list[Zone]:
        if self.membrane is None:
            return ["aqueous"] * len(self.structure)
        return self.membrane.zones(self.structure)

    @cached_property
    def depths(self) -> np.ndarray:
        if self.membrane is None:
            return np.zeros(len(self.structure))
        return self.membrane.depths(self.structure)

    def zone(self, resi: int) -> Zone:
        return self.zones[resi - 1]

    def depth(self, resi: int) -> float:
        return float(self.depths[resi - 1])

    def is_lipid_facing(self, resi: int) -> bool:
        """Exposed *and* inside the bilayer -- the inverted-physics case.

        These positions want hydrophobic residues, the exact opposite of what
        a soluble-protein surface rule would put there.
        """
        return self.zone(resi) == "lipid_core" and self.layer(resi) == "surface"

    def is_membrane_buried(self, resi: int) -> bool:
        """Inside the bilayer but packed against other protein.

        This is where polar and hydrogen-bonding residues legitimately belong
        in a membrane protein -- buried helix-helix interfaces, polar relays,
        and the transport pathway itself.
        """
        return self.zone(resi) == "lipid_core" and self.layer(resi) in ("core", "boundary")

    # ----------------------------------------------------------------- views

    def in_layer(self, *names: str) -> list[int]:
        return [p for p in self.designable if self.layer(p) in names]

    def in_ss(self, *codes: str) -> list[int]:
        return [p for p in self.designable if self.ss_at(p) in codes]

    def in_zone(self, *names: str) -> list[int]:
        return [p for p in self.designable if self.zone(p) in names]

    def ss_segments(self, code: str) -> list[tuple[int, int]]:
        """Contiguous runs of one SS type as inclusive (start, end) pairs."""
        out, start = [], None
        for i, c in enumerate(self.ss, start=1):
            if c == code and start is None:
                start = i
            elif c != code and start is not None:
                out.append((start, i - 1))
                start = None
        if start is not None:
            out.append((start, len(self.ss)))
        return out

    def summary(self) -> str:
        from collections import Counter
        lay = Counter(self.layers)
        lines = [
            f"{len(self.structure)} residues, {len(self.frozen)} frozen",
            f"layers: core {lay['core']}, boundary {lay['boundary']}, surface {lay['surface']}",
            f"ss: H {self.ss.count('H')}, E {self.ss.count('E')}, L {self.ss.count('L')}",
        ]
        if self.membrane is not None:
            from collections import Counter as C
            z = C(self.zones)
            lines.append(f"membrane: {self.membrane.describe()}")
            lines.append(f"zones: lipid_core {z['lipid_core']}, "
                         f"interface {z['interface']}, aqueous {z['aqueous']}")
            lines.append(f"lipid-facing positions: "
                         f"{sum(1 for p in self.positions if self.is_lipid_facing(p))}")
        return "\n".join(lines)
