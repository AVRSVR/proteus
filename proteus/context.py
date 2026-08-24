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

    # ------------------------------------------------- precomputed for scoring
    #
    # These depend only on the backbone, never on the sequence threaded onto
    # it, so they are computed once per context and reused for every scoring
    # call. Without this the aggregation term is O(n^2) in Python on every
    # evaluation, which puts real proteins out of reach.

    @cached_property
    def layer_index(self) -> np.ndarray:
        """0 = core, 1 = boundary, 2 = surface, as an array."""
        order = {"core": 0, "boundary": 1, "surface": 2}
        return np.array([order[l] for l in self.layers], dtype=np.int8)

    @cached_property
    def is_core(self) -> np.ndarray:
        return self.layer_index == 0

    @cached_property
    def is_surface(self) -> np.ndarray:
        return self.layer_index == 2

    @cached_property
    def is_helix(self) -> np.ndarray:
        return np.array([c == "H" for c in self.ss])

    @cached_property
    def is_strand(self) -> np.ndarray:
        return np.array([c == "E" for c in self.ss])

    @cached_property
    def in_lipid_core(self) -> np.ndarray:
        return np.array([z == "lipid_core" for z in self.zones])

    def patch_neighbors(self, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sparse neighbour list over aggregation-relevant positions.

        Returns ``(env_idx, rows, cols)`` where ``env_idx`` maps subset indices
        back to residue indices, and ``rows``/``cols`` enumerate pairs within
        ``radius``.

        A dense mask would make the aggregation term an O(n^2) matrix product
        on every evaluation -- 120 ms for a 3000-residue protein, which is far
        too slow for a loop that scores tens of thousands of candidates. With
        an 8 A cutoff the true neighbour graph is very sparse, so the pair list
        is built once per structure and each evaluation costs O(pairs).
        """
        key = ("patch_neighbors", round(float(radius), 3))
        cached = self._cache.get(key)
        if cached is None:
            env_idx = np.flatnonzero(self.aggregation_environment)
            if env_idx.size == 0:
                cached = (env_idx, np.empty(0, np.intp), np.empty(0, np.intp))
            else:
                sub = self.cb_dist[np.ix_(env_idx, env_idx)] <= radius
                np.fill_diagonal(sub, False)
                rows, cols = np.nonzero(sub)
                cached = (env_idx, rows, cols)
            self._cache[key] = cached
        return cached

    @cached_property
    def cap_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks for helix N-cap and C-cap positions.

        The N-cap is the residue immediately preceding a helix, whose sidechain
        can satisfy the first backbone NH groups; the C-cap is the residue
        following it. Both are real stabilization sites, and without them in
        the objective the capping strategy is invisible to selection.
        """
        n_res = len(self.structure)
        n_cap = np.zeros(n_res, dtype=bool)
        c_cap = np.zeros(n_res, dtype=bool)
        for start, end in self.ss_segments("H"):
            if end - start + 1 < 5:
                continue
            if start - 1 >= 1:
                n_cap[start - 2] = True
            if end + 1 <= n_res:
                c_cap[end] = True
        return n_cap, c_cap

    @cached_property
    def is_loop(self) -> np.ndarray:
        return np.array([c == "L" for c in self.ss])

    @cached_property
    def aggregation_environment(self) -> np.ndarray:
        """Positions where an exposed hydrophobic is a liability.

        Surface positions, except those facing lipid -- inside the bilayer an
        exposed hydrophobic is correct, not an aggregation risk.
        """
        env = self.is_surface.copy()
        if self.is_membrane:
            env &= ~self.in_lipid_core
        return env

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
