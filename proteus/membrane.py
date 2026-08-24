"""Membrane geometry and depth-aware zoning.

Membrane proteins invert the rule every soluble-protein heuristic depends on.
In a soluble protein, exposed means "make it polar". In the lipid-facing belt
of a membrane protein, exposed means "make it hydrophobic", and buried inside
the bundle is where polar and hydrogen-bonding residues belong. A stabilization
engine that does not know where the bilayer is will confidently do the exact
wrong thing -- the original prototype's ``distance > 11 A -> polar`` rule would
have stripped the lipid-facing surface of a GPCR.

So Proteus treats membrane depth as a first-class coordinate. Burial (how much
protein is in front of this sidechain) and depth (where it sits relative to the
bilayer) are orthogonal, and the two together decide what belongs at a position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .structure import Structure

Zone = Literal["lipid_core", "interface", "aqueous"]

# Kyte-Doolittle hydropathy, used only to orient the bilayer.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2, "X": 0.0,
}

# Default bilayer dimensions (Angstrom, half-widths from the midplane).
LIPID_CORE_HALF = 13.0
INTERFACE_HALF = 18.0


@dataclass
class MembraneModel:
    """A planar bilayer: a midplane point, a normal, and its thickness."""

    center: np.ndarray
    normal: np.ndarray
    lipid_core_half: float = LIPID_CORE_HALF
    interface_half: float = INTERFACE_HALF
    source: str = "estimated"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("membrane normal must be non-zero")
        self.normal = n / norm

    def depth(self, xyz: np.ndarray) -> np.ndarray:
        """Signed distance from the midplane along the normal, in Angstrom."""
        return np.atleast_1d((np.asarray(xyz, dtype=float) - self.center) @ self.normal)

    def zone_of(self, depth: float) -> Zone:
        d = abs(float(depth))
        if d <= self.lipid_core_half:
            return "lipid_core"
        if d <= self.interface_half:
            return "interface"
        return "aqueous"

    def zones(self, structure: Structure) -> list[Zone]:
        d = self.depth(structure.coords("ca"))
        return [self.zone_of(x) for x in d]

    def depths(self, structure: Structure) -> np.ndarray:
        return self.depth(structure.coords("ca"))

    def describe(self) -> str:
        c = ", ".join(f"{v:.1f}" for v in self.center)
        n = ", ".join(f"{v:.2f}" for v in self.normal)
        return (f"bilayer center ({c}) normal ({n}) "
                f"core +/-{self.lipid_core_half:.0f}A ({self.source})")


def from_opm_dummies(path: str) -> MembraneModel | None:
    """Read the bilayer from OPM-style DUM/membrane pseudo-atoms if present.

    OPM deposits two planes of dummy atoms marking the hydrophobic boundaries.
    When they are there, they are a far better answer than any estimate.
    """
    zs: list[float] = []
    xyz: list[list[float]] = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line[:6] not in ("HETATM", "ATOM  "):
                continue
            if line[17:20].strip() not in ("DUM", "MEM"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            xyz.append([x, y, z])
            zs.append(z)

    if len(xyz) < 8:
        return None

    pts = np.array(xyz)
    # OPM aligns the bilayer normal to z, so the planes separate along z.
    upper = np.array([p for p in pts if p[2] > np.mean(zs)])
    lower = np.array([p for p in pts if p[2] <= np.mean(zs)])
    if len(upper) == 0 or len(lower) == 0:
        return None

    half = float(abs(upper[:, 2].mean() - lower[:, 2].mean()) / 2.0)
    center = np.array([pts[:, 0].mean(), pts[:, 1].mean(),
                       (upper[:, 2].mean() + lower[:, 2].mean()) / 2.0])
    return MembraneModel(
        center=center,
        normal=np.array([0.0, 0.0, 1.0]),
        lipid_core_half=half,
        interface_half=half + 5.0,
        source="OPM dummy atoms",
    )


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    """Roughly uniform directions on a hemisphere (antipodes are equivalent)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - i / n)          # 0 .. pi/2
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def estimate(
    structure: Structure,
    exposure: np.ndarray | None = None,
    n_directions: int = 400,
    lipid_core_half: float = LIPID_CORE_HALF,
) -> MembraneModel:
    """Estimate the bilayer from the structure's hydrophobic belt.

    A membrane protein presents a band of exposed hydrophobic residues around
    its waist. We search directions for the one whose perpendicular slab best
    concentrates exposed hydrophobicity, then slide the slab along that axis to
    the best offset. This is a coarse but honest geometric fit; when OPM dummy
    atoms are available, prefer :func:`from_opm_dummies`.
    """
    from .geometry import sidechain_neighbors

    ca = structure.coords("ca")
    if exposure is None:
        counts = sidechain_neighbors(structure)
        # Exposure weight: surface residues count, buried ones do not.
        exposure = np.clip(1.0 - counts / 5.2, 0.0, 1.0)

    hydro = np.array([KD.get(r.aa, 0.0) for r in structure])
    weight = exposure * hydro                     # exposed + hydrophobic -> large

    centroid = ca.mean(axis=0)
    best = (-np.inf, np.array([0.0, 0.0, 1.0]), 0.0)

    for direction in _fibonacci_hemisphere(n_directions):
        proj = (ca - centroid) @ direction
        # Slide the slab; evaluate weight captured inside it.
        for offset in np.linspace(proj.min(), proj.max(), 25):
            inside = np.abs(proj - offset) <= lipid_core_half
            if inside.sum() < 3:
                continue
            score = float(weight[inside].sum() - 0.35 * weight[~inside].sum())
            if score > best[0]:
                best = (score, direction, float(offset))

    _, normal, offset = best
    return MembraneModel(
        center=centroid + normal * offset,
        normal=normal,
        lipid_core_half=lipid_core_half,
        interface_half=lipid_core_half + 5.0,
        source="estimated from hydrophobic belt",
    )
