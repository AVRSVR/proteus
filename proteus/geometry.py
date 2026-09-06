"""Geometric analysis: burial, secondary structure, and pairwise distances.

Burial is computed with the cone-based *sidechain neighbour* count rather than
distance from the centroid. Centroid distance only behaves sensibly for roughly
spherical globules; for a helical bundle or any elongated fold it mislabels
buried termini as "surface" and exposed loops as "core". The neighbour count
asks the question that actually matters -- how much protein is stacked in front
of this sidechain -- and is orientation-aware, so it transfers to any topology.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .structure import Structure

Layer = Literal["core", "boundary", "surface"]

# Rosetta LayerSelector defaults for the sidechain_neighbors measure.
DIST_MIDPOINT = 9.0
DIST_STEEPNESS = 1.0
ANGLE_SHIFT = 0.5
ANGLE_EXPONENT = 2.0

# Weight of the direction-free neighbour count blended into the cone.
#
# The cone alone answers "is this sidechain pointing into protein", which is
# not the same question as "is this residue packed". Its angular term is
# ((cos t + 0.5) / 1.5) ** 2, so a neighbour more than 120 degrees off the
# CA->CB axis contributes exactly nothing and one at 90 degrees contributes
# 0.11. A residue on a helix-packing face often has the neighbouring helix in
# precisely that discarded arc.
#
# Measured on a three-helix design: W11 sits 6.5 A from the molecular centre
# with 17 neighbours inside 10 A and scored 1.98, while a genuinely exposed
# lysine 27 A out with 5 neighbours scored 0.68. The two were nearly
# indistinguishable, and mutating W11 collapsed the fold from 1.2 A to 27 A
# while the classifier called it surface.
#
# Blending in an isotropic term fixes the ordering without discarding the
# directional signal, which is still worth most of the weight. At 0.15 the
# fold-critical positions separate from genuinely exposed ones while the
# global layer proportions stay where they were.
ISOTROPIC_WEIGHT = 0.15

# Cutoffs are set from the distribution over 23 natural proteins (S669), at
# the percentiles that give roughly a fifth core and two fifths surface --
# the proportions a globular protein is expected to show.
CORE_CUTOFF = 7.0
SURFACE_CUTOFF = 2.8


def sidechain_neighbors(structure: Structure) -> np.ndarray:
    """Per-residue neighbour count; higher means more buried.

    For residue *i* the sidechain direction is CA->CB. Every other residue *j*
    contributes a product of two terms: a sigmoid in the CB_i->CA_j distance,
    and an angular term weighting neighbours in the cone the sidechain points
    into. A residue with protein packed in front of its sidechain scores high
    even when it sits far from the molecular centre.

    The cone is then blended with the same distance term summed over every
    neighbour regardless of direction. On its own the cone answers a narrower
    question than the one burial is asked -- see :data:`ISOTROPIC_WEIGHT` for
    the case that made the difference matter.
    """
    ca = structure.coords("ca")
    cb = structure.coords("cb")
    n = len(structure)
    if n < 2:
        return np.zeros(n)

    # Sidechain direction, one unit vector per residue.
    vec = cb - ca
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    direction = vec / norms

    # displacement[i, j] = CA_j - CB_i
    displacement = ca[None, :, :] - cb[:, None, :]
    dist = np.linalg.norm(displacement, axis=2)
    np.fill_diagonal(dist, np.inf)            # a residue is not its own neighbour

    safe = np.where(np.isfinite(dist) & (dist > 0), dist, 1.0)
    unit = displacement / safe[:, :, None]

    dist_term = 1.0 / (1.0 + np.exp(DIST_STEEPNESS * (dist - DIST_MIDPOINT)))
    dist_term[~np.isfinite(dist)] = 0.0

    cos_theta = np.einsum("ik,ijk->ij", direction, unit)
    angle_term = np.clip((cos_theta + ANGLE_SHIFT) / (1.0 + ANGLE_SHIFT), 0.0, 1.0)
    angle_term = angle_term ** ANGLE_EXPONENT

    directional = (dist_term * angle_term).sum(axis=1)
    isotropic = dist_term.sum(axis=1)
    return directional + ISOTROPIC_WEIGHT * isotropic


def layers(
    structure: Structure,
    core_cutoff: float = CORE_CUTOFF,
    surface_cutoff: float = SURFACE_CUTOFF,
) -> list[Layer]:
    """Assign each residue to core / boundary / surface."""
    counts = sidechain_neighbors(structure)
    out: list[Layer] = []
    for c in counts:
        if c >= core_cutoff:
            out.append("core")
        elif c <= surface_cutoff:
            out.append("surface")
        else:
            out.append("boundary")
    return out


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral in degrees, IUPAC convention."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.linalg.norm(b1)
    if n1 == 0:
        return float("nan")
    b1 = b1 / n1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def dihedrals(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    """Backbone phi/psi in degrees; NaN where undefined (termini, chain breaks)."""
    n_res = len(structure)
    phi = np.full(n_res, np.nan)
    psi = np.full(n_res, np.nan)
    breaks = set(structure.chain_breaks())

    for i in range(n_res):
        r = structure.residues[i]
        if i > 0 and (i) not in breaks:
            prev = structure.residues[i - 1]
            if np.linalg.norm(r.n - prev.c) <= 2.0:
                phi[i] = _dihedral(prev.c, r.n, r.ca, r.c)
        if i < n_res - 1 and (i + 1) not in breaks:
            nxt = structure.residues[i + 1]
            if np.linalg.norm(nxt.n - r.c) <= 2.0:
                psi[i] = _dihedral(r.n, r.ca, r.c, nxt.n)
    return phi, psi


def secondary_structure(structure: Structure, min_run: int = 3) -> str:
    """Assign H (helix), E (strand), L (loop) from backbone dihedrals.

    This is a deliberately simple Ramachandran-region assignment followed by a
    run-length filter, not a hydrogen-bond method like DSSP. Proteus uses SS to
    decide *where a strategy is allowed to act* -- capping a helix terminus,
    rigidifying a loop -- and for that a robust dependency-free assignment is
    worth more than the last few percent of accuracy. Pass an externally
    computed DSSP string to any strategy that needs the real thing.
    """
    phi, psi = dihedrals(structure)
    raw = []
    for f, p in zip(phi, psi):
        if np.isnan(f) or np.isnan(p):
            raw.append("L")
        elif -160.0 <= f <= -20.0 and -120.0 <= p <= 50.0:
            raw.append("H")
        elif -180.0 <= f <= -40.0 and (p >= 90.0 or p <= -150.0):
            raw.append("E")
        else:
            raw.append("L")

    # Drop runs shorter than min_run -- an isolated "helix" residue is noise.
    out = list(raw)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if out[i] in "HE" and (j - i) < min_run:
            for k in range(i, j):
                out[k] = "L"
        i = j
    return "".join(out)


def cb_distances(structure: Structure) -> np.ndarray:
    """Full CB-CB distance matrix (virtual CB used for glycine)."""
    cb = structure.coords("cb")
    return np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=2)


def ca_distances(structure: Structure) -> np.ndarray:
    ca = structure.coords("ca")
    return np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=2)


def sequence_separation(n_res: int) -> np.ndarray:
    idx = np.arange(n_res)
    return np.abs(idx[:, None] - idx[None, :])
