"""Ideal-geometry backbone builder for tests.

Builds real N/CA/C coordinates from phi/psi via NeRF (natural extension
reference frame), so geometric code can be tested against structures whose
answer is known by construction rather than against a downloaded PDB.
"""

from __future__ import annotations

import numpy as np

from proteus.structure import Structure, from_arrays

# Ideal peptide geometry.
B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329
A_N_CA_C, A_CA_C_N, A_C_N_CA = 111.2, 116.2, 121.7
OMEGA = 180.0

PHI_PSI = {
    "helix": (-57.0, -47.0),
    "strand": (-139.0, 135.0),
    "polyproline": (-75.0, 145.0),
}


def _place(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           bond: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    angle, torsion = np.deg2rad(angle_deg), np.deg2rad(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    nrm = np.cross(b - a, bc)
    nrm /= np.linalg.norm(nrm)
    m = np.stack([bc, np.cross(nrm, bc), nrm], axis=1)
    d = np.array([
        -bond * np.cos(angle),
        bond * np.sin(angle) * np.cos(torsion),
        bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + m @ d


def build_backbone(phi_psi: list[tuple[float, float]]) -> tuple[np.ndarray, ...]:
    """Return (N, CA, C) arrays for the given per-residue phi/psi."""
    n_res = len(phi_psi)
    N = np.zeros((n_res, 3))
    CA = np.zeros((n_res, 3))
    C = np.zeros((n_res, 3))

    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = np.array([B_N_CA, 0.0, 0.0])
    ang = np.deg2rad(A_N_CA_C)
    C[0] = CA[0] + B_CA_C * np.array([-np.cos(ang), np.sin(ang), 0.0])

    for i in range(1, n_res):
        psi_prev = phi_psi[i - 1][1]
        N[i] = _place(N[i - 1], CA[i - 1], C[i - 1], B_C_N, A_CA_C_N, psi_prev)
        CA[i] = _place(CA[i - 1], C[i - 1], N[i], B_N_CA, A_C_N_CA, OMEGA)
        C[i] = _place(C[i - 1], N[i], CA[i], B_CA_C, A_N_CA_C, phi_psi[i][0])
    return N, CA, C


def make(sequence: str, motif: str | list[tuple[float, float]] = "helix") -> Structure:
    """Build a Structure with ideal geometry for the given sequence."""
    if isinstance(motif, str):
        phi_psi = [PHI_PSI[motif]] * len(sequence)
    else:
        phi_psi = list(motif)
        if len(phi_psi) != len(sequence):
            raise ValueError("motif length must match sequence length")
    N, CA, C = build_backbone(phi_psi)
    return from_arrays(sequence, ca=CA, n=N, c=C)


def make_bundle(n_helices: int = 4, per_helix: int = 18, radius: float = 10.0,
                sequence: str | None = None) -> Structure:
    """Assemble ideal helices into a parallel bundle -- gives a real core."""
    helix = make("A" * per_helix, "helix")
    ca = helix.coords("ca")
    axis = ca[-1] - ca[0]
    axis /= np.linalg.norm(axis)

    # Rotate the helix so its own axis lies along +z.
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z)
    s, c = np.linalg.norm(v), float(np.dot(axis, z))
    if s < 1e-8:
        rot = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    all_n, all_ca, all_c = [], [], []
    for h in range(n_helices):
        theta = 2 * np.pi * h / n_helices
        offset = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        for arr, sink in (("n", all_n), ("ca", all_ca), ("c", all_c)):
            pts = helix.coords(arr) @ rot.T + offset
            sink.append(pts)

    N = np.concatenate(all_n)
    CA = np.concatenate(all_ca)
    C = np.concatenate(all_c)
    seq = sequence or "A" * len(CA)
    return from_arrays(seq[: len(CA)].ljust(len(CA), "A"), ca=CA, n=N, c=C)
