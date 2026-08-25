"""Backend-independent structure representation.

Proteus deliberately does not depend on PyRosetta for its core reasoning. A
``Structure`` holds only what the strategy layer needs: per-residue identity,
backbone geometry, and enough vectors to compute burial. Heavy machinery
(packing, minimisation, scoring) lives behind the optional backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Ideal tetrahedral geometry for reconstructing a CB that isn't there (Gly, or
# a backbone-only model). Constants are the standard Rosetta/Bio idealisation.
_CB_BOND_LENGTH = 1.522
_CB_ROT = np.array([-0.58273431, 0.56802827, -0.54067466])


@dataclass(frozen=True)
class Residue:
    """One residue, indexed by its position in the chain (1-based, contiguous).

    ``resi`` is Proteus' internal index and is always 1..N with no gaps. The
    original PDB numbering is preserved separately in ``pdb_number`` so that
    reports can speak the user's language without the internal maths ever
    having to cope with insertion codes or numbering jumps.
    """

    resi: int
    name3: str
    chain: str
    pdb_number: int
    icode: str
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    cb: np.ndarray
    has_real_cb: bool
    #: CA B-factor. Structure predictors write per-residue confidence here --
    #: pLDDT for AlphaFold and ESMFold -- so it is carried through rather than
    #: discarded.
    bfactor: float = 0.0

    @property
    def aa(self) -> str:
        return THREE_TO_ONE.get(self.name3, "X")

    @property
    def is_glycine(self) -> bool:
        return self.name3 == "GLY"


@dataclass
class Structure:
    residues: list[Residue] = field(default_factory=list)
    source: str | None = None

    def __len__(self) -> int:
        return len(self.residues)

    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)

    def __getitem__(self, resi: int) -> Residue:
        """1-based access, matching ``Residue.resi``."""
        if not 1 <= resi <= len(self.residues):
            raise KeyError(f"residue {resi} out of range 1..{len(self.residues)}")
        return self.residues[resi - 1]

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)

    def coords(self, atom: str = "ca") -> np.ndarray:
        return np.array([getattr(r, atom) for r in self.residues])

    def chain_breaks(self) -> list[int]:
        """Indices after which the chain is discontinuous.

        A break is any consecutive C(i)->N(i+1) distance beyond 2.0 A, or a
        change of chain ID. Loop strategies must not span these.
        """
        breaks = []
        for i in range(len(self.residues) - 1):
            a, b = self.residues[i], self.residues[i + 1]
            if a.chain != b.chain or np.linalg.norm(b.n - a.c) > 2.0:
                breaks.append(a.resi)
        return breaks

    def label(self, resi: int) -> str:
        """Human-facing residue label, e.g. ``A:L47``."""
        r = self[resi]
        return f"{r.chain}:{r.aa}{r.pdb_number}{r.icode.strip()}"


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Place a CB from backbone atoms using ideal tetrahedral geometry."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    basis = np.stack([a, b, np.cross(a, b)], axis=-1)
    norms = np.linalg.norm(basis, axis=0)
    norms[norms == 0] = 1.0
    return ca + _CB_BOND_LENGTH * (basis / norms) @ _CB_ROT


def from_pdb(path: str, chain: str | None = None, model: int = 0) -> Structure:
    """Load a Structure from a PDB or mmCIF file.

    Only residues with a complete N/CA/C backbone are kept; anything else
    cannot be reasoned about geometrically and silently keeping it would
    produce wrong burial numbers rather than an honest error.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    parser = (MMCIFParser(QUIET=True) if str(path).lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    bio = parser.get_structure("s", str(path))
    models = list(bio)
    if not models:
        raise ValueError(f"no models in {path}")

    residues: list[Residue] = []
    skipped = 0
    for ch in models[model]:
        if chain is not None and ch.id != chain:
            continue
        for res in ch:
            if res.id[0] != " ":          # hetero/water
                continue
            if res.get_resname() not in THREE_TO_ONE:
                skipped += 1
                continue
            try:
                n = res["N"].get_coord().astype(float)
                ca = res["CA"].get_coord().astype(float)
                c = res["C"].get_coord().astype(float)
            except KeyError:
                skipped += 1
                continue
            if "CB" in res:
                cb, real = res["CB"].get_coord().astype(float), True
            else:
                cb, real = _virtual_cb(n, ca, c), False
            try:
                bfac = float(res["CA"].get_bfactor())
            except Exception:
                bfac = 0.0
            residues.append(Residue(
                resi=len(residues) + 1,
                name3=res.get_resname(),
                chain=ch.id,
                pdb_number=res.id[1],
                icode=res.id[2],
                n=n, ca=ca, c=c, cb=cb, has_real_cb=real,
                bfactor=bfac,
            ))

    if not residues:
        raise ValueError(f"no usable protein residues found in {path}")
    return Structure(residues=residues, source=str(path))


def from_arrays(
    sequence: str,
    ca: np.ndarray,
    cb: np.ndarray | None = None,
    n: np.ndarray | None = None,
    c: np.ndarray | None = None,
    chain: str = "A",
) -> Structure:
    """Build a Structure directly from coordinate arrays.

    Used by the test suite and by callers who already hold coordinates and do
    not want a round-trip through a PDB file.
    """
    one_to_three = {v: k for k, v in THREE_TO_ONE.items()}
    ca = np.asarray(ca, dtype=float)
    if len(sequence) != len(ca):
        raise ValueError(f"sequence length {len(sequence)} != {len(ca)} CA coords")

    residues = []
    for i, aa in enumerate(sequence):
        ca_i = ca[i]
        n_i = np.asarray(n[i], float) if n is not None else ca_i + np.array([-1.46, 0.0, 0.0])
        c_i = np.asarray(c[i], float) if c is not None else ca_i + np.array([1.52, 0.0, 0.0])
        if cb is not None:
            cb_i, real = np.asarray(cb[i], float), True
        else:
            cb_i, real = _virtual_cb(n_i, ca_i, c_i), False
        residues.append(Residue(
            resi=i + 1, name3=one_to_three.get(aa, "ALA"), chain=chain,
            pdb_number=i + 1, icode=" ",
            n=n_i, ca=ca_i, c=c_i, cb=cb_i, has_real_cb=real,
        ))
    return Structure(residues=residues, source="<arrays>")


def to_pdb(structure: Structure, path: str, sequence: str | None = None) -> None:
    """Write backbone coordinates as a PDB file.

    Only N/CA/C/CB are written -- Proteus reasons over backbone geometry and
    does not build full sidechains. Pass ``sequence`` to thread a designed
    sequence onto the same backbone.
    """
    one_to_three = {v: k for k, v in THREE_TO_ONE.items()}
    lines: list[str] = []
    serial = 1
    for i, res in enumerate(structure.residues):
        name3 = one_to_three.get(sequence[i], res.name3) if sequence else res.name3
        atoms = [("N", res.n), ("CA", res.ca), ("C", res.c)]
        if name3 != "GLY":
            atoms.append(("CB", res.cb))
        for atom_name, xyz in atoms:
            # Fixed-column PDB layout: serial 7-11, atom name 13-16 (single
            # letter elements are offset by one), altLoc 17, resName 18-20,
            # chain 22, resSeq 23-26, iCode 27, coordinates from 31.
            lines.append(
                f"ATOM  {serial:5d}  {atom_name:<3}{'':1}{name3:>3} "
                f"{res.chain}{res.pdb_number:4d}{res.icode:1}   "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {atom_name[0]:>2}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
