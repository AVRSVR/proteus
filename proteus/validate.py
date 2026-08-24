"""Refold self-consistency: does the designed sequence still encode the fold?

This is the check every other number in Proteus is conditional on. A scorer can
only say a sequence looks better on the terms it measures; it cannot say the
sequence still folds to the backbone the strategies were reasoning about. The
standard answer in the field is self-consistency: predict a structure from the
designed sequence alone and measure how far it lands from the intended one.

The measurement has one classic trap, and it is the same one the earlier
prototype fell into with its MD evaluation:

    diff = predicted - reference
    rmsd = sqrt(mean(sum(diff**2)))

That is not RMSD. Without removing rigid-body translation and rotation first,
it measures how far the molecule drifted and tumbled, not how much its shape
differs. A perfectly rigid structure rotated 30 degrees scores terribly. Every
RMSD here is superposed with Kabsch first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .structure import Structure

#: Field-standard self-consistency thresholds for a designed backbone.
DEFAULT_RMSD_CUTOFF = 2.0
DEFAULT_PLDDT_CUTOFF = 80.0


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal rotation aligning ``mobile`` onto ``target`` (both centred)."""
    covariance = mobile.T @ target
    u, _s, vt = np.linalg.svd(covariance)
    # Guard against a reflection: a mirror image is not a valid superposition.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    return vt.T @ correction @ u.T


def superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``mobile`` rigidly fitted onto ``target``."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape:
        raise ValueError(f"shape mismatch: {mobile.shape} vs {target.shape}")
    mob_centre = mobile.mean(axis=0)
    tgt_centre = target.mean(axis=0)
    rotation = kabsch_rotation(mobile - mob_centre, target - tgt_centre)
    return (mobile - mob_centre) @ rotation.T + tgt_centre


def rmsd(mobile: np.ndarray, target: np.ndarray, superposed: bool = False) -> float:
    """Root-mean-square deviation after optimal superposition.

    Set ``superposed=True`` only if the coordinates are already aligned. The
    default exists because an unsuperposed RMSD is almost always a bug.
    """
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if not superposed:
        mobile = superpose(mobile, target)
    diff = mobile - target
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


@dataclass
class RefoldResult:
    """Outcome of a self-consistency check."""

    sequence: str
    sc_rmsd: float
    plddt: float | None
    passed: bool
    reason: str
    predictor: str = ""

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        plddt = f", pLDDT {self.plddt:.1f}" if self.plddt is not None else ""
        return f"{verdict}  scRMSD {self.sc_rmsd:.2f} A{plddt}  ({self.reason})"


class RefoldGate(ABC):
    """Predicts a structure from sequence and scores self-consistency."""

    name: str = "gate"

    def __init__(self, rmsd_cutoff: float = DEFAULT_RMSD_CUTOFF,
                 plddt_cutoff: float | None = DEFAULT_PLDDT_CUTOFF) -> None:
        self.rmsd_cutoff = rmsd_cutoff
        self.plddt_cutoff = plddt_cutoff

    @abstractmethod
    def predict(self, sequence: str) -> tuple[np.ndarray, float | None]:
        """Return (CA coordinates, mean pLDDT or None) for ``sequence``."""

    def check(self, sequence: str, reference: Structure) -> RefoldResult:
        """Fold ``sequence`` and compare with the backbone it was designed on."""
        if len(sequence) != len(reference):
            raise ValueError(
                f"sequence length {len(sequence)} != reference {len(reference)}"
            )
        coords, plddt = self.predict(sequence)
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (len(reference), 3):
            raise ValueError(
                f"predictor returned {coords.shape}, expected ({len(reference)}, 3)"
            )

        value = rmsd(coords, reference.coords("ca"))
        reasons = []
        passed = True
        if value > self.rmsd_cutoff:
            passed = False
            reasons.append(f"scRMSD {value:.2f} > {self.rmsd_cutoff:.1f} A")
        if self.plddt_cutoff is not None and plddt is not None:
            if plddt < self.plddt_cutoff:
                passed = False
                reasons.append(f"pLDDT {plddt:.1f} < {self.plddt_cutoff:.0f}")
        if passed:
            reasons.append("refolds to the intended backbone")

        return RefoldResult(
            sequence=sequence, sc_rmsd=value, plddt=plddt, passed=passed,
            reason="; ".join(reasons), predictor=self.name,
        )


class ESMFoldGate(RefoldGate):
    """Self-consistency via ESMFold.

    The model is loaded lazily and cached, because it is large (~2.6 GB) and
    slow on CPU. This is emphatically *not* an in-loop filter: a single
    prediction takes minutes without a GPU, so it belongs at the end of a run,
    applied to finalists.
    """

    name = "esmfold"

    def __init__(self, rmsd_cutoff: float = DEFAULT_RMSD_CUTOFF,
                 plddt_cutoff: float | None = DEFAULT_PLDDT_CUTOFF,
                 device: str | None = None,
                 chunk_size: int | None = 64) -> None:
        super().__init__(rmsd_cutoff, plddt_cutoff)
        self.device = device
        self.chunk_size = chunk_size
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as exc:
            raise ImportError(
                "ESMFoldGate needs torch and transformers:\n"
                "    pip install 'transformers>=4.35' accelerate\n"
                "The model weights (~2.6 GB) download on first use."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        if self.chunk_size is not None:
            # Trades speed for a much smaller activation footprint, which is
            # what makes CPU inference feasible at all.
            model.trunk.set_chunk_size(self.chunk_size)
        self._model = model
        self._device = device

    def predict(self, sequence: str) -> tuple[np.ndarray, float | None]:
        # _load first: it raises the actionable install message. Importing
        # torch here beforehand would surface a bare "No module named 'torch'"
        # and make that message unreachable.
        self._load()
        import torch

        tokens = self._tokenizer([sequence], return_tensors="pt",
                                 add_special_tokens=False)
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self._model(**tokens)

        # positions: (layers, batch, residues, atoms, 3); atom 1 is CA.
        ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()
        plddt = float(out["plddt"][0, :, 1].mean().cpu().numpy())
        # ESMFold reports pLDDT in [0, 1] in some versions and [0, 100] in
        # others; normalise to the familiar 0-100 scale.
        if plddt <= 1.0:
            plddt *= 100.0
        return ca, plddt


class NullGate(RefoldGate):
    """A gate that always passes, for runs without a structure predictor.

    Explicit rather than implicit: a run with no refold check has *not* been
    verified, and the result says so rather than quietly omitting the field.
    """

    name = "none"

    def predict(self, sequence: str) -> tuple[np.ndarray, float | None]:
        raise NotImplementedError("NullGate does not predict structures")

    def check(self, sequence: str, reference: Structure) -> RefoldResult:
        return RefoldResult(
            sequence=sequence, sc_rmsd=float("nan"), plddt=None, passed=True,
            reason="no refold check performed", predictor=self.name,
        )
