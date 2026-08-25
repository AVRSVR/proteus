"""Superposition, RMSD, and the refold gate.

The RMSD tests exist because the prototype computed deviation as a raw
coordinate difference with no superposition, which measures rigid-body drift
rather than shape change. Each test below fails against that implementation.
"""

import numpy as np
import pytest

from proteus import validate
from proteus.validate import (ESMFoldGate, NullGate, RefoldGate, rmsd,
                              superpose)

from . import _synthetic as syn


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(degrees)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


# ----------------------------------------------------------------------- rmsd

def test_rmsd_of_identical_structures_is_zero():
    coords = syn.make("A" * 20, "helix").coords("ca")
    assert rmsd(coords, coords) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_is_invariant_to_translation():
    """A translated copy is the same shape and must score zero."""
    coords = syn.make("A" * 20, "helix").coords("ca")
    moved = coords + np.array([100.0, -50.0, 25.0])
    assert rmsd(moved, coords) == pytest.approx(0.0, abs=1e-8)


def test_rmsd_is_invariant_to_rotation():
    """The bug this guards: an unsuperposed difference measures tumbling.

    A rigidly rotated structure has identical shape. Without superposition
    this scores in the tens of angstroms and would fail any sane cutoff.
    """
    coords = syn.make("A" * 25, "helix").coords("ca")
    rotated = coords @ _rotation([0.3, 1.0, 0.2], 30.0).T
    assert rmsd(rotated, coords) == pytest.approx(0.0, abs=1e-8)

    naive = float(np.sqrt(((rotated - coords) ** 2).sum(axis=1).mean()))
    assert naive > 5.0, "test structure should tumble far enough to matter"


def test_rmsd_detects_real_deformation():
    coords = syn.make("A" * 25, "helix").coords("ca")
    bent = coords.copy()
    bent[12:] += np.array([3.0, 0.0, 0.0])
    assert rmsd(bent, coords) > 0.5


def test_superpose_does_not_mirror():
    """A reflection is not a valid superposition.

    Without the determinant correction in Kabsch, a mirrored structure
    superposes onto its original with near-zero RMSD, which would pass a
    mirror-image fold as self-consistent.
    """
    coords = syn.make("A" * 25, "helix").coords("ca")
    mirrored = coords * np.array([1.0, 1.0, -1.0])
    assert rmsd(mirrored, coords) > 1.0


def test_superpose_returns_aligned_coordinates():
    coords = syn.make("A" * 20, "helix").coords("ca")
    moved = coords @ _rotation([0, 0, 1], 45.0).T + np.array([10.0, 0.0, 0.0])
    fitted = superpose(moved, coords)
    assert np.allclose(fitted, coords, atol=1e-8)


def test_rmsd_rejects_shape_mismatch():
    a = np.zeros((10, 3))
    b = np.zeros((11, 3))
    with pytest.raises(ValueError):
        rmsd(a, b)


# ----------------------------------------------------------------------- gate

class FakeGate(RefoldGate):
    """A predictor with a controllable answer, for testing gate logic."""

    name = "fake"

    def __init__(self, coords, plddt=None, **kw):
        super().__init__(**kw)
        self._coords = coords
        self._plddt = plddt

    def predict(self, sequence):
        return self._coords, self._plddt


def test_gate_passes_a_self_consistent_prediction():
    st = syn.make("A" * 20, "helix")
    gate = FakeGate(st.coords("ca"), plddt=92.0)
    result = gate.check(st.sequence, st)
    assert result.passed
    assert result.sc_rmsd == pytest.approx(0.0, abs=1e-8)
    assert "refolds" in result.reason


def test_gate_fails_on_high_rmsd():
    st = syn.make("A" * 20, "helix")
    wrong = syn.make("A" * 20, "strand").coords("ca")
    result = FakeGate(wrong, plddt=95.0).check(st.sequence, st)
    assert not result.passed
    assert "scRMSD" in result.reason


def test_gate_fails_on_low_plddt():
    """A confident-looking backbone the model does not believe in still fails."""
    st = syn.make("A" * 20, "helix")
    result = FakeGate(st.coords("ca"), plddt=41.0).check(st.sequence, st)
    assert not result.passed
    assert "pLDDT" in result.reason


def test_gate_tolerates_a_predictor_without_plddt():
    st = syn.make("A" * 20, "helix")
    result = FakeGate(st.coords("ca"), plddt=None).check(st.sequence, st)
    assert result.passed
    assert result.plddt is None


def test_gate_is_rotation_invariant():
    """A correct fold in a different frame must pass."""
    st = syn.make("A" * 22, "helix")
    rotated = st.coords("ca") @ _rotation([1, 1, 0], 75.0).T + 40.0
    assert FakeGate(rotated, plddt=90.0).check(st.sequence, st).passed


def test_gate_rejects_length_mismatch():
    st = syn.make("A" * 20, "helix")
    gate = FakeGate(st.coords("ca"))
    with pytest.raises(ValueError):
        gate.check("AAA", st)


def test_gate_rejects_wrong_prediction_shape():
    st = syn.make("A" * 20, "helix")
    gate = FakeGate(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        gate.check(st.sequence, st)


def test_null_gate_is_explicit_about_not_checking():
    """An unverified run must say so, not silently omit the field."""
    st = syn.make("A" * 20, "helix")
    result = NullGate().check(st.sequence, st)
    assert result.passed
    assert np.isnan(result.sc_rmsd)
    assert "no refold check" in result.reason


def test_esmfold_gate_reports_a_useful_error_without_transformers():
    gate = ESMFoldGate()
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformers"):
            gate.predict("AAAA")


# ------------------------------------------------- prediction-from-file gate

def test_predicted_structure_gate_accepts_a_structure_object():
    from proteus.validate import PredictedStructureGate
    st = syn.make("A" * 20, "helix")
    result = PredictedStructureGate(st, plddt=90.0).check(st.sequence, st)
    assert result.passed
    assert result.sc_rmsd == pytest.approx(0.0, abs=1e-8)


def test_predicted_structure_gate_reads_a_pdb_file(tmp_path):
    from proteus.structure import to_pdb
    from proteus.validate import PredictedStructureGate
    st = syn.make("A" * 20, "helix")
    path = tmp_path / "pred.pdb"
    to_pdb(st, str(path))
    result = PredictedStructureGate(str(path), plddt=95.0).check(st.sequence, st)
    assert result.passed


def test_predicted_structure_gate_is_rotation_invariant(tmp_path):
    """A correct fold saved in a different frame must still pass."""
    from proteus.structure import from_arrays, to_pdb
    from proteus.validate import PredictedStructureGate
    st = syn.make("A" * 22, "helix")
    rot = _rotation([1, 1, 0], 70.0)
    moved = from_arrays(st.sequence, ca=st.coords("ca") @ rot.T + 30.0,
                        n=st.coords("n") @ rot.T + 30.0,
                        c=st.coords("c") @ rot.T + 30.0)
    path = tmp_path / "rot.pdb"
    to_pdb(moved, str(path))
    assert PredictedStructureGate(str(path), plddt=90.0).check(st.sequence, st).passed


def test_predicted_structure_gate_fails_on_a_different_fold():
    from proteus.validate import PredictedStructureGate
    st = syn.make("A" * 20, "helix")
    wrong = syn.make("A" * 20, "strand")
    result = PredictedStructureGate(wrong, plddt=95.0).check(st.sequence, st)
    assert not result.passed


def test_plddt_read_from_bfactor_column(tmp_path):
    """AlphaFold and ESMFold write per-residue confidence into B-factors."""
    from proteus.structure import from_pdb, to_pdb
    from proteus.validate import PredictedStructureGate
    st = syn.make("A" * 20, "helix")
    path = tmp_path / "pred.pdb"
    to_pdb(st, str(path))
    text = path.read_text().replace("  1.00  0.00", "  1.00 42.00")
    path.write_text(text)
    loaded = from_pdb(str(path))
    assert loaded[1].bfactor == pytest.approx(42.0)
    # 42 is below the default pLDDT cutoff, so this must fail on confidence.
    result = PredictedStructureGate(str(path)).check(st.sequence, st)
    assert not result.passed
    assert "pLDDT" in result.reason
