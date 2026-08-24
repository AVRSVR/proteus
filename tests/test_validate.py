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


# -------------------------------------------------------------------- rosetta

def test_rosetta_backend_imports_without_pyrosetta():
    """The module must be importable on a machine with no Rosetta licence."""
    from proteus.backends import rosetta
    assert isinstance(rosetta.available(), bool)


def test_rosetta_require_explains_installation():
    from proteus.backends import rosetta
    if not rosetta.available():
        with pytest.raises(ImportError, match="pyrosetta-installer"):
            rosetta.require()


def test_rosetta_selects_membrane_energy_function():
    """Scoring a membrane protein with ref2015 is the wrong function."""
    from proteus.backends.rosetta import RosettaBackend
    assert RosettaBackend(membrane=False).weights == "ref2015"
    assert RosettaBackend(membrane=True).weights == "franklin2019"


def test_resolution_from_sequence_pins_only_changed_positions():
    from proteus.backends.rosetta import build_resolution_from_sequence
    from proteus import DesignContext

    st = syn.make_bundle(3, 20, radius=7.0)
    ctx = DesignContext(structure=st)
    seq = list(st.sequence)
    seq[5] = "W"
    seq[9] = "E"
    res = build_resolution_from_sequence(ctx, "".join(seq))
    assert set(res.allowed) == {6, 10}
    assert res.allowed[6] == frozenset("W")


def test_resolution_from_sequence_respects_frozen():
    from proteus.backends.rosetta import build_resolution_from_sequence
    from proteus import DesignContext

    st = syn.make_bundle(3, 20, radius=7.0)
    ctx = DesignContext(structure=st, frozen=frozenset({6}))
    seq = list(st.sequence)
    seq[5] = "W"
    res = build_resolution_from_sequence(ctx, "".join(seq))
    assert 6 not in res.allowed


def test_resfile_from_rosetta_resolution_defaults_to_natro():
    from proteus.backends.rosetta import build_resolution_from_sequence
    from proteus.proposals import to_resfile
    from proteus import DesignContext

    st = syn.make_bundle(3, 20, radius=7.0)
    ctx = DesignContext(structure=st)
    seq = list(st.sequence)
    seq[5] = "W"
    text = to_resfile(build_resolution_from_sequence(ctx, "".join(seq)), st)
    assert text.splitlines()[0] == "NATRO"
    assert len([l for l in text.splitlines() if "PIKAA" in l]) == 1
