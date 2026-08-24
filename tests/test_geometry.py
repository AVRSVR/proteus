"""Geometry tests against structures whose answer is known by construction."""

import numpy as np
import pytest

from proteus import geometry as G

from . import _synthetic as syn


def test_helix_dihedrals_match_construction():
    h = syn.make("A" * 20, "helix")
    phi, psi = G.dihedrals(h)
    # Interior residues should reproduce the phi/psi they were built from.
    assert phi[10] == pytest.approx(-57.0, abs=1.0)
    assert psi[10] == pytest.approx(-47.0, abs=1.0)


def test_strand_dihedrals_match_construction():
    s = syn.make("A" * 20, "strand")
    phi, psi = G.dihedrals(s)
    assert phi[10] == pytest.approx(-139.0, abs=1.0)
    assert psi[10] == pytest.approx(135.0, abs=1.0)


def test_termini_dihedrals_are_undefined():
    h = syn.make("A" * 10, "helix")
    phi, psi = G.dihedrals(h)
    assert np.isnan(phi[0])       # no preceding residue
    assert np.isnan(psi[-1])      # no following residue


def test_secondary_structure_assignment():
    assert "H" * 15 in G.secondary_structure(syn.make("A" * 20, "helix"))
    assert "E" * 15 in G.secondary_structure(syn.make("A" * 20, "strand"))


def test_short_runs_are_filtered_to_loop():
    """An isolated helical residue is noise, not a helix."""
    motif = [syn.PHI_PSI["strand"]] * 10
    motif[5] = syn.PHI_PSI["helix"]
    ss = G.secondary_structure(syn.make("A" * 10, motif), min_run=3)
    assert ss[5] != "H"


def test_bundle_interior_is_more_buried_than_exterior():
    """The burial measure must rank interior above exterior on a real fold."""
    b = syn.make_bundle(4, 20, radius=7.0)
    counts = G.sidechain_neighbors(b)
    ca = b.coords("ca")
    axis_centre = np.array([ca[:, 0].mean(), ca[:, 1].mean()])
    radial = np.linalg.norm(ca[:, :2] - axis_centre, axis=1)
    # Residues nearer the bundle axis should score as more buried.
    assert np.corrcoef(counts, radial)[0, 1] < -0.2


def test_isolated_helix_has_no_core():
    """A single helix has nothing to bury against."""
    h = syn.make("A" * 25, "helix")
    assert "core" not in G.layers(h)


def test_virtual_cb_is_finite_for_glycine():
    h = syn.make("G" * 10, "helix")
    for r in h:
        assert np.all(np.isfinite(r.cb))
        assert not r.has_real_cb


def test_chain_breaks_detected():
    h = syn.make("A" * 20, "helix")
    # Displace the second half far away to create a break.
    from proteus.structure import from_arrays
    ca, n, c = h.coords("ca"), h.coords("n"), h.coords("c")
    for arr in (ca, n, c):
        arr[10:] += 50.0
    broken = from_arrays("A" * 20, ca=ca, n=n, c=c)
    assert 10 in broken.chain_breaks()
