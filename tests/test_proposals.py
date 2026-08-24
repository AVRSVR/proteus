"""Conflict resolution and resfile generation.

These lock in the behaviour that replaced the prototype's shared-resfile
approach, where the last strategy to write a line silently won.
"""

import pytest

from proteus.proposals import (ALL, AROMATIC, CHARGED, HYDROPHOBIC, NEGATIVE,
                               Proposal, resolve, to_resfile)

from . import _synthetic as syn


def test_compatible_proposals_intersect():
    r = resolve([
        Proposal(5, HYDROPHOBIC, "core_packing", "buried"),
        Proposal(5, AROMATIC, "cavity_fill", "cavity"),
    ])
    assert r.allowed[5] == AROMATIC & HYDROPHOBIC
    assert not r.conflicts


def test_incompatible_proposals_are_recorded_not_silent():
    r = resolve([
        Proposal(9, CHARGED, "surface_depolarize", "exposed", weight=1.0),
        Proposal(9, HYDROPHOBIC, "core_packing", "buried", weight=2.0),
    ])
    assert len(r.conflicts) == 1
    assert r.allowed[9] == HYDROPHOBIC          # higher weight wins
    assert "core_packing" in r.conflicts[0].resolved_by


def test_frozen_positions_reject_proposals():
    """Freezing is a hard constraint, not an overridable default."""
    r = resolve([Proposal(3, HYDROPHOBIC, "core_packing", "buried")], frozen={3})
    assert 3 not in r.allowed
    assert r.blocked_frozen[3] == ("core_packing",)


def test_attribution_is_retained():
    r = resolve([
        Proposal(7, HYDROPHOBIC, "core_packing", "a"),
        Proposal(7, AROMATIC, "cavity_fill", "b"),
    ])
    assert set(r.attribution[7]) == {"core_packing", "cavity_fill"}


def test_resfile_defaults_to_natro():
    """Unmentioned positions must be held fixed, not thrown open to design.

    The prototype defaulted to ALLAA, which meant every position no strategy
    named -- including its entire nominally-frozen binding face -- was
    redesigned on every generation.
    """
    st = syn.make("A" * 10, "helix")
    r = resolve([Proposal(4, AROMATIC, "cavity_fill", "cavity")])
    text = to_resfile(r, st)
    assert text.splitlines()[0] == "NATRO"
    assert "4 A PIKAA FWY" in text
    # No line for any other position.
    assert len([ln for ln in text.splitlines() if "PIKAA" in ln]) == 1


def test_resfile_uses_pdb_numbering():
    st = syn.make("A" * 5, "helix")
    r = resolve([Proposal(2, NEGATIVE, "salt_bridge", "pair")])
    assert f"{st[2].pdb_number} {st[2].chain} PIKAA DE" in to_resfile(r, st)


def test_allaa_rendered_when_unrestricted():
    st = syn.make("A" * 5, "helix")
    r = resolve([Proposal(2, ALL, "anything", "unrestricted")])
    assert "2 A ALLAA" in to_resfile(r, st)


def test_empty_proposal_rejected():
    with pytest.raises(ValueError):
        Proposal(1, frozenset(), "bad", "empty")


def test_non_canonical_residue_rejected():
    with pytest.raises(ValueError):
        Proposal(1, frozenset("BZX"), "bad", "non-canonical")
