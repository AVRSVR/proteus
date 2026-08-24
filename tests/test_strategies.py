"""Strategy behaviour, focused on the geometric gates.

Each test here corresponds to a way the original prototype produced
chemically meaningless output: ungated disulfides, periodic proline, glycines
mutated regardless of backbone conformation, and soluble rules applied inside
a bilayer.
"""

import random

import numpy as np
import pytest

from proteus import DesignContext, REGISTRY
from proteus import membrane as M
from proteus.strategies.base import MEMBRANE, SOLUBLE

from . import _synthetic as syn

RNG = random.Random(0)


def soluble_ctx(sequence=None, n_helices=4, per_helix=20, radius=7.0, frozen=frozenset()):
    b = syn.make_bundle(n_helices, per_helix, radius=radius, sequence=sequence)
    return DesignContext(structure=b, frozen=frozen)


def membrane_ctx(frozen=frozenset()):
    """TM bundle with a correctly hydrophobic belt, so the estimator can work."""
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    seq = "".join("LIVF"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "KEDS"[i % 4]
                  for i in range(len(b0)))
    b = syn.make_bundle(4, 26, radius=9.0, sequence=seq)
    return DesignContext(structure=b, membrane=M.estimate(b), frozen=frozen)


# --------------------------------------------------------------- environment

def test_membrane_strategies_excluded_from_soluble_context():
    names = [s.name for s in REGISTRY.for_context(soluble_ctx())]
    assert "lipid_facing_hydrophobic" not in names
    assert "aromatic_belt" not in names


def test_soluble_only_strategies_excluded_from_membrane_context():
    names = [s.name for s in REGISTRY.for_context(membrane_ctx())]
    assert "surface_depolarize" not in names
    assert "core_packing" not in names


def test_every_strategy_declares_a_mechanism():
    for s in REGISTRY.all():
        assert s.mechanism.strip(), f"{s.name} has no mechanism description"
        assert s.applies_to <= {SOLUBLE, MEMBRANE}


# ------------------------------------------------------------------ geometry

def test_disulfide_requires_real_geometry():
    """A single extended helix has no CB pair at disulfide distance."""
    st = syn.make("A" * 30, "helix")
    ctx = DesignContext(structure=st)
    assert REGISTRY.get("disulfide").diagnose(ctx) == []


def test_disulfide_pairs_satisfy_distance_criteria():
    ctx = soluble_ctx()
    strat = REGISTRY.get("disulfide")
    for i, j in strat._pairs(ctx):
        assert 3.0 <= ctx.cb_dist[i - 1, j - 1] <= 4.5
        assert 4.0 <= ctx.ca_dist[i - 1, j - 1] <= 6.5
        assert abs(i - j) >= 4


def test_disulfide_proposes_cysteine_in_pairs():
    ctx = soluble_ctx()
    props = REGISTRY.get("disulfide").run(ctx, RNG)
    if props:                                   # only if geometry allows one
        assert len(props) == 2
        assert all(p.allowed == frozenset("C") for p in props)


def test_proline_only_where_phi_permits():
    """Proline is gated on backbone conformation, not on sequence position."""
    from proteus.geometry import dihedrals
    ctx = soluble_ctx()
    phi, _ = dihedrals(ctx.structure)
    for p in REGISTRY.get("loop_rigidify")._proline_sites(ctx):
        assert -90.0 <= phi[p - 1] <= -40.0
        assert ctx.ss_at(p) == "L"


def test_proline_never_proposed_inside_a_helix():
    ctx = soluble_ctx()
    props = REGISTRY.get("loop_rigidify").run(ctx, RNG)
    for p in props:
        if p.allowed == frozenset("P"):
            assert ctx.ss_at(p.resi) == "L"


def test_positive_phi_glycine_is_protected():
    """Glycine at positive phi is conformationally required and must survive."""
    from proteus.geometry import dihedrals
    motif = [syn.PHI_PSI["helix"]] * 12
    motif[6] = (60.0, 40.0)                     # left-handed: glycine-only
    st = syn.make("AAAAAAGAAAAA", motif)
    ctx = DesignContext(structure=st)
    phi, _ = dihedrals(st)
    assert phi[6] > 0                           # the construction held
    assert 7 not in REGISTRY.get("loop_rigidify")._glycine_sites(ctx)


def test_salt_bridge_pairs_are_within_reach():
    ctx = soluble_ctx()
    for i, j in REGISTRY.get("salt_bridge")._pairs(ctx):
        assert 4.0 <= ctx.cb_dist[i - 1, j - 1] <= 8.0
        assert abs(i - j) >= 3


def test_salt_bridge_proposes_complementary_charges():
    ctx = soluble_ctx()
    props = REGISTRY.get("salt_bridge").run(ctx, RNG)
    if props:
        charges = {frozenset("DE"), frozenset("KR")}
        assert all(p.allowed in charges for p in props)
        assert len(props) % 2 == 0              # always emitted in pairs


# ------------------------------------------------------------------ membrane

def test_lipid_facing_positions_get_hydrophobic_proposals():
    ctx = membrane_ctx()
    strat = REGISTRY.get("lipid_facing_hydrophobic")
    for p in strat.run(ctx, RNG, max_positions=20):
        assert ctx.is_lipid_facing(p.resi)
        assert p.allowed <= frozenset("AVLIMF")


def test_burial_and_depth_are_independent_axes():
    """If these collapsed into one signal the membrane path would be a flag."""
    ctx = membrane_ctx()
    corr = np.corrcoef(ctx.neighbors, np.abs(ctx.depths))[0, 1]
    assert abs(corr) < 0.5


def test_aromatic_belt_targets_the_interface_only():
    ctx = membrane_ctx()
    for p in REGISTRY.get("aromatic_belt").run(ctx, RNG, max_positions=20):
        assert ctx.zone(p.resi) == "interface"
        assert p.allowed == frozenset("WY")


def test_membrane_normal_recovered_on_well_formed_bundle():
    ctx = membrane_ctx()
    assert abs(float(np.dot(ctx.membrane.normal, [0, 0, 1]))) > 0.9


def test_belt_estimator_needs_a_plausible_sequence():
    """Honest limitation: a badly designed sequence hides the belt.

    The estimator locates the bilayer from exposed hydrophobicity. If the
    design under repair has that backwards -- which is exactly the case
    Proteus exists to fix -- the estimate is unreliable and the membrane
    should be supplied explicitly instead.
    """
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    inverted = "".join("KEDS"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "LIVF"[i % 4]
                       for i in range(len(b0)))
    b = syn.make_bundle(4, 26, radius=9.0, sequence=inverted)
    mm = M.estimate(b)
    assert abs(float(np.dot(mm.normal, [0, 0, 1]))) < 0.99


def test_explicit_membrane_overrides_estimation():
    b = syn.make_bundle(4, 26, radius=9.0)
    explicit = M.MembraneModel(center=[0, 0, 0], normal=[0, 0, 1], source="given")
    ctx = DesignContext(structure=b, membrane=explicit)
    assert ctx.membrane.source == "given"
    assert np.allclose(ctx.membrane.normal, [0, 0, 1])


# --------------------------------------------------------------- frozen face

def test_no_strategy_ever_proposes_at_a_frozen_position():
    """The central guarantee: frozen means frozen, for every strategy."""
    frozen = frozenset(range(1, 15))
    for ctx in (soluble_ctx(frozen=frozen), membrane_ctx(frozen=frozen)):
        for strat in REGISTRY.for_context(ctx):
            for prop in strat.run(ctx, RNG, max_positions=25):
                assert prop.resi not in frozen, f"{strat.name} touched a frozen residue"
