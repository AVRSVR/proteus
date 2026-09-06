"""The nature-inspired strategy families added after the first audit.

Each family is checked for the two failure modes the earlier audit found: a
gate so permissive that its diagnosis carries no information, and a proposal
that contradicts the mechanism it claims to implement.
"""

import random

import pytest

from proteus import DesignContext, REGISTRY
from proteus import membrane as M
from proteus.structure import from_arrays

from . import _synthetic as syn

RNG = random.Random(0)


def ctx_with(sequence, n_helices=4, per_helix=20, radius=7.0, frozen=frozenset()):
    """A bundle carrying a chosen sequence, padded or trimmed to fit."""
    b = syn.make_bundle(n_helices, per_helix, radius=radius)
    n = len(b)
    seq = (sequence * (n // len(sequence) + 1))[:n]
    b = syn.make_bundle(n_helices, per_helix, radius=radius, sequence=seq)
    return DesignContext(structure=b, frozen=frozen)


def membrane_ctx():
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    seq = "".join("LIVF"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "KEDS"[i % 4]
                  for i in range(len(b0)))
    b = syn.make_bundle(4, 26, radius=9.0, sequence=seq)
    return DesignContext(structure=b, membrane=M.estimate(b))


# --------------------------------------------------------------- liabilities

def test_deamidation_motif_detects_ng():
    """Asn-Gly is the fastest deamidation motif and must be found."""
    ctx = ctx_with("NGAAEKAAEKAAEKAAEKAA")
    sites = REGISTRY.get("deamidation_motif")._sites(ctx)
    seq = ctx.structure.sequence
    assert sites
    for p in sites:
        assert seq[p - 1] == "N"
        assert seq[p:p + 1] in "GSNTA"


def test_deamidation_ignores_asn_without_a_partner():
    """Asn followed by a bulky residue is not a fast deamidation site."""
    ctx = ctx_with("NWAAEKAAEKAAEKAAEKAA")
    for p in REGISTRY.get("deamidation_motif")._sites(ctx):
        assert ctx.structure.sequence[p:p + 1] != "W"


def test_isomerisation_motif_detects_dg():
    ctx = ctx_with("DGAAEKAAEKAAEKAAEKAA")
    sites = REGISTRY.get("isomerisation_motif")._sites(ctx)
    assert sites
    assert all(ctx.structure.sequence[p - 1] == "D" for p in sites)


def test_glycosylation_sequon_requires_the_full_motif():
    """N-X-S/T, and X must not be proline."""
    strat = REGISTRY.get("glycosylation_sequon")
    assert strat._sites(ctx_with("NASAEKAAEKAAEKAAEKAA"))       # N-A-S: a sequon
    assert not strat._sites(ctx_with("NPSAEKAAEKAAEKAAEKAA"))   # N-P-S: not one
    assert not strat._sites(ctx_with("NAAAEKAAEKAAEKAAEKAA"))   # no Ser/Thr


def test_liability_strategies_never_propose_the_residue_they_remove():
    """Replacing Asn with Asn, or Cys with Cys, would be a no-op."""
    cases = {
        "deamidation_motif": ("NGAAEKAAEKAAEKAAEKAA", "N"),
        "isomerisation_motif": ("DGAAEKAAEKAAEKAAEKAA", "D"),
        "free_cysteine": ("CAAAEKAAEKAAEKAAEKAA", "C"),
        "methionine_oxidation": ("MAAAEKAAEKAAEKAAEKAA", "M"),
    }
    for name, (seq, removed) in cases.items():
        ctx = ctx_with(seq)
        for prop in REGISTRY.get(name).run(ctx, RNG, max_positions=99):
            assert removed not in prop.allowed, f"{name} re-proposes {removed}"


def test_free_cysteine_spares_a_bonded_pair():
    """A cysteine pair at disulfide distance is a bond, not a liability."""
    ctx = ctx_with("AAAAEKAAEKAAEKAAEKAA")
    disulfide = REGISTRY.get("disulfide")
    pairs = disulfide.candidate_pairs(ctx)
    if not pairs:
        pytest.skip("test geometry has no disulfide-compatible pair")
    i, j = pairs[0]
    seq = list(ctx.structure.sequence)
    seq[i - 1] = seq[j - 1] = "C"
    st = from_arrays("".join(seq), ca=ctx.structure.coords("ca"),
                     cb=ctx.structure.coords("cb"), n=ctx.structure.coords("n"),
                     c=ctx.structure.coords("c"))
    after = DesignContext(structure=st)
    assert i not in REGISTRY.get("free_cysteine")._unpaired(after)


# --------------------------------------------------------------- thermophile

def test_arginine_preference_only_targets_surface_lysine():
    ctx = ctx_with("KAAAEKAAEKAAEKAAEKAA")
    strat = REGISTRY.get("arginine_preference")
    for p in strat.diagnose(ctx):
        assert ctx.aa(p) == "K"
        assert ctx.layer(p) == "surface"
    for prop in strat.run(ctx, RNG, max_positions=99):
        assert prop.allowed == frozenset("R")


def test_thermolabile_amide_targets_asn_and_gln_only():
    ctx = ctx_with("NQAAEKAAEKAAEKAAEKAA")
    for p in REGISTRY.get("thermolabile_amide").diagnose(ctx):
        assert ctx.aa(p) in "NQ"
        assert ctx.layer(p) == "surface"


def test_helix_dipole_places_the_correct_sign_at_each_end():
    """Negative near the N-terminus, positive near the C-terminus."""
    ctx = ctx_with("AAAAAAAAAAAAAAAAAAAA")
    strat = REGISTRY.get("helix_dipole")
    sites = strat._sites(ctx)
    if not sites:
        pytest.skip("no helix long enough in the test structure")
    for prop in strat.run(ctx, RNG, max_positions=99):
        end = sites.get(prop.resi)
        if end == "N":
            assert prop.allowed == frozenset("DE")
        elif end == "C":
            assert prop.allowed == frozenset("KR")


def test_salt_bridge_network_requires_existing_charges_nearby():
    """A network seed must actually sit inside a cluster of charges."""
    ctx = ctx_with("AAAAAAAAAAAAAAAAAAAA")     # no charges at all
    assert REGISTRY.get("salt_bridge_network").diagnose(ctx) == []


def test_capping_box_targets_n3_of_a_helix():
    ctx = ctx_with("AAAAAAAAAAAAAAAAAAAA")
    strat = REGISTRY.get("capping_box")
    starts = {s for s, e in ctx.ss_segments("H") if e - s + 1 >= 7}
    for p in strat.diagnose(ctx):
        assert (p - 2) in starts


# ------------------------------------------------------------------- packing

def test_cation_pi_is_capped_and_selective():
    """The gate is approximate, so it must under-claim rather than flood."""
    ctx = ctx_with("FAKAAEKAAEKAAEKAAEKA")
    strat = REGISTRY.get("cation_pi")
    sites = strat.diagnose(ctx)
    assert len(sites) <= strat.MAX_SITES
    for p in sites:
        assert ctx.aa(p) not in "KR"


def test_aromatic_cluster_pairs_with_an_existing_aromatic():
    ctx = ctx_with("FAAAEKAAEKAAEKAAEKAA")
    strat = REGISTRY.get("aromatic_cluster")
    from proteus.strategies.packing import STACK_MAX, STACK_MIN
    for p, q in strat.candidate_pairs(ctx):
        assert ctx.aa(q) in "FWY"
        assert STACK_MIN <= ctx.cb_dist[p - 1, q - 1] <= STACK_MAX


def test_aromatic_cluster_never_targets_the_surface():
    """A ring on the surface is an aggregation liability, not a cluster."""
    ctx = ctx_with("FAAAEKAAEKAAEKAAEKAA")
    for p, _q in REGISTRY.get("aromatic_cluster").candidate_pairs(ctx):
        assert ctx.layer(p) != "surface"


def test_buried_unsatisfied_polar_requires_burial_and_no_partner():
    ctx = ctx_with("AAAAEKAAEKAAEKAAEKAA")
    strat = REGISTRY.get("buried_unsatisfied_polar")
    from proteus.strategies.packing import BURIED_POLAR, HBOND_MAX
    for p in strat.diagnose(ctx):
        assert ctx.layer(p) == "core"
        assert ctx.aa(p) in BURIED_POLAR
        row = ctx.cb_dist[p - 1]
        partners = [q for q in ctx.positions
                    if q != p and row[q - 1] <= HBOND_MAX
                    and ctx.aa(q) in BURIED_POLAR]
        assert not partners


def test_beta_edge_protection_proposes_only_blockers():
    """Charges and proline are what stop a strand recruiting a partner."""
    strand = syn.make("VTVTVTVTVTVTVTVTVTVT", "strand")
    ctx = DesignContext(structure=strand)
    for prop in REGISTRY.get("beta_edge_protection").run(ctx, RNG, max_positions=99):
        assert prop.allowed <= frozenset("KRDE")


# ------------------------------------------------------------------ membrane

def test_glycine_zipper_only_inside_the_bilayer():
    ctx = membrane_ctx()
    strat = REGISTRY.get("glycine_zipper")
    for p in strat.diagnose(ctx):
        assert ctx.is_membrane_buried(p)
        assert ctx.ss_at(p) == "H"


def test_membrane_only_strategies_absent_from_soluble_context():
    names = [s.name for s in REGISTRY.for_context(ctx_with("AAAAEKAAEKAAEKAAEKAA"))]
    for name in ("glycine_zipper", "terminal_anchor"):
        assert name not in names


# ----------------------------------------------------------------- invariants

def test_every_registered_strategy_has_a_mechanism_and_valid_environments():
    from proteus.strategies.base import MEMBRANE, SOLUBLE
    for s in REGISTRY.all():
        assert len(s.mechanism.strip()) > 40, f"{s.name} lacks a real description"
        assert s.applies_to <= {SOLUBLE, MEMBRANE}
        assert s.applies_to, f"{s.name} applies to nothing"


def test_no_strategy_touches_a_frozen_position():
    """The central guarantee, re-checked across the whole expanded library."""
    frozen = frozenset(range(1, 30))
    for ctx in (ctx_with("NGDGCMKQSTFAAEKAAEKA", frozen=frozen), membrane_ctx()):
        if not ctx.frozen:
            ctx = DesignContext(structure=ctx.structure, membrane=ctx.membrane,
                                frozen=frozen)
        for strat in REGISTRY.for_context(ctx):
            for prop in strat.run(ctx, RNG, max_positions=99):
                assert prop.resi not in frozen, f"{strat.name} touched a frozen residue"


def test_no_strategy_diagnoses_more_than_two_thirds_of_a_protein():
    """A gate that fires almost everywhere carries no information."""
    ctx = ctx_with("NGDGCMKQSTFAAEKAAEKA")
    for strat in REGISTRY.for_context(ctx):
        n = len(strat.diagnose(ctx))
        assert n <= 0.67 * len(ctx), f"{strat.name} diagnosed {n} of {len(ctx)}"


def test_packed_aromatics_are_not_stripped_for_helix_propensity():
    """A buried ring is paying for its propensity with packing.

    Chou-Fasman rates tyrosine at 0.69, so buried tyrosines look like free
    stability to recover. On a three-helix design replacing the five packed
    ones collapsed the predicted fold from 1.2 A to 27 A.
    """
    ctx = ctx_with("YAAAEKAAEKYAAEKAAEKA")
    strat = REGISTRY.get("helix_propensity")
    for p in strat.diagnose(ctx):
        if ctx.aa(p) in "FWY":
            assert ctx.layer(p) == "surface", (
                f"{ctx.aa(p)}{p} is {ctx.layer(p)} and should be left alone")


def test_exposed_aromatics_are_still_fair_game():
    """On the surface there is nothing to pack against, so the rule applies."""
    from proteus.strategies.soluble import HelixPropensity
    ctx = ctx_with("YAAAEKAAEKYAAEKAAEKA")
    exposed = [p for p in ctx.positions
               if ctx.aa(p) in "FWY" and ctx.layer(p) == "surface"]
    for p in exposed:
        assert not HelixPropensity._earns_its_place(ctx, p)
