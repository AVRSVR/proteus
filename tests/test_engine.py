"""Selection, scoring and the optimization loop.

Several of these correspond directly to defects in the original prototype:
the unreachable absolute-energy target, the unbounded strategy weights, the
pure hill climber, and the binding face that was never actually frozen.
"""

import random

import pytest

from proteus import DesignContext
from proteus import membrane as M
from proteus.engine import Engine, realize_sequence
from proteus.proposals import Proposal, resolve
from proteus.scoring import HeuristicScorer
from proteus.selection import UCB1, Thompson, make_policy

from . import _synthetic as syn

POOR_SEQUENCE = "AGSTAGSTKLAGSTGGVAST"


def poor_bundle(n_helices=4, per_helix=20, radius=7.0):
    return syn.make_bundle(n_helices, per_helix, radius=radius,
                           sequence=POOR_SEQUENCE * n_helices)


# ------------------------------------------------------------------- scoring

def test_score_is_size_independent():
    """Per-residue scoring is what makes a target reachable at any length.

    The prototype compared *total* Rosetta energy against a fixed -250 REU.
    Total energy scales with chain length, so that target was unreachable for
    a small protein and trivial for a large one.
    """
    scorer = HeuristicScorer()
    small = poor_bundle(n_helices=3, per_helix=20)
    large = poor_bundle(n_helices=6, per_helix=20)
    s_small = scorer.score(DesignContext(structure=small), small.sequence)
    s_large = scorer.score(DesignContext(structure=large), large.sequence)
    assert len(large) == 2 * len(small)
    # Totals differ with size; per-residue values stay comparable.
    assert abs(s_small.per_residue - s_large.per_residue) < 0.5 * abs(s_small.per_residue) + 0.05


def test_aggregation_term_penalises_exposed_hydrophobic_patch():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    scorer = HeuristicScorer()
    surface = [p for p in ctx.positions if ctx.layer(p) == "surface"]
    seq = list(st.sequence)
    for p in surface:
        seq[p - 1] = "K"
    polar = scorer.score(ctx, "".join(seq)).terms["aggregation"]
    for p in surface:
        seq[p - 1] = "I"
    greasy = scorer.score(ctx, "".join(seq)).terms["aggregation"]
    assert greasy > polar


def test_membrane_inverts_the_burial_term():
    """A lipid-facing hydrophobic must score better than a lipid-facing charge."""
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    good = "".join("LIVF"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "KEDS"[i % 4]
                   for i in range(len(b0)))
    st = syn.make_bundle(4, 26, radius=9.0, sequence=good)
    ctx = DesignContext(structure=st, membrane=M.estimate(st))
    scorer = HeuristicScorer()

    lipid_facing = [p for p in ctx.positions if ctx.is_lipid_facing(p)]
    assert lipid_facing, "test structure must expose lipid-facing positions"

    seq = list(good)
    for p in lipid_facing:
        seq[p - 1] = "L"
    hydrophobic = scorer.score(ctx, "".join(seq)).terms["burial"]
    for p in lipid_facing:
        seq[p - 1] = "E"
    charged = scorer.score(ctx, "".join(seq)).terms["burial"]
    assert hydrophobic < charged


def test_aromatic_surface_is_flagged_as_aggregation_prone():
    """Aggregation is not scored on a hydropathy scale.

    Kyte-Doolittle rates Trp and Tyr as hydrophilic because they are
    amphipathic, so scoring aggregation with it gave a tryptophan-covered
    surface a free pass -- precisely the degenerate corner a greedy optimizer
    finds. Aromatics must rank as aggregation-prone, and charged residues must
    act as gatekeepers.
    """
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    scorer = HeuristicScorer()
    surface = [p for p in ctx.positions if ctx.layer(p) == "surface"]

    def surface_agg(aa):
        seq = list(st.sequence)
        for p in surface:
            seq[p - 1] = aa
        return scorer.score(ctx, "".join(seq)).terms["aggregation"]

    for aromatic in "WYF":
        assert surface_agg(aromatic) > 1.0, f"{aromatic} surface not flagged"
    for gatekeeper in "KED":
        assert surface_agg(gatekeeper) == pytest.approx(0.0, abs=1e-9)


def test_net_charge_term_punishes_runaway_charge():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    scorer = HeuristicScorer()
    neutral = scorer.score(ctx, "A" * len(st)).terms["net_charge"]
    all_glu = scorer.score(ctx, "E" * len(st)).terms["net_charge"]
    assert all_glu > neutral
    assert neutral == pytest.approx(0.0, abs=1e-9)


def test_sequence_length_mismatch_rejected():
    st = poor_bundle()
    with pytest.raises(ValueError):
        HeuristicScorer().score(DesignContext(structure=st), "AAA")


# ----------------------------------------------------------------- selection

def test_untried_arms_are_selected_first():
    pol = UCB1(["a", "b", "c"], random.Random(0))
    pol.update(["a"], reward=10.0, success=True)
    chosen = pol.select(["a", "b", "c"], k=2)
    assert "a" not in chosen          # b and c have no evidence yet


def test_ucb_prefers_the_better_arm_once_explored():
    pol = UCB1(["good", "bad"], random.Random(0))
    for _ in range(20):
        pol.update(["good"], reward=1.0, success=True)
        pol.update(["bad"], reward=0.0, success=False)
    assert pol.select(["good", "bad"], k=1) == ["good"]


def test_leaderboard_is_ordered_by_mean_reward():
    pol = UCB1(["a", "b"], random.Random(0))
    pol.update(["a"], reward=5.0, success=True)
    pol.update(["b"], reward=1.0, success=True)
    assert pol.leaderboard()[0].name == "a"


def test_weights_cannot_run_away():
    """Rewards are averaged, not accumulated.

    The prototype added 0.5 per success with no upper bound, so an early
    winner's sampling weight grew without limit and exploration stopped.
    """
    pol = UCB1(["a"], random.Random(0))
    for _ in range(1000):
        pol.update(["a"], reward=1.0, success=True)
    assert pol.stats["a"].mean_reward == pytest.approx(1.0)


def test_selection_ignores_inapplicable_strategies():
    """A strategy that does not apply is never charged a failure."""
    pol = UCB1(["a", "b"], random.Random(0))
    assert pol.select(["a"], k=2) == ["a"]


def test_thompson_policy_runs():
    pol = make_policy("thompson", ["a", "b"], random.Random(0))
    assert isinstance(pol, Thompson)
    for _ in range(10):
        pol.update(pol.select(["a", "b"], k=1), reward=1.0, success=True)
    assert pol.total_pulls == 10


def test_unknown_policy_rejected():
    with pytest.raises(KeyError):
        make_policy("nope", ["a"])


# -------------------------------------------------------------------- engine

def test_frozen_positions_are_unchanged_in_the_output():
    """The guarantee the prototype claimed but did not implement."""
    st = poor_bundle()
    frozen = frozenset(range(1, 16))
    ctx = DesignContext(structure=st, frozen=frozen)
    result = Engine(ctx, seed=0).run(generations=25)
    for p in frozen:
        assert result.best_sequence[p - 1] == st.sequence[p - 1]


def test_run_terminates_on_its_budget():
    """No unreachable target, so the loop is always bounded."""
    ctx = DesignContext(structure=poor_bundle())
    result = Engine(ctx, seed=0).run(generations=12)
    assert len(result.trajectory) <= 12


def test_best_is_never_worse_than_start():
    ctx = DesignContext(structure=poor_bundle())
    result = Engine(ctx, seed=0).run(generations=25)
    assert result.best_score <= result.start_score
    assert result.improvement >= 0.0


def test_run_is_deterministic_given_a_seed():
    ctx = DesignContext(structure=poor_bundle())
    a = Engine(ctx, seed=7).run(generations=15)
    b = Engine(ctx, seed=7).run(generations=15)
    assert a.best_sequence == b.best_sequence
    assert a.best_score == pytest.approx(b.best_score)


def test_temperature_is_calibrated_from_observed_deltas():
    """A fixed temperature is either far too hot or far too cold."""
    ctx = DesignContext(structure=poor_bundle())
    eng = Engine(ctx, seed=0, calibration_moves=4)
    assert eng.t0 is None
    eng.run(generations=15)
    assert eng.t0 is not None and eng.t0 > 0
    assert eng.t1 < eng.t0


def test_acceptance_rate_is_not_degenerate():
    """Neither a random walk (accept everything) nor a pure hill climber."""
    ctx = DesignContext(structure=poor_bundle())
    result = Engine(ctx, seed=0).run(generations=40)
    rate = result.n_accepted / len(result.trajectory)
    assert 0.1 < rate < 0.95


def test_tabu_blocks_immediate_reversion():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    scorer = HeuristicScorer()
    res = resolve([Proposal(20, frozenset("CF"), "s", "test")])
    barred = realize_sequence(ctx, res, scorer, st.sequence, random.Random(0),
                              tabu={(20, "C"): 10}, generation=1)
    assert barred[19] != "C"


def test_tabu_never_empties_a_position():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    res = resolve([Proposal(20, frozenset("C"), "s", "only option")])
    out = realize_sequence(ctx, res, HeuristicScorer(), st.sequence,
                           random.Random(0), tabu={(20, "C"): 10}, generation=1)
    assert out[19] == "C"          # falls back rather than failing


def test_engine_records_strategy_attribution():
    ctx = DesignContext(structure=poor_bundle())
    result = Engine(ctx, seed=0).run(generations=10)
    assert all(m.strategies for m in result.trajectory)
    assert result.policy is not None
    assert result.policy.total_pulls > 0


def test_membrane_run_uses_membrane_strategies_only():
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    seq = "".join("LIVF"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "KEDS"[i % 4]
                  for i in range(len(b0)))
    st = syn.make_bundle(4, 26, radius=9.0, sequence=seq)
    ctx = DesignContext(structure=st, membrane=M.estimate(st))
    result = Engine(ctx, seed=0).run(generations=15)
    used = {s for m in result.trajectory for s in m.strategies}
    assert "core_packing" not in used
    assert "surface_depolarize" not in used


# --------------------------------------------------------------- provenance

def test_every_surviving_mutation_is_attributed():
    """A mutation with no proposing strategy would mean the loop is not
    actually driven by the strategy library."""
    st = poor_bundle()
    ctx = DesignContext(structure=st, frozen=frozenset(range(1, 9)))
    # Mutation cost off, so the run definitely produces changes to attribute.
    result = Engine(ctx, seed=0, min_gain_per_mutation=0.0).run(generations=25)
    prov = result.provenance()
    assert prov, "run produced no changes at all"
    unattributed = [p for p, (_o, _n, why) in prov.items() if not why]
    assert not unattributed, f"unattributed positions: {unattributed}"


def test_provenance_describes_the_returned_sequence():
    """Provenance must describe ``best_sequence`` -- the thing handed back.

    Reporting the accepted walk instead listed changes absent from the result,
    which produced output showing nine changed positions beside a summary
    claiming 100% identity retained.
    """
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    result = Engine(ctx, seed=0, min_gain_per_mutation=0.0).run(generations=25)

    prov = result.provenance()
    for pos, (old, new, _why) in prov.items():
        assert old == st.sequence[pos - 1]
        assert new == result.best_sequence[pos - 1]

    truly_changed = {i + 1 for i, (a, b) in
                     enumerate(zip(st.sequence, result.best_sequence)) if a != b}
    assert set(prov) == truly_changed
    assert len(prov) == result.n_mutations


def test_provenance_excludes_positions_changed_and_reverted():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    result = Engine(ctx, seed=1).run(generations=30)
    for pos, (old, new, _why) in result.provenance().items():
        assert old != new


def test_frozen_positions_never_appear_in_provenance():
    st = poor_bundle()
    frozen = frozenset(range(1, 16))
    ctx = DesignContext(structure=st, frozen=frozen)
    result = Engine(ctx, seed=0).run(generations=25)
    assert not (set(result.provenance()) & frozen)


def test_credit_only_counts_surviving_mutations():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    result = Engine(ctx, seed=0).run(generations=25)
    credit = result.credit()
    if credit:
        assert all(v > 0 for v in credit.values())
        assert set(credit) <= {s for m in result.trajectory for s in m.strategies}


# --------------------------------------------------------- mutation budget

def test_mutation_budget_is_never_exceeded():
    """A tool that rewrites half the sequence has redesigned the protein."""
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    result = Engine(ctx, seed=0, mutation_budget=0.10, min_gain_per_mutation=0.0).run(
        generations=40)
    assert result.n_mutations <= result.max_mutations
    assert result.n_mutations <= 0.10 * len(st) + 1


def test_tighter_budget_yields_fewer_mutations():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    loose = Engine(ctx, seed=0, mutation_budget=0.30, min_gain_per_mutation=0.0).run(
        generations=40)
    tight = Engine(ctx, seed=0, mutation_budget=0.05, min_gain_per_mutation=0.0).run(
        generations=40)
    assert tight.n_mutations <= loose.n_mutations


def test_min_gain_threshold_suppresses_neutral_edits():
    """A mutation must earn its place rather than merely not hurt."""
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    free = Engine(ctx, seed=0, min_gain_per_mutation=0.0).run(generations=40)
    priced = Engine(ctx, seed=0, min_gain_per_mutation=0.01).run(generations=40)
    assert priced.n_mutations <= free.n_mutations


def test_min_gain_threshold_is_size_independent():
    """The bar must not get cheaper because the protein is larger.

    An earlier form divided the price by chain length, which made the same
    setting six times cheaper per mutation on a 400-residue protein than on a
    66-residue one, so large proteins accumulated dozens of marginal edits.
    """
    small = poor_bundle(n_helices=3, per_helix=20)
    large = poor_bundle(n_helices=6, per_helix=20)
    threshold = 0.002
    a = Engine(DesignContext(structure=small), seed=0,
               min_gain_per_mutation=threshold)
    b = Engine(DesignContext(structure=large), seed=0,
               min_gain_per_mutation=threshold)
    # The penalty for one mutation is identical regardless of chain length.
    assert a._mutation_penalty(1) == pytest.approx(b._mutation_penalty(1))
    assert a._mutation_penalty(5) == pytest.approx(threshold * 5)


def test_identity_and_mutation_count_agree():
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    r = Engine(ctx, seed=0, min_gain_per_mutation=0.0).run(generations=25)
    assert r.identity == pytest.approx(1.0 - r.n_mutations / len(st))


def test_frozen_positions_are_excluded_from_the_budget():
    st = poor_bundle()
    frozen = frozenset(range(1, 41))
    ctx = DesignContext(structure=st, frozen=frozen)
    r = Engine(ctx, seed=0, mutation_budget=0.5, min_gain_per_mutation=0.0).run(generations=20)
    assert r.max_mutations <= 0.5 * len(ctx.designable) + 1
    assert not (set(r.provenance()) & frozen)


def test_rationale_names_the_strategy_that_chose_the_residue():
    """A proline must not be credited to a mechanism that never proposes one."""
    from proteus import REGISTRY
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    r = Engine(ctx, seed=0, min_gain_per_mutation=0.0).run(generations=30)
    for pos, (_old, new, why) in r.provenance().items():
        if not why or ":" not in why:
            continue
        name = why.split(":", 1)[0]
        strat = REGISTRY.get(name)
        proposals = strat.run(DesignContext(structure=st), random.Random(0), 999)
        allowed = {p.allowed for p in proposals if p.resi == pos}
        if allowed:
            assert any(new in a for a in allowed), (
                f"{name} credited for {new} at {pos} but never proposes it")


# ------------------------------------------------------------ fitted scorer

def test_fitted_scorer_uses_the_same_terms_as_the_heuristic():
    """The fit reweights the existing terms; it does not invent new ones."""
    from proteus.scoring import FittedScorer
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    h = HeuristicScorer().score(ctx, st.sequence)
    f = FittedScorer().score(ctx, st.sequence)
    assert set(f.terms) == set(h.terms)
    assert f.terms == h.terms          # term values identical, only the sum differs


def test_fitted_scorer_differs_from_hand_tuned_total():
    """If the fitted weights reproduced the hand-tuned sum, the fit did nothing."""
    from proteus.scoring import FittedScorer
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    assert (FittedScorer().score(ctx, st.sequence).total
            != pytest.approx(HeuristicScorer().score(ctx, st.sequence).total))


def test_fitted_weights_cover_every_term():
    """A term with no fitted weight would be silently dropped from the total."""
    from proteus.scoring import FITTED_DDG_WEIGHTS
    st = poor_bundle()
    terms = HeuristicScorer().score(DesignContext(structure=st), st.sequence).terms
    missing = set(terms) - set(FITTED_DDG_WEIGHTS)
    assert not missing, f"terms with no fitted weight: {sorted(missing)}"


# ------------------------------------------------------------- ProteinMPNN

def test_mpnn_module_imports_without_torch():
    """The core package must stay usable on a machine with no ML stack."""
    from proteus import mpnn
    assert isinstance(mpnn.available(), bool)


def test_mpnn_require_explains_installation():
    from proteus import mpnn
    if not mpnn.available():
        with pytest.raises(ImportError, match="proteinmpnn"):
            mpnn.require()


def test_mpnn_alphabet_matches_the_package():
    """An alphabet off by one silently scores the wrong residue."""
    from proteus import mpnn
    if not mpnn.available():
        pytest.skip("proteinmpnn not installed")
    import re, inspect
    from proteinmpnn import protein_mpnn_utils as U
    found = re.search(r"alphabet\s*=\s*['\"]([A-Z]+)['\"]",
                      inspect.getsource(U.tied_featurize))
    assert found, "could not locate the alphabet in the package"
    assert mpnn.ALPHABET == found.group(1)


def test_mpnn_sequence_scorer_rejects_synthetic_structures():
    """It reads a file; a structure built from arrays has none."""
    from proteus.mpnn import MPNNSequenceScorer
    st = poor_bundle()
    ctx = DesignContext(structure=st)
    with pytest.raises(ValueError, match="source path"):
        MPNNSequenceScorer().score(ctx, st.sequence)
