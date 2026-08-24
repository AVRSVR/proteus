"""Fingerprints, the cross-protein knowledge base, and prior-guided selection.

These cover the part of Proteus that is supposed to generalise: a leaderboard
is only worth keeping if what it learns on one protein is usable on the next.
"""

import random

import pytest

from proteus import DesignContext, KnowledgeBase
from proteus import fingerprint as fp
from proteus import membrane as M
from proteus.engine import Engine
from proteus.knowledge import SCHEMA_VERSION, Observation, Prior
from proteus.selection import PriorGuidedUCB, UCB1, make_policy

from . import _synthetic as syn

POOR = "AGSTAGSTKLAGSTGGVAST"


def helical(n_helices=4, per_helix=20, radius=7.0, sequence=None):
    st = syn.make_bundle(n_helices, per_helix, radius=radius,
                         sequence=(sequence or POOR * n_helices))
    return DesignContext(structure=st)


def membrane_ctx():
    b0 = syn.make_bundle(4, 26, radius=9.0)
    ca = b0.coords("ca")
    zmid = ca[:, 2].mean()
    seq = "".join("LIVF"[i % 4] if abs(ca[i, 2] - zmid) <= 13 else "KEDS"[i % 4]
                  for i in range(len(b0)))
    st = syn.make_bundle(4, 26, radius=9.0, sequence=seq)
    return DesignContext(structure=st, membrane=M.estimate(st))


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_is_size_independent():
    """The same architecture at two lengths must land close together.

    Every feature is a fraction, a density or a log precisely so that a small
    design and a large one are comparable at all. Note this means *same
    architecture, more residues* -- a three-helix bundle and a six-helix
    bundle are genuinely different contexts, because the three-helix case has
    no buried core at all, and the fingerprint is right to separate them.
    """
    short = fp.compute(helical(n_helices=4, per_helix=18))
    long = fp.compute(helical(n_helices=4, per_helix=30))
    assert len(long.to_array()) == len(short.to_array())
    assert short.similarity(long) > 0.5


def test_fingerprint_separates_different_architectures():
    """The flip side: differing burial profiles must not look alike."""
    three = fp.compute(helical(n_helices=3, per_helix=20))
    six = fp.compute(helical(n_helices=6, per_helix=20))
    assert three.frac_core < six.frac_core        # the real difference
    assert three.similarity(six) < 0.5


def test_similar_folds_score_higher_than_dissimilar():
    helix = fp.compute(helical())
    strand_st = syn.make("VIYFVIYFVIYFVIYFVIYF", "strand")
    strand = fp.compute(DesignContext(structure=strand_st))
    other_helix = fp.compute(helical(n_helices=5))
    assert helix.similarity(other_helix) > helix.similarity(strand)


def test_identity_features_outweigh_damage_features():
    """Fold identity has to anchor the metric.

    Unweighted, aggregation load and exposed hydrophobic fraction contributed
    95% of the distance, so two identical folds looked unrelated purely
    because one was damaged.
    """
    clean = helical(sequence="AEKLAEKLAEKLAEKLAEKL" * 4)
    damaged = helical(sequence="AWKLAWWLAWWLAWWLAWWL" * 4)
    same_fold = fp.compute(clean).similarity(fp.compute(damaged))

    strand_st = syn.make("VIYFVIYFVIYFVIYFVIYF", "strand")
    different_fold = fp.compute(clean).similarity(
        fp.compute(DesignContext(structure=strand_st)))
    assert same_fold > different_fold


def test_aggregation_load_is_bounded():
    """An unbounded feature dwarfs every bounded one in the distance."""
    greasy = fp.compute(helical(sequence="WWWWWWWWWWWWWWWWWWWW" * 4))
    assert 0.0 <= greasy.aggregation_load < 1.0


def test_membrane_and_soluble_are_distinguishable():
    assert fp.compute(helical()).similarity(fp.compute(membrane_ctx())) < 0.5


def test_fingerprint_round_trips_through_dict():
    original = fp.compute(helical())
    assert fp.Fingerprint.from_dict(original.to_dict()) == original


def test_identical_fingerprints_are_maximally_similar():
    f = fp.compute(helical())
    assert f.similarity(f) == pytest.approx(1.0)
    assert f.distance(f) == pytest.approx(0.0)


# ------------------------------------------------------------- knowledge base

def test_priors_are_similarity_weighted():
    """A near-identical protein must count for more than an unrelated one."""
    near = fp.compute(helical(n_helices=4))
    far = fp.compute(DesignContext(structure=syn.make("VIYF" * 8, "strand")))
    query = fp.compute(helical(n_helices=4))

    kb = KnowledgeBase()
    kb.record(near, "core_packing", reward=1.0, success=True, protein="near")
    kb.record(far, "core_packing", reward=0.0, success=False, protein="far")

    prior = kb.priors(query)["core_packing"]
    # Weighted toward the similar protein's observation.
    assert prior.mean_reward > 0.5


def test_priors_respect_the_strategy_filter():
    f = fp.compute(helical())
    kb = KnowledgeBase()
    kb.record(f, "core_packing", 1.0, True)
    kb.record(f, "aromatic_belt", 1.0, True)
    assert set(kb.priors(f, ["core_packing"])) == {"core_packing"}


def test_knowledge_survives_a_round_trip(tmp_path):
    f = fp.compute(helical())
    kb = KnowledgeBase()
    kb.record(f, "core_packing", 0.5, True, protein="test")
    path = tmp_path / "kb.json"
    kb.save(path)

    reloaded = KnowledgeBase.load(path)
    assert len(reloaded) == 1
    assert reloaded.observations[0].strategy == "core_packing"
    assert reloaded.observations[0].fingerprint == f


def test_loading_a_missing_file_gives_an_empty_base(tmp_path):
    assert len(KnowledgeBase.load(tmp_path / "nope.json")) == 0


def test_stale_schema_is_rejected(tmp_path):
    """Old observations are not comparable once the features change."""
    import json
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": SCHEMA_VERSION - 1, "observations": []}))
    with pytest.raises(ValueError, match="schema"):
        KnowledgeBase.load(path)


def test_neighbours_ranks_by_similarity():
    query = fp.compute(helical(n_helices=4))
    kb = KnowledgeBase()
    kb.record(fp.compute(helical(n_helices=4)), "s", 1.0, True, protein="twin")
    kb.record(fp.compute(DesignContext(structure=syn.make("VIYF" * 8, "strand"))),
              "s", 1.0, True, protein="stranger")
    ranked = kb.neighbours(query)
    assert ranked[0][1] == "twin"


def test_report_conditions_on_context():
    f = fp.compute(helical())
    kb = KnowledgeBase()
    kb.record(f, "core_packing", 1.0, True, protein="a")
    assert "core_packing" in kb.report(f)
    assert "empty" in KnowledgeBase().report()


# ------------------------------------------------------- prior-guided policy

def test_prior_gives_an_untried_arm_a_head_start():
    priors = {"good": Prior("good", mean_reward=5.0, success_rate=1.0,
                            weight=3.0, n_observations=3)}
    pol = PriorGuidedUCB(["good", "unknown"], random.Random(0), priors=priors)
    # "unknown" has no evidence at all, so it is explored first; "good" then
    # outranks it on the strength of transferred evidence.
    assert pol.select(["good", "unknown"], k=1) == ["unknown"]
    pol.update(["unknown"], reward=0.0, success=False)
    assert pol.select(["good", "unknown"], k=1) == ["good"]


def test_evidence_eventually_overrides_the_prior():
    """Transfer is a head start, not a verdict."""
    priors = {"overrated": Prior("overrated", mean_reward=10.0, success_rate=1.0,
                                 weight=2.0, n_observations=2)}
    pol = PriorGuidedUCB(["overrated", "solid"], random.Random(0), priors=priors)
    for _ in range(60):
        pol.update(["overrated"], reward=0.0, success=False)
        pol.update(["solid"], reward=1.0, success=True)
    mean, _ = pol._blended(pol.stats["overrated"])
    assert mean < 1.0
    assert pol.select(["overrated", "solid"], k=1) == ["solid"]


def test_make_policy_upgrades_to_prior_guided_when_priors_given():
    priors = {"a": Prior("a", 1.0, 1.0, 1.0, 1)}
    assert isinstance(make_policy("ucb1", ["a"], priors=priors), PriorGuidedUCB)
    assert isinstance(make_policy("ucb1", ["a"]), UCB1)


# ---------------------------------------------------------------- integration

def test_engine_records_observations_into_the_knowledge_base():
    kb = KnowledgeBase()
    ctx = helical()
    result = Engine(ctx, seed=0, knowledge=kb, protein="test").run(generations=12)
    assert len(kb) > 0
    assert result.fingerprint is not None
    assert all(o.protein == "test" for o in kb.observations)


def test_second_run_starts_with_priors_from_the_first():
    kb = KnowledgeBase()
    ctx = helical()
    Engine(ctx, seed=0, knowledge=kb, protein="first").run(generations=12)
    engine = Engine(helical(n_helices=5), seed=1, knowledge=kb, protein="second")
    assert isinstance(engine.policy, PriorGuidedUCB)
    assert engine.policy.priors, "no knowledge transferred to the second run"


def test_engine_without_knowledge_is_unchanged():
    ctx = helical()
    engine = Engine(ctx, seed=0)
    assert engine.fingerprint is None
    assert not isinstance(engine.policy, PriorGuidedUCB)
