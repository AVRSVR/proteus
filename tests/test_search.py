"""The propose-fold-check loop.

The loop exists because a single run usually fails the refold check. What
makes it more than a retry is the schedule: it shrinks the edit, because the
usual reason a design stops folding is the number of simultaneous mutations
rather than any individual one.
"""

import pytest

from proteus import DesignContext
from proteus.search import MIN_BUDGET_MUTATIONS, budget_schedule, search

from . import _synthetic as syn

POOR = "AGSTAGSTKLAGSTGGVAST"


def bundle(n=4, per=20):
    return syn.make_bundle(n, per, radius=7.0, sequence=POOR * n)


# ---------------------------------------------------------------- schedule

def test_budget_holds_full_size_while_exploring():
    """Early attempts vary the seed at full budget rather than shrinking."""
    early = [budget_schedule(i, 30, 0.15, 60) for i in range(1, 11)]
    assert all(b == pytest.approx(0.15) for b in early)


def test_budget_shrinks_monotonically_after_exploring():
    vals = [budget_schedule(i, 30, 0.15, 60) for i in range(1, 31)]
    tail = vals[10:]
    assert all(a >= b for a, b in zip(tail, tail[1:])), tail


def test_budget_never_falls_below_the_floor():
    """Shrinking past a couple of mutations gives up the edit entirely."""
    n = 60
    floor = MIN_BUDGET_MUTATIONS / n
    for i in range(1, 51):
        assert budget_schedule(i, 50, 0.15, n) >= floor - 1e-9


def test_budget_leaves_an_already_small_budget_alone():
    tiny = 0.01
    assert budget_schedule(40, 50, tiny, 60) == pytest.approx(tiny)


# -------------------------------------------------------------------- loop

def test_loop_stops_at_the_first_pass():
    ctx = DesignContext(structure=bundle())
    calls = []

    def folder(seq):
        calls.append(seq)
        return (1.0, 90.0, True, "")           # passes immediately

    result = search(ctx, folder, max_attempts=10, generations=10)
    assert result.winner is not None
    assert len(calls) == 1
    assert "passed on attempt" in result.stopped_because


def test_loop_finds_a_small_edit_when_large_ones_fail():
    """The point of the schedule: shrink until the design survives."""
    st = bundle()
    ctx = DesignContext(structure=st)

    def folder(seq):
        n = sum(1 for a, b in zip(st.sequence, seq) if a != b)
        rmsd = 1.0 + 0.4 * n
        return rmsd, 88.0, rmsd <= 2.0, ""

    result = search(ctx, folder, max_attempts=30, generations=15)
    assert result.winner is not None, result.summary()
    assert result.winner.n_mutations <= 3


def test_service_failure_does_not_shrink_the_budget():
    """A fold the service never returned says nothing about the design.

    Treating it as a failed design would quietly abandon large edits because
    the network had a bad minute.
    """
    ctx = DesignContext(structure=bundle())
    result = search(ctx, lambda s: (None, None, False, "HTTP 504"),
                    max_attempts=6, generations=10)
    budgets = {round(a.mutation_budget, 6) for a in result.attempts}
    assert len(budgets) == 1, budgets
    assert result.winner is None
    assert result.n_folded == 0


def test_closest_attempt_is_kept_when_nothing_passes():
    st = bundle()
    ctx = DesignContext(structure=st)
    seen = []

    def folder(seq):
        n = sum(1 for a, b in zip(st.sequence, seq) if a != b)
        rmsd = 3.0 + 0.1 * n                    # never passes
        seen.append(rmsd)
        return rmsd, 70.0, False, ""

    result = search(ctx, folder, max_attempts=8, generations=10)
    assert result.winner is None
    assert result.closest is not None
    assert result.closest.sc_rmsd == pytest.approx(min(seen))


def test_duplicate_candidates_are_not_folded_twice():
    """Folding costs minutes; re-folding an identical sequence learns nothing."""
    ctx = DesignContext(structure=bundle())
    folded = []

    def folder(seq):
        folded.append(seq)
        return 5.0, 70.0, False, ""

    search(ctx, folder, max_attempts=12, generations=10)
    assert len(folded) == len(set(folded))


def test_cancellation_is_honoured():
    ctx = DesignContext(structure=bundle())
    state = {"n": 0}

    def folder(seq):
        state["n"] += 1
        return 5.0, 70.0, False, ""

    result = search(ctx, folder, max_attempts=20, generations=10,
                    should_stop=lambda: state["n"] >= 2)
    assert result.stopped_because == "cancelled"
    assert state["n"] <= 3


def test_every_attempt_is_reported():
    ctx = DesignContext(structure=bundle())
    result = search(ctx, lambda s: (4.0, 70.0, False, ""),
                    max_attempts=5, generations=10)
    assert len(result.attempts) == 5
    assert all(a.sequence for a in result.attempts)


# ------------------------------------------------- burial-first relaxation

def test_stages_relax_burial_before_edit_size():
    """Burial is the stronger predictor, so it is relaxed first."""
    from proteus.search import allowed_layers
    assert allowed_layers(1, 12) == ("core", "boundary", "surface")
    assert allowed_layers(5, 12) == ("boundary", "surface")
    assert allowed_layers(12, 12) == ("surface",)


def test_budget_holds_until_the_final_stage():
    """Edit size only shrinks once restricting burial has already failed."""
    early = [budget_schedule(i, 12, 0.15, 58) for i in range(1, 9)]
    assert all(b == pytest.approx(0.15) for b in early)
    assert budget_schedule(12, 12, 0.15, 58) < 0.15


def test_later_attempts_leave_the_core_alone():
    """A restricted stage must not return any buried mutation."""
    st = bundle()
    ctx = DesignContext(structure=st)
    core = {p for p in ctx.positions if ctx.layer(p) == "core"}
    if not core:
        pytest.skip("test structure has no core positions")

    touched = []

    def folder(seq):
        changed = {i + 1 for i, (a, b) in enumerate(zip(st.sequence, seq)) if a != b}
        touched.append(changed)
        return 9.0, 70.0, False, ""          # never passes, so every stage runs

    search(ctx, folder, max_attempts=12, generations=12)
    # The final third runs surface-only; none of those may touch the core.
    for changed in touched[-2:]:
        assert not (changed & core), sorted(changed & core)


def test_restricting_layers_still_respects_user_frozen_positions():
    """Layer restriction adds to the frozen set rather than replacing it."""
    st = bundle()
    frozen = frozenset(range(1, 12))
    ctx = DesignContext(structure=st, frozen=frozen)
    seen = []

    def folder(seq):
        seen.append({i + 1 for i, (a, b) in enumerate(zip(st.sequence, seq)) if a != b})
        return 9.0, 70.0, False, ""

    search(ctx, folder, max_attempts=9, generations=12)
    for changed in seen:
        assert not (changed & frozen), sorted(changed & frozen)


def test_attempt_records_which_layers_it_could_touch():
    ctx = DesignContext(structure=bundle())
    result = search(ctx, lambda s: (9.0, 70.0, False, ""),
                    max_attempts=9, generations=12)
    assert all(a.layers for a in result.attempts)
    assert result.attempts[0].layers == ("core", "boundary", "surface")
    assert result.attempts[-1].layers == ("surface",)
