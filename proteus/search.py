"""Search for a design that both scores better and still folds.

A single stabilization run optimises the objective and stops. Whether the
result still adopts the intended backbone is a separate question, answered
afterwards by folding it, and in practice the answer is often no. Running once
and checking once therefore fails most of the time.

This closes the loop: propose, fold, check, and if the fold check fails, try
again differently. The important word is *differently*, and what to vary was
settled by measurement rather than assumption.

The first version shrank the number of mutations, on the theory that a design
stops folding because too many things changed at once. That theory is wrong,
at least for the case that tested it. On a three-helix design that refolds at
1.21 A, an attempt making *two* mutations collapsed it to 27 A while an
attempt making *four* left it at 1.21 A. Count was not the variable.

The variable was burial. The two-mutation failure contained A22I -- alanine to
isoleucine at a buried position, adding some 78 cubic angstroms of sidechain
into a packed core. The four-mutation success touched only the surface. A
buried substitution has to be accommodated by the entire fold; a surface one
mostly has to be tolerated by solvent.

So the schedule relaxes in two stages, burial first:

    stage 1   every position, full budget, different seeds
    stage 2   surface and boundary only -- the core is left alone
    stage 3   surface only, with the budget shrinking as well

If nothing passes, the attempt that came closest is reported rather than
discarded, because a run that reached 2.4 A is more useful than a bare
failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .context import DesignContext
from .engine import Engine

#: Below this many mutations there is little left to give up, so shrinking
#: further mostly wastes attempts.
MIN_BUDGET_MUTATIONS = 2

#: Layers the loop is willing to mutate at each stage, in order. Burial is
#: relaxed before edit size because it is the stronger predictor of whether a
#: design survives refolding -- see the module docstring for the measurement.
LAYER_STAGES = (
    ("core", "boundary", "surface"),
    ("boundary", "surface"),
    ("surface",),
)


@dataclass
class Attempt:
    """One proposal-and-check cycle."""

    index: int
    seed: int
    mutation_budget: float
    #: Layers this attempt was allowed to mutate.
    layers: tuple[str, ...]
    n_mutations: int
    identity: float
    improvement: float
    sequence: str
    #: None when the fold could not be obtained at all, which is a failure of
    #: the folding service rather than of the design.
    sc_rmsd: float | None = None
    plddt: float | None = None
    passed: bool = False
    note: str = ""

    def describe(self) -> str:
        if self.sc_rmsd is None:
            return (f"attempt {self.index}: {self.n_mutations} mutations, "
                    f"fold unavailable ({self.note})")
        verdict = "PASS" if self.passed else "fail"
        plddt = f", pLDDT {self.plddt:.1f}" if self.plddt is not None else ""
        where = "+".join(layer[0] for layer in self.layers)
        return (f"attempt {self.index}: {self.n_mutations} mutations [{where}], "
                f"scRMSD {self.sc_rmsd:.2f} A{plddt} -- {verdict}")


@dataclass
class SearchResult:
    """Everything the loop tried, and the best thing it found."""

    attempts: list[Attempt] = field(default_factory=list)
    winner: Attempt | None = None
    #: Best attempt by scRMSD even if it never passed, so a near miss is
    #: visible rather than thrown away.
    closest: Attempt | None = None
    stopped_because: str = ""

    @property
    def n_folded(self) -> int:
        return sum(1 for a in self.attempts if a.sc_rmsd is not None)

    def summary(self) -> str:
        lines = [f"{len(self.attempts)} attempts, {self.n_folded} folded",
                 f"stopped: {self.stopped_because}"]
        if self.winner:
            lines.append(f"found: {self.winner.describe()}")
        elif self.closest and self.closest.sc_rmsd is not None:
            lines.append(f"closest: {self.closest.describe()}")
        else:
            lines.append("nothing folded successfully")
        return "\n".join(lines)


def stage_for(attempt: int, total: int) -> int:
    """Which relaxation stage an attempt belongs to (0, 1 or 2).

    Split into thirds. The first third explores seeds with everything on the
    table, since a design that tolerates core changes is worth discovering
    rather than assuming away.
    """
    third = max(1, total // 3)
    if attempt <= third:
        return 0
    if attempt <= 2 * third:
        return 1
    return 2


def allowed_layers(attempt: int, total: int) -> tuple[str, ...]:
    """Layers this attempt may mutate."""
    return LAYER_STAGES[stage_for(attempt, total)]


def budget_schedule(attempt: int, total: int, start_budget: float,
                    n_designable: int) -> float:
    """Mutation budget for a given attempt.

    Holds full size until the final stage. Burial is relaxed first, so the
    budget only starts shrinking once the loop has already restricted itself
    to surface positions and still not succeeded.
    """
    explore = max(1, total // 3) * 2
    if attempt <= explore:
        return start_budget

    floor = MIN_BUDGET_MUTATIONS / max(n_designable, 1)
    if start_budget <= floor:
        return start_budget
    # Geometric decay from start_budget to floor across the remaining attempts.
    span = max(total - explore, 1)
    frac = (attempt - explore) / span
    return start_budget * (floor / start_budget) ** frac


def search(
    ctx: DesignContext,
    fold_and_check: Callable[[str], tuple[float | None, float | None, bool, str]],
    max_attempts: int = 30,
    generations: int = 40,
    start_budget: float = 0.15,
    scorer=None,
    allowed_strategies: list[str] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> SearchResult:
    """Propose, fold, check; repeat with a smaller edit until something passes.

    ``fold_and_check`` takes a sequence and returns
    ``(sc_rmsd, plddt, passed, note)``. ``sc_rmsd`` is None when the fold could
    not be obtained, which is treated as an inconclusive attempt rather than a
    rejection -- the folding service being down says nothing about the design.

    ``should_stop`` is polled between attempts so a caller can cancel.

    Attempts that produce no fold verdict -- the service failed, no mutations
    were proposed, or the candidate duplicates an earlier one -- do not advance
    the shrinking schedule, though they do count against ``max_attempts``.
    """
    result = SearchResult()
    n_designable = max(len(ctx.designable), 1)
    seen: set[str] = set()
    # The schedule advances only on attempts that actually told us something.
    # A fold the service failed to return is evidence about the service, not
    # about the design, and shrinking the budget in response would quietly
    # abandon large edits because the network had a bad minute.
    conclusive = 0

    for i in range(1, max_attempts + 1):
        if should_stop and should_stop():
            result.stopped_because = "cancelled"
            return result

        step = conclusive + 1
        budget = budget_schedule(step, max_attempts, start_budget, n_designable)
        layers = allowed_layers(step, max_attempts)

        # Restricting burial is expressed by freezing the positions that are
        # off limits, which the engine already guarantees it will not touch.
        attempt_ctx = ctx
        if len(layers) < 3:
            off_limits = frozenset(
                p for p in ctx.positions if ctx.layer(p) not in layers)
            if off_limits:
                attempt_ctx = DesignContext(
                    structure=ctx.structure, membrane=ctx.membrane,
                    frozen=ctx.frozen | off_limits)

        engine = Engine(attempt_ctx, seed=i, scorer=scorer,
                        allowed_strategies=allowed_strategies,
                        mutation_budget=budget,
                        min_gain_per_mutation=None if scorer else 0.0003)
        run = engine.run(generations=generations)

        if run.n_mutations == 0:
            # Nothing proposed at this budget; a smaller one will not help.
            attempt = Attempt(i, i, budget, layers, 0, 1.0, 0.0,
                              run.best_sequence,
                              note="no mutations proposed")
            result.attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            continue

        if run.best_sequence in seen:
            # The same candidate as a previous attempt; folding it again would
            # spend minutes to learn nothing.
            attempt = Attempt(i, i, budget, layers, run.n_mutations,
                              run.identity, run.improvement,
                              run.best_sequence,
                              note="duplicate of an earlier attempt")
            result.attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            continue
        seen.add(run.best_sequence)

        rmsd, plddt, passed, note = fold_and_check(run.best_sequence)
        attempt = Attempt(i, i, budget, layers, run.n_mutations, run.identity,
                          run.improvement, run.best_sequence,
                          sc_rmsd=rmsd, plddt=plddt, passed=passed, note=note)
        result.attempts.append(attempt)
        if on_attempt:
            on_attempt(attempt)

        if rmsd is not None:
            conclusive += 1
            if (result.closest is None or result.closest.sc_rmsd is None
                    or rmsd < result.closest.sc_rmsd):
                result.closest = attempt
        if passed:
            result.winner = attempt
            result.stopped_because = f"passed on attempt {i}"
            return result

    result.stopped_because = f"exhausted {max_attempts} attempts"
    return result
