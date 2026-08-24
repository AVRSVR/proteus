"""The optimization loop.

Three changes from the prototype's loop, each fixing a specific failure:

*Acceptance.* The prototype accepted a move only if it improved the score, so
it was a pure hill climber: once in a local minimum it could never leave, and
"evolution" was a misnomer -- there was no population and no exploration. Here
acceptance is Metropolis with an annealed temperature, so uphill moves are
taken with a probability that falls as the run proceeds.

*Objective.* It optimised total Rosetta energy toward a fixed constant. Here
the objective is the change per residue relative to the input structure, which
is both size-independent and the quantity that actually answers "is this more
stable than what I was handed".

*Termination.* Its ``while best > TARGET`` loop could not terminate for an
unreachable target. Here the run is bounded by a generation budget, with
optional early stopping on stagnation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .context import DesignContext
from .proposals import Proposal, Resolution, resolve
from .scoring import HeuristicScorer, ScoreBreakdown, Scorer
from .selection import Policy, make_policy
from .strategies import REGISTRY


@dataclass
class Move:
    """One generation of the loop, kept whole for post-hoc analysis."""

    generation: int
    strategies: tuple[str, ...]
    sequence: str
    score: float
    delta: float
    accepted: bool
    temperature: float
    n_designed: int
    n_conflicts: int
    mutations: tuple[tuple[int, str, str], ...] = ()

    def describe(self) -> str:
        verdict = "accept" if self.accepted else "reject"
        muts = ", ".join(f"{o}{r}{n}" for r, o, n in self.mutations[:5])
        if len(self.mutations) > 5:
            muts += f" (+{len(self.mutations) - 5} more)"
        return (f"gen {self.generation:3d}  {'+'.join(self.strategies):<45} "
                f"delta {self.delta:+7.4f}  {verdict}  [{muts}]")


@dataclass
class RunResult:
    """Everything a run produced."""

    best_sequence: str
    best_score: float
    start_score: float
    trajectory: list[Move] = field(default_factory=list)
    policy: Policy | None = None
    breakdown_start: ScoreBreakdown | None = None
    breakdown_best: ScoreBreakdown | None = None

    @property
    def improvement(self) -> float:
        """Score improvement per residue; positive means better."""
        return self.start_score - self.best_score

    @property
    def n_accepted(self) -> int:
        return sum(1 for m in self.trajectory if m.accepted)

    def summary(self) -> str:
        lines = [
            f"generations      : {len(self.trajectory)}",
            f"accepted         : {self.n_accepted}",
            f"score (start)    : {self.start_score:.4f} /residue",
            f"score (best)     : {self.best_score:.4f} /residue",
            f"improvement      : {self.improvement:+.4f} /residue",
        ]
        if self.policy is not None:
            lines.append("")
            lines.append(self.policy.table())
        return "\n".join(lines)


def realize_sequence(
    ctx: DesignContext,
    resolution: Resolution,
    scorer: Scorer,
    start_sequence: str,
    rng: random.Random,
    passes: int = 2,
    tabu: dict[tuple[int, str], int] | None = None,
    generation: int = 0,
) -> str:
    """Choose concrete residues from the allowed sets.

    With a real packer this is Rosetta's job. Without one, positions are
    optimised by iterated conditional modes: visit each designed position in
    random order and take the allowed residue that minimises the objective
    given everything else. Crude, but deterministic given a seed and good
    enough to exercise the loop honestly.

    ``tabu`` forbids specific (position, residue) choices that a recent move
    already moved away from, which is what stops two strategies overwriting
    each other's work indefinitely.
    """
    seq = list(start_sequence)
    positions = list(resolution.allowed)
    if not positions:
        return "".join(seq)

    for _ in range(passes):
        rng.shuffle(positions)
        for p in positions:
            allowed = sorted(resolution.allowed[p])
            if tabu:
                permitted = [a for a in allowed
                             if tabu.get((p, a), -1) < generation]
                # Never let the tabu list empty a position entirely.
                allowed = permitted or allowed
            if len(allowed) == 1:
                seq[p - 1] = allowed[0]
                continue
            best_aa, best_val = seq[p - 1], math.inf
            for aa in allowed:
                seq[p - 1] = aa
                val = scorer.total(ctx, "".join(seq))
                if val < best_val:
                    best_aa, best_val = aa, val
            seq[p - 1] = best_aa
    return "".join(seq)


class Engine:
    """Runs strategy selection, application, scoring and acceptance."""

    def __init__(
        self,
        ctx: DesignContext,
        scorer: Scorer | None = None,
        policy: Policy | str = "ucb1",
        strategies_per_move: int = 2,
        max_positions_per_strategy: int = 6,
        temperature: float | None = None,
        final_temperature: float | None = None,
        calibration_moves: int = 6,
        tabu_tenure: int = 4,
        seed: int | None = None,
    ) -> None:
        self.ctx = ctx
        self.scorer = scorer or HeuristicScorer()
        self.rng = random.Random(seed)
        self.k = strategies_per_move
        self.max_positions = max_positions_per_strategy
        # Temperature is meaningless in absolute units: the delta scale depends
        # on the scorer, the protein size and how large a move the strategies
        # make. A fixed default is therefore either far too hot (accept
        # everything, a random walk) or far too cold (pure hill climbing). When
        # not given explicitly it is calibrated from the first few observed
        # deltas instead.
        self.t0 = temperature
        self.t1 = final_temperature
        self.calibration_moves = calibration_moves
        self.tabu_tenure = tabu_tenure
        self._calibrated = temperature is not None

        available = [s.name for s in REGISTRY.for_context(ctx)]
        self.policy = (make_policy(policy, available, self.rng)
                       if isinstance(policy, str) else policy)
        for name in available:
            self.policy.add_arm(name)

    # ------------------------------------------------------------------ loop

    def _temperature(self, gen: int, total: int) -> float:
        """Geometric anneal from t0 to t1, once both are known."""
        if self.t0 is None or self.t1 is None:
            return 0.0                      # calibrating: greedy acceptance
        if total <= 1:
            return self.t1
        frac = gen / (total - 1)
        return self.t0 * (self.t1 / self.t0) ** frac

    def _calibrate(self, deltas: list[float]) -> None:
        """Set the temperature schedule from the observed delta scale.

        The starting temperature is chosen so a typical uphill move is accepted
        with probability about 1/2, and the final temperature is an order of
        magnitude colder, so the run ends close to greedy.
        """
        uphill = [abs(d) for d in deltas if d > 0]
        scale = (sorted(uphill)[len(uphill) // 2] if uphill
                 else max((abs(d) for d in deltas), default=1e-4))
        scale = max(scale, 1e-9)
        self.t0 = scale / math.log(2.0)
        self.t1 = self.t0 / 10.0
        self._calibrated = True

    def _applicable_now(self, sequence: str) -> list[str]:
        """Strategies that diagnose something on the *current* sequence.

        Re-diagnosing each generation matters: once core packing has filled the
        core, it correctly stops proposing, and the budget moves to mechanisms
        that still have something to offer.
        """
        live = DesignContext(
            structure=_threaded(self.ctx, sequence),
            frozen=self.ctx.frozen,
            membrane=self.ctx.membrane,
            dssp=self.ctx.ss,
            core_cutoff=self.ctx.core_cutoff,
            surface_cutoff=self.ctx.surface_cutoff,
        )
        return [s.name for s in REGISTRY.for_context(live) if s.diagnose(live)]

    def run(self, generations: int = 50, patience: int | None = None,
            verbose: bool = False) -> RunResult:
        start_seq = self.ctx.structure.sequence
        start = self.scorer.score(self.ctx, start_seq)
        start_per_res = start.per_residue

        current_seq, current = start_seq, start_per_res
        best_seq, best = start_seq, start_per_res
        trajectory: list[Move] = []
        since_improvement = 0
        calibration_deltas: list[float] = []
        # (position, residue) -> generation until which that choice is barred.
        tabu: dict[tuple[int, str], int] = {}

        for gen in range(1, generations + 1):
            temp = self._temperature(gen - 1, generations)
            live_ctx = DesignContext(
                structure=_threaded(self.ctx, current_seq),
                frozen=self.ctx.frozen,
                membrane=self.ctx.membrane,
                dssp=self.ctx.ss,
                core_cutoff=self.ctx.core_cutoff,
                surface_cutoff=self.ctx.surface_cutoff,
            )
            available = [s.name for s in REGISTRY.for_context(live_ctx)
                         if s.diagnose(live_ctx)]
            chosen = self.policy.select(available, k=self.k)
            if not chosen:
                break

            proposals: list[Proposal] = []
            for name in chosen:
                proposals.extend(
                    REGISTRY.get(name).run(live_ctx, self.rng, self.max_positions)
                )
            if not proposals:
                self.policy.update(chosen, reward=0.0, success=False)
                continue

            res = resolve(proposals, frozen=self.ctx.frozen)
            candidate = realize_sequence(live_ctx, res, self.scorer,
                                         current_seq, self.rng,
                                         tabu=tabu, generation=gen)
            cand_score = self.scorer.score(live_ctx, candidate).per_residue
            delta = cand_score - current

            # Metropolis: always take improvements, take regressions with a
            # probability that decays as the run cools. While calibrating,
            # acceptance is greedy and the deltas are only being measured.
            if delta <= 0:
                accept = True
            elif not self._calibrated:
                accept = False
            else:
                accept = self.rng.random() < math.exp(-delta / max(temp, 1e-12))

            if not self._calibrated:
                calibration_deltas.append(delta)
                if len(calibration_deltas) >= self.calibration_moves:
                    self._calibrate(calibration_deltas)

            mutations = tuple(
                (i + 1, a, b) for i, (a, b) in enumerate(zip(current_seq, candidate))
                if a != b
            )
            move = Move(
                generation=gen, strategies=tuple(chosen), sequence=candidate,
                score=cand_score, delta=delta, accepted=accept, temperature=temp,
                n_designed=res.n_designed, n_conflicts=len(res.conflicts),
                mutations=mutations,
            )
            trajectory.append(move)
            if verbose:
                print(move.describe())

            # Reward is improvement per residue, scaled so typical moves land
            # in a usable range for the bandit.
            reward = max(-delta, 0.0) * 100.0
            self.policy.update(chosen, reward=reward, success=delta < 0)

            if accept:
                # Bar reverting each change for a few generations. Without
                # this, two strategies with opposing views of the same
                # position overwrite each other indefinitely -- one writing
                # cysteine for a disulfide, the other writing it back to a
                # core hydrophobic -- and the run stops making progress.
                for pos, old, _new in mutations:
                    tabu[(pos, old)] = gen + self.tabu_tenure
                current_seq, current = candidate, cand_score
            if cand_score < best:
                best_seq, best = candidate, cand_score
                since_improvement = 0
            else:
                since_improvement += 1

            if patience is not None and since_improvement >= patience:
                break

        best_ctx = DesignContext(
            structure=_threaded(self.ctx, best_seq), frozen=self.ctx.frozen,
            membrane=self.ctx.membrane, dssp=self.ctx.ss,
        )
        return RunResult(
            best_sequence=best_seq, best_score=best, start_score=start_per_res,
            trajectory=trajectory, policy=self.policy,
            breakdown_start=start, breakdown_best=self.scorer.score(best_ctx, best_seq),
        )


def _threaded(ctx: DesignContext, sequence: str):
    """Return a Structure with ``sequence`` threaded onto the same backbone.

    Coordinates are unchanged. With the heuristic backend that is exactly
    right, since no repacking happens; with the Rosetta backend the pose
    carries real sidechains and this is not used.
    """
    from .structure import THREE_TO_ONE, Structure
    from dataclasses import replace

    one_to_three = {v: k for k, v in THREE_TO_ONE.items()}
    residues = [
        replace(r, name3=one_to_three.get(sequence[i], r.name3))
        for i, r in enumerate(ctx.structure.residues)
    ]
    return Structure(residues=residues, source=ctx.structure.source)
