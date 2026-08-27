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
    #: position -> the strategies that proposed it on this move. Recorded live
    #: rather than reconstructed: strategies sample their positions randomly,
    #: so a replay with a fresh RNG attributes the wrong ones.
    attribution: dict[int, tuple[str, ...]] = field(default_factory=dict)
    #: position -> why, in words, from the winning proposal.
    rationale: dict[int, str] = field(default_factory=dict)

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
    fingerprint: object | None = None
    start_sequence: str = ""
    max_mutations: int | None = None
    #: Index into ``trajectory`` of the move that produced ``best_sequence``.
    #: Reasons must be gathered only up to here: the walk continues past the
    #: best point, and a later move's rationale does not explain the result.
    best_move_index: int | None = None

    @property
    def improvement(self) -> float:
        """Score improvement per residue; positive means better."""
        return self.start_score - self.best_score

    @property
    def raw_improvement(self) -> float:
        """Improvement on the objective alone, before the mutation price."""
        if self.breakdown_best is None:
            return self.improvement
        return self.start_score - self.breakdown_best.per_residue

    @property
    def n_mutations(self) -> int:
        if not self.start_sequence:
            return 0
        return sum(1 for a, b in zip(self.start_sequence, self.best_sequence)
                   if a != b)

    @property
    def identity(self) -> float:
        """Fraction of the input sequence retained."""
        if not self.start_sequence:
            return 1.0
        return 1.0 - self.n_mutations / len(self.start_sequence)

    @property
    def n_accepted(self) -> int:
        return sum(1 for m in self.trajectory if m.accepted)

    def provenance(self) -> dict[int, tuple[str, str, str]]:
        """Every position changed in the returned sequence, and why.

        Maps position -> (original, final, reason). Reasons are gathered by
        walking the accepted moves, but the set of positions is taken from
        ``best_sequence`` -- the thing actually handed back. Reporting the
        accepted walk instead would list changes absent from the result, which
        is exactly the confusion that showed 9 changed positions alongside a
        summary saying 100% identity retained.
        """
        reasons: dict[int, str] = {}
        cutoff = (len(self.trajectory) if self.best_move_index is None
                  else self.best_move_index + 1)
        for move in self.trajectory[:cutoff]:
            if not move.accepted:
                continue
            for pos, _old, _new in move.mutations:
                why = move.rationale.get(pos, "")
                if not why and pos in move.attribution:
                    why = "+".join(move.attribution[pos])
                if why:
                    reasons[pos] = why

        if not self.start_sequence:
            return {}
        out: dict[int, tuple[str, str, str]] = {}
        for i, (old, new) in enumerate(zip(self.start_sequence,
                                           self.best_sequence), start=1):
            if old != new:
                out[i] = (old, new, reasons.get(i, ""))
        return out

    def credit(self) -> dict[str, int]:
        """How many mutations in the returned sequence each strategy caused.

        Counted once per surviving position, attributed to the strategy that
        set its final residue. Summing every touch across the whole trajectory
        instead reported 62 credits for 8 mutations, because a position gets
        proposed repeatedly on the way to its final value.
        """
        from collections import Counter
        counts: Counter[str] = Counter()
        final_setter: dict[int, str] = {}
        cutoff = (len(self.trajectory) if self.best_move_index is None
                  else self.best_move_index + 1)
        for move in self.trajectory[:cutoff]:
            if not move.accepted:
                continue
            for pos, names in move.attribution.items():
                if names:
                    final_setter[pos] = names[0]
        for pos in self.provenance():
            name = final_setter.get(pos)
            if name:
                counts[name] += 1
        return dict(counts)

    def explain(self, limit: int | None = None) -> str:
        """Human-readable account of what was changed and on what grounds."""
        prov = self.provenance()
        if not prov:
            return "no positions changed"
        lines = [f"{len(prov)} positions changed"]
        for pos in sorted(prov)[:limit]:
            old, new, why = prov[pos]
            lines.append(f"  {pos:>4} {old}->{new}  {why or '(unattributed)'}")
        if limit and len(prov) > limit:
            lines.append(f"  ... and {len(prov) - limit} more")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"generations      : {len(self.trajectory)}",
            f"accepted         : {self.n_accepted}",
            f"score (start)    : {self.start_score:.4f} /residue",
            f"score (best)     : {self.best_score:.4f} /residue  (incl. mutation cost)",
            f"improvement      : {self.improvement:+.4f} /residue"
            + (f"  ({self.raw_improvement:+.4f} before mutation cost)"
               if self.breakdown_best is not None else ""),
            f"mutations        : {self.n_mutations}"
            + (f" of at most {self.max_mutations} allowed" if self.max_mutations else ""),
            f"sequence identity: {self.identity:.1%} retained",
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
    original: str | None = None,
    max_mutations: int | None = None,
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

    ``original`` and ``max_mutations`` cap how far the result may drift from
    the input sequence. Once the budget is spent, a position may still change
    if it is already mutated, but no previously-native position may be opened.
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
            if original is not None and max_mutations is not None:
                spent = sum(1 for a, b in zip(original, seq) if a != b)
                if spent >= max_mutations and seq[p - 1] == original[p - 1]:
                    # Budget exhausted: leave still-native positions alone.
                    continue
                if spent >= max_mutations:
                    allowed = [a for a in allowed if a != original[p - 1]] or allowed
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
        mutation_budget: float = 0.15,
        min_gain_per_mutation: float = 0.0003,
        knowledge=None,
        protein: str = "",
        allowed_strategies: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.ctx = ctx
        self.knowledge = knowledge
        self.protein = protein
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
        self._calibrated = temperature is not None
        self.tabu_tenure = tabu_tenure

        # A stabilization tool that rewrites half the sequence has not
        # stabilized the protein, it has designed a different one. Established
        # campaigns (PROSS, FRESCO) change single-digit percentages of
        # positions. Two mechanisms keep the edit small:
        #
        #   mutation_budget  a hard ceiling, as a fraction of designable
        #                    positions, on how far the design may drift from
        #                    the input sequence.
        #   min_gain_per_mutation
        #                    the improvement in per-residue score a mutation
        #                    must deliver to be worth keeping. This is what
        #                    stops the loop accumulating neutral edits, and it
        #                    is also what makes the tool *specific*.
        #
        # Gain-per-mutation is the quantity that separates a broken design from
        # a sound one. Measured across a panel: a score-hacked design offers
        # 0.00057 per mutation, an experimentally validated design 0.00016, and
        # natural evolved proteins 0.00001 or less. An absolute floor near
        # 0.0003 therefore repairs the first and leaves the rest alone, which
        # is the correct behaviour for a repair tool.
        #
        # Two earlier formulations failed, and both failures are instructive.
        # Expressing the price as cost/n_residues made it size-dependent: the
        # same setting was six times cheaper per mutation on a 400-residue
        # protein than on a 66-residue one, so large proteins accumulated
        # dozens of marginal edits. Expressing it *relative* to the gain
        # observed in the run normalised away exactly the signal that
        # distinguishes a broken protein from a sound one, and every protein
        # ran to the budget ceiling.
        #
        # The threshold is absolute and therefore tied to this scorer's scale.
        # Changing the scorer's terms or weights means re-measuring it; the
        # panel above is the procedure.
        self.mutation_budget = mutation_budget
        self.min_gain_per_mutation = max(min_gain_per_mutation, 0.0)
        n_designable = max(len(ctx.designable), 1)
        self.max_mutations = max(1, int(round(mutation_budget * n_designable)))

        available = [s.name for s in REGISTRY.for_context(ctx)]
        if allowed_strategies is not None:
            # Manual mode: the caller has chosen which mechanisms to apply, so
            # the bandit selects only among those. Names not valid in this
            # environment are dropped rather than silently failing later.
            wanted = set(allowed_strategies)
            available = [n for n in available if n in wanted]
            if not available:
                raise ValueError(
                    "none of the requested strategies apply here: "
                    f"{sorted(wanted)}")

        # If a knowledge base is supplied, describe this protein and pull
        # forward what worked on structurally similar ones. The prior is a
        # head start, not a verdict -- evidence from this run washes it out.
        self.fingerprint = None
        priors = None
        if knowledge is not None:
            from .fingerprint import compute as compute_fingerprint

            self.fingerprint = compute_fingerprint(ctx, self.scorer)
            priors = knowledge.priors(self.fingerprint, available)

        self.policy = (make_policy(policy, available, self.rng, priors=priors)
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

    def _mutation_penalty(self, n_mutations: int) -> float:
        """Price of having drifted this far from the input.

        Deliberately *not* divided by chain length. The question a mutation has
        to answer is "did this change earn its keep", and that question does
        not get easier because the protein is larger.
        """
        return self.min_gain_per_mutation * n_mutations

    def _calibrate(self, deltas: list[float]) -> None:
        """Set the temperature schedule from the observed delta scale.

        Temperature has no meaningful absolute value -- it must be compared
        against the size of the moves actually being made -- so it is measured
        rather than chosen. The starting point accepts a typical uphill move
        with probability about one half, cooling by an order of magnitude over
        the run.
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
        start_per_res = start.per_residue        # zero mutations, no penalty

        current_seq, current = start_seq, start_per_res
        best_seq, best = start_seq, start_per_res
        trajectory: list[Move] = []
        since_improvement = 0
        calibration_deltas: list[float] = []
        best_index: int | None = None
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
                                         tabu=tabu, generation=gen,
                                         original=start_seq,
                                         max_mutations=self.max_mutations)
            cand_muts = _n_mutations(start_seq, candidate)

            # Hard ceiling: past the budget the candidate is a different
            # protein, not a repaired one, so it is not considered at all.
            if cand_muts > self.max_mutations:
                self.policy.update(chosen, reward=0.0, success=False)
                continue

            cand_raw = self.scorer.score(live_ctx, candidate).per_residue
            cand_score = cand_raw + self._mutation_penalty(cand_muts)
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
            chosen_by = {pos: _winning_rationale(proposals, pos, new)
                         for pos, _old, new in mutations}
            move = Move(
                generation=gen, strategies=tuple(chosen), sequence=candidate,
                score=cand_score, delta=delta, accepted=accept, temperature=temp,
                n_designed=res.n_designed, n_conflicts=len(res.conflicts),
                mutations=mutations,
                attribution={p: (name,) for p, (name, _why) in chosen_by.items()
                             if name},
                rationale={p: why for p, (_name, why) in chosen_by.items() if why},
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
                best_index = len(trajectory) - 1
                since_improvement = 0
            else:
                since_improvement += 1

            if patience is not None and since_improvement >= patience:
                break

        # Fold this run's evidence back into the shared memory, so the next
        # protein starts from a better prior than this one did.
        if self.knowledge is not None and self.fingerprint is not None:
            self.knowledge.record_run(self.fingerprint, self.policy, self.protein)

        best_ctx = DesignContext(
            structure=_threaded(self.ctx, best_seq), frozen=self.ctx.frozen,
            membrane=self.ctx.membrane, dssp=self.ctx.ss,
        )
        return RunResult(
            best_sequence=best_seq, best_score=best, start_score=start_per_res,
            trajectory=trajectory, policy=self.policy,
            breakdown_start=start, breakdown_best=self.scorer.score(best_ctx, best_seq),
            fingerprint=self.fingerprint, start_sequence=start_seq,
            max_mutations=self.max_mutations, best_move_index=best_index,
        )


def _n_mutations(original: str, candidate: str) -> int:
    return sum(1 for a, b in zip(original, candidate) if a != b)


def _winning_rationale(proposals: list[Proposal], position: int,
                       chosen: str) -> tuple[str, str]:
    """The strategy that actually determined ``chosen`` at ``position``.

    Several strategies may propose at one position; only the one whose allowed
    set contains the residue finally picked explains the outcome. Reporting the
    first proposal instead produced attributions that contradicted themselves,
    such as a proline credited to helix capping -- a mechanism that proposes
    only Ser, Thr, Asp or Asn.
    """
    fallback = ("", "")
    for p in proposals:
        if p.resi != position:
            continue
        if chosen in p.allowed:
            return p.strategy, f"{p.strategy}: {p.rationale}"
        if not fallback[0]:
            fallback = (p.strategy, f"{p.strategy}: {p.rationale}")
    return fallback


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
