"""Mutation proposals and explicit conflict resolution.

In the original prototype, strategies wrote lines into a shared resfile. When
two strategies targeted the same position the later line silently won, so the
meaning of a strategy *combination* was decided by dict iteration order rather
than by anything principled -- and nothing was reported.

Proteus makes that explicit. A strategy emits ``Proposal`` objects describing
which residues it would allow at a position and why. The resolver intersects
overlapping proposals, records every conflict it had to arbitrate, and refuses
outright to touch frozen positions. Every design decision is attributable to
the strategy that made it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# Chemically-motivated residue sets that strategies reuse.
HYDROPHOBIC = frozenset("AVLIMFWY")
AROMATIC = frozenset("FWY")
POLAR = frozenset("STNQCY")
CHARGED = frozenset("DEKR")
NEGATIVE = frozenset("DE")
POSITIVE = frozenset("KR")
HELIX_FORMER = frozenset("AELMQKR")
BETA_FORMER = frozenset("VIYFWT")
SMALL = frozenset("AGSC")
ALL = frozenset(CANONICAL)


@dataclass(frozen=True)
class Proposal:
    """One strategy's opinion about one position.

    ``allowed`` is the set of residues the strategy will permit here. A
    proposal never dictates a single identity unless the mechanism genuinely
    requires one (a disulfide needs cysteine); otherwise it narrows the choice
    and lets the packer pick within that.
    """

    resi: int
    allowed: frozenset[str]
    strategy: str
    rationale: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        bad = set(self.allowed) - set(CANONICAL)
        if bad:
            raise ValueError(f"{self.strategy}: non-canonical residues {sorted(bad)}")
        if not self.allowed:
            raise ValueError(f"{self.strategy}: empty proposal at {self.resi}")


@dataclass(frozen=True)
class Conflict:
    """Record of two or more strategies disagreeing about a position."""

    resi: int
    strategies: tuple[str, ...]
    proposed: tuple[frozenset[str], ...]
    resolution: frozenset[str]
    resolved_by: str

    def describe(self) -> str:
        parts = ", ".join(
            f"{s}->{''.join(sorted(a))}" for s, a in zip(self.strategies, self.proposed)
        )
        return (f"residue {self.resi}: {parts} | kept "
                f"{''.join(sorted(self.resolution))} ({self.resolved_by})")


@dataclass
class Resolution:
    """Merged design specification plus a full account of how it was reached."""

    allowed: dict[int, frozenset[str]] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    attribution: dict[int, tuple[str, ...]] = field(default_factory=dict)
    blocked_frozen: dict[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def n_designed(self) -> int:
        return len(self.allowed)

    def summary(self) -> str:
        lines = [f"{self.n_designed} positions designed, "
                 f"{len(self.conflicts)} conflicts arbitrated"]
        if self.blocked_frozen:
            n = len(self.blocked_frozen)
            lines.append(f"{n} proposals rejected at frozen positions")
        return "\n".join(lines)


def resolve(
    proposals: list[Proposal],
    frozen: frozenset[int] | set[int] = frozenset(),
) -> Resolution:
    """Merge proposals into one design specification.

    Rules, in order:

    1. Any proposal at a frozen position is rejected and recorded. Freezing is
       a hard constraint, not a default that later rules can override.
    2. Proposals at the same position are intersected -- a residue must satisfy
       every strategy that has an opinion about it.
    3. If the intersection is empty the strategies are genuinely incompatible
       here. The highest-weighted proposal wins and the conflict is recorded,
       so a combination that keeps fighting itself is visible rather than
       quietly producing whatever the last strategy asked for.
    """
    frozen = frozenset(frozen)
    grouped: dict[int, list[Proposal]] = defaultdict(list)
    blocked: dict[int, list[str]] = defaultdict(list)

    for p in proposals:
        if p.resi in frozen:
            blocked[p.resi].append(p.strategy)
        else:
            grouped[p.resi].append(p)

    result = Resolution(
        blocked_frozen={k: tuple(v) for k, v in blocked.items()},
    )

    for resi, group in sorted(grouped.items()):
        names = tuple(p.strategy for p in group)
        result.attribution[resi] = names

        if len(group) == 1:
            result.allowed[resi] = group[0].allowed
            continue

        merged = frozenset.intersection(*(p.allowed for p in group))
        if merged:
            result.allowed[resi] = merged
            continue

        winner = max(group, key=lambda p: p.weight)
        result.allowed[resi] = winner.allowed
        result.conflicts.append(Conflict(
            resi=resi,
            strategies=names,
            proposed=tuple(p.allowed for p in group),
            resolution=winner.allowed,
            resolved_by=f"highest weight: {winner.strategy}",
        ))

    return result


def to_resfile(
    resolution: Resolution,
    structure,
    default: str = "NATRO",
) -> str:
    """Render a Resolution as a Rosetta resfile.

    The default is ``NATRO`` -- native rotamer, no design, no repack. Anything
    Proteus has not explicitly decided to change is held fixed. The prototype
    used ``ALLAA`` as its default, which meant every position the strategies
    did not mention (including the entire "frozen" binding face) was silently
    thrown open to full redesign.
    """
    lines = [default, "start"]
    for resi in sorted(resolution.allowed):
        allowed = resolution.allowed[resi]
        res = structure[resi]
        if allowed == ALL:
            lines.append(f"{res.pdb_number} {res.chain} ALLAA")
        else:
            lines.append(f"{res.pdb_number} {res.chain} PIKAA {''.join(sorted(allowed))}")
    return "\n".join(lines) + "\n"
