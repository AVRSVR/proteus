"""Strategy interface and registry.

A Proteus strategy is a *mechanism*, not a mutation rule. Each one names the
biophysical effect it is trying to exploit, decides for itself whether the
structure in front of it actually presents that opportunity, and then proposes
where to act. The two halves matter separately:

``diagnose``  -- where, if anywhere, does this structure exhibit the weakness
                 this mechanism addresses? A strategy that diagnoses nothing is
                 never sampled, so the engine spends its budget on mechanisms
                 that are relevant to *this* protein.
``propose``   -- what residue choices implement the mechanism there?

Keeping diagnosis separate is what lets the leaderboard learn something
transferable. "Loop rigidification helps proteins with long flexible loops" is
a claim about structural context; "mutation L47P helps" is not.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..context import DesignContext
from ..proposals import Proposal

# Which environments a strategy is valid in.
SOLUBLE = "soluble"
MEMBRANE = "membrane"


class Strategy(ABC):
    """Base class for all stabilization mechanisms."""

    name: str = "unnamed"
    mechanism: str = ""
    applies_to: frozenset[str] = frozenset({SOLUBLE, MEMBRANE})
    #: Strategies that target the same physics and should not be combined.
    conflicts_with: frozenset[str] = frozenset()

    @abstractmethod
    def diagnose(self, ctx: DesignContext) -> list[int]:
        """Positions where this mechanism has something to offer."""

    @abstractmethod
    def propose(self, ctx: DesignContext, positions: list[int],
                rng: random.Random) -> list[Proposal]:
        """Residue choices implementing the mechanism at ``positions``."""

    # ------------------------------------------------------------------ glue

    def valid_for(self, ctx: DesignContext) -> bool:
        env = MEMBRANE if ctx.is_membrane else SOLUBLE
        return env in self.applies_to

    def applicable(self, ctx: DesignContext) -> bool:
        return self.valid_for(ctx) and bool(self.diagnose(ctx))

    def run(self, ctx: DesignContext, rng: random.Random,
            max_positions: int | None = None) -> list[Proposal]:
        """Diagnose then propose, optionally sampling a subset of positions.

        Sampling matters: applying a mechanism at every eligible position at
        once is usually too large a jump for the optimizer to accept. Smaller
        moves make the accept/reject signal interpretable.
        """
        if not self.valid_for(ctx):
            return []
        positions = self.diagnose(ctx)
        if not positions:
            return []
        if max_positions is not None and len(positions) > max_positions:
            positions = rng.sample(positions, max_positions)
        return self.propose(ctx, positions, rng)

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"


class PairStrategy(Strategy):
    """Base for mechanisms acting on *pairs* of positions.

    Disulfides and salt bridges both install a two-residue feature, and both
    need the same discipline: only report as diagnosed what the strategy will
    actually act on. Reporting every position in every candidate pair badly
    overstates applicability -- 82 sites for a strategy that then emitted a
    single pair -- which tells the selector a mechanism is far more relevant
    than it is, and corrupts the leaderboard it is supposed to inform.
    """

    #: Cap on features installed per move. Small on purpose: each one
    #: constrains the fold, and applying many at once makes the accept/reject
    #: signal uninterpretable.
    MAX_PAIRS = 3

    @abstractmethod
    def candidate_pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        """Position pairs whose geometry can host this feature."""

    def select_pairs(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily take non-overlapping pairs, up to the cap."""
        chosen: list[tuple[int, int]] = []
        used: set[int] = set()
        for i, j in pairs:
            if i in used or j in used:
                continue
            chosen.append((i, j))
            used |= {i, j}
            if len(chosen) >= self.MAX_PAIRS:
                break
        return chosen

    def diagnose(self, ctx: DesignContext) -> list[int]:
        pairs = self.select_pairs(self.candidate_pairs(ctx))
        return sorted({x for pair in pairs for x in pair})


class REGISTRY:
    """Global strategy registry."""

    _items: dict[str, Strategy] = {}

    @classmethod
    def register(cls, strategy: Strategy) -> Strategy:
        if strategy.name in cls._items:
            raise ValueError(f"duplicate strategy name: {strategy.name}")
        cls._items[strategy.name] = strategy
        return strategy

    @classmethod
    def get(cls, name: str) -> Strategy:
        if name not in cls._items:
            raise KeyError(f"unknown strategy {name!r}; "
                           f"available: {sorted(cls._items)}")
        return cls._items[name]

    @classmethod
    def all(cls) -> list[Strategy]:
        return [cls._items[k] for k in sorted(cls._items)]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._items)

    @classmethod
    def for_context(cls, ctx: DesignContext) -> list[Strategy]:
        """Strategies valid in this environment (membrane vs soluble)."""
        return [s for s in cls.all() if s.valid_for(ctx)]

    @classmethod
    def applicable(cls, ctx: DesignContext) -> list[Strategy]:
        """Strategies that both fit the environment and diagnose something."""
        return [s for s in cls.for_context(ctx) if s.diagnose(ctx)]


def register(strategy_cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: instantiate and register a strategy."""
    REGISTRY.register(strategy_cls())
    return strategy_cls
