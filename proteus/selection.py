"""Strategy selection: the leaderboard.

The prototype kept a weight per strategy, added 0.5 on success and subtracted
0.1 on failure, and sampled proportionally. That is a rich-get-richer rule with
no notion of confidence: weights grow without bound, a strategy that happened
to succeed twice early dominates sampling, and exploration stops long before
the evidence justifies it.

Selection here is treated as what it actually is -- a multi-armed bandit under
a fixed evaluation budget. Two policies are provided:

``UCB1``      deterministic, optimistic in the face of uncertainty. Every arm
              is tried once, then arms are picked by mean reward plus a
              confidence bonus that shrinks as evidence accumulates.
``Thompson``  Bayesian, samples from each arm's posterior. Usually better with
              noisy rewards, which is what a stochastic design protocol gives.

Both expose the same interface, so the engine does not care which is in use,
and both record enough history to answer the question the whole project is
about: *which mechanism works for which kind of protein?*
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ArmStats:
    """Running record for one strategy."""

    name: str
    pulls: int = 0
    total_reward: float = 0.0
    successes: int = 0
    #: Reward from each pull, in order, for post-hoc analysis.
    history: list[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.pulls if self.pulls else 0.0

    def update(self, reward: float, success: bool) -> None:
        self.pulls += 1
        self.total_reward += reward
        self.successes += int(success)
        self.history.append(reward)


class Policy(ABC):
    """Base class for strategy-selection policies."""

    def __init__(self, arms: list[str], rng: random.Random | None = None) -> None:
        self.stats: dict[str, ArmStats] = {a: ArmStats(a) for a in arms}
        self.rng = rng or random.Random()
        self.total_pulls = 0

    def add_arm(self, name: str) -> None:
        """Register a strategy discovered after construction."""
        self.stats.setdefault(name, ArmStats(name))

    @abstractmethod
    def _priority(self, arm: ArmStats) -> float:
        """Higher means more worth trying next."""

    def select(self, available: list[str], k: int = 1) -> list[str]:
        """Pick up to ``k`` distinct strategies from ``available``.

        Only strategies the caller says are applicable to the current structure
        are considered, so a mechanism that does not apply is never charged a
        failure for not being tried.
        """
        pool = [a for a in available if a in self.stats]
        if not pool:
            return []
        k = min(k, len(pool))

        # Untried arms first -- no evidence means no basis for ranking.
        untried = [a for a in pool if self.stats[a].pulls == 0]
        self.rng.shuffle(untried)
        chosen = untried[:k]
        if len(chosen) == k:
            return chosen

        ranked = sorted(
            (a for a in pool if a not in chosen),
            key=lambda a: self._priority(self.stats[a]),
            reverse=True,
        )
        return chosen + ranked[: k - len(chosen)]

    def update(self, arms: list[str], reward: float, success: bool) -> None:
        """Credit every strategy that contributed to one evaluation.

        Joint credit is a real limitation: when three mechanisms are applied
        together and the result improves, all three are rewarded even though
        perhaps only one helped. Larger sample counts disentangle this, and
        ``per_arm_history`` retains the raw record so the ambiguity can be
        analysed rather than hidden.
        """
        for a in arms:
            if a in self.stats:
                self.stats[a].update(reward, success)
        self.total_pulls += 1

    # ------------------------------------------------------------- reporting

    def leaderboard(self) -> list[ArmStats]:
        return sorted(
            self.stats.values(),
            key=lambda s: (s.mean_reward, s.pulls),
            reverse=True,
        )

    def per_arm_history(self) -> dict[str, list[float]]:
        return {name: list(s.history) for name, s in self.stats.items()}

    def table(self, top: int | None = None) -> str:
        rows = self.leaderboard()
        if top:
            rows = rows[:top]
        width = max((len(r.name) for r in rows), default=8)
        out = [f"{'strategy'.ljust(width)}  pulls  win%   mean reward"]
        out.append("-" * len(out[0]))
        for r in rows:
            if r.pulls == 0:
                out.append(f"{r.name.ljust(width)}      -     -             -")
            else:
                out.append(f"{r.name.ljust(width)}  {r.pulls:5d}  "
                           f"{100 * r.success_rate:4.0f}  {r.mean_reward:12.3f}")
        return "\n".join(out)


class UCB1(Policy):
    """Upper confidence bound. Optimistic, deterministic, no tuning beyond c."""

    def __init__(self, arms, rng=None, c: float = 1.4) -> None:
        super().__init__(arms, rng)
        self.c = c

    def _priority(self, arm: ArmStats) -> float:
        if arm.pulls == 0:
            return math.inf
        total = max(self.total_pulls, 1)
        return arm.mean_reward + self.c * math.sqrt(math.log(total) / arm.pulls)


class Thompson(Policy):
    """Beta-Bernoulli Thompson sampling on the success indicator.

    Rewards are continuous, but whether a move was *accepted* is a clean
    Bernoulli signal and is what the posterior tracks. With noisy design
    protocols this explores more gracefully than UCB1.
    """

    def __init__(self, arms, rng=None, prior_a: float = 1.0, prior_b: float = 1.0) -> None:
        super().__init__(arms, rng)
        self.prior_a = prior_a
        self.prior_b = prior_b

    def _priority(self, arm: ArmStats) -> float:
        a = self.prior_a + arm.successes
        b = self.prior_b + (arm.pulls - arm.successes)
        return self.rng.betavariate(a, b)


def make_policy(name: str, arms: list[str], rng: random.Random | None = None) -> Policy:
    policies = {"ucb1": UCB1, "thompson": Thompson}
    key = name.lower()
    if key not in policies:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(policies)}")
    return policies[key](arms, rng)
