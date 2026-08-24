"""Cross-protein memory: what worked, and on what kind of protein.

A bandit inside a single run learns which mechanisms help *this* protein and
then throws that away. The knowledge base keeps the observations, tagged with
the structural fingerprint of the protein they came from, so that the next run
starts with an informed prior instead of from scratch.

The unit of knowledge is deliberately ``(structural context, mechanism,
outcome)`` rather than ``(position, mutation, outcome)``. The former is a claim
that can transfer between proteins; the latter is a fact about one position in
one structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .fingerprint import Fingerprint

#: Bumped when the fingerprint feature set changes, which invalidates
#: previously stored observations.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Observation:
    """One strategy application and what came of it."""

    fingerprint: Fingerprint
    strategy: str
    reward: float
    success: bool
    protein: str = ""

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "strategy": self.strategy,
            "reward": self.reward,
            "success": self.success,
            "protein": self.protein,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        return cls(
            fingerprint=Fingerprint.from_dict(data["fingerprint"]),
            strategy=data["strategy"],
            reward=float(data["reward"]),
            success=bool(data["success"]),
            protein=data.get("protein", ""),
        )


@dataclass(frozen=True)
class Prior:
    """A similarity-weighted expectation for one strategy in one context."""

    strategy: str
    mean_reward: float
    success_rate: float
    #: Sum of similarity weights -- an effective sample size, not a raw count.
    #: Two observations from near-identical proteins are worth more than ten
    #: from unrelated ones.
    weight: float
    n_observations: int

    @property
    def is_informative(self) -> bool:
        return self.weight >= 0.5


@dataclass
class KnowledgeBase:
    """Accumulated observations across runs and proteins."""

    observations: list[Observation] = field(default_factory=list)
    bandwidth: float = 0.5

    def __len__(self) -> int:
        return len(self.observations)

    # ------------------------------------------------------------- recording

    def record(self, fingerprint: Fingerprint, strategy: str, reward: float,
               success: bool, protein: str = "") -> None:
        self.observations.append(
            Observation(fingerprint, strategy, reward, success, protein)
        )

    def record_run(self, fingerprint: Fingerprint, policy, protein: str = "") -> int:
        """Absorb every pull from a finished run's policy.

        Reward is the per-pull improvement, and the engine only awards a
        positive reward to a move that improved the score, so a positive
        reward and a success are the same event.
        """
        added = 0
        for name, rewards in policy.per_arm_history().items():
            for reward in rewards:
                self.record(fingerprint, name, reward, reward > 0.0, protein)
                added += 1
        return added

    # -------------------------------------------------------------- querying

    def priors(self, fingerprint: Fingerprint,
               strategies: list[str] | None = None) -> dict[str, Prior]:
        """Similarity-weighted expectations for a new protein.

        Every stored observation contributes in proportion to how similar its
        protein was to this one, so a nearly identical fold dominates and an
        unrelated one barely registers.
        """
        buckets: dict[str, list[tuple[float, Observation]]] = {}
        for obs in self.observations:
            if strategies is not None and obs.strategy not in strategies:
                continue
            w = fingerprint.similarity(obs.fingerprint, self.bandwidth)
            if w <= 1e-6:
                continue
            buckets.setdefault(obs.strategy, []).append((w, obs))

        out: dict[str, Prior] = {}
        for name, items in buckets.items():
            total_w = sum(w for w, _ in items)
            if total_w <= 0:
                continue
            out[name] = Prior(
                strategy=name,
                mean_reward=sum(w * o.reward for w, o in items) / total_w,
                success_rate=sum(w * float(o.success) for w, o in items) / total_w,
                weight=total_w,
                n_observations=len(items),
            )
        return out

    def neighbours(self, fingerprint: Fingerprint, k: int = 5) -> list[tuple[float, str]]:
        """The most similar proteins seen before, as (similarity, name)."""
        seen: dict[str, float] = {}
        for obs in self.observations:
            if not obs.protein:
                continue
            sim = fingerprint.similarity(obs.fingerprint, self.bandwidth)
            seen[obs.protein] = max(seen.get(obs.protein, 0.0), sim)
        return sorted(((s, n) for n, s in seen.items()), reverse=True)[:k]

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        payload = {
            "schema": SCHEMA_VERSION,
            "bandwidth": self.bandwidth,
            "observations": [o.to_dict() for o in self.observations],
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        p = Path(path)
        if not p.exists():
            return cls()
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"knowledge base at {p} uses schema {payload.get('schema')}, "
                f"this build expects {SCHEMA_VERSION}. The fingerprint feature "
                f"set changed, so old observations are not comparable."
            )
        return cls(
            observations=[Observation.from_dict(d) for d in payload["observations"]],
            bandwidth=float(payload.get("bandwidth", 0.5)),
        )

    # -------------------------------------------------------------- reporting

    def report(self, fingerprint: Fingerprint | None = None,
               top: int | None = None) -> str:
        """Leaderboard, optionally conditioned on a structural context."""
        if not self.observations:
            return "knowledge base is empty"

        if fingerprint is None:
            agg: dict[str, list[float]] = {}
            for o in self.observations:
                agg.setdefault(o.strategy, []).append(o.reward)
            rows = [(name, sum(v) / len(v), float(len(v)), len(v))
                    for name, v in agg.items()]
            header = f"unconditional leaderboard ({len(self.observations)} observations)"
        else:
            priors = self.priors(fingerprint)
            rows = [(p.strategy, p.mean_reward, p.weight, p.n_observations)
                    for p in priors.values()]
            header = f"leaderboard for: {fingerprint.describe()}"

        rows.sort(key=lambda r: r[1], reverse=True)
        if top:
            rows = rows[:top]
        if not rows:
            return header + "\n  (no comparable observations)"

        width = max(len(r[0]) for r in rows)
        lines = [header, "",
                 f"{'strategy'.ljust(width)}  mean reward   weight    n"]
        lines.append("-" * len(lines[-1]))
        for name, mean, weight, n in rows:
            lines.append(f"{name.ljust(width)}  {mean:11.4f}  {weight:7.2f}  {n:3d}")
        return "\n".join(lines)
