"""Objective functions.

Two things went wrong with scoring in the prototype and both are structural
rather than cosmetic.

First, it optimised *total* Rosetta energy against a fixed target of -250 REU.
Total energy scales with chain length, so for a 54-residue protein that target
sits outside the physically reachable range and the loop could never terminate.
Everything downstream of it -- including the entire MD validation branch -- was
unreachable code. Scores here are therefore reported per residue, and the
engine optimises *change relative to the starting structure*, which is the
quantity that actually means "more stable than what I was given".

Second, greedy descent on a single energy term is a licence to exploit that
term. Any scorer used inside an optimisation loop needs terms that push back on
each other -- packing against aggregation, charge against burial -- or the
optimiser finds the degenerate corner. The heuristic scorer below is built that
way deliberately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .context import DesignContext

# Kyte-Doolittle hydropathy, rescaled to roughly [-1, 1].
HYDROPATHY = {
    "A": 0.40, "R": -1.00, "N": -0.78, "D": -0.78, "C": 0.56, "Q": -0.78,
    "E": -0.78, "G": -0.09, "I": 1.00, "L": 0.84, "K": -0.87, "M": 0.42,
    "F": 0.62, "P": -0.36, "S": -0.18, "T": -0.16, "W": -0.20, "Y": -0.29,
    "V": 0.93, "H": -0.71, "X": 0.0,
}

# Sidechain volume (A^3), used for packing terms.
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0, "X": 130.0,
}

# Chou-Fasman-style propensities (>1 favours that secondary structure).
HELIX_PROP = {
    "E": 1.51, "M": 1.45, "A": 1.42, "L": 1.21, "K": 1.16, "F": 1.13,
    "Q": 1.11, "W": 1.08, "I": 1.08, "V": 1.06, "D": 1.01, "H": 1.00,
    "R": 0.98, "T": 0.83, "S": 0.77, "C": 0.70, "Y": 0.69, "N": 0.67,
    "P": 0.57, "G": 0.57, "X": 1.00,
}
SHEET_PROP = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "C": 1.19, "W": 1.37, "F": 1.38,
    "L": 1.30, "T": 1.19, "M": 1.05, "A": 0.83, "R": 0.93, "G": 0.75,
    "D": 0.54, "K": 0.74, "S": 0.75, "H": 0.87, "N": 0.89, "Q": 1.10,
    "P": 0.55, "E": 0.37, "X": 1.00,
}

CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# Aggregation propensity, which is *not* the same axis as hydropathy.
#
# Kyte-Doolittle measures membrane-insertion free energy and rates tryptophan
# and tyrosine as hydrophilic, because they are amphipathic: a bulky nonpolar
# ring carrying a polar group. For aggregation the aromatics are among the
# worst offenders -- aromatic stacking is a principal driver of amyloid, and
# experimental scales (Aggrescan, TANGO) rank them at the top. Scoring
# aggregation on a hydropathy scale therefore gives a protein with a
# tryptophan-covered surface a free pass, which is exactly the failure mode
# a greedy optimizer will find and exploit.
#
# Charged residues take negative values: they act as gatekeepers, actively
# suppressing aggregation of the patch around them.
AGGREGATION = {
    "I": 1.00, "F": 1.00, "W": 0.95, "L": 0.94, "V": 0.90, "Y": 0.82,
    "M": 0.70, "C": 0.60, "A": 0.32, "T": 0.24, "G": 0.15, "S": 0.14,
    "H": 0.10, "Q": 0.08, "N": 0.06, "P": -0.10, "R": -0.45, "K": -0.55,
    "E": -0.60, "D": -0.60, "X": 0.0,
}


@dataclass
class ScoreBreakdown:
    """Per-term decomposition, so a score can be argued with."""

    total: float
    terms: dict[str, float] = field(default_factory=dict)
    n_residues: int = 0

    @property
    def per_residue(self) -> float:
        return self.total / self.n_residues if self.n_residues else 0.0

    def table(self) -> str:
        width = max((len(k) for k in self.terms), default=8)
        lines = [f"{'term'.ljust(width)}  value"]
        lines.append("-" * len(lines[0]))
        for k, v in sorted(self.terms.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"{k.ljust(width)}  {v:8.3f}")
        lines.append(f"{'TOTAL'.ljust(width)}  {self.total:8.3f} "
                     f"({self.per_residue:.3f}/residue)")
        return "\n".join(lines)


class Scorer(ABC):
    """Objective interface. Lower is better, following energy convention."""

    name: str = "scorer"

    @abstractmethod
    def score(self, ctx: DesignContext, sequence: str) -> ScoreBreakdown:
        """Score ``sequence`` threaded onto the structure in ``ctx``."""

    def total(self, ctx: DesignContext, sequence: str) -> float:
        return self.score(ctx, sequence).total


class HeuristicScorer(Scorer):
    """A transparent, dependency-free objective.

    This is a *screening* function, not a force field. It exists so the whole
    engine runs and can be tested without a licensed Rosetta install, and so
    the terms driving a decision are readable. For real work, swap in
    :class:`RosettaScorer`; the engine takes either.

    The terms are deliberately in tension:

    ``burial``       rewards hydrophobics where buried, polars where exposed
                     -- inverted inside the bilayer, where lipid-facing wants
                     hydrophobic and the bundle interior tolerates polar.
    ``packing``      rewards filling core volume, penalises overpacking.
    ``ss_propensity`` rewards residues suited to their local secondary structure.
    ``aggregation``  penalises contiguous exposed hydrophobic patches -- this
                     is what stops ``burial`` being gamed by making everything
                     hydrophobic.
    ``net_charge``   penalises extreme net charge, which otherwise runs away
                     when surface strategies pile on charged residues.
    """

    name = "heuristic"

    # Weights chosen so no single term can dominate the others.
    W_BURIAL = 1.0
    W_PACKING = 0.6
    W_SS = 0.8
    W_AGGREGATION = 1.2
    W_CHARGE = 0.5

    IDEAL_CORE_VOLUME = 165.0        # roughly leucine
    PATCH_RADIUS = 8.0

    def score(self, ctx: DesignContext, sequence: str) -> ScoreBreakdown:
        if len(sequence) != len(ctx):
            raise ValueError(f"sequence length {len(sequence)} != {len(ctx)} residues")

        terms = {
            "burial": self._burial(ctx, sequence),
            "packing": self._packing(ctx, sequence),
            "ss_propensity": self._ss(ctx, sequence),
            "aggregation": self._aggregation(ctx, sequence),
            "net_charge": self._charge(sequence),
        }
        return ScoreBreakdown(total=sum(terms.values()), terms=terms,
                              n_residues=len(sequence))

    # ------------------------------------------------------------------ terms

    def _burial(self, ctx: DesignContext, seq: str) -> float:
        """Hydrophobicity should track the environment, whichever way it points."""
        score = 0.0
        for p in ctx.positions:
            aa = seq[p - 1]
            h = HYDROPATHY.get(aa, 0.0)
            if ctx.is_membrane and ctx.zone(p) == "lipid_core":
                # Inverted: lipid exposure wants hydrophobic, bundle interior
                # tolerates (and uses) polar.
                want = 1.0 if ctx.layer(p) == "surface" else -0.2
            else:
                # Soluble: buried wants hydrophobic, exposed wants polar.
                want = {"core": 1.0, "boundary": 0.2, "surface": -0.6}[ctx.layer(p)]
            score -= h * want
        return self.W_BURIAL * score / max(len(seq), 1)

    def _packing(self, ctx: DesignContext, seq: str) -> float:
        core = [p for p in ctx.positions if ctx.layer(p) == "core"]
        if not core:
            return 0.0
        dev = [abs(VOLUME.get(seq[p - 1], 130.0) - self.IDEAL_CORE_VOLUME) / 100.0
               for p in core]
        return self.W_PACKING * float(np.mean(dev))

    def _ss(self, ctx: DesignContext, seq: str) -> float:
        score = 0.0
        for p in ctx.positions:
            aa, ss = seq[p - 1], ctx.ss_at(p)
            if ss == "H":
                score -= np.log(max(HELIX_PROP.get(aa, 1.0), 0.1))
            elif ss == "E":
                score -= np.log(max(SHEET_PROP.get(aa, 1.0), 0.1))
        return self.W_SS * float(score) / max(len(seq), 1)

    def _aggregation(self, ctx: DesignContext, seq: str) -> float:
        """Contiguous exposed hydrophobic surface, in 3D not in sequence.

        Aggregation nucleates on spatial patches, so neighbours are found by
        distance rather than by sequence position. Inside the bilayer an
        exposed hydrophobic is correct, not a liability, so the lipid-facing
        zone is exempt.
        """
        exposed = [p for p in ctx.positions
                   if ctx.layer(p) == "surface"
                   and not (ctx.is_membrane and ctx.zone(p) == "lipid_core")]
        if not exposed:
            return 0.0

        patch = 0.0
        for p in exposed:
            a = AGGREGATION.get(seq[p - 1], 0.0)
            if a <= 0:
                continue
            row = ctx.cb_dist[p - 1]
            # Neighbours contribute with sign: aggregation-prone residues add
            # to the patch, charged gatekeepers subtract from it. A patch that
            # is well policed by charges contributes nothing.
            neigh = sum(
                AGGREGATION.get(seq[q - 1], 0.0)
                for q in exposed
                if q != p and row[q - 1] <= self.PATCH_RADIUS
            )
            patch += a * max(neigh, 0.0)     # quadratic in local propensity
        return self.W_AGGREGATION * patch / max(len(exposed), 1)

    def _charge(self, seq: str) -> float:
        net = sum(CHARGE.get(a, 0.0) for a in seq)
        per_res = abs(net) / max(len(seq), 1)
        # Free up to ~0.05 |charge|/residue, quadratic beyond.
        excess = max(per_res - 0.05, 0.0)
        return self.W_CHARGE * (excess ** 2) * 100.0


class RosettaScorer(Scorer):
    """PyRosetta energy, per residue.

    Imported lazily so the package works without a Rosetta licence. Requires a
    backend that produces real poses -- see ``proteus.backends.rosetta``.
    """

    name = "rosetta"

    def __init__(self, weights: str = "ref2015", membrane: bool = False) -> None:
        self.weights = "franklin2019" if membrane else weights
        self._sfxn = None

    def _scorefxn(self):
        if self._sfxn is None:
            try:
                import pyrosetta
            except ImportError as exc:                    # pragma: no cover
                raise ImportError(
                    "RosettaScorer needs pyrosetta. Install it separately "
                    "(licence required) or use HeuristicScorer."
                ) from exc
            self._sfxn = pyrosetta.create_score_function(self.weights)
        return self._sfxn

    def score(self, ctx: DesignContext, sequence: str) -> ScoreBreakdown:  # pragma: no cover
        raise NotImplementedError(
            "RosettaScorer scores poses, not threaded sequences. Use it via "
            "proteus.backends.rosetta.RosettaBackend, which carries a live pose."
        )

    def score_pose(self, pose) -> ScoreBreakdown:                          # pragma: no cover
        total = float(self._scorefxn()(pose))
        return ScoreBreakdown(total=total, terms={self.weights: total},
                              n_residues=pose.total_residue())
