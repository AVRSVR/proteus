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


# Chemical degradation routes. These are not folding terms -- they are the
# ways a protein falls apart in a tube over weeks, and nature avoids them in
# proteins that must last. Each is a sequence motif with a known mechanism:
#
#   N-G, N-S, N-N   asparagine deamidation, fastest when followed by a small
#                   flexible residue; converts Asn to Asp and adds a charge
#   D-G, D-P, D-S   aspartate isomerisation to iso-Asp, which kinks the chain
#   N-X-S/T         N-linked glycosylation sequon (X not proline)
#
# Weights are relative severities, not rates.
DEAMIDATION = {"NG": 1.0, "NS": 0.6, "NN": 0.5, "NT": 0.4, "NA": 0.3}
ISOMERISATION = {"DG": 1.0, "DP": 0.6, "DS": 0.5, "DD": 0.4}


def _byte_table(mapping: dict[str, float], default: float = 0.0,
                transform=None) -> np.ndarray:
    """Build a 256-entry array indexed by ASCII byte.

    Scoring is called tens of thousands of times per run, so per-residue table
    lookups are done by indexing a flat array with the encoded sequence rather
    than by dict access in a Python loop.
    """
    table = np.full(256, default, dtype=np.float64)
    for aa, value in mapping.items():
        if len(aa) == 1:
            table[ord(aa)] = transform(value) if transform else value
    return table


_LOOKUP = {
    "hydropathy": _byte_table(HYDROPATHY),
    "volume": _byte_table(VOLUME, default=130.0),
    "aggregation": _byte_table(AGGREGATION),
    "charge": _byte_table(CHARGE),
    # Propensities are consumed as logs, so the log is taken once here.
    "helix": _byte_table(HELIX_PROP, default=0.0,
                         transform=lambda v: float(np.log(max(v, 0.1)))),
    "sheet": _byte_table(SHEET_PROP, default=0.0,
                         transform=lambda v: float(np.log(max(v, 0.1)))),
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
    ``bb_entropy``   glycine costs unfolded-state entropy, proline in loops
                     recovers it.
    ``capping``      rewards satisfied helix N- and C-caps.
    ``liabilities``  penalises chemical degradation motifs -- deamidation,
                     isomerisation, glycosylation sequons, unpaired cysteine
                     and exposed methionine.
    ``interactions`` rewards aromatic stacking and cation-pi contacts, which
                     are real packing energy that a per-residue burial term
                     cannot see.

    The last two exist because of a systematic bias found during auditing: a
    mechanism the objective cannot measure will always lose on the leaderboard
    regardless of its merit. Loop rigidification and helix capping both scored
    exactly 0.00000 before these terms existed, so selection could never learn
    anything about them.
    """

    name = "heuristic"

    # Weights chosen so no single term can dominate the others.
    W_BURIAL = 1.0
    W_PACKING = 0.6
    W_SS = 0.8
    W_AGGREGATION = 1.2
    W_CHARGE = 0.5
    W_BB_ENTROPY = 0.5
    W_CAPPING = 0.4
    W_LIABILITY = 0.6
    W_INTERACTION = 0.5

    #: Aromatic ring-centre separation admitting a stacking interaction.
    STACK_MIN, STACK_MAX = 4.5, 7.5
    #: Cation to ring-centre separation for a cation-pi contact.
    CATION_PI_MAX = 6.5

    IDEAL_CORE_VOLUME = 165.0        # roughly leucine
    PATCH_RADIUS = 8.0

    # Backbone conformational entropy, in arbitrary units consistent with the
    # other terms. Glycine is the most flexible residue and pays for it in the
    # unfolded state; proline is the most restricted and is rewarded, but only
    # in loops -- inside a helix or strand it is a breaker and the propensity
    # term penalises it there.
    GLY_ENTROPY_COST = 1.0
    PRO_LOOP_BONUS = 0.8

    N_CAP_GOOD = frozenset("STDN")
    C_CAP_GOOD = frozenset("GN")

    def score(self, ctx: DesignContext, sequence: str) -> ScoreBreakdown:
        if len(sequence) != len(ctx):
            raise ValueError(f"sequence length {len(sequence)} != {len(ctx)} residues")

        # One pass over the sequence produces every property vector the terms
        # need; the terms themselves are then pure array arithmetic.
        idx = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        hydro = _LOOKUP["hydropathy"][idx]
        volume = _LOOKUP["volume"][idx]
        helix = _LOOKUP["helix"][idx]
        sheet = _LOOKUP["sheet"][idx]
        aggreg = _LOOKUP["aggregation"][idx]
        charge = _LOOKUP["charge"][idx]

        terms = {
            "burial": self._burial(ctx, hydro),
            "packing": self._packing(ctx, volume),
            "ss_propensity": self._ss(ctx, helix, sheet),
            "aggregation": self._aggregation(ctx, aggreg),
            "net_charge": self._charge(charge),
            "bb_entropy": self._bb_entropy(ctx, idx),
            "capping": self._capping(ctx, idx),
            "liabilities": self._liabilities(ctx, sequence),
            "interactions": self._interactions(ctx, sequence),
        }
        return ScoreBreakdown(total=float(sum(terms.values())), terms=terms,
                              n_residues=len(sequence))

    # ------------------------------------------------------------------ terms

    def _burial(self, ctx: DesignContext, hydro: np.ndarray) -> float:
        """Hydrophobicity should track the environment, whichever way it points."""
        want = np.where(ctx.is_core, 1.0, np.where(ctx.is_surface, -0.6, 0.2))
        if ctx.is_membrane:
            # Inverted inside the bilayer: lipid exposure wants hydrophobic,
            # the bundle interior tolerates (and uses) polar.
            lipid = ctx.in_lipid_core
            want = np.where(lipid, np.where(ctx.is_surface, 1.0, -0.2), want)
        return self.W_BURIAL * float(-(hydro * want).sum()) / max(len(hydro), 1)

    def _packing(self, ctx: DesignContext, volume: np.ndarray) -> float:
        core = ctx.is_core
        if not core.any():
            return 0.0
        dev = np.abs(volume[core] - self.IDEAL_CORE_VOLUME) / 100.0
        return self.W_PACKING * float(dev.mean())

    def _ss(self, ctx: DesignContext, helix: np.ndarray, sheet: np.ndarray) -> float:
        score = -(helix[ctx.is_helix].sum() + sheet[ctx.is_strand].sum())
        return self.W_SS * float(score) / max(len(helix), 1)

    def _aggregation(self, ctx: DesignContext, aggreg: np.ndarray) -> float:
        """Contiguous aggregation-prone surface, measured in 3D not in sequence.

        Aggregation nucleates on spatial patches, so neighbours are found by
        distance rather than by sequence position. The neighbour mask depends
        only on the backbone, so it is computed once per structure and this
        term reduces to a matrix-vector product.
        """
        env_idx, rows, cols = ctx.patch_neighbors(self.PATCH_RADIUS)
        n_env = env_idx.size
        if n_env == 0:
            return 0.0

        prop = aggreg[env_idx]
        # Signed neighbourhood sum: aggregation-prone residues build the patch,
        # charged gatekeepers subtract from it. A well-policed patch is free.
        neighbourhood = np.bincount(rows, weights=prop[cols], minlength=n_env)
        np.clip(neighbourhood, 0.0, None, out=neighbourhood)
        seeds = np.where(prop > 0.0, prop, 0.0)
        patch = float((seeds * neighbourhood).sum())
        return self.W_AGGREGATION * patch / n_env

    def _bb_entropy(self, ctx: DesignContext, idx: np.ndarray) -> float:
        """Unfolded-state backbone entropy.

        Glycine samples far more backbone conformations than any other
        residue, so every glycine raises the entropy of the unfolded state and
        destabilises the fold. Proline does the reverse. Rigidifying a loop is
        real stabilization and the objective has to be able to see it.
        """
        is_gly = idx == ord("G")
        is_pro = idx == ord("P")
        cost = self.GLY_ENTROPY_COST * float(is_gly.sum())
        gain = self.PRO_LOOP_BONUS * float((is_pro & ctx.is_loop).sum())
        return self.W_BB_ENTROPY * (cost - gain) / max(len(idx), 1)

    def _capping(self, ctx: DesignContext, idx: np.ndarray) -> float:
        """Reward satisfied helix N- and C-caps."""
        n_cap, c_cap = ctx.cap_positions
        if not (n_cap.any() or c_cap.any()):
            return 0.0
        good_n = np.isin(idx, [ord(a) for a in self.N_CAP_GOOD])
        good_c = np.isin(idx, [ord(a) for a in self.C_CAP_GOOD])
        satisfied = float((n_cap & good_n).sum() + (c_cap & good_c).sum())
        return -self.W_CAPPING * satisfied / max(len(idx), 1)

    def _liabilities(self, ctx: DesignContext, seq: str) -> float:
        """Chemical degradation motifs, weighted by exposure.

        A buried motif is far less reactive than an exposed one -- the solvent
        has to reach it -- so each hit is scaled by how exposed the residue is.
        """
        n = len(seq)
        if n < 2:
            return 0.0
        exposure = np.where(ctx.is_surface, 1.0,
                            np.where(ctx.is_core, 0.15, 0.5))
        total = 0.0

        for i in range(n - 1):
            pair = seq[i:i + 2]
            total += DEAMIDATION.get(pair, 0.0) * exposure[i]
            total += ISOMERISATION.get(pair, 0.0) * exposure[i]

        # N-linked glycosylation sequon: Asn, any residue but proline, Ser/Thr.
        for i in range(n - 2):
            if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
                total += 1.0 * exposure[i]

        # Unpaired cysteine: free thiols oxidise, scramble and cross-link.
        cys = [p for p in ctx.positions if seq[p - 1] == "C"]
        if cys:
            paired = set()
            for a_idx, i in enumerate(cys):
                for j in cys[a_idx + 1:]:
                    if ctx.cb_dist[i - 1, j - 1] <= 4.5:
                        paired.update((i, j))
            total += 1.2 * sum(exposure[p - 1] for p in cys if p not in paired)

        # Exposed methionine oxidises readily.
        total += 0.5 * sum(exposure[p - 1] for p in ctx.positions
                           if seq[p - 1] == "M" and ctx.is_surface[p - 1])

        return self.W_LIABILITY * total / n

    def _interactions(self, ctx: DesignContext, seq: str) -> float:
        """Aromatic stacking and cation-pi contacts.

        Both are real packing energy that a per-residue hydrophobicity term
        cannot represent: they depend on which *pair* of residues sit near each
        other, not on either one alone.
        """
        n = len(seq)
        aromatic = [p for p in ctx.positions if seq[p - 1] in "FWY"]
        cations = [p for p in ctx.positions if seq[p - 1] in "KR"]
        if not aromatic:
            return 0.0

        reward = 0.0
        for a_idx, i in enumerate(aromatic):
            for j in aromatic[a_idx + 1:]:
                d = ctx.cb_dist[i - 1, j - 1]
                if self.STACK_MIN <= d <= self.STACK_MAX:
                    # Buried stacks are worth more; solvent competes at the surface.
                    reward += 1.0 if ctx.is_core[i - 1] or ctx.is_core[j - 1] else 0.4
            for j in cations:
                if ctx.cb_dist[i - 1, j - 1] <= self.CATION_PI_MAX:
                    reward += 0.6

        return -self.W_INTERACTION * reward / max(n, 1)

    def _charge(self, charge: np.ndarray) -> float:
        per_res = abs(float(charge.sum())) / max(len(charge), 1)
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
