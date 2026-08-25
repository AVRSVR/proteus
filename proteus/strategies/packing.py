"""Pairwise interactions and beta-sheet architecture.

These mechanisms are invisible to any per-residue rule, because they depend on
which *pair* of residues happen to sit near each other. Aromatic stacking,
cation-pi contacts and buried hydrogen bonds are all real packing energy that a
hydrophobicity scale cannot represent, and the beta-sheet strategies address a
problem that only exists because of how sheets propagate.
"""

from __future__ import annotations

import numpy as np

from ..context import DesignContext
from ..proposals import AROMATIC, Proposal
from .base import MEMBRANE, SOLUBLE, PairStrategy, Strategy, register

AROMATICS = frozenset("FWY")
CATIONS = frozenset("KR")
#: Polar residues whose buried hydrogen-bonding capacity must be satisfied.
BURIED_POLAR = frozenset("STNQDEKRH")
HYDROPHOBIC_CORE = frozenset("AVLIMF")

# Ring-centre separations. Measured from CB as a proxy, so the windows are
# wider than true centroid criteria would be.
STACK_MIN, STACK_MAX = 4.5, 7.5
CATION_PI_MAX = 6.5
HBOND_MAX = 6.5


@register
class AromaticCluster(PairStrategy):
    name = "aromatic_cluster"
    mechanism = ("Build aromatic clusters. Stacked rings contribute real "
                 "packing energy beyond simple hydrophobic burial, and "
                 "thermophile proteins show markedly larger aromatic clusters "
                 "than their mesophile equivalents. A position already sitting "
                 "at stacking distance from an aromatic is the cheapest place "
                 "to add another.")
    applies_to = frozenset({SOLUBLE})
    MAX_PAIRS = 2

    def candidate_pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        """Buried non-aromatic positions adjacent to an existing aromatic."""
        existing = [p for p in ctx.positions if ctx.aa(p) in AROMATICS]
        if not existing:
            return []
        out = []
        for p in ctx.designable:
            if ctx.aa(p) in AROMATICS or ctx.layer(p) == "surface":
                continue
            row = ctx.cb_dist[p - 1]
            for q in existing:
                if q != p and STACK_MIN <= row[q - 1] <= STACK_MAX:
                    out.append((p, q))
                    break
        return out

    def propose(self, ctx, positions, rng):
        pairs = self.candidate_pairs(ctx)
        if not pairs:
            return []
        rng.shuffle(pairs)
        out = []
        for p, partner in self.select_pairs(pairs):
            d = ctx.cb_dist[p - 1, partner - 1]
            out.append(Proposal(
                p, AROMATIC, self.name,
                f"{d:.1f}A from aromatic {ctx.label(partner)}; stacking "
                f"distance",
            ))
        return out

    def select_pairs(self, pairs):
        """Only the first element is designed, so partners may repeat."""
        chosen, used = [], set()
        for p, q in pairs:
            if p in used:
                continue
            chosen.append((p, q))
            used.add(p)
            if len(chosen) >= self.MAX_PAIRS:
                break
        return chosen

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted({p for p, _ in self.select_pairs(self.candidate_pairs(ctx))})


@register
class CationPi(Strategy):
    name = "cation_pi"
    mechanism = ("Complete cation-pi contacts. An arginine or lysine stacked "
                 "over an aromatic ring face is worth several kcal/mol -- "
                 "comparable to a salt bridge and less sensitive to solvent. "
                 "These are common in thermophile proteins and in interfaces "
                 "that have to hold under load.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    #: Distance window, not merely an upper bound. Below this the sidechains
    #: clash; a cation sitting further away is simply a nearby charge.
    MIN_DISTANCE = 4.5
    #: Cap on sites reported and acted on -- see :meth:`_sites`.
    MAX_SITES = 4

    def _sites(self, ctx: DesignContext) -> dict[int, int]:
        """Positions that could place a cation over an aromatic ring face.

        This gate is *necessarily approximate* and is written to admit that.
        A cation-pi contact requires the charge to sit over the ring face at
        about 4-6 A from its centroid, but Proteus carries only backbone atoms
        and CB, so neither the ring centroid nor its normal can be computed.
        Distance between CB atoms is a weak proxy for both.

        An upper bound alone flagged 80% of an aromatic-rich design and a third
        of a 387-residue protein, which tells the selector nothing. Narrowing
        the window and requiring both partners partially buried helped but not
        enough. So the strategy is additionally capped: it proposes only the
        few best-scoring candidates rather than every position that might
        qualify. Better to under-claim a mechanism that cannot be verified from
        the available coordinates than to flood the leaderboard with guesses.
        """
        aromatic = [p for p in ctx.positions
                    if ctx.aa(p) in AROMATICS and ctx.layer(p) != "surface"]
        if not aromatic:
            return {}

        scored: list[tuple[float, int, int]] = []
        for p in ctx.designable:
            if ctx.aa(p) in CATIONS or ctx.layer(p) != "boundary":
                continue
            row = ctx.cb_dist[p - 1]
            best = None
            for q in aromatic:
                d = row[q - 1]
                if q != p and self.MIN_DISTANCE <= d <= CATION_PI_MAX:
                    if best is None or d < best[0]:
                        best = (d, q)
            if best is not None:
                scored.append((best[0], p, best[1]))

        scored.sort()
        return {p: q for _d, p, q in scored[: self.MAX_SITES]}

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted(self._sites(ctx))

    def propose(self, ctx, positions, rng):
        sites = self._sites(ctx)
        out = []
        for p in positions:
            q = sites.get(p)
            if q is None:
                continue
            out.append(Proposal(
                p, CATIONS, self.name,
                f"{ctx.cb_dist[p - 1, q - 1]:.1f}A from aromatic "
                f"{ctx.label(q)}; cation-pi geometry",
            ))
        return out


@register
class BuriedUnsatisfiedPolar(Strategy):
    name = "buried_unsatisfied_polar"
    mechanism = ("Remove buried polar groups with no partner. Burying a "
                 "hydrogen-bond donor or acceptor costs its hydration energy, "
                 "and that is only repaid if something inside answers it. An "
                 "unsatisfied buried polar is one of the most destabilising "
                 "things a design can contain, and it is a characteristic "
                 "failure of sequence design that optimises burial alone.")
    applies_to = frozenset({SOLUBLE})

    def _unsatisfied(self, ctx: DesignContext) -> list[int]:
        out = []
        for p in ctx.designable:
            if ctx.layer(p) != "core" or ctx.aa(p) not in BURIED_POLAR:
                continue
            row = ctx.cb_dist[p - 1]
            partners = [
                q for q in ctx.positions
                if q != p and row[q - 1] <= HBOND_MAX
                and ctx.aa(q) in BURIED_POLAR
            ]
            if not partners:
                out.append(p)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return self._unsatisfied(ctx)

    def propose(self, ctx, positions, rng):
        return [Proposal(p, HYDROPHOBIC_CORE, self.name,
                         f"buried {ctx.aa(p)} at {ctx.label(p)} has no "
                         f"hydrogen-bond partner within {HBOND_MAX:.0f}A")
                for p in positions]


@register
class BetaEdgeProtection(Strategy):
    name = "beta_edge_protection"
    mechanism = ("Protect exposed beta-sheet edges. A sheet propagates by "
                 "hydrogen bonding along its edge, so a free edge strand is an "
                 "invitation for another copy of the protein to join -- this is "
                 "the amyloid growth mechanism. Nature blocks it deliberately: "
                 "edge strands carry charged residues, prolines and inward "
                 "kinks that make an incoming strand pay a penalty. Richardson "
                 "and Richardson called this negative design.")
    applies_to = frozenset({SOLUBLE})

    #: Charges repel an approaching strand; proline cannot donate the backbone
    #: hydrogen bond that propagation requires.
    BLOCKERS = frozenset("KRDEP")
    NEIGHBOUR_RADIUS = 6.0

    def _edge_strands(self, ctx: DesignContext) -> list[int]:
        """Strand positions with sheet neighbours on only one side.

        An interior strand is flanked by partners on both sides; an edge strand
        is not, and only the edge can nucleate polymerisation.
        """
        strand_positions = [p for p in ctx.positions if ctx.ss_at(p) == "E"]
        if len(strand_positions) < 6:
            return []

        out = []
        for p in strand_positions:
            if p not in ctx.designable or ctx.layer(p) != "surface":
                continue
            row = ctx.cb_dist[p - 1]
            # Neighbouring strand residues that are not sequence-adjacent.
            partners = [q for q in strand_positions
                        if abs(q - p) > 2 and row[q - 1] <= self.NEIGHBOUR_RADIUS]
            if len(partners) <= 1 and ctx.aa(p) not in self.BLOCKERS:
                out.append(p)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return self._edge_strands(ctx)

    def propose(self, ctx, positions, rng):
        return [Proposal(p, frozenset("KRDE"), self.name,
                         f"{ctx.label(p)} sits on an exposed sheet edge; "
                         f"a charge blocks strand propagation")
                for p in positions]


@register
class BetaTurn(Strategy):
    name = "beta_turn"
    mechanism = ("Optimise beta turns. A tight two-residue turn connecting "
                 "antiparallel strands has strong positional preferences: "
                 "glycine and asparagine at the position needing a positive "
                 "backbone angle, proline where the chain must change "
                 "direction. Getting these right shortens the loop and "
                 "pre-organises the hairpin.")
    applies_to = frozenset({SOLUBLE})

    def _turn_sites(self, ctx: DesignContext) -> dict[int, str]:
        """Short loops flanked by strands on both sides."""
        sites: dict[int, str] = {}
        from ..geometry import dihedrals
        phi, _ = dihedrals(ctx.structure)

        for start, end in ctx.ss_segments("L"):
            length = end - start + 1
            if not 2 <= length <= 4:
                continue
            before = start - 1
            after = end + 1
            if before < 1 or after > len(ctx):
                continue
            if ctx.ss_at(before) != "E" or ctx.ss_at(after) != "E":
                continue
            for p in range(start, end + 1):
                if p not in ctx.designable:
                    continue
                val = phi[p - 1]
                if not np.isnan(val) and val > 0:
                    # Positive phi: only glycine and asparagine manage this well.
                    if ctx.aa(p) not in "GN":
                        sites[p] = "positive-phi"
        return sites

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted(self._turn_sites(ctx))

    def propose(self, ctx, positions, rng):
        return [Proposal(p, frozenset("GN"), self.name,
                         f"{ctx.label(p)} is a positive-phi position in a "
                         f"beta turn")
                for p in positions]
