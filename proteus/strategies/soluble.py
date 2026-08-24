"""Stabilization mechanisms for soluble, globular proteins."""

from __future__ import annotations

import random

import numpy as np

from ..context import DesignContext
from ..geometry import dihedrals
from ..proposals import AROMATIC, HYDROPHOBIC, NEGATIVE, POSITIVE, Proposal
from .base import MEMBRANE, SOLUBLE, PairStrategy, Strategy, register

# Residues too small to fill a core position well.
UNDERSIZED = frozenset("AGSCT")
LARGE_HYDROPHOBIC = frozenset("LIMFVWY")
POLAR_SURFACE = frozenset("DEKRNQSTH")
HELIX_GOOD_CORE = frozenset("ALMIF")
HELIX_GOOD_SURFACE = frozenset("AEQKRL")
BETA_GOOD_CORE = frozenset("VIFLYW")
BETA_GOOD_SURFACE = frozenset("TVIYKRE")

# Ideal disulfide geometry. A real SS bond needs the CB atoms this close;
# anything else is not a crosslink, it is two free cysteines.
SS_CB_MIN, SS_CB_MAX = 3.0, 4.5
SS_CA_MIN, SS_CA_MAX = 4.0, 6.5
SS_MIN_SEQSEP = 4

# Salt-bridge geometry.
#
# Distance between CB atoms alone is far too permissive -- on a typical fold
# nearly every adjacent surface pair falls inside any reasonable window, which
# flagged 90-95% of all residues and told the selector nothing.
#
# Requiring the two CA->CB vectors to point at each other is also wrong, and
# fails in the opposite direction: for two surface residues on a helix face,
# both vectors point outward from the backbone and are close to parallel, so
# that test rejected every real pair. The bridge is not made by the CB atoms,
# it is made by sidechains reaching laterally.
#
# So the criterion models the reach: project a notional charged tip out from
# each CB along its sidechain direction and require the two tips to land within
# hydrogen-bonding range of one another.
SB_CB_MIN, SB_CB_MAX = 4.0, 9.0
SB_MIN_SEQSEP = 3
SB_TIP_REACH = 3.0        # CB to charged group, averaged over Asp/Glu/Lys/Arg
SB_TIP_MAX = 5.0          # tip-tip distance admitting a salt bridge


@register
class CorePacking(Strategy):
    name = "core_packing"
    mechanism = ("Fill the hydrophobic core. Buried positions carrying small "
                 "residues leave voids; larger hydrophobics recover the "
                 "van der Waals contacts that hold the fold together.")
    applies_to = frozenset({SOLUBLE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.layer(p) == "core" and ctx.aa(p) in UNDERSIZED]

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            allowed = LARGE_HYDROPHOBIC
            if ctx.ss_at(p) == "H":
                allowed = allowed & HELIX_GOOD_CORE or LARGE_HYDROPHOBIC
            out.append(Proposal(
                p, allowed, self.name,
                f"buried {ctx.aa(p)} (neighbors {ctx.burial(p):.1f}) leaves a void",
            ))
        return out


@register
class CavityFill(Strategy):
    name = "cavity_fill"
    mechanism = ("Close packing defects. Deeply buried positions whose local "
                 "neighbourhood carries less sidechain volume than the core "
                 "average indicate a void; aromatics are the largest thing "
                 "available to fill one.")
    applies_to = frozenset({SOLUBLE})
    conflicts_with = frozenset({"core_packing"})

    #: Only fire this far above the core threshold. Marginally-buried
    #: positions are boundary-like, and putting a bulky aromatic there
    #: exposes ring surface and costs more in aggregation than it gains in
    #: packing -- the original version targeted exactly those positions and
    #: made the score consistently worse.
    DEPTH_MARGIN = 1.0
    VOLUME_RADIUS = 8.0

    def _volume_deficit(self, ctx: DesignContext) -> dict[int, float]:
        """Local sidechain volume relative to the core average.

        A genuine cavity is not "few neighbours" -- it is neighbours that do
        not fill the space they enclose. Summing neighbour volume in a shell
        distinguishes the two.
        """
        from ..scoring import VOLUME

        core = [p for p in ctx.positions
                if ctx.burial(p) >= ctx.core_cutoff + self.DEPTH_MARGIN]
        if len(core) < 3:
            return {}

        local: dict[int, float] = {}
        for p in core:
            row = ctx.cb_dist[p - 1]
            local[p] = sum(VOLUME.get(ctx.aa(q), 130.0)
                           for q in ctx.positions
                           if q != p and row[q - 1] <= self.VOLUME_RADIUS)

        values = sorted(local.values())
        median = values[len(values) // 2]
        return {p: median - v for p, v in local.items() if v < median}

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in self._volume_deficit(ctx)
                if p in ctx.designable and ctx.aa(p) not in AROMATIC]

    def propose(self, ctx, positions, rng):
        deficit = self._volume_deficit(ctx)
        out = []
        for p in positions:
            # Tryptophan only where burial is deep enough to bury the whole
            # ring; otherwise the aromatic surface becomes an exposed patch.
            allowed = AROMATIC if ctx.burial(p) >= ctx.core_cutoff + 2.0 else frozenset("FY")
            out.append(Proposal(
                p, allowed, self.name,
                f"buried (neighbors {ctx.burial(p):.1f}) with a local volume "
                f"deficit of {deficit.get(p, 0.0):.0f} A^3",
            ))
        return out


@register
class SurfaceDepolarize(Strategy):
    name = "surface_depolarize"
    mechanism = ("Reduce aggregation. Solvent-exposed hydrophobic residues "
                 "nucleate intermolecular contacts; replacing them with polar "
                 "or charged residues raises solubility without touching the "
                 "core that determines the fold.")
    applies_to = frozenset({SOLUBLE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.layer(p) == "surface" and ctx.aa(p) in HYDROPHOBIC]

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            allowed = POLAR_SURFACE
            if ctx.ss_at(p) == "H":
                allowed = allowed & HELIX_GOOD_SURFACE or HELIX_GOOD_SURFACE
            elif ctx.ss_at(p) == "E":
                allowed = allowed & BETA_GOOD_SURFACE or BETA_GOOD_SURFACE
            out.append(Proposal(
                p, allowed, self.name,
                f"exposed {ctx.aa(p)} is an aggregation liability",
            ))
        return out


@register
class SaltBridge(PairStrategy):
    name = "salt_bridge"
    mechanism = ("Add favourable electrostatics. Pairs of surface positions "
                 "whose sidechains can reach each other are given "
                 "complementary charges, the mechanism thermophilic proteins "
                 "use most heavily.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def candidate_pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        """Surface pairs whose charged groups could reach each other.

        Selectivity comes from the tip-reach test rather than from CB-CB
        distance; see the geometry constants above for why. Pairs that already
        carry complementary charges are skipped -- that bridge exists.
        """
        cb = ctx.cb_dist
        coords_cb = ctx.structure.coords("cb")
        coords_ca = ctx.structure.coords("ca")

        direction = coords_cb - coords_ca
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tips = coords_cb + SB_TIP_REACH * (direction / norms)

        # Charged residues need solvation, so restrict to genuine surface.
        eligible = [p for p in ctx.designable if ctx.layer(p) == "surface"]
        cand = []
        for a_idx, i in enumerate(eligible):
            for j in eligible[a_idx + 1:]:
                if abs(i - j) < SB_MIN_SEQSEP:
                    continue
                if not (SB_CB_MIN <= cb[i - 1, j - 1] <= SB_CB_MAX):
                    continue
                # Within a helix only i,i+3 and i,i+4 present the right geometry.
                if ctx.ss_at(i) == "H" and ctx.ss_at(j) == "H":
                    if abs(i - j) not in (3, 4):
                        continue
                # Can the two charged groups actually reach each other?
                if float(np.linalg.norm(tips[i - 1] - tips[j - 1])) > SB_TIP_MAX:
                    continue
                # Skip pairs that already form a complementary bridge.
                residues = {ctx.aa(i), ctx.aa(j)}
                if residues & NEGATIVE and residues & POSITIVE:
                    continue
                cand.append((i, j))
        return cand

    def propose(self, ctx, positions, rng):
        pairs = self.candidate_pairs(ctx)
        if not pairs:
            return []
        rng.shuffle(pairs)
        out = []
        for i, j in self.select_pairs(pairs):
            d = ctx.cb_dist[i - 1, j - 1]
            # Weighted above 1.0: a pair is only meaningful if both halves land.
            out.append(Proposal(i, NEGATIVE, self.name,
                                f"CB-CB {d:.1f}A salt bridge with {ctx.label(j)}",
                                weight=1.5))
            out.append(Proposal(j, POSITIVE, self.name,
                                f"CB-CB {d:.1f}A salt bridge with {ctx.label(i)}",
                                weight=1.5))
        return out


@register
class Disulfide(PairStrategy):
    name = "disulfide"
    mechanism = ("Cross-link the fold. A disulfide lowers the entropy of the "
                 "unfolded state, which is why secreted proteins rely on them "
                 "-- but only where the geometry genuinely supports one.")
    applies_to = frozenset({SOLUBLE})

    def _existing(self, ctx: DesignContext) -> set[int]:
        """Positions already participating in a disulfide.

        A cysteine pair that already satisfies the geometry is a bond that
        exists. Proposing to "add" it again wastes budget and, on a
        disulfide-rich protein such as gp120, dominates the candidate list.
        """
        cys = [p for p in ctx.positions if ctx.aa(p) == "C"]
        bonded = set()
        for a_idx, i in enumerate(cys):
            for j in cys[a_idx + 1:]:
                if SS_CB_MIN <= ctx.cb_dist[i - 1, j - 1] <= SS_CB_MAX:
                    bonded.update((i, j))
        return bonded

    def candidate_pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        """Position pairs whose backbone geometry can actually host an SS bond.

        The prototype picked two random surface residues and mutated both to
        cysteine. Random pairs essentially never satisfy disulfide geometry, so
        that produced two free cysteines -- an oxidation and aggregation
        liability -- rather than a crosslink. Here the geometry is the filter.
        """
        cb, ca = ctx.cb_dist, ctx.ca_dist
        existing = self._existing(ctx)
        out = []
        pos = [p for p in ctx.designable if p not in existing]
        for a_idx, i in enumerate(pos):
            for j in pos[a_idx + 1:]:
                if abs(i - j) < SS_MIN_SEQSEP:
                    continue
                if not (SS_CB_MIN <= cb[i - 1, j - 1] <= SS_CB_MAX):
                    continue
                if not (SS_CA_MIN <= ca[i - 1, j - 1] <= SS_CA_MAX):
                    continue
                out.append((i, j))
        return out

    def propose(self, ctx, positions, rng):
        pairs = self.candidate_pairs(ctx)
        if not pairs:
            return []
        rng.shuffle(pairs)
        cys = frozenset("C")
        out = []
        for i, j in self.select_pairs(pairs):
            why = (f"CB-CB {ctx.cb_dist[i - 1, j - 1]:.1f}A, CA-CA "
                   f"{ctx.ca_dist[i - 1, j - 1]:.1f}A: disulfide geometry satisfied")
            out.append(Proposal(i, cys, self.name, why, weight=2.0))
            out.append(Proposal(j, cys, self.name, why, weight=2.0))
        return out


@register
class HelixCapping(Strategy):
    name = "helix_capping"
    mechanism = ("Cap helix termini. The first backbone NH groups of a helix "
                 "have no intrahelical partner; an N-cap sidechain satisfies "
                 "them, and glycine at the C-cap adopts the left-handed "
                 "conformation that terminates the helix cleanly.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    N_CAP = frozenset("STDN")
    C_CAP = frozenset("GN")

    def _sites(self, ctx: DesignContext) -> list[tuple[int, str]]:
        sites = []
        for start, end in ctx.ss_segments("H"):
            if end - start + 1 < 5:
                continue
            n_cap, c_cap = start - 1, end + 1
            if n_cap >= 1 and n_cap in ctx.designable and ctx.aa(n_cap) not in self.N_CAP:
                sites.append((n_cap, "N"))
            if c_cap <= len(ctx) and c_cap in ctx.designable and ctx.aa(c_cap) not in self.C_CAP:
                sites.append((c_cap, "C"))
        return sites

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p, _ in self._sites(ctx)]

    def propose(self, ctx, positions, rng):
        out = []
        for p, kind in self._sites(ctx):
            if p not in positions:
                continue
            allowed = self.N_CAP if kind == "N" else self.C_CAP
            out.append(Proposal(p, allowed, self.name,
                                f"{kind}-cap of helix at {ctx.label(p)}"))
        return out


@register
class LoopRigidify(Strategy):
    name = "loop_rigidify"
    mechanism = ("Restrict backbone entropy in loops. Proline removes a "
                 "backbone degree of freedom and glycine adds one, so "
                 "replacing loop glycines and introducing proline where the "
                 "backbone already sits in its allowed region pre-pays part "
                 "of the folding entropy cost.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _proline_sites(self, ctx: DesignContext) -> list[int]:
        """Loop positions whose existing phi is compatible with proline.

        The proline ring pins phi near -60 degrees and proline cannot donate a
        backbone hydrogen bond. Placing it anywhere else -- at every fourth
        residue, say -- breaks helices and strains loops. So the backbone has
        to already be in the proline region before we propose it.
        """
        phi, _ = dihedrals(ctx.structure)
        breaks = ctx.chain_breaks
        out = []
        for p in ctx.designable:
            if ctx.ss_at(p) != "L" or ctx.aa(p) == "P":
                continue
            if p <= 1 or (p - 1) in breaks or p in breaks:
                continue
            if ctx.aa(p - 1) == "P":
                continue
            val = phi[p - 1]
            if np.isnan(val) or not (-90.0 <= val <= -40.0):
                continue
            out.append(p)
        return out

    def _glycine_sites(self, ctx: DesignContext) -> list[int]:
        """Loop glycines that are not conformationally required.

        A glycine sitting at positive phi is there for a reason -- no other
        residue can adopt that backbone -- so it is left alone.
        """
        phi, _ = dihedrals(ctx.structure)
        return [p for p in ctx.designable
                if ctx.aa(p) == "G" and ctx.ss_at(p) == "L"
                and not np.isnan(phi[p - 1]) and phi[p - 1] < 0.0]

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted(set(self._proline_sites(ctx)) | set(self._glycine_sites(ctx)))

    def propose(self, ctx, positions, rng):
        pro = set(self._proline_sites(ctx))
        gly = set(self._glycine_sites(ctx))
        out = []
        for p in positions:
            if p in pro:
                out.append(Proposal(p, frozenset("P"), self.name,
                                    f"loop phi at {ctx.label(p)} admits proline"))
            elif p in gly:
                out.append(Proposal(p, frozenset("ASTND"), self.name,
                                    "non-essential loop glycine (negative phi)"))
        return out


@register
class HelixPropensity(Strategy):
    name = "helix_propensity"
    mechanism = ("Match residues to the secondary structure they sit in. "
                 "Beta-branched and helix-breaking residues inside a helix "
                 "cost stability that costs nothing to recover.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    POOR = frozenset("GPVIT")

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.ss_at(p) == "H" and ctx.aa(p) in self.POOR]

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            allowed = HELIX_GOOD_CORE if ctx.layer(p) == "core" else HELIX_GOOD_SURFACE
            out.append(Proposal(p, allowed, self.name,
                                f"{ctx.aa(p)} is a poor helix former"))
        return out


@register
class BetaPropensity(Strategy):
    name = "beta_propensity"
    mechanism = ("Strengthen strands. Beta-branched and aromatic residues "
                 "favour the extended backbone that sheets require.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    POOR = frozenset("GPDNS")

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.ss_at(p) == "E" and ctx.aa(p) in self.POOR]

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            allowed = BETA_GOOD_CORE if ctx.layer(p) == "core" else BETA_GOOD_SURFACE
            out.append(Proposal(p, allowed, self.name,
                                f"{ctx.aa(p)} is a poor strand former"))
        return out
