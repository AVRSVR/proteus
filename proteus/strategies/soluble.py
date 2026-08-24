"""Stabilization mechanisms for soluble, globular proteins."""

from __future__ import annotations

import random

import numpy as np

from ..context import DesignContext
from ..geometry import dihedrals
from ..proposals import AROMATIC, HYDROPHOBIC, NEGATIVE, POSITIVE, Proposal
from .base import MEMBRANE, SOLUBLE, Strategy, register

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

# Salt-bridge reach: CB-CB range over which two charged sidechains can pair.
SB_CB_MIN, SB_CB_MAX = 4.0, 8.0
SB_MIN_SEQSEP = 3


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
    mechanism = ("Close packing defects. Positions that sit in the core but "
                 "have unusually few neighbours for their layer indicate a "
                 "cavity; aromatics are the largest way to fill one.")
    applies_to = frozenset({SOLUBLE})
    conflicts_with = frozenset({"core_packing"})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        # Marginally-core positions: buried enough to matter, loosely packed.
        lo, hi = ctx.core_cutoff, ctx.core_cutoff + 1.5
        return [p for p in ctx.designable
                if lo <= ctx.burial(p) < hi and ctx.aa(p) not in AROMATIC]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, AROMATIC, self.name,
                         f"loosely packed core position (neighbors {ctx.burial(p):.1f})")
                for p in positions]


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
class SaltBridge(Strategy):
    name = "salt_bridge"
    mechanism = ("Add favourable electrostatics. Pairs of surface positions "
                 "whose sidechains can reach each other are given "
                 "complementary charges, the mechanism thermophilic proteins "
                 "use most heavily.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        cb, ca = ctx.cb_dist, ctx.ca_dist
        cand = []
        eligible = [p for p in ctx.designable
                    if ctx.layer(p) in ("surface", "boundary")]
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
                if ca[i - 1, j - 1] > 12.0:
                    continue
                cand.append((i, j))
        return cand

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted({x for pair in self._pairs(ctx) for x in pair})

    def propose(self, ctx, positions, rng):
        pairs = [pr for pr in self._pairs(ctx)
                 if pr[0] in positions or pr[1] in positions]
        if not pairs:
            return []
        rng.shuffle(pairs)
        used: set[int] = set()
        out = []
        for i, j in pairs:
            if i in used or j in used:
                continue
            used |= {i, j}
            d = ctx.cb_dist[i - 1, j - 1]
            # Weighted above 1.0: a pair is only meaningful if both halves land.
            out.append(Proposal(i, NEGATIVE, self.name,
                                f"CB-CB {d:.1f}A salt bridge with {ctx.label(j)}",
                                weight=1.5))
            out.append(Proposal(j, POSITIVE, self.name,
                                f"CB-CB {d:.1f}A salt bridge with {ctx.label(i)}",
                                weight=1.5))
            if len(out) >= 6:
                break
        return out


@register
class Disulfide(Strategy):
    name = "disulfide"
    mechanism = ("Cross-link the fold. A disulfide lowers the entropy of the "
                 "unfolded state, which is why secreted proteins rely on them "
                 "-- but only where the geometry genuinely supports one.")
    applies_to = frozenset({SOLUBLE})

    def _pairs(self, ctx: DesignContext) -> list[tuple[int, int]]:
        """Position pairs whose backbone geometry can actually host an SS bond.

        The prototype picked two random surface residues and mutated both to
        cysteine. Random pairs essentially never satisfy disulfide geometry, so
        that produced two free cysteines -- an oxidation and aggregation
        liability -- rather than a crosslink. Here the geometry is the filter.
        """
        cb, ca = ctx.cb_dist, ctx.ca_dist
        out = []
        pos = ctx.designable
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

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted({x for pair in self._pairs(ctx) for x in pair})

    def propose(self, ctx, positions, rng):
        pairs = self._pairs(ctx)
        if not pairs:
            return []
        rng.shuffle(pairs)
        i, j = pairs[0]
        why = (f"CB-CB {ctx.cb_dist[i - 1, j - 1]:.1f}A, CA-CA "
               f"{ctx.ca_dist[i - 1, j - 1]:.1f}A: disulfide geometry satisfied")
        cys = frozenset("C")
        return [Proposal(i, cys, self.name, why, weight=2.0),
                Proposal(j, cys, self.name, why, weight=2.0)]


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
