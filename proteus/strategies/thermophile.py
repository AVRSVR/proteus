"""Adaptations that let proteins work near boiling, and how to copy them.

Comparing thermophile proteins against their mesophile orthologues gives an
unusually clean natural experiment: same fold, same function, different
operating temperature. The compositional differences that come out of that
comparison are consistent enough to copy, and they are not what a naive
stability heuristic would guess. Thermophiles do not simply bury more
hydrophobics -- the core is already packed in both. They change the *surface*:
more charged residues, arranged into networks rather than isolated pairs, with
arginine favoured over lysine and the thermolabile amides depleted.

Each strategy here implements one of those observations.
"""

from __future__ import annotations

import numpy as np

from ..context import DesignContext
from ..proposals import NEGATIVE, POSITIVE, Proposal
from .base import MEMBRANE, SOLUBLE, Strategy, register

#: Polar but uncharged residues, the ones thermophiles trade away at the surface.
POLAR_UNCHARGED = frozenset("STNQ")
#: Thermolabile amides: both deamidate, and faster as temperature rises.
THERMOLABILE = frozenset("NQ")
CHARGED_SURFACE = frozenset("DEKR")


@register
class ArgininePreference(Strategy):
    name = "arginine_preference"
    mechanism = ("Prefer arginine to lysine on the surface. Thermophile "
                 "proteins are consistently enriched in Arg relative to Lys. "
                 "The guanidinium group is planar and makes bidentate hydrogen "
                 "bonds where an ammonium makes one, its charge is delocalised "
                 "so it stays protonated across a wider pH range, and it lacks "
                 "the primary amine that makes lysine reactive.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.aa(p) == "K" and ctx.layer(p) == "surface"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, frozenset("R"), self.name,
                         f"surface lysine at {ctx.label(p)}; Arg is the "
                         f"thermophile-favoured equivalent")
                for p in positions]


@register
class ThermolabileAmide(Strategy):
    name = "thermolabile_amide"
    mechanism = ("Deplete surface asparagine and glutamine. Both amides "
                 "deamidate, and the rate climbs steeply with temperature, so "
                 "hyperthermophiles carry markedly fewer of them than their "
                 "mesophile counterparts. Replacing them with charged residues "
                 "removes the liability and adds surface charge at the same "
                 "time.")
    applies_to = frozenset({SOLUBLE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.aa(p) in THERMOLABILE and ctx.layer(p) == "surface"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, CHARGED_SURFACE, self.name,
                         f"surface {ctx.aa(p)} at {ctx.label(p)} is "
                         f"thermolabile")
                for p in positions]


@register
class SurfaceChargeEnrichment(Strategy):
    name = "surface_charge_enrichment"
    mechanism = ("Trade uncharged polar surface for charged surface. The "
                 "clearest compositional signature of thermophilic proteins is "
                 "a surface richer in Asp, Glu, Lys and Arg and poorer in Ser, "
                 "Thr, Asn and Gln. Charged surfaces raise solubility, "
                 "suppress aggregation and create the raw material for the "
                 "ion-pair networks below.")
    applies_to = frozenset({SOLUBLE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.aa(p) in POLAR_UNCHARGED and ctx.layer(p) == "surface"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, CHARGED_SURFACE, self.name,
                         f"uncharged polar surface at {ctx.label(p)}")
                for p in positions]


@register
class SaltBridgeNetwork(Strategy):
    name = "salt_bridge_network"
    mechanism = ("Extend ion pairs into networks. Hyperthermophile proteins do "
                 "not merely have more salt bridges, they have larger *linked* "
                 "clusters of them. A network is worth more than the sum of "
                 "its pairs: each charge is stabilised by several partners, so "
                 "losing one does not unravel the group, and the entropic cost "
                 "of ordering the sidechains is paid once for the whole "
                 "cluster.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    NETWORK_RADIUS = 8.0
    #: A position already flanked by this many charges is a network seed.
    MIN_NEIGHBOURS = 2

    def _seeds(self, ctx: DesignContext) -> dict[int, int]:
        """Uncharged surface positions sitting inside an existing charge cluster."""
        charged = [p for p in ctx.positions if ctx.aa(p) in CHARGED_SURFACE]
        if len(charged) < self.MIN_NEIGHBOURS:
            return {}
        out: dict[int, int] = {}
        for p in ctx.designable:
            if ctx.layer(p) != "surface" or ctx.aa(p) in CHARGED_SURFACE:
                continue
            row = ctx.cb_dist[p - 1]
            near = [q for q in charged
                    if q != p and row[q - 1] <= self.NETWORK_RADIUS]
            if len(near) >= self.MIN_NEIGHBOURS:
                out[p] = len(near)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return sorted(self._seeds(ctx))

    def propose(self, ctx, positions, rng):
        seeds = self._seeds(ctx)
        out = []
        for p in positions:
            # Complete the network with whichever sign is under-represented
            # locally, so the cluster ends up mixed rather than all one charge.
            row = ctx.cb_dist[p - 1]
            pos_near = sum(1 for q in ctx.positions
                           if ctx.aa(q) in "KR" and row[q - 1] <= self.NETWORK_RADIUS)
            neg_near = sum(1 for q in ctx.positions
                           if ctx.aa(q) in "DE" and row[q - 1] <= self.NETWORK_RADIUS)
            allowed = NEGATIVE if pos_near > neg_near else POSITIVE
            out.append(Proposal(
                p, allowed, self.name,
                f"{seeds.get(p, 0)} charges within {self.NETWORK_RADIUS:.0f}A "
                f"of {ctx.label(p)}; completing the network",
            ))
        return out


@register
class HelixDipole(Strategy):
    name = "helix_dipole"
    mechanism = ("Satisfy the helix macrodipole. Aligned backbone carbonyls "
                 "give every helix a net positive charge at its N-terminus and "
                 "negative at its C-terminus. Placing an acidic residue near "
                 "the start and a basic one near the end pays that back. The "
                 "effect is strong enough that natural helices show the bias "
                 "clearly across whole proteomes.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    #: How many residues in from each terminus the dipole is worth answering.
    SPAN = 2

    def _sites(self, ctx: DesignContext) -> dict[int, str]:
        sites: dict[int, str] = {}
        for start, end in ctx.ss_segments("H"):
            if end - start + 1 < 7:
                continue
            for offset in range(self.SPAN):
                n_pos, c_pos = start + offset, end - offset
                if n_pos in ctx.designable and ctx.aa(n_pos) not in "DE":
                    sites.setdefault(n_pos, "N")
                if c_pos in ctx.designable and ctx.aa(c_pos) not in "KR":
                    sites.setdefault(c_pos, "C")
        return sites

    def diagnose(self, ctx: DesignContext) -> list[int]:
        # Only worth doing where the sidechain is solvated.
        return [p for p, _ in self._sites(ctx).items() if ctx.layer(p) != "core"]

    def propose(self, ctx, positions, rng):
        sites = self._sites(ctx)
        out = []
        for p in positions:
            end = sites.get(p)
            if end == "N":
                out.append(Proposal(p, NEGATIVE, self.name,
                                    f"{ctx.label(p)} near a helix N-terminus; "
                                    f"the dipole is positive there"))
            elif end == "C":
                out.append(Proposal(p, POSITIVE, self.name,
                                    f"{ctx.label(p)} near a helix C-terminus; "
                                    f"the dipole is negative there"))
        return out


@register
class CappingBox(Strategy):
    name = "capping_box"
    mechanism = ("Build a capping box at helix starts. Beyond a plain N-cap, "
                 "the classic motif is reciprocal: the N-cap sidechain hydrogen "
                 "bonds to the backbone of residue N3, and the N3 sidechain "
                 "bonds back to the N-cap backbone. Glutamate at N3 with "
                 "serine or threonine at the cap is the commonest form, and it "
                 "is worth appreciably more than either half alone.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _sites(self, ctx: DesignContext) -> list[int]:
        out = []
        for start, end in ctx.ss_segments("H"):
            if end - start + 1 < 7:
                continue
            n3 = start + 2          # N-cap is start-1, so N3 is start+2
            if n3 in ctx.designable and ctx.aa(n3) != "E":
                if ctx.layer(n3) != "core":
                    out.append(n3)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return self._sites(ctx)

    def propose(self, ctx, positions, rng):
        return [Proposal(p, frozenset("EQ"), self.name,
                         f"N3 of a helix at {ctx.label(p)}; completes the "
                         f"reciprocal capping box")
                for p in positions]
