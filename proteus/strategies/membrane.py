"""Stabilization mechanisms for membrane-embedded proteins.

These are not soluble strategies with a flag flipped. The bilayer inverts the
sign of the dominant term: a solvent-exposed position in a soluble protein
wants to be polar, while a lipid-exposed position wants to be hydrophobic, and
the polar residues that a soluble protein puts on its surface belong *inside*
the helix bundle of a membrane protein. Every strategy here is gated on
membrane depth as well as burial, because neither coordinate alone is enough to
say what belongs at a position.
"""

from __future__ import annotations

import numpy as np

from ..context import DesignContext
from ..proposals import CHARGED, POSITIVE, Proposal
from .base import MEMBRANE, Strategy, register

# Residues that tolerate direct lipid contact.
LIPID_FACING = frozenset("AVLIMF")
# The interfacial aromatic belt: Trp and Tyr concentrate at the headgroup
# region, where their amphipathic character anchors the protein vertically.
BELT_AROMATIC = frozenset("WY")
# Long-chain basics that can "snorkel" -- aliphatic stem in the hydrophobic
# region, charged tip reaching up to the phosphate headgroups.
SNORKELERS = frozenset("KR")
# Small polars able to form interhelical hydrogen bonds inside the bundle.
INTERHELICAL_POLAR = frozenset("STNQ")

POLAR_OR_CHARGED = frozenset("DEKRNQSTH")


@register
class LipidFacingHydrophobic(Strategy):
    name = "lipid_facing_hydrophobic"
    mechanism = ("Match the lipid environment. Positions exposed to the "
                 "bilayer core pay a large desolvation penalty for polar or "
                 "charged sidechains; hydrophobic residues there are the "
                 "single largest term in membrane protein stability.")
    applies_to = frozenset({MEMBRANE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.is_lipid_facing(p) and ctx.aa(p) in POLAR_OR_CHARGED]

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            allowed = LIPID_FACING
            if ctx.ss_at(p) == "H":
                allowed = allowed & frozenset("ALMIF") or LIPID_FACING
            out.append(Proposal(
                p, allowed, self.name,
                f"lipid-facing {ctx.aa(p)} at depth {ctx.depth(p):+.1f}A "
                f"pays a desolvation penalty",
            ))
        return out


@register
class AromaticBelt(Strategy):
    name = "aromatic_belt"
    mechanism = ("Anchor the protein vertically in the bilayer. Tryptophan "
                 "and tyrosine cluster at the lipid headgroup interface in "
                 "essentially every known membrane protein: the ring sits in "
                 "the acyl region while the polar edge hydrogen bonds to the "
                 "headgroups, pinning the protein against vertical drift.")
    applies_to = frozenset({MEMBRANE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.zone(p) == "interface"
                and ctx.layer(p) in ("surface", "boundary")
                and ctx.aa(p) not in BELT_AROMATIC]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, BELT_AROMATIC, self.name,
                         f"interfacial position at depth {ctx.depth(p):+.1f}A "
                         f"suits an aromatic anchor")
                for p in positions]


@register
class Snorkeling(Strategy):
    name = "snorkeling"
    mechanism = ("Exploit snorkeling. Lysine and arginine placed near the "
                 "bilayer boundary bury their aliphatic stem in the acyl "
                 "region while the charged tip reaches the phosphate "
                 "headgroups, so they stabilise positions that would punish "
                 "any other charged residue.")
    applies_to = frozenset({MEMBRANE})

    def diagnose(self, ctx: DesignContext) -> list[int]:
        out = []
        for p in ctx.designable:
            if ctx.zone(p) != "interface" or ctx.layer(p) != "surface":
                continue
            # Only near the inner edge of the interface, where the stem can
            # actually span to the headgroups.
            if abs(ctx.depth(p)) < ctx.membrane.lipid_core_half - 1.0:
                continue
            if ctx.aa(p) in SNORKELERS:
                continue
            out.append(p)
        return out

    def propose(self, ctx, positions, rng):
        return [Proposal(p, SNORKELERS, self.name,
                         f"depth {ctx.depth(p):+.1f}A permits a snorkeling basic")
                for p in positions]


@register
class PositiveInside(Strategy):
    name = "positive_inside"
    mechanism = ("Reinforce the positive-inside rule. Cytoplasmic loops of "
                 "membrane proteins are strongly enriched in lysine and "
                 "arginine; the bias sets and locks in membrane topology, and "
                 "strengthening it stabilises the intended orientation.")
    applies_to = frozenset({MEMBRANE})

    def _cytoplasmic_sign(self, ctx: DesignContext) -> float:
        """Infer which side is cytoplasmic from existing charge asymmetry.

        Topology is a fact about the protein that geometry cannot supply. When
        it is unknown we infer it from the existing basic-residue asymmetry --
        the same signal the rule itself describes -- and say so in the
        rationale. Pass a known topology instead whenever you have one.
        """
        pos_side = neg_side = 0
        for p in ctx.positions:
            if ctx.zone(p) == "lipid_core":
                continue
            if ctx.aa(p) not in SNORKELERS:
                continue
            if ctx.depth(p) > 0:
                pos_side += 1
            else:
                neg_side += 1
        return 1.0 if pos_side >= neg_side else -1.0

    def diagnose(self, ctx: DesignContext) -> list[int]:
        sign = self._cytoplasmic_sign(ctx)
        return [p for p in ctx.designable
                if ctx.zone(p) in ("interface", "aqueous")
                and np.sign(ctx.depth(p)) == sign
                and ctx.layer(p) == "surface"
                and ctx.aa(p) not in SNORKELERS]

    def propose(self, ctx, positions, rng):
        sign = self._cytoplasmic_sign(ctx)
        side = "positive-depth" if sign > 0 else "negative-depth"
        return [Proposal(p, POSITIVE, self.name,
                         f"{side} side inferred cytoplasmic; "
                         f"positive-inside bias at depth {ctx.depth(p):+.1f}A")
                for p in positions]


@register
class InterhelicalPolar(Strategy):
    name = "interhelical_polar"
    mechanism = ("Add buried hydrogen bonds between helices. Inside a "
                 "membrane bundle there is no water to compete, so a buried "
                 "polar pair is worth far more than the same pair would be in "
                 "a soluble protein -- this is how tightly packed TM helices "
                 "hold each other in register.")
    applies_to = frozenset({MEMBRANE})

    # Small polars only: a buried charge inside the bilayer is destabilising
    # unless it has a dedicated partner, which this strategy cannot guarantee.
    MAX_SITES = 4

    def diagnose(self, ctx: DesignContext) -> list[int]:
        out = []
        for p in ctx.designable:
            if not ctx.is_membrane_buried(p):
                continue
            if ctx.layer(p) != "core":
                continue
            if ctx.aa(p) in INTERHELICAL_POLAR:
                continue
            # Needs a partner position close enough to hydrogen bond to.
            row = ctx.cb_dist[p - 1]
            partners = [
                q for q in ctx.positions
                if q != p and abs(q - p) > 4
                and 4.0 <= row[q - 1] <= 7.0
                and ctx.is_membrane_buried(q)
            ]
            if partners:
                out.append(p)
        return out

    def propose(self, ctx, positions, rng):
        chosen = positions[: self.MAX_SITES]
        return [Proposal(p, INTERHELICAL_POLAR, self.name,
                         f"buried in bilayer at depth {ctx.depth(p):+.1f}A "
                         f"with an interhelical partner in range")
                for p in chosen]


@register
class HydrophobicMismatch(Strategy):
    name = "hydrophobic_mismatch"
    mechanism = ("Correct hydrophobic mismatch. When the protein's "
                 "hydrophobic band is longer or shorter than the bilayer is "
                 "thick, the lipids deform to compensate and the protein pays "
                 "for it. Trimming or extending the hydrophobic band at the "
                 "boundaries relieves that strain.")
    applies_to = frozenset({MEMBRANE})

    def _boundary_positions(self, ctx: DesignContext) -> list[int]:
        half = ctx.membrane.lipid_core_half
        out = []
        for p in ctx.designable:
            if ctx.layer(p) != "surface":
                continue
            d = abs(ctx.depth(p))
            # Within 2.5 A either side of the hydrophobic boundary.
            if abs(d - half) <= 2.5:
                out.append(p)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        out = []
        for p in self._boundary_positions(ctx):
            inside = abs(ctx.depth(p)) <= ctx.membrane.lipid_core_half
            hydrophobic = ctx.aa(p) in LIPID_FACING
            # Mismatched: hydrophobic outside the band, or polar inside it.
            if inside != hydrophobic:
                out.append(p)
        return out

    def propose(self, ctx, positions, rng):
        out = []
        for p in positions:
            inside = abs(ctx.depth(p)) <= ctx.membrane.lipid_core_half
            allowed = LIPID_FACING if inside else (BELT_AROMATIC | frozenset("STNQKR"))
            where = "inside" if inside else "outside"
            out.append(Proposal(
                p, allowed, self.name,
                f"depth {ctx.depth(p):+.1f}A is {where} the hydrophobic band "
                f"but carries {ctx.aa(p)}",
            ))
        return out
