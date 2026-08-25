"""Chemical degradation routes, and how nature avoids them.

Folding stability is only half of what keeps a protein intact. The other half
is chemistry: over days to weeks, asparagine deamidates, aspartate isomerises,
free thiols oxidise and scramble, methionine oxidises. Proteins that have to
last -- secreted enzymes, structural proteins, antibodies in serum -- are
visibly depleted in the motifs that make these fast, and that depletion is
selection acting on chemistry rather than on folding.

Every mechanism here removes a degradation route rather than adding a
stabilizing interaction. They matter most for the case Proteus is aimed at: a
designed protein has no evolutionary history filtering these motifs out, so it
carries them at background frequency.
"""

from __future__ import annotations

from ..context import DesignContext
from ..proposals import Proposal
from .base import MEMBRANE, SOLUBLE, Strategy, register

# Asn followed by a small flexible residue deamidates fastest: the backbone
# nitrogen of residue i+1 attacks the sidechain carbonyl to form a succinimide,
# and a small i+1 leaves room for it.
DEAMIDATION_PAIRS = ("NG", "NS", "NN", "NT", "NA")
# Asp isomerises to iso-Asp through the same succinimide chemistry.
ISOMERISATION_PAIRS = ("DG", "DP", "DS", "DD")

#: Conservative replacements for Asn that keep size and polarity.
ASN_ALTERNATIVES = frozenset("QSTDH")
#: Replacements for Asp that keep the negative charge where possible.
ASP_ALTERNATIVES = frozenset("EQSN")


@register
class DeamidationMotif(Strategy):
    name = "deamidation_motif"
    mechanism = ("Remove asparagine deamidation sites. Asn followed by a small "
                 "flexible residue cyclises to a succinimide and hydrolyses to "
                 "Asp, introducing a negative charge that was never designed "
                 "for. It is the single most common chemical degradation route "
                 "in proteins, and long-lived natural proteins are measurably "
                 "depleted in these motifs.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _sites(self, ctx: DesignContext) -> list[int]:
        seq = ctx.structure.sequence
        out = []
        for p in ctx.designable:
            if seq[p - 1] != "N" or p >= len(seq):
                continue
            if seq[p - 1:p + 1] in DEAMIDATION_PAIRS:
                out.append(p)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        # Exposed sites react far faster; buried ones are largely protected.
        return [p for p in self._sites(ctx) if ctx.layer(p) != "core"]

    def propose(self, ctx, positions, rng):
        seq = ctx.structure.sequence
        return [Proposal(p, ASN_ALTERNATIVES, self.name,
                         f"{seq[p - 1:p + 1]} motif at {ctx.label(p)} deamidates")
                for p in positions]


@register
class IsomerisationMotif(Strategy):
    name = "isomerisation_motif"
    mechanism = ("Remove aspartate isomerisation sites. Asp-Gly and related "
                 "pairs convert to iso-aspartate through a succinimide, which "
                 "inserts an extra backbone carbon and kinks the chain. In a "
                 "loop this is tolerated; in a strand or near an interface it "
                 "is not.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _sites(self, ctx: DesignContext) -> list[int]:
        seq = ctx.structure.sequence
        return [p for p in ctx.designable
                if seq[p - 1] == "D" and p < len(seq)
                and seq[p - 1:p + 1] in ISOMERISATION_PAIRS]

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in self._sites(ctx) if ctx.layer(p) != "core"]

    def propose(self, ctx, positions, rng):
        seq = ctx.structure.sequence
        return [Proposal(p, ASP_ALTERNATIVES, self.name,
                         f"{seq[p - 1:p + 1]} motif at {ctx.label(p)} isomerises")
                for p in positions]


@register
class GlycosylationSequon(Strategy):
    name = "glycosylation_sequon"
    mechanism = ("Remove unintended N-linked glycosylation sites. The sequon "
                 "Asn-X-Ser/Thr, with X not proline, is recognised by the "
                 "cellular machinery and carries a glycan. In a design that "
                 "did not ask for one this changes mass, solubility and "
                 "immunogenicity, and can block the intended interface.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    def _sites(self, ctx: DesignContext) -> list[int]:
        seq = ctx.structure.sequence
        out = []
        for p in ctx.designable:
            if p + 2 > len(seq):
                continue
            if seq[p - 1] == "N" and seq[p] != "P" and seq[p + 1] in "ST":
                out.append(p)
        return out

    def diagnose(self, ctx: DesignContext) -> list[int]:
        # Only surface sequons are actually reachable by the transferase.
        return [p for p in self._sites(ctx) if ctx.layer(p) == "surface"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, ASN_ALTERNATIVES, self.name,
                         f"N-X-S/T sequon starting at {ctx.label(p)}")
                for p in positions]


@register
class FreeCysteine(Strategy):
    name = "free_cysteine"
    mechanism = ("Remove unpaired cysteines. A free thiol oxidises, forms "
                 "intermolecular bridges that aggregate the protein, and "
                 "scrambles existing disulfides. Intracellular proteins keep "
                 "cysteine reduced and rare; secreted ones pair it. An "
                 "unpaired, exposed cysteine is neither.")
    applies_to = frozenset({SOLUBLE, MEMBRANE})

    #: Serine is the classic isosteric replacement; Ala and Thr also work.
    ALTERNATIVES = frozenset("SAT")
    PAIR_DISTANCE = 4.5

    def _unpaired(self, ctx: DesignContext) -> list[int]:
        cys = [p for p in ctx.positions if ctx.aa(p) == "C"]
        if not cys:
            return []
        paired = set()
        for a_idx, i in enumerate(cys):
            for j in cys[a_idx + 1:]:
                if ctx.cb_dist[i - 1, j - 1] <= self.PAIR_DISTANCE:
                    paired.update((i, j))
        return [p for p in cys if p not in paired and p in ctx.designable]

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in self._unpaired(ctx) if ctx.layer(p) != "core"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, self.ALTERNATIVES, self.name,
                         f"unpaired cysteine at {ctx.label(p)} will oxidise")
                for p in positions]


@register
class MethionineOxidation(Strategy):
    name = "methionine_oxidation"
    mechanism = ("Replace exposed methionine. The thioether oxidises to a "
                 "sulfoxide under ordinary storage and handling, adding an "
                 "oxygen and a large polarity change at a position that was "
                 "chosen to be hydrophobic. Leucine is the standard isosteric "
                 "substitute.")
    applies_to = frozenset({SOLUBLE})

    ALTERNATIVES = frozenset("LQKI")

    def diagnose(self, ctx: DesignContext) -> list[int]:
        return [p for p in ctx.designable
                if ctx.aa(p) == "M" and ctx.layer(p) == "surface"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, self.ALTERNATIVES, self.name,
                         f"exposed methionine at {ctx.label(p)} oxidises")
                for p in positions]
