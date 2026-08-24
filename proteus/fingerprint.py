"""Structural fingerprints: describing *what kind of protein* this is.

This is the piece that makes the leaderboard worth keeping. A bandit that
learns "core packing is a good strategy" has learned something about the last
protein it saw and nothing more. The claim that transfers is conditional:
*core packing works on proteins with an underpacked core*. To learn that, the
system needs a description of structural context that is comparable across
proteins.

A fingerprint is therefore deliberately size-independent -- every feature is a
fraction, a density or a log -- so a 70-residue miniprotein and a 900-residue
enzyme land in the same space and can be compared.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from .context import DesignContext

#: Feature order is fixed; changing it invalidates stored knowledge bases.
FEATURE_NAMES = (
    "log_size",
    "frac_core",
    "frac_boundary",
    "frac_surface",
    "frac_helix",
    "frac_strand",
    "frac_loop",
    "mean_burial",
    "is_membrane",
    "frac_lipid_facing",
    "exposed_hydrophobic_frac",
    "aggregation_load",
    "net_charge_per_residue",
    "frac_glycine",
    "frac_proline",
    "helix_segment_density",
)

# A fingerprint carries two different kinds of information and they need
# different weight in the distance.
#
# *Identity* features say what kind of protein this is -- its size, fold
# composition and burial profile. *State* features say how damaged it is: how
# much exposed hydrophobic surface it carries, how lopsided its charge is.
#
# Both matter for choosing a strategy, but identity has to anchor the metric.
# Left unweighted, the state features dominated completely: two ~70-residue
# all-helical bundles came out only 0.18 similar because one was damaged and
# the other was not, with aggregation load and exposed hydrophobic fraction
# together contributing 95% of the distance. Fold identity was invisible.
#
# Values are the spread each feature is expected to show across proteins,
# divided by how much it should count. Larger means less influence.
_SCALE = {
    # identity
    "log_size": 1.2,
    "frac_core": 0.20,
    "frac_boundary": 0.20,
    "frac_surface": 0.20,
    "frac_helix": 0.25,
    "frac_strand": 0.25,
    "frac_loop": 0.25,
    "mean_burial": 1.5,
    "is_membrane": 0.20,
    "frac_lipid_facing": 0.25,
    "helix_segment_density": 0.05,
    # state -- deliberately damped so it modulates rather than dominates
    "exposed_hydrophobic_frac": 0.60,
    "aggregation_load": 0.60,
    "net_charge_per_residue": 0.40,
    "frac_glycine": 0.40,
    "frac_proline": 0.40,
}
_DEFAULT_SCALE = 0.25


def _bounded(x: float) -> float:
    """Map an unbounded non-negative quantity into [0, 1).

    The aggregation term has no upper limit, so as a raw feature it dwarfed
    every bounded fraction in the distance calculation.
    """
    return x / (1.0 + x) if x > 0 else 0.0


@dataclass(frozen=True)
class Fingerprint:
    """A size-independent description of a protein's structural context."""

    log_size: float = 0.0
    frac_core: float = 0.0
    frac_boundary: float = 0.0
    frac_surface: float = 0.0
    frac_helix: float = 0.0
    frac_strand: float = 0.0
    frac_loop: float = 0.0
    mean_burial: float = 0.0
    is_membrane: float = 0.0
    frac_lipid_facing: float = 0.0
    exposed_hydrophobic_frac: float = 0.0
    #: Bounded into [0, 1); see :func:`_bounded`.
    aggregation_load: float = 0.0
    net_charge_per_residue: float = 0.0
    frac_glycine: float = 0.0
    frac_proline: float = 0.0
    helix_segment_density: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Fingerprint":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})

    def distance(self, other: "Fingerprint") -> float:
        """Scaled Euclidean distance in feature space."""
        scales = np.array([_SCALE.get(n, _DEFAULT_SCALE) for n in FEATURE_NAMES])
        diff = (self.to_array() - other.to_array()) / scales
        return float(np.linalg.norm(diff) / math.sqrt(len(FEATURE_NAMES)))

    def similarity(self, other: "Fingerprint", bandwidth: float = 0.5) -> float:
        """Gaussian kernel on the distance; 1.0 means identical context."""
        d = self.distance(other)
        return float(math.exp(-0.5 * (d / bandwidth) ** 2))

    def describe(self) -> str:
        env = "membrane" if self.is_membrane >= 0.5 else "soluble"
        return (f"{env}, ~{int(round(math.exp(self.log_size)))} res, "
                f"core {self.frac_core:.0%} / surface {self.frac_surface:.0%}, "
                f"H {self.frac_helix:.0%} E {self.frac_strand:.0%}, "
                f"aggregation load {self.aggregation_load:.2f}")


def compute(ctx: DesignContext, scorer=None) -> Fingerprint:
    """Derive a fingerprint from an analysed structure."""
    from .scoring import AGGREGATION, HeuristicScorer

    n = len(ctx)
    if n == 0:
        return Fingerprint()
    scorer = scorer or HeuristicScorer()
    seq = ctx.structure.sequence
    breakdown = scorer.score(ctx, seq)

    layers = ctx.layers
    surface_positions = [p for p in ctx.positions if ctx.layer(p) == "surface"]
    exposed_hydrophobic = sum(
        1 for p in surface_positions if AGGREGATION.get(ctx.aa(p), 0.0) > 0.5
    )

    lipid_facing = (sum(1 for p in ctx.positions if ctx.is_lipid_facing(p))
                    if ctx.is_membrane else 0)

    charge = sum({"D": -1, "E": -1, "K": 1, "R": 1}.get(a, 0) for a in seq)

    return Fingerprint(
        log_size=math.log(n),
        frac_core=layers.count("core") / n,
        frac_boundary=layers.count("boundary") / n,
        frac_surface=layers.count("surface") / n,
        frac_helix=ctx.ss.count("H") / n,
        frac_strand=ctx.ss.count("E") / n,
        frac_loop=ctx.ss.count("L") / n,
        mean_burial=float(np.mean(ctx.neighbors)),
        is_membrane=1.0 if ctx.is_membrane else 0.0,
        frac_lipid_facing=lipid_facing / n,
        exposed_hydrophobic_frac=exposed_hydrophobic / max(len(surface_positions), 1),
        aggregation_load=_bounded(float(breakdown.terms.get("aggregation", 0.0))),
        net_charge_per_residue=charge / n,
        frac_glycine=seq.count("G") / n,
        frac_proline=seq.count("P") / n,
        helix_segment_density=len(ctx.ss_segments("H")) / n,
    )
