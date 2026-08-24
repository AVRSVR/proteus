"""Proteus quickstart: the soluble path and the membrane path side by side.

Run with:  python examples/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proteus import DesignContext, REGISTRY, from_pdb
from proteus import membrane as membrane_mod
from proteus.engine import Engine

HERE = Path(__file__).parent


def show(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def stabilize(pdb: Path, membrane: bool, frozen=frozenset(), generations: int = 30):
    structure = from_pdb(str(pdb))
    mem = membrane_mod.estimate(structure) if membrane else None
    ctx = DesignContext(structure=structure, frozen=frozen, membrane=mem)

    print(ctx.summary())
    print()
    print("diagnosis -- what this fold actually admits:")
    for s in REGISTRY.applicable(ctx):
        print(f"  {s.name:<26} {len(s.diagnose(ctx)):3d} sites")

    result = Engine(ctx, seed=0).run(generations=generations)
    print()
    print(result.summary())
    print()
    print("start:", structure.sequence)
    print("best :", result.best_sequence)

    if frozen:
        held = all(result.best_sequence[p - 1] == structure.sequence[p - 1]
                   for p in frozen)
        print(f"\nfrozen region preserved: {held}")
    return result


show("SOLUBLE -- four-helix bundle, deliberately poor sequence")
stabilize(HERE / "soluble_bundle.pdb", membrane=False, frozen=frozenset(range(1, 9)))

show("MEMBRANE -- transmembrane bundle, burial rules inverted in the bilayer")
stabilize(HERE / "tm_bundle.pdb", membrane=True)

show("The point")
print("""
The two runs select different mechanisms because the diagnosis differs, not
because a flag was set. In the bilayer, core_packing and surface_depolarize are
not merely down-weighted -- they are not valid mechanisms at all, while
aromatic_belt and snorkeling become available. That separation is what makes
the leaderboard's answer transferable: it is a claim about structural context,
not about a particular position in a particular protein.
""".strip())
