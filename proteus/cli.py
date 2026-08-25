"""Command-line interface.

    proteus analyze design.pdb --membrane
    proteus run design.pdb --freeze 1-10,47-53 --generations 60
"""

from __future__ import annotations

import argparse
import sys

from . import membrane as membrane_mod
from .context import DesignContext
from .engine import Engine
from .scoring import HeuristicScorer
from .strategies import REGISTRY
from .structure import from_pdb


def parse_ranges(spec: str | None) -> frozenset[int]:
    """Parse ``1-10,47-53,88`` into a set of residue indices."""
    if not spec:
        return frozenset()
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return frozenset(out)


def build_context(args) -> DesignContext:
    structure = from_pdb(args.pdb, chain=args.chain)
    mem = None
    if args.membrane:
        mem = membrane_mod.from_opm_dummies(args.pdb)
        if mem is None:
            mem = membrane_mod.estimate(structure)
            print("note: no OPM dummy atoms found; bilayer estimated from the "
                  "hydrophobic belt. If this design's surface is already wrong, "
                  "that estimate is unreliable -- supply an OPM-oriented "
                  "structure for best results.", file=sys.stderr)
    frozen = parse_ranges(args.freeze)
    invalid = {p for p in frozen if not 1 <= p <= len(structure)}
    if invalid:
        raise SystemExit(f"--freeze refers to residues outside 1..{len(structure)}: "
                         f"{sorted(invalid)}")
    return DesignContext(structure=structure, frozen=frozen, membrane=mem)


def cmd_analyze(args) -> int:
    ctx = build_context(args)
    print(ctx.summary())
    print()
    scorer = HeuristicScorer()
    print(scorer.score(ctx, ctx.structure.sequence).table())
    print()
    applicable = REGISTRY.applicable(ctx)
    print(f"applicable strategies ({len(applicable)} of "
          f"{len(REGISTRY.for_context(ctx))} valid in this environment):")
    for s in applicable:
        sites = len(s.diagnose(ctx))
        print(f"  {s.name:<26} {sites:3d} sites")
        if args.verbose:
            print(f"      {s.mechanism}")
    return 0


def cmd_run(args) -> int:
    from pathlib import Path

    from .knowledge import KnowledgeBase

    ctx = build_context(args)
    print(ctx.summary())
    print()

    knowledge = None
    if args.knowledge:
        knowledge = KnowledgeBase.load(args.knowledge)
        print(f"knowledge base: {len(knowledge)} observations from previous runs")

    engine = Engine(
        ctx,
        policy=args.policy,
        strategies_per_move=args.strategies_per_move,
        knowledge=knowledge,
        protein=Path(args.pdb).stem,
        seed=args.seed,
    )

    if knowledge is not None and engine.fingerprint is not None:
        print(f"this protein   : {engine.fingerprint.describe()}")
        near = knowledge.neighbours(engine.fingerprint, k=3)
        if near:
            print("most similar seen before: " +
                  ", ".join(f"{name} ({sim:.2f})" for sim, name in near))
        if getattr(engine.policy, "priors", None):
            print(f"transferred priors for {len(engine.policy.priors)} strategies")
        else:
            print("no comparable proteins yet; starting without priors")
    print()

    result = engine.run(generations=args.generations, patience=args.patience,
                        verbose=not args.quiet)
    print()
    print(result.summary())
    print()
    print("start:", ctx.structure.sequence)
    print("best :", result.best_sequence)

    if args.explain:
        print()
        print(result.explain())
        credit = result.credit()
        if credit:
            print()
            print("surviving mutations per strategy:")
            for name, n in sorted(credit.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<26} {n}")

    if knowledge is not None:
        knowledge.save(args.knowledge)
        print(f"\nknowledge base now holds {len(knowledge)} observations "
              f"-> {args.knowledge}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(f">{args.pdb}|proteus|improvement="
                     f"{result.improvement:+.4f}/residue\n")
            fh.write(result.best_sequence + "\n")
        print(f"\nwrote {args.out}")
    return 0


def cmd_leaderboard(args) -> int:
    from .fingerprint import compute as compute_fingerprint
    from .knowledge import KnowledgeBase

    knowledge = KnowledgeBase.load(args.knowledge)
    if not len(knowledge):
        print(f"{args.knowledge}: no observations yet. Run 'proteus run "
              f"--knowledge {args.knowledge}' on a few structures first.")
        return 0

    if not args.target:
        print(knowledge.report(top=args.top))
        return 0

    structure = from_pdb(args.target, chain=args.chain)
    mem = membrane_mod.estimate(structure) if args.membrane else None
    ctx = DesignContext(structure=structure, membrane=mem)
    print_fp = compute_fingerprint(ctx)

    near = knowledge.neighbours(print_fp, k=5)
    if near:
        print("most similar proteins seen before:")
        for sim, name in near:
            print(f"  {sim:.3f}  {name}")
        print()
    print(knowledge.report(print_fp, top=args.top))
    return 0


def cmd_strategies(args) -> int:
    for s in REGISTRY.all():
        envs = ", ".join(sorted(s.applies_to))
        print(f"{s.name}  [{envs}]")
        print(f"    {s.mechanism}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="proteus",
        description="Stabilize designed proteins with mechanism-level strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("pdb", help="input PDB or mmCIF file")
        p.add_argument("--chain", help="restrict to one chain")
        p.add_argument("--membrane", action="store_true",
                       help="treat as membrane-embedded (inverts burial rules)")
        p.add_argument("--freeze", help="residues to hold fixed, e.g. 1-10,47-53")

    p_an = sub.add_parser("analyze", help="diagnose without modifying anything")
    add_common(p_an)
    p_an.add_argument("-v", "--verbose", action="store_true",
                      help="print each strategy's mechanism")
    p_an.set_defaults(func=cmd_analyze)

    p_run = sub.add_parser("run", help="run the stabilization loop")
    add_common(p_run)
    p_run.add_argument("-n", "--generations", type=int, default=50)
    p_run.add_argument("--policy", default="ucb1", choices=["ucb1", "thompson"])
    p_run.add_argument("--strategies-per-move", type=int, default=2)
    p_run.add_argument("--patience", type=int, default=None,
                       help="stop after this many generations without improvement")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--out", help="write the best sequence to this FASTA file")
    p_run.add_argument("--knowledge", metavar="PATH",
                       help="accumulate and reuse what works across runs; the "
                            "file is created if it does not exist")
    p_run.add_argument("--explain", action="store_true",
                       help="print every changed position and the mechanism "
                            "that proposed it")
    p_run.add_argument("-q", "--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("strategies", help="list the strategy library")
    p_st.set_defaults(func=cmd_strategies)

    p_lb = sub.add_parser("leaderboard",
                          help="what has worked, optionally for a given protein")
    p_lb.add_argument("knowledge", help="knowledge base file")
    p_lb.add_argument("--for", dest="target", metavar="PDB",
                      help="condition the leaderboard on this structure")
    p_lb.add_argument("--chain")
    p_lb.add_argument("--membrane", action="store_true")
    p_lb.add_argument("--top", type=int, default=None)
    p_lb.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
