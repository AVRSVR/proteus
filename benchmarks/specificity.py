"""Does Proteus distinguish a broken design from a sound protein?

This is the project's most interesting claim and the one most worth testing,
because nothing in the design aims at it. Proteus is never told which class a
structure belongs to. It diagnoses mechanisms, applies them, and measures what
it gained. The claim is that *gain per mutation* -- how much score a single
change buys -- falls out as a discriminator:

    a badly designed protein has a lot available per change,
    an evolved or validated one has almost nothing.

If that holds, the mutation floor is not an arbitrary knob. It is a decision
boundary, and setting it correctly makes the tool specific: it repairs what is
broken and leaves alone what is not.

Run with a manifest of structures grouped by class:

    python benchmarks/specificity.py manifest.json

where the manifest is

    {"natural": [{"label": "CDC42", "path": "...", "chain": null}, ...],
     "designed": [...]}

Measurement runs with the mutation floor switched *off*, so it records the raw
gain the strategies can find rather than what survives a threshold. Applying
the threshold first would assume the answer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proteus import DesignContext, from_pdb          # noqa: E402
from proteus.engine import Engine                     # noqa: E402

MIN_RESIDUES = 40
MAX_RESIDUES = 900


def measure(path: str, chain: str | None, generations: int, seed: int) -> dict | None:
    """Raw gain per mutation for one structure, floor disabled."""
    try:
        structure = from_pdb(path, chain=chain)
    except Exception as exc:
        print(f"  skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    if not MIN_RESIDUES <= len(structure) <= MAX_RESIDUES:
        print(f"  skipped ({len(structure)} residues, outside "
              f"{MIN_RESIDUES}-{MAX_RESIDUES})", file=sys.stderr)
        return None

    ctx = DesignContext(structure=structure)
    result = Engine(ctx, seed=seed, min_gain_per_mutation=0.0).run(
        generations=generations)
    raw = result.start_score - result.breakdown_best.per_residue
    return {
        "n_residues": len(structure),
        "start_score": result.start_score,
        "n_mutations": result.n_mutations,
        "raw_gain": raw,
        "gain_per_mutation": raw / result.n_mutations if result.n_mutations else 0.0,
    }


def summarise(rows: list[dict]) -> None:
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    print(f"\n{'class':<20} {'n':>3} {'median gain/mut':>17} {'min':>11} {'max':>11}")
    print("-" * 66)
    medians: dict[str, float] = {}
    for name, group in sorted(by_class.items()):
        values = sorted(r["gain_per_mutation"] for r in group)
        medians[name] = statistics.median(values)
        print(f"{name:<20} {len(group):3d} {medians[name]:17.6f} "
              f"{values[0]:11.6f} {values[-1]:11.6f}")

    if len(medians) >= 2:
        hi = max(medians.values())
        lo = min(v for v in medians.values() if v > 0) if any(
            v > 0 for v in medians.values()) else 0.0
        if lo > 0:
            print(f"\nseparation between class medians: {hi / lo:.1f}x")

    # The threshold that best separates the largest class from the rest, which
    # is the honest way to report how usable this is as a decision boundary.
    labelled = [(r["gain_per_mutation"], r["class"]) for r in rows]
    biggest = max(by_class, key=lambda k: len(by_class[k]))
    positive = [v for v, c in labelled if c == biggest]
    negative = [v for v, c in labelled if c != biggest]
    if positive and negative:
        best_acc, best_thr = 0.0, 0.0
        for threshold in sorted({v for v, _ in labelled}):
            tp = sum(1 for v in positive if v >= threshold)
            fp = sum(1 for v in negative if v >= threshold)
            acc = (tp + (len(negative) - fp)) / (len(positive) + len(negative))
            if acc > best_acc:
                best_acc, best_thr = acc, threshold
        print(f"\nbest separation of '{biggest}' (n={len(positive)}) from the "
              f"rest (n={len(negative)}):")
        print(f"  {best_acc:.0%} accuracy at gain/mutation >= {best_thr:.6f}")

    print("\nCaveats worth stating with any result from this script:")
    print("  - structures from one design run are highly correlated; a class of")
    print("    20 outputs from the same loop is closer to n=1 than to n=20")
    print("  - gain is measured against HeuristicScorer, so this shows the")
    print("    classes differ under *that* objective, not under real energetics")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", help="JSON mapping class name -> entries")
    parser.add_argument("-n", "--generations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="specificity.json")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for class_name, entries in manifest.items():
        for entry in entries:
            label = entry.get("label") or Path(entry["path"]).stem
            print(f"{class_name:<18} {label[:34]:<36}", end="", flush=True)
            stats = measure(entry["path"], entry.get("chain"),
                            args.generations, args.seed)
            if stats is None:
                continue
            print(f" {stats['n_residues']:4d} res  "
                  f"gain/mut {stats['gain_per_mutation']:.6f}", flush=True)
            rows.append({"class": class_name, "label": label, **stats})

    if not rows:
        print("no structures measured", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summarise(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
