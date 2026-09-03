"""Does HeuristicScorer predict anything real?

Benchmarks the scorer against S669: 669 single-point mutations across 94
proteins with experimentally measured ddG, curated to share at most 30%
sequence identity with the datasets the learned predictors were trained on.

The test is deliberately the one the project could fail. For each mutation the
scorer sees the wild-type structure and the mutant sequence threaded onto it,
and its score difference is correlated against the measured ddG. Published
predictors on the same rows are reported alongside, because a correlation
means nothing without knowing what the field already achieves.
"""
from __future__ import annotations
import sys, csv, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from proteus import DesignContext, from_pdb
from proteus.scoring import HeuristicScorer

ROOT = Path(__file__).parent / "s669"


def spearman(a, b):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, float)), np.asarray(rank(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = list(csv.DictReader(open(ROOT / "ddg_experimental.csv", encoding="utf-8")))
    scorer = HeuristicScorer()
    cache: dict[str, tuple] = {}

    preds, exps, skipped = [], [], 0
    per_row = []
    for r in rows:
        pdb_id, chain = r["pdb_id"], r["chain"]
        pre, post = r["pre"], r["post"]
        try:
            pos = int(r["pos"])
            exp = float(r["ddG_experimental"])
        except (ValueError, KeyError):
            skipped += 1; continue

        key = f"{pdb_id}_{chain}"
        if key not in cache:
            path = ROOT / "pdb" / f"{pdb_id}.pdb"
            if not path.exists():
                cache[key] = None
            else:
                try:
                    st = from_pdb(str(path), chain=chain)
                    ctx = DesignContext(structure=st)
                    # map PDB numbering -> internal index
                    pdbmap = {res.pdb_number: res.resi for res in st}
                    base = scorer.score(ctx, st.sequence).total
                    cache[key] = (st, ctx, pdbmap, base)
                except Exception:
                    cache[key] = None
        entry = cache[key]
        if entry is None:
            skipped += 1; continue
        st, ctx, pdbmap, base = entry

        idx = pdbmap.get(pos)
        if idx is None or st.sequence[idx - 1] != pre:
            skipped += 1; continue

        seq = list(st.sequence)
        seq[idx - 1] = post
        mut = scorer.score(ctx, "".join(seq)).total
        # score is "lower is better"; a destabilising mutation raises it.
        delta = mut - base
        preds.append(delta); exps.append(exp)
        per_row.append((pdb_id, chain, f"{pre}{pos}{post}", delta, exp))

    preds, exps = np.array(preds), np.array(exps)
    print(f"scored {len(preds)} of {len(rows)} mutations ({skipped} skipped)\n")

    pear = float(np.corrcoef(preds, exps)[0, 1])
    spear = spearman(preds, exps)
    print(f"{'HeuristicScorer':<24} Pearson r = {pear:+.3f}   Spearman = {spear:+.3f}")

    # Published predictors on the same dataset, for scale.
    for name, path, col in [
        ("ProteinMPNN-ddG", "proteinmpnn.csv", "ProteinMPNN-ddG"),
        ("ProteinMPNN", "proteinmpnn.csv", "ProteinMPNN"),
        ("RaSP", "rasp.csv", "RaSP"),
    ]:
        f = ROOT / path
        if not f.exists():
            continue
        p, e = [], []
        for row in csv.DictReader(open(f, encoding="utf-8")):
            try:
                p.append(float(row[col])); e.append(float(row["ddG_experimental"]))
            except (ValueError, KeyError):
                continue
        if len(p) > 10:
            print(f"{name:<24} Pearson r = {float(np.corrcoef(p,e)[0,1]):+.3f}   "
                  f"Spearman = {spearman(p,e):+.3f}   (n={len(p)})")

    # Predictors bundled in the main file.
    others = ["FoldX_dir", "DDGun3D_dir", "ACDC-NN_dir", "ThermoNet_dir",
              "PremPS_dir", "INPS3D_dir"]
    print()
    for col in others:
        p, e = [], []
        for row in rows:
            try:
                p.append(float(row[col])); e.append(float(row["ddG_experimental"]))
            except (ValueError, KeyError, TypeError):
                continue
        if len(p) > 10:
            print(f"{col.replace('_dir',''):<24} Pearson r = "
                  f"{float(np.corrcoef(p,e)[0,1]):+.3f}   Spearman = {spearman(p,e):+.3f}")

    out = Path(__file__).parent / "ddg_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain", "variant", "heuristic_delta", "ddg_experimental"])
        w.writerows(per_row)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
