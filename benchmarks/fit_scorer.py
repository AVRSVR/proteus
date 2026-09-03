"""Can the scorer's own terms be reweighted into a real predictor?

HeuristicScorer computes nine physically-motivated terms and combines them with
weights chosen by hand. Those weights were never fitted to anything. This asks
whether the terms carry real signal that the hand-tuning is wasting.

Evaluation is grouped k-fold by protein: every mutation from a given PDB sits
in the same fold, so the model is always scored on proteins it has never seen.
Without that, mutations from the same protein leak across the split and the
correlation is inflated.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from proteus import DesignContext, from_pdb
from proteus.scoring import HeuristicScorer

ROOT = Path(__file__).parent / "s669"
TERMS = ["burial", "packing", "ss_propensity", "aggregation", "net_charge",
         "bb_entropy", "capping", "liabilities", "interactions"]


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        order = np.argsort(v); r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        return (np.bincount(inv, weights=r) / cnt)[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def build():
    rows = list(csv.DictReader(open(ROOT / "ddg_experimental.csv", encoding="utf-8")))
    scorer = HeuristicScorer()
    cache: dict[str, object] = {}
    X, y, groups, total_delta = [], [], [], []

    for r in rows:
        try:
            pos, exp = int(r["pos"]), float(r["ddG_experimental"])
        except (ValueError, KeyError):
            continue
        key = f'{r["pdb_id"]}_{r["chain"]}'
        if key not in cache:
            p = ROOT / "pdb" / f'{r["pdb_id"]}.pdb'
            try:
                st = from_pdb(str(p), chain=r["chain"])
                ctx = DesignContext(structure=st)
                cache[key] = (st, ctx, {x.pdb_number: x.resi for x in st},
                              scorer.score(ctx, st.sequence))
            except Exception:
                cache[key] = None
        if cache[key] is None:
            continue
        st, ctx, pdbmap, base = cache[key]
        idx = pdbmap.get(pos)
        if idx is None or st.sequence[idx - 1] != r["pre"]:
            continue
        seq = list(st.sequence); seq[idx - 1] = r["post"]
        mut = scorer.score(ctx, "".join(seq))
        # per-term deltas, scaled to the whole chain so terms stay comparable
        n = len(st)
        X.append([(mut.terms[t] - base.terms[t]) * n for t in TERMS])
        total_delta.append(mut.total - base.total)
        y.append(exp); groups.append(r["pdb_id"])
    return np.array(X), np.array(y), np.array(groups), np.array(total_delta)


def main():
    X, y, groups, hand = build()
    print(f"{len(y)} mutations, {len(set(groups))} proteins\n")

    print(f"{'hand-tuned weights':<28} Pearson |r| = {abs(np.corrcoef(hand,y)[0,1]):.3f}"
          f"   Spearman |rho| = {abs(spearman(hand,y)):.3f}")

    # Grouped CV: never evaluate on a protein seen in training.
    gkf = GroupKFold(n_splits=10)
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        m = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    print(f"{'refitted (grouped 10-fold)':<28} Pearson |r| = "
          f"{abs(np.corrcoef(pred,y)[0,1]):.3f}   Spearman |rho| = {abs(spearman(pred,y)):.3f}")

    # Leakage check: the same fit without grouping, which is the number an
    # careless benchmark would report.
    from sklearn.model_selection import KFold
    naive = np.zeros_like(y)
    for tr, te in KFold(n_splits=10, shuffle=True, random_state=0).split(X):
        naive[te] = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(X[tr], y[tr]).predict(X[te])
    print(f"{'  (ungrouped, leaks)':<28} Pearson |r| = "
          f"{abs(np.corrcoef(naive,y)[0,1]):.3f}   <- inflated, do not quote\n")

    full = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(X, y)
    print("fitted weight per term (sign shows direction against ddG):")
    for t, w in sorted(zip(TERMS, full.coef_), key=lambda kv: -abs(kv[1])):
        print(f"  {t:<16} {w:+.4f}")
    np.save(Path(__file__).parent / "fitted_weights.npy", full.coef_)
    print(f"\nintercept {full.intercept_:+.4f}, alpha {full.alpha_:.3g}")


if __name__ == "__main__":
    main()
