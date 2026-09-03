import sys, csv, warnings, time; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
from pathlib import Path
import numpy as np
from proteus import from_pdb
from proteus.mpnn import MPNNScorer, _INDEX
from proteinmpnn import protein_mpnn_utils as U

ROOT=Path("benchmarks/s669")
def spearman(a,b):
    def rank(v):
        v=np.asarray(v,float); _,inv,cnt=np.unique(v,return_inverse=True,return_counts=True)
        o=np.argsort(v); r=np.empty(len(v),float); r[o]=np.arange(len(v),dtype=float)
        return (np.bincount(inv,weights=r)/cnt)[inv]
    return float(np.corrcoef(rank(a),rank(b))[0,1])

rows=list(csv.DictReader(open(ROOT/"ddg_experimental.csv",encoding="utf-8")))
by={}
for r in rows: by.setdefault((r["pdb_id"],r["chain"]),[]).append(r)

sc=MPNNScorer(); preds=[]; exps=[]; skip=0; mismatch=0
t0=time.time()
for i,((pid,ch),grp) in enumerate(sorted(by.items()),1):
    p=ROOT/"pdb"/(pid+".pdb")
    if not p.exists(): skip+=len(grp); continue
    try:
        st=from_pdb(str(p),chain=ch)
        lp,mseq=sc.log_probs(str(p),chain=ch)
    except Exception:
        skip+=len(grp); continue
    if st.sequence!=mseq:
        mismatch+=1; skip+=len(grp); continue     # indices would not correspond
    pm={x.pdb_number:x.resi for x in st}
    for r in grp:
        try: pos=int(r["pos"]); wt=r["pre"]; mu=r["post"]; e=float(r["ddG_experimental"])
        except Exception: skip+=1; continue
        idx=pm.get(pos)
        if idx is None or st.sequence[idx-1]!=wt: skip+=1; continue
        wi,mi=_INDEX.get(wt),_INDEX.get(mu)
        if wi is None or mi is None: skip+=1; continue
        preds.append(float(lp[idx-1,mi]-lp[idx-1,wi])); exps.append(e)
    if i%30==0: print(f"  {i}/{len(by)} ({time.time()-t0:.0f}s)",flush=True)

preds,exps=np.array(preds),np.array(exps)
print(f"\nscored {len(preds)}/{len(rows)}  skipped {skip}  seq-mismatch proteins {mismatch}")
print(f"ProteinMPNN  |r| = {abs(np.corrcoef(preds,exps)[0,1]):.3f}   |rho| = {abs(spearman(preds,exps)):.3f}")
np.save("/tmp/mpnn_ok.npy",np.vstack([preds,exps]))
