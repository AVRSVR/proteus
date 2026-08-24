"""Run Proteus + Rosetta refinement on Kaggle or Colab.

PyRosetta ships no Windows wheel, so the Rosetta backend needs a Linux host.
Kaggle and Colab both work and both have a free tier. Paste this into a
notebook cell, or run it as a script with the paths edited.

The division of labour is the point: the cheap heuristic loop explores locally
on any machine, and Rosetta adjudicates the finalist where a licence and a
Linux wheel are available.
"""

# --- 1. Dependencies -------------------------------------------------------
# In a notebook, run these as a cell first:
#
#   !pip install -q pyrosetta-installer
#   import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()
#   !pip install -q git+https://github.com/<you>/proteus   # or upload the package
#
# PyRosetta is free for academic use. Commercial use needs a licence.

import glob
import sys

from proteus import DesignContext, from_pdb
from proteus import membrane as membrane_mod
from proteus.backends import rosetta
from proteus.engine import Engine

# --- 2. Inputs -------------------------------------------------------------

PDB = next(iter(glob.glob("/kaggle/input/**/*.pdb", recursive=True)), None)
if PDB is None:
    sys.exit("upload a PDB to the Kaggle input directory")

IS_MEMBRANE = False
FROZEN = frozenset()          # e.g. frozenset(range(1, 11)) | frozenset(range(47, 54))
GENERATIONS = 60

# --- 3. Cheap exploration --------------------------------------------------

structure = from_pdb(PDB)
membrane = membrane_mod.estimate(structure) if IS_MEMBRANE else None
ctx = DesignContext(structure=structure, frozen=FROZEN, membrane=membrane)

print(ctx.summary())
print()

result = Engine(ctx, seed=0, protein=PDB).run(generations=GENERATIONS, verbose=True)
print()
print(result.summary())
print()
print(result.explain(limit=20))

# --- 4. Rosetta adjudication ----------------------------------------------

if not rosetta.available():
    rosetta.require()

resolution = rosetta.build_resolution_from_sequence(ctx, result.best_sequence)
print(f"\n{len(resolution.allowed)} positions to apply in Rosetta")

backend = rosetta.RosettaBackend(membrane=IS_MEMBRANE)
refined = backend.refine(
    PDB, resolution, structure,
    frozen=FROZEN,
    out_pdb="/kaggle/working/proteus_refined.pdb",
)
print(refined.describe())

# A negative delta means Rosetta agrees the design is better. A positive one
# means the heuristic objective and the real energy function disagree, which
# is information worth having rather than a failure to hide.
verdict = "agrees" if refined.delta < 0 else "DISAGREES with"
print(f"\nRosetta {verdict} the heuristic objective.")

# --- 5. Optional: refold check --------------------------------------------
#
#   !pip install -q "transformers>=4.35" accelerate
#
#   from proteus.validate import ESMFoldGate
#   check = ESMFoldGate().check(result.best_sequence, structure)
#   print(check.describe())
#
# On a GPU runtime this takes seconds; on CPU, minutes.
