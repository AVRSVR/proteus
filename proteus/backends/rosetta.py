"""PyRosetta backend: real packing, minimisation and energies.

Proteus explores with a cheap heuristic objective and refines with this. That
ordering is deliberate. A Rosetta FastRelax costs seconds to minutes per call,
so putting it inside a loop that evaluates tens of thousands of candidates is
the wrong place for it; the loop narrows the field, and Rosetta answers whether
the survivor is genuinely better under a real energy function.

Two things the earlier prototype got wrong are fixed structurally here:

*Frozen means frozen.* The prototype listed a binding face as "frozen" but left
the resfile default at ``ALLAA`` and gave FastRelax an unrestricted MoveMap, so
those residues were redesigned and their backbone moved every cycle. Here the
frozen set is applied in both places -- ``NATRO`` in the resfile and false in
the MoveMap -- because either alone is insufficient.

*Energies are compared, not thresholded.* The prototype drove total REU toward
a fixed -250, which is unreachable for a small protein and trivial for a large
one. This backend reports the change relative to the input pose, per residue.

PyRosetta is licence-gated and not on PyPI, so every import is lazy and this
module is importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..context import DesignContext
from ..proposals import Resolution, to_resfile
from ..structure import Structure

#: Energy function for each environment. franklin2019 carries an implicit
#: membrane term; scoring a membrane protein with ref2015 applies soluble
#: burial physics to a bilayer and is simply the wrong function.
WEIGHTS_SOLUBLE = "ref2015"
WEIGHTS_MEMBRANE = "franklin2019"

DEFAULT_INIT_FLAGS = (
    "-ex1 -ex2aro -use_input_sc -flip_HNQ "
    "-no_optH false -ignore_unrecognized_res true -mute all"
)


@dataclass
class RefinementResult:
    """What a Rosetta refinement produced."""

    sequence: str
    score_before: float
    score_after: float
    n_residues: int
    weights: str
    pdb_path: str | None = None

    @property
    def delta(self) -> float:
        """Total energy change; negative is an improvement."""
        return self.score_after - self.score_before

    @property
    def delta_per_residue(self) -> float:
        return self.delta / max(self.n_residues, 1)

    def describe(self) -> str:
        return (f"{self.weights}: {self.score_before:.2f} -> {self.score_after:.2f} REU "
                f"({self.delta:+.2f} total, {self.delta_per_residue:+.4f}/residue)")


def available() -> bool:
    """Whether PyRosetta can be imported in this interpreter."""
    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        return False
    return True


def require() -> None:
    if not available():
        raise ImportError(
            "PyRosetta is not installed. It is licence-gated and not on PyPI:\n"
            "    pip install pyrosetta-installer\n"
            "    python -c \"import pyrosetta_installer; "
            "pyrosetta_installer.install_pyrosetta()\"\n"
            "Free for academic use; commercial use needs a licence."
        )


class RosettaBackend:
    """Packing, relax and scoring through PyRosetta."""

    def __init__(self, membrane: bool = False, weights: str | None = None,
                 init_flags: str = DEFAULT_INIT_FLAGS) -> None:
        self.membrane = membrane
        self.weights = weights or (WEIGHTS_MEMBRANE if membrane else WEIGHTS_SOLUBLE)
        self.init_flags = init_flags
        self._initialised = False
        self._scorefxn = None

    # ------------------------------------------------------------------ setup

    def init(self) -> None:
        """Initialise PyRosetta once per process."""
        if self._initialised:
            return
        require()
        import pyrosetta

        pyrosetta.init(self.init_flags)
        self._initialised = True

    def scorefxn(self):
        if self._scorefxn is None:
            self.init()
            import pyrosetta

            self._scorefxn = pyrosetta.create_score_function(self.weights)
        return self._scorefxn

    # ------------------------------------------------------------------ poses

    def pose_from_pdb(self, path: str):
        self.init()
        import pyrosetta

        pose = pyrosetta.pose_from_pdb(str(path))
        if self.membrane:
            self._add_membrane(pose)
        return pose

    def _add_membrane(self, pose) -> None:
        """Attach an implicit membrane, which franklin2019 requires.

        Without a membrane residue and a span definition the membrane score
        terms have no geometry to act on, and franklin2019 silently degrades
        toward a soluble energy -- the failure is quiet, which is why this is
        not optional.
        """
        from pyrosetta.rosetta.protocols.membrane import AddMembraneMover

        AddMembraneMover("from_structure").apply(pose)

    # -------------------------------------------------------------- operations

    def score(self, pose) -> float:
        return float(self.scorefxn()(pose))

    def design(self, pose, resfile_path: str, frozen: frozenset[int] = frozenset()):
        """Repack and design according to a resfile, then relax.

        ``frozen`` is applied to the MoveMap in addition to the resfile. The
        resfile controls which *identities* may change; the MoveMap controls
        which *coordinates* may move. Freezing a binding face requires both --
        the prototype set neither and moved everything.
        """
        self.init()
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task import operation as task_op
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        sfxn = self.scorefxn()

        task_factory = TaskFactory()
        task_factory.push_back(task_op.InitializeFromCommandline())
        task_factory.push_back(task_op.IncludeCurrent())
        task_factory.push_back(task_op.ReadResfile(str(resfile_path)))

        packer = PackRotamersMover()
        packer.task_factory(task_factory)
        packer.score_function(sfxn)
        packer.apply(pose)

        move_map = MoveMap()
        n = pose.total_residue()
        for i in range(1, n + 1):
            movable = i not in frozen
            move_map.set_bb(i, movable)
            move_map.set_chi(i, movable)

        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.set_movemap(move_map)
        relax.constrain_relax_to_start_coords(True)
        relax.apply(pose)
        return pose

    # ------------------------------------------------------------------ entry

    def refine(
        self,
        pdb_path: str,
        resolution: Resolution,
        structure: Structure,
        frozen: frozenset[int] = frozenset(),
        out_pdb: str | None = None,
    ) -> RefinementResult:
        """Score a structure, apply a design specification, relax, rescore."""
        import tempfile
        from pathlib import Path

        pose = self.pose_from_pdb(pdb_path)
        before = self.score(pose)

        resfile_text = to_resfile(resolution, structure)
        tmp = Path(tempfile.gettempdir()) / "proteus_design.resfile"
        tmp.write_text(resfile_text, encoding="utf-8")

        self.design(pose, str(tmp), frozen)
        after = self.score(pose)

        if out_pdb:
            pose.dump_pdb(str(out_pdb))

        return RefinementResult(
            sequence=pose.sequence(),
            score_before=before,
            score_after=after,
            n_residues=pose.total_residue(),
            weights=self.weights,
            pdb_path=out_pdb,
        )


def build_resolution_from_sequence(ctx: DesignContext, sequence: str) -> Resolution:
    """Turn a designed sequence into a resfile specification.

    Positions whose identity differs from the input are pinned to the new
    residue; everything else is left at the resfile default of ``NATRO``. This
    is how a sequence produced by the cheap loop is handed to Rosetta for real
    packing without reopening the whole protein to design.
    """
    from ..proposals import Resolution as _Resolution

    original = ctx.structure.sequence
    if len(sequence) != len(original):
        raise ValueError(
            f"sequence length {len(sequence)} != structure {len(original)}"
        )
    allowed = {
        i + 1: frozenset(new)
        for i, (old, new) in enumerate(zip(original, sequence))
        if old != new and (i + 1) not in ctx.frozen
    }
    return _Resolution(allowed=allowed)
