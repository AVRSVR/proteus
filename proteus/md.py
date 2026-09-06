"""A short molecular-dynamics screen, run before and after a design.

Every other check in Proteus is a score or a structure prediction. This one is
physics: put the structure in a forcefield, let it move, and watch whether it
holds its shape. It answers a question the scorer cannot -- does this thing
stay folded when it is allowed to fall apart?

What it is not: a free-energy calculation. Comparing radius of gyration across
a few tens of picoseconds tells you whether a design comes apart quickly, not
what its dG of folding is. A protein can sit still for the whole trajectory and
still be less stable than the one it replaced. Read a pass here as "nothing
obviously broke", and read a fail as real.

Three properties make the difference between a screen and a coin flip, and all
three are deliberate:

* **Replicates.** A single trajectory's drift is dominated by which velocities
  it happened to start with. Run the same structure three times from different
  seeds and the spread across them is the noise floor -- a difference smaller
  than that is not a result. One run each would produce confident nonsense.
* **Equilibration is discarded.** The first picoseconds are the structure
  relaxing out of whatever the predictor handed us, which is not the signal.
* **The comparison is paired.** Both structures get identical treatment, so
  systematic error in the forcefield or the solvent model largely cancels.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

#: Discarded before measurement: the structure settling into the forcefield.
EQUILIBRATION_PS = 5.0
#: Measured. Short on purpose -- this is a screen, not a production run.
PRODUCTION_PS = 45.0
#: Below this the spread between replicates swamps any real difference.
MIN_REPLICATES = 2
DEFAULT_REPLICATES = 3

TIMESTEP_FS = 2.0
TEMPERATURE_K = 310.0
FRICTION_PER_PS = 1.0
REPORT_INTERVAL_PS = 1.0


def available() -> bool:
    try:
        import openmm  # noqa: F401
        import pdbfixer  # noqa: F401
        return True
    except Exception:
        return False


def require() -> None:
    if not available():
        raise ImportError(
            "molecular dynamics needs OpenMM and PDBFixer, which are not "
            "installed. They are conda packages rather than wheels: "
            "conda install -c conda-forge openmm pdbfixer")


@dataclass
class Trajectory:
    """One replicate."""

    rg: list[float] = field(default_factory=list)
    rmsd: list[float] = field(default_factory=list)
    seed: int = 0

    @property
    def rg_mean(self) -> float:
        return sum(self.rg) / len(self.rg) if self.rg else float("nan")

    @property
    def rg_drift(self) -> float:
        """Late-trajectory expansion, in angstrom.

        The last quarter of the window minus the *second* quarter. Positive
        means the structure was still opening up when the run ended, which is
        what coming apart looks like at this length.

        The first quarter is skipped rather than used as the baseline. Five
        picoseconds of equilibration does not always finish settling a
        predicted structure into the forcefield, and a single early jump that
        is followed by a flat trajectory is the minimiser letting go, not the
        protein expanding -- measured from frame zero it would be reported as
        drift equal to the whole jump. The cost is that expansion happening
        entirely within the first quarter is invisible here; that is the right
        trade, because that is exactly the region where relaxation and
        expansion cannot be told apart anyway.
        """
        if len(self.rg) < 8:
            return float("nan")
        q = len(self.rg) // 4
        early = sum(self.rg[q:2 * q]) / q
        late = sum(self.rg[-q:]) / q
        return late - early

    @property
    def rmsd_final(self) -> float:
        return self.rmsd[-1] if self.rmsd else float("nan")


@dataclass
class Screen:
    """Every replicate of one structure."""

    trajectories: list[Trajectory] = field(default_factory=list)
    n_atoms: int = 0
    n_residues: int = 0

    @staticmethod
    def _spread(values: list[float]) -> float:
        usable = [v for v in values if v == v]
        if len(usable) < 2:
            return float("nan")
        m = sum(usable) / len(usable)
        return math.sqrt(sum((v - m) ** 2 for v in usable) / (len(usable) - 1))

    @staticmethod
    def _mean(values: list[float]) -> float:
        usable = [v for v in values if v == v]
        return sum(usable) / len(usable) if usable else float("nan")

    @property
    def rg_mean(self) -> float:
        return self._mean([t.rg_mean for t in self.trajectories])

    @property
    def rg_drift(self) -> float:
        return self._mean([t.rg_drift for t in self.trajectories])

    @property
    def rg_drift_spread(self) -> float:
        """The noise floor.

        Standard deviation of drift across replicates of the *same* structure.
        A before/after difference smaller than this is thermal noise wearing a
        result's clothing.
        """
        return self._spread([t.rg_drift for t in self.trajectories])

    @property
    def rmsd_final(self) -> float:
        return self._mean([t.rmsd_final for t in self.trajectories])

    @property
    def rmsd_spread(self) -> float:
        return self._spread([t.rmsd_final for t in self.trajectories])


def _prepare(pdb_text: str):
    """Protonate and repair a predicted structure so a forcefield accepts it."""
    from pdbfixer import PDBFixer

    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.findMissingResidues()
    # Terminal gaps are dropped rather than modelled: inventing a loop nothing
    # predicted would put atoms in the trajectory that no method placed there.
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain_index, position = key
        chain = chains[chain_index]
        if position == 0 or position == len(list(chain.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def _rg_and_rmsd(positions, masses, reference):
    """Mass-weighted radius of gyration, and RMSD against the start frame.

    No superposition: the Langevin thermostat does not translate or rotate the
    system appreciably over this window, and a design that tumbles has bigger
    problems than this measurement.
    """
    total = sum(masses)
    cx = sum(m * p[0] for m, p in zip(masses, positions)) / total
    cy = sum(m * p[1] for m, p in zip(masses, positions)) / total
    cz = sum(m * p[2] for m, p in zip(masses, positions)) / total
    rg2 = sum(m * ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)
              for m, p in zip(masses, positions)) / total
    n = len(positions)
    sq = sum((positions[i][0] - reference[i][0]) ** 2
             + (positions[i][1] - reference[i][1]) ** 2
             + (positions[i][2] - reference[i][2]) ** 2 for i in range(n))
    return math.sqrt(rg2), math.sqrt(sq / n)


def simulate(pdb_text: str, *, replicates: int = DEFAULT_REPLICATES,
             production_ps: float = PRODUCTION_PS,
             equilibration_ps: float = EQUILIBRATION_PS,
             seed: int = 0, should_stop=None) -> Screen:
    """Run several short implicit-solvent trajectories of one structure."""
    require()
    if replicates < MIN_REPLICATES:
        raise ValueError(
            f"{replicates} replicate(s) cannot separate a real difference from "
            f"thermal noise; use at least {MIN_REPLICATES}")

    import openmm
    from openmm import app, unit

    fixer = _prepare(pdb_text)
    # Implicit solvent: there is no water box to equilibrate, which is the only
    # reason a run this short says anything at all.
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(
        fixer.topology, nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer, constraints=app.HBonds)

    heavy = [a.index for a in fixer.topology.atoms()
             if a.element is not None and a.element.symbol != "H"]
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in heavy]

    screen = Screen(n_atoms=system.getNumParticles(),
                    n_residues=fixer.topology.getNumResidues())
    steps_per_report = max(1, int(REPORT_INTERVAL_PS * 1000 / TIMESTEP_FS))
    equil_steps = int(equilibration_ps * 1000 / TIMESTEP_FS)
    n_reports = max(8, int(production_ps / REPORT_INTERVAL_PS))
    platform = openmm.Platform.getPlatformByName("CPU")

    for r in range(replicates):
        if should_stop is not None and should_stop():
            break
        integrator = openmm.LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond,
            TIMESTEP_FS * unit.femtosecond)
        integrator.setRandomNumberSeed(seed + r + 1)
        sim = app.Simulation(fixer.topology, system, integrator, platform)
        sim.context.setPositions(fixer.positions)
        sim.minimizeEnergy(maxIterations=500)
        sim.context.setVelocitiesToTemperature(
            TEMPERATURE_K * unit.kelvin, seed + r + 1)
        sim.step(equil_steps)

        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        # Drift is measured from the post-equilibration frame, not the input:
        # from where the forcefield put it, not where the predictor did.
        reference = [pos[i] for i in heavy]

        traj = Trajectory(seed=seed + r + 1)
        for _ in range(n_reports):
            if should_stop is not None and should_stop():
                break
            sim.step(steps_per_report)
            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            rg, rmsd = _rg_and_rmsd([pos[i] for i in heavy], masses, reference)
            traj.rg.append(float(rg))
            traj.rmsd.append(float(rmsd))
        screen.trajectories.append(traj)

    return screen


def compare(before_pdb: str, after_pdb: str, **kw) -> dict:
    """Screen two structures identically and report the paired difference.

    The verdict is deliberately conservative. A design is called worse only
    when its extra drift exceeds the noise floor measured from the replicates
    themselves; anything inside that band is reported as indistinguishable
    rather than resolved in whichever direction the means happen to fall.
    """
    before = simulate(before_pdb, **kw)
    after = simulate(after_pdb, **kw)

    d_drift = after.rg_drift - before.rg_drift
    noise = max(before.rg_drift_spread, after.rg_drift_spread)

    if d_drift != d_drift or noise != noise:
        verdict = "unknown"
        note = "not enough replicates finished to compare the two structures"
    elif abs(d_drift) <= noise:
        verdict = "indistinguishable"
        note = (f"the {abs(d_drift):.2f} A difference in drift is inside the "
                f"{noise:.2f} A spread between replicates of the same "
                "structure, so this run cannot separate them")
    elif d_drift > 0:
        verdict = "worse"
        note = (f"the design expands {d_drift:.2f} A more than the input over "
                f"the same window, against a {noise:.2f} A noise floor")
    else:
        verdict = "better"
        note = (f"the design holds {abs(d_drift):.2f} A tighter than the input "
                f"over the same window, against a {noise:.2f} A noise floor")

    def summary(s: Screen) -> dict:
        return {"rg_mean": s.rg_mean, "rg_drift": s.rg_drift,
                "rg_drift_spread": s.rg_drift_spread,
                "rmsd_final": s.rmsd_final, "rmsd_spread": s.rmsd_spread,
                "replicates": len(s.trajectories), "n_residues": s.n_residues}

    return {"verdict": verdict, "note": note, "delta_rg_drift": d_drift,
            "noise_floor": noise,
            "before": summary(before), "after": summary(after)}
