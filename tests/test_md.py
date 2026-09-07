"""The MD screen's arithmetic and its refusal to over-claim.

The simulation itself needs OpenMM, which is a conda package and not present
everywhere, so the trajectory-running tests skip when it is missing. The parts
that decide *what a run means* -- the drift measure, the noise floor, and the
verdict -- are pure arithmetic and are tested unconditionally, because those
are where a wrong answer would be confidently wrong rather than absent.
"""

import pytest

from proteus import md


def traj(rg_values, rmsd=1.0, seed=0):
    t = md.Trajectory(seed=seed)
    t.rg = list(rg_values)
    t.rmsd = [rmsd] * len(rg_values)
    return t


def screen(*trajectories):
    return md.Screen(trajectories=list(trajectories))


# ------------------------------------------------------------------ drift

def test_drift_is_the_last_quarter_minus_the_second():
    """The first quarter is skipped, so a steady ramp reports 13 - 11."""
    t = traj([10.0] * 4 + [11.0] * 4 + [12.0] * 4 + [13.0] * 4)
    assert t.rg_drift == pytest.approx(2.0)


def test_a_structure_holding_its_size_has_no_drift():
    assert traj([12.0] * 16).rg_drift == pytest.approx(0.0)


def test_drift_is_nan_when_there_are_too_few_frames():
    """Better to say nothing than to average four numbers into a verdict."""
    assert traj([10.0] * 4).rg_drift != traj([10.0] * 4).rg_drift


def test_early_transient_does_not_dominate_drift():
    """The first frames are the minimiser relaxing, not the signal.

    A structure that jumps once at the start and is flat afterwards should not
    be reported as expanding, which is what a slope over the whole window would
    have said.
    """
    t = traj([10.0] * 2 + [12.0] * 14)
    flat = traj([12.0] * 16)
    assert t.rg_drift == pytest.approx(0.0)
    assert flat.rg_drift == pytest.approx(0.0)


# ------------------------------------------------------------- noise floor

def test_noise_floor_needs_more_than_one_replicate():
    s = screen(traj([10.0] * 8 + [11.0] * 8))
    assert s.rg_drift_spread != s.rg_drift_spread      # NaN


def test_noise_floor_is_the_spread_between_identical_runs():
    a = traj([10.0] * 8 + [11.0] * 8)                  # drift +1
    b = traj([10.0] * 8 + [13.0] * 8)                  # drift +3
    s = screen(a, b)
    assert s.rg_drift == pytest.approx(2.0)
    assert s.rg_drift_spread == pytest.approx(1.4142, abs=1e-3)


def test_summary_statistics_ignore_unmeasurable_replicates():
    """A replicate that was cancelled mid-run must not poison the mean."""
    good = traj([10.0] * 8 + [11.0] * 8)
    stopped = traj([10.0, 10.0])                       # too few frames
    s = screen(good, stopped)
    assert s.rg_drift == pytest.approx(1.0)


# ---------------------------------------------------------------- verdict

def _verdict(before_trajs, after_trajs, monkeypatch):
    runs = iter([screen(*before_trajs), screen(*after_trajs)])
    monkeypatch.setattr(md, "simulate", lambda *a, **k: next(runs))
    return md.compare("BEFORE", "AFTER")


def test_a_difference_inside_the_noise_floor_is_not_a_result(monkeypatch):
    """The failure mode this whole module exists to avoid.

    Two runs of the same structure differ by chance. If that spread is larger
    than the before/after difference, reporting a direction would be reporting
    thermal noise as a finding.
    """
    before = [traj([10.0] * 8 + [11.0] * 8), traj([10.0] * 8 + [13.0] * 8)]
    after = [traj([10.0] * 8 + [11.2] * 8), traj([10.0] * 8 + [13.2] * 8)]
    out = _verdict(before, after, monkeypatch)
    assert out["verdict"] == "indistinguishable"
    assert "inside" in out["note"]


def test_a_design_that_expands_more_is_called_worse(monkeypatch):
    before = [traj([10.0] * 8 + [10.1] * 8), traj([10.0] * 8 + [10.2] * 8)]
    after = [traj([10.0] * 8 + [14.0] * 8), traj([10.0] * 8 + [14.1] * 8)]
    out = _verdict(before, after, monkeypatch)
    assert out["verdict"] == "worse"
    assert out["delta_rg_drift"] > 0


def test_a_design_that_holds_tighter_is_called_better(monkeypatch):
    before = [traj([10.0] * 8 + [14.0] * 8), traj([10.0] * 8 + [14.1] * 8)]
    after = [traj([10.0] * 8 + [10.1] * 8), traj([10.0] * 8 + [10.2] * 8)]
    out = _verdict(before, after, monkeypatch)
    assert out["verdict"] == "better"
    assert out["delta_rg_drift"] < 0


def test_verdict_is_unknown_when_nothing_could_be_measured(monkeypatch):
    out = _verdict([traj([10.0, 10.0])], [traj([10.0, 10.0])], monkeypatch)
    assert out["verdict"] == "unknown"


def test_one_replicate_is_refused():
    """A single trajectory has no noise floor, so it cannot be interpreted."""
    with pytest.raises(ValueError, match="thermal noise"):
        md.simulate("", replicates=1)


# ------------------------------------------------------------- integration

@pytest.mark.skipif(not md.available(), reason="OpenMM/PDBFixer not installed")
def test_a_real_trajectory_runs_and_reports_finite_numbers():
    from pathlib import Path

    pdb = Path("examples/2a3d.pdb").read_text(encoding="utf-8")
    s = md.simulate(pdb, replicates=2, production_ps=10, equilibration_ps=1)
    assert len(s.trajectories) == 2
    for t in s.trajectories:
        assert len(t.rg) >= 8
        assert 5.0 < t.rg_mean < 60.0
    assert s.rg_drift_spread == s.rg_drift_spread      # not NaN


@pytest.mark.skipif(not md.available(), reason="OpenMM/PDBFixer not installed")
def test_a_clashing_structure_is_refused_before_it_wastes_a_run():
    """The synthetic bundles cannot be simulated, and must say so quickly.

    Built from ideal phi/psi via NeRF with virtual CB atoms, they start at
    +6.3e6 kJ/mol and rise under minimisation because atoms overlap. Left to
    run, dynamics produces `Particle coordinate is NaN` after nine minutes of
    setup, naming neither the structure nor the cause.
    """
    from pathlib import Path

    pdb = Path("examples/soluble_bundle.pdb").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        md.simulate(pdb, replicates=2, production_ps=10, equilibration_ps=1)


@pytest.mark.skipif(not md.available(), reason="OpenMM/PDBFixer not installed")
def test_the_fastest_available_platform_is_chosen():
    """CPU is 138x slower, which is the difference between usable and not."""
    _plat, name = md.best_platform()
    assert name in md.PLATFORM_PREFERENCE
    available = set()
    import openmm
    for i in range(openmm.Platform.getNumPlatforms()):
        available.add(openmm.Platform.getPlatform(i).getName())
    for preferred in md.PLATFORM_PREFERENCE:
        if preferred in available:
            assert name == preferred, "a faster platform was available"
            break


@pytest.mark.skipif(not md.available(), reason="OpenMM/PDBFixer not installed")
def test_replicates_of_one_structure_differ_only_by_noise():
    """Establishes that the noise floor is measuring what it claims to.

    The same structure run twice must not look like a different structure --
    if it did, every verdict would be meaningless.
    """
    from pathlib import Path

    pdb = Path("examples/2a3d.pdb").read_text(encoding="utf-8")
    s = md.simulate(pdb, replicates=2, production_ps=10, equilibration_ps=1)
    a, b = (t.rg_mean for t in s.trajectories)
    assert abs(a - b) < 2.0, "replicates of one structure diverged implausibly"


@pytest.mark.skipif(not md.available(), reason="OpenMM/PDBFixer not installed")
def test_a_structure_is_not_better_than_itself():
    """The null control: the same structure on both sides of the comparison.

    Any difference here is thermal noise by construction, so a large delta
    would mean the measure is reading something other than what it claims.
    Measured at the shipped settings (3 replicates, 45 ps) the delta is 0.06 A
    against a 0.09 A noise floor, which the verdict correctly refuses to
    resolve. This test uses a shorter run and asserts only the robust half --
    that the delta stays small -- because whether a 0.06 A difference lands
    inside a 0.09 A floor is itself a coin flip at low replicate counts, and a
    test that flips is worse than no test.
    """
    from pathlib import Path

    pdb = Path("examples/2a3d.pdb").read_text(encoding="utf-8")
    out = md.compare(pdb, pdb, replicates=2, production_ps=15,
                     equilibration_ps=2)
    assert abs(out["delta_rg_drift"]) < 0.5, (
        f"a structure drifted {out['delta_rg_drift']:.2f} A differently from "
        "itself; the measure is not reading thermal noise")
    assert out["before"]["n_residues"] == out["after"]["n_residues"] == 73
