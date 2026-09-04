"""Minimal local server for the Proteus frontend. No auth, localhost only."""
import sys, tempfile, traceback
import threading, time, urllib.request, urllib.error
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, request, jsonify, send_from_directory

from proteus import DesignContext, from_pdb
from proteus import membrane as membrane_mod
from proteus.engine import Engine
from proteus.scoring import FittedScorer, HeuristicScorer
from proteus.strategies import REGISTRY
from proteus.validate import PredictedStructureGate
from proteus import mpnn as _mpnn

app = Flask(__name__, static_folder=str(Path(__file__).parent))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _load(pdb_text, chain, membrane):
    tmp = Path(tempfile.gettempdir()) / "proteus_web_input.pdb"
    tmp.write_text(pdb_text, encoding="utf-8")
    structure = from_pdb(str(tmp), chain=chain or None)
    mem = membrane_mod.estimate(structure) if membrane else None
    return DesignContext(structure=structure, membrane=mem), structure


@app.post("/api/analyze")
def analyze():
    data = request.get_json(force=True)
    try:
        ctx, structure = _load(data["pdb"], data.get("chain"), data.get("membrane"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    breakdown = HeuristicScorer().score(ctx, structure.sequence)
    fitted = FittedScorer().score(ctx, structure.sequence)
    applicable = REGISTRY.applicable(ctx)

    return jsonify({
        "n_residues": len(structure),
        "sequence": structure.sequence,
        "score": breakdown.per_residue,
        "terms": breakdown.terms,
        "fitted_score": fitted.total,
        # Measured accuracy, so neither number is ever shown without the
        # context that says how much to trust it. Both are |Pearson r| against
        # 669 experimental ddG values (S669), 10-fold CV grouped by protein.
        "accuracy": {
            "heuristic_r": 0.125,
            "fitted_r": 0.296,
            "mpnn_r": 0.331,
            "mpnn_available": _mpnn.available(),
            "best_published_r": 0.460,
            "best_published_name": "ACDC-NN",
            "foldx_r": 0.214,
            "benchmark": "S669 (669 mutations, 94 proteins)",
        },
        "strategies": [
            {"name": s.name, "sites": len(s.diagnose(ctx)), "mechanism": s.mechanism}
            for s in sorted(applicable, key=lambda s: -len(s.diagnose(ctx)))
        ],
    })


GATES = {
 "core_packing": "layer = core AND residue in {A,C,G,S,T}",
 "cavity_fill": "burial >= core+1.0 AND local sidechain volume below the core median within 8 A",
 "surface_depolarize": "layer = surface AND residue hydrophobic",
 "salt_bridge": "CB-CB 4-9 A, charged-group tips within 4 A (acid reach 3.2 A, base reach 5.2 A), |i-j|>=3; in a helix only i,i+3 / i,i+4",
 "disulfide": "CB-CB 3.0-4.5 A, CA-CA 4.0-6.5 A, |i-j|>=4, excludes existing pairs, max 3 new",
 "helix_capping": "N-cap -> S/T/D/N, C-cap -> G/N, helix length >= 5",
 "loop_rigidify": "Pro only where SS=L and -90 <= phi <= -40; Gly replaced only if phi < 0",
 "helix_propensity": "SS=H AND Chou-Fasman helix propensity < 0.9",
 "beta_propensity": "SS=E AND sheet propensity < 0.9",
 "deamidation_motif": "residue N followed by G/S/N/T/A, not buried",
 "isomerisation_motif": "residue D followed by G/P/S/D, not buried",
 "glycosylation_sequon": "N-X-[S/T] with X != P, surface only",
 "free_cysteine": "Cys with no partner within CB-CB 4.5 A, not buried",
 "methionine_oxidation": "Met on the surface",
 "arginine_preference": "surface Lys -> Arg",
 "thermolabile_amide": "surface N or Q -> charged",
 "surface_charge_enrichment": "surface S/T/N/Q -> D/E/K/R",
 "salt_bridge_network": ">= 2 existing charges within 8 A of an uncharged surface position; completes with the under-represented sign",
 "helix_dipole": "within 2 of a helix terminus (helix >= 7): N-term -> D/E, C-term -> K/R",
 "capping_box": "position N3 of a helix (>= 7) -> E/Q, pairs with the N-cap",
 "aromatic_cluster": "buried non-aromatic with an existing aromatic at CB-CB 4.5-7.5 A, max 2",
 "cation_pi": "boundary layer, 4.5-6.5 A from a buried aromatic, capped at 4 sites (ring geometry not computable from CB alone)",
 "buried_unsatisfied_polar": "core polar with no polar partner within 6.5 A",
 "beta_edge_protection": "surface strand residue with <= 1 non-adjacent strand neighbour within 6 A",
 "beta_turn": "positive-phi position inside a 2-4 residue loop flanked by strands",
 "lipid_facing_hydrophobic": "zone = lipid_core AND layer = surface AND currently polar/charged",
 "aromatic_belt": "zone = interface AND exposed AND not already W/Y",
 "snorkeling": "zone = interface, |depth| >= core_half - 1, surface",
 "positive_inside": "non-lipid zone, surface, on the inferred cytoplasmic side",
 "interhelical_polar": "buried in bilayer, core layer, partner at CB-CB 4-7 A, max 4",
 "hydrophobic_mismatch": "within 2.5 A of the hydrophobic boundary with the wrong character",
 "glycine_zipper": "membrane-buried helix position with a small residue at i+/-4, max 3",
 "terminal_anchor": "helix terminus within 4 A of the bilayer boundary",
}

FAMILY = {
 "core": ["core_packing","cavity_fill","surface_depolarize","salt_bridge","disulfide",
          "helix_capping","loop_rigidify","helix_propensity","beta_propensity"],
 "liability": ["deamidation_motif","isomerisation_motif","glycosylation_sequon",
               "free_cysteine","methionine_oxidation"],
 "thermophile": ["arginine_preference","thermolabile_amide","surface_charge_enrichment",
                 "salt_bridge_network","helix_dipole","capping_box"],
 "packing": ["aromatic_cluster","cation_pi","buried_unsatisfied_polar",
             "beta_edge_protection","beta_turn"],
 "membrane": ["lipid_facing_hydrophobic","aromatic_belt","snorkeling","positive_inside",
              "interhelical_polar","hydrophobic_mismatch","glycine_zipper","terminal_anchor"],
}
FAMILY_NOTE = {
 "core": "Classic protein engineering: pack the interior, clean the surface, cap and rigidify.",
 "liability": "Chemical degradation routes -- how a protein falls apart over weeks rather than unfolding in seconds. Long-lived natural proteins are depleted in these motifs; a fresh design carries them at background frequency.",
 "thermophile": "Taken from comparing thermophile proteins with mesophile orthologues: same fold, same function, different operating temperature. Notably the differences are almost all on the surface, not in the core.",
 "packing": "Energy that depends on which pair of residues sit near each other, so it is invisible to any per-residue rule.",
 "membrane": "Inside the bilayer the burial rules invert: lipid-exposed wants hydrophobic, and the bundle interior tolerates polar.",
}


@app.get("/api/strategies")
def strategies():
    out = []
    for family, names in FAMILY.items():
        for name in names:
            try:
                s = REGISTRY.get(name)
            except KeyError:
                continue
            out.append({"name": name, "family": family,
                        "environments": sorted(s.applies_to),
                        "mechanism": s.mechanism,
                        "gate": GATES.get(name, "")})
    return jsonify({"strategies": out, "notes": FAMILY_NOTE})


@app.post("/api/run")
def run():
    data = request.get_json(force=True)
    try:
        ctx, structure = _load(data["pdb"], data.get("chain"), data.get("membrane"))
        frozen = frozenset()
        if data.get("freeze"):
            frozen = set()
            for chunk in data["freeze"].split(","):
                chunk = chunk.strip()
                if "-" in chunk:
                    lo, hi = chunk.split("-")
                    frozen.update(range(int(lo), int(hi) + 1))
                elif chunk:
                    frozen.add(int(chunk))
            frozen = frozenset(frozen)
            ctx = DesignContext(structure=structure, membrane=ctx.membrane, frozen=frozen)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400

    generations = int(data.get("generations", 40))
    chosen = data.get("strategies") or None

    # Scorer choice. The gain threshold is left to calibrate itself for
    # anything but the hand-tuned default, whose 0.0003 was measured against
    # that scorer alone and means nothing on another scale.
    which = (data.get("scorer") or "heuristic").lower()
    scorer, min_gain = None, 0.0003
    if which == "fitted":
        scorer, min_gain = FittedScorer(), None
    elif which in ("mpnn", "proteinmpnn"):
        if not _mpnn.available():
            return jsonify({"error": "ProteinMPNN is not installed here. "
                                     "pip install torch proteinmpnn"}), 400
        scorer = _mpnn.MPNNSequenceScorer(chain=data.get("chain") or None)
        min_gain = None

    try:
        engine = Engine(ctx, seed=0, allowed_strategies=chosen,
                        scorer=scorer, min_gain_per_mutation=min_gain)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result = engine.run(generations=min(generations, 200))

    prov = result.provenance()
    changes = [
        {"pos": p, "from": old, "to": new, "why": why}
        for p, (old, new, why) in sorted(prov.items())
    ]

    return jsonify({
        "start_sequence": structure.sequence,
        "best_sequence": result.best_sequence,
        "n_mutations": result.n_mutations,
        "identity": result.identity,
        "improvement": result.improvement,
        "changes": changes,
        "credit": result.credit(),
        "mode": "manual" if chosen else "autonomous",
        "scorer": which,
        "min_gain": engine.min_gain_per_mutation,
        "leaderboard": [
            {"name": a.name, "pulls": a.pulls, "win_rate": a.success_rate,
             "mean_reward": a.mean_reward}
            for a in result.policy.leaderboard() if a.pulls
        ] if result.policy else [],
    })


#: Public ESMFold endpoint. Free, no key, but it is a shared service: it rate
#: limits, refuses long sequences, and goes down. Every failure mode below is
#: reported to the caller rather than swallowed, because a silent failure here
#: would leave a design looking verified when nothing checked it.
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_LEN = 400


def fold_with_esmfold(sequence: str, timeout: int = 120,
                      attempts: int = 7) -> tuple[str | None, str | None]:
    """Fold a sequence remotely, retrying transient failures.

    The endpoint is free and unauthenticated, and it behaves like it. Measured
    directly while building this: the same sequence succeeded three times, then
    504'd eight times consecutively, then succeeded again. A second sequence
    showed the opposite pattern minutes later. Failures are neither
    sequence-specific nor a clean outage -- they arrive in correlated bursts,
    at roughly coin-flip odds overall.

    Retrying is therefore worth doing but cannot be relied on. Gateway errors
    and timeouts back off and retry; a persistent failure is reported plainly
    so the manual path can be used instead.

    Client errors (4xx) are not retried: those mean the request itself is
    wrong, and repeating it will not help.
    """
    if len(sequence) > ESMFOLD_MAX_LEN:
        return None, (f"sequence is {len(sequence)} residues; the public ESMFold "
                      f"endpoint accepts up to about {ESMFOLD_MAX_LEN}. Fold this "
                      f"one elsewhere and upload the result.")

    last = ""
    for attempt in range(attempts):
        if attempt:
            time.sleep(min(2.0 * attempt, 8.0))
        req = urllib.request.Request(
            ESMFOLD_URL, data=sequence.encode("ascii"), method="POST",
            headers={"Content-Type": "text/plain"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code < 500:
                return None, (f"ESMFold rejected the request ({last}). This usually "
                              f"means the sequence contains characters it cannot fold.")
            continue
        except Exception as exc:
            last = type(exc).__name__
            continue
        if "ATOM" in body:
            return body, None
        last = body[:120]

    return None, (f"ESMFold did not respond after {attempts} attempts ({last}). "
                  f"It is a free shared service and goes down regularly -- "
                  f"fold externally and upload the result instead.")


# --------------------------------------------------------------- fold jobs
#
# The upstream service fails in bursts, so a single synchronous request is a
# coin flip. Folding therefore runs as a background job that keeps retrying
# for several minutes while the page stays usable, and the client polls. This
# turns an unreliable dependency into a slow but dependable one.
FOLD_JOBS: dict[str, dict] = {}
FOLD_JOB_LOCK = threading.Lock()
FOLD_MAX_MINUTES = 8


def _fold_worker(job_id: str, sequence: str, reference_pdb: str,
                 chain: str | None, rmsd_cutoff: float) -> None:
    deadline = time.time() + FOLD_MAX_MINUTES * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        with FOLD_JOB_LOCK:
            FOLD_JOBS[job_id]["attempt"] = attempt
        pdb_text, err = fold_with_esmfold(sequence, attempts=1)
        if pdb_text:
            result = {"state": "done", "pdb": pdb_text, "attempt": attempt}
            if reference_pdb:
                try:
                    _, reference = _load(reference_pdb, chain, False)
                    tmp = Path(tempfile.gettempdir()) / f"proteus_fold_{job_id}.pdb"
                    tmp.write_text(pdb_text, encoding="utf-8")
                    predicted = from_pdb(str(tmp))
                    if len(predicted) != len(reference):
                        result["compare_error"] = (
                            f"folded {len(predicted)} residues but the reference has "
                            f"{len(reference)}")
                    else:
                        check = PredictedStructureGate(
                            predicted, rmsd_cutoff=rmsd_cutoff
                        ).check(predicted.sequence, reference)
                        result.update({"passed": check.passed, "rmsd": check.sc_rmsd,
                                       "plddt": check.plddt, "reason": check.reason})
                except Exception as exc:
                    result["compare_error"] = str(exc)
            with FOLD_JOB_LOCK:
                FOLD_JOBS[job_id] = result
            return
        # Non-retryable errors come back worded differently; stop on those.
        if err and "rejected the request" in err:
            with FOLD_JOB_LOCK:
                FOLD_JOBS[job_id] = {"state": "failed", "error": err}
            return
        time.sleep(6)

    with FOLD_JOB_LOCK:
        FOLD_JOBS[job_id] = {
            "state": "failed",
            "error": (f"ESMFold did not respond in {FOLD_MAX_MINUTES} minutes "
                      f"({attempt} attempts). The free service is in a bad "
                      f"patch -- fold externally and upload the result."),
        }


@app.post("/api/fold/start")
def fold_start():
    data = request.get_json(force=True)
    sequence = (data.get("sequence") or "").strip()
    if not sequence:
        return jsonify({"error": "no sequence supplied"}), 400
    if len(sequence) > ESMFOLD_MAX_LEN:
        return jsonify({"error": f"sequence is {len(sequence)} residues; the public "
                                 f"ESMFold endpoint accepts up to about "
                                 f"{ESMFOLD_MAX_LEN}. Fold externally instead."}), 400

    job_id = uuid.uuid4().hex[:12]
    with FOLD_JOB_LOCK:
        FOLD_JOBS[job_id] = {"state": "running", "attempt": 0}
    threading.Thread(
        target=_fold_worker, daemon=True,
        args=(job_id, sequence, data.get("pdb") or "", data.get("chain"),
              float(data.get("rmsd", 2.0)))).start()
    return jsonify({"job": job_id, "max_minutes": FOLD_MAX_MINUTES})


@app.get("/api/fold/status/<job_id>")
def fold_status(job_id):
    with FOLD_JOB_LOCK:
        job = FOLD_JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.post("/api/fold")
def fold():
    """Fold a sequence and compare it against the reference backbone in one step."""
    data = request.get_json(force=True)
    sequence = (data.get("sequence") or "").strip()
    if not sequence:
        return jsonify({"error": "no sequence supplied"}), 400

    pdb_text, err = fold_with_esmfold(sequence)
    if err:
        return jsonify({"error": err}), 502

    result = {"pdb": pdb_text, "n_residues": len(sequence)}

    # If a reference was supplied, do the self-consistency check immediately.
    if data.get("pdb"):
        try:
            _, reference = _load(data["pdb"], data.get("chain"), False)
            tmp = Path(tempfile.gettempdir()) / "proteus_folded.pdb"
            tmp.write_text(pdb_text, encoding="utf-8")
            predicted = from_pdb(str(tmp))
            if len(predicted) != len(reference):
                result["compare_error"] = (
                    f"folded {len(predicted)} residues but the reference has "
                    f"{len(reference)}; they must correspond position for position")
            else:
                gate = PredictedStructureGate(
                    predicted, rmsd_cutoff=float(data.get("rmsd", 2.0)))
                check = gate.check(predicted.sequence, reference)
                result.update({"passed": check.passed, "rmsd": check.sc_rmsd,
                               "plddt": check.plddt, "reason": check.reason})
        except Exception as exc:
            traceback.print_exc()
            result["compare_error"] = str(exc)
    return jsonify(result)


@app.post("/api/validate")
def validate():
    data = request.get_json(force=True)
    try:
        _, reference = _load(data["pdb"], data.get("chain"), False)
        pred_tmp = Path(tempfile.gettempdir()) / "proteus_web_pred.pdb"
        pred_tmp.write_text(data["predicted_pdb"], encoding="utf-8")
        predicted = from_pdb(str(pred_tmp), chain=data.get("predicted_chain") or None)
        if len(predicted) != len(reference):
            return jsonify({"error": f"length mismatch: reference "
                            f"{len(reference)} vs prediction {len(predicted)}"}), 400
        gate = PredictedStructureGate(predicted, rmsd_cutoff=float(data.get("rmsd", 2.0)))
        res = gate.check(predicted.sequence, reference)
        return jsonify({"passed": res.passed, "rmsd": res.sc_rmsd,
                        "plddt": res.plddt, "reason": res.reason})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(port=8420, debug=False)
