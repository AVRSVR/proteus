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
from proteus import md as _md
from proteus import mpnn as _mpnn

app = Flask(__name__, static_folder=str(Path(__file__).parent))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


#: Below this there is not enough structure to diagnose a mechanism against.
#: Burial, secondary-structure segments and packing all need neighbours; a
#: short peptide has none, and every gate would either fire on everything or
#: nothing. Refusing is more honest than returning a confident empty answer.
MIN_RESIDUES = 20


def _load(pdb_text, chain, membrane):
    tmp = Path(tempfile.gettempdir()) / "proteus_web_input.pdb"
    tmp.write_text(pdb_text, encoding="utf-8")
    try:
        structure = from_pdb(str(tmp), chain=chain or None)
    except ValueError as exc:
        # Biopython raises a bare int() conversion error on two common inputs,
        # and the raw message names neither the file nor the cause. A solvated
        # system past 99,999 atoms switches to hybrid-36 numbering ("A000"),
        # which its parser cannot read.
        if "invalid literal for int()" in str(exc):
            raise ValueError(
                "this file's atom or residue numbering could not be read. It is "
                "usually a solvated or topology file past 99,999 atoms, which "
                "switches to a numbering scheme the parser does not support. "
                "Strip the waters and ions and upload the protein alone."
            ) from exc
        raise
    if len(structure) < MIN_RESIDUES:
        raise ValueError(
            f"only {len(structure)} residues with a complete N/CA/C backbone "
            f"were found; at least {MIN_RESIDUES} are needed to diagnose a "
            "mechanism. Check the chain selection, and note that residues "
            "missing backbone atoms are dropped rather than guessed at.")
    mem = membrane_mod.estimate(structure) if membrane else None
    return DesignContext(structure=structure, membrane=mem), structure


def _parse_freeze(spec):
    """Parse "1-10,47,53-60" into a frozenset of residue numbers."""
    if not spec:
        return frozenset()
    out = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return frozenset(out)


def _scorer_for(which, chain):
    """Build the requested scorer, or None for the hand-tuned default.

    Returning None rather than HeuristicScorer() matters: the engine treats a
    missing scorer as "use the default and its calibrated threshold", which is
    the only combination whose gain threshold was measured rather than derived.
    """
    which = (which or "heuristic").lower()
    if which == "fitted":
        return FittedScorer()
    if which in ("mpnn", "proteinmpnn"):
        _mpnn.require()
        return _mpnn.MPNNSequenceScorer(chain=chain or None)
    return None


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
            "heuristic_r": 0.179,
            "fitted_r": 0.303,
            "mpnn_r": 0.331,
            "mpnn_available": _mpnn.available(),
            "md_available": _md.available(),
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
        frozen = _parse_freeze(data.get("freeze"))
        if frozen:
            ctx = DesignContext(structure=structure, membrane=ctx.membrane,
                                frozen=frozen)
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


# ------------------------------------------------------------ search loop
#
# Propose, fold, check, and retry with a smaller edit until something folds.
# Runs as a background job because a single fold can take minutes and thirty
# of them will not fit inside a request.
SEARCH_JOBS: dict[str, dict] = {}
SEARCH_LOCK = threading.Lock()

#: Per-fold retry budget inside the loop. Deliberately smaller than the
#: standalone fold endpoint uses: there the user is waiting on one answer, here
#: a stuck fold blocks every remaining attempt, so it is better to give up on
#: one candidate quickly and come back to folding on the next.
SEARCH_FOLD_ATTEMPTS = 3


def _search_worker(job_id: str, pdb_text: str, chain: str | None,
                   membrane: bool, frozen, which: str, chosen,
                   max_attempts: int, generations: int, budget: float,
                   rmsd_cutoff: float, plddt_cutoff: float | None,
                   deadline: float) -> None:
    from proteus.search import search as run_search

    def publish(**kw):
        with SEARCH_LOCK:
            SEARCH_JOBS[job_id].update(kw)

    try:
        ctx, structure = _load(pdb_text, chain, membrane)
        if frozen:
            ctx = DesignContext(structure=structure, membrane=ctx.membrane,
                                frozen=frozen)
        scorer = _scorer_for(which, chain)
    except Exception as exc:
        traceback.print_exc()
        publish(state="failed", error=str(exc))
        return

    reference = structure

    def fold_and_check(sequence):
        if time.time() > deadline:
            return None, None, False, "time budget exhausted"
        pdb_out, err = fold_with_esmfold(sequence, attempts=SEARCH_FOLD_ATTEMPTS)
        if not pdb_out:
            return None, None, False, (err or "fold unavailable")[:120]
        try:
            tmp = Path(tempfile.gettempdir()) / f"proteus_search_{job_id}.pdb"
            tmp.write_text(pdb_out, encoding="utf-8")
            predicted = from_pdb(str(tmp))
            if len(predicted) != len(reference):
                return None, None, False, "length mismatch"
            check = PredictedStructureGate(
                predicted, rmsd_cutoff=rmsd_cutoff,
                plddt_cutoff=plddt_cutoff,
            ).check(predicted.sequence, reference)
            with SEARCH_LOCK:
                SEARCH_JOBS[job_id]["last_pdb"] = pdb_out
            return check.sc_rmsd, check.plddt, check.passed, check.reason
        except Exception as exc:
            return None, None, False, str(exc)[:120]

    # Fold the input sequence first. If the design does not refold to its own
    # backbone, no edit to it can, and every attempt below would spend minutes
    # proving the same thing. This has happened on a real input: an earlier
    # prototype output measured 3.62 A and pLDDT 64.7 unmutated, so its search
    # could never have succeeded regardless of what the strategies proposed.
    publish(phase="baseline")
    base_rmsd, base_plddt, base_passed, base_note = fold_and_check(
        structure.sequence)
    # Kept, not discarded. The MD screen needs a "before" structure from the
    # same predictor as the "after": comparing an ESMFold model against the
    # crystal or predicted input would measure the difference between two
    # modelling methods as much as the difference between two sequences.
    with SEARCH_LOCK:
        SEARCH_JOBS[job_id]["baseline_pdb"] = SEARCH_JOBS[job_id].get("last_pdb")
    publish(baseline={"sc_rmsd": base_rmsd, "plddt": base_plddt,
                      "passed": base_passed, "note": base_note})
    if base_rmsd is not None and not base_passed:
        publish(state="done", found=False, n_folded=1,
                stopped_because=(
                    "the input sequence does not refold to its own backbone "
                    f"(scRMSD {base_rmsd:.2f} A"
                    + (f", pLDDT {base_plddt:.1f}" if base_plddt is not None else "")
                    + "). No edit can repair a structure that was not "
                    "self-consistent to begin with, so the search was not run."),
                start_sequence=structure.sequence, best=None)
        return
    publish(phase="searching")

    def on_attempt(a):
        with SEARCH_LOCK:
            job = SEARCH_JOBS[job_id]
            job["attempts"].append({
                "index": a.index, "n_mutations": a.n_mutations,
                "identity": a.identity, "improvement": a.improvement,
                "sc_rmsd": a.sc_rmsd, "plddt": a.plddt,
                "passed": a.passed, "note": a.note,
                "budget": a.mutation_budget,
                "layers": list(a.layers),
                # Kept so a failure can be traced back to the exact edit.
                # Without it a 27 A result is a number with no explanation,
                # and reconstructing the attempt from its index reproduces
                # different mutations than the run actually made.
                "sequence": a.sequence,
            })
            if a.passed:
                job["winner_pdb"] = job.get("last_pdb")

    def should_stop():
        with SEARCH_LOCK:
            return SEARCH_JOBS[job_id].get("cancel") or time.time() > deadline

    try:
        res = run_search(ctx, fold_and_check, max_attempts=max_attempts,
                         generations=generations, start_budget=budget,
                         scorer=scorer, allowed_strategies=chosen,
                         should_stop=should_stop, on_attempt=on_attempt)
    except Exception as exc:
        traceback.print_exc()
        publish(state="failed", error=str(exc))
        return

    best = res.winner or res.closest
    publish(state="done",
            stopped_because=res.stopped_because,
            n_folded=res.n_folded,
            found=res.winner is not None,
            best=None if best is None else {
                "sequence": best.sequence, "n_mutations": best.n_mutations,
                "identity": best.identity, "improvement": best.improvement,
                "sc_rmsd": best.sc_rmsd, "plddt": best.plddt,
                "passed": best.passed, "index": best.index,
            },
            start_sequence=structure.sequence)


@app.post("/api/search/start")
def search_start():
    data = request.get_json(force=True)
    try:
        frozen = _parse_freeze(data.get("freeze"))
    except Exception as exc:
        return jsonify({"error": f"could not parse freeze: {exc}"}), 400

    which = (data.get("scorer") or "heuristic").lower()
    if which in ("mpnn", "proteinmpnn") and not _mpnn.available():
        return jsonify({"error": "ProteinMPNN is not installed here."}), 400

    max_attempts = max(1, min(int(data.get("max_attempts", 30)), 50))
    minutes = max(1, min(int(data.get("max_minutes", 45)), 180))
    job_id = uuid.uuid4().hex[:12]
    with SEARCH_LOCK:
        SEARCH_JOBS[job_id] = {"state": "running", "attempts": [],
                               "max_attempts": max_attempts}
    threading.Thread(
        target=_search_worker, daemon=True,
        args=(job_id, data["pdb"], data.get("chain") or None,
              bool(data.get("membrane")), frozen, which,
              data.get("strategies") or None, max_attempts,
              int(data.get("generations", 40)),
              float(data.get("mutation_budget", 0.15)),
              float(data.get("rmsd", 2.0)),
              # pLDDT 80 suits a natural protein; small de novo designs
              # routinely score lower even when the fold is right -- 2A3D, an
              # experimentally validated three-helix bundle, refolds at 78.9.
              # Exposed so the bar can match the kind of protein being judged,
              # and disabled entirely with 0.
              (lambda v: None if v is not None and float(v) <= 0
                         else (70.0 if v is None else float(v)))(
                  data.get("plddt")),
              time.time() + minutes * 60)).start()
    return jsonify({"job": job_id, "max_attempts": max_attempts,
                    "max_minutes": minutes})


@app.get("/api/search/status/<job_id>")
def search_status(job_id):
    with SEARCH_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        # last_pdb can be large; it is fetched separately when wanted.
        return jsonify({k: v for k, v in job.items()
                        if k not in ("last_pdb", "winner_pdb",
                                     "baseline_pdb")})


@app.post("/api/search/cancel/<job_id>")
def search_cancel(job_id):
    with SEARCH_LOCK:
        if job_id not in SEARCH_JOBS:
            return jsonify({"error": "unknown job"}), 404
        SEARCH_JOBS[job_id]["cancel"] = True
    return jsonify({"cancelled": True})


@app.get("/api/search/structure/<job_id>")
def search_structure(job_id):
    """The winning structure, or with ?which=baseline the unmutated one."""
    which = request.args.get("which", "")
    with SEARCH_LOCK:
        job = SEARCH_JOBS.get(job_id) or {}
        if which == "baseline":
            pdb_out = job.get("baseline_pdb")
        else:
            pdb_out = job.get("winner_pdb") or job.get("last_pdb")
    if not pdb_out:
        return jsonify({"error": "no folded structure for this job yet"}), 404
    return jsonify({"pdb": pdb_out})


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


MD_JOBS: dict[str, dict] = {}
MD_LOCK = threading.Lock()

#: A screen this size is minutes of CPU. Past it the wait stops being worth
#: what a run this short can tell you, and the answer belongs on a cluster.
MD_MAX_RESIDUES = 200


def _md_worker(job_id, before_pdb, after_pdb, replicates, production_ps):
    def publish(**kw):
        with MD_LOCK:
            MD_JOBS[job_id].update(kw)

    def cancelled():
        with MD_LOCK:
            return bool(MD_JOBS.get(job_id, {}).get("cancel"))

    try:
        publish(phase="input")
        result = _md.compare(before_pdb, after_pdb, replicates=replicates,
                             production_ps=production_ps,
                             should_stop=cancelled)
        publish(state="done", **result)
    except Exception as exc:
        traceback.print_exc()
        publish(state="failed", error=str(exc))


@app.post("/api/md/start")
def md_start():
    """Screen the input and the design under identical short MD runs."""
    data = request.get_json(force=True)
    if not _md.available():
        return jsonify({"error": (
            "molecular dynamics needs OpenMM and PDBFixer, which are conda "
            "packages and are not installed in this environment. The hosted "
            "build does not carry them; run the server locally to use this."
        )}), 400

    before_pdb = (data.get("before") or "").strip()
    after_pdb = (data.get("after") or "").strip()
    if not before_pdb or not after_pdb:
        return jsonify({"error": "both structures are required"}), 400

    # Counted off the input rather than trusted from the client.
    n_res = len({line[22:27] for line in before_pdb.splitlines()
                 if line.startswith("ATOM")})
    if n_res > MD_MAX_RESIDUES:
        return jsonify({"error": (
            f"{n_res} residues is past the {MD_MAX_RESIDUES} this screen will "
            "attempt; a run that size needs a cluster, not a web request."
        )}), 400

    replicates = max(_md.MIN_REPLICATES, min(5, int(data.get("replicates", 3))))
    production_ps = max(10.0, min(200.0, float(data.get("ps", 45.0))))

    job_id = uuid.uuid4().hex[:12]
    with MD_LOCK:
        MD_JOBS[job_id] = {"state": "running", "phase": "starting",
                           "replicates": replicates,
                           "production_ps": production_ps}
    threading.Thread(target=_md_worker, daemon=True,
                     args=(job_id, before_pdb, after_pdb, replicates,
                           production_ps)).start()
    return jsonify({"job": job_id})


@app.get("/api/md/status/<job_id>")
def md_status(job_id):
    with MD_LOCK:
        job = MD_JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.post("/api/md/cancel/<job_id>")
def md_cancel(job_id):
    with MD_LOCK:
        if job_id not in MD_JOBS:
            return jsonify({"error": "unknown job"}), 404
        MD_JOBS[job_id]["cancel"] = True
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Local development only. In a deployment gunicorn imports `app` directly
    # and this block never runs -- see render.yaml for the served command.
    import os
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 8420)), debug=False)
