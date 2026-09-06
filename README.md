# Proteus

**A strategy library for stabilizing designed proteins — soluble and membrane-embedded.**

**[Live demo](https://proteus-5kpn.onrender.com/)** · **[Source](https://github.com/AVRSVR/proteus)**

*(Hosted on Render's free tier — the instance sleeps when idle, so the first request after a pause pays a ~30s cold start.)*

Generative models produce protein backbones and sequences that look right. Whether they *hold together* is a separate question. Proteus is the layer after generation: given a structure, it diagnoses which stabilization mechanisms the fold actually admits, applies them, and keeps a record of which ones earned their place.

The distinguishing idea is that Proteus reasons at the level of **mechanisms**, not mutations. Most stability tooling asks "what is ΔΔG for L47I?". Proteus asks "does this fold have an underpacked core, an exposed hydrophobic patch, or an uncapped helix — and which of those is worth fixing here?" That abstraction is interpretable, and unlike a per-mutation model it can transfer between proteins.

```bash
proteus analyze design.pdb --membrane
proteus run design.pdb --freeze 1-10,47-53 --generations 60
```

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AVRSVR/proteus)

---

## Why membrane proteins are a first-class path

The bilayer inverts the rule that every soluble-protein heuristic depends on.

|                      | Soluble protein        | Membrane protein (inside the bilayer) |
| -------------------- | ---------------------- | ------------------------------------- |
| Solvent-exposed      | wants **polar**        | wants **hydrophobic**                 |
| Buried against protein | wants **hydrophobic** | tolerates and uses **polar**          |

A stabilization engine that doesn't know where the membrane is will confidently do the exact wrong thing — a naive "exposed → make it polar" rule strips the lipid-facing surface off a GPCR. So Proteus carries **membrane depth as a coordinate orthogonal to burial**, everywhere. Burial answers *how much protein is in front of this sidechain*; depth answers *where is it relative to the bilayer*. Neither alone determines what belongs at a position.

On the test structures these two signals correlate at |r| < 0.5, which is the property that makes the membrane path a genuine second code path rather than a flag.

---

## Running it

Core dependencies are `numpy` and `biopython`. Nothing else.

```bash
pip install -e ".[dev]"

proteus analyze design.pdb                          # diagnose, change nothing
proteus analyze design.pdb --membrane               # invert the burial rules
proteus run design.pdb --freeze 1-10,47-53 --explain
proteus run design.pdb --knowledge kb.json          # accumulate across runs
proteus leaderboard kb.json --for other.pdb         # what should work here?
proteus strategies                                  # the library, with mechanisms
proteus validate design.pdb --predicted folded.pdb  # does it still fold?
```

### How much it is allowed to change

A stabilization tool that rewrites half the sequence has not stabilized the protein, it has designed a different one. Established campaigns (PROSS, FRESCO) change single-digit percentages of positions.

Two mechanisms hold the edit down, and they do different jobs:

- `--mutation-budget 0.15` — a hard ceiling, as a fraction of designable positions.
- `--min-gain 0.0003` — the per-residue score improvement a mutation must deliver to be kept.

The second is also what makes the tool **specific** -- see "Does it know what to fix?" below for the 28-structure test of that claim. In short: gain-per-mutation separates a broken design from a sound one by 63x, which is why an absolute floor near 0.0003 repairs a broken design and leaves evolved proteins alone.

Two earlier formulations failed, and both failures are worth recording. Expressing the price as `cost / n_residues` made it **size-dependent** — the same setting was six times cheaper per mutation on a 400-residue protein than on a 66-residue one, so large proteins accumulated dozens of marginal edits. Expressing it *relative* to the gain observed within each run **normalised away exactly the signal that distinguishes a broken protein from a sound one**, and every protein ran to the budget ceiling.

The threshold is absolute and therefore tied to this scorer's scale. Changing the scorer's terms or weights means re-measuring it; the table above is the procedure.

```
mutations        : 8 of at most 9 allowed
sequence identity: 87.9% retained
```

---

## The web app

The same engine behind a browser UI: load a structure, see which mechanisms the
fold admits, run a search, and check the result actually refolds.

```bash
pip install -r requirements.txt
python webapp/server.py            # http://127.0.0.1:8420
```

`PORT` and `HOST` are read from the environment if you need to move it.

Folding is delegated to the public **ESMFold API** rather than a local model,
which is what keeps the deployment small enough to be free. The consequence is
that the fold gate depends on a third-party service: when it is down or rate
limiting, folds fail and the app says so instead of pretending. Folds run as
background jobs the browser polls, so a slow one does not hold a request open.

### Deploying

`render.yaml` is a working Render blueprint. The **Deploy to Render** button
above reads it directly — click it, connect the GitHub repo, and Render builds
from `requirements.txt`. Deploying manually is the same thing without the
button: create a Blueprint instance in the Render dashboard and point it at
this repo.

Two details in it are load-bearing rather than cosmetic:

- **One worker, eight threads.** Fold and search jobs live in per-process
  dictionaries driven by background threads. A second worker would serve a
  status poll from a process that never saw the job and report it missing.
  Threads are the right axis anyway, since almost all of the wall time is spent
  waiting on the ESMFold API.
- **A 600-second timeout.** A search holds its worker thread for the whole run,
  and a single fold is allowed eight minutes.

`requirements.txt` deliberately omits torch. ProteinMPNN scoring is detected at
import (`proteus.mpnn.available`) and the two analytic scorers are used when it
is missing — so the hosted build runs without it and the UI reports MPNN as
unavailable rather than failing. Installing `torch` and `proteinmpnn` in the
environment is all that is needed to turn it on, at roughly 900 MB, which is
past what a free instance will hold.

Free-tier instances sleep when idle, so the first request after a pause pays a
cold start.

---

## How it works

```
structure ──► DesignContext ──► strategies diagnose ──► proposals
                  │                                         │
       burial ────┤                                    conflict
       secondary  │                                    resolution
       structure  │                                         │
       membrane   │                                         ▼
       depth  ────┘                                   realize sequence
                                                            │
                   in-run bandit  ◄── accept/reject ────────┘
                         │             (Metropolis)
                         ▼
                  KnowledgeBase  ◄──► fingerprint
                  (across runs)       (structural context)
```

1. **Analyze once.** `DesignContext` computes burial, secondary structure and membrane depth. Strategies never recompute geometry; they ask the context questions.
2. **Diagnose.** Each strategy reports where — if anywhere — the fold exhibits the weakness its mechanism addresses. A strategy that diagnoses nothing is never sampled and is never charged a failure.
3. **Propose.** Strategies emit `Proposal` objects: allowed residues at a position, plus the reasoning.
4. **Resolve.** Overlapping proposals are intersected. Genuine incompatibilities are arbitrated by weight **and recorded**, so a combination that fights itself is visible rather than silent.
5. **Accept or reject.** Metropolis with an annealed temperature calibrated from the observed score scale.
6. **Credit.** A bandit policy (UCB1 or Thompson) updates the in-run leaderboard, and the run is folded back into a persistent knowledge base.

## The leaderboard that transfers

A bandit inside one run learns which mechanisms help *this* protein, then throws it away. The claim worth keeping is conditional — *core packing works on proteins with an underpacked core* — so Proteus tags every observation with a **structural fingerprint** and stores it.

```bash
proteus run design_a.pdb --knowledge kb.json     # cold start
proteus run design_b.pdb --knowledge kb.json     # arrives with priors
proteus leaderboard kb.json --for design_c.pdb   # what should work here?
```

```
knowledge base: 40 observations from previous runs
this protein   : soluble, ~66 res, core 5% / surface 74%, H 94% E 0%
most similar seen before: 2a3d_raw (0.68)
transferred priors for 6 strategies
```

The fingerprint is 16 size-independent features, so a 70-residue miniprotein and a 900-residue enzyme land in the same space. Priors enter as *pseudo-observations* weighted by similarity, not as a ranking — a strategy that worked on near-identical folds gets a head start that real evidence from the current run washes out. Transfer is a hint, never a verdict.

The result is a leaderboard that changes with context. Conditioned on a β-rich protein, `beta_propensity` appears; conditioned on an all-helical one it vanishes and `helix_capping` takes its place. That conditionality is the whole point — an unconditional ranking is a fact about the last protein you ran.

Deliberate limitation: credit is **joint**. When two mechanisms are applied together and the result improves, both are rewarded. Raw per-arm history is retained so the ambiguity can be analysed rather than hidden.

---

## Strategy library

33 mechanisms in five families, each carrying a description of the biophysics it exploits and each gated on real geometry or a real sequence motif.

**Core stability** (soluble) — `core_packing`, `cavity_fill`, `surface_depolarize`, `salt_bridge`, `disulfide`, `helix_capping`, `loop_rigidify`, `helix_propensity`, `beta_propensity`

**Chemical liabilities** — the routes by which a protein degrades over weeks rather than unfolds in seconds. Long-lived natural proteins are measurably depleted in these motifs; a design has no history filtering them out.
`deamidation_motif`, `isomerisation_motif`, `glycosylation_sequon`, `free_cysteine`, `methionine_oxidation`

**Thermophile-inspired** — from comparing thermophile proteins against mesophile orthologues: same fold, same function, different operating temperature. The differences are mostly on the *surface*, not in the core.
`arginine_preference`, `thermolabile_amide`, `surface_charge_enrichment`, `salt_bridge_network`, `helix_dipole`, `capping_box`

**Pairwise interactions and sheet architecture** — energy invisible to any per-residue rule, because it depends on which *pair* of residues sit near each other.
`aromatic_cluster`, `cation_pi`, `buried_unsatisfied_polar`, `beta_edge_protection`, `beta_turn`

**Membrane** — burial rules inverted inside the bilayer.
`lipid_facing_hydrophobic`, `aromatic_belt`, `snorkeling`, `positive_inside`, `interhelical_polar`, `hydrophobic_mismatch`, `glycine_zipper`, `terminal_anchor`

```bash
proteus strategies    # full descriptions
```

### What is deliberately absent

The prototype this replaces had strategies called `zipper`, `teflon`, `alanine_shave` and `lactam_staple`. None survived. The first three lower a score without a mechanism behind them. Lactam stapling is a real technique, but a real lactam needs i,i+4 or i,i+7 on the same helical face and a modelled covalent bond; picking two surface residues and making one lysine and one glutamate produces an unrelated Lys and Glu and no staple.

The bar for inclusion is that a mechanism can be *gated* — that there is a geometric or sequence criterion distinguishing where it applies from where it does not. A strategy that fires everywhere carries no information for the selector, which is why `cation_pi` is capped: ring-face geometry cannot be computed from backbone atoms and CB alone, so it under-claims rather than guesses.

Adding one requires no geometry code:

```python
from proteus import Strategy, register
from proteus.proposals import Proposal, AROMATIC

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    mechanism = "One sentence on the biophysics being exploited."
    applies_to = frozenset({"soluble"})

    def diagnose(self, ctx):
        return [p for p in ctx.designable if ctx.layer(p) == "core"]

    def propose(self, ctx, positions, rng):
        return [Proposal(p, AROMATIC, self.name, "why here") for p in positions]
```

---

## Engineering notes

This is a rewrite of an earlier prototype. The defects it fixes are worth stating, because each one is a way a plausible-looking design loop produces meaningless output:

**Frozen residues weren't frozen.** The prototype excluded its binding face from the strategy rules, but the resfile default was `ALLAA`, so every unmentioned position — including the entire nominally-protected face — was redesigned every generation. Proteus defaults to `NATRO` and rejects proposals at frozen positions in the resolver. There is a test asserting no strategy can touch a frozen residue, for every strategy in both environments.

**The target was unreachable.** It optimised *total* Rosetta energy toward a fixed −250 REU. Total energy scales with chain length, so for a ~54-residue protein that target sits outside the physically reachable range — the loop could never terminate, and everything downstream of it, including the entire MD validation branch, was unreachable code. Scores here are per residue, and the objective is improvement relative to the input.

**Disulfides had no geometry.** It picked two random surface residues and mutated both to cysteine. Random pairs essentially never satisfy disulfide geometry, so the result was two free cysteines — an oxidation and aggregation liability — rather than a crosslink. Pairs are now gated on Cβ–Cβ (3.0–4.5 Å), Cα–Cα (4.0–6.5 Å) and sequence separation.

**Proline was periodic.** It placed proline at every fourth residue. Proline pins φ near −60° and cannot donate a backbone hydrogen bond, so this breaks helices and strains loops. Proline is now proposed only where the backbone already sits in its allowed region. Relatedly, glycines at *positive* φ are conformationally required and are left alone.

**Burial was distance-from-centroid.** That only behaves for roughly spherical globules; on a helical bundle it labels buried termini as surface. Proteus uses the cone-based sidechain-neighbour count (the measure Rosetta's `LayerSelector` uses).

**Strategies overwrote each other silently.** They wrote into a shared resfile where the last line won, so the meaning of a strategy *combination* was decided by dict ordering. Conflicts are now explicit and logged.

**The leaderboard couldn't stop exploring or start exploiting.** Weights grew by +0.5 per success without bound, so an early winner dominated sampling forever. Selection is now a bandit with proper confidence accounting.

### Found by auditing the strategies against real structures

Every strategy was run in isolation on a validated de novo design, a prototype output, gp120, and an AlphaFold model. Five more defects surfaced:

- **A mechanism the objective cannot measure always loses, regardless of merit.** Loop rigidification and helix capping both scored exactly `0.00000` because the scorer had no term for backbone entropy or cap satisfaction — so selection could never learn anything about them. This is the systematic version of the bug, and it is why `bb_entropy` and `capping` terms exist.
- **`salt_bridge` diagnosed 90–95% of all residues** (68/73 on 2A3D), which carries no information. Two attempts were needed: requiring the CA→CB vectors to point at each other rejected *every* real pair, because on a helix face both vectors point outward and are nearly parallel — the bridge is made by sidechains reaching laterally, not by the CB atoms. A tip-reach model fixed it. Now 1–8%, and on 2A3D the hits are i,i+3 and i,i+4.
- **`disulfide` reported 82 sites and emitted one pair**, overstating applicability to the selector by an order of magnitude, and re-proposed bonds that already existed on a disulfide-rich protein.
- **`cavity_fill` made the score consistently worse.** It targeted marginally-buried positions, which are boundary-like, so the aromatics it proposed exposed ring surface and cost more in aggregation than they gained in packing.
- **Scoring didn't scale.** The aggregation term was an O(n²) Python loop on every evaluation — 121 ms for a 3000-residue protein, and the audit that found this had timed out. Now vectorised with a cached sparse neighbour list: 3.6 ms, a 33× improvement.

### Found while demonstrating the tool

- **Temperature has no meaningful absolute scale.** A fixed default was 40× the observed delta scale, making acceptance 97% — a random walk. Now calibrated from the first few moves, giving ~50%.
- **Strategies oscillated across generations.** `disulfide` wrote F31C, `core_packing` wrote C31F, forever. The resolver can't see this because the conflict spans generations. A short tabu list fixes it.
- **Aggregation was scored on the wrong axis.** Kyte-Doolittle rates Trp and Tyr as hydrophilic because they are amphipathic, so a surface covered entirely in tryptophan scored `0.000` — identical to poly-lysine. Aromatics are among the worst aggregation offenders. Now on a propensity scale with charged residues as gatekeepers.
- **The fingerprint measured damage, not identity.** Aggregation load and exposed hydrophobic fraction contributed 95% of the distance, so two ~70-residue helical bundles came out 0.18 similar because one was damaged. Features are now split into identity and state groups with the latter damped; the same pair now scores 0.68.

---

## Honest limitations

- **The scorer is a screening heuristic, not a force field.** This is the limitation everything else is downstream of. `HeuristicScorer` keeps the terms driving a decision readable, and holds them in deliberate tension (packing against aggregation, charge against burial) to resist the degenerate corners a single-term objective invites. It correctly separates a score-hacked design from an experimentally validated one. It is still not a stability prediction.
- **No repacking or minimisation.** Sequences are threaded onto a fixed backbone; sidechains are never rebuilt. Real packing needs an energy function and a packer, neither of which is here.
- **The membrane estimator needs a plausible starting sequence.** It locates the bilayer from exposed hydrophobicity. If the design under repair has that backwards — exactly the case Proteus exists to fix — the estimate is unreliable. Supply an OPM-oriented structure, or an explicit `MembraneModel`. There is a test asserting this failure mode rather than hiding it.
- **Credit assignment is joint.** When two mechanisms are applied together and the result improves, both are rewarded. Raw per-arm history is retained so the ambiguity can be analysed.
- **The knowledge base is only as good as the objective.** It faithfully learns which mechanisms improve `HeuristicScorer`. Whether that tracks real stability is the open question, and matters more than adding more strategies.
- **One chain, under 400 residues.** The fold gate calls the public ESMFold endpoint, which is capped at roughly 400 residues; longer sequences are refused before folding starts. More importantly, every chain in an uploaded file is read as a **single unit with no interface treatment** — a two-chain complex is scored as though the chains were one protein, so burial at the interface is counted as if it were interior packing. Neither the scorer nor the fold gate is doing anything meaningful on a complex. Use the `chain` field to pick one chain out, and treat complexes as outside the domain.
- **Some files will not load at all.** Only residues with a complete N/CA/C backbone are kept, and anything else is dropped rather than guessed at, so a file can load with far fewer residues than it appears to contain. Two inputs fail outright: solvated or topology files past 99,999 atoms, which switch to a numbering scheme the parser cannot read (strip waters and ions first), and anything under 20 residues, which has too few neighbours for burial or packing to mean anything. Both now report the cause instead of surfacing a parser error.
- **The optimiser itself still cannot see the fold.** The scorer knows nothing about whether a sequence folds; that check is a separate gate applied afterwards, and the search loop exists precisely because the optimiser proposes sequences the gate then rejects. The loop makes the failure recoverable, not absent — a design that passes has been refolded and measured, but the objective driving the search remains a heuristic that a determined optimiser can walk away from.

---

## Verification

Nothing in the optimisation confirms the design still folds. That check is separate, and deliberately decoupled from any particular predictor:

```bash
proteus validate design.pdb --predicted esmfold_output.pdb
```

```
PASS  scRMSD 1.34 A, pLDDT 88.2  (refolds to the intended backbone)
FAIL  scRMSD 4.94 A  (scRMSD 4.94 > 2.0 A)
```

Fold the designed sequence with whatever you have — ESMFold or AlphaFold on Colab, a cluster, a colleague's GPU — and hand the file over. Proteus does the part that needs care. pLDDT is read from the B-factor column, where AlphaFold and ESMFold both write it.

Every RMSD is superposed with Kabsch first, including a determinant correction so a mirror image cannot pass. This is not incidental: computing deviation as a raw coordinate difference, as the earlier prototype's MD did, measures rigid-body tumbling rather than shape change — a correct structure rotated 30° scores over 5 Å that way. The tests assert rotation and translation invariance directly, and check the naive quantity is large in the same test so the guard cannot silently stop testing anything.

`ESMFoldGate` will run the model in-process if you have the memory for it (roughly 16 GB; `esmfold_v1` bundles ESM-2 3B). It has not been executed here — the file-based path has.

---

## A negative result worth keeping

Protein language model likelihood looks like a cheap way to check a designed sequence is plausible. It was tested and rejected.

Scoring 2A3D, an experimentally validated de novo three-helix bundle, against a synthetic poly-Trp/Tyr repeat with ESM-2 650M:

| sequence | mean log-likelihood / residue |
| --- | --- |
| 2A3D (validated design) | −0.326 |
| poly-WY junk | **−0.013** |

The junk scores *better*. A repetitive sequence is trivially predictable — given `WWWWYYYY`, the next `W` is easy — so likelihood rewards low complexity. As a quality gate this would rank exactly the score-hacked aromatic designs this project exists to catch above a real one. (The measurement was a single-pass approximation rather than true masked pseudo-likelihood, but the direction is a documented failure mode and disqualifying either way.)

Verifying a design needs structure, not sequence plausibility. There is no cheap version, which is why nothing here claims to do it.

---

## Does it know what to fix?

The mutation floor (`--min-gain`) is only defensible if gain-per-mutation actually separates broken designs from sound proteins -- otherwise it is an arbitrary knob. This was tested rather than assumed.

```bash
python benchmarks/specificity.py benchmarks/manifest.json
```

23 outputs from an earlier, defective design loop (`God_Particle_*`, the direct predecessor to this rewrite), one experimentally validated de novo design (2A3D), and four natural proteins (CDC42, PGK1, GRIN1, gp120). Proteus is not told which class anything belongs to; it only reports what it found.

| class | n | median gain/mutation |
| --- | --- | --- |
| broken AI designs | 23 | 0.000458 |
| validated design (2A3D) | 1 | 0.000158 |
| natural proteins | 4 | 0.000007 |

**63x separation** between the broken-design median and the natural-protein median, and a threshold exists (0.000282) that classifies this set with 100% accuracy. The default floor of 0.0003 sits almost exactly there, which is not a coincidence -- it was set from an earlier four-protein version of this same measurement.

Two things keep this from being stronger evidence than it is, and both are worth stating plainly. All 23 broken designs came from one defective loop, most from the same starting backbone at different checkpoints -- they are highly correlated, closer to n=2 than n=23. And there is exactly one validated design, which is the case that actually matters most: a *good* design that the tool must learn to leave alone. A single point cannot establish where that boundary sits.

The honest reading is not "specificity is proven." It is "a discriminating signal was found where none was designed in, the effect size is large, and the experiment that would actually test it -- many independent validated designs across different folds -- has not been run." That experiment is the natural next step for anyone extending this project, and `benchmarks/specificity.py` is built to make it a manifest edit rather than a rewrite.

## Roadmap

- Strategy interaction map: which mechanisms are synergistic, which interfere
- Per-strategy credit assignment — currently joint when mechanisms are co-applied
- Cross-protein transfer evaluated properly: does a prior from similar folds measurably beat a cold start?
- A real energy function behind the objective, which is the limitation everything else is downstream of
- **A molecular-dynamics stability screen**, written and parked on the `md-screen` branch rather than merged. It runs both structures through an implicit-solvent forcefield with replicates, so the spread between runs of the *same* structure sets a noise floor and a smaller before/after difference is reported as indistinguishable instead of resolved. Two measured things stop it shipping: 10 ps of dynamics on 1140 atoms takes 332 s on the CPU platform, making a default screen roughly 2.5 hours, so it needs the OpenCL path; and the synthetic test bundle starts at +6.3e6 kJ/mol and cannot be minimised, so the integration tests need a real PDB. The arithmetic deciding what a run *means* is already covered by 12 tests that need no OpenMM.
- Mutational recovery on S669: apply a known experimentally-destabilizing mutation to the wild type and measure how often the tool reverts that exact position against a random-position baseline. This is the missing piece — it would test whether the **strategies** work, where the benchmark above only tests whether the **scorer** correlates.

## Tests

```bash
pytest -q     # 162 tests
```

Tests run against synthetic structures built from ideal φ/ψ via NeRF, so the geometry has a known answer by construction rather than depending on downloaded PDBs.

## License

MIT
