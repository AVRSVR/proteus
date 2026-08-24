# Proteus

**A strategy library for stabilizing designed proteins — soluble and membrane-embedded.**

Generative models produce protein backbones and sequences that look right. Whether they *hold together* is a separate question. Proteus is the layer after generation: given a structure, it diagnoses which stabilization mechanisms the fold actually admits, applies them, and keeps a record of which ones earned their place.

The distinguishing idea is that Proteus reasons at the level of **mechanisms**, not mutations. Most stability tooling asks "what is ΔΔG for L47I?". Proteus asks "does this fold have an underpacked core, an exposed hydrophobic patch, or an uncapped helix — and which of those is worth fixing here?" That abstraction is interpretable, and unlike a per-mutation model it can transfer between proteins.

```bash
proteus analyze design.pdb --membrane
proteus run design.pdb --freeze 1-10,47-53 --generations 60
```

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

## Install

```bash
pip install -e ".[dev]"
```

Core requirements are `numpy` and `biopython` only. PyRosetta and OpenMM are optional extras — the library, its tests, and the built-in scorer all run without them, so the whole thing is reproducible without a licensed install.

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

15 mechanisms, each carrying a description of the biophysics it exploits.

**Soluble** — `core_packing`, `cavity_fill`, `surface_depolarize`, `salt_bridge`, `disulfide`, `helix_capping`, `loop_rigidify`, `helix_propensity`, `beta_propensity`

Each is gated on real geometry: disulfides on Cβ–Cβ and Cα–Cα distance, salt bridges on whether the charged groups can actually reach, proline on whether the backbone φ already permits it, cavity filling on a measured local volume deficit.

**Membrane** — `lipid_facing_hydrophobic`, `aromatic_belt`, `snorkeling`, `positive_inside`, `interhelical_polar`, `hydrophobic_mismatch`

```bash
proteus strategies    # full descriptions
```

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

- **The built-in scorer is a screening heuristic, not a force field.** `HeuristicScorer` exists so the engine runs and is testable without PyRosetta, and so the terms driving a decision are readable. Its terms are deliberately in tension (packing against aggregation, charge against burial) to resist the degenerate corners a single-term objective invites — but it is not a substitute for Rosetta, and improvements measured against it are *not* stability predictions.
- **No repacking or minimisation in the default backend.** Sequences are threaded onto a fixed backbone. Real sidechain placement needs the Rosetta backend.
- **The membrane estimator needs a plausible starting sequence.** It locates the bilayer from exposed hydrophobicity. If the design under repair has that backwards — exactly the case Proteus exists to fix — the estimate is unreliable. Supply an OPM-oriented structure, or an explicit `MembraneModel`. There is a test asserting this failure mode rather than hiding it.
- **Credit assignment is joint.** When two mechanisms are applied together and the result improves, both are rewarded. Raw per-arm history is retained so the ambiguity can be analysed.
- **The knowledge base is only as good as the objective.** It faithfully learns which mechanisms improve `HeuristicScorer`. Whether that tracks real stability is exactly the open question, and is why the Rosetta backend and a refold gate matter more than more strategies.
- **No refold check yet.** The single most valuable missing filter is threading the designed sequence back through a structure predictor and requiring self-consistency. Until that exists, nothing here verifies the sequence still encodes the fold.

---

## Roadmap

- Rosetta backend (`ref2015` for soluble, `franklin2019` for membrane) with real packing and relax
- ESMFold refold self-consistency gate — the highest-value missing filter
- OpenMM backend with **superposition-corrected** RMSD (unaligned RMSD measures tumbling, not deformation)
- Strategy interaction map: which mechanisms are synergistic, which interfere
- Disentangling joint credit, so a mechanism is rewarded for its own contribution

---

## Tests

```bash
pytest -q     # 83 tests
```

Tests run against synthetic structures built from ideal φ/ψ via NeRF, so the geometry has a known answer by construction rather than depending on downloaded PDBs.

## License

MIT
