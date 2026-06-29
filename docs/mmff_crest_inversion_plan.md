# MMFF↔CREST/MACE inversion — cheap levers to reach MMFF-disfavoured basins

Branch: `19-mmff-crestmace-inversion-investigation`. Implements issue #19 (follow-up to #17).

## 20260625

This thread follows directly from the v0.3 root-cause finding
(`docs/concerted_moves_v0_3_plan.md`): the `cremp_sharp` coverage gap is an
**MMFF↔MACE energy rank inversion**, not a sampling-move deficiency. The
CREST-dominant conformer (conf0) is ranked ~2.4 kcal/mol *above* our best sampled
basin by MMFF (which drives exploration) but ~10 kcal/mol *below* by MACE (which
only re-scores). Because MACE is downstream of MMFF, it can re-rank the basins
MMFF finds but cannot make MMFF *discover* a basin MMFF avoids. The goal here is
to find **cheap, MMFF/MC-only levers** that give the explorer access to
MMFF-disfavoured-but-CREST-real geometries — **without MACE relaxation** (out of
scope; it defeats the low-compute objective).

## Goals and constraints

**Goal.** Improve CREST-ensemble coverage on the inversion subset (canonical case
`cremp_sharp`) at low compute, while not regressing the agreement case
(`cremp_typical`, currently ~0.99). "Coverage" = `cov_bw_ceil` (Boltzmann-weighted
coverage of the CREST ceiling), `match_rmsd = 0.125 Å`.

**Constraints (locked).**
- MMFF/MC drive exploration; MACE is a single-point re-scorer ONLY. No
  MACE-relaxation anywhere in the inner loop or final relax.
- Reproduce CREST as closely as possible at lowest compute — every lever is
  judged on coverage-gained-per-GPU-second.
- Don't regress cremp_typical (≥ ~0.99) or any of the 497-passing test suite.
- Benchmark dyad: `cremp_sharp` (inversion / failure) vs `cremp_typical`
  (agreement / success). Definitions in `docs/concerted_moves_v0_3_plan.md`.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Foundational diagnostic: MMFF-iteration reachability of conf0 (does lighter/no MMFF preserve geometries near conf0 that full MMFF collapses away?) | ✓ complete (2026-06-25; light MMFF RULED OUT — ETKDG never gets near conf0 at any cap) |
| 2 | Lever 2 — partial/light MMFF relaxation (`mmff_max_iters`) | ✗ ruled out by Step 1 (no near-conf0 geometries for MMFF to preserve) |
| 3 | Lever 1 + 4 — de-novo reachability of conf0 vs seed budget and sampling distribution | ✓ complete (2026-06-25; uniform/less-biased sampling RULED OUT — worse than ETKDG prior; all plateau ~1 Å, none reach conf0) |
| 4 | MACE-optimization wall-clock cost benchmark (user request) | ✓ complete (2026-06-25; ~580× MMFF; inner-loop ~11 h/peptide → de-novo fix stays out; final-relax ~14 min feasible but can't reach unsampled conf0) |
| 4b | geomeTRIC (internal-coord TRIC) vs ASE LBFGS for MACE relaxation — does a better-conditioned optimizer cut the step count / wall-clock (Step 4 was step-count-bound)? | ✓ complete (2026-06-25; RULED OUT — same step count, ~26× slower wall-clock; ASE LBFGS stays) |
| 5 | Inversion pre-screen: cheap predictor of which peptides suffer the inversion | ✓ complete (2026-06-25; `scripts/inversion_prescreen.py`; inversion is pervasive (~63% relaxed dMMFF>2) & NOT feature-predictable; necessary-not-sufficient for coverage — calibration gap flagged) |
| 5b | Lever 5 — learned dihedral prediction (feasibility) | ✓ complete (2026-06-25; cis-ω ruled out as signature; inaccessible feature is SPARSE (median 2 backbone-dihedral flips, 67% ≤3) → viable but a real structured-predictor ML build, not a trivial add-on) |
| 6 | ~~Productionize reference-seeding (Lever 1) + document ceiling + figures~~ | **dropped (2026-06-25)** — reference-seeding needs a per-peptide CREST ensemble, so it cannot generalize to molecules distinct from CREMP (the actual goal). Superseded by Lever 5. |
| 7 | **Lever 5 — learned dihedral prediction → promoted to its own issue/branch.** The generalizable de-novo fix: predict the dominant conformer's sparse backbone dihedrals from sequence/topology, seed via constrained-DG. Needs only the trained model at inference, not a reference. | in progress (issue drafted) |

## Lever menu (cheap, no MACE relaxation)

Priority subset marked ★. Cost/impact are rough first estimates.

- ★ **Lever 2 — lighter / partial MMFF relaxation.** Cap MMFF iterations so a
  geometry that starts near conf0 (from ETKDG or a move) doesn't fully collapse
  into the lower-MMFF neighbour before MACE scores it. Cost: ~free (fewer MMFF
  steps = cheaper). Impact: directly targets the identified mechanism (MMFF pulls
  proposals off conf0). Risk: residual strain degrades MACE single-point accuracy
  — find the iteration count that balances reachability vs scoring noise.
- ★ **Lever 1 — seeding diversity.** Richer/larger ETKDG seed pools, basins
  accumulated across independent runs, or (where a reference exists) direct
  CREMP-seeding (confirmed to close cremp_sharp → 0.724). Cost: linear in pool
  size. Impact: high if a de-novo pool can land in conf0's basin; bounded by how
  narrow that basin is.
- ★ **Lever 4 — uniform / unbiased torsion sampling (the "fill in the
  distribution" idea).** ETKDG samples torsions from a non-uniform,
  CSD-derived prior (`useExpTorsionAnglePrefs`) plus basic-knowledge and
  macrocycle priors; the inverted basin (conf0) lives where that prior assigns
  low probability, so ETKDG rarely embeds there regardless of seed count.
  Replace the biased distribution with a more uniform one (turn off
  `useExpTorsionAnglePrefs`, optionally `useBasicKnowledge` /
  `useMacrocycleTorsions`, with `useRandomCoords`) so under-sampled torsion
  regions get covered. Cost: ~free (same embed, different flags); may need more
  seeds since uniform sampling is less efficient on the easy basins. Impact:
  potentially high — directly attacks the de-novo reachability gap Step 1
  identified. Risk: uniform DG yields looser/invalid geometries (mitigated by
  the downstream MMFF relax + MACE filter, or by keeping basic-knowledge on).
- ★ **Lever 5 — learned dihedral prediction (user idea, 2026-06-25).** The
  de-novo explorer can't reach the inverted basin, but we *have* the answer for
  the CREMP training peptides: their CREST minimum-energy conformers. Train a
  lightweight ML model to predict the inaccessible (MMFF-disfavoured) dihedral
  values — backbone ω/φ/ψ and key side-chain χ — from sequence/topology, then
  seed the MC walk from constrained-DG embeds at the predicted dihedrals (the
  Pool-B `torsional_sampling` machinery already does constrained-DG from target
  dihedrals). Unlike reference-seeding (Lever 1, needs a per-peptide CREST
  ensemble), this needs only the *trained model* at inference → a genuine de-novo
  fix for novel peptides in the inversion class. Cost: model training (one-off)
  + cheap inference; no MACE relaxation. Open questions: do the inverted-basin
  dihedrals generalise across sequences, what target representation (per-residue
  ω cis/trans + φ/ψ bins?), and does seeding from predicted dihedrals actually
  let the MMFF/MC+MACE pipeline land and retain the basin. Step 5b explores
  feasibility.
- **Lever 3 — accept + bound the ceiling.** Quantify how often the inversion
  occurs and document cremp_sharp as the canonical case. Cost: analysis only.
  Impact: sets expectations; not a fix.
- **(reserve) skip-MMFF-on-a-fraction.** `skip_mmff_relax` already exists for
  side-chain moves; extend selectively. Risk: strained geometries → unreliable
  MACE. Use only if light MMFF (Lever 2) underperforms.

## Risks to instrument from day one

- **MACE single-point unreliability on under-relaxed geometries.** Lighter MMFF
  leaves strain; MACE energies on strained geometries may be noisy/wrong.
  Diagnostic: track MACE-energy spread of a basin under varying MMFF iterations;
  flag if acceptance/ranking destabilises. Mitigation: a minimum iteration floor.
- **cremp_typical regression.** Any lever that helps cremp_sharp must be checked
  on cremp_typical in the same sweep (the v0.2/v0.3 sweeps sometimes dropped
  typical from cells — do not repeat that; run both peptides every cell).
- **Bursty exhaustive-ETKDG saturation.** Seed-pool quality varies run-to-run
  (seen in v0.3: 1 vs 6 basins at n=10k). Diagnostic: report pool e_min and
  nearest-to-conf0 per run; average over seeds, don't trust a single draw.

### Findings 2026-06-25 — Step 4b: geomeTRIC ruled out for MACE relaxation

Head-to-head, same 10 cremp_sharp conformers, identical float32 MACE-OFF
calculator, matched convergence (gmax 1e-3 Ha/Bohr = ASE fmax 0.05 eV/Å binding,
other criteria non-binding), maxiter 500:

| optimizer | mean steps | wall-clock | non-converged |
|---|---|---|---|
| ASE LBFGS | 145.5 | 3.79 s/conf | 0/10 |
| geomeTRIC TRIC | 146.0 | 100.6 s/conf | 2/10 |

- **Same step count (~146)** — internal coords + model Hessian + trust-radius RFO
  did NOT reduce steps. The float32 MACE PES is noise-limited near convergence
  (MACE warns to use float64 for geometry opt) and the macrocycle has many soft
  torsions, so step count is set by the noise floor, not optimizer conditioning.
- **~26× slower wall-clock at equal step count** — ~690 ms/step vs ASE ~26 ms/step,
  i.e. ~660 ms/step of intrinsic internal-coordinate overhead (Wilson B-matrix
  pseudoinverse, Hessian transforms, trust-step root-finding, per-step file I/O)
  on a 93-atom / ~270-internal-coord system, dwarfing the MACE force eval. The
  trust-step root-finding adds no force calls, so this is linear-algebra/I-O, not
  extra MACE evals.
- **494 meV energy divergence** between converged geomeTRIC and ASE — they land in
  different minima (different paths on the flexible/noisy PES). Unit conversions
  verified (Hartree/Bohr ↔ eV/Å), so genuine path divergence, not a bug.

**Verdict: ASE LBFGS remains the better relaxer; geomeTRIC makes MACE-relaxation
cost worse, not better.** The Step-4 wall-clock verdict (de-novo inner-loop
infeasible) stands. (Caveat: float64 MACE might let geomeTRIC converge cleaner /
fewer steps, but its ~660 ms/step overhead would still dominate at this molecule
size — not competitive.)

### Findings 2026-06-25 — Step 5: MMFF↔CREST inversion is pervasive, not feature-predictable

`scripts/inversion_prescreen.py` (CPU; 600-peptide stratified CREMP sample, 300
sharp + 300 control). For each peptide, MMFF-score every CREST conformer and
measure `dMMFF(dominant) = MMFF(CREST-dominant) − min MMFF`, both single-point
(on the GFN2-xTB geometry) and MMFF-relaxed.

| dMMFF(dominant) | median | 75th | 90th | max |
|---|---|---|---|---|
| single-point | 7.29 | 15.30 | 20.97 | 43.79 |
| MMFF-relaxed | 2.98 | 5.82 | 9.02 | 16.01 |

Anchors (separate cleanly): cremp_typical 0.18, cremp_sharp 7.10 kcal/mol (relaxed).

- **Single-point inflates relaxed by 2.4×** — cross-method strain (GFN2-xTB
  geometries aren't MMFF minima). The relaxed metric is the sampling-relevant
  one; both columns kept in the CSV to document the disparity.
- **The inversion is pervasive:** relaxed dMMFF >1 kcal/mol in 75% of peptides,
  >2 in 63%, >3 in 50%. MMFF and GFN2-xTB(CREST) disagree on the dominant
  conformer for most macrocyclic peptides — this is the norm, not a cremp_sharp
  one-off. (Plausible: classical FF vs semi-empirical QM diverge by several
  kcal/mol on flexible H-bonding macrocycles.)
- **No cheap predictor:** sharp vs control shows NO separation (non-sharp
  slightly higher); flagged-vs-not differ negligibly on poplowestpct (44 vs 49%),
  n_nme (0.9 vs 0.6), aromatic (61 vs 53%). A sequence-feature pre-screen for
  inversion risk is not feasible.

**Crucial caveat — calibration gap.** `dMMFF` measures energy-model
*disagreement*, which is *necessary but not sufficient* for a *coverage* failure:
the sampler can still reach a disfavoured dominant basin if it is geometrically
reachable (cremp_sharp's was disfavoured AND narrow). We have coverage labels for
only 2 peptides (typical 0.99, sharp 0.000), so the fraction of the ~63% flagged
that *actually* fail coverage is unknown. Two readings, both important:
(a) most flagged peptides still get good coverage → dMMFF over-flags, a
reachability term is needed; (b) coverage really is poor on a majority → the
package's CREST-reproduction is worse than the 2-point picture suggested. **A
coverage-labelled sample (run sampler + union_basin_count on ~30–50 peptides
spanning the dMMFF range) is the missing calibration** — recorded as a deferred
follow-up. This also raises the value of Lever 5 (learned dihedral prediction):
if the inversion is pervasive, a model that generalises would help broadly, not
just on rare cases.

### Findings 2026-06-25 — Step 5b: Lever 5 feasibility — sparse target, viable but a real ML build

Two probes on inverted peptides (relaxed dMMFF > 2 from the Step-5 screen):

1. **cis-ω is NOT the signature.** Inverted vs control CREST-dominant ω geometry:
   mean cis-ω/peptide 0.40 vs 0.20; only **25%** of inverted peptides have *any*
   cis-ω; cis fraction at NMe ω is actually *lower* in inverted (18% vs 27%);
   corr(dMMFF, #cis-ω) = −0.12. So cremp_sharp's cis-ω angle is not
   representative — ~75% of inverted peptides invert for other reasons. The
   inaccessible feature is not a simple sparse dihedral *type*.
2. **But the inaccessible feature is SPARSE.** Backbone dihedrals differing >60°
   between the CREST-dominant and the MMFF-best conformer (120 inverted peptides;
   median 12 backbone dihedrals/peptide): **median 2, mean 2.7, 75th 4, max 12;
   67% differ by ≤3, 32% by ≤1.** The CREST-dominant is usually a *local* change
   (~2 dihedral flips) from the easily-found MMFF-best, not a global refold.

**Verdict: Lever 5 is viable but a genuine (modest) ML project, not a lightweight
add-on.** Pro: the target is sparse (median 2 flips), and the MMFF-best is a
strong prior, so a model needs only a few dihedrals right; the constrained-DG
seeding machinery already exists (`torsional_sampling` Pool B). Con: it's not a
fixed rule (cis-ω ruled out), so the model must predict *variable* per-residue
backbone dihedral states (φ/ψ/ω) from sequence/topology — a structured predictor
needing CREMP training-data prep, a train/val generalization split, and seeding
integration. ~1/3 of cases are more diffuse (heavier tail) → it would help the
sparse majority, not all. Being a new ML model = a new major feature → its own
branch + issue per the project workflow before any build.

## Deferred follow-ups

- **Coverage-label calibration of the inversion pre-screen (2026-06-25).** Step 5
  shows energy disagreement (relaxed dMMFF) is pervasive but only 2 peptides have
  coverage labels, so the dMMFF→coverage-failure mapping is uncalibrated. Run the
  sampler + `union_basin_count` on ~30–50 CREMP peptides spanning the dMMFF range
  to (a) learn what dMMFF threshold predicts cov_bw_ceil≈0 and (b) measure the
  package's actual CREST-reproduction rate (currently a 2-point picture). **Trigger:**
  needed before claiming an inversion *prevalence* or shipping dMMFF as a
  reliability flag; GPU cost ~30–50 sampler runs.
- **NVIDIA Alchemi BGR for batched GPU inner-loop relaxation (2026-06-25, user
  pointer) — potentially reopens the de-novo fix.** Alchemi BGR (NIM microservice)
  provides a GPU-batched FIRE2 optimizer with dynamic batching — exactly the
  batched-NNP-relaxer the Step-4 cost benchmark identified as missing (serial
  ASE-LBFGS at 4 s/conf / ~11 h per 10k-seed run was the bottleneck). Crucially,
  **custom MACE `.model` checkpoints can be mounted** (`ALCHEMI_NIM_MODEL_TYPE=mace`
  + volume-mount + `NIM_DISABLE_MODEL_DOWNLOAD=true`), and MACE-OFF23 ships in
  exactly that `.model` format — so we could batch-relax with our *own* energy
  model, no substitution, no benchmark redefinition.
  **Open questions before adopting:**
  (1) Alchemi's MACE path is documented "periodic only (`pbc: true`)"; cyclic
  peptides are isolated. Workaround: a large vacuum box (MACE's ~5–6 Å cutoff
  makes a big enough box equivalent to isolated) — verify Alchemi accepts it and
  reproduces MACE-OFF23 isolated energies.
  (2) No published throughput numbers — measure batched relax rate vs the
  ~11 h/peptide serial-MACE figure.
  (3) Reachability is still unverified: even with fast batched MACE-OFF relax,
  does inner-loop relaxation reach conf0 / reproduce CREST? (MACE-relax from
  ~1.5 Å did not converge to conf0; conf0 is not even a MACE-OFF minimum — a
  deeper MACE-OFF min sits ~0.15 Å away at −66779.65. The "target" under
  MACE-OFF relaxation needs its own re-examination.)
  (4) Operational overhead — NIM container + REST API + NGC key.
  **Trigger:** worth a focused spike if reviving the de-novo fix is desired —
  stand up Alchemi with a mounted MACE-OFF23 in a vacuum box, validate energies
  vs our ASE MACE-OFF, benchmark throughput, then test de-novo reachability of
  conf0 with batched inner-loop relaxation.

- Constrained MMFF (freeze the rotated dihedral) as a middle ground between full
  and skipped relaxation — trigger: if `mmff_max_iters` capping (Lever 2) helps
  but leaves too much strain for reliable MACE scoring.
- Graceful-degradation for Move A (concerted) on non-aromatic peptides (carried
  over from v0.3; same class as the ω/large-window fixes) — trigger: if any
  production mix re-enables `concerted_dihedral_weight` across a mixed dataset.

## Findings

(append-only, dated)

### Findings 2026-06-25 — Step 1: light MMFF ruled out; the gap is de-novo ETKDG reachability

Embedded a 300-conformer ETKDG pool (macrocycle params) of cremp_sharp and
measured nearest-to-conf0 heavy-atom RMSD as MMFF relaxation is capped at
{0 (raw), 5, 20, 50, 200, 2000} iterations:

| MMFF cap | nearest RMSD | n ≤ 0.125 Å | n ≤ 0.36 Å |
|---|---|---|---|
| raw (0) | 0.987 Å | 0 | 0 |
| 5 | 0.980 | 0 | 0 |
| 20 | 0.979 | 0 | 0 |
| 50 | 0.972 | 0 | 0 |
| 200 | 0.977 | 0 | 0 |
| 2000 (full) | 1.187 | 0 | 0 |

**Light MMFF (Lever 2) is ruled out.** The pool never gets within ~1 Å of conf0
at *any* MMFF cap — so MMFF is not collapsing near-conf0 geometries (there are
none to preserve). ETKDG simply does not generate conf0's fold de novo at this
pool size. Full MMFF (2000) nudges the pool slightly *farther* (0.99 → 1.19 Å),
consistent with MMFF pulling toward its own basins, but it's moot since nothing
is in range. **The mechanism is upstream of the relaxer: de-novo ETKDG
reachability, not MMFF collapse.**

Seed-budget scaling so far: 300 raw seeds → 0.99 Å nearest; the v0.3 exhaustive
run at 10k seeds → 0.36 Å nearest basin; never < 0.125 Å. → Step 3 quantifies
whether more seeds ever cross 0.125 Å or asymptote short (the de-novo ceiling
for Lever 1).

### Findings 2026-06-25 — Step 3: uniform sampling ruled out; de-novo levers exhausted

Compared three embedding distributions for raw de-novo reachability of conf0 at
matched seed budgets (heavy-atom Kabsch RMSD, nvmolkit embed, no MMFF):

| distribution | nearest @1k | @10k | @30k | n≤0.36 Å |
|---|---|---|---|---|
| etkdgv3_macro (CSD-biased, default) | 1.335 | 1.015 | 1.015 | 0 |
| no_exp_tors (CSD torsion prior off) | 1.587 | 1.399 | 1.399 | 0 |
| plain_dg (all priors off + random, "uniform") | 1.719 | 1.521 | 1.410 | 0 |

**Lever 4 (uniform / "fill-in-the-distribution") is RULED OUT.** Removing
ETKDG's torsion prior makes conf0 reachability *worse* (1.0 → 1.4 Å), not better:
the CSD/macrocycle prior concentrates sampling in plausible regions that get
*closest* to conf0; a uniform distribution dilutes that coverage rather than
filling conf0's region. (Consistent with the v0.3 exhaustive-ETKDG mode
comparison: the macrocycle prior helps, not hurts.) All distributions plateau by
~10k seeds at ~1.0–1.4 Å raw; none place a conformer within even 0.36 Å of conf0,
let alone the 0.125 Å match threshold.

(Reconciliation: this is *raw* reachability (~1.0 Å); the v0.3 "0.36 Å" was a
*post-MMFF basin* reaching conf0's +453 meV *neighbour*. Both agree conf0 itself
is de-novo unreachable.)

**The de-novo levers are exhausted:** light MMFF (Step 1 ✗), uniform sampling
(Step 3 ✗), more seeds (plateaus past ~10k). conf0 is *both* MMFF-disfavoured
(+2.4 kcal/mol) *and* ETKDG-disfavoured (~1 Å unreachable). Within the cheap
"MMFF/MC explore + MACE-score-only" design, the dominant CREST basin is genuinely
unreachable de novo — the only thing that closes it is reference-seeding
(confirmed 0.724 in v0.3). The sole remaining de-novo fix would be MACE-*guided*
sampling, which violates the compute constraint (out of scope).

**Pivot:** the practical deliverable is no longer a de-novo fix but (1) a cheap
**inversion pre-screen** (Step 4) so users know which peptides are de-novo
unreliable and should supply a reference, (2) productionised **reference-seeding**
(Lever 1, Step 5) for those cases, and (3) documenting the structural ceiling
(Lever 3).

### Findings 2026-06-25 — Step 4: MACE-optimization wall-clock cost (de-novo fix stays out)

Measured on cremp_sharp (93 atoms), GV100:

| operation | per-conformer | vs MMFF |
|---|---|---|
| MMFF relax (RDKit, all cores) | 7.1 ms | 1× |
| MACE batched single-point (batch=200) | 39.8 ms | — |
| MACE relax (ASE LBFGS, mean 155 steps) | 4.08 s | ~580× |

Extrapolated per peptide: final-relax (~200 basins) MMFF 1.4 s vs MACE ~14 min;
**inner-loop (~10k seeds) MMFF ~1 min vs MACE ~11 hours.**

**Verdict.** MACE relaxation does NOT revive the de-novo fix. The inner-loop
version — the only one that could reach conf0 de novo — is ~11 h/peptide
(~580× MMFF), infeasible for a low-compute tool. A final-relax of survivors
(~14 min) is feasible as an optional polish but cannot reach conf0 (you can only
relax conformers already sampled; conf0 is never sampled). The earlier
"avoid MACE relaxation" decision is confirmed with hard wall-clock numbers.

Caveat: the "batched MACE relax" estimate is a rough ceiling — at batch=200 the
GV100 was not saturated (batched 39.8 ms/conf was slower than serial 26 ms/conf
single-conf force), so a larger batch could lower it, but even an optimistic 4×
leaves inner-loop at ~3 h/peptide → still infeasible. A precise batched ceiling
(SP throughput vs batch size) is a deferred refinement; it does not change the
inner-loop verdict.
