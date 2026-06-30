# Learned dihedral prediction — disfavoured-basin seeding (issue #20, Lever 5)

Branch: `20-disfavored-dihedral-prediction`. Follows from issue #19 (the MMFF↔CREST
inversion is pervasive but sparse — the CREST-dominant differs from the MMFF-best by
a median of ~2 backbone dihedrals). Goal: predict the dominant conformer's backbone
dihedrals from topology and seed constrained-DG so the sampler reaches the otherwise
MMFF-inaccessible dominant basin — a fix that generalises (needs only the trained
model at inference, not a per-peptide CREST reference).

## 20260630

## Goals and constraints

- **Goal.** A lightweight model that, from per-residue topology, predicts the dominant
  backbone (phi/psi/omega) well enough that constrained-DG seeds reach the dominant
  basin where de-novo sampling fails — improving `cov_bw_ceil` on the inverted subset.
- **Generalises:** inference needs only the model + the molecule's topology.
- Don't regress the existing 58-test suite; reuse `torsional_sampling` constrained DG
  and `union_basin_count` coverage metric.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Module foundation: `src/dihedral_predictor/` (residues/data/model/train/seed) + README + 12 tests | ✓ complete (2026-06-30) |
| 2 | Ring-ordering bug fix (get_backbone_dihedrals not ring-ordered → corrupted omega + neighbour features for non-ring-ordered peptides) | ✓ complete (2026-06-30; cis fraction 0.66 artifact → 0.17 real; 58 tests pass) |
| 3 | Dataset build (36,198 peptides) + train | ✓ complete (2026-06-30; val peptide_all_ok 0.36 vs 0.00 majority baseline) |
| 4 | Seeding-speed diagnosis | ✓ complete (2026-06-30; tol30 embed ~90 s → tol60 ~2-5 s; tol60 set as default) |
| 5 | Seeding validation v1 (RMSD proxy: seed → relax → min RMSD to dominant, vs unconstrained ETKDG) | ✓ complete (2026-06-30; mixed — signal but no clear basin-landing win) |
| 6 | Tighter-tolerance precision probe | ✓ complete (2026-06-30; tol40 ≈ tol60, baseline still wins <0.5 Å → tolerance is NOT the limiter, prediction accuracy is; kept tol60) |
| 7 | **Real MCMM + coverage integration** — feed predicted-dihedral seeds into the actual confsweeper sampler (`extra_seed_coords` param, wired + smoke-tested), measure `cov_bw_ceil` via `union_basin_count`. The true test (MC+MACE refine the seed; the RMSD proxy omits this). | in progress |
| 8 | **Model improvements** — larger model / wider neighbour window / top-K (multi-modal) targets / topology-split generalisation eval, then re-validate. | pending |

## Architecture (Steps 1-4)

- Per-residue features derived from the mol in **ring order** (chained via each
  residue's `n_next`), so inputs and (phi/psi/omega) targets are aligned by
  construction. Features: NMe/Gly/D/Pro flags + side-chain heavy count / aromatic /
  H-bond donors+acceptors, plus cyclic ±1 neighbour augmentation.
- Targets: phi/psi into 24 circular bins (15°), omega cis/trans. Dominant =
  max-Boltzmann-weight CREST conformer.
- Model: per-residue transformer (ring size injected as a global feature; no absolute
  positional encoding — a macrocycle has no canonical start). 3 heads (phi/psi/omega).
- Seeding: predicted bins → bin-centre angles → constrained-DG bounds (phi/psi **and**
  omega) → `embed_constrained`. Tolerance ±60° (tighter thrashes the DG embed and is
  false precision — rotamers are 120° apart, prediction is ±15-22° accurate, MMFF
  relax snaps to the basin).

### Findings 2026-06-30 — Step 3: predictor trains, beats majority baseline

Val (36,198 CREMP peptides, peptide split), model vs majority baseline: phi within-1-bin
0.72 vs 0.37, psi 0.69 vs 0.28, omega 0.97 vs 0.94, **peptide_all_ok 0.36 vs 0.00**
(fraction of peptides with *every* backbone dihedral within tolerance — the end-to-end
seeding proxy). The omega baseline flipping from 0.15 (pre-fix) to 0.94 (post-fix)
re-confirmed the ring-ordering bug fix.

### Findings 2026-06-30 — Step 5: seeding validation is mixed (signal, no clear win)

`scripts/validate_seeding.py` — for inverted test peptides (relaxed dMMFF > 2), min RMSD
to the CREST-dominant from predicted-dihedral seeding vs unconstrained macrocycle ETKDG,
both unrelaxed and MMFF-relaxed.

- **Heavy-atom RMSD (50 peptides):** seeded median 2.10-2.15 Å vs baseline 2.46-2.52;
  within 2 Å 44% vs 26%; within 1 Å **0% both**. Seeded closer on 70% of peptides.
- **Backbone RMSD (N/Cα/C, 50 peptides):** seeded median **0.85 = baseline 0.85**;
  within 1.0 Å seeded **86% vs 68%** (more consistent); within 0.5 Å seeded **0% vs
  baseline 4-14%** (tol60 too loose to land tightly — ETKDG occasionally nails it).
- **MMFF-relaxation confound disproven:** relaxed ≈ unrelaxed (2.10 vs 2.15 heavy;
  0.91 vs 0.85 backbone), so MMFF relax does NOT pull seeds off the dominant — the
  dominant is metastable under MMFF (consistent with conf0 being a narrow stable MMFF
  basin in issue #19).

**Read:** the predictor biases the backbone toward the dominant fold (reaches ~1 Å more
consistently than ETKDG) but at tol60 cannot land the basin tightly, so it does not yet
beat unconstrained ETKDG on the strict basin criterion. Two caveats: tol60 trades away
precision (Step 6), and the RMSD proxy seeds → relaxes → measures *without the MC walk*
that the real pipeline would run, so it likely undersells the method (Step 7).

### Findings 2026-06-30 — Step 6: tolerance is not the precision limiter

tol40 vs the tol60 default (12 inverted test peptides, backbone RMSD): within 0.5 Å
seeded **0% vs baseline 17% (unrelaxed) / 33% (relaxed)**; median 0.89 vs 0.82; seeded
strictly closer only 42% (worse than tol60's 56%). Tightening did NOT improve tight
landing — seeded conformers cluster at ~0.85 Å backbone regardless of tolerance,
floored by the prediction's residual error, while unconstrained ETKDG's higher variance
occasionally lands <0.5 Å by luck. **The limiter is prediction accuracy (Step 8) and/or
the seed needing MC refinement (Step 7), not the DG tolerance.** Kept tol60 (same quality,
~10× faster embed). Note: baseline reaching <0.5 Å *backbone* for some inverted peptides
is not a contradiction with issue #19 — the coverage basin match is heavy-atom symmetric
at 0.125 Å (much stricter than backbone-0.5 Å), so the backbone fold can be reachable
while the full tight basin is not.

### Findings 2026-06-30 — Step 7: backbone-only seeding does not lift coverage (side chains matter)

`scripts/validate_seeding_coverage.py` on cremp_sharp (in TRAIN — mechanism test, not
generalisation), n_seeds=6400, learned seeds injected via `extra_seed_coords`:

| run | final basins | cov_bw_ceil | max_missed_bw |
|---|---|---|---|
| baseline (pure-DBT) | 2 | 0.000 | 0.724 |
| seeded (pure-DBT) | 4 | **0.000** | 0.724 |

Seeding added exploration (87 vs 42 basins in memory) but did NOT cover the dominant.
**Why:** coverage matching is heavy-atom symmetric RMSD at 0.5 Å, but the learned seed
only sets the **backbone** (~0.85 Å backbone ≈ 2 Å heavy-atom — side chains from DG are
random). The cremp_sharp dominant is defined by backbone **+ specific side-chain
rotamers** (issue-17's concerted-χ territory). **Confound:** this run used the default
pure-DBT (backbone-only) proposer, so the MC walk had no side-chain moves to refine the
seed's side chains — an unfair test. Re-running with side-chain χ moves enabled (the
issue-17 move set) to see whether good-backbone seed + side-chain MC reaches the dominant.

**Implication for Step 8:** backbone-only prediction is insufficient — the model (or the
seeding) must also place **side-chain χ**, at least for the rotamer-sensitive residues.

### Findings 2026-06-30 — Step 7 ORACLE CONTROL FAILED: coverage measurement is invalid

Before trusting any seeded-vs-baseline number, ran the control: inject the TRUE CREMP
dominant conformer as the seed (`--oracle`). Expected ≈ the issue-19 direct-seeding result
(cov 0.724). Got cov_bw_ceil = 0.010. Isolating further (no MCMM, direct
`_boltzmann_coverage`):

- **UNRELAXED true dominant → cov 0.000**, MMFF-relaxed → 0.010, and MMFF relax shifts the
  dominant **0.59 Å** heavy-atom (> the 0.5 Å match threshold).

**The CREST minimum-energy geometry is the oracle / ground truth and MUST register as
covering the ceiling** (it is, by construction, one of the ceiling basins). It currently
does not — `calc_coverage` per-basin RMSDs put the oracle 1.64 Å from the 0.724 ceiling
basin and 0.85-1.79 Å from EVERY basin (never <0.5). The mapped oracle geometry is valid
(MMFF relaxes it by only 0.59 Å), so this is a **coverage-comparison bug** (cross-provenance
atom-order / graph-perception between the smi-built mol and the ceiling SDF mol), not a real
miss. Until the CREST-min oracle registers ≈ its weight (0.724), NO Step-7 number
(baseline/seeded) can be trusted. **BLOCKER → fix:** make the oracle (CREST min-energy)
register correctly — likely by computing coverage with provenance-consistent mols
(build the sampler mol from the CREMP/ceiling template rather than the smi string), and
confirm against how the existing benchmark + issue-19 obtained the 0.724.

**Isolation (decisive):** feeding the ceiling basins back as the sampler gives
self-coverage = **1.0** — so `calc_coverage` / `_boltzmann_coverage` are correct. The bug
is purely **cross-provenance**: a CREST/smi-built geometry matched against the ceiling SDF
mol via spyrmsd-symmetric fails for cremp_sharp (oracle 1.64 Å from its own basin) despite
ceiling-vs-ceiling being exact. Fix direction: express the ceiling reference in the *same
mol object* as the sampler (map the CREST ceiling conformers onto the smi-built mol so all
RMSDs are single-provenance), then oracle → ≈0.724 and seeded/baseline become trustworthy.

**Provenance RULED OUT (2026-06-30):** redid the oracle coverage with provenance-consistent
heavy-atom Kabsch (both ceiling and oracle mapped to the smi-built mol, one atom order) — it
agrees exactly with spyrmsd: CREST min-energy oracle is **1.64 Å** from the 0.724-weight
ceiling basin and ≥0.85 Å from every basin. So it is not a matching artifact. The real
finding: `cov_bw_ceil` weights ceiling basins by **MACE** energy, so the 0.724 basin is the
*MACE-favored* geometry, and the CREST min-energy conformer sits 1.64 Å away (nearest the
0.158-weight basin). The inversion is GEOMETRIC in the ceiling: MACE's preferred geometry ≠
CREST's. This collides with the issue-19 note that conf0 is both the CREST-dominant and the
MACE global min (same geometry). **NEEDS user ground truth before proceeding:** (1) how are
the ceiling SDF basins built (raw CREST? MMFF/MACE-relaxed? centroids)? (2) which weight
defines the basin to cover — CREST-Boltzmann or MACE-Boltzmann? (3) what geometry was the
oracle in the issue-19 0.724 direct-seeding result, and against which weighting? Step 8
(model improvements) is independent of this and can proceed — the RMSD-proxy validation
needs no ceiling.

### Findings 2026-06-30 — Step 7 coverage FIXED (CREST weighting + raw-CREST ceiling)

User ground truth: (1) ceiling = raw CREST geometries (rebuild from CREMP); (2) the target
is **CREST Boltzmann** population, NOT MACE; (3) issue-19's 0.724 seeded from the MMFF-relaxed
geometry. Project objective restated: reproduce the CREST conformer *distribution* cheaply.

Rewrote `scripts/validate_seeding_coverage.py`: ceiling = CREST conformers (mapped to the
smi-built mol's atom order via substructure match) clustered at 0.125 Å heavy-atom Kabsch,
each basin weighted by summed CREST Boltzmann population; coverage = CREST-weighted fraction
of ceiling basins within `match_rmsd` (heavy-atom Kabsch, single-provenance) of a sampler
basin. **Oracle now registers correctly: the CREST dominant self-covers at 0.535** (= the
cremp_sharp dominant basin's CREST weight, matching poplowestpct 53.48%). The earlier
MACE-weighted 0.724 and the cov-0 results are superseded. Re-running baseline / seeded /
oracle through the real MCMM with this corrected metric is the live Step 7.

### Findings 2026-06-30 — Step 7 (corrected): pipeline VALIDATED; learned backbone seed still insufficient

Real MCMM (n_seeds=6400), CREST-weighted coverage, best model (w2_d256_l6):

| run | cov @0.5 | cov @0.75 | cov @1.0 |
|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 |
| seeded (learned backbone) | 0.000 | 0.000 | 0.000 |
| oracle (true CREST dominant) | 0.000 | **0.536** | 0.580 |

- **Pipeline validated:** the oracle recovers cov 0.536 = the dominant basin's CREST weight
  (0.535). A good seed through MMFF relax + MC + MACE DOES reproduce the dominant — seeding
  mechanism and metric are correct.
- **match_rmsd must be ~0.75, not 0.5:** MMFF relaxation shifts the dominant 0.59 Å from its
  CREST geometry, so 0.5 is too tight (oracle reads 0). 0.75 accommodates the MMFF↔CREST gap.
- **Learned backbone seeding still insufficient (0.000):** the backbone-only seed (~2 Å
  heavy-atom off, side chains random) does not land the basin even at match 1.0. Gap =
  prediction quality + side-chain χ placement → Step 8.

### Findings 2026-06-30 — Step 8 capacity/window sweep: capacity is the lever

Val peptide_all_ok (random split; baseline w1_d128_l3 = 0.366):

| config | peptide_all_ok | phi_w1 | psi_w1 |
|---|---|---|---|
| w2_d128_l3 | 0.366 | 0.72 | 0.69 |
| w1_d256_l6 | 0.429 | 0.73 | 0.71 |
| **w2_d256_l6** | **0.437** | 0.74 | 0.71 |

Capacity (d256/l6) lifts peptide_all_ok 0.37 → 0.43; window=2 adds a little. Next Step-8
levers: side-chain χ prediction (Step 7 shows it's required for the heavy-atom basin match),
topology-split generalisation check, and a longer/larger model; then re-validate coverage.

### Findings 2026-06-30 — MMFF relaxation destroys the inverted-dominant seed

For an inverted peptide the dominant basin is MMFF-disfavoured, so MMFF relaxation pushes a
good seed off it. Seed coverage (cremp_sharp, dominant CREST weight 0.535):

| seed | unrelaxed @0.5 | unrelaxed @0.75 | MMFF-relaxed @0.5 | MMFF-relaxed @0.75 |
|---|---|---|---|---|
| oracle (true dominant) | **0.535** | 0.576 | **0.000** | 0.536 |
| learned (backbone) | 0.000 | 0.000 | 0.000 | 0.000 |

The MCMM MMFF-relaxes seeds because it explores the MMFF PES (cheap) with MACE as the final
re-scorer. But that step is hostile to the inverted dominant: **omitting MMFF relax recovers
the oracle's coverage at the strict 0.5 threshold.** The learned seed is 0 either way — its
problem is upstream (backbone-only, ~2 Å heavy-atom off). → motivates a **no-MMFF-relax seed
path** (keep predicted geometry, MACE-score, add to the basin set).

### Findings 2026-06-30 — MACE vs MMFF cost (cremp_sharp, 93 atoms, GV100)

| op | MMFF | MACE | ratio |
|---|---|---|---|
| single-point | 1.3 ms/conf | 26 ms/conf (batched 64) | ~20× |
| full relaxation | 4.2 ms/conf | ~3.8 s/conf (LBFGS, 145 steps) | ~600–900× |

MCMM budget (≈12,800 in-loop relaxations): MMFF ~1 min/peptide; **MACE relaxation in-loop
≈ 13 h/peptide (infeasible)**; MACE energy-only acceptance ≈ +5.5 min/peptide (feasible but
doesn't fix the geometry problem). Conclusion: don't convert the explorer to MACE — the cost
is only prohibitive over the full MCMM. Apply the no-relax (or cheap MACE-relax) treatment to
the **handful of seeds** only.

### Findings 2026-06-30 — no-relax seed path implemented + validated

Added `get_mol_PE_mcmm(relax_seeds=...)`: when False, injected `extra_seed_coords` are NOT
MMFF-relaxed; they are MACE-scored at their predicted geometry and appended to the output
basins as protected basins after the tail. Real-pipeline coverage (cremp_sharp, dominant
CREST weight 0.535):

| run | cov @0.5 | cov @0.75 |
|---|---|---|
| baseline | 0.000 | 0.000 |
| seeded (learned, no-relax) | 0.000 | 0.000 |
| oracle (no-relax) | **0.535** | 0.576 |

The no-relax path **fixes the oracle at the strict 0.5 threshold** (was 0.000 with MMFF
relax) — a good seed now survives end-to-end and reproduces the dominant. The learned seed
is still 0.000: its seeds are added as protected basins but stay ~2 Å heavy-atom off (wrong
side chains). **The architecture is now sound; the sole remaining gap is prediction quality —
specifically side-chain χ placement (next Step-8 lever).**

### Findings 2026-06-30 — Step 8: separate chi model built; binned prediction can't hit the 0.5 Å match

Built a SEPARATE side-chain chi predictor (`ChiPredictor`, own checkpoint — backbone model
untouched, so no fidelity risk): chi extraction (`residues.sidechain_chi_quads`, canonical-rank
tie-breaking so chi slots are consistent between the CREMP extraction mol and the smi seed
mol), chi dataset targets, `train_chi`, and seeding via `SetDihedralDeg`
(`seed_conformers(chi_model=...)`). 17 tests pass.

Chi model (d256/l6/window2, val): **chi_within1 = 0.57, chi_peptide_ok = 0.12** — harder than
backbone (rotamers depend on packing context the per-residue features capture less of).

Coverage with backbone+chi seeding (cremp_sharp, no-relax path):

| run | cov @0.5 | @0.75 | @1.0 |
|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 |
| seeded (backbone+chi) | **0.000** | 0.000 | 0.000 |
| oracle (true dominant) | 0.535 | 0.576 | 0.581 |

**Key finding: binned-dihedral prediction fundamentally cannot reach the 0.5 Å heavy-atom
basin match.** Even all-dihedrals-within-1-bin (±22°) compounds across ~12 backbone + ~8 chi
dihedrals to ~1-2 Å heavy-atom. The oracle covers only because it is the *exact* geometry.

**The missing piece — MACE-relax the seed (not MMFF).** MACE's global min for cremp_sharp IS
the dominant, so MACE-relaxing an approximately-correct predicted seed should pull it onto the
dominant, correcting the binning error. Affordable for a handful of seeds (~3.8 s each) even
though MACE-relax over the whole MCMM is infeasible (~600× MMFF). This is the next test.

### Findings 2026-06-30 — MACE-relax-seed also fails on cremp_sharp; accuracy is the wall

MACE-relaxing the backbone+chi seed before injection (cheap: ~20 seeds × ~3.8 s) still gives
cov 0.000 at all thresholds (oracle 0.535). cov@1.0 = 0 means the seed is >1 Å heavy-atom
from the dominant — it sits in a *neighbouring* MACE basin, so MACE-relax descends to that
neighbour, not conf0. Complete chain: oracle 0.535 ✓ | backbone-only 0 | backbone+chi 0 |
backbone+chi+MACE-relax 0.

**Conclusion:** the Lever-5 infrastructure and mechanism are fully validated (a correct seed
reproduces the dominant end-to-end), but the learned prediction is not accurate enough to
land in cremp_sharp's dominant attraction basin. cremp_sharp is the hardest case (the
canonical deep inversion, dominant CREST weight 0.535). Open directions: (a) measure
*aggregate* coverage lift over many inverted test peptides — seeding may help the broader,
milder-inversion population even if it misses the extreme case; (b) substantially improve
prediction (regression for finer angles, chi conditioned on predicted backbone, more
capacity/data); (c) document the validated infrastructure + the accuracy wall.

### Findings 2026-06-30 — REFRAMING: cremp_sharp's inversion is a side-chain problem, not backbone

Coverage uses heavy-atom (all non-H) RMSD. Re-measuring on BACKBONE atoms only (N/Cα/C,
`--backbone`) is revealing:

| atoms=backbone | cov @0.5 | @0.75 | @1.0 |
|---|---|---|---|
| baseline (de-novo) | 0.002 | **1.000** | 1.000 |
| oracle | 1.000 | 1.000 | 1.000 |

The cremp_sharp CREST ensemble is **0.995 one backbone fold** (10 backbone basins, one
dominant), and **de-novo MMFF/MC already covers that backbone fold (cov 1.0 @0.75 Å).** So
the backbone was never the bottleneck — the all-atom failure (0.000) is *entirely*
side-chain rotamers. This aligns with issue-17 (cremp_sharp needs concerted Trp-χ moves).

**Reframing:** for cremp_sharp, learned *backbone* seeding cannot help (backbone already
reachable). The lever is precise *side-chain χ* placement, and the current χ accuracy
(peptide_ok 0.12) plus the strict 0.5 Å all-atom match is the wall. (Note Step 5b's
"~2 backbone-dihedral flips" measured CREST-dominant vs MMFF-*best*; MMFF/MC *exploration*
still reaches the backbone even if the single MMFF-best differs.) Open question this raises:
how much of the *broader* inverted population is backbone-driven vs side-chain-driven —
backbone seeding may still help peptides whose inversion is genuinely a backbone refold,
even though cremp_sharp's is not.

## Deferred follow-ups

- **Top-K / multi-modal targets** — predict more than the single dominant conformer to
  seed the whole CREST ensemble, not just the ceiling basin (part of Step 8).
- **Topology-stratified split** — stronger generalisation test than the random peptide
  split (part of Step 8).
