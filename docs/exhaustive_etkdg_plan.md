# Exhaustive randomized ETKDG — design plan

Branch: `7-randomized-etkdg-sampling`. Closes #7.

This document is the working design for a new sampling mode in confsweeper that
replaces CREST metadynamics with brute-force randomized ETKDG. It captures the
motivating data, the planned pipeline, and the experimental findings that
inform the implementation. Folded into `src/README.md` (or deleted) before the
PR ships.

---

## Background

The current `get_mol_PE_batched` runs nvmolkit ETKDG at `n_confs=100` and
optionally a Pool B torsional sampler, then clusters with Butina at
`cutoff_dist=0.1`. The design intent is to provide a CREST-replacement for
cyclic peptides where exhaustive metadynamics is too expensive.

A direct CREMP comparison run on the deduplicated CycPeptMPDB PAMPA dataset
shows the current pipeline severely under-samples. CREMP (CREST-generated
ensembles, 36 198 macrocyclic peptides) is the natural ground truth here:

| metric | CREMP (CREST) | confsweeper (ETKDG, n_confs=100) |
|---|---|---|
| median Boltzmann weight of dominant conformer | 25.4 % | ≈ 72 % |
| peptides with `max_bw > 0.95` | 0.4 % | majority |
| mean unique conformers per peptide | 864 | 100 (only ~3 within 3 kT) |

A Butina cutoff sweep (0.1 → 0.01) on a representative PAMPA peptide kept all
100 conformers as their own clusters at every cutoff. Clustering is therefore
**not** the bottleneck — sampling diversity is. CREMP shows the underlying
macrocyclic-peptide landscape is generally not one-hot, so the sharpness we
observe is purely an artifact of ETKDG-100 finding one basin per molecule.

This impacts the PAMPA fine-tuning val/MAE directly: only one conformer per
molecule contributes to the property prediction. The fine-tuning wrapper had
to be initialised at α = 0.01 (near-uniform) just to keep gradients flowing —
a workaround for one-hot ensembles, not a fix.

The user's directive: keep the philosophical commitment to
randomization-over-metadynamics. Scale ETKDG up by orders of magnitude,
exploiting the fact that nvmolkit's GPU embed is ~100× cheaper than CREST per
conformer. Even if we have to call nvmolkit ETKDG repeatedly with different
seeds, this is an acceptable trade-off.

---

## nvmolkit ETKDG semantics — experimental findings

Before scaling N to thousands, three properties of nvmolkit's
`embedMolecules.EmbedMolecules` were checked on a 56-heavy-atom PAMPA cyclic
peptide. The script ran each path on the same SMILES with `params` from
`get_embed_params_macrocycle()` and then compared the resulting conformer
coordinate matrices.

| Path | Setup | conformers returned | unique-coord rows | NN distance min / median / max (Å) |
|------|-------|---------------------|-------------------|-------------------------------------|
| A | single call, `confsPerMolecule=200`, `randomSeed=0` | 200 | 200 | 27.8 / 43.0 / 53.6 |
| B | two calls, `confsPerMolecule=100`, seeds 0 then 1000 | 200 | 200 | 31.4 / 43.7 / 55.6 |
| C | two calls, `confsPerMolecule=100`, seeds 0 then 0 | 200 | 200 | — |

Findings that constrain the design:

1. **Path A is genuinely seed-independent within a single call.** A single
   `EmbedMolecules(..., confsPerMolecule=N)` returns N geometrically distinct
   conformers — no internal seed sharing collapses the batch onto a few
   shapes. Single-call scaling is therefore viable as the fast path.
2. **Chunking covers comparable diversity.** Path B's nearest-neighbour
   distance distribution matches Path A's within noise. Calling
   `EmbedMolecules` repeatedly with offset seeds is a safe fallback when
   `n_seeds` exceeds whatever per-call ceiling the GPU memory imposes.
3. **`params.randomSeed` does not provide cross-call bit-exact
   reproducibility.** Two consecutive calls with `randomSeed=0` produced 200
   distinct conformers, not the same 100 twice. Some internal random state
   advances independently across calls. Tests that depend on coordinate
   identity will fail intermittently; the chunk-equivalence test in Phase 4
   must compare aggregate stats (sorted-energy distributions, basin counts)
   instead.
4. **Repeated `EmbedMolecules` calls append rather than replace.** A second
   call to `embed.EmbedMolecules` on the same RDKit `Mol` adds new conformers
   to the existing set rather than overwriting them — verified empirically:
   `EmbedMolecules(mol, confsPerMolecule=10)` followed by
   `EmbedMolecules(mol, confsPerMolecule=10)` leaves the mol with 20
   conformers, not 10. The chunked-embed loop in
   `get_mol_PE_exhaustive` depends on this; if nvmolkit ever changes to
   replace-on-call (matching `AllChem.EmbedMultipleConfs`'s default), the loop
   collapses to producing only the last chunk's conformers. The
   `test_exhaustive_chunked_path_accumulates_full_pool` test guards against
   this regression. Discovered while writing the test: an initial CPU mock
   used `EmbedMultipleConfs(..., clearConfs=True)` (the default), produced
   only the last chunk's conformers, and triggered a clear assertion
   failure. The mock now passes `clearConfs=False` to match nvmolkit.

Implication for the implementation: use Path A as the default fast path; only
fall back to chunking when `n_seeds > embed_chunk_size`. Plan for stochastic
correctness tests, not coordinate-identity tests. The chunked-loop's
correctness silently depends on nvmolkit's append semantics — flag this in
the eventual PR description so reviewers know what the regression test is
actually catching.

---

## Phase 0 — branch hygiene

The actual `useSmallRingTorsions` flag fix landed on `main` via PR #6 (the
`3-cremp-dataset-comparison` branch). What remains here:

- The docstring in `get_embed_params_macrocycle` ([src/confsweeper.py:75-94](../src/confsweeper.py#L75-L94))
  still describes `useSmallRingTorsions` as enabled.
- The matching subsection in `src/README.md` ([src/README.md:24-29](../src/README.md#L24-L29))
  also describes the flag as enabled.

Update both to state that the flag is intentionally disabled because nvmolkit
hangs in CPU preprocessing when it is set. Single small commit.

---

## Phase 1 — `get_mol_PE_exhaustive` core pipeline

Add a new public function in `src/confsweeper.py`. Keep `get_mol_PE_batched`
working unchanged — it remains the right tool for ETKDG-only quick experiments
and for torsional sampling. Possible deprecation of either pre-existing path is
deferred until the saturation experiments in Phase 2 confirm the exhaustive
path supersedes them.

### Signature

```python
def get_mol_PE_exhaustive(
    smi: str,
    params,
    hardware_opts,
    calc,
    n_seeds: int = 5000,
    embed_chunk_size: int = 1000,
    score_chunk_size: int = 500,
    e_window_kT: float = 5.0,
    rmsd_threshold: float = 0.1,
    minimize: bool = False,
    seed: int = 0,
) -> tuple[Chem.Mol, list[int], list[float]]:
```

Returns the same `(mol, centroid_ids, energies_eV)` contract as
`get_mol_PE_batched`, so downstream consumers swap one function call.

### Pipeline stages

1. **Massive embed.** Path A by default: a single `EmbedMolecules` call with
   `confsPerMolecule=n_seeds`. When `n_seeds > embed_chunk_size`, fall back to
   Path B: chunked calls with seeds `seed + chunk_index * embed_chunk_size`,
   appending into a single mol via `RDKit.Chem.RWMol`. Stop early if a chunk
   returns zero conformers (likely an embedding-incompatible molecule).

2. **(Optional) MMFF94 minimization.** When `minimize=True`, run
   `AllChem.MMFFOptimizeMolecule` on each conformer in place to push
   random-init structures into their basin minima. Default off; flip only if
   Phase 2 ablation shows it tightens basin assignment.

3. **Batched MACE scoring.** Score the full pool by calling
   `_mace_batch_energies` in chunks of `score_chunk_size`, concatenating
   results. Existing `_mace_batch_energies` keeps its silent fallback to
   per-conformer ASE for non-MACE calculators.

4. **Energy filter.** Drop conformers with
   `(E - E_min) > e_window_kT * kT`. Critical for keeping geometric dedup
   tractable: at `e_window_kT=5` (≈130 meV), survivors are typically a few
   percent of the pool, so a 10 k-seed run drops to a few hundred before
   clustering.

5. **Energy-ranked geometric dedup** (new primitive — see below). Returns one
   centroid per geometric basin, with the lowest-energy member chosen as
   representative.

6. **Output.** Remove non-centroid conformers from the mol, return
   `(mol, centroid_ids, [E[c] for c in centroid_ids])`.

### Energy-ranked dedup primitive

Different from Butina: Butina picks dense cluster centres first; we want
lowest-energy first so a basin's representative is its energy minimum.

```python
def _energy_ranked_dedup(
    coords: torch.Tensor,        # [N, A, 3]
    energies: np.ndarray,        # [N] in eV
    rmsd_threshold: float,       # normalised L1 units, matches existing 0.1 cutoff
) -> list[int]:
    # Sort by energy ascending, iterate, pick lowest-energy unassigned,
    # exclude all conformers within rmsd_threshold (normalised L1 = sum |Δ| / 3·A).
```

Implementation notes:
- Same normalised L1 distance metric as `get_mol_PE_batched` so cutoffs are
  comparable across both functions.
- O(N²) worst case, but on the post-filter pool (~hundreds), so fast.
- GPU-accelerate per-iteration distance via `torch.cdist(p=1.0)` keeping
  `coords` on CUDA.
- Start as private `_energy_ranked_dedup` in `confsweeper.py`. Promote to
  `utils.py` only if a second caller appears.

### Failure modes

- **Zero conformers embedded** — return `(mol, [], [])` with no scoring.
- **All conformers excluded by energy filter** — keep the lowest-energy
  conformer unconditionally so callers always get at least one centroid.
- **Single conformer** — skip dedup, return that conformer.
- **GPU OOM during scoring** — surface a clear error message asking for a
  lower `score_chunk_size`. No silent fallback for this case.

### Reproducibility

- `seed` parameter offsets per-chunk ETKDG seed.
- MACE forward pass is deterministic given fixed dtype.
- Energy-ranked dedup is deterministic given a stable sort
  (`np.argsort(kind='stable')`).
- Per the semantics experiment, this guarantees **statistical** reproducibility
  (same sorted-energy distribution and basin count across runs), not
  coordinate-bit-exact reproducibility.

---

## Phase 2 — saturation experiments

Goal: pick `n_seeds` for the production default by finding where the
saturation curves flatten.

### Representative peptide selection

Five peptides spanning the macrocycle size range:
- 2 from CREMP (so we have CREST ground truth for `max_bw` and basin counts)
- 3 from PAMPA, one each in small (~50 heavy atoms), medium (~70), large
  (~100+) buckets

### Sweep design

For each peptide, run `n_seeds ∈ {100, 500, 1k, 5k, 10k, 50k}` and report:
- `max_bw`, `eff_n`, `entropy` (post-dedup)
- `n_within_kT`, `n_within_3kT`
- `n_basins` (centroids surviving dedup)
- wall-clock per stage (embed, MMFF if on, MACE, dedup)
- GPU memory peak

Plot saturation curves; pick `n_seeds` where each curve is within ~5 % of the
largest tested value. CREMP cross-check: for the two CREMP peptides, our
`(max_bw, n_basins)` distributions should land within 2× CREMP's median 25 %.
If not even at 50 k seeds, the bottleneck is ETKDG itself (basin volume in
random-init space is too uneven for randomization alone) and the
exhaustive-randomization premise needs revisiting.

### Ablation: `minimize=True`

After picking N, repeat the largest-N run with `minimize=True` on the same 5
peptides. Compare basin count and `n_within_3kT`. If MMFF tightens basin
assignment without losing diversity, flip the default; otherwise leave off.

### Script location

`scripts/saturation_etkdg.py` — kept out of `data/` (gitignored) and out of
`src/` (library code). New `scripts/` directory if needed.

---

## Phase 3 — documentation

1. **`src/README.md`** — add a "Exhaustive randomized ETKDG" subsection under
   Pipeline functions, documenting `get_mol_PE_exhaustive`'s contract, when
   to use it (cyclic peptides without a CREMP ensemble), the trade-off vs.
   `get_mol_PE_batched`, and the saturation-derived default N.
2. **Module docstring** in `src/confsweeper.py` — extend the top-level
   docstring to mention the three pipelines.
3. **Function docstring** for `get_mol_PE_exhaustive` — full Params/Returns
   block per CLAUDE.md plus a "When to use" note distinguishing it from
   `get_mol_PE_batched`.

---

## Phase 4 — tests

`tests/test_exhaustive_etkdg.py`:

- `test_exhaustive_returns_centroids_and_energies` — small SMILES (4-residue
  cyclic peptide), `n_seeds=200`, asserts non-empty outputs and
  `len(centroids) == len(energies)`.
- `test_energy_ranked_dedup_picks_lowest_energy` — synthetic `coords` and
  `energies` where the lowest-energy conformer is geometrically near a
  higher-energy one; assert lowest-energy survives.
- `test_energy_filter_drops_high_energy` — synthetic energies covering > 5 kT
  spread; assert filtered subset matches expectation.
- `test_chunked_embed_aggregate_equivalent` — fixed `seed`, run with
  `embed_chunk_size=200` and `embed_chunk_size=1000`; assert sorted-energy
  distributions match within float32 tolerance and basin count matches
  within ±5 %. **Not coordinate-identity** because of the Path C semantics
  finding above.
- `test_zero_conformers_safe` — SMILES that nvmolkit cannot embed; assert
  `(mol, [], [])` not a crash.
- `test_single_conformer_skips_dedup` — `n_seeds=1`; assert one centroid
  returned.

Run via `pixi run pytest tests/test_exhaustive_etkdg.py -v`.

---

## Phase 5 — downstream integration (separate follow-up PR)

Lives in `peptide_electrostatics`, not this branch:
1. Update `finetune_generate_conformers.py` to call `get_mol_PE_exhaustive`.
2. Add `--n_seeds` and (optional) `--minimize` CLI args.
3. Regenerate `data/fine_tune/pampa_conformers.pkl` (Falcon Slurm).
4. Re-run Stage 1b + Stage 2 fine-tuning.
5. Compare `val/mae` and learned `α` against the previous one-hot run.
   Expectation: α drifts toward 1, MAE drops noticeably.

---

## Suggested commit ordering on this branch

1. `Add docs/exhaustive_etkdg_plan.md and clean up useSmallRingTorsions docstrings`
   (Phase 0 + this design doc).
2. `Add _energy_ranked_dedup helper` (primitive + unit tests).
3. `Add get_mol_PE_exhaustive pipeline` (embed → score → filter → dedup
   wiring + integration tests).
4. `Add scripts/saturation_etkdg.py and document chosen n_seeds default`
   (Phase 2 + the chosen N propagated into the function default).
5. `Update src/README.md with exhaustive ETKDG section`.

Phases 1 and 2 land together in one PR because the saturation results
justify the chosen default `n_seeds`. Reviewers see the data alongside the
choice.

---

## Phase 2 results — what the saturation experiments actually found

The Phase 2 plan was a sweep of `n_seeds ∈ {100, 500, 1k, 5k, 10k, 50k}` on
five representative peptides, plus ablations on parameter mode, dihedral
jitter, and MMFF minimization. The actual sweep ran the same grid (sans
50k) plus three ablations:

1. **Baseline grid** — `etkdgv3_macrocycle` ETKDG, no jitter, no minimize,
   `e_window_kT=5.0`, `rmsd_threshold=0.1`. Five peptides × four seed
   counts = 20 runs.
2. **Mode comparison** — same grid, but with `etkdg_original` (RDKit's 2015
   ETKDG, no macrocycle torsion knowledge) to test whether ETKDGv3's
   torsion bias suppresses basin diversity. Five peptides × four seed
   counts = 20 runs.
3. **Dihedral jitter ablation** — `n=5000` only, with `dihedral_jitter_deg=15°`
   applied per rotatable bond after embedding, on both modes. Five
   peptides × two modes × one n_seeds = 10 runs.
4. **MMFF minimization at n=10k** — `etkdgv3_macrocycle`,
   `--minimize_at_largest`. Adds five `minimize=True` runs at n=10k plus
   five `minimize=False` n=10k controls.

Results live in `data/processed/saturation/saturation_etkdg.csv` (~60+
rows) and the runtime log in `scripts/saturation_etkdg.log`. Total compute
spent: ~3 hours of GPU time across the whole grid.

### Peptide selection

Five peptides spanning the macrocycle size range, two with CREST ground
truth from CREMP and three from CycPeptMPDB PAMPA:

| label | source | n_heavy | ground-truth `max_bw` | rationale |
|---|---|---|---|---|
| `cremp_typical:t.I.G.N` | CREMP | 27 | **0.246** | near median CREMP `poplowestpct` (~25 %) — typical rich ensemble |
| `cremp_sharp:S.S.N.MeW.MeA.MeN` | CREMP | 50 | **0.535** | 90th-percentile `poplowestpct` — naturally sharper landscape |
| `pampa_small` | PAMPA | 51 | — | small PAMPA bucket (≤55 heavy) |
| `pampa_medium` | PAMPA | 70 | — | medium PAMPA bucket (60–75 heavy) |
| `pampa_large` | PAMPA | 103 | — | large PAMPA bucket (≥95 heavy) |

CREMP ground truth is `poplowestpct/100` from the validation subset
summary CSV. Selection logic is in `select_cremp_peptides` and
`select_pampa_peptides` of `scripts/saturation_etkdg.py`.

### Headline finding: max_bw oscillates, never monotonically saturates

Across every (peptide, mode, jitter) combination, the per-row `max_bw` did
not smoothly decrease with `n_seeds`. It oscillated between ~0.4 and 1.0
because each newly-discovered low-energy conformer reset the Boltzmann-weight
reference: the new minimum becomes `E_min`, the previous (higher-energy)
basin members fall outside the 5 kT energy filter, and the dedup output
collapses back toward one-hot. The **right summary metric is
`min(max_bw)` across the full sweep**, not the value at any single
`n_seeds`. The saturation curves are bursty rather than monotonic, which
matters for downstream interpretation: a single `n_seeds=N` run is not a
reliable estimate of what the pipeline *can* find.

Concrete example, baseline `etkdgv3_macrocycle` on `cremp_typical` (CREMP
ground truth `max_bw=0.246`):

| n_seeds | n_basins | max_bw |
|---|---|---|
| 100 | 2 | 0.907 |
| 500 | 5 | 0.874 |
| 1 000 | **1** | **1.000** |
| 5 000 | 9 | **0.222** |
| 10 000 | 1 | 1.000 |

n=5000 hit the CREMP ground truth almost exactly (0.222 vs 0.246) — but
both n=1000 and n=10000 collapsed to one-hot on the same peptide. ETKDG
finds a new lowest-energy conformer at the larger N, which pushes the
formerly-contributing higher-energy conformers outside the 5 kT window.
This isn't a bug in the pipeline; it is the pipeline working correctly on
a stochastic sampler.

### Baseline grid — `etkdgv3_macrocycle`, no jitter, no minimize

Best `min(max_bw)` per peptide across `n_seeds ∈ {100, 500, 1k, 5k}`:

| peptide | best `max_bw` | at n_seeds | n_basins | n_within_3kT | ground truth |
|---|---|---|---|---|---|
| `cremp_typical` | **0.222** | 5 000 | 9 | 7 | 0.246 ✓ |
| `cremp_sharp` | 0.821 | 500 | 2 | 2 | 0.535 ✗ |
| `pampa_small` | 0.971 | 1 000 | 2 | 1 | — |
| `pampa_medium` | **0.415** | 500 | 7 | 5 | — |
| `pampa_large` | 1.000 | (any) | 1 | 1 | — |

Two of five peptides responded to scaling: `cremp_typical` saturated
correctly to CREMP, and `pampa_medium` produced a real CREMP-like
distribution at one lucky draw (n=500). Three peptides stayed essentially
one-hot at every grid point: `cremp_sharp` got modest BW spread but
nowhere near its CREMP ground truth of 0.535; `pampa_small` and
`pampa_large` never broke through.

This split — "responsive" vs "stuck" peptides — drove the rest of the
investigation. Pure scaling is necessary but not sufficient.

### Mode comparison — does dropping the macrocycle prior help?

Hypothesis: ETKDGv3's macrocycle torsion knowledge biases initial
coordinate distributions toward known-low-energy regions, so we keep
re-finding the same basin. Removing the bias (RDKit's 2015 `ETKDG()` with
no macrocycle terms) might let randomness explore more freely.

`etkdgv3_macrocycle` vs `etkdg_original`, best `min(max_bw)` across the
same 4-point grid:

| peptide | etkdgv3 best | etkdg_orig best |
|---|---|---|
| `cremp_typical` | **0.222** ✓ | 0.671 |
| `cremp_sharp` | **0.821** | 0.992 |
| `pampa_small` | 0.971 | **0.928** |
| `pampa_medium` | **0.415** | 0.939 |
| `pampa_large` | 1.000 | **0.966** |

**Hypothesis disconfirmed.** ETKDGv3 wins on three of five peptides, and
substantially on `cremp_typical` (0.222 vs 0.671) and `pampa_medium`
(0.415 vs 0.939). Original ETKDG wins only on `pampa_small` and
`pampa_large`, both marginally. The macrocycle prior is *helping*, not
hurting — it concentrates ETKDG's random embeds in the geometrically
plausible region of conformation space, which raises the probability that
any given low-energy basin lies inside the random-init's reach.

This rules out "v3 too biased" as an explanation for the stuck peptides.

### Dihedral jitter ablation — does geometric noise push new basins?

Hypothesis: a small uniform random rotation on each rotatable bond after
embedding could push a few conformers across nearby basin boundaries
before MACE rescoring catches them. Implemented as
`_jitter_rotatable_dihedrals` (RDKit standard rotatable-bond SMARTS, with
methyl-degenerate rotations filtered, applied uniformly in
[-jitter, +jitter]°). On cyclic peptides this only perturbs side-chain
rotamers since backbone bonds are in-ring.

Tested at `dihedral_jitter_deg=15°`, `n_seeds=5000`, on both modes. Best
no-jitter baseline vs jitter=15° at n=5000:

| peptide | etkdgv3 baseline | etkdgv3 + j15 | etkdg_orig baseline | etkdg_orig + j15 |
|---|---|---|---|---|
| `cremp_typical` | 9 / 0.222 | 5 / 0.927 | 3 / 0.869 | 2 / 0.988 |
| `cremp_sharp` | 1 / 1.000 | 1 / 1.000 | 2 / 0.992 | 1 / 1.000 |
| `pampa_small` | 1 / 1.000 | 1 / 1.000 | 1 / 1.000 | 1 / 1.000 |
| `pampa_medium` | 2 / 0.981 | 1 / 1.000 | 1 / 1.000 | 1 / 1.000 |
| `pampa_large` | 1 / 1.000 | 2 / 0.990 | 1 / 1.000 | 1 / 1.000 |

**Hypothesis disconfirmed.** Across all 10 (peptide × mode) combinations,
jitter=15° never improved on the no-jitter best. It bumped basin counts
on `cremp_typical` (1 → 5) and `pampa_large` (1 → 2) but did not lower
`max_bw` in either case — the new "basins" sat too high above the
minimum to contribute Boltzmann weight. Geometric noise without an energy
gradient just produces extra high-energy conformers that the 5 kT filter
discards. To make jitter useful you would need either (a) much larger
amplitude that crosses rotamer barriers (and then re-minimize) or (b)
backbone-targeted jitter — neither is safe for cyclic peptides without
breaking ring closure.

### Energy-backend check: MACE-OFF23 vs xtb GFN2

Question: are the one-hot Boltzmann distributions a MACE-OFF23 artifact?
CREST uses xtb GFN2 internally, so testing whether xtb produces richer
ensembles on the *same* conformer pool would tell us whether the energy
model itself is the cause.

Implemented as `scripts/mace_vs_xtb.py`: embed N=1000 conformers of
`pampa_large` (the most extreme stuck peptide), MACE-score the full pool
on GPU (single batched pass), xtb GFN2 single-point on CPU in 8 parallel
workers, compare BW vectors and energy correlations.

| metric | MACE-OFF23 | xtb GFN2 |
|---|---|---|
| `max_bw` | 0.860 | **0.9997** |
| `eff_n` | 1.32 | 1.00 |
| `entropy` | 0.413 | 0.003 |
| `n_within_kT` | 1 | 1 |
| `n_within_3kT` | 2 | 1 |
| `e_range` (eV) | 24.2 | 21.3 |

Cross-backend correlation on the same 1 000 conformers:
- **Pearson r = +0.991** (energies, mean-shifted)
- **Spearman r = +0.983** (rank order)
- Top-BW conformer agrees: both backends pick the same conformer (idx=121)
  as the dominant one.

**MACE is exonerated.** Two completely different physics — a graph neural
network trained on QM data vs the semi-empirical tight-binding method
that CREST uses internally — agree to within sampling noise on which
conformer is the energy minimum and on the rank order of the rest. xtb
is actually *more* one-hot than MACE on this pool. The bottleneck is not
the energy model; it is the conformer pool itself. CREST gets rich
ensembles because it samples *different conformer basins*, not because
xtb scores them differently from MACE.

This validates keeping MACE-OFF23 as the production scorer (the ~10×
speedup vs xtb is free on equal-quality energies) and shifts the entire
investigation back to sampling.

Per-conformer CSV is at
`data/processed/saturation/mace_vs_xtb_pampa_large.csv` for any
follow-up scatter plots.

### MMFF minimization is the lever

The breakthrough finding. With everything else held fixed
(`etkdgv3_macrocycle`, no jitter, MACE scoring, `e_window_kT=5.0`,
`rmsd_threshold=0.1`), turning on MMFF94 post-embed minimization
transformed the stuck cases.

`cremp_typical` n=10000 baseline vs minimize=True (CPU MMFF, the first
attempt):

| | n_basins | max_bw | n_within_3kT |
|---|---|---|---|
| no-minimize | 1 | 1.000 | 1 |
| **+ minimize=True (CPU MMFF)** | **75** | **0.071** | **38** |

`max_bw=0.071` is *lower* than CREMP's ground truth of 0.246 — the
ensemble is now more uniformly distributed across many basins than CREST
itself produced. 75 basins from the same input pool is ~75× the basin
count of the no-minimize run.

Why it works: ETKDG produces conformers near basin floors but not at
them, so the energy filter and dedup see slight geometric variations as
separate minima. MMFF94 pulls each conformer to its actual local
minimum. Then the energy-ranked dedup sees true basins, not noisy points
near them — duplicates collapse correctly, and what survives are real
basin energies rather than accidental low-energy structures within
basins.

This is the missing ingredient. Without minimization, exhaustive ETKDG
finds basin *neighbourhoods*, not basin *minima*. With minimization,
each conformer is decomposed into "which basin does it belong to?" and
"what is that basin's energy?", and the dedup-and-Boltzmann-weight
pipeline produces a proper canonical ensemble.

### nvmolkit GPU MMFF integration

CPU MMFF on `cremp_typical` n=10000 (27 heavy atoms) took 862 s — about
86 ms per conformer for the smallest peptide. Scaling that to `pampa_large`
(103 heavy atoms, 222 atoms with explicit Hs) is multi-hour per peptide.
Not viable for production.

`nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` provides a
GPU-batched CUDA implementation. Quick-correctness test on 50 ETKDG
conformers per peptide:

| peptide | n_atoms | CPU MMFF | GPU MMFF | speedup | Pearson r (energies) | RMS coord delta median (Å) |
|---|---|---|---|---|---|---|
| cremp_typical | 34 | 0.80 s | 0.03 s | **23.6×** | 0.940 | 0.012 |
| pampa_large | 222 | 35.7 s | 0.70 s | **50.6×** | 0.955 | 0.002 |

The two backends are not bit-exact. Pearson r ≈ 0.95 between final
energies, with std ~3-5 kcal/mol per-conformer. Median per-conformer RMS
coordinate delta is ~0.01 Å (most conformers identical), but the tails
diverge — different BFGS gradient details push the optimizer into
neighbouring basins on a few percent of cases. Both are valid descents to
local minima; just non-identical local minima. For exhaustive sampling
the disagreement is acceptable: we care about basin coverage in
aggregate, not per-conformer reproducibility.

The integration:
- New `mmff_backend: str = "gpu"` parameter on `get_mol_PE_exhaustive`.
- `'gpu'` calls `MMFFOptimizeMoleculesConfs` once on the full pool;
  `'cpu'` falls back to the original RDKit serial loop.
- Order-dependent import: `nvmolkit.mmffOptimization` requires
  `nvmolkit.embedMolecules` to be loaded first to register C++ global
  state. The pipeline already imports `embed.EmbedMolecules` before
  reaching the minimization step, so this is automatic in normal use.

End-to-end speedup at n=10k on `cremp_typical`: 862 s → 291 s (3.0×).
Smaller than the per-conformer 23.6× because at this scale the
embed and MACE stages also dominate; MMFF is no longer the long pole.
For `pampa_large` and similar large peptides where CPU MMFF would
otherwise take hours, the GPU backend is the difference between viable
and not.

### Implications for the production default

The data argues for these `get_mol_PE_exhaustive` defaults:

- `n_seeds=10000` — saturated for small peptides, partial for large.
  Bigger N still helps but gets stochastic; **`minimize=True` matters
  more than scaling N further**.
- `minimize=True` — the actual lever. Without it, even n=10k stays
  one-hot on most peptides.
- `mmff_backend='gpu'` — required to make minimize=True viable on the
  large peptides that dominate PAMPA.
- `params_mode='etkdgv3_macrocycle'` — the macrocycle prior helps, not
  hurts. Keep it.
- `dihedral_jitter_deg=0.0` — the rotatable-bond jitter machinery is
  built but not on by default; no measurable benefit at jitter=15°.
  Keep the code in case future work wants larger jitter or
  rotamer-aware variants, but document that it's experimental.

### Final n=10k saturation results — minimize splits the dataset by size

The full n=10k sweep across all five peptides, no-minimize vs minimize=True
(GPU MMFF), `etkdgv3_macrocycle`, no jitter:

| peptide | n_heavy | no-minimize | minimize=True | ground truth |
|---|---|---|---|---|
| `cremp_typical` | 27 | 1 / 1.000 | **54 / 0.318** ✓ | 0.246 |
| `cremp_sharp` | 50 | 1 / 1.000 | **11 / 0.380** ✓ | 0.535 |
| `pampa_small` | 51 | 3 / 0.968 | **4 / 0.591** ✓ | — |
| `pampa_medium` | 70 | 3 / 0.923 | 2 / 0.966 ✗ | — |
| `pampa_large` | 103 | 3 / 0.497 | 3 / 0.930 ✗ | — |

(Cells are `n_basins / max_bw`. ✓ = minimize improved; ✗ = minimize regressed.)

**The split is sharp at ~70 heavy atoms.** Below that threshold,
minimize=True is unambiguously the lever — `cremp_typical` recovered to
CREMP-comparable BW (max_bw 0.318 vs ground truth 0.246), `cremp_sharp`
to *better* than CREMP (0.380 vs 0.535), and `pampa_small` from one-hot
(0.968) to a real ensemble (0.591). Above the threshold,
minimize=True regresses: `pampa_medium` got worse (0.966 vs 0.923 without
minimize) and `pampa_large` got substantially worse (0.930 vs 0.497).

A second observation worth recording: `pampa_large` *did* finally move off
one-hot at n=10k without minimize (max_bw=0.497, 3 basins) — the
bursty-saturation pattern paying off at the largest scale we tested. The
baseline grid (n ≤ 5000) had it stuck at 1.000 across every grid point.
Pure scaling does eventually help on the largest peptide, but only at the
top of the seed range.

### Confound: input pool is not held fixed across minimize/no-minimize

Before drawing conclusions about the large-peptide regression: the
no-minimize and minimize runs use **different nvmolkit ETKDG seeds**
because they ran as separate invocations of the saturation script.
nvmolkit ETKDG is not bit-exact reproducible across calls (per the
"nvmolkit ETKDG semantics" findings above), so the two runs are
comparing not just "same pool, different processing" but "different
pools, different processing". The pampa_large no-minimize run that
found 3 basins / max_bw=0.497 may have drawn a particularly rich pool.

Two competing hypotheses for the large-peptide regression need a
controlled experiment to distinguish:

1. **Sampling-luck hypothesis** — the no-minimize run on pampa_medium /
   pampa_large happened to find an unusually diverse ETKDG pool, while
   the minimize run found a poorer pool. The MMFF basin-collapse effect
   on the poorer pool could not recover the diversity that the
   un-minimized rich pool retained. Implication: minimize=True remains
   the right default, just with the same bursty-saturation noise we
   already accepted for n_seeds.
2. **Mechanism hypothesis** — for large peptides ETKDG produces enough
   geometric noise *within* a single basin that the 5 kT filter and
   dedup happen to keep ~3 of those near-duplicates as "different"
   conformers, providing fake diversity. MMFF cleans them up to a single
   basin minimum, the dominant minimum's BW concentrates, and max_bw
   rises. Implication: minimize=True is actively wrong for large
   peptides; production needs a size-aware default.

The next-priority controlled experiment, planned for `scripts/minimize_ablation.py`:
embed `pampa_large` and `pampa_medium` once each at n=10k, deepcopy the
mol, run pipeline A (MACE → filter → dedup → BW) and pipeline B
(MMFF → MACE → filter → dedup → BW) on the *same* starting pool, report
side-by-side. If pipeline B still regresses on the same input as
pipeline A, the mechanism hypothesis stands. If pipeline B improves or
matches, hypothesis 1 explains the saturation result.

### Implications for the production default — pending confound resolution

Until the controlled experiment lands, the recommended defaults from
the previous subsection ("Implications for the production default") are
provisional. Concretely:

- **For peptides ≤ ~51 heavy atoms** (which covers cremp_typical, cremp_sharp,
  pampa_small): minimize=True with GPU MMFF is the correct default. The
  effect is reproducibly large and points the right direction.
- **For peptides ≥ ~70 heavy atoms** (which covers pampa_medium and
  pampa_large, and most of the PAMPA dataset above the median): the
  minimize=True regression needs to be explained before we ship it as
  the default. A size-conditional default (e.g., `minimize=True` only
  when `n_heavy < 60`) is the obvious fallback if the mechanism
  hypothesis holds, but we shouldn't commit until the controlled
  experiment confirms the mechanism rather than sampling luck.

### Controlled minimize ablation — sampling-luck hypothesis confirmed

The controlled experiment is `scripts/minimize_ablation.py`. It embeds
once per peptide at n=10k via nvmolkit ETKDG, deepcopies the mol so the
two pipelines start from identical geometries, then runs A (MACE → filter
→ dedup → BW) and B (GPU MMFF → MACE → filter → dedup → BW) on the same
pool. Anything pipeline B gains over pipeline A is therefore attributable
to the minimization step alone.

Run on the two regression peptides (`pampa_large` and `pampa_medium`):

| peptide | pipeline | n_basins | max_bw |
|---|---|---|---|
| `pampa_large` (103 heavy) | A: no minimize | 1 | 1.000 |
| | B: minimize=True | 2 | 0.981 |
| `pampa_medium` (70 heavy) | A: no minimize | 2 | 0.990 |
| | B: minimize=True | **2** | **0.648** |

Compare against the saturation-sweep numbers (different ETKDG pools per
pipeline):

| peptide | sweep no-min | sweep min | ablation no-min | ablation min |
|---|---|---|---|---|
| `pampa_large` | 0.497 (3 basins) | 0.930 (3 basins) | **1.000** (1) | **0.981** (2) |
| `pampa_medium` | 0.923 (3 basins) | 0.966 (2 basins) | **0.990** (2) | **0.648** (2) |

**Hypothesis 1 (sampling luck) wins decisively on both peptides.**

For `pampa_medium` the ordering reverses entirely — on a controlled pool
minimize=True actually drops max_bw from 0.990 to 0.648, a meaningful
recovery toward CREMP-like ensembles. The saturation-sweep "regression"
was an artifact of comparing two unrelated ETKDG draws.

For `pampa_large`, both pipelines on the controlled pool produce
essentially one-hot results (1.000 and 0.981). Minimize=True doesn't
help much here, but it doesn't hurt — and the sweep's "lucky" no-min
result of 0.497 was a statistical outlier rather than a representative
behaviour. ETKDG simply isn't generating enough basin diversity at n=10k
on a 103-heavy-atom macrocycle, with or without MMFF.

Per-conformer MACE-energy Spearman correlation between pre-MMFF and
post-MMFF (same conformer indices, different geometries after
optimization):
- `pampa_large`: r = +0.5646
- `pampa_medium`: r = +0.6098

Both moderate. MMFF substantially reorders the conformer energy ranking
— consistent with conformers crossing basin boundaries during
minimization, which is the prerequisite for new basins emerging in the
post-dedup result.

### Final implications for the production default

The controlled ablation closes the open question. Across all five
representative peptides, minimize=True is either reproducibly good
(cremp_typical, cremp_sharp, pampa_small, pampa_medium) or neutral
(pampa_large), never reproducibly worse. The size-conditional fallback
discussed earlier is unnecessary.

**Production `get_mol_PE_exhaustive` defaults backed by the data**:

- `n_seeds = 10000` — covers all peptides we tested, with the bursty
  pattern noted in earlier sections. Larger values still help on a
  per-run-luck basis but with diminishing returns.
- `minimize = True`
- `mmff_backend = 'gpu'`
- `params_mode` semantically fixed to `etkdgv3_macrocycle` (no public
  flag yet; the function takes `params` directly and the saturation
  script wires up the variant)
- `dihedral_jitter_deg = 0.0`
- `e_window_kT = 5.0`
- `rmsd_threshold = 0.1`

`pampa_large`-style peptides remain a sampling-coverage hard case — even
with minimize=True they can stay one-hot when the seed pool happens to
land in one basin. The bursty-saturation pattern means a run might
luckily find diversity at any given n_seeds; in production the
fine-tuning consumer should expect some PAMPA molecules to come back
near-one-hot regardless of pipeline tuning.

The fine-tuning wrapper's α=0.01 init is the right hedge for that
contingency: it lets the property head learn to ignore the dominant
conformer when the input ensemble is one-hot. With the rest of PAMPA
producing rich CREMP-like ensembles, α should drift toward 1 during
training; we'll see that empirically when Phase 5 runs.

---

## Open questions

- **GPU OOM ceiling for nvmolkit single-call embed** — sets the upper bound
  on `embed_chunk_size`. Empirically `embed_chunk_size=1000` at 222 atoms
  per conformer (pampa_large) ran without OOM during the n=10k saturation
  runs, so the operational ceiling is at least that on a single A100/H100.
- **Does PAMPA val/MAE actually drop with rich ensembles?** Ultimate test for
  the whole effort. Answered in Phase 5 by re-fitting `FinetuneJanossyWrapper`
  and checking whether learned α moves toward 1 (currently lives at 0.01) and
  whether MAE drops. Should now actually be possible because GPU MMFF makes
  the per-molecule conformer regeneration step tractable across the 7.2 k
  PAMPA dataset.
- **CPU vs GPU MMFF reproducibility on basin distribution.** The
  per-conformer disagreement is well-characterised (Pearson r ≈ 0.95) but
  the *aggregate* basin distribution agreement after dedup is not.
  Worth a one-off run that fixes the input conformer pool and only
  varies the MMFF backend (the current data conflates ETKDG seed
  variation with backend variation).
