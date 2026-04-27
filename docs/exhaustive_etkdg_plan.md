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

Implication for the implementation: use Path A as the default fast path; only
fall back to chunking when `n_seeds > embed_chunk_size`. Plan for stochastic
correctness tests, not coordinate-identity tests.

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

## Open questions

- **GPU OOM ceiling for nvmolkit single-call embed** — sets the upper bound
  on `embed_chunk_size`. Plan: start at 1000, profile during Phase 2.
- **Does ETKDG's internal cleanup land conformers in basin minima**, or is
  explicit MMFF / MACE local minimization required for geometric dedup to be
  meaningful? Phase 2 ablation answers this.
- **Does PAMPA val/MAE actually drop with rich ensembles?** Ultimate test for
  the whole effort. Answered in Phase 5 by re-fitting `FinetuneJanossyWrapper`
  and checking whether learned α moves toward 1 (currently lives at 0.01) and
  whether MAE drops.
