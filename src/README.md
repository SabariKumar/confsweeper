# src/

Core confsweeper library. Five modules handle distinct responsibilities:
`confsweeper.py` owns the full generation-to-scoring pipeline and is general-purpose —
it works for any molecule type. `torsional_sampling.py` provides the backbone
dihedral-constrained Phase 2 conformer pool and is **macrocycle-specific**: it
assumes a head-to-tail cyclic peptide backbone and should not be used for acyclic
molecules. `mcmm.py` and `concerted_rotation.py` implement the Multiple Minimum
Monte Carlo with replica exchange (MCMM-REMD) sampler: a non-ETKDG basin-hopping
search built around Dodd-Boone-Theodorou (DBT) concerted-rotation moves on
backbone windows. Both are **macrocycle-specific** and consumed by
`get_mol_PE_mcmm` for issue #10's sampler benchmark. `utils.py` provides
geometry comparison utilities used by the validation layer.

---

## confsweeper.py

The main pipeline module. Its public surface is a family of `get_mol_PE*` functions
that each take a SMILES string and return `(mol, conf_ids, energies)` — a molecule
with attached conformers, the integer IDs of the Butina-representative subset, and
the neural-network potential energies for those representatives in eV.

### Embedding parameters

Two helper functions build RDKit `EmbedParameters` objects for nvmolkit:

- **`get_embed_params()`** — baseline ETKDGv3 with `useRandomCoords=True`. Suitable
  for small molecules.
- **`get_embed_params_macrocycle()`** — adds `useMacrocycleTorsions` and
  `useMacrocycle14config`, the ETKDGv3 flags that enable a special sampling regime for
  ring-closure constraints on cyclic backbones. Use this for any cyclic peptide input.
  `useSmallRingTorsions` is intentionally left disabled — nvmolkit does not support
  the flag and hangs indefinitely in CPU preprocessing when it is set.

### Energy backends

Three calculators are available, all returning energies in eV for compatibility with
downstream Boltzmann-weighting code:

| Function | Backend | Environment | Status |
|----------|---------|-------------|--------|
| `get_mace_calc()` | MACE-OFF (small/medium/large) | `mace` pixi env | **Supported** |
| `get_mol_PE_mmff` | RDKit MMFF94 | any | Supported |
| `get_uma_calc()` | FairChem UMA-S, omol task | `default` pixi env | **Not supported** |

UMA (`get_uma_calc`) is present in the codebase but not actively maintained. Use MACE-OFF.

MMFF94 energies are converted from kcal/mol via the constant `_KCAL_TO_EV = 0.043364`.
The conversion is applied in `get_mol_PE_mmff` so callers always see eV regardless of
which backend scored the conformers.

### Pipeline functions

**`get_mol_PE`** — the reference implementation. Scores each Butina representative
in a separate ASE calculator call. Works with both MACE and UMA but is slow for
large representative sets because it iterates conformers sequentially.

**`get_mol_PE_batched`** — preferred for MACE, and the only function that supports
torsional sampling. After Pool A embedding it optionally calls `sample_constrained_confs`
to append Pool B conformers to the same mol object, then runs a single Butina pass over
the merged pool before scoring. All Butina representatives are scored in one MACE forward
pass via `_mace_batch_energies`. If the MACE batching API is unavailable or raises
(e.g. when `calc` is a UMA calculator), `_mace_batch_energies` falls back silently
to sequential scoring.

Torsional sampling is activated by passing `grids` (loaded from the CREMP Ramachandran
`.npz` via `load_ramachandran_grids`) and setting `n_constrained_samples > 0`. When
`grids=None` (the default), behaviour is identical to before — no overhead is added.

**`get_mol_PE_exhaustive`** — randomized-saturation pipeline for cyclic peptides
and other molecules where `get_mol_PE_batched` produces near-one-hot Boltzmann
distributions because ETKDG-100 misses low-energy basins. Embeds an order of
magnitude more conformers (`n_seeds=10000` by default), optionally
MMFF-minimises them on the GPU via `nvmolkit.mmffOptimization`, MACE-scores
in chunks, applies a 5 kT energy filter, and dedups by basin energy minimum
rather than cluster density. Returns the same `(mol, conf_ids, energies)`
contract as the other pipelines, so downstream consumers swap one function
call. See `docs/exhaustive_etkdg_plan.md` for the saturation experiments
behind the chosen defaults; the headline result is that exhaustive ETKDG +
MMFF reproduces CREST-quality Boltzmann distributions on most cyclic
peptides we tested (e.g. `cremp_typical`: 27 heavy atoms, max_bw 0.318 vs
CREST ground truth 0.246) at a fraction of CREST's compute cost.

The seven-stage pipeline:
1. Massive ETKDG embed (chunked when `n_seeds > embed_chunk_size`).
2. Optional rotatable-bond dihedral jitter (off by default; experimental).
3. Optional MMFF94 minimisation (on by default), `gpu` or `cpu` backend.
4. Batched MACE scoring in chunks of `score_chunk_size`.
5. Energy filter: drop conformers with `(E - E_min) > 5 kT`.
6. Energy-ranked geometric dedup via `_energy_ranked_dedup` — picks the
   lowest-energy conformer of each geometric basin, in contrast to Butina
   which picks dense cluster centres.
7. Returns one centroid per basin, ordered by ascending energy.

**`get_mol_PE_pool_b`** — sister function to `get_mol_PE_exhaustive` for the
sampler benchmark (issue #10). Same `(mol, conf_ids, energies)` contract and
the same post-sampling tail (MMFF → MACE batched scoring → 5 kT energy filter
→ `_energy_ranked_dedup` → non-centroid pruning), but Phase 1 swaps nvmolkit
ETKDG for `sample_constrained_confs`, which embeds each conformer with the
bounds matrix tightened to a CREMP-derived (phi, psi) target. **Macrocyclic
peptides only** — `sample_constrained_confs` requires backbone (phi, psi)
atoms. Defaults are calibrated for matched-budget benchmarking against
`get_mol_PE_exhaustive`: `n_samples=10000`, `n_attempts=1` (caps the raw pool
near 10k), `strategy='inverse'` (oversamples rare-but-accessible Ramachandran
cells, which is the design intent — fill the gaps ETKDG misses).

The shared post-sampling tail is implemented in the private
`_minimize_score_filter_dedup(mol, all_conf_ids, hardware_opts, calc, ...)`
helper. `get_mol_PE_exhaustive`, `get_mol_PE_pool_b`, and `get_mol_PE_mcmm`
are all thin wrappers around their respective Phase 1 sampler plus a single
call to that helper.

**`get_mol_PE_mcmm`** — third sampler for the issue-#10 benchmark.
**Macrocyclic peptides only.** Phase 1 is Multiple Minimum Monte Carlo with
replica exchange (MCMM-REMD) implemented in `src/mcmm.py`: an ETKDG seed (or
`n_init_confs` seeds, distributed round-robin across walkers per lever C9)
is MMFF-relaxed and used to initialise a stack of walkers across a geometric
temperature ladder. Each walker proposes a DBT concerted-rotation move (or a
GOAT-style Cartesian kick, routed by `cartesian_weight`) on a backbone
window, MMFF-relaxes the result, MACE-scores it, and decides via Metropolis
with a Saunders 1990 `1/usage^p` novelty bias against re-visiting basins
already in the shared `BasinMemory`. Swaps between adjacent temperatures
fire every `swap_interval` steps. After `n_steps` per walker, the basin set
in memory is the input pool to the shared `_minimize_score_filter_dedup`
tail. The recommended production setting per `docs/mcmm_plan.md`'s
2026-05-06 findings: `drive_sigma_rad=0.3, closure_tol=0.05,
kt_high=8 × kT_298, n_init_confs=8, cartesian_weight=0.5, e_window_kT=10,
saunders_exponent=1.0`. In-code defaults preserve the original Saunders /
5 kT pipeline conventions; production settings are locked in
`scripts/sampler_benchmark.py:_run_mcmm` so the benchmark stays
self-documenting.

**`get_mol_PE_mmff`** — like `get_mol_PE_batched` but scores with MMFF94. No GPU
required for the scoring step; GPU is still used for embedding and Butina.

### Clustering and dedup

Two distinct geometric grouping strategies are used in different code paths:

**Butina clustering** (used by `get_mol_PE` and `get_mol_PE_batched`) runs on the
pairwise L1 distance matrix between flattened conformer coordinate tensors,
normalised by `3 * n_atoms` so the cutoff is in Å-per-atom units. The GPU Butina
path (`nvmolkit.clustering.butina`) and the CPU fallback
(`rdkit.ML.Cluster.Butina.ClusterData`) both return 0-based *row indices* into
the distance matrix, not RDKit conformer IDs. In the single-pool (ETKDG-only)
case these happen to be the same; in the two-pool case (Pool A + Pool B from
torsional sampling), `get_mol_PE_batched` maps centroid row indices back through
`mol.GetConformers()` to recover actual conformer IDs. The normalisation
`/ (3 * n_atoms)` means `cutoff_dist=0.1` corresponds to a mean coordinate
deviation of 0.1 Å per atom — a coarse criterion, but cheap to compute across
thousands of conformers.

**Energy-ranked Kabsch dedup** (used by `_energy_ranked_dedup` and
`BasinMemory`, consumed by `get_mol_PE_exhaustive`, `get_mol_PE_pool_b`, and
`get_mol_PE_mcmm`) operates on heavy-atom Kabsch-aligned RMSD in Å. The
default threshold is `0.125 Å` — the same value CREST / CREMP / GOAT use for
conformer uniqueness, so `n_basins` is directly comparable to CREMP
`uniqueconfs`. The metric uses pure torch SVD (`_kabsch_rmsd_pairwise` in
`src/mcmm.py`) with rotation-matrix determinant correction to reject
reflections; atom-permutation symmetry is not handled (methyl flips remain
counted as distinct, ~5 % over-count vs. CREST's permutation-aware matching).
The dedup keeps the lowest-energy conformer of each basin rather than the dense
cluster centre, which matters when the energy landscape is rough and the
densest cluster isn't the deepest well.

Two `dedup_mode` values are available on `_energy_ranked_dedup` and
`BasinMemory`:

- **`'kabsch'`** (default) — the metric above, RMSD-only. The right default for
  day-to-day work; clean basin definitions; chosen for the AE pretraining and
  in-run MCMM dynamics.
- **`'crest'`** — CREST's three-criteria AND-test: two conformers are merged
  iff their Kabsch RMSD < threshold AND their energy difference <
  `energy_threshold_eV` (default 0.05 eV — the MACE float32 noise floor) AND
  their rotational-constant anisotropy < `rotconst_anisotropy_threshold`
  (default 0.01). Used to publish CREMP-comparable `n_basins` numbers for the
  paper. CREST mode requires `atomic_numbers` to derive inertia tensors and
  is mutually-aligned across the in-run `BasinMemory` and the post-sampler
  `_minimize_score_filter_dedup` so the count is internally consistent.

### Data contracts

**`get_mol_PE` / `get_mol_PE_mmff`** (shared params)

| Argument | Type | Notes |
|----------|------|-------|
| `smi` | `str` | SMILES; Hs are added internally via `Chem.AddHs` |
| `params` | `EmbedParameters` | from `get_embed_params_macrocycle()` for cyclic peptides |
| `hardware_opts` | `HardwareOptions` | from `get_hardware_opts()` |
| `calc` | ASE calculator | from `get_mace_calc()` or `get_uma_calc()` |
| `n_confs` | `int` | conformers to embed before Butina (default 1000) |
| `cutoff_dist` | `float` | Butina threshold in normalised L1 units (default 0.1) |
| `gpu_clustering` | `bool` | `True` to use nvmolkit GPU Butina (default) |

**`get_mol_PE_batched`** — all of the above, plus:

| Argument | Type | Notes |
|----------|------|-------|
| `grids` | `dict \| None` | from `load_ramachandran_grids()`; `None` disables Pool B |
| `n_constrained_samples` | `int` | Pool B (phi, psi) draws (default 0) |
| `torsion_strategy` | `str` | `'uniform'` or `'inverse'` (default `'uniform'`) |
| `torsion_seed` | `int` | RNG seed for Pool B reproducibility (default 0) |

**`get_mol_PE_pool_b`** — same shape as `get_mol_PE_exhaustive` minus the
ETKDG-specific knobs (`embed_chunk_size`, `dihedral_jitter_deg`), plus the
constrained-DG knobs:

| Argument | Type | Notes |
|----------|------|-------|
| `smi`, `hardware_opts`, `calc` | — | as above (no `params` — Pool B uses CPU ETKDGv3 internally) |
| `grids` | `dict` | from `load_ramachandran_grids()`; required |
| `n_samples` | `int` | (phi, psi) draw budget (default 10000) |
| `n_attempts` | `int` | ETKDGv3 attempts per draw (default 1) |
| `tolerance_deg` | `float` | dihedral constraint half-width (default 30°) |
| `strategy` | `str` | `'inverse'` (default) or `'uniform'` |
| `score_chunk_size`, `e_window_kT`, `rmsd_threshold`, `dedup_mode`, `minimize`, `mmff_backend`, `seed` | — | as in `get_mol_PE_exhaustive` |

**`get_mol_PE_exhaustive`** — different parameter space (no Butina cutoff, no
torsional sampling, no `gpu_clustering` toggle):

| Argument | Type | Notes |
|----------|------|-------|
| `smi`, `params`, `hardware_opts`, `calc` | — | as above |
| `n_seeds` | `int` | total ETKDG seeds (default 10000, saturation-validated) |
| `embed_chunk_size` | `int` | nvmolkit per-call cap before chunking (default 1000) |
| `score_chunk_size` | `int` | MACE per-batch forward-pass cap (default 500) |
| `e_window_kT` | `float` | energy filter window in `kT_298K` units (default 5.0) |
| `rmsd_threshold` | `float` | basin-dedup exclusion radius — Kabsch heavy-atom RMSD in Å (default `0.125`, matches CREMP / CREST / GOAT) |
| `dedup_mode` | `str` | `'kabsch'` (default) or `'crest'` (three-criteria AND-test; see *Clustering and dedup*) |
| `energy_threshold_eV` | `float` | CREST-mode energy criterion in eV (default 0.05; MACE float32 noise floor) |
| `rotconst_anisotropy_threshold` | `float` | CREST-mode rotational-constant criterion (default 0.01 = 1 %) |
| `minimize` | `bool` | MMFF94 minimise post-embed (default `True`, the lever) |
| `mmff_backend` | `str` | `'gpu'` (default; nvmolkit batched CUDA) or `'cpu'` (RDKit serial) |
| `dihedral_jitter_deg` | `float` | rotatable-bond jitter ±deg (default 0; experimental) |
| `seed` | `int` | base ETKDG seed; chunk i uses `seed + i*embed_chunk_size` (default 0) |

**`get_mol_PE_mcmm`** — MCMM-REMD sampler for cyclic peptides. Pipeline-tail
knobs (`score_chunk_size`, `e_window_kT`, `rmsd_threshold`, `dedup_mode`,
`energy_threshold_eV`, `rotconst_anisotropy_threshold`, `minimize`,
`mmff_backend`) match `get_mol_PE_exhaustive`. The MCMM-specific knobs:

| Argument | Type | Notes |
|----------|------|-------|
| `smi`, `params`, `hardware_opts`, `calc` | — | as above |
| `n_walkers_per_temp` | `int` | walkers at each temperature (default 8) |
| `n_temperatures` | `int` | replica-exchange ladder size (default 8) — total walkers = product |
| `kt_low` | `float \| None` | coldest replica kT in eV (default `_KT_EV_298K` ≈ 300 K) |
| `kt_high` | `float \| None` | hottest replica kT in eV (default `2 × _KT_EV_298K` ≈ 600 K; production `8 × kT_298`) |
| `n_steps` | `int` | MC steps per walker (default 200; benchmark adapter uses `max(1, n_seeds // 64)`) |
| `swap_interval` | `int` | steps between adjacent-rank swap attempts (default 20) |
| `drive_sigma_rad` | `float` | DBT drive-angle Gaussian σ (default 0.1; production 0.3) |
| `closure_tol` | `float` | DBT closure tolerance in Å (default 0.01; production 0.05) |
| `sigma_kick_a` | `float` | GOAT-style Cartesian-kick σ in Å (default 0.1; consulted only when `cartesian_weight > 0`) |
| `cartesian_weight` | `float` | per-step routing weight for Cartesian kick vs. DBT (default 0.0 = pure DBT; production 0.5) |
| `n_init_confs` | `int` | distinct ETKDG seeds (default 1; production 8 — round-robin walker distribution per lever C9) |
| `saunders_exponent` | `float` | Saunders novelty bias exponent (default 0.5 = `1/√usage`; production 1.0 = `1/usage`) |
| `seed` | `int` | base seed; derived seeds for walkers / proposer / swap RNG are deterministic offsets |

Returns `(mol, conf_ids, pe)`:
- `mol`: `Chem.Mol` with only basin-representative conformers attached
- `conf_ids`: `List[int]` of conformer IDs on `mol`, ordered by ascending energy
- `pe`: `List[float]` of energies in eV, same order as `conf_ids`

**`write_sdf`**

Writes one SDF file per molecule to `output_dir/<uuid>.sdf`. Each conformer record
carries a `MACE_ENERGY` property (eV) regardless of which backend was used. When
`save_lowest_energy=True`, only the lowest-energy conformer is written.

### Critical parameters

- **`n_confs` vs. macrocycle ring closure**: ETKDGv3 cannot always close the ring for
  every requested conformer. The actual number of embedded conformers may be substantially
  less than `n_confs`, especially for larger macrocycles. Always check `mol.GetNumConformers()`
  after embedding before assuming `n_confs` conformers are available.
- **GPU ID**: `get_hardware_opts` defaults to `gpuIds=[0]`. On multi-GPU nodes, set
  `gpuIds` explicitly; nvmolkit does not auto-select a free GPU.
- **`get_mol_PE_exhaustive` is bursty, not monotone**: `max_bw` does not smoothly
  decrease with `n_seeds`. It oscillates because each newly-discovered low-energy
  conformer resets `E_min`, sometimes pushing previously-contributing basins outside
  the 5 kT filter. The right summary metric across a sweep is `min(max_bw)`, not the
  value at any specific `n_seeds`. A single large run is more reliable than a
  fine-grained scan.
- **`get_mol_PE_exhaustive` requires `minimize=True` for cyclic peptides**: turning
  off MMFF post-minimisation reverts to one-hot Boltzmann distributions on most
  peptides above ~50 heavy atoms, defeating the purpose of the pipeline. Only set
  `minimize=False` if you specifically want pre-minimisation energies (e.g. ablation
  studies).
- **`get_mol_PE_exhaustive` GPU MMFF stochasticity**: the `mmff_backend='gpu'` path
  produces final geometries that are not bit-exact identical to `'cpu'`. Pearson r
  between final energies is ~0.95; basin distributions after dedup are similar but
  not identical. For batch saturation runs this is fine. Switch to `'cpu'` only for
  bit-exact RDKit reference behaviour, accepting a 25-50× slowdown.

---

## torsional_sampling.py

> **Macrocyclic peptides only.** This module targets the backbone (phi, psi) dihedral
> space of head-to-tail cyclic peptides. It is not suitable for acyclic molecules or
> non-peptide macrocycles. For general-purpose conformer generation, use the ETKDG
> functions in `confsweeper.py` directly.

Implements backbone dihedral-constrained conformer generation (Phase 2 of the two-pool
pipeline). The key idea is that RDKit's distance geometry algorithm (ETKDGv3) is guided
by a *bounds matrix* — a matrix of lower and upper interatomic distance constraints.
Injecting tighter distance bounds for the 1,4-atom pairs of a backbone dihedral effectively
constrains the embedded conformer to a target (phi, psi) region, without modifying the
embedding algorithm itself.

### Why distance bounds rather than torsion driving

ETKDGv3 does not expose a direct torsion angle constraint API. The bounds matrix is the
correct abstraction level: it is the input that ETKDGv3 actually reads, and it subsumes
all geometry constraints (bond lengths, bond angles, torsions) in a unified representation.
The 1,4-distance formula used here (`_d14`) is an exact analytic expression derived from
the law of cosines applied iteratively through the four-atom chain.

Note that constrained DG runs on **CPU ETKDGv3**, not nvmolkit. Each (phi, psi) draw
requires its own `EmbedParameters` object with a distinct bounds matrix. nvmolkit's batch
embedding API does not support per-molecule custom bounds matrices, so GPU batching is
not available for Phase 2. GPU time is preserved for Butina and MACE scoring.

### Residue classification

`classify_backbone_residues(mol)` returns one label per backbone residue in the same
order as `get_backbone_dihedrals(mol)`. Labels: `"L"`, `"D"`, `"NMe"`, `"Gly"`.
Detection uses atom-graph inspection on the amide N and Cα, not SMARTS on the SMILES
string, so it works on ring-opened representations and on molecules built by RDKit.
Requires `Chem.AssignStereochemistry` (called internally).

Classification is stratified because the four classes occupy distinct Ramachandran
regions: L and D are mirror images of each other; NMe has a restricted phi due to
the N-methyl steric clash with the preceding carbonyl; Gly has a much broader accessible
region with no side-chain constraint.

### Sampling strategies

**`sample_constrained_confs`** is the main entry point. It draws `n_samples` (phi, psi)
targets from the CREMP Ramachandran grids, builds a bounds matrix for each draw, and
attempts to embed a conformer. Ring-closure failures are discarded silently — the caller
receives however many conformers the constraint geometry allowed.

Two strategies are available via the `strategy` argument:

- **`uniform`** — equal weight to every non-zero cell of the CREMP Ramachandran grid.
  Samples from the full CREMP-accessible region without bias. Good for a first pass.
- **`inverse`** — weight proportional to `1/p` for each non-zero cell. Rare-but-accessible
  cells are oversampled relative to the CREMP distribution. This is the designed mode
  for gap-filling: cells with high CREMP probability are already visited by ETKDGv3;
  cells with low probability are the ones worth targeting.

### Data contracts

**`sample_constrained_confs`**

| Argument | Type | Notes |
|----------|------|-------|
| `mol` | `Chem.Mol` | Modified in-place; new conformers are appended. Must have explicit Hs. |
| `grids` | `dict` | From `load_ramachandran_grids()`; keys `L`, `D`, `NMe`, `Gly`, `bin_centers` |
| `n_samples` | `int` | Number of (phi, psi) draws to attempt |
| `n_attempts` | `int` | ETKDGv3 attempts per draw (default 5) |
| `tolerance_deg` | `float` | Dihedral constraint half-width in degrees (default 30°) |
| `strategy` | `str` | `'uniform'` or `'inverse'` |
| `seed` | `int` | NumPy RNG seed |

Returns `List[int]` of conformer IDs added to `mol`.

**`load_ramachandran_grids`**

```python
grids = load_ramachandran_grids("data/processed/cremp/ramachandran_grids.npz")
grids["L"]            # (36, 36) float64, sums to 1.0 per class
grids["bin_centers"]  # (36,) degrees
```

Grids are built by `data/scripts/build_ramachandran_grids.py` (see `data/scripts/README.md`).

### Critical parameters

- **`tolerance_deg`**: controls the half-width of the dihedral constraint window.
  Wider tolerance → higher embedding acceptance rate but less precise targeting.
  Narrower tolerance → more precise but more infeasible samples (ring closure fails).
  Default 30° is a reasonable starting point; macrocycles with many residues may
  need wider tolerance.
- **`n_attempts` per draw**: each draw gets this many ETKDGv3 trials with different
  random seeds. A single draw rarely produces more than one conformer (the bounds
  matrix already tightly constrains the geometry), so `n_attempts=5` is usually
  sufficient.

---

## mcmm.py

> **Macrocyclic peptides only.** This module assumes a head-to-tail cyclic
> peptide backbone for window enumeration. It is not suitable for acyclic
> molecules or non-peptide macrocycles. Imported by `get_mol_PE_mcmm` in
> `confsweeper.py`; the rest of the public pipeline never touches it
> directly.

Multiple Minimum Monte Carlo (Saunders 1990) with replica exchange and
Dodd-Boone-Theodorou (DBT) concerted-rotation moves on backbone windows
(issue #11; design in `docs/mcmm_plan.md`). The premise is that pure
randomisation (exhaustive ETKDG) saturates near one-hot Boltzmann
distributions on `pampa_large`-style peptides regardless of seed budget — a
*connectivity* problem rather than a sampling-density one. MCMM walks
adaptively from a known minimum with shared basin memory and a
high-temperature replica that crosses barriers; the basin set is then fed
into the same MMFF → MACE → 5 kT filter → Kabsch dedup tail that
`get_mol_PE_exhaustive` and `get_mol_PE_pool_b` use.

### Architecture

```
ETKDG seeds (n_init_confs)
   │
   │  MMFF-relax each; MACE-score; build initial BasinMemory
   ▼
ReplicaExchangeMCMMDriver
   ├── n_temperatures geometric ladder (kt_low … kt_high)
   ├── n_walkers_per_temp walkers per rung (total = n_temperatures × n_walkers_per_temp)
   ├── round-robin walker → seed assignment when n_init_confs > 1
   │
   │  Repeat n_steps times:
   │     each walker proposes a move via the composite proposer
   │       ├── DBT concerted-rotation (weight 1 - cartesian_weight)
   │       └── GOAT-style Cartesian kick (weight cartesian_weight)
   │     MMFF-relax + MACE-score the proposal (batched across walkers)
   │     Metropolis accept/reject with kT_walker + Saunders 1/usage^p bias
   │     accepted basins added to shared BasinMemory
   │  Every swap_interval steps: propose adjacent-rank swap, Metropolis on ΔE/Δβ
   │
   ▼
basin set from BasinMemory → _minimize_score_filter_dedup (shared tail)
```

### Core classes and helpers

**`BasinMemory`** — visit-weighted novelty memory shared across all walkers
(Saunders 1990). Stores per-basin heavy-atom coords, MACE energy, visit count,
and (in `dedup_mode='crest'`) the rotational-constant eigenvalues. Methods
`query_novelty(coords, energy=None)` and `query_novelty_batch(...)` return
the closest stored basin under the configured `dedup_mode` and a flag for
whether the candidate is novel; `add_basin(...)` appends a new basin or
increments the visit count of an existing one. The `1/usage^p` Saunders bias
applied during Metropolis acceptance is computed from the visit count.

**`MCMMWalker`** — single-walker driver. Holds `state` (current coords,
energy), `kT`, RNG, references to the proposer and the shared `BasinMemory`.
Per step: ask the proposer for a candidate, query memory for novelty / closest
basin, compute Metropolis acceptance with the Saunders bias, update state and
basin counts. Per-walker acceptance / closure-pass / proposal stats are
exposed for diagnostics.

**`ParallelMCMMDriver` and `ReplicaExchangeMCMMDriver`** — batched walker
execution. `ParallelMCMMDriver` runs all walkers in lockstep (one batched
MMFF + MACE call per step across all walkers via the proposer). The
replica-exchange driver adds a swap-attempt loop every `swap_interval` steps:
adjacent-rank pairs swap configurations with Metropolis on `(βᵢ - βⱼ)(Eᵢ - Eⱼ)`,
preserving detailed balance via `_swap_walker_configs`. `enable_swaps=False`
recovers the independent-T-worker variant for ablations (Step 13).

**`make_mcmm_proposer`, `make_cartesian_kick_proposer`,
`make_composite_proposer`** — proposer factories. The DBT proposer enumerates
backbone windows (8 atoms each = 4-residue window), samples a drive-angle
perturbation, solves the closure problem numerically (Coutsias 2004 reformulation
in `concerted_rotation.py`), couples side chains to the rotated backbone via
rigid-body transport, MMFF-relaxes the result, and batches MACE scoring across
all walkers' proposals in one forward pass. The Cartesian-kick proposer
applies an isotropic Gaussian perturbation (σ = `sigma_kick_a`) to every atom
and MMFF-relaxes — MMFF's bond-stretch / angle-bend gradients pull bonds and
ring sp² angles back to equilibrium for `sigma_kick_a ≤ 0.3 Å` without
explicit SHAKE-style constraints. `make_composite_proposer` routes each
walker per step between the two via configurable weights and reassembles
results in walker order. All three proposers expose `.stats` dicts
(`n_proposed`, `n_relax_failures`, `n_relax_successes`, etc.) for diagnostic
logging.

### Kabsch and inertia helpers

**`_kabsch_rmsd_pairwise(queries, refs)`** — pure-torch SVD with
determinant-correction to reject reflections; row-vector convention; returns
the `(n_queries, n_refs)` RMSD matrix in Å. Operates on already-sliced
heavy-atom coords (slicing is the caller's responsibility).
`_heavy_atom_kabsch_rmsd(a, b)` is the single-pair scalar form.

**`_inertia_eigvals(coords, masses)`** — builds the inertia tensor
`I_ab = Σ_a m_a (||r_a||² δ_ab - r_a r_b)` from centred coords and returns
`torch.linalg.eigvalsh(I)`. Three numbers per conformer, translation- and
rotation-invariant by construction. Consumed by `BasinMemory` in CREST mode
to compute rotational-constant anisotropy for the AND-test third criterion.

**`_max_relative_eig_diff(query_eigs, stored_eigs)`** — the relative-anisotropy
distance metric used inside CREST mode; symmetric difference of each eigenvalue
pair normalised by the larger of the two.

### Sampler benchmark adapter

`scripts/sampler_benchmark.py:_run_mcmm` adapts `get_mol_PE_mcmm` to the
benchmark's `(peptide, n_seeds, hardware_opts, calc, grids)` signature. The
matched-budget mapping is `n_steps = max(1, n_seeds // 64)`, keeping total
MMFF work proportional to `n_seeds` across the three samplers
(`exhaustive_etkdg`, `pool_b`, `mcmm`). The adapter is also where the
production settings from `docs/mcmm_plan.md`'s 2026-05-06 findings are
applied — in-code defaults preserve the original Saunders 1990 / 5 kT
conventions, so the benchmark stays self-documenting.

---

## concerted_rotation.py

> **Macrocyclic peptides only.** Standalone Dodd-Boone-Theodorou (DBT)
> concerted-rotation geometry. Consumed by `mcmm.py`; has no MCMM-specific
> state, so it is reusable for any macrocycle-MC code.

DBT 1993 backbone moves with Coutsias 2004's better-conditioned reformulation
of the ring-closure polynomial. The idea: rotating two adjacent backbone
dihedrals on a window of 4 residues breaks the ring; rotating the remaining
six dihedrals concertedly closes it again, producing a geometrically-valid
new backbone conformation. Coutsias 2004 reformulated the polynomial in terms
of half-angle tangents and showed the resulting expression is numerically
well-behaved up to the closure manifold's boundary, where DBT 1993's original
recipe loses precision.

### Move proposal

**`propose_move(coords, window, drive_sigma_rad, closure_tol)`** is the entry
point. Takes the current atomic coordinates, a backbone window (8 atoms = 4
residues), and the drive perturbation σ; returns a `MoveProposal(coords, det_j,
success)` tuple. `det_j` is the determinant of the Wu-Deem 1999 Jacobian needed
for detailed balance under the Metropolis acceptance criterion — the proposer
returns it so the MCMM walker can compute the correct acceptance ratio.
`success=False` means the closure problem had no solution at the chosen
drive perturbation, in which case the walker keeps its current state and
counts the failure as a "rejected closure" for diagnostic stats.

### Closure tolerance

`DEFAULT_CLOSURE_TOL = 0.01 Å` is the maximum r5 + r6 displacement norm
considered "ring-closed". Relaxing it monotonically improves closure
acceptance and basin coverage up to ~0.1 Å (MMFF's bond-stretch tolerance);
beyond that, MMFF can drift the structure into a different basin than the
move targeted, defeating the algorithmic point. Production runs use 0.05 Å
per the Findings 2026-05-06 tuning.

### Helpers

`dihedral_angle(p0, p1, p2, p3)`, `apply_dihedral_changes(positions, deltas)`,
and `apply_dihedral_changes_full_mol(...)` are general-purpose dihedral
geometry utilities — vectorised against torch tensors so the proposer can
apply moves to a batch of walker states in one call. Used internally by
`propose_move` and exposed for tests.

---

## utils.py

A single function, `compare_geometries`, used by the validation layer to check
whether a confsweeper-generated conformer matches a CREMP reference conformer.

**`compare_geometries(mol_a, mol_b, calc, rmsd_threshold, energy_threshold)`**

Returns `(is_match, rmsd, energy_diff_kcal)`. A match requires both:
- Rigid-body aligned RMSD ≤ `rmsd_threshold` (default 0.125 Å) via `spyrmsd`
- |ΔE| ≤ `energy_threshold` (default 6.0 kcal/mol) from the ASE calculator

The combined RMSD + energy criterion is stricter than geometry-only RMSD: two
conformers can be geometrically close but energetically distinct if one is near
a saddle point. The 6 kcal/mol threshold is approximately the thermal energy
available at room temperature (kT ≈ 0.6 kcal/mol × ~10), chosen to be
permissive enough to avoid penalising genuine local minima that differ only due
to force-field vs. ML-potential energy offsets.
