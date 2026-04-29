# src/

Core confsweeper library. Three modules handle distinct responsibilities:
`confsweeper.py` owns the full generation-to-scoring pipeline and is general-purpose —
it works for any molecule type. `torsional_sampling.py` provides the backbone
dihedral-constrained Phase 2 conformer pool and is **macrocycle-specific**: it
assumes a head-to-tail cyclic peptide backbone and should not be used for acyclic
molecules. `utils.py` provides geometry comparison utilities used by the validation
layer.

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

> **Refactor flag**: the post-sampling tail is duplicated between
> `get_mol_PE_exhaustive` and `get_mol_PE_pool_b`. Once two more samplers
> (CREST-fast, MCMM, REMD; see issue #10) land, refactor the shared MMFF /
> MACE / energy-filter / dedup / prune block into a private
> `_minimize_score_filter_dedup(mol, calc, hardware_opts, ...)` helper. The
> duplication is intentional in this PR to avoid touching the production-
> default exhaustive path inside a benchmarking branch.

**`get_mol_PE_mmff`** — like `get_mol_PE_batched` but scores with MMFF94. No GPU
required for the scoring step; GPU is still used for embedding and Butina.

### Clustering

Butina clustering runs on the pairwise L1 distance matrix between flattened conformer
coordinate tensors, normalised by `3 * n_atoms` so the cutoff is in Å-per-atom units
rather than raw L1 distance. The GPU Butina path (`nvmolkit.clustering.butina`) and the
CPU fallback (`rdkit.ML.Cluster.Butina.ClusterData`) both return 0-based *row indices*
into the distance matrix, not RDKit conformer IDs.

In the single-pool (ETKDG-only) case these happen to be the same, because nvmolkit
assigns conformer IDs starting at 0 sequentially. In the two-pool case, Pool B
conformers receive IDs starting from the Pool A count, so row index ≠ conformer ID.
`get_mol_PE_batched` handles this by collecting `all_conf_ids = [c.GetId() for c in
mol.GetConformers()]` before building the distance matrix and mapping centroid row
indices back through that list to recover actual conformer IDs.

The normalisation `/ (3 * n_atoms)` means `cutoff_dist=0.1` corresponds to a mean
coordinate deviation of 0.1 Å per atom, not a 0.1 Å RMSD. This is a coarser criterion
than symmetric RMSD but is much cheaper to compute across thousands of conformers.

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
| `n_attempts` | `int` | ETKDGv3 attempts per draw (default 1, see refactor flag) |
| `tolerance_deg` | `float` | dihedral constraint half-width (default 30°) |
| `strategy` | `str` | `'inverse'` (default) or `'uniform'` |
| `score_chunk_size`, `e_window_kT`, `rmsd_threshold`, `minimize`, `mmff_backend`, `seed` | — | as in `get_mol_PE_exhaustive` |

**`get_mol_PE_exhaustive`** — different parameter space (no Butina cutoff, no
torsional sampling, no `gpu_clustering` toggle):

| Argument | Type | Notes |
|----------|------|-------|
| `smi`, `params`, `hardware_opts`, `calc` | — | as above |
| `n_seeds` | `int` | total ETKDG seeds (default 10000, saturation-validated) |
| `embed_chunk_size` | `int` | nvmolkit per-call cap before chunking (default 1000) |
| `score_chunk_size` | `int` | MACE per-batch forward-pass cap (default 500) |
| `e_window_kT` | `float` | energy filter window in `kT_298K` units (default 5.0) |
| `rmsd_threshold` | `float` | basin-dedup exclusion radius in normalised L1 units (default 0.1) |
| `minimize` | `bool` | MMFF94 minimise post-embed (default `True`, the lever) |
| `mmff_backend` | `str` | `'gpu'` (default; nvmolkit batched CUDA) or `'cpu'` (RDKit serial) |
| `dihedral_jitter_deg` | `float` | rotatable-bond jitter ±deg (default 0; experimental) |
| `seed` | `int` | base ETKDG seed; chunk i uses `seed + i*embed_chunk_size` (default 0) |

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
