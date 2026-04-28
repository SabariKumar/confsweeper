# scripts/

Diagnostic and saturation-experiment scripts that drive `confsweeper`'s
public pipeline functions on real data. Distinct from `src/` (library code)
and `tests/` (correctness tests): these are research tools meant to be run
from the command line on a GPU node, and their outputs feed design
decisions rather than CI.

## Why this directory exists

`/data/` is `.gitignore`d in this repo, which is the right policy for raw
datasets but the wrong place for analysis code we want to track. Similarly,
`/src/` is reserved for the importable library — runnable diagnostic
scripts that consume the library don't belong there. `scripts/` fills the
gap.

## Module contents

### `saturation_etkdg.py`

Sweeps `n_seeds` over a fixed set of representative cyclic peptides for
`get_mol_PE_exhaustive` and writes per-run Boltzmann-weight statistics to a
CSV. The Phase 2 diagnostic from `docs/exhaustive_etkdg_plan.md`: its job
is to tell us where the saturation curves of `max_bw` and `n_within_3kT`
flatten so we can pick a defensible production default for `n_seeds`.

`--params_mode` selects the ETKDG variant. `etkdgv3_macrocycle` (default) is
the production setting from `get_embed_params_macrocycle`. `etkdg_original`
is RDKit's 2015 ETKDG with no macrocycle-specific torsion knowledge; it
exists to test the hypothesis that ETKDGv3's macrocycle bias suppresses
low-energy basin diversity in the post-MACE-rescore step.

`--dihedral_jitter_deg` (default 0) applies a uniform random rotation in
`[-jitter, +jitter]` degrees to every rotatable-bond dihedral on every
embedded conformer before MACE scoring. Used to push side-chain rotamer
exploration past ETKDG's torsion-knowledge bias. Macrocycle backbone
bonds are excluded by the rotatable-bond SMARTS, so on cyclic peptides
this jitters side chains only.

All three knobs (mode, n_seeds, jitter) can write into the same output
CSV — rows are keyed by `(peptide_id, params_mode, n_seeds, minimize,
dihedral_jitter_deg)`.

Selects 5 peptides (2 CREMP — one near median `poplowestpct`, one in the
high tail; 3 PAMPA — small, medium, large heavy-atom buckets) so the run
covers the full chemical-size range and gives us CREST ground truth on the
two CREMP picks for sanity-checking. CREMP `poplowestpct/100` and
`uniqueconfs` are passed through to the output CSV in the
`ground_truth_*` columns.

Writes results one row at a time and is resume-safe: rerunning with the
same `--out_csv` skips `(peptide_id, n_seeds, minimize)` tuples that are
already in the file.

### Output CSV columns

See the `OUTPUT_COLUMNS` constant in `saturation_etkdg.py` for the full
list. The diagnostic columns are `n_basins`, `max_bw`, `eff_n`, `entropy`,
`n_within_kT`, `n_within_3kT` per run, plus per-stage timings.
`ground_truth_max_bw` and `ground_truth_n_confs` are populated only for
CREMP rows (NaN for PAMPA).

### `minimize_ablation.py`

Controlled ablation: same nvmolkit ETKDG starting pool, two scoring
pipelines side-by-side. Pipeline A is MACE → energy filter → energy-ranked
dedup → BW. Pipeline B is GPU MMFF → MACE → same filter → same dedup → BW.
Removes the confound from the saturation sweep, where the no-minimize and
minimize=True runs used different ETKDG seeds and therefore different
starting pools.

Defaults to `pampa_large` (the most extreme regression case from the
saturation results). Pass `--peptides` comma-separated to run on multiple
peptides in one invocation. Sources SMILES + heavy-atom counts from
`saturation_etkdg.csv` so the comparison is on the exact same molecules
the sweep evaluated.

### `mace_vs_xtb.py`

Side-by-side energy benchmark on a single peptide: embeds one conformer
pool, scores it with MACE-OFF23 (GPU, batched) and xtb GFN2 (CPU,
parallel subprocess pool), then reports Boltzmann-weight statistics
under each backend plus Pearson and Spearman correlations between the
two energy vectors.

The point is to rule MACE in or out as the cause of the near-one-hot
Boltzmann distributions on cyclic peptides. If both backends produce
similar BW vectors on the same conformer set, sampling is the bottleneck
and the energy model is fine. If they disagree, the production pipeline's
backend choice matters and we'd need to rescore (or replace MACE).

Self-contained: takes a SMILES, runs the pool through both backends, and
prints results. Does not write into the saturation CSV. Optional
per-conformer CSV (`--out_csv`) preserves both energies for follow-up.

## Critical parameters or constraints

- The script imports from `src/confsweeper.py` directly via
  `sys.path.insert`, so it must be run with the confsweeper pixi
  environment active. `pixi run python scripts/saturation_etkdg.py …`.
- nvmolkit and MACE both run on GPU 0 by default. There is no multi-GPU
  scheduler — long runs block on a single device.
- Default `n_seeds_grid` (`100,500,1000,5000,10000,50000`) is the grid
  recommended by the design plan. Smaller grids are fine for quick
  validation; the published default is meant to be replaced by whatever
  the saturation curve actually requires.

## Dependencies on other modules

- Reads from `data/processed/cremp/validation_subset.csv` (in this repo,
  but gitignored — generated upstream from CREMP raw data).
- Reads from `peptide_electrostatics:data/fine_tune/CycPeptMPDB_PAMPA_deduped.csv`
  (in the sibling repo).
- Calls `confsweeper.get_mol_PE_exhaustive`,
  `get_embed_params_macrocycle`, `get_hardware_opts`, `get_mace_calc`,
  and the private constant `_KT_EV_298K`.
