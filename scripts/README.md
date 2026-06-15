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

### `sampler_benchmark.py`

Head-to-head benchmark of structurally distinct conformer samplers
against the same MACE-OFF scoring path on the same five peptides used by
`saturation_etkdg.py`. Companion script for issue #10: `saturation_etkdg.py`
sweeps the *budget* knob for one sampler; `sampler_benchmark.py` sweeps the
*sampler* knob at a fixed budget. The motivating question is which (if
any) sampler clears the `pampa_large`-style ceiling that randomized
ETKDG + MMFF cannot push through, regardless of `n_seeds`.

Sampler dispatch table — keyed by name, each entry is an adapter that
takes `(peptide, n_seeds, hardware_opts, calc, grids)` and returns a
list of MACE energies for the basin centroids. Currently:

- `exhaustive_etkdg` — `get_mol_PE_exhaustive` at saturation-validated
  defaults (the production baseline).
- `pool_b` — `get_mol_PE_pool_b` with `strategy='inverse'` and
  `n_attempts=1` (matched-budget default).
- `mcmm` — `get_mol_PE_mcmm` (MCMM-REMD with DBT + Cartesian + side-chain
  dihedral composite proposer, basin memory, replica exchange). The
  adapter maps the benchmark's `n_seeds` knob onto MCMM step count via
  `n_steps = max(1, n_seeds // 64)`, keeping total MMFF work
  proportional across the three samplers at the same `--n_seeds`. The
  adapter also locks the production tuning from
  `docs/mcmm_plan.md`'s 2026-05-06 findings
  (`drive_sigma_rad=0.3, closure_tol=0.05, kt_high=8 × kT_298,
  n_init_confs=8, cartesian_weight=0.5, e_window_kT=10,
  saunders_exponent=1.0`); the in-code function defaults preserve the
  original Saunders 1990 / 5 kT conventions.

**MCMM proposer-mix CLI flags (issue #12).** Three knobs exposed at the
CLI for the dihedral-kick proposer; each defaults to "off / locked" so
existing benchmark commands keep their issue-#10 behaviour:

- `--cartesian_weight FLOAT` (default `0.0`) — routing weight for the
  GOAT-style Cartesian kick alongside DBT. Step 12 of
  `docs/mcmm_plan.md`.
- `--dihedral_weight FLOAT` (default `0.0`) — routing weight for the
  side-chain dihedral kick. DBT residual weight =
  `1 − cartesian_weight − dihedral_weight`; the sum must be `≤ 1`.
  Issue #12 / `docs/dihedral_kick_plan.md`.
- `--p_rotamer_jump FLOAT` (default `0.3`) — probability per walker per
  step of the dihedral kick taking a discrete rotamer-jump
  (sampled uniformly from `rotamer_wells_deg`) instead of a Gaussian
  Δχ. Exposed for the Step-7 snap-back diagnostic; ignored when
  `--dihedral_weight=0`.

**Production tuning lock (issue #12 closes, 2026-06-15).** The 4-cell
phase-1 sweep at n_seeds=5000 + 2-run phase-2 sweep at n_seeds=10000
(driver scripts `scripts/sweep_step7.sh`, `scripts/sweep_step7_phase2.sh`)
identified `--cartesian_weight=0.33 --dihedral_weight=0.33
--p_rotamer_jump=0.30` as the production mix. On `cremp_typical`
(`t.I.G.N`) this lifts Boltzmann coverage from `0.83` (DBT + Cart at the
same budget) to **`0.991`** — 20/22 ceiling basins covered, `max_missed_bw
= 0.006`. Raising `--p_rotamer_jump` to `0.70` *regresses* cremp_typical
to `0.971` (too few Gaussian refinement steps), so the lock is precise.
On `cremp_sharp` (`S.S.N.MeW.MeA.MeN`) the mix does NOT recover the
dominant ceiling basin at any tested setting; the residual is documented
as a v0.2 follow-up (issue #13: aromatic-aware rotamer wells, no-MMFF
ablation). See `docs/dihedral_kick_plan.md` Step-7 Findings for the
full empirical record.

Future entries (CREST-fast, independent-T MCMM) plug in as a new adapter
function plus a single dispatch-table key. The benchmark protocol stays
unchanged.

The benchmark also dumps per-peptide basin SDFs to a configurable output
directory when run with `--dump_basin_sdfs`. Each SDF carries the
basin-representative conformers tagged with `MACE_ENERGY`; these are the
inputs that `union_basin_count.py` consumes for post-hoc cross-method
overlap analysis.

Output CSV is keyed by `(peptide_id, sampler, n_seeds)` and is resume-aware.
Failed runs are logged and skipped — the loop moves on to the next cell.

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

### `analyze_basin_sdf.py`

Quick diagnostic that takes a dumped basin SDF (from
`sampler_benchmark.py --dump_basin_sdfs`) and reports the pairwise
heavy-atom Kabsch RMSD matrix plus the per-conformer MACE-energy spread.
The tool that surfaced the metric-noise pathology of the pre-Step-11
normalised-L1 dedup (`cremp_sharp` 4 "basins" all within 0.21 Å of each
other; `pampa_small` 16 basins with median pairwise 0.25 Å). Used to
discriminate between the "deep-basin trap" hypothesis and the
"sub-Å wobble passing as distinct basins" hypothesis — see Findings
2026-05-05 in `docs/mcmm_plan.md`.

Self-contained, no CSV input, no GPU required. CLI is a single
positional argument (the SDF path). Prints to stdout.

### `union_basin_count.py`

Post-hoc union analysis across two `sampler_benchmark.py` runs (DBT-only
vs DBT + Cartesian-kick). Loads the dumped basin SDFs from each run,
concatenates the conformers onto a single template mol per peptide, and
reports four metrics (Step 18):

- **Discovery diversity** (`n_union_all`): all union conformers deduped
  at the configured Kabsch threshold with no energy filter. Independent
  of where the global `e_min` lands — the right "how many distinct
  basins did either method find?" number for the paper.
- **Filtered union** (`n_union_filtered_5kT`): standard 5 kT filter
  relative to union `e_min`, then dedup. Comparable to single-method
  `n_basins`.
- **Per-method split**: `n_dbt_only`, `n_cart_only`, `n_overlap` —
  answers "what did each proposer contribute that the other missed?"
- **Coverage % vs the CREMP-rescored ceiling**: union count divided by
  `post_mmff_kabsch_0125` from `cremp_collapse_test.py` when a
  matching peptide exists; the legacy count-ratio metric kept for
  backward continuity.
- **Boltzmann-weighted coverage** (when `--ceiling_sdf_dir` is given):
  reports `coverage_bw_ceiling` (fraction of the CREMP ceiling's 298 K
  Boltzmann population the sampler recovers), `coverage_count_matched`
  (a *true* matched fraction via spyrmsd symmetric RMSD, replacing the
  count ratio that could exceed 1), joint-reference masses, and a
  **new-basins / discovery** metric (`n_new_basins`,
  `new_basin_mass_joint`, `delta_emin_vs_ceiling`,
  `found_new_global_min`) capturing basins the sampler found that CREMP
  missed plus their thermodynamic weight. **Two-threshold design:**
  basin sets are deduped at the within-method `--rmsd_threshold`
  (0.125 Å, the CREMP / CREST / GOAT convention), but cross-method
  ceiling↔sampler matching uses `--match_rmsd` (default **0.5 Å**, the
  `validation/cremp_coverage.py` convention) because MMFF and
  GFN2-xTB relax the same basin to geometries 0.3–0.5 Å apart. Inputs
  come from `cremp_collapse_test.py run --dump_ceiling_sdf_dir`. See
  the 2026-05-21 Findings entry in `docs/mcmm_plan.md` for the
  rotation-naive pre-filter and threshold rationale.

Energies come from the SDF's `MACE_ENERGY` per-conformer property — no
GPU needed. Outputs one row per peptide to the configured `--out_csv`.

### `cremp_collapse_test.py`

Two-subcommand CLI (`run` and `summarize`) for the CREMP basin-collapse
benchmark — diagnostic for Step 16 on the original two-peptide sanity
check, scaled to ~1500 peptides for Step 19's at-scale variance study.

**`run`** — per-peptide, feeds CREMP's GFN2-xTB-relaxed conformer set
through five pipeline stages and reports the conformer count at each:

1. Pre-MMFF Kabsch dedup at 0.125 Å (xtb energies for the ranking).
2. Pre-MMFF Kabsch dedup at 0.5 Å (looser threshold for sanity).
3. Pre-MMFF CREST three-criteria dedup at 0.125 Å + 0.05 kcal/mol +
   1 % rotational-constant anisotropy.
4. MMFF-relax in-place (nvmolkit batched), then MACE-score every
   relaxed conformer.
5. Post-MMFF: 5 kT filter relative to MACE `e_min`, then Kabsch dedup
   at 0.125 / 0.5 Å and CREST dedup at 0.125 Å.

CLI consumes a CSV with a `sequence` column via `--peptide_list_csv`
(the output of `sample_cremp_peptides.py` is the canonical input).
Feature columns (`topology`, `has_proline`, `has_glycine`,
`num_monomers`) on the input CSV are passed through to the output CSV
so the summarize step can stratify without re-deriving.
**Resume-aware** — at start, reads the existing `--out_csv` and skips
sequences already present. Per-peptide try/except so a single bad
pickle doesn't abort the run; per-row append + flush guarantees no
in-memory loss on interrupt.

Optional `--dump_ceiling_sdf_dir PATH` writes the post-MMFF Kabsch
ceiling basins (geometries + per-conformer `MACE_ENERGY`) to
`<dir>/<sequence>.sdf` for each peptide. These are the canonical
"ceiling" basins consumed by `union_basin_count.py`'s Boltzmann-weighted
coverage analysis.

**`summarize`** — reads a completed collapse-test CSV and emits a
plot-ready per-stratum summary: median + IQR of four collapse ratios
(pre-MMFF kabsch, pre-MMFF crest, post-MMFF kabsch, post-MMFF crest)
across the 16 `(topology, has_proline, has_glycine)` cells, the
`num_monomers` marginal, and the sample aggregate. Plus the fraction
of peptides with post-MMFF Kabsch `n_basins` < `uniqueconfs / 10` per
stratum — the "≥ 10× collapse" fraction that headlines the Step-19
paper claim.

### `sample_cremp_peptides.py`

Stratified sampler over the full CREMP `summary.csv` (36k peptides)
for Step 19's at-scale collapse benchmark. The stratification grid is
**4 topology classes × 2 has-Proline × 2 has-Glycine = 16 cells**;
the default target is 100 peptides per cell with a 30-peptide floor
(cells below the floor are dropped with a logged warning).

Topology comes from `validation.make_validation_sets_cremp.parse_topology`,
the same parser the validation-subset script uses, so the topology
labels agree across both subset definitions. The Pro / Gly axes are
binary presence flags — derived from each sequence's residue tokens
after stripping any `Me` prefix and case-folding. The output CSV
mirrors `validation_subset.csv`'s schema plus the new feature columns
`topology`, `has_proline`, `has_glycine`, and `cell`.

Deterministic via `--seed` (default 42). One-shot script — once the
sample is generated for a given seed, downstream consumers
(`cremp_collapse_test.py`, `cremp_overlap_figure.py`) work on the same
file.

### `cremp_overlap_figure.py`

3-panel paper figure for Step 19's at-scale CREMP overlap statistics.
Reads a completed `cremp_collapse_test.py run` output (with the
feature columns the sampler emits) and renders:

- **Panel A** — pre-MMFF Kabsch collapse ratio boxplot grouped by
  topology. Tests whether NMe / D-amino-acid topologies have more
  CREST AND-test inflation than canonical L peptides.
- **Panel B** — post-MMFF Kabsch collapse ratio boxplot grouped by
  Pro / Gly bucket (`neither` / `Pro-only` / `Gly-only` / `both`).
  Tests whether the MMFF-vs-xtb basin disagreement concentrates in
  conformationally-special residue classes.
- **Panel C** — heatmap of post-MMFF Kabsch median collapse ratio
  across the full 16-cell grid, with cell counts overlaid. Direct
  visual of whether the inflation is uniform across the
  `(topology × Pro × Gly)` grid.

SVG is the default and required output (vector, editable in Illustrator
/ Inkscape, lossless for the paper). PDF and PNG are optional via
`--out_pdf` and `--out_png`. One function per panel (`_panel_a / _b
/ _c`) so any panel can be re-rendered or extracted in isolation; layout
tweaks live in `main`.

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
  `confsweeper.get_mol_PE_pool_b` (`sampler_benchmark.py` only),
  `get_embed_params_macrocycle`, `get_hardware_opts`, `get_mace_calc`,
  and the private constant `_KT_EV_298K`.
- `sampler_benchmark.py` reuses `select_cremp_peptides`,
  `select_pampa_peptides`, and `_bw_metrics` from `saturation_etkdg.py`
  so both benchmarks operate on the same peptide library and produce
  directly-comparable rows.
- `pool_b` requires `data/processed/cremp/ramachandran_grids.npz` (the
  CREMP-derived backbone Ramachandran prior).
