# dihedral_predictor

## Purpose

Predict a macrocyclic peptide's **dominant-conformer dihedrals** (backbone φ/ψ/ω and
side-chain χ) from per-residue topology, and seed conformer generation from the
prediction — so the sampler can reach the CREST-dominant basin it otherwise misses.

This module targets the MMFF↔CREST energy-rank **inversion** (issue #19): for many
cyclic peptides the CREST-dominant conformer is MMFF-disfavoured, so the MMFF/MC
explorer never visits it and MACE (a re-scorer) cannot recover an unsampled basin.
The project goal is to reproduce the **CREST conformer distribution** cheaply; unlike
per-peptide reference-seeding, a trained model **generalizes** — it needs only the
molecule's topology at inference.

**Backbone and side-chain χ are two SEPARATE models** (`DihedralPredictor` and
`ChiPredictor`), trained independently with no shared weights, so adding χ cannot
regress backbone fidelity. At seeding, the backbone model drives constrained
distance-geometry (φ/ψ/ω) and the χ model sets side chains via `SetDihedralDeg`.

### Status (2026-06-30)

The end-to-end mechanism is **validated**: injecting the true CREST-dominant geometry as
a seed reproduces the dominant basin (`cov_bw_ceil` ≈ the dominant's CREST weight). The
current **learned** prediction, however, is not yet accurate enough to land the strict
0.5 Å all-atom basin match on the hardest peptides — binned-dihedral error compounds
across the molecule. A key reframing: for cremp_sharp the inversion is **side-chain-rotamer
driven** (the backbone fold is already reachable de-novo), so χ precision is the active
lever there. See `docs/dihedral_predictor_plan.md` for the full findings ledger.

## Module contents

- **`residues.py`** — feature and dihedral extraction, all derived from the RDKit mol in
  **ring order** (`ordered_backbone_dihedrals` chains residues by their amide
  connectivity; `get_backbone_dihedrals` alone is NOT ring-sequential for many
  macrocycles, which would corrupt ω and the cyclic neighbour features). `residue_features`
  builds per-residue physico-chemical descriptors (NMe, Gly, D/L, proline, side-chain
  heavy count / aromaticity / H-bond donors+acceptors); `neighbor_augment` concatenates
  cyclic ±window neighbour descriptors. `omega_quads` builds ω from consecutive residues;
  `sidechain_chi_quads`/`sidechain_chi_values` enumerate side-chain χ (χ1…χ`MAX_CHI`),
  walking from Cβ outward and emitting a χ only when its rotated bond is non-ring, with
  **canonical-rank** branch tie-breaking so a χ slot is the same chemical dihedral on the
  CREMP extraction mol and the smi seed mol. Binning helpers discretise the circular
  targets (φ/ψ/χ into `PHI_PSI_BINS`, ω into cis/trans).

- **`data.py`** — `extract_record` builds one record per peptide (base features + the
  **dominant** conformer's binned φ/ψ/ω **and** χ targets + a χ presence mask).
  `build_dataset` runs this over CREMP and saves a peptide-split pickle (no sequence
  leaks). `DihedralDataset`/`collate` apply per-peptide cyclic augmentation and pad per
  batch; both backbone and χ targets are carried so one dataset serves both models.

- **`model.py`** — `DihedralPredictor` (backbone: φ/ψ/ω heads) and `ChiPredictor`
  (separate: per-residue χ1…χ`MAX_CHI` heads). Both: a small transformer encoder over
  residues, **ring size injected** as a global feature (ω cis/trans is strongly ring-size
  dependent), and **no absolute positional encoding** (a macrocycle has no canonical
  start; local order from neighbour augmentation, longer range from attention).

- **`train.py`** — `train` (backbone; checkpoints best by `peptide_all_ok`) and `train_chi`
  (separate χ model; checkpoints best by `chi_peptide_ok`). Masked cross-entropy; reports
  within-1-bin accuracy and the per-peptide all-correct seeding proxies.

- **`seed.py`** — `load_model`/`load_chi_model`, `predict_dihedrals`/`predict_chi`, and
  `seed_conformers`: predict backbone → constrained-DG embed (φ/ψ **and** ω, the base
  `make_constrained_bounds` does only φ/ψ) → optionally set predicted χ on each seed with
  `apply_chi` (`SetDihedralDeg`; χ bonds are not ring-closure-constrained). Constraint
  tolerance defaults to **±60°** (tighter thrashes the from-scratch DG embed: ~90 s vs
  ~2 s at 30° vs 60°; ±60° is also the right precision since rotamers are 120° apart).

Runnable entry points (in `scripts/`): `build_dihedral_dataset.py`,
`train_dihedral_predictor.py`, `train_chi_predictor.py`, `sweep_dihedral_model.py`
(capacity/window sweep), `resplit_topology.py` (composition split for generalisation),
`validate_seeding.py` (RMSD proxy), `validate_seeding_coverage.py` (real-MCMM CREST
coverage, with `--oracle`/`--no_relax_seed`/`--chi_ckpt`/`--mace_relax_seed`/`--backbone`),
and `aggregate_seeding_coverage.py` (coverage lift over an inverted test set).

## Data contracts

- **CREMP pickle** (extraction input): `{"rd_mol": Chem.Mol with N conformers (explicit
  Hs), "conformers": [{"boltzmannweight": float}, … N], "smiles": str}`.
- **Dataset pickle**: dict with `seqs` (list[str]), `feats` (list of `(n_res, 8)` float32),
  `phi_bin`/`psi_bin`/`omega_bin` (lists of `(n_res,)` int64), `chi_bin` (list of
  `(n_res, MAX_CHI)` int64), `chi_mask` (list of `(n_res, MAX_CHI)` bool), `split` (object
  array), `phi_psi_bins`, `max_chi`.
- **Backbone model I/O**: input `x` `(B, L, F)` (F = `8*(2*window+1)`), `mask` `(B, L)`;
  output logits `(B,L,24)`, `(B,L,24)`, `(B,L,2)`.
- **Chi model I/O**: same input; output logits `(B, L, MAX_CHI, 24)`.
- **`seed_conformers`**: takes a mol with explicit Hs (built as
  `Chem.AddHs(Chem.MolFromSmiles(smi))`), adds conformers in place, returns their IDs
  (empty if the predicted backbone is geometrically infeasible — a free DG rejection).

## Critical parameters and constraints

- **`window`** (backbone) and the χ model's window must each match between training and
  inference; both are stored in the checkpoint and returned by the loaders.
- **`PHI_PSI_BINS = 24`** (15°), **`OMEGA_CIS_CUTOFF = 90°`**, **`MAX_CHI = 4`** — changing
  any invalidates existing checkpoints/datasets.
- **Ring-order + canonical-χ invariant**: features and targets are produced in
  `ordered_backbone_dihedrals` order, and χ slots are canonical-rank-consistent so a model
  trained on CREMP mols applies correctly to smi-built seed mols. Preserve both.
- **Seeding tolerance ±60°** (not the Pool-B 30°) — tighter makes constrained DG thrash.
- **`get_mol_PE_mcmm(extra_seed_coords=, relax_seeds=)`** (in `confsweeper.py`) is the
  injection point. `relax_seeds=False` keeps seeds at their predicted geometry (MACE-scored,
  appended as protected basins) — required for inverted peptides, where MMFF relaxation
  pushes a good seed off the MMFF-disfavoured dominant.
- The split is by peptide; do not reshuffle a trained checkpoint's dataset or test metrics
  leak. `resplit_topology.py` provides a stronger composition (permutation-aware) split.

## Dependencies on other modules

- Consumes `torsional_sampling` (`ordered`/`get_backbone_dihedrals` ordering,
  `set_dihedral_bounds`, `embed_constrained`) and the CREMP raw pickles under
  `data/raw/cremp/`.
- Produces seed conformers for the confsweeper sampler: injected via
  `confsweeper.get_mol_PE_mcmm(extra_seed_coords=…)`. Coverage is scored against the
  raw-CREST, CREST-Boltzmann-weighted ceiling (single-provenance Kabsch).
- Requires `torch` (models) and `rdkit` (features/embedding); seeding/coverage use the
  MACE calculator from `confsweeper`.
```
