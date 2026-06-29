# dihedral_predictor

## Purpose

Predict a macrocyclic peptide's **dominant-conformer backbone dihedrals** from
per-residue topology, and seed constrained distance-geometry from the prediction.

This module is the generalizable fix for the MMFF↔CREST energy-rank **inversion**
characterized in issue #19: for a large fraction of cyclic peptides the
CREST-dominant conformer is MMFF-disfavoured, so the MMFF/MC explorer never visits
it and MACE (a re-scorer) cannot recover a basin that was never sampled. Issue #19
showed (a) the inversion is pervasive and not predictable from cheap features, and
(b) crucially, the inaccessible feature is **sparse** — the CREST-dominant differs
from the easily-found MMFF-best conformer by only ~2 backbone dihedrals (median).
So predicting the dominant fold is tractable, and unlike reference-seeding (which
needs a per-peptide CREST ensemble) a trained model **generalizes** to novel
peptides: it needs only the molecular topology at inference.

The predicted dihedrals are turned into seed conformers via the constrained-DG
machinery in [`torsional_sampling.py`](../torsional_sampling.py); those seeds are
added to the sampler pool so the MC walk can start from and retain the dominant
basin.

## Module contents

- **`residues.py`** — feature and target extraction. Everything is derived from
  the RDKit mol in `get_backbone_dihedrals` order, so per-residue inputs and
  (phi, psi, omega) targets are aligned by construction (no sequence-string ↔
  atom-index matching). `residue_features` builds per-residue physico-chemical
  descriptors (N-methylation, glycine, D/L chirality, proline, side-chain heavy
  count / aromaticity / H-bond donors+acceptors); these are mol-derived so they
  generalize to non-standard residues. `neighbor_augment` concatenates cyclic
  ±window neighbour descriptors (neighbours strongly drive backbone propensities).
  `omega_quads` builds omega atom-quads from consecutive residues (aligned by
  construction); `backbone_dihedral_values` reads phi/psi/omega for a conformer.
  Binning helpers (`angle_to_bin`/`bin_to_center`, `omega_to_bin`/
  `omega_bin_to_center`) discretise the circular targets for classification.

- **`data.py`** — `extract_record` builds one record per peptide (base features +
  the **dominant**, i.e. max-Boltzmann-weight, conformer's binned dihedrals).
  `build_dataset` runs this over CREMP and saves a peptide-split pickle (split is
  by peptide so no sequence leaks). `DihedralDataset` applies cyclic neighbour
  augmentation per peptide on its true length (padding never leaks across the ring
  closure); `collate` pads per batch with a mask.

- **`model.py`** — `DihedralPredictor`, a small transformer encoder over residues
  with three classification heads (phi, psi, omega). Two deliberate choices:
  **ring size is injected** as a global feature (omega cis/trans is strongly
  ring-size dependent — strained tetrapeptides favour cis), and there is **no
  absolute positional encoding** (a head-to-tail macrocycle has no canonical start;
  local order comes from neighbour augmentation, longer range from attention).

- **`train.py`** — masked cross-entropy training; reports per-residue exact and
  within-1-bin accuracy plus the seeding proxy `peptide_all_ok` (fraction of
  peptides whose every backbone dihedral is within tolerance), against a majority
  baseline. Checkpoints the best model by `peptide_all_ok`.

- **`seed.py`** — `load_model`, `predict_dihedrals` (argmax bin → angle), and
  `seed_conformers` (predict → constrained-DG embed). Bounds constrain phi, psi
  **and** omega (the base `make_constrained_bounds` does only phi/psi).

Runnable entry points: `scripts/build_dihedral_dataset.py`,
`scripts/train_dihedral_predictor.py`.

## Data contracts

- **CREMP pickle** (input to extraction): `{"rd_mol": Chem.Mol with N conformers
  (explicit Hs), "conformers": [{"boltzmannweight": float}, ... len N]}`.
- **Dataset pickle** (`build_dataset` output): dict with `seqs` (list[str]),
  `feats` (list of `(n_res, 8)` float32), `phi_bin`/`psi_bin`/`omega_bin` (lists of
  `(n_res,)` int64), `split` (object array of 'train'/'val'/'test'), `phi_psi_bins`.
- **Model I/O**: input `x` `(B, L, F)` float (F = `8 * (2*window+1)`), `mask`
  `(B, L)` bool; output three logit tensors `(B, L, 24)`, `(B, L, 24)`, `(B, L, 2)`.
- **`seed_conformers`**: takes an RDKit mol with explicit Hs, adds conformers in
  place, returns their conformer IDs (empty if the predicted dihedral set is
  geometrically infeasible — a free ring-closure rejection).

## Critical parameters and constraints

- **`window`** must match between training and inference (it sets the input feature
  dim). It is stored in the checkpoint and returned by `load_model`.
- **`PHI_PSI_BINS = 24`** (15° bins): a bin is ~the constrained-DG tolerance, so a
  within-1-bin prediction lands inside the ±30° seeding window. Changing this
  invalidates existing checkpoints and datasets.
- **`OMEGA_CIS_CUTOFF = 90°`** defines the cis/trans split for the omega target.
- **Alignment invariant**: features and targets are both produced in
  `get_backbone_dihedrals` order. Any new feature/target must preserve this order.
- The split is by peptide; do not reshuffle a trained checkpoint's dataset or test
  metrics leak.

## Dependencies on other modules

- Consumes `torsional_sampling.get_backbone_dihedrals` (residue ordering),
  `set_dihedral_bounds`, and `embed_constrained` (constrained-DG seeding).
- Consumes the CREMP raw pickles under `data/raw/cremp/`.
- Produces seed conformers for the confsweeper sampler pool (Pool B augmentation):
  the downstream consumer is the MMFF/MC + MACE pipeline in `confsweeper.py`.
- Requires `torch` (model/training) and `rdkit` (features/embedding).
```
