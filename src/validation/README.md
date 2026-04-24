# src/validation/

Benchmarking utilities that measure how well confsweeper's conformer pools cover
reference datasets. The primary benchmark compares against CREMP (a GFN2-xTB-level
macrocyclic peptide conformer database); a secondary benchmark covers GEOM-Drugs
(a broader drug-like molecule dataset). There are also helper modules for test
peptide generation and topological barcode analysis.

---

## cremp.py

Data loader and symmetric RMSD utilities for the CREMP benchmark.

**`iter_validation_mols(subset_csv, pickle_dir)`** — yields `(smiles, rd_mol)` pairs
for each molecule in the validation subset. Each CREMP pickle contains an RDKit mol
with all unique GFN2-xTB-optimised conformers plus energy and Boltzmann metadata.
Hs are stored as explicit atoms; callers should not call `Chem.AddHs` on the loaded mol.

The symmetric RMSD utilities in this module use `spyrmsd` with graph-automorphism-aware
alignment. Symmetric RMSD is required for macrocyclic peptides because the SMARTS
backbone traversal order is not guaranteed to match the atom order of the CREMP pickle,
and standard RMSD without symmetry handling gives artificially large values when the
two atom orderings are equivalent by ring automorphism.

---

## cremp_coverage.py

The main confsweeper benchmark CLI. For each molecule in the validation subset, it:

1. Generates conformers with confsweeper (either ETKDG-only or ETKDG + torsional)
2. For each CREMP reference conformer, checks whether any confsweeper conformer
   is within `rmsd_cutoff` (default 0.5 Å symmetric RMSD)
3. Records coverage fraction = (CREMP conformers matched) / (total CREMP conformers)

**Sampling modes:**

- **`etkdg`** (default): standard nvmolkit ETKDGv3 → GPU Butina → MACE scoring.
- **`etkdg+torsional`**: adds a Pool B from `torsional_sampling.sample_constrained_confs`.
  Pool B conformers are MACE-scored and filtered to energies within `mean(A) + 2*std(A)`
  before merging; this prevents the torsional pool from flooding the Butina deduplication
  step with high-energy ring-closure failures.

**Checkpointing:** the output CSV is written row-by-row. If the script is interrupted,
re-running with the same `--output_csv` path skips any `(sequence, n_confs, sampling_mode)`
triple already present. This makes it safe to run on a cluster where jobs are preempted.

**n_confs sweep:** passing `--n_confs 500,1000,2000` processes each molecule at all
three values in a single pass. The expensive step is loading and parsing the CREMP
pickle; the embedding cost scales with `n_confs` but pickle I/O is amortised across
all requested values.

**Usage:**

```bash
# ETKDG only, sweep conformer counts
python src/validation/cremp_coverage.py \
    --subset_csv   data/processed/cremp/validation_subset.csv \
    --pickle_dir   data/raw/cremp/pickle \
    --output_csv   data/processed/cremp/coverage.csv \
    --n_confs      500,1000,2000

# ETKDG + torsional
python src/validation/cremp_coverage.py \
    --subset_csv   data/processed/cremp/validation_subset.csv \
    --pickle_dir   data/raw/cremp/pickle \
    --output_csv   data/processed/cremp/coverage.csv \
    --n_confs      1000 \
    --torsional_sampling \
    --torsional_n_samples 200
```

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `sequence` | CREMP dot-separated sequence string |
| `n_confs` | requested conformer count |
| `sampling_mode` | `etkdg` or `etkdg+torsional` |
| `coverage` | fraction of CREMP conformers matched (0–1) |
| `n_ref_confs` | number of CREMP reference conformers |
| `n_generated` | number of Butina-representative conformers generated |

---

## make_validation_sets_cremp.py

One-time script to build the stratified validation subset from the CREMP summary CSV.
Stratification is across three axes to ensure the validation subset is representative
of the full CREMP distribution:

- **Topology**: `all-L`, `D-only`, `NMe-only`, `D+NMe` (derived from CREMP sequence notation)
- **Ring size**: 4, 5, or 6 monomers
- **Heavy atom count**: tertile-binned within each ring size (`small`, `medium`, `large`)

The topology parser infers D and NMe residues from CREMP's dot-separated sequence format:
a lowercase residue name indicates a D-amino acid; a `Me` prefix on any residue indicates
N-methylation. This is the same classification logic used for building the Ramachandran
grids, so the validation subset and the Ramachandran prior are stratified consistently.

**Usage:**

```bash
python src/validation/make_validation_sets_cremp.py \
    --summary_csv data/raw/cremp/summary.csv \
    --output_csv  data/processed/cremp/validation_subset.csv \
    --n_per_stratum 14 \
    --seed 42
```

---

## make_validation_sets.py

Builds validation subsets for non-CREMP datasets (e.g. GEOM-Drugs). See file-level
docstring for dataset-specific stratification logic.

---

## geom_drugs.py

Coverage benchmark against GEOM-Drugs, a large dataset of drug-like molecules with
MMFF-optimised conformer ensembles. The benchmark logic mirrors `cremp_coverage.py`
but uses symmetric RMSD from `spyrmsd` without the backbone-residue-aware matching.

The primary purpose of the GEOM-Drugs benchmark is to verify that confsweeper's
macrocycle-specific flags (`useMacrocycleTorsions`, `useMacrocycle14config`) do not
degrade coverage for non-macrocyclic drug-like molecules relative to standard ETKDGv3.
Stratification is by number of rotatable bonds, since conformational coverage scales
most directly with rotational degrees of freedom.

---

## peptides.py and peptides_utils.py

Utilities for generating test peptide inputs programmatically. `peptides.py` builds
SMILES strings for cyclic homo-oligomers of standard amino acids and passes them through
the confsweeper pipeline. `peptides_utils.py` provides helpers for SMILES assembly,
sequence validation, and RDKit mol construction.

These modules are used to generate the `tests/peptides/` inputs for the test suite.
They are not part of the benchmarking pipeline and are not needed for normal usage.

---

## barcodes.py

Topological data analysis (TDA) utilities using GUDHI. Computes alpha-complex or Rips-complex
persistent homology from conformer coordinate point clouds.

The motivation for TDA-based conformer characterisation is that coverage metrics based on
pairwise RMSD may miss differences in the *topological* structure of the conformer ensemble —
e.g. two ensembles could have similar pairwise RMSDs but very different distributions of
dihedral angle basins. Persistent barcodes encode the multi-scale shape of the ensemble
without choosing a particular cutoff distance.

This module is experimental and not connected to the main benchmarking pipeline. It requires
GUDHI (`pixi install` includes it in the `default` environment).
