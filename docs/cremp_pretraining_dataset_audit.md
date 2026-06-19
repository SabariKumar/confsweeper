# CREMP pretraining dataset — construction audit

> **Scope:** All statistics and findings in this document are for the **CREMP dataset** (36,198 head-to-tail macrocyclic peptides, hosted in this repo at [`data/raw/cremp/`](../data/raw/cremp/); see [`dataset_summary.md`](../data/raw/cremp/dataset_summary.md)). The **code being audited lives in the `peptide_electrostatics` repo** (`../peptide_electrostatics`), which consumes this CREMP dataset for autoencoder pretraining. Source paths below (e.g. `data/dataset.py`, `training/train.py`) are relative to `peptide_electrostatics/src/peptide_electrostatics/` unless otherwise noted; they are not files in this repo.

Audit of all code that builds the pretraining dataset from CREMP before conformer embeddings reach the encoder: data loading, filtering, preprocessing, splitting, augmentation. Answers reflect **what the current code does**; where the code contradicts a comment or planning doc, the code is reported and the discrepancy flagged.

Primary files (in `peptide_electrostatics`): `src/peptide_electrostatics/data/dataset.py`, `src/peptide_electrostatics/training/data_module.py`, `src/peptide_electrostatics/training/train.py`. Raw data is the CREMP/Zenodo dataset in this repo at `data/raw/cremp/`.

---

## 1. CREMP filtering

**No filtering is applied to raw CREMP sequences before pretraining.** The dataset includes every `.pickle` file in `pickle_dir`, and `train.py` never passes a `sequences` subset.

- Default sequence selection takes *all* pickles, unfiltered — `data/dataset.py:57-64`:
  ```python
  if sequences is not None:
      self.sequences = list(sequences)
  else:
      self.sequences = sorted(
          os.path.splitext(f)[0]
          for f in os.listdir(self.pickle_dir)
          if f.endswith(".pickle")
      )
  ```
- `train.py` constructs `CREMPDataModule` without a `sequences` argument, so it always falls through to "all pickles" — `training/train.py:195-203`. `CREMPDataModule` likewise defaults `sequences=None` — `training/data_module.py:37`.

**Minimum conformer count per molecule:** None.
- The only conformer-count branch is the *upper* bound for c_max selection — `data/dataset.py:114`: `if n_confs > self.c_max:`. A molecule with 1 conformer is accepted.
- Reinforced by the raw data: CREMP `uniqueconfs` **min = 1** (`data/raw/cremp/dataset_summary.md:45`), so single-conformer molecules exist in the source and pass through unfiltered.

**Boltzmann entropy / max_bw cutoff:** None.
- No entropy or max_bw computation exists in the pretraining data path (a grep for `entropy|max_bw` returns hits only in `data/fine_tune/`, not in `data/` or `training/`). Near-one-hot distributions are **not** excluded.

**Residue type / non-canonical / sequence-length bounds:** None.
- No residue, backbone, or sequence-length logic exists in `data/dataset.py` or `data_module.py` (grep for `residue|seq_len|canonical|backbone` returns only ESP "canonical frame" comments, unrelated to filtering).
- The raw set spans `num_monomers ∈ {4, 5, 6}` (`dataset_summary.md:36-40`) and includes D-amino acids and N-methylated residues (`dataset_summary.md:7`), all head-to-tail cyclic — none of which the code screens on.

**Conformers dropped beyond c_max selection:** No.
- The only conformer reduction is the top-c_max-by-Boltzmann-weight selection — `data/dataset.py:113-118`:
  ```python
  # Select top-c_max conformers by Boltzmann weight, preserving index order
  if n_confs > self.c_max:
      top_k = np.argsort(bw)[::-1][: self.c_max]
      selected = np.sort(top_k)
  else:
      selected = np.arange(n_confs)
  ```
  Padding conformers added to reach c_max (`dataset.py:128-131`) are later skipped in `collate_fn` (`dataset.py:198-199`), so they never reach the encoder — but that is padding removal, not a drop of real conformers.

> **Note (not a filter, but relevant):** molecules missing a precomputed topo/ESP `.pt` file are **not** excluded — the item simply omits that target key (`dataset.py:159-171`, guarded by `if os.path.exists(...)`). Precompute coverage does not gate dataset membership.

---

## 2. Boltzmann weights

**Energy model / source:** The CREMP-provided weights are used **directly** — read straight from the pickle's `boltzmannweight` field. They are **not** recalculated with MACE-OFF23 or any other model — `data/dataset.py:104-110`:
```python
# Per-conformer energies and Boltzmann weights from CREMP (GFN2-xTB)
energies = np.array(
    [conf_meta[i]["totalenergy"] for i in range(n_confs)], dtype=np.float32
)
bw = np.array(
    [conf_meta[i]["boltzmannweight"] for i in range(n_confs)], dtype=np.float32
)
```
- The comment attributes them to GFN2-xTB (`dataset.py:104`). The raw-data summary confirms these are GFN2-xTB Boltzmann populations (`poplowestpct` / `temperature` columns, `dataset_summary.md:27-28`).

**Renormalized / clipped / temperature-rescaled:**
- Renormalized: **yes**, divided by their sum — `data/dataset.py:111`: `bw = bw / bw.sum()  # renormalize to sum to 1 after any numerical drift`. After top-c_max truncation the selected subset is **not** re-renormalized (`bw_out[:C] = bw[selected]` at `dataset.py:136` keeps the full-set normalization, so the kept weights need not sum to 1).
- Clipped: **no** clipping in the pretraining path.
- Temperature-rescaled: **no.** There is no α/temperature parameter in pretraining.

> **Clarifying boundary (fine-tune vs. pretrain):** the confsweeper integration introduces a *MACE/UMA-energy* Boltzmann reweighting (`boltzmann_weights_from_energies`, `finetune/finetune_conformer_adapter.py:24-36`), but that path is **fine-tuning only**. It does not touch the pretraining weights, which remain CREMP GFN2-xTB.

---

## 3. Train/val split

**Strategy:** Random split — `training/data_module.py:68-72`:
```python
self.train_dataset, self.val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)
```
It is **not** scaffold-based or sequence-based. (This contrasts with the fine-tune pipeline, which does use a scaffold split — pretraining does not.)

**Ratio:** `val_fraction` default `0.1`, i.e. 90/10 — `data_module.py:41` and CLI default `train.py:66`. Computed as `n_val = max(1, int(len(dataset) * self.val_fraction))`, `n_train = len(dataset) - n_val` — `data_module.py:66-67`.

**Seed:** Fixed at **42** — `data_module.py:71`: `generator=torch.Generator().manual_seed(42)`.

---

## 4. Dataset size

**Not pinned in code, but the referenced data directory gives a concrete figure.**

- There is no hardcoded count, no filtering that fixes a size, and no summary log that prints the molecule/conformer total. `__len__` is purely dynamic on the contents of `pickle_dir` — `data/dataset.py:66-75`:
  ```python
  def __len__(self) -> int:
      ...
      return len(self.sequences)
  ```
- The CREMP pickle directory referenced by the test fixture, `data/raw/cremp/pickle` in this repo (`tests/test_dataset.py:15`), contains **36,198 `.pickle` files** (one per molecule), and `dataset_summary.md:5` states **"Total molecules: 36,198 macrocyclic peptides"**. So if pretraining is pointed at this directory, `len(dataset)` = 36,198.
- **Caveat:** `--pickle_dir` is a required CLI argument with no default (`training/train.py:60`); the actual molecule count is whatever that directory contains at runtime. No code emits the total.
- Conformers per molecule: raw CREMP has median **655.5** unique conformers/molecule (mean 863.6, min 1, max 12,268; `dataset_summary.md:42-50`), but each molecule is capped at `c_max` (CLI default **20** — `train.py:63`; class default 50 — `dataset.py:48`) by descending Boltzmann weight. The realized per-molecule count is `C = len(selected)` (`dataset.py:120`) and varies; no code emits a total conformer count.

---

## 5. Additional sequences beyond CREMP

**No.** The pretraining dataset is exclusively CREMP per-molecule pickles.

- The only dataset class in the pretraining path is `CREMPDataset`, which reads `rd_mol` / `conformers` from pickle files in a single `pickle_dir` — `data/dataset.py:90-94`:
  ```python
  with open(path, "rb") as f:
      raw = pickle.load(f)
  mol = raw["rd_mol"]
  conf_meta = raw["conformers"]
  ```
- There is **no** code mixing in confsweeper/ETKDG-generated conformers, no second source directory, and no concatenation of datasets in `data_module.py` (it wraps a single `CREMPDataset` — `data_module.py:59-65`).

> **Ruling out a likely misread:** although confsweeper is a package dependency of `peptide_electrostatics` (`pixi.toml:52`) and is used to *generate* conformers, that generation feeds **only the fine-tuning Stage 1a path** (`finetune/finetune_generate_conformers.py:171-176`), never pretraining. The pretraining pickle directory happens to be *stored* inside this confsweeper repo (`data/raw/cremp/`), but it is the unmodified CREMP/Zenodo dataset (`dataset_summary.md:2,86-87`), not confsweeper-sampler output.

---

## Discrepancy flags (code vs. docs)

- The planning docs describe CREMP as "~15k macrocyclic peptide sequences"; the actual raw directory holds **36,198** molecules (`dataset_summary.md:5`), and the code imposes **no filtering and logs no count** — the realized pretraining size is entirely determined by the `--pickle_dir` contents at runtime.
- Several "intended" filters implied by the broader project framing — entropy/max_bw exclusion of near-one-hot ensembles, minimum conformer counts, non-canonical-backbone screening — are **implemented only for the fine-tune/PAMPA path** (`data/fine_tune/`), **not** for CREMP pretraining. As implemented, pretraining passes every CREMP molecule through unfiltered.
- The MACE/UMA Boltzmann reweighting is **fine-tune only**; pretraining uses CREMP's GFN2-xTB weights directly. Do not conflate the two pipelines.

---

## Appendix — Backbone flexibility and torsional degrees of freedom (CREMP dataset)

Computed over all 36,198 molecules of the **CREMP dataset** from `data/raw/cremp/summary.csv` (RDKit parse of each SMILES). Three flexibility metrics are compared, plus a combined effective-DOF estimate.

- **naive RDKit (strict):** `rdMolDescriptors.CalcNumRotatableBonds` (Default). Excludes ring bonds and amide C–N bonds.
- **NonStrict:** `CalcNumRotatableBonds(..., NumRotatableBondsOptions.NonStrict)`. Simpler rotatable-bond SMARTS, but still excludes ring bonds (`-&!@-`).
- **Backbone dihedrals (φ, ψ):** 2 per backbone residue, counted via this repo's backbone SMARTS (`src/torsional_sampling.py:45`), matching `get_backbone_dihedrals`:
  ```
  [C:1](=O)[N:2][CX4:3][C:4](=O)[N:5]
  ```
  The ω amide bond is excluded as planar. `[N:2]` has no H-count constraint and `[CX4:3]` matches any sp³ Cα, so tertiary backbone nitrogens are captured.
- **Total effective torsional DOF:** naive (side-chain/exocyclic) + backbone dihedrals.

| Ring | Molecules | naive RDKit (strict) | NonStrict | Backbone dihedrals (φ,ψ) | Total effective DOF |
|---|---|---|---|---|---|
| 4-mer | 17,842 | 5.9 | 5.9 | 8.0 | 13.9 |
| 5-mer | 13,644 | 7.0 | 7.0 | 10.0 | 17.0 |
| 6-mer | 4,712 | 8.4 | 8.4 | 12.0 | 20.4 |
| **all** | 36,198 | **6.6** | **6.6** | **9.3** | **15.9** |

**SMARTS coverage (proline + N-methyl):** the number of residues matched by the backbone SMARTS equals `num_monomers` for **100.0%** of all CREMP molecules — including **100.0%** of the 17,422 N-methyl-containing sequences and **100.0%** of the 8,034 proline-containing sequences. Tertiary backbone nitrogens (proline ring N, N-methylated N) are matched correctly.

**Interpretation:**
- **naive ≡ NonStrict** (identical to 0.1) because every backbone amide is *in the macrocycle ring*, and both RDKit definitions exclude in-ring bonds. The only bonds either metric counts are side-chain / exocyclic.
- **Backbone dihedral count is deterministic = 2 × num_monomers** (8/10/12, zero variance), capturing the in-ring φ/ψ torsional freedom that both RDKit metrics miss entirely — the flexibility that drives the large CREMP conformer ensembles (~700–1,000 unique conformers/molecule).
- **Total effective DOF (~14–20)** exceeds the RDKit rotatable-bond count by roughly 2–2.5×; the RDKit metric alone substantially undercounts true conformational flexibility for these macrocycles.
