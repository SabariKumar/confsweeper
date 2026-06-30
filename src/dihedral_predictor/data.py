"""
Dataset construction and loading for the dihedral predictor.

Extracts one training record per CREMP peptide: per-residue base features
(`residues.residue_features`) plus the **dominant** (max-Boltzmann-weight)
conformer's per-residue (phi, psi, omega) binned targets. Base features are
stored unpadded so the cyclic ±window neighbour augmentation is applied per
peptide on its true length (padding never leaks across the ring closure); the
torch Dataset augments and the collate fn pads per batch with a mask.

The split is by peptide (random, fixed seed) so no sequence leaks between
train/val/test. A topology-stratified split is a stronger generalisation test
and is left as a follow-up.
"""

import csv
import os
import pickle

import numpy as np
import torch

from .residues import (
    MAX_CHI,
    PHI_PSI_BINS,
    angle_to_bin,
    backbone_dihedral_values,
    neighbor_augment,
    omega_to_bin,
    residue_features,
    sidechain_chi_values,
)


def extract_record(mol, boltzmann_weights: np.ndarray) -> dict | None:
    """
    Build one peptide record: base features + dominant-conformer binned targets.

    Params:
        mol: Chem.Mol : CREMP peptide with conformers (explicit Hs)
        boltzmann_weights: np.ndarray : per-conformer CREST Boltzmann weights
    Returns:
        dict with keys feats (n_res, F), phi_bin/psi_bin/omega_bin (n_res,), or
        None if the peptide has no backbone residues / inconsistent conformers
    """
    feats = residue_features(mol)
    n_res = feats.shape[0]
    if n_res == 0:
        return None
    if mol.GetNumConformers() != len(boltzmann_weights):
        return None
    dom = int(np.asarray(boltzmann_weights).argmax())
    dom_id = mol.GetConformers()[dom].GetId()
    phi, psi, omega = backbone_dihedral_values(mol, dom_id)
    if len(phi) != n_res:
        return None
    chi_deg, chi_mask = sidechain_chi_values(mol, dom_id)  # (n_res, MAX_CHI)
    chi_bin = np.zeros((n_res, MAX_CHI), dtype=np.int64)
    for i in range(n_res):
        for k in range(MAX_CHI):
            if chi_mask[i, k]:
                chi_bin[i, k] = angle_to_bin(chi_deg[i, k])
    return {
        "feats": feats,
        "phi_bin": np.array([angle_to_bin(a) for a in phi], dtype=np.int64),
        "psi_bin": np.array([angle_to_bin(a) for a in psi], dtype=np.int64),
        "omega_bin": np.array([omega_to_bin(a) for a in omega], dtype=np.int64),
        "chi_bin": chi_bin,
        "chi_mask": chi_mask,
    }


def build_dataset(
    summary_csv: str,
    pickle_dir: str,
    out_path: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    """
    Extract records for every CREMP peptide and save a split dataset pickle.

    Params:
        summary_csv: str : CREMP summary.csv (provides the sequence list)
        pickle_dir: str : directory of per-sequence CREMP pickles
        out_path: str : output .pkl path
        val_frac: float : fraction of peptides for validation
        test_frac: float : fraction of peptides for test
        seed: int : split RNG seed
        limit: int | None : cap on #peptides (for smoke tests)
    Returns:
        the saved dataset dict
    """
    rows = list(csv.DictReader(open(summary_csv)))
    if limit:
        rows = rows[:limit]
    seqs, feats, phi, psi, omega, chi, chi_mask = [], [], [], [], [], [], []
    n_skip = 0
    for r in rows:
        seq = r["sequence"]
        path = os.path.join(pickle_dir, f"{seq}.pickle")
        try:
            d = pickle.load(open(path, "rb"))
            bw = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            rec = extract_record(d["rd_mol"], bw)
            if rec is None:
                n_skip += 1
                continue
            seqs.append(seq)
            feats.append(rec["feats"])
            phi.append(rec["phi_bin"])
            psi.append(rec["psi_bin"])
            omega.append(rec["omega_bin"])
            chi.append(rec["chi_bin"])
            chi_mask.append(rec["chi_mask"])
        except Exception:  # noqa: BLE001
            n_skip += 1

    n = len(seqs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    split = np.empty(n, dtype=object)
    split[perm[:n_test]] = "test"
    split[perm[n_test : n_test + n_val]] = "val"
    split[perm[n_test + n_val :]] = "train"

    out = {
        "seqs": seqs,
        "feats": feats,
        "phi_bin": phi,
        "psi_bin": psi,
        "omega_bin": omega,
        "chi_bin": chi,
        "chi_mask": chi_mask,
        "split": split,
        "phi_psi_bins": PHI_PSI_BINS,
        "max_chi": MAX_CHI,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"extracted {n} peptides ({n_skip} skipped) -> {out_path}")
    print(
        f"  split: train {(split=='train').sum()} / val {(split=='val').sum()} / test {(split=='test').sum()}"
    )
    return out


class DihedralDataset(torch.utils.data.Dataset):
    """One item per peptide: cyclic-neighbour-augmented features + binned targets."""

    def __init__(self, data: dict, split: str, window: int = 1):
        """
        Params:
            data: dict : dataset dict from build_dataset / load_dataset
            split: str : 'train' | 'val' | 'test'
            window: int : neighbour augmentation half-width
        Returns:
            None
        """
        self.window = window
        idx = [i for i, s in enumerate(data["split"]) if s == split]
        self.feats = [data["feats"][i] for i in idx]
        self.phi = [data["phi_bin"][i] for i in idx]
        self.psi = [data["psi_bin"][i] for i in idx]
        self.omega = [data["omega_bin"][i] for i in idx]
        self.chi = [data["chi_bin"][i] for i in idx]
        self.chi_mask = [data["chi_mask"][i] for i in idx]
        self.seqs = [data["seqs"][i] for i in idx]

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, i: int) -> dict:
        x = neighbor_augment(self.feats[i], window=self.window)
        return {
            "x": torch.from_numpy(x),
            "phi": torch.from_numpy(self.phi[i]),
            "psi": torch.from_numpy(self.psi[i]),
            "omega": torch.from_numpy(self.omega[i]),
            "chi": torch.from_numpy(self.chi[i]),  # (n_res, MAX_CHI)
            "chi_mask": torch.from_numpy(self.chi_mask[i]),  # (n_res, MAX_CHI) bool
        }


def collate(batch: list[dict]) -> dict:
    """
    Pad a batch of variable-length peptides to the max residue count + mask.

    Params:
        batch: list[dict] : items from DihedralDataset
    Returns:
        dict of padded tensors x (B,L,F), phi/psi/omega (B,L), mask (B,L) bool
    """
    L = max(b["x"].shape[0] for b in batch)
    B, F = len(batch), batch[0]["x"].shape[1]
    n_chi = batch[0]["chi"].shape[1]
    x = torch.zeros(B, L, F)
    phi = torch.zeros(B, L, dtype=torch.long)
    psi = torch.zeros(B, L, dtype=torch.long)
    omega = torch.zeros(B, L, dtype=torch.long)
    chi = torch.zeros(B, L, n_chi, dtype=torch.long)
    chi_mask = torch.zeros(B, L, n_chi, dtype=torch.bool)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n] = b["x"]
        phi[i, :n] = b["phi"]
        psi[i, :n] = b["psi"]
        omega[i, :n] = b["omega"]
        chi[i, :n] = b["chi"]
        chi_mask[i, :n] = b["chi_mask"]
        mask[i, :n] = True
    return {
        "x": x,
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "chi": chi,
        "chi_mask": chi_mask,
        "mask": mask,
    }


def load_dataset(path: str) -> dict:
    """
    Load a dataset pickle written by build_dataset.

    Params:
        path: str : .pkl path
    Returns:
        the dataset dict
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)
