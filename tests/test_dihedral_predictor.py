"""Tests for src/dihedral_predictor/ (issue #20)."""
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dihedral_predictor.data import DihedralDataset, collate, extract_record
from dihedral_predictor.model import DihedralPredictor
from dihedral_predictor.residues import (
    N_BASE_FEATURES,
    PHI_PSI_BINS,
    angle_to_bin,
    backbone_dihedral_values,
    bin_to_center,
    neighbor_augment,
    omega_bin_to_center,
    omega_quads,
    omega_to_bin,
    residue_atoms,
    residue_features,
)
from dihedral_predictor.seed import (
    make_bounds_phi_psi_omega,
    predict_dihedrals,
    seed_conformers,
)
from torsional_sampling import get_backbone_dihedrals

_CYCLOALA4 = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"


def _cycloala4():
    mol = Chem.AddHs(Chem.MolFromSmiles(_CYCLOALA4))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


# --- binning -------------------------------------------------------------


def test_angle_bin_roundtrip_center():
    for b in range(PHI_PSI_BINS):
        assert angle_to_bin(bin_to_center(b)) == b


def test_angle_bin_wraps():
    assert angle_to_bin(-180.0) == angle_to_bin(180.0)
    assert 0 <= angle_to_bin(179.9) < PHI_PSI_BINS


def test_omega_bins():
    assert omega_to_bin(180.0) == 1 and omega_to_bin(-175.0) == 1  # trans
    assert omega_to_bin(0.0) == 0 and omega_to_bin(10.0) == 0  # cis
    assert omega_bin_to_center(0) == 0.0 and omega_bin_to_center(1) == 180.0


# --- neighbour augmentation ---------------------------------------------


def test_neighbor_augment_shape_and_cyclic():
    f = np.arange(3 * 2).reshape(3, 2).astype("float32")  # 3 residues, 2 feats
    aug = neighbor_augment(f, window=1)
    assert aug.shape == (3, 6)
    # res0 = [last, self, next]
    assert list(aug[0]) == [4.0, 5.0, 0.0, 1.0, 2.0, 3.0]


# --- feature/target extraction + alignment ------------------------------


def test_residue_features_shape_and_alignment():
    mol = _cycloala4()
    feats = residue_features(mol)
    assert feats.shape == (4, N_BASE_FEATURES)
    # features, dihedrals, omega all in the same residue order
    assert len(get_backbone_dihedrals(mol)) == feats.shape[0]
    assert len(omega_quads(mol)) == feats.shape[0]
    assert len(residue_atoms(mol)) == feats.shape[0]
    # Ala: not NMe, not Gly, side chain has 1 heavy atom (CB)
    assert feats[:, 0].sum() == 0  # is_nme
    assert feats[:, 1].sum() == 0  # is_gly
    assert np.allclose(feats[:, 4], 1.0)  # sc_heavy == 1


def test_backbone_dihedral_values_lengths():
    mol = _cycloala4()
    phi, psi, om = backbone_dihedral_values(mol, mol.GetConformer().GetId())
    assert len(phi) == len(psi) == len(om) == 4


def test_extract_record():
    mol = _cycloala4()
    rec = extract_record(mol, np.array([1.0]))  # one conformer, dominant=0
    assert rec is not None
    for k in ("phi_bin", "psi_bin", "omega_bin"):
        assert rec[k].shape == (4,) and rec[k].dtype == np.int64
        assert rec[k].min() >= 0


def test_extract_record_rejects_mismatched_weights():
    mol = _cycloala4()
    assert extract_record(mol, np.array([1.0, 0.5])) is None  # 2 weights, 1 conf


# --- dataset / collate ---------------------------------------------------


def test_dataset_and_collate():
    mol = _cycloala4()
    rec = extract_record(mol, np.array([1.0]))
    data = {
        "seqs": ["x", "y"],
        "feats": [rec["feats"], rec["feats"]],
        "phi_bin": [rec["phi_bin"], rec["phi_bin"]],
        "psi_bin": [rec["psi_bin"], rec["psi_bin"]],
        "omega_bin": [rec["omega_bin"], rec["omega_bin"]],
        "split": np.array(["train", "train"], dtype=object),
    }
    ds = DihedralDataset(data, "train", window=1)
    assert len(ds) == 2
    batch = collate([ds[0], ds[1]])
    assert batch["x"].shape == (2, 4, N_BASE_FEATURES * 3)
    assert batch["mask"].sum().item() == 8


# --- model ---------------------------------------------------------------


def test_model_forward_shapes():
    m = DihedralPredictor(in_features=N_BASE_FEATURES * 3, d_model=32, n_layers=2)
    x = torch.randn(2, 4, N_BASE_FEATURES * 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    pl, sl, ol = m(x, mask)
    assert pl.shape == (2, 4, PHI_PSI_BINS)
    assert ol.shape == (2, 4, 2)


def test_model_handles_large_macrocycle_and_padding():
    """>10-residue macrocycle plus a short peptide in the same padded batch."""
    m = DihedralPredictor(in_features=N_BASE_FEATURES * 3, d_model=32, n_layers=2)
    x = torch.randn(2, 14, N_BASE_FEATURES * 3)
    mask = torch.zeros(2, 14, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :14] = True  # 14-mer
    pl, _, ol = m(x, mask)
    assert pl.shape == (2, 14, PHI_PSI_BINS)
    assert torch.isfinite(pl).all()


# --- seeding -------------------------------------------------------------


def test_predict_and_seed_pipeline():
    mol = _cycloala4()
    n_res = len(get_backbone_dihedrals(mol))
    m = DihedralPredictor(in_features=N_BASE_FEATURES * 3, d_model=32, n_layers=2)
    phi, psi, om = predict_dihedrals(mol, m, window=1)
    assert len(phi) == len(psi) == len(om) == n_res
    bounds = make_bounds_phi_psi_omega(mol, phi, psi, om)
    assert bounds is None or isinstance(bounds, np.ndarray)
    # full pipeline must not crash; returns a (possibly empty) list of conf ids
    ids = seed_conformers(Chem.Mol(mol), m, window=1, n_attempts=3)
    assert isinstance(ids, list)
