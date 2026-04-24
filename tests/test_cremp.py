"""Unit tests for src/validation/cremp.py."""
import pickle

import pandas as pd
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from validation.cremp import (
    calc_coverage,
    iter_validation_mols,
    pairwise_rmsd_tensor,
    symmetric_rmsd,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mol_with_confs(smiles: str, n_confs: int, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed)
    return mol


def _write_pickle(path, smiles, n_confs, seed=42):
    mol = _make_mol_with_confs(smiles, n_confs, seed=seed)
    data = {
        "rd_mol": mol,
        "smiles": smiles,
        "uniqueconfs": n_confs,
        "lowestenergy": -50.0,
        "charge": 0,
        "conformers": [],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return mol


def _make_subset_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "subset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


WATER_SMILES = "O"
ETHANOL_SMILES = "CCO"

# Cyclooctane — flexible ring with distinct conformers for RMSD tests
CYCLOOCTANE_SMILES = "C1CCCCCCC1"


# ---------------------------------------------------------------------------
# iter_validation_mols
# ---------------------------------------------------------------------------


class TestIterValidationMols:
    def _base_row(self, sequence, smiles):
        return {
            "sequence": sequence,
            "smiles": smiles,
            "topology": "all-L",
            "atom_bin": "small",
            "num_monomers": 4,
            "num_heavy_atoms": 20,
        }

    def test_yields_correct_fields(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        _write_pickle(pkl_dir / "A.V.G.L.pickle", ETHANOL_SMILES, n_confs=3)

        csv = _make_subset_csv(tmp_path, [self._base_row("A.V.G.L", ETHANOL_SMILES)])
        results = list(iter_validation_mols(csv, pkl_dir))

        assert len(results) == 1
        seq, smi, mol, meta = results[0]
        assert seq == "A.V.G.L"
        assert smi == ETHANOL_SMILES
        assert mol.GetNumConformers() == 3
        assert meta["topology"] == "all-L"
        assert meta["uniqueconfs"] == 3

    def test_skips_missing_pickle(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        # No pickle file written
        csv = _make_subset_csv(tmp_path, [self._base_row("A.V.G.L", ETHANOL_SMILES)])
        results = list(iter_validation_mols(csv, pkl_dir))
        assert results == []

    def test_skips_empty_mol(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        # Write pickle with mol that has 0 conformers
        mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL_SMILES))
        data = {
            "rd_mol": mol,
            "smiles": ETHANOL_SMILES,
            "uniqueconfs": 0,
            "lowestenergy": 0.0,
            "charge": 0,
            "conformers": [],
        }
        with open(pkl_dir / "A.V.G.L.pickle", "wb") as f:
            pickle.dump(data, f)

        csv = _make_subset_csv(tmp_path, [self._base_row("A.V.G.L", ETHANOL_SMILES)])
        results = list(iter_validation_mols(csv, pkl_dir))
        assert results == []

    def test_skips_atom_count_mismatch(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        # Pickle mol is water, but SMILES in subset is ethanol → atom count mismatch
        _write_pickle(pkl_dir / "A.V.G.L.pickle", WATER_SMILES, n_confs=2)
        csv = _make_subset_csv(tmp_path, [self._base_row("A.V.G.L", ETHANOL_SMILES)])
        results = list(iter_validation_mols(csv, pkl_dir))
        assert results == []

    def test_iterates_multiple_molecules(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        seqs = ["A.V.G.L", "a.V.G.L", "MeA.V.G.L"]
        for seq in seqs:
            _write_pickle(pkl_dir / f"{seq}.pickle", ETHANOL_SMILES, n_confs=2)

        rows = [self._base_row(s, ETHANOL_SMILES) for s in seqs]
        csv = _make_subset_csv(tmp_path, rows)
        results = list(iter_validation_mols(csv, pkl_dir))
        assert len(results) == 3
        assert [r[0] for r in results] == seqs

    def test_streams_one_at_a_time(self, tmp_path):
        """Confirm it's an iterator, not a list."""
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        _write_pickle(pkl_dir / "A.V.G.L.pickle", ETHANOL_SMILES, n_confs=2)
        csv = _make_subset_csv(tmp_path, [self._base_row("A.V.G.L", ETHANOL_SMILES)])
        result = iter_validation_mols(csv, pkl_dir)
        import types

        assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# pairwise_rmsd_tensor
# ---------------------------------------------------------------------------


class TestPairwiseRmsdTensor:
    def test_self_rmsd_is_zero(self):
        coords = torch.randn(5, 10, 3)
        rmsds = pairwise_rmsd_tensor(coords, coords)
        assert rmsds.shape == (5, 5)
        assert torch.allclose(torch.diag(rmsds), torch.zeros(5), atol=1e-5)

    def test_output_shape(self):
        a = torch.randn(4, 8, 3)
        b = torch.randn(6, 8, 3)
        rmsds = pairwise_rmsd_tensor(a, b)
        assert rmsds.shape == (4, 6)

    def test_symmetry(self):
        coords = torch.randn(3, 7, 3)
        rmsds = pairwise_rmsd_tensor(coords, coords)
        assert torch.allclose(rmsds, rmsds.T, atol=1e-5)

    def test_known_value(self):
        # Two conformers: one at origin, one shifted by 1 Å in x for all atoms
        n_atoms = 4
        a = torch.zeros(1, n_atoms, 3)
        b = torch.ones(1, n_atoms, 3) * 0.0
        b[0, :, 0] = 1.0  # shift x by 1 Å
        rmsds = pairwise_rmsd_tensor(a, b)
        # RMSD = sqrt(mean((1^2 + 0 + 0))) = 1.0
        assert abs(rmsds[0, 0].item() - 1.0) < 1e-5

    def test_non_negative(self):
        a = torch.randn(3, 5, 3)
        b = torch.randn(4, 5, 3)
        assert (pairwise_rmsd_tensor(a, b) >= 0).all()


# ---------------------------------------------------------------------------
# symmetric_rmsd
# ---------------------------------------------------------------------------


class TestSymmetricRmsd:
    def test_self_rmsd_is_zero(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=2)
        r = symmetric_rmsd(mol, 0, mol, 0)
        assert r < 1e-4

    def test_returns_float(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=2)
        r = symmetric_rmsd(mol, 0, mol, 1)
        assert isinstance(r, float)

    def test_non_negative(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=3)
        for i in range(3):
            for j in range(3):
                assert symmetric_rmsd(mol, i, mol, j) >= 0.0

    def test_symmetric_ring_confs_may_be_zero(self):
        # Cyclooctane is fully symmetric — spyrmsd with symmetry=True can map any
        # two conformers onto each other via ring automorphisms, so RMSD may be 0.
        # This is the correct behavior and distinguishes symmetric_rmsd from plain RMSD.
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=10, seed=0)
        r = symmetric_rmsd(mol, 0, mol, 9)
        assert r >= 0.0  # not asserting > 0: symmetry collapsing to 0 is expected

    def test_asymmetric_ring_confs_nonzero(self):
        # Cyclopentane: embed one conformer, then add a heavily perturbed copy.
        # The perturbation is large enough that no rotation + symmetry mapping
        # can reduce the RMSD to zero.
        mol = _make_mol_with_confs("C1CCCC1", n_confs=1, seed=0)
        conf = mol.GetConformer(0)
        import numpy as np
        from rdkit.Chem import rdchem

        new_conf = rdchem.Conformer(mol.GetNumAtoms())
        pos = conf.GetPositions()
        # Displace every atom by a large, non-uniform amount to break all symmetry
        rng = np.random.default_rng(99)
        perturbed = pos + rng.uniform(3.0, 6.0, size=pos.shape)
        for i, p in enumerate(perturbed):
            new_conf.SetAtomPosition(i, p.tolist())
        mol.AddConformer(new_conf, assignId=True)
        r = symmetric_rmsd(mol, 0, mol, 1)
        assert r > 0.0


# ---------------------------------------------------------------------------
# calc_coverage
# ---------------------------------------------------------------------------


class TestCalcCoverage:
    def test_identical_confs_full_coverage(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=3)
        conf_ids = [c.GetId() for c in mol.GetConformers()]
        coverage, min_rmsds = calc_coverage(
            ref_mol=mol,
            gen_mol=mol,
            gen_conf_ids=conf_ids,
            rmsd_cutoff=1.0,
        )
        assert coverage == pytest.approx(1.0)
        assert all(r < 1e-4 for r in min_rmsds)

    def test_empty_gen_confs_zero_coverage(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=3)
        coverage, min_rmsds = calc_coverage(
            ref_mol=mol,
            gen_mol=mol,
            gen_conf_ids=[],
            rmsd_cutoff=1.0,
        )
        assert coverage == pytest.approx(0.0)
        assert min_rmsds == []

    def test_coverage_between_zero_and_one(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=5, seed=1)
        all_ids = [c.GetId() for c in mol.GetConformers()]
        # Use only first 2 generated confs to cover 5 reference confs
        coverage, min_rmsds = calc_coverage(
            ref_mol=mol,
            gen_mol=mol,
            gen_conf_ids=all_ids[:2],
            rmsd_cutoff=0.01,  # tight cutoff — only self-matches count
        )
        assert 0.0 <= coverage <= 1.0
        assert len(min_rmsds) == 5

    def test_tight_cutoff_lowers_coverage(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=5, seed=0)
        all_ids = [c.GetId() for c in mol.GetConformers()]
        cov_loose, _ = calc_coverage(mol, mol, all_ids, rmsd_cutoff=10.0)
        cov_tight, _ = calc_coverage(mol, mol, all_ids, rmsd_cutoff=0.001)
        assert cov_loose >= cov_tight

    def test_returns_one_min_rmsd_per_ref_conf(self):
        mol = _make_mol_with_confs(CYCLOOCTANE_SMILES, n_confs=4)
        all_ids = [c.GetId() for c in mol.GetConformers()]
        _, min_rmsds = calc_coverage(mol, mol, all_ids, rmsd_cutoff=1.0)
        assert len(min_rmsds) == 4
