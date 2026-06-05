"""Unit tests for get_mol_PE_pool_b in src/confsweeper.py.

Mirrors the structure of test_exhaustive_etkdg.py: mocks the slow / GPU
stages (constrained-DG sampling, MACE scoring, GPU MMFF) and exercises
the contract of the post-sampling pipeline (MMFF dispatch, energy filter,
energy-ranked dedup, non-centroid pruning).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from confsweeper import _KT_EV_298K, get_mol_PE_pool_b

# Cyclic tetrapeptide. Pool B's real sample_constrained_confs needs a
# macrocyclic backbone, but we mock that call out — any RDKit-embeddable
# molecule works for these tests.
TEST_SMILES = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"


def _mock_pool_b_sample(
    mol, grids, n_samples, n_attempts, tolerance_deg, strategy, seed
):
    """
    Drop-in for sample_constrained_confs that adds n_samples conformers via
    plain CPU ETKDG so the test covers the post-sampling pipeline without
    depending on the Ramachandran grids data file or the constrained-DG
    feasibility rate (which varies with the tolerance and ring closure).
    """
    del grids, n_attempts, tolerance_deg, strategy  # unused by the mock
    AllChem.EmbedMultipleConfs(
        mol, numConfs=n_samples, randomSeed=seed, clearConfs=False
    )
    return [c.GetId() for c in mol.GetConformers()]


def _make_seq_mock_mace():
    """
    Mock for `_mace_batch_energies` that yields globally monotone-increasing
    energies (i * 0.01 eV ≈ 0.39 kT_298K per step). Reused from the exhaustive
    tests' pattern — keeps energy filter / dedup behaviour deterministic.
    """
    counter = [0]

    def _mock(_calc, ase_mols):
        out = [(counter[0] + i) * 0.01 for i in range(len(ase_mols))]
        counter[0] += len(ase_mols)
        return out

    return _mock


@pytest.fixture
def pool_b_mocks():
    """
    Patch constrained-DG sampling, MACE scoring, and GPU MMFF with CPU-only
    drop-ins so tests run on any machine. Ramachandran grids are unused
    because the sample mock ignores them; tests pass an empty dict.
    """
    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.sample_constrained_confs", side_effect=_mock_pool_b_sample),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
    ):
        yield


def test_pool_b_returns_centroids_and_energies(pool_b_mocks):
    """Smoke: pipeline returns aligned centroids + energies in ascending order
    and the returned mol contains exactly the centroid conformers."""
    mol, centroid_ids, energies = get_mol_PE_pool_b(
        TEST_SMILES,
        grids={},
        hardware_opts=None,
        calc=MagicMock(),
        n_samples=20,
        score_chunk_size=20,
        rmsd_threshold=0.0,
    )
    assert len(centroid_ids) > 0
    assert len(centroid_ids) == len(energies)
    assert energies == sorted(energies)
    assert mol.GetNumConformers() == len(centroid_ids)


def test_pool_b_zero_conformers_safe():
    """If sample_constrained_confs adds no conformers (all infeasible), the
    contract is (mol, [], []) — no crash, no MACE call, no MMFF call."""

    def _embed_nothing(
        mol, grids, n_samples, n_attempts, tolerance_deg, strategy, seed
    ):
        return []

    mock_mace = MagicMock()
    mock_mmff = MagicMock()
    with (
        patch("confsweeper.sample_constrained_confs", side_effect=_embed_nothing),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            side_effect=mock_mmff,
        ),
    ):
        mol, centroid_ids, energies = get_mol_PE_pool_b(
            TEST_SMILES,
            grids={},
            hardware_opts=None,
            calc=MagicMock(),
            n_samples=20,
        )
    assert centroid_ids == []
    assert energies == []
    assert mol.GetNumConformers() == 0
    mock_mace.assert_not_called()
    mock_mmff.assert_not_called()


def test_pool_b_energy_filter_drops_high_energy(pool_b_mocks):
    """Conformers more than e_window_kT * kT above the minimum are dropped.
    Same arithmetic as the exhaustive test: with E_i = i * 0.01 eV and
    e_window_kT=2.0 the first 6 conformers (i = 0..5) survive."""
    n_samples = 20
    e_window_kT = 2.0
    expected_kept = int(np.floor(e_window_kT * _KT_EV_298K / 0.01)) + 1

    _, centroid_ids, energies = get_mol_PE_pool_b(
        TEST_SMILES,
        grids={},
        hardware_opts=None,
        calc=MagicMock(),
        n_samples=n_samples,
        score_chunk_size=n_samples,
        e_window_kT=e_window_kT,
        rmsd_threshold=0.0,
    )
    assert len(centroid_ids) == expected_kept
    assert max(energies) - min(energies) <= e_window_kT * _KT_EV_298K + 1e-12


def test_pool_b_minimize_gpu_invokes_batched_mmff(pool_b_mocks):
    """mmff_backend='gpu' calls MMFFOptimizeMoleculesConfs exactly once with
    the full conformer set."""
    n_samples = 6
    with patch(
        "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs", return_value=[[]]
    ) as mock_mmff:
        get_mol_PE_pool_b(
            TEST_SMILES,
            grids={},
            hardware_opts=None,
            calc=MagicMock(),
            n_samples=n_samples,
            score_chunk_size=n_samples,
            minimize=True,
            mmff_backend="gpu",
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    mock_mmff.assert_called_once()
    args, _ = mock_mmff.call_args
    assert len(args[0]) == 1
    assert args[0][0].GetNumConformers() == n_samples


def test_pool_b_minimize_cpu_invokes_per_conformer_mmff(pool_b_mocks):
    """mmff_backend='cpu' triggers AllChem.MMFFOptimizeMolecule once per conformer."""
    n_samples = 6
    with patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", return_value=0) as mock_mmff:
        get_mol_PE_pool_b(
            TEST_SMILES,
            grids={},
            hardware_opts=None,
            calc=MagicMock(),
            n_samples=n_samples,
            score_chunk_size=n_samples,
            minimize=True,
            mmff_backend="cpu",
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    assert mock_mmff.call_count == n_samples


def test_pool_b_minimize_unknown_backend_raises(pool_b_mocks):
    """An unrecognised mmff_backend value raises ValueError."""
    with pytest.raises(ValueError, match="unknown mmff_backend"):
        get_mol_PE_pool_b(
            TEST_SMILES,
            grids={},
            hardware_opts=None,
            calc=MagicMock(),
            n_samples=2,
            score_chunk_size=2,
            minimize=True,
            mmff_backend="something_else",
        )


def test_pool_b_forwards_strategy_and_n_attempts():
    """The function forwards strategy and n_attempts unchanged to
    sample_constrained_confs. Defaults are strategy='inverse' and
    n_attempts=1, both important for matched-budget benchmarking against
    get_mol_PE_exhaustive."""
    captured: dict = {}

    def _capturing_sample(mol, grids, **kwargs):
        captured.update(kwargs)
        # Add one conformer so the rest of the pipeline has something to score.
        AllChem.EmbedMolecule(mol, randomSeed=0)
        return [c.GetId() for c in mol.GetConformers()]

    with (
        patch("confsweeper.sample_constrained_confs", side_effect=_capturing_sample),
        patch(
            "confsweeper._mace_batch_energies",
            side_effect=lambda c, ms: [0.0] * len(ms),
        ),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
    ):
        get_mol_PE_pool_b(
            TEST_SMILES,
            grids={},
            hardware_opts=None,
            calc=MagicMock(),
            n_samples=5,
        )

    assert captured["strategy"] == "inverse"
    assert captured["n_attempts"] == 1
    assert captured["n_samples"] == 5
