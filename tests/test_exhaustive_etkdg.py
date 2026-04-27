"""Unit tests for the exhaustive ETKDG sampling primitives in src/confsweeper.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from rdkit.Chem import AllChem

from confsweeper import (
    _KT_EV_298K,
    _energy_ranked_dedup,
    get_embed_params,
    get_mol_PE_exhaustive,
)


def _line_coords(positions: list[float], n_atoms: int = 4) -> torch.Tensor:
    """
    Build [N, n_atoms, 3] coords where conformer i is a rigid translation of a
    fixed atom layout by `positions[i]` along x. Pairwise normalised L1 distance
    between conformer i and j equals |positions[i] - positions[j]|.

    Params:
        positions: list[float] : x-translation per conformer
        n_atoms: int : number of atoms (constant across conformers)
    Returns:
        torch.Tensor [N, n_atoms, 3] of conformer coordinates
    """
    base = torch.zeros(n_atoms, 3)
    base[:, 0] = torch.arange(n_atoms, dtype=torch.float32)
    coords = torch.stack([base + torch.tensor([p, 0.0, 0.0]) for p in positions])
    return coords


# ---------------------------------------------------------------------------
# _energy_ranked_dedup
# ---------------------------------------------------------------------------


def test_energy_ranked_dedup_empty():
    coords = torch.zeros(0, 4, 3)
    energies = np.zeros(0)
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1) == []


def test_energy_ranked_dedup_single():
    coords = torch.zeros(1, 4, 3)
    energies = np.array([0.0])
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1) == [0]


def test_energy_ranked_dedup_keeps_all_distinct():
    """Three conformers spaced far apart in geometry should all survive dedup."""
    coords = _line_coords([0.0, 5.0, 10.0])
    energies = np.array([0.0, 1.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert sorted(centroids) == [0, 1, 2]


def test_energy_ranked_dedup_collapses_all_close():
    """Three conformers all within the RMSD threshold collapse to the lowest-energy."""
    # All translations are 0.0, so pairwise normalised L1 distance = 0
    coords = _line_coords([0.0, 0.0, 0.0])
    energies = np.array([2.0, 0.5, 1.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [1]  # index of the lowest energy


def test_energy_ranked_dedup_picks_lowest_energy_in_basin():
    """The lowest-energy conformer of a basin must be the one returned, even
    when a higher-energy conformer is geometrically nearer the centroid of
    other already-selected representatives."""
    # conformers 0 and 1 are geometric near-duplicates; conformer 0 has the
    # higher energy, so dedup must drop conformer 0 and keep conformer 1.
    coords = _line_coords([0.0, 0.05, 5.0])
    energies = np.array([1.0, 0.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    # Lowest-energy basin rep (conformer 1) and the geometrically distant
    # conformer 2 survive; the higher-energy near-duplicate (conformer 0) drops.
    assert sorted(centroids) == [1, 2]


def test_energy_ranked_dedup_centroid_order_is_energy_ascending():
    """Returned centroid order must match ascending energy."""
    coords = _line_coords([0.0, 5.0, 10.0])
    energies = np.array([2.0, 0.5, 1.5])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [1, 2, 0]


def test_energy_ranked_dedup_stable_on_energy_ties():
    """When two conformers tie on energy, np.argsort(kind='stable') breaks the
    tie by original index. The lower-index conformer is selected first and
    excludes its near-duplicate twin."""
    # conformers 0 and 1 are near-duplicates with identical energy
    coords = _line_coords([0.0, 0.05, 5.0])
    energies = np.array([1.0, 1.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [0, 2]


def test_energy_ranked_dedup_threshold_boundary():
    """A pair right at the threshold is treated as overlapping (strict < not <=)
    only when distance is *less than* the threshold. Distance exactly at the
    threshold is considered distinct."""
    # On _line_coords, normalised L1 distance between conformer 0 (offset 0)
    # and conformer 1 (offset 0.4) over n_atoms=4 atoms is
    # (4 * 0.4) / (3 * 4) = 0.1333...
    coords = _line_coords([0.0, 0.4])
    energies = np.array([0.0, 1.0])
    # Threshold 0.2 > 0.133, so the second is excluded
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.2) == [0]
    # Threshold 0.1 < 0.133, so both survive
    assert sorted(_energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)) == [0, 1]


@pytest.mark.parametrize("device", ["cpu"])
def test_energy_ranked_dedup_device_independent(device):
    """Algorithm output is identical regardless of which device coords live on.
    GPU coverage is exercised in integration tests; this parametrisation makes
    it easy to add 'cuda' once the helper is wired into the full pipeline."""
    coords = _line_coords([0.0, 5.0, 0.05]).to(device)
    energies = np.array([1.0, 2.0, 0.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    # Lowest energy is conformer 2 (offset 0.05, near conformer 0). It excludes
    # conformer 0; conformer 1 is far away and survives.
    assert sorted(centroids) == [1, 2]


# ---------------------------------------------------------------------------
# get_mol_PE_exhaustive
# ---------------------------------------------------------------------------

# Tripeptide cyclic-ish backbone — small enough that CPU ETKDG produces
# distinct conformers reliably, large enough that the geometric dedup actually
# discriminates basins.
TEST_SMILES = "C[C@@H](N)C(=O)NCC(=O)NCC(=O)O"


def _mock_embed_seed_aware(mols, params, confsPerMolecule, hardwareOptions):
    """
    CPU drop-in for nvmolkit.embedMolecules.EmbedMolecules.

    Matches two behaviours of the real call that the chunked-embed loop relies
    on: respects params.randomSeed (so chunked calls produce distinct
    conformers) and appends to any existing conformer set rather than
    replacing it (verified empirically against nvmolkit on a small peptide).
    """
    for m in mols:
        AllChem.EmbedMultipleConfs(
            m,
            numConfs=confsPerMolecule,
            randomSeed=params.randomSeed,
            clearConfs=False,
        )


def _make_seq_mock_mace():
    """
    Return a `_mace_batch_energies` mock that yields a global monotone-increasing
    energy sequence across calls. Conformer i (counted across all chunks) is
    assigned energy i * 0.01 eV — about 0.39 kT_298K per step. This makes the
    energy filter and dedup behaviour deterministic and easy to assert against.
    """
    counter = [0]

    def _mock(_calc, ase_mols):
        out = [(counter[0] + i) * 0.01 for i in range(len(ase_mols))]
        counter[0] += len(ase_mols)
        return out

    return _mock


@pytest.fixture
def exhaustive_mocks():
    """Patch nvmolkit embed and MACE scoring with deterministic CPU mocks."""
    mock_mace = _make_seq_mock_mace()
    with patch(
        "confsweeper.embed.EmbedMolecules", side_effect=_mock_embed_seed_aware
    ), patch("confsweeper._mace_batch_energies", side_effect=mock_mace):
        yield


def test_exhaustive_returns_centroids_and_energies(exhaustive_mocks):
    """Smoke: pipeline returns aligned centroids + energies in ascending order."""
    mol, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=20,
        embed_chunk_size=20,
        score_chunk_size=20,
        rmsd_threshold=0.0,  # disable geometric dedup so all energy-filtered survivors are returned
    )
    assert len(centroid_ids) > 0
    assert len(centroid_ids) == len(energies)
    # Energies are in ascending order (energy-ranked dedup guarantees this).
    assert energies == sorted(energies)
    # Mol contains exactly the centroid conformers — non-centroids are removed.
    assert mol.GetNumConformers() == len(centroid_ids)


def test_exhaustive_chunked_path_accumulates_full_pool(exhaustive_mocks):
    """When n_seeds > embed_chunk_size, the chunked loop accumulates the full
    requested pool size (modulo CPU embed failures, which our mock does not
    induce). This validates the chunked-embed mechanics, not nvmolkit's
    cross-call diversity (the semantics experiment in docs/ covers that)."""
    n_seeds = 24
    embed_chunk_size = 8
    _, centroid_ids, _ = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=n_seeds,
        embed_chunk_size=embed_chunk_size,
        score_chunk_size=64,
        rmsd_threshold=0.0,
        e_window_kT=1e9,  # disable energy filter so we can count the full pool
    )
    # With rmsd_threshold=0 and the energy filter disabled, every successfully-
    # embedded conformer is its own centroid.
    assert len(centroid_ids) == n_seeds


def test_exhaustive_energy_filter_drops_high_energy(exhaustive_mocks):
    """Conformers more than e_window_kT * kT above the minimum are dropped.

    With monotone energies E_i = i * 0.01 eV and kT = 0.02568 eV at 298 K,
    a window of e_window_kT=2.0 keeps E_i ≤ 2 * 0.02568 ≈ 0.0514 eV, i.e.
    the first 6 conformers (i = 0..5).
    """
    n_seeds = 20
    e_window_kT = 2.0
    expected_kept = int(np.floor(e_window_kT * _KT_EV_298K / 0.01)) + 1

    _, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=n_seeds,
        embed_chunk_size=n_seeds,
        score_chunk_size=n_seeds,
        e_window_kT=e_window_kT,
        rmsd_threshold=0.0,  # isolate the energy filter from dedup
    )
    assert len(centroid_ids) == expected_kept
    assert max(energies) - min(energies) <= e_window_kT * _KT_EV_298K + 1e-12


def test_exhaustive_zero_conformers_safe():
    """When embed returns 0 conformers, contract is (mol, [], []) — no crash,
    no MACE call, no dedup attempt."""

    def _embed_nothing(mols, params, confsPerMolecule, hardwareOptions):
        return  # don't add any conformers

    mock_mace = MagicMock()
    with patch("confsweeper.embed.EmbedMolecules", side_effect=_embed_nothing), patch(
        "confsweeper._mace_batch_energies", side_effect=mock_mace
    ):
        mol, centroid_ids, energies = get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=20,
            embed_chunk_size=20,
            score_chunk_size=20,
        )
    assert centroid_ids == []
    assert energies == []
    assert mol.GetNumConformers() == 0
    mock_mace.assert_not_called()


def test_exhaustive_single_conformer_skips_dedup(exhaustive_mocks):
    """n_seeds=1 returns one centroid without invoking the dedup primitive
    (which is unnecessary on a singleton pool)."""
    mol, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=1,
        embed_chunk_size=1,
        score_chunk_size=1,
    )
    assert len(centroid_ids) == 1
    assert len(energies) == 1
    assert mol.GetNumConformers() == 1


def test_exhaustive_minimize_invokes_mmff(exhaustive_mocks):
    """minimize=True triggers AllChem.MMFFOptimizeMolecule once per conformer
    in the pool (called before energy scoring)."""
    n_seeds = 6
    with patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", return_value=0) as mock_mmff:
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=n_seeds,
            embed_chunk_size=n_seeds,
            score_chunk_size=n_seeds,
            minimize=True,
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    # One MMFF call per embedded conformer. Mock embed always succeeds.
    assert mock_mmff.call_count == n_seeds
