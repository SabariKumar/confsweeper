"""Unit tests for the exhaustive ETKDG sampling primitives in src/confsweeper.py."""

import numpy as np
import pytest
import torch

from confsweeper import _energy_ranked_dedup


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
