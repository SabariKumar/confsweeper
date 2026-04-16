"""Unit tests for src/utils.py."""
from unittest.mock import MagicMock

import ase
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils import EV_TO_KCAL, compare_geometries

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _water(positions=None):
    """Create a water molecule as an ase.Atoms object."""
    default_positions = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]]
    return ase.Atoms(
        "OHH", positions=positions if positions is not None else default_positions
    )


def _mock_mace(energy_ev: float = -10.0):
    """Return a mock MACE calculator that yields a fixed energy."""
    calc = MagicMock()
    calc.get_potential_energy = MagicMock(return_value=energy_ev)
    return calc


# ---------------------------------------------------------------------------
# Tests: identical geometries
# ---------------------------------------------------------------------------


def test_identical_geometries():
    mol_a = _water()
    mol_b = _water()
    calc = _mock_mace(-10.0)

    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert is_match
    assert rmsd == pytest.approx(0.0, abs=1e-6)
    assert energy_diff == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: within thresholds
# ---------------------------------------------------------------------------


def test_small_perturbation_within_threshold():
    mol_a = _water()
    perturbed = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]])
    perturbed += np.random.default_rng(42).normal(scale=0.05, size=perturbed.shape)
    mol_b = _water(positions=perturbed)

    calc = _mock_mace(-10.0)
    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert rmsd < 0.125
    assert is_match


# ---------------------------------------------------------------------------
# Tests: exceeds thresholds
# ---------------------------------------------------------------------------


def test_large_perturbation_exceeds_rmsd():
    mol_a = _water()
    # Non-uniform perturbation so rigid-body alignment can't fix it
    perturbed = np.array([[0.5, -0.3, 0.2], [0.96, 0.4, -0.5], [0.24, 0.93, 0.6]])
    mol_b = _water(positions=perturbed)

    calc = _mock_mace(-10.0)
    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert rmsd > 0.125
    assert not is_match


def test_energy_exceeds_threshold():
    mol_a = _water()
    mol_b = _water()

    # 7 kcal/mol difference -> exceeds 6 kcal/mol threshold
    energy_diff_ev = 7.0 / EV_TO_KCAL

    # Use a calculator that returns different energies for each molecule
    energies = iter([-10.0, -10.0 + energy_diff_ev])
    calc = MagicMock()
    calc.get_potential_energy = MagicMock(side_effect=lambda atoms: next(energies))

    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert energy_diff > 6.0
    assert not is_match


# ---------------------------------------------------------------------------
# Tests: rigid-body alignment
# ---------------------------------------------------------------------------


def test_translated_molecule_matches():
    mol_a = _water()
    shifted = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]])
    shifted += [5.0, 3.0, -2.0]  # pure translation
    mol_b = _water(positions=shifted)

    calc = _mock_mace(-10.0)
    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert rmsd == pytest.approx(0.0, abs=1e-4)
    assert is_match


def test_rotated_molecule_matches():
    mol_a = _water()
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]])
    rot = Rotation.from_euler("z", 45, degrees=True)
    rotated = rot.apply(positions)
    mol_b = _water(positions=rotated)

    calc = _mock_mace(-10.0)
    is_match, rmsd, energy_diff = compare_geometries(mol_a, mol_b, calc)

    assert rmsd == pytest.approx(0.0, abs=1e-4)
    assert is_match


# ---------------------------------------------------------------------------
# Tests: validation errors
# ---------------------------------------------------------------------------


def test_mismatched_atom_count_raises():
    mol_a = _water()
    mol_b = ase.Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])

    calc = _mock_mace()
    with pytest.raises(ValueError, match="Atom count mismatch"):
        compare_geometries(mol_a, mol_b, calc)


def test_mismatched_atomic_numbers_raises():
    mol_a = _water()  # OHH
    mol_b = ase.Atoms(
        "NHH", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]]
    )

    calc = _mock_mace()
    with pytest.raises(ValueError, match="Atomic numbers do not match"):
        compare_geometries(mol_a, mol_b, calc)
