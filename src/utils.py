import ase
import numpy as np
import torch
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

EV_TO_KCAL = 23.0609


def compare_geometries(
    mol_a: ase.Atoms,
    mol_b: ase.Atoms,
    mace_calc,
    rmsd_threshold: float = 0.125,
    energy_threshold: float = 6.0,
) -> tuple[bool, float, float]:
    """
    Compare two geometries of the same molecule using rigid-body aligned RMSD
    and MACE potential energy difference.

    Params:
        mol_a: ase.Atoms : first geometry
        mol_b: ase.Atoms : second geometry
        mace_calc: MACE calculator instance
        rmsd_threshold: float : max allowed RMSD in Angstroms (default 0.125)
        energy_threshold: float : max allowed energy difference in kcal/mol (default 6.0)
    Returns:
        tuple[bool, float, float] : (is_match, rmsd, energy_diff_kcal)
    """
    if len(mol_a) != len(mol_b):
        raise ValueError(
            f"Atom count mismatch: mol_a has {len(mol_a)} atoms, mol_b has {len(mol_b)}"
        )

    if not np.array_equal(mol_a.get_atomic_numbers(), mol_b.get_atomic_numbers()):
        raise ValueError("Atomic numbers do not match between mol_a and mol_b")

    # Rigid-body aligned RMSD via spyrmsd
    ref = Molecule(
        atomicnums=mol_a.get_atomic_numbers(),
        coordinates=mol_a.get_positions(),
    )
    comp = Molecule(
        atomicnums=mol_b.get_atomic_numbers(),
        coordinates=mol_b.get_positions(),
    )
    rmsd_val = rmsdwrapper(ref, comp, minimize=True, strip=False, symmetry=False)[0]

    # MACE potential energies
    mol_a.calc = mace_calc
    energy_a = mol_a.get_potential_energy()
    mol_a.calc = None
    torch.cuda.empty_cache()

    mol_b.calc = mace_calc
    energy_b = mol_b.get_potential_energy()
    mol_b.calc = None
    torch.cuda.empty_cache()

    energy_diff_kcal = abs(energy_a - energy_b) * EV_TO_KCAL

    is_match = rmsd_val <= rmsd_threshold and energy_diff_kcal <= energy_threshold
    return is_match, rmsd_val, energy_diff_kcal
