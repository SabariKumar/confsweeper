"""Seeding probe — experiment A: can the randomized exhaustive-ETKDG pipeline
DISCOVER the dominant cremp_sharp ceiling basin (conf0) de novo?

Runs get_mol_PE_exhaustive(n_seeds=10000) on cremp_sharp, then reports the
nearest-basin heavy-atom RMSD to conf0 and the energy gap. If a basin lands
near conf0 (RMSD < ~1 Å) and at conf0's energy, ETKDG can reach the fold and the
MCMM gap is a seeding problem (just feed it a richer pool). If not, the fold is
ETKDG-unreachable and the gap is deep.
"""

import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from confsweeper import (
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE_exhaustive,
)

CREMP_SHARP = (
    "C[C@H]1C(=O)N(C)[C@@H](CC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)"
    "C(=O)N[C@@H](CC(N)=O)C(=O)N(C)[C@@H](Cc2c[nH]c3ccccc23)C(=O)N1C"
)
CEILING = "results/cremp_ceiling_sdfs/S.S.N.MeW.MeA.MeN.sdf"


def main():
    calc = get_mace_calc()
    hw = get_hardware_opts()
    params = get_embed_params_macrocycle()

    conf0 = list(Chem.SDMolSupplier(CEILING, removeHs=False))[0]
    e_conf0 = float(conf0.GetProp("MACE_ENERGY"))
    conf0_heavy = Chem.RemoveHs(copy.deepcopy(conf0))

    mol, centroid_ids, energies = get_mol_PE_exhaustive(
        CREMP_SHARP, params, hw, calc, n_seeds=10000, minimize=True, mmff_backend="gpu"
    )
    energies = np.array(energies)

    # nearest exhaustive basin to conf0 (heavy-atom RMSD), and energy gap
    rmsds = []
    single = Chem.Mol(mol)
    for cid in centroid_ids:
        probe = Chem.Mol(mol, False, cid)  # one-conformer copy
        rmsds.append(
            rdMolAlign.GetBestRMS(Chem.RemoveHs(copy.deepcopy(probe)), conf0_heavy)
        )
    rmsds = np.array(rmsds)
    j = int(rmsds.argmin())

    print("=== exhaustive-ETKDG de-novo reachability of conf0 (n_seeds=10000) ===")
    print(f"  n basins found: {len(centroid_ids)}")
    print(f"  exhaustive e_min:  {energies.min():.4f} eV")
    print(f"  conf0 (ceiling):   {e_conf0:.4f} eV")
    print(f"  e_min - conf0:     {(energies.min() - e_conf0)*1000:.0f} meV")
    print(
        f"  nearest basin to conf0: {rmsds[j]:.3f} Å  (its energy {energies[j]:.4f} eV)"
    )
    if rmsds[j] < 1.0:
        print("  >>> ETKDG REACHES conf0's fold → MCMM gap is a SEEDING problem")
    else:
        print("  >>> ETKDG does NOT reach conf0's fold → gap is deeper than seeding")


if __name__ == "__main__":
    main()
