"""Seeding probe — experiment B: does seeding the MC walk directly from the
dominant ceiling basin (conf0) close the coverage gap?

Injects conf0's geometry as the MCMM seed (monkeypatching the ETKDG embed so
get_mol_PE_mcmm starts its walkers at conf0), runs a short walk, dumps the basin
set, and reports coverage vs the CREMP ceiling. If cov_bw_ceil jumps from 0.000
to >= 0.724, direct CREMP seeding fixes the gap (conf0 is retained as a basin).
"""

import copy

import numpy as np
from rdkit import Chem

import confsweeper
from confsweeper import (
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE_mcmm,
)

CREMP_SHARP = (
    "C[C@H]1C(=O)N(C)[C@@H](CC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)"
    "C(=O)N[C@@H](CC(N)=O)C(=O)N(C)[C@@H](Cc2c[nH]c3ccccc23)C(=O)N1C"
)
CEILING = "results/cremp_ceiling_sdfs/S.S.N.MeW.MeA.MeN.sdf"

conf0 = list(Chem.SDMolSupplier(CEILING, removeHs=False))[0]


def _conf0_coords_in(mol):
    """Map conf0's coordinates onto `mol`'s atom order via substructure match."""
    match = mol.GetSubstructMatch(conf0)  # mol atom for each conf0 atom
    if len(match) != mol.GetNumAtoms():
        raise RuntimeError(f"atom map incomplete: {len(match)}/{mol.GetNumAtoms()}")
    pos = conf0.GetConformer().GetPositions()
    coords = np.zeros((mol.GetNumAtoms(), 3))
    for conf0_idx, mol_idx in enumerate(match):
        coords[mol_idx] = pos[conf0_idx]
    return coords


def _fake_embed(mols, params, confsPerMolecule, hardwareOptions):
    """Drop-in for embed.EmbedMolecules: seed each mol with conf0's geometry
    instead of a fresh ETKDG embed."""
    for m in mols:
        coords = _conf0_coords_in(m)
        conf = Chem.Conformer(m.GetNumAtoms())
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(
                i, (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
            )
        m.AddConformer(conf, assignId=True)


def main():
    calc = get_mace_calc()
    hw = get_hardware_opts()
    params = get_embed_params_macrocycle()

    orig = confsweeper.embed.EmbedMolecules
    confsweeper.embed.EmbedMolecules = _fake_embed
    try:
        mol, conf_ids, energies = get_mol_PE_mcmm(
            CREMP_SHARP,
            params,
            hw,
            calc,
            n_walkers_per_temp=8,
            n_temperatures=8,
            n_steps=30,
            n_init_confs=1,
            dedup_mode="kabsch",
            cartesian_weight=0.33,
            dihedral_weight=0.33,
            aromatic_wells_deg=(-90.0, 0.0, 90.0, 180.0),
            skip_mmff_relax=True,
        )
    finally:
        confsweeper.embed.EmbedMolecules = orig

    energies = np.array(energies)
    e_conf0 = float(conf0.GetProp("MACE_ENERGY"))
    # write basins to an SDF for union_basin_count
    out = "results/seed_probe_conf0_basins.sdf"
    w = Chem.SDWriter(out)
    for cid, e in zip(conf_ids, energies):
        m1 = Chem.Mol(mol, False, cid)
        m1.SetProp("MACE_ENERGY", str(float(e)))
        m1.SetProp("id", "S.S.N.MeW.MeA.MeN")
        w.write(m1)
    w.close()
    print(
        f"seeded-from-conf0 run: {len(conf_ids)} basins, e_min={energies.min():.4f} eV"
    )
    print(
        f"conf0 energy={e_conf0:.4f} eV  | e_min-conf0={(energies.min()-e_conf0)*1000:.0f} meV"
    )
    print(f"basins written to {out}")


if __name__ == "__main__":
    main()
