"""
Generate before/after macrocycle structures for the MCMM proposer cartoons,
to be ray-traced in PyMOL (replacing the hand-drawn SVG cartoons in
figures/20260619_mcmm_steps_overview.svg).

Loads one real CREMP conformer (cremp_sharp = S.S.N.MeW.MeA.MeN, which carries
an N-Me-Trp aromatic side chain) and writes four aligned PDBs sharing one atom
ordering, so PyMOL can overlay start (ghost) against each moved state:

  - start.pdb : the unmodified conformer (same for all three proposers)
  - dbt.pdb   : one DBT concerted backbone rotation (ring stays closed) via
                src/concerted_rotation
  - dih.pdb   : one aromatic side-chain dihedral rotated to a new rotamer
  - cart.pdb  : every atom perturbed by an isotropic Gaussian kick

Also writes selection.json recording the moved-atom indices and the backbone
window, so the render script can colour / arrow the moved region.

Usage:
    pixi run python scripts/mcmm_pymol_structures.py --out_dir results/mcmm_pymol
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolTransforms  # noqa: E402

import concerted_rotation as cr  # noqa: E402
from mcmm import enumerate_backbone_windows  # noqa: E402
from proposers import (  # noqa: E402
    _backbone_atom_set,
    _compute_window_downstream_sets,
    _enumerate_side_chain_dihedrals,
)


def _write_pdb(mol: Chem.Mol, coords: np.ndarray, path: Path) -> None:
    """
    Write a single-conformer PDB with the given coordinates.

    Params:
        mol: Chem.Mol : template molecule (atom ordering preserved)
        coords: np.ndarray (n, 3) : coordinates to stamp onto the conformer
        path: Path : output PDB path
    Returns:
        None
    """
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(float(v) for v in coords[i]))
    Chem.MolToPDBFile(m, str(path))


def _dbt_move(mol: Chem.Mol, coords: np.ndarray, seed: int) -> tuple:
    """
    Apply one successful DBT concerted backbone rotation.

    Params:
        mol: Chem.Mol : molecule
        coords: np.ndarray (n, 3) : starting coordinates
        seed: int : RNG seed for window / drive choice
    Returns:
        tuple : (new_coords (n,3), window tuple, moved-atom index list)
    """
    rng = np.random.default_rng(seed)
    windows = enumerate_backbone_windows(mol)
    bb_set = _backbone_atom_set(mol)
    # try windows/drives until closure succeeds with a visible drive
    for _ in range(200):
        window = tuple(windows[rng.integers(len(windows))])
        drive_idx = int(rng.integers(cr.N_DIHEDRALS))
        drive_delta = float(rng.choice([-1, 1])) * rng.uniform(0.35, 0.55)
        pos7 = coords[list(window)]
        prop = cr.propose_move(pos7, drive_idx, drive_delta)
        if not prop.success:
            continue
        downstream = _compute_window_downstream_sets(mol, window, bb_set)
        new_coords = cr.apply_dihedral_changes_full_mol(
            coords, window, prop.deltas, downstream
        )
        moved = sorted(set().union(*[set(s) for s in downstream]))
        # require a visibly large move so the cartoon reads
        if np.max(np.linalg.norm(new_coords - coords, axis=1)) > 0.8:
            return new_coords, window, moved
    raise RuntimeError("no visible DBT closure found")


def _dihedral_move(mol: Chem.Mol, coords: np.ndarray) -> tuple:
    """
    Rotate one aromatic side-chain dihedral to a new rotamer.

    Params:
        mol: Chem.Mol : molecule
        coords: np.ndarray (n, 3) : starting coordinates
    Returns:
        tuple : (new_coords (n,3), (a,b,c,d) dihedral atoms, moved-atom list)
    """
    dihedrals = _enumerate_side_chain_dihedrals(mol)
    # prefer a dihedral whose pivot atom c is aromatic (the Trp χ case)
    chosen = None
    for (a, b, c, d) in dihedrals:
        if mol.GetAtomWithIdx(c).GetIsAromatic():
            chosen = (a, b, c, d)
            break
    if chosen is None:
        chosen = dihedrals[0]
    a, b, c, d = chosen

    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(float(v) for v in coords[i]))
    cur = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
    rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, cur + 110.0)
    new_coords = conf.GetPositions()
    moved = [
        i
        for i in range(m.GetNumAtoms())
        if np.linalg.norm(new_coords[i] - coords[i]) > 1e-3
    ]
    return new_coords, chosen, moved


def _cartesian_move(coords: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply an isotropic Gaussian kick to every atom.

    Params:
        coords: np.ndarray (n, 3) : starting coordinates
        seed: int : RNG seed
    Returns:
        np.ndarray (n, 3) : perturbed coordinates
    """
    rng = np.random.default_rng(seed)
    return coords + rng.normal(0.0, 0.30, size=coords.shape)


@click.command()
@click.option(
    "--sdf",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/cremp_ceiling_sdfs/S.S.N.MeW.MeA.MeN.sdf"),
    show_default=True,
)
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    default=Path("results/mcmm_pymol"),
    show_default=True,
)
@click.option("--seed", type=int, default=7, show_default=True)
def main(sdf: Path, out_dir: Path, seed: int) -> None:
    """
    Build and write the four aligned PDBs plus a selection-index JSON.

    Params:
        sdf: Path : input CREMP conformer SDF
        out_dir: Path : output directory
        seed: int : RNG seed for the DBT / Cartesian moves
    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    coords = mol.GetConformer().GetPositions()

    dbt_coords, window, dbt_moved = _dbt_move(mol, coords, seed)
    dih_coords, dih_atoms, dih_moved = _dihedral_move(mol, coords)
    cart_coords = _cartesian_move(coords, seed)

    _write_pdb(mol, coords, out_dir / "start.pdb")
    _write_pdb(mol, dbt_coords, out_dir / "dbt.pdb")
    _write_pdb(mol, dih_coords, out_dir / "dih.pdb")
    _write_pdb(mol, cart_coords, out_dir / "cart.pdb")

    sel = {
        "n_atoms": int(mol.GetNumAtoms()),
        "backbone_window": [int(i) for i in window],
        "dbt_moved": [int(i) for i in dbt_moved],
        "dih_dihedral": [int(i) for i in dih_atoms],
        "dih_moved": [int(i) for i in dih_moved],
        "dbt_max_disp": float(np.max(np.linalg.norm(dbt_coords - coords, axis=1))),
        "dih_max_disp": float(np.max(np.linalg.norm(dih_coords - coords, axis=1))),
    }
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=2))
    click.echo(f"wrote 4 PDBs + selection.json to {out_dir}")
    click.echo(f"  backbone window: {window}")
    click.echo(
        f"  DBT moved atoms: {len(dbt_moved)}  max disp {sel['dbt_max_disp']:.2f} A"
    )
    click.echo(
        f"  dihedral {dih_atoms} aromatic-pivot; moved {len(dih_moved)} atoms, "
        f"max disp {sel['dih_max_disp']:.2f} A"
    )


if __name__ == "__main__":
    main()
