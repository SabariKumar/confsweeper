"""
Quick analysis of dumped basin SDFs: pairwise heavy-atom RMSD and energy
spread. Tells us whether the conformers a sampler returned are truly
distinct minima (trap hypothesis) or near-duplicates that escaped dedup
(dedup-threshold hypothesis).

Usage:
    pixi run -e mace python scripts/analyze_basin_sdf.py <sdf_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def heavy_atom_rmsd(mol: Chem.Mol, i: int, j: int) -> float:
    """
    Best-fit heavy-atom RMSD between conformers i and j of mol.

    Params:
        mol: Chem.Mol : mol with multiple conformers, all sharing the
            atom ordering of the same SMILES
        i: int : conformer index for reference
        j: int : conformer index for probe
    Returns:
        float : RMSD in Angstroms
    """
    return AllChem.GetBestRMS(mol, mol, prbId=j, refId=i)


def _energy_ranked_dedup_at(
    rmsd: np.ndarray, energies: np.ndarray, threshold: float
) -> list[int]:
    """
    Reproduce `_energy_ranked_dedup`'s logic on an already-computed RMSD
    matrix. Used here to retro-score existing SDFs at multiple thresholds
    without rerunning the sampler.
    """
    n = rmsd.shape[0]
    if n == 0:
        return []
    order = np.argsort(energies, kind="stable")
    excluded = np.zeros(n, dtype=bool)
    centroids: list[int] = []
    for idx in order.tolist():
        if excluded[idx]:
            continue
        centroids.append(idx)
        excluded |= rmsd[idx] < threshold
    return centroids


def main(sdf_path: Path) -> None:
    """
    Load all confs from an SDF, report energies, pairwise heavy-atom Kabsch
    RMSD, and the corrected n_basins under several thresholds.

    Params:
        sdf_path: Path : path to SDF written by sampler_benchmark.py
    Returns:
        None : prints summary to stdout
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    confs = [m for m in suppl if m is not None]
    if not confs:
        print(f"No conformers in {sdf_path}")
        return

    template = Chem.Mol(confs[0])
    template.RemoveAllConformers()
    for c in confs:
        template.AddConformer(c.GetConformer(), assignId=True)

    energies = []
    for c in confs:
        if c.HasProp("MACE_ENERGY"):
            energies.append(float(c.GetProp("MACE_ENERGY")))
        else:
            energies.append(float("nan"))
    energies = np.array(energies)

    n = template.GetNumConformers()
    rmsd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = heavy_atom_rmsd(template, i, j)
            rmsd[i, j] = r
            rmsd[j, i] = r

    print(f"\n=== {sdf_path.name} ===")
    print(f"n_confs: {n}")
    print(f"e range: {energies.min():.3f} ... {energies.max():.3f} eV")
    print(
        f"e spread: {energies.max() - energies.min():.4f} eV "
        f"({(energies.max() - energies.min()) / 0.0257:.1f} kT_298K)"
    )
    if n >= 2:
        triu = np.triu_indices(n, k=1)
        print(
            f"rmsd: min={rmsd[triu].min():.3f} "
            f"median={np.median(rmsd[triu]):.3f} "
            f"max={rmsd[triu].max():.3f} A"
        )
        print("\ncorrected n_basins under Kabsch heavy-atom RMSD dedup:")
        for thresh in (0.125, 0.5, 1.0):
            centroids = _energy_ranked_dedup_at(rmsd, energies, thresh)
            print(f"  threshold={thresh:.3f} A → n_basins={len(centroids)}")
    print(f"\nenergies (eV, sorted):")
    for k, idx in enumerate(np.argsort(energies)):
        de_kt = (energies[idx] - energies.min()) / 0.0257
        print(
            f"  rank {k:2d}: conf {idx:2d}  E={energies[idx]:.4f}  "
            f"ΔE={energies[idx] - energies.min():.4f} eV  ({de_kt:.1f} kT)"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
