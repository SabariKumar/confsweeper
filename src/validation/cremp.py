"""
CREMP dataset loader and symmetric RMSD utilities for coverage benchmarking.

Pickle structure (per molecule):
    rd_mol       : RDKit Mol with all unique conformers (Hs included)
    smiles       : canonical SMILES
    conformers   : list of dicts with energy/Boltzmann metadata per conformer set
    charge       : molecular charge
    uniqueconfs  : number of conformers in rd_mol
    lowestenergy : lowest GFN2-xTB energy (kcal/mol)
    ... (other summary stats matching summary.csv)
"""

import logging
import os
import pickle
import warnings
from typing import Iterator

import numpy as np
import pandas as pd
import rdkit
import torch
from rdkit import Chem
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def iter_validation_mols(
    subset_csv: os.PathLike | str,
    pickle_dir: os.PathLike | str,
) -> Iterator[tuple[str, str, rdkit.Chem.Mol, dict]]:
    """
    Iterates over the validation subset, loading each CREMP pickle on demand.

    Yields one molecule at a time so the full dataset is never held in memory.
    Missing or corrupt pickles are logged and skipped.

    Params:
        subset_csv  : path to validation_subset.csv (output of make_validation_sets_cremp.py)
        pickle_dir  : path to the CREMP pickle/ directory
    Yields:
        sequence    : dot-separated amino acid sequence (also the pickle stem)
        smiles      : canonical SMILES from the pickle
        rd_mol      : RDKit Mol with all unique conformers (Hs included, GFN2-xTB geometries)
        meta        : dict of scalar metadata (energies, Boltzmann weights, topology, atom_bin, etc.)
    """
    subset = pd.read_csv(subset_csv)
    pickle_dir = os.fspath(pickle_dir)

    for row in subset.itertuples(index=False):
        sequence = row.sequence
        pickle_path = os.path.join(pickle_dir, f"{sequence}.pickle")

        if not os.path.exists(pickle_path):
            logger.warning("Missing pickle for sequence %s, skipping", sequence)
            continue

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load pickle for %s: %s", sequence, e)
            continue

        rd_mol = data.get("rd_mol")
        if rd_mol is None or rd_mol.GetNumConformers() == 0:
            logger.warning("No conformers in pickle for %s, skipping", sequence)
            continue

        n_atoms_pickle = rd_mol.GetNumAtoms()
        n_atoms_smiles = Chem.AddHs(Chem.MolFromSmiles(row.smiles)).GetNumAtoms()
        if n_atoms_pickle != n_atoms_smiles:
            logger.warning(
                "Atom count mismatch for %s: pickle has %d, SMILES gives %d, skipping",
                sequence,
                n_atoms_pickle,
                n_atoms_smiles,
            )
            continue

        meta = {
            "sequence": sequence,
            "topology": row.topology,
            "atom_bin": row.atom_bin,
            "num_monomers": row.num_monomers,
            "num_heavy_atoms": row.num_heavy_atoms,
            "uniqueconfs": data["uniqueconfs"],
            "lowestenergy": data["lowestenergy"],
            "charge": data["charge"],
        }

        yield sequence, data["smiles"], rd_mol, meta


# ---------------------------------------------------------------------------
# RMSD utilities
# ---------------------------------------------------------------------------


def pairwise_rmsd_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Fast vectorised pairwise RMSD between two conformer sets.
    Does NOT account for rotation or symmetry — use as a pre-filter only.

    Params:
        a : (N, M, 3) tensor — N conformers, M atoms
        b : (K, M, 3) tensor — K conformers, M atoms
    Returns:
        (N, K) tensor of RMSDs
    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)  # (N, K, M, 3)
    return torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1), dim=-1))


def symmetric_rmsd(
    mol_a: rdkit.Chem.Mol,
    conf_id_a: int,
    mol_b: rdkit.Chem.Mol,
    conf_id_b: int,
) -> float:
    """
    Symmetry-aware, rotation-minimised RMSD between two conformers via spyrmsd.

    Uses graph automorphisms to find the atom permutation that minimises RMSD,
    correctly handling macrocycle ring traversal direction and equivalent substituents.

    Params:
        mol_a      : RDKit Mol containing conformer a
        conf_id_a  : conformer index in mol_a
        mol_b      : RDKit Mol containing conformer b
        conf_id_b  : conformer index in mol_b
    Returns:
        float : minimum symmetric RMSD in Angstroms
    """
    ref = Molecule.from_rdkit(Chem.Mol(mol_a, confId=conf_id_a))
    comp = Molecule.from_rdkit(Chem.Mol(mol_b, confId=conf_id_b))
    return float(rmsdwrapper(ref, comp, minimize=True, strip=False, symmetry=True)[0])


def calc_coverage(
    ref_mol: rdkit.Chem.Mol,
    gen_mol: rdkit.Chem.Mol,
    gen_conf_ids: list[int],
    rmsd_cutoff: float = 1.0,
    filter_factor: float = 2.0,
) -> tuple[float, list[float]]:
    """
    Calculates what fraction of CREMP reference conformers are covered by
    confsweeper-generated conformers.

    Strategy:
        1. Fast pre-filter: vectorised tensor RMSD (no rotation) at filter_factor × rmsd_cutoff
        2. Precise re-score: symmetric spyrmsd for candidate pairs only

    A reference conformer is "covered" if at least one generated conformer has
    symmetric RMSD <= rmsd_cutoff.

    Params:
        ref_mol      : CREMP RDKit Mol with reference conformers (Hs included)
        gen_mol      : confsweeper RDKit Mol with generated conformers (Hs included)
        gen_conf_ids : list of conformer IDs to use from gen_mol
        rmsd_cutoff  : coverage threshold in Angstroms (default 1.0)
        filter_factor: pre-filter uses rmsd_cutoff * filter_factor (default 2.0)
    Returns:
        coverage     : fraction of reference conformers covered (0.0–1.0)
        ref_min_rmsds: for each reference conformer, the minimum symmetric RMSD
                       to any generated conformer (useful for analysis)
    """
    ref_conf_ids = [c.GetId() for c in ref_mol.GetConformers()]
    n_ref = len(ref_conf_ids)
    n_gen = len(gen_conf_ids)

    if n_ref == 0 or n_gen == 0:
        return 0.0, []

    # Build coordinate tensors: (N, M, 3)
    ref_coords = torch.tensor(
        np.array([ref_mol.GetConformer(i).GetPositions() for i in ref_conf_ids]),
        dtype=torch.float32,
    )
    gen_coords = torch.tensor(
        np.array([gen_mol.GetConformer(i).GetPositions() for i in gen_conf_ids]),
        dtype=torch.float32,
    )

    # Center each conformer to its centroid so the pre-filter RMSD reflects
    # shape difference, not translational offset between independent coordinate frames.
    ref_coords = ref_coords - ref_coords.mean(dim=1, keepdim=True)
    gen_coords = gen_coords - gen_coords.mean(dim=1, keepdim=True)

    # Center each conformer to remove translational offset before pre-filtering.
    # ETKDG and GFN2-xTB place molecules in arbitrary frames; without this,
    # raw pairwise RMSDs are dominated by translation and every candidate fails.
    ref_coords = ref_coords - ref_coords.mean(dim=1, keepdim=True)
    gen_coords = gen_coords - gen_coords.mean(dim=1, keepdim=True)

    # Fast pre-filter: (n_ref, n_gen) RMSD matrix, rotation not yet minimised
    fast_rmsds = pairwise_rmsd_tensor(ref_coords, gen_coords)
    filter_thresh = rmsd_cutoff * filter_factor

    ref_min_rmsds = []
    n_covered = 0

    for i, ref_cid in enumerate(ref_conf_ids):
        # Indices of generated conformers that pass the pre-filter
        candidate_idxs = (
            (fast_rmsds[i] < filter_thresh).nonzero(as_tuple=True)[0].tolist()
        )

        if not candidate_idxs:
            ref_min_rmsds.append(float("inf"))
            continue

        # Precise symmetric RMSD for candidates only
        min_rmsd = float("inf")
        for j in candidate_idxs:
            gen_cid = gen_conf_ids[j]
            r = symmetric_rmsd(ref_mol, ref_cid, gen_mol, gen_cid)
            if r < min_rmsd:
                min_rmsd = r

        ref_min_rmsds.append(min_rmsd)
        if min_rmsd <= rmsd_cutoff:
            n_covered += 1

    coverage = n_covered / n_ref
    return coverage, ref_min_rmsds
