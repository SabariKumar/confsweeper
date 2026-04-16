# Performs benchmarking on geom-drugs to find the number of
# conformers recovered using confsweeper for different n_confs.

import json
import pickle
import sys
import warnings
from collections import Counter
from copy import deepcopy

import click
import numpy as np
import pandas as pd
import rdkit
import spyrmsd
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors
from spyrmsd import io, rmsd
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper
from tqdm import tqdm

from confsweeper import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_max_rotatable_bonds(
    n_smis: int = 10000,
    geom_base_dir: str
    | os.PathLike = "/mnt/data/sabari/geom-drugs/geom-drugs/rdkit_folder",
    dump_pickles=False,
):
    """
    TODO: Remove duplicate code!
    Finds the top n_smis smiles with the most rotatable bonds from the geom summary files.
    Params:
        n_smis: int : number of smiles to test for each summary json
        geom_base_dir : str|os.PathLike : path to geom dataset's rdkit-folder
        dump_pickles : bool : save the max rotatable bond geom data objects as pickles
    Returns:
        dict: dict of length n_smis, with drugs smiles as keys and number of rotatable bonds as values
        dict: dict of length n_smis, with qm9 smiles as keys and number of rotatable bonds as values
    """
    drugs_json = "../../data/raw/geom-drugs/summary_drugs.json"
    qm9_json = ".././data/raw/geom-drugs/summary_qm9.json"

    def get_rotatable_bonds(smi: str):
        # Returns number of rotatable bonds for a given smiles string
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.CalcNumRotatableBonds(mol)

    with open(drugs_json, "r") as infile:
        drugs_summary = drugs_json.load(infile)

    with open(qm9_json, "r") as infile:
        qm9_summary = qm9_json.load(infile)

    drugs_rot_bonds = {}
    for smi in tqdm(drugs_summary.keys()):
        drugs_rot_bonds[smi] = get_rotatable_bonds(smi)

    qm9_rot_bonds = {}
    for smi in tqdm(qm9_summary.keys()):
        qm9_rot_bonds[smi] = get_rotatable_bonds(smi)

    drugs_max_rotatable = Counter(drugs_rot_bonds).most_common(n_smis * 5)
    qm9_max_rotatable = Counter(qm9_rot_bonds).most_common(n_smis * 5)

    drugs_selected_rotatable = {}
    drugs_selected_pickles = {}
    for ind, smi in tqdm([(ind, x[0]) for ind, x in enumerate(drugs_max_rotatable)]):
        if len(drugs_selected_rotatable) < n_smis:
            path = drugs_summary[smi].get("pickle_path", None)
            if path is not None:
                path = os.path.join(geom_base_dir, path)
                drugs_selected_rotatable[smi] = drugs_max_rotatable[ind][1]
                with open(path, "rb") as picklefile:
                    drugs_selected_pickles[smi] = pickle.load(picklefile)

    for smi, n_rot in drugs_selected_rotatable.items():
        drugs_selected_pickles[smi]["rotatablebonds"] = n_rot

    qm9_selected_rotatable = {}
    qm9_selected_pickles = {}
    for ind, smi in tqdm([(ind, x[0]) for ind, x in enumerate(qm9_max_rotatable)]):
        if len(qm9_selected_rotatable) < n_smis:
            path = qm9_summary[smi].get("pickle_path", None)
            if path is not None:
                path = os.path.join(geom_base_dir, path)
                qm9_selected_rotatable[smi] = qm9_max_rotatable[ind][1]
                with open(path, "rb") as picklefile:
                    qm9_selected_pickles[smi] = pickle.load(picklefile)

    for smi, n_rot in qm9_selected_rotatable.items():
        qm9_selected_pickles[smi]["rotatablebonds"] = n_rot

    if dump_pickles:
        with open(
            "../data/processed/geom-drugs/drugs_maxrotatable.pickle", "wb"
        ) as outfile:
            pickle.dump(drugs_selected_pickles, outfile)
        with open(
            "../data/processed/geom-drugs/qm9_maxrotatable.pickle", "wb"
        ) as outfile:
            pickle.dump(qm9_selected_pickles, outfile)

    return drugs_selected_pickles, qm9_selected_pickles


def pairwise_rmsd(a: torch.Tensor, b: torch.Tensor):
    """
    Calculate pairwise RMSDs between two tensors; for an (A,M,3) and a (B,M,3) tensor,
    this returns a (A,B) dimensional tensor
    Params:
        a: torch.Tensor : 1st input tensor
        b: torch.Tensor : 2nd input tensor
    Returns:
        torch.Tensor : symmetric pairwise distance matrix
    """
    x = a.unsqueeze(1) - b.unsqueeze(0)
    x = torch.sum(x**2, axis=-1)
    return torch.sqrt(torch.mean(x, axis=-1))


def calc_coverage(
    selected_pickle: dict,
    n_confs: int = 1000,
    butina_thresh: float = 0.1,
    cutoff_rmsd: float = 0.05,
    use_spyrmsd: bool = True,
) -> Tuple:

    """
    Performs conformer coverage calculation for drugs and qm9 datasets.
    Params:
        n_confs: int : Number of random conformers to use in confsweeper
        butina_thres: float : Confsweeper butina clustering threshold
        cutoff_dist: float : Cutoff RMSD for determining same conformer, in A
        drugs_pickle: str|os.PathLike|None : Path to processed (subsampled) drugs pickle file
        qm9_pickle: str|os.PathLike|None : Path to processed (subsampled) qm9 pickle file
        generate_max_rotatable: bool : Regenerate subsampled dataset pickle files
    Returns:
        List: percent coverage per molecule
        List:
    """

    params = get_embed_params()
    hardware_opts = get_hardware_opts(
        preprocessingThreads=8,
        batch_size=1000,
        batchesPerGpu=2,
    )
    macemp = get_mace_calc()
    coverages = []
    multiple_matches = []
    for smi in tqdm(selected_pickle.keys()):
        mol, conf_ids, pe = get_mol_PE(
            smi=smi,
            params=params,
            hardware_opts=hardware_opts,
            mace_calc=macemp,
            n_confs=n_confs,
            cutoff_dist=butina_thresh,
            gpu_clustering=True,
        )
        # Check if each conformer is within 0.5 A rmsd to a conf in drugs_selected_pickle
        matches = []
        rmsds = []
        for dat in selected_pickle[smi]["conformers"]:
            geom_cid = dat["rd_mol"].GetConformer().GetId()
            mol_ = deepcopy(mol)
            match = 0
            for cid in conf_ids:
                if use_spyrmsd:
                    ref = Molecule.from_rdkit(Chem.Mol(mol_, confId=cid))
                    comp = Molecule.from_rdkit(dat["rd_mol"])
                    rmsd = rmsdwrapper(ref, comp)[0]
                else:
                    rmsd = rdMolAlign.AlignMol(mol_, dat["rd_mol"], cid, geom_cid)
                rmsds.append(rmsd)
                if rmsd < cutoff_rmsd:
                    match += 1
            matches.append(match)
        matches = torch.tensor(matches)
        matches_counts = torch.bincount(matches)
        if len(matches_counts) > 1:
            coverage = matches_counts[1] / len(matches)
        else:
            coverage = 0
        coverages.append(coverage)
        if matches_counts[2:].any():
            multiple_matches.append(True)
        else:
            multiple_matches.append(False)

        # geom_coords = torch.tensor(
        #     np.array(
        #         [
        #             _["rd_mol"].GetConformer().GetPositions()
        #             for _ in list(selected_pickle.values())[0]["conformers"]
        #         ]
        #     )
        # )
        # cs_pos = []
        # for conf in conf_ids:
        #     cs_pos.append(mol.GetConformer(conf).GetPositions())
        # cs_pos = torch.tensor(np.array(cs_pos))
        # rmsds = pairwise_rmsd(cs_pos, geom_coords)
        # # TODO: Check whether multiple confsweeper confs match one geom -> adjust Butina thresh!
        # # Just see if multiple True on geom axis
        # # Count number of zeros, ones, and more than ones - any more than ones means that
        # # butina threshold is probably too high.
        # matches = rmsds < cutoff_dist
        # matches_ = matches.sum(axis=1).int()
        # matches_counts = torch.bincount(matches_)
        # coverage = matches_counts[1] / len(matches_)
        # coverages.append(coverage)
        # if matches_counts[2:].any():
        #     multiple_matches.append(True)
        # else:
        #     multiple_matches.append(False)

    return coverage, multiple_matches


@click.command()
@click.option("--drugs_pickle")
@click.option("--qm9_pickle")
@click.option("--generate_max_rotatable")
def calc_defaults_coverage(
    drugs_pickle: str
    | os.PathLike
    | None = "../data/processed/geom-drugs/drugs_maxrotatable.pickle",
    qm9_pickle: str | os.PathLike | None = None,
    generate_max_rotatable: bool = False,
):
    """
    Calculates conformer coverage
    """
    if generate_max_rotatable:
        drugs_selected_pickles, qm9_selected_pickles = get_max_rotatable_bonds()
    else:
        if drugs_pickle:
            with open(drugs_pickle, "rb") as infile:
                drugs_selected_pickles = pickle.load(infile)
        else:
            drugs_selected_pickles = None
        if qm9_pickle:
            with open(qm9_pickle, "rb") as infile:
                qm9_selected_pickles = pickle.load(infile)
        else:
            qm9_selected_pickles = None
    if drugs_selected_pickles:
        drugs_coverage, drugs_multiple = calc_coverage(drugs_selected_pickles)
        pd.DataFrame(
            {
                "smis": drugs_selected_pickles.keys(),
                "coverage": drugs_coverage,
                "multiple": drugs_multiple,
            }
        ).to_csv("./drugs_coverage.csv", index=False)
    if qm9_selected_pickles:
        qm9_coverage, qm9_multiple = calc_coverage(qm9_selected_pickles)
        pd.DataFrame(
            {
                "smis": qm9_selected_pickles.keys(),
                "coverage": qm9_coverage,
                "multiple": qm9_multiple,
            }
        ).to_csv("./qm9_coverage.csv", index=False)


if __name__ == "__main__":
    calc_defaults_coverage()
