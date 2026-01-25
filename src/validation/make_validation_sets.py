# import sys
# sys.path.insert(1, '/home/sabari/confsweeper')
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

RANDOM_SEED = 42
rd = random.Random()
rd.seed(RANDOM_SEED)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 24})
plt.rcParams["svg.fonttype"] = "none"

SUMMARY_DICT = {
    "drugs": "/home/sabari/confsweeper/data/raw/geom-drugs/summary_drugs.json",
    "qm9": "/home/sabari/confsweeper/data/raw/geom-drugs/summary_qm9.json",
}


def get_rotatable_bonds(
    smi,
) -> np.array:
    """
    Gets number of rotatble bonds using rdkit built in.
    Params:
        smi: str : input smiles string

    Returns:
        np.array : number of rotatble bonds in input smiles
    """
    mol = Chem.MolFromSmiles(smi)
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def get_max_rotatable(
    summary_prefix: str,
    summary_dict: dict,
    make_plots: str
    | os.PathLike
    | None = "/home/sabari/confsweeper/data/processed/geom-drugs/plots",
    geom_drugs_dir: str
    | os.PathLike = "/mnt/data/sabari/geom-drugs/geom-drugs/rdkit_folder",
    validation_pickle_dir: str
    | os.PathLike = "/home/sabari/confsweeper/data/processed/geom-drugs/",
) -> None:
    """
    Use the max number of rotatble bonds to sample from the summary dict, save the number of rotatble bonds as
    a dict attribute, and write the result.
    Params:
        summary_prefix: str : dataset label from SUMMARY_DICT keys
        summary_dict: str : parsed geom summary dict from SUMMARY_DICT value
        make_plots: str|os.PathLike|None : if a path is provided, save dataset histograms to this directory
        geom_drugs_dir: str|os.PathLike : parent folder for geom-drugs pickles
        validation_pickle_dir: str|os.PathLike : where to save sampled pickle files
    Returns:
        None
    """
    rot_bonds = {}
    for smi in tqdm(summary_dict.keys()):
        rot_bonds[smi] = get_rotatable_bonds(smi)

    max_rotatable = Counter(rot_bonds).most_common(
        10000
    )  # Just sample the top 10,000 for now

    selected_rotatable = {}
    selected_pickles = {}
    for ind, smi in tqdm([(ind, x[0]) for ind, x in enumerate(max_rotatable)]):
        if len(selected_rotatable) < 1000:
            path = summary_dict[smi].get("pickle_path", None)
            if path is not None:
                path = os.path.join(geom_drugs_dir, path)
                selected_rotatable[smi] = max_rotatable[ind][1]
                with open(path, "rb") as picklefile:
                    selected_pickles[smi] = pickle.load(picklefile)

    for smi, n_rot in selected_rotatable.items():
        selected_pickles[smi]["rotatablebonds"] = n_rot

    # Sample 1000 random as well
    random_pickles = {}
    for key in selected_pickles.keys():
        del summary_dict[key]

    key_ = random.sample(list(summary_dict.keys()), 1000)
    val_ = [summary_dict[x] for x in key_]
    random_pickles = dict(zip(key_, val_))
    for smi in tqdm(random_pickles.keys()):
        random_pickles[smi]["rotatablebonds"] = get_rotatable_bonds(smi)

    with open(
        os.path.join(validation_pickle_dir, f"{summary_prefix}_maxrotatable.pickle"),
        "wb",
    ) as outfile:
        pickle.dump(selected_pickles, outfile)

    with open(
        os.path.join(validation_pickle_dir, f"{summary_prefix}_random.pickle"), "wb"
    ) as outfile:
        pickle.dump(random_pickles, outfile)

    if make_plots:
        for pre_, set_ in {"max_rotatable": selected_pickles, "random": random_pickles}:
            n_rot_ = []
            n_heavy_ats = []
            for smi in random_pickles.keys():
                n_rot_.append(set_[smi]["rotatablebonds"])
                n_heavy_ats.append(
                    rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(set_[smi]))
                )
            n_rot_ = np.array(n_rot_)
            plt.hist(n_rot_, bins="auto")
            plt.title("Number of Rotatable Bonds")
            plt.savefig(
                os.path.join(make_plots, f"{pre_}_rotatable_bond_distribution.svg")
            )
            plt.hist(n_heavy_ats, bins="auto")
            plt.title("Number of Heavy Atoms")
            plt.savefig(os.path.join(make_plots, f"{pre_}_heavy_atom_distribution.svg"))


def get_all_rot(summary_file_dict: dict) -> None:
    """
    Given a dict of {summary file label: summary file path}, do rotatable bond sampling and dump sample
    dict as pickle.
    Params:
        summary_file_dict: dict : dictionary of geom summary files and dataset labels
    """
    for key, val in summary_file_dict.items():
        with open(val, "r") as infile:
            summary_file = json.load(infile)
        get_max_rotatable(key, summary_file)


if __name__ == "__main__":
    get_all_rot(SUMMARY_DICT)
