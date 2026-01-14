# Performs benchmarking on geom-drugs to find the number of
# conformers recovered using confsweeper for different n_confs.

import json
from collections import Counter

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

from src.confsweeper import *


def get_max_rotatable_bonds(n_smis: int = 1000):
    """
    Finds the top n_smis smiles with the most rotatable bonds from the geom summary files.
    Params:
        n_smis: int : number of smiles to test for each summary json
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

    drugs_max_rotatable = Counter(drugs_rot_bonds).most_common(n_smis)
    drugs_max_rotatable = {x[0]: x[1] for x in drugs_max_rotatable}
    qm9_max_rotatable = Counter(qm9_rot_bonds).most_common(n_smis)
    qm9_max_rotatable = {x[0]: x[1] for x in qm9_max_rotatable}

    return drugs_max_rotatable, qm9_max_rotatable
