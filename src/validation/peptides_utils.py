from typing import List

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem


def get_backbone_atoms(mol: rdkit.Chem.Mol) -> List:
    """
    Returns the atom indices of backbone atoms
    Params:
        mol: rdkit.Chem.Mol : input rdkit mol object
    Returns:
        List
    """
    amide_backbone = Chem.MolFromSmarts("[N]-[C]-[C]=[O]")
    matches = mol.GetSubstructMatches(amide_backbone)
    backbone_ats = set(idx for match in matches for idx in match)
    return backbone_ats


def get_ramachandran_atoms(mol: rdkit.Chem.Mol) -> List:
    """
    Returns a dict with phi and psi dihedral atom indices for
    all AAs in the input mol object. Use ComputeDihedralAngle to get
    Ramachandran angles.
    """
    phi_struc = Chem.MolFromSmarts("[C]-[NH1]-[CH1]-[C]")
    psi_struc = Chem.MolFromSmarts("")
