# Contains code to generate the CREMP++ dataset. Cyclic peptide
# generation functions courtesy of PepINVENT code
# (https://github.com/MolecularAI/PepINVENT/blob/master/data/query_peptide_preparation/CHUCKLES_representation.ipynb)
#
import itertools
import os

import click
import numpy as np
import pandas as pd
import rdkit
from openbabel import openbabel, pybel
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from src.confsweeper import *


class PeptideBuilder:
    """
    Wrapper for CHUCKLES based peptide generation scripts
    taken from the PepINVENTrepo - credit to @gokcegeylan
    for original code!
    """

    def __init__(self):
        return

    def convert_to_chuckles(self, aa_smiles):
        mol = pybel.readstring("smi", aa_smiles)
        try:
            n_term_pat = pybel.Smarts("[$([ND1,ND2]CC(O)=O)]")
            c_term_pat = pybel.Smarts("[$([OD1]C(=O)C[ND1,ND2])]")
            n_term_idx = n_term_pat.findall(mol)[0][0]
            c_term_idx = c_term_pat.findall(mol)[0][0]
        except:
            print("Cannot produce CHUCKLES from", aa_smiles)

        rearranger = openbabel.OBConversion()
        rearranger.SetInAndOutFormats("smi", "smi")
        rearranger.AddOption("f", openbabel.OBConversion.OUTOPTIONS, str(n_term_idx))
        rearranger.AddOption("l", openbabel.OBConversion.OUTOPTIONS, str(c_term_idx))
        outmol = openbabel.OBMol()
        rearranger.ReadString(outmol, aa_smiles)
        return rearranger.WriteString(outmol).strip()

    def uncharger(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        uncharger = rdMolStandardize.Uncharger()
        if mol:
            uncharged_mol = uncharger.uncharge(mol)
            uncharged_smi = Chem.MolToSmiles(uncharged_mol, isomericSmiles=True)
        else:
            uncharged_smi = None
        return uncharged_smi

    def remove_backbone_charges(self, original_smiles):
        if "[O-]" in original_smiles:
            modified_smiles = re.sub(r"\[O\-\]", "O", original_smiles)
        else:
            modified_smiles = original_smiles
        return modified_smiles

    def smiles_to_chuckles(self, amino_acid):
        # Preprocessing
        uncharged_aa = self.uncharger(amino_acid)
        uncharged_aa = self.remove_backbone_charges(uncharged_aa)

        # Convert to CHUCKLES
        chuckles_aa = self.convert_to_chuckles(uncharged_aa)
        return chuckles_aa

    def peptide2chuckles(self, amino_acids):
        aas = [self.smiles_to_chuckles(aa) for aa in amino_acids]
        return aas

    def cycle_mapping(self, smiles):
        map_num = 0
        numbers = []
        for s in smiles:
            num = [int(x) for x in [*s] if x.isdigit()]
            numbers.extend(num)
        if len(numbers) > 1:
            map_num = max(numbers)
        return map_num + 1

    def linear(self, smiles):
        smiles[-1] = smiles[-1] + "O"
        return smiles

    def head_to_tail(self, smiles):
        mapping_num = self.cycle_mapping(smiles)

        start_pos, end_pos = 0, -1
        cycle_start = smiles[start_pos]
        smiles[start_pos] = cycle_start[0] + f"{mapping_num}" + cycle_start[1:]

        cycle_end = smiles[end_pos]
        smiles[end_pos] = cycle_end[:-5] + f"{mapping_num}" + cycle_end[-5:]
        return smiles

    def smarts_pattern_match(self, smi, smarts):
        substructure = Chem.MolFromSmarts(smarts)
        mol = Chem.MolFromSmiles(smi)
        smarts_match = mol.HasSubstructMatch(substructure)
        return smarts_match

    def disulfide_bridge(self, smiles, cyclization_info):
        mapping_num = self.cycle_mapping(smiles)

        for pos in cyclization_info:
            sulfhdryl_smarts = "[S;D1,D2;!$(S(=O));!$(S(=[C,c]));!$(S([C,c])[C,c])]"
            smarts_match = self.smarts_pattern_match(smiles[pos], sulfhdryl_smarts)
            if not smarts_match:
                raise ValueError(
                    f"The chosen amino acid at position {pos} does not have a sulfhydryl (-SH) group."
                )

        start_pos, end_pos = cyclization_info[0], cyclization_info[1]

        cycle_start = smiles[start_pos]
        smiles[start_pos] = re.sub(r"(S)", rf"S{mapping_num}", cycle_start)

        cycle_end = smiles[end_pos]
        smiles[end_pos] = re.sub(r"(S)", rf"S{mapping_num}", cycle_end)
        smiles[-1] = smiles[-1] + "O"
        return smiles

    def sidechain_to_tail(self, smiles, cyclization_info):
        mapping_num = self.cycle_mapping(smiles)

        start_pos, end_pos = cyclization_info[0], cyclization_info[1]

        cycle_start = smiles[start_pos]
        cycle_starting_smi = self.prepare_sidechain(cycle_start, "Sidechain-To-Tail")
        if not cycle_starting_smi:
            raise ValueError(
                f"The chosen amino acid at position {start_pos} does not have a amino group."
            )

        cycle_starting_smi = cycle_starting_smi.replace("[U]", str(mapping_num))
        smiles[start_pos] = cycle_starting_smi

        cycle_end = smiles[end_pos]
        smiles[end_pos] = cycle_end[:-5] + str(mapping_num) + cycle_end[-5:]
        return smiles

    def head_to_sidechain(self, smiles, cyclization_info):
        mapping_num = self.cycle_mapping(smiles)

        start_pos, end_pos = cyclization_info[0], cyclization_info[1]

        cycle_end = smiles[end_pos]
        cycle_ending_smi = self.prepare_sidechain(cycle_end, "Head-To-Sidechain")
        if not cycle_ending_smi:
            raise ValueError(
                f"The chosen amino acid at position {end_pos} does not have a carboxylic acid group."
            )

        cycle_ending_smi = cycle_ending_smi.replace("[U]", str(mapping_num))
        smiles[end_pos] = cycle_ending_smi

        cycle_start = smiles[start_pos]
        smiles[start_pos] = cycle_start[0] + f"{mapping_num}" + cycle_start[1:]
        smiles[-1] = smiles[-1] + "O"
        return smiles

    def mutate_sidechain(self, smi, topology):
        """Temporarily replaces cyclization point to [U]"""
        if topology == "Sidechain-To-Tail":
            # Smarts for the amine group
            sidechain_group_smarts = "[ND1;!+;!$(NCc);$(N[C]);!$(N*=*);!$(NCC(=O)O):55]"
            rxnSmarts = f"{sidechain_group_smarts}.[*:99]>>[*:55][*:99]"

        if topology == "Head-To-Sidechain":
            # Smarts for the carboxylic group
            sidechain_group_smarts = "[O;$([O;D1]C(=O)[A]);!$(OC(=O)C[ND1,ND2]):55]"
            rxnSmarts = f"{sidechain_group_smarts}.[*:99]>>[*:55][*:99]"

        smarts_match = self.smarts_pattern_match(smi, sidechain_group_smarts)
        if not smarts_match:
            return None
        else:
            mutated_aa = self.mutation_rxn(smi, rxnSmarts)
        return mutated_aa

    def mutation_rxn(self, smi, rxnSmarts):
        mol = Chem.MolFromSmiles(smi)
        rxn = AllChem.ReactionFromSmarts(rxnSmarts)
        reacts = (mol, Chem.MolFromSmiles("[U]"))
        products = rxn.RunReactants(reacts)
        return Chem.MolToSmiles(products[0][0])

    def prepare_sidechain(self, smi, topology):
        mutated_smi = self.mutate_sidechain(smi, topology)
        mutated_smi = self.convert_to_chuckles(mutated_smi)
        return mutated_smi

    def assign_topology(self, peptide_chuckles, topology, cyclization_info=None):
        """Converts list of amino acid smiles into amino acids chuckles with the desired topological information

        peptide_chuckles:
                A list of amino acids ordered according to their position in the intended peptide chain

        topology:
                The choice of topology from the selection
                [ Linear, Head-To-Tail, Sidechain-To-Tail, Head-To-Sidechain, Disulfide-Bridge ]

        cyclization_info:
                A list of the positions of the chosen amino acids participating in the peptide cyclization.
                Is only necessary for Sidechain-To-Tail, Head-To-Sidechain, Disulfide-Bridge
        """
        if topology == "Linear":
            peptide_with_topology = self.linear(peptide_chuckles)

        elif topology == "Head-To-Tail":
            peptide_with_topology = self.ead_to_tail(peptide_chuckles)

        elif topology == "Sidechain-To-Tail":
            if cyclization_info:
                peptide_with_topology = self.sidechain_to_tail(
                    peptide_chuckles, cyclization_info
                )
            else:
                raise ValueError(
                    "The positions of the amino acids in the cyclization has to be specified."
                )

        elif topology == "Head-To-Sidechain":
            if cyclization_info:
                peptide_with_topology = self.head_to_sidechain(
                    peptide_chuckles, cyclization_info
                )
            else:
                raise ValueError(
                    "The positions of the amino acids in the cyclization has to be specified."
                )

        elif topology == "Disulfide-Bridge":
            if cyclization_info:
                peptide_with_topology = self.disulfide_bridge(
                    peptide_chuckles, cyclization_info
                )
            else:
                raise ValueError(
                    "The positions of the amino acids in the cyclization has to be specified."
                )

        else:
            raise ValueError(
                "This topology is not one of the available topologies: \n\t1) Linear\n\t2) Head-To-Tail\n\t3) Sidechain-To-Tail\n\t4) Head-To-Sidechain\n\t5) Disulfide-Bridge"
            )

        return peptide_with_topology

    def merge_original_peptide(self, peptide_chuckles):
        peptide = [x[:-1] for x in peptide_chuckles]
        peptide = "".join(peptide)
        return peptide

    def build_peptide(
        self,
        amino_acids: List,
        topology: str,
        masking_positions: List,
        cyclization_info=None,
    ):
        """Builds the peptide from a list of amino acids"""
        aa_chuckles = self.peptide2chuckles(amino_acids)
        peptide_chuckles = self.assign_topology(aa_chuckles, topology, cyclization_info)

        original_peptide = self.merge_original_peptide(peptide_chuckles)

        return original_peptide


# Combinatorially make peptides


@click.command()
@click.option("--amino_acid_lib_file")
@click.option("--peptide_save_dir")
def make_peptides(
    amino_acid_lib_file: os.PathLike | str,
    peptide_save_dir: os.PathLike | str,
    max_length: int = 12,
    topologies: List[str] = ["Head-To-Tail"],
) -> None:
    """
    Function to combinatorially generate macrocyclic peptide smiles
    Params:
        amino_acid_lib_file: os.PathLike|str : path to amino acid defintion file
        peptide_save_dir: os.PathLike|str : path to save generated peptides
    Returns:
        None
    """
    pb = PeptideBuilder()
    aa_lib = pd.read_csv(amino_acid_lib_file, comment="#")
    for length in range(4, max_length + 1):
        init_aa_seqs = itertools.permutations(aa_lib["Smiles"], length)
        smis = []
        seqs = []
        topology = []
        for seq in tqdm(init_aa_seqs, desc=f"Building macrocycles of length {length}"):
            for top in topologies:
                smis.append(pb.build_peptide(seq, topology=top))
                topology.append(top)
                seqs.append(seq)
        lens = [length] * len(smis)
        pd.DataFrame(
            {
                "initial_aa_seqs": init_aa_seqs,
                "smiles": smis,
                "topology": topology,
                "length": lens,
            }
        ).to_csv(
            os.path.join(peptide_save_dir, f"macrocyc_peptide_{length}.csv"),
            index=False,
        )


if __name__ == "__main__":
    make_peptides()
