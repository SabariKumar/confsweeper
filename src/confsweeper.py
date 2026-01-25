import os
import random
import uuid
from pathlib import Path
from typing import Callable, List, Tuple

import ase
import click
import matplotlib.pyplot as plt
import numpy as np
import nvmolkit
import nvmolkit.embedMolecules as embed
import pandas as pd
import rdkit
import torch
from ase.build import molecule
from mace.calculators import mace_mp
from nvmolkit import clustering
from nvmolkit.types import HardwareOptions
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3
from rdkit.ML.Cluster.Butina import ClusterData
from tqdm import tqdm

RANDOM_SEED = 42
rd = random.Random()
rd.seed(RANDOM_SEED)


def read_csv(input_csv: os.PathLike | str, write_uuids: bool = True) -> pd.DataFrame:
    """
    Reads a smiles csv from disk and assigns uuids, writing a new
    csv to disk with the new uuids.
    Params:
        input_csv: os.PathLike|str : input csv path. Must contain a column called 'smiles'.
                                     If write_uuids == False, must also contain an index column called 'uuid'
        write_uuids: bool : whether to assign uuids

    Returns:
        pd.DataFrame: smiles df

    """
    smi_df = pd.read_csv(input_csv)
    if write_uuids:
        input_csv = Path(input_csv)
        basename = input_csv.stem
        uuids = [
            str(uuid.UUID(int=rd.getrandbits(128), version=4).hex)
            for _ in smi_df["smiles"]
        ]
        smi_df["uuid"] = uuids
        smi_df.to_csv(input_csv.with_name(f"{basename}_uuids.csv"), index=False)
    return smi_df


def get_embed_params() -> rdkit.Chem.rdDistGeom.EmbedParameters:
    """
    ETKDG setup for nvmolkit
    Params:
        None
    Returns:
        rdkit.Chem.rdDistGeom.EmbedParameters
    """
    params = ETKDGv3()
    params.useRandomCoords = True
    return params


def get_hardware_opts(
    preprocessingThreads: int = 4,
    batch_size: int = 500,
    batchesPerGpu: int = 2,
    gpuIds: List = [0],
) -> nvmolkit.types.HardwareOptions:
    """
    GPU setup options for nvmolkit
    Params:
        preprocessingThreads: int : Number of CPU threads to use
        batch_size: int : Molecules per batch
        batchesPerGpu: int : Concurrent batches on a single GPU
        gpuIds: List[int] : GPU CUDA IDs to use
    Returns:
        nvmolkit.types.HardwareOptions : NVMolKit hardware params
    """

    return HardwareOptions(
        preprocessingThreads=preprocessingThreads,
        batchSize=batch_size,
        batchesPerGpu=batchesPerGpu,
        gpuIds=gpuIds,
    )


def get_mace_calc():
    """
    Convenience hook for subbing in different mace models.
    Params:
        None
    Returns:
        Callable: MACE calculator object
    """
    return mace_mp()


def get_mol_PE(
    smi: str,
    params,
    hardware_opts,
    mace_calc,
    n_confs: int = 1000,
    cutoff_dist: float = 0.1,
    gpu_clustering: bool = True,
) -> Tuple:
    """
    Save a multiSDF for a single smiles string.
    Energy saved as "MACE_ENERGY" in the SDF for each conformer.
    Params:
        smi: str : input smiles string
        params : ETKDG params from get_embed_params
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        mace_calc : ASE MACE calculator from get_mace_calc
        n_confs: int : Number of confomers to use
        cutoff_dist: float : Distance threshold for Butina clustering
    Returns:
        rdkit.Chem.Mol : input mol object
        List : list of valid rdkit conformer ids
        List : list of ASE mol objects
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_confs, hardwareOptions=hardware_opts
    )
    coords = []
    for conf in mol.GetConformers():
        coords.append(conf.GetPositions())
    coords = torch.tensor(np.array(coords))  # (N_CONFS, n_atoms, 3)
    dists = torch.cdist(
        torch.flatten(coords, start_dim=1), torch.flatten(coords, start_dim=1), p=1.0
    ) / (3 * n_confs)
    if gpu_clustering:
        # TODO: refactor to support dists on multiple gpus
        clusters = clustering.butina(dists.to("cuda:0"), cutoff=cutoff_dist).numpy()
        cluster_confs = {}
        for cluster_id in set(clusters.tolist()):
            conf_ids = np.argwhere(clusters == cluster_id).flatten()
            geoms = []
            for conf in conf_ids:
                geoms.append(mol.GetConformer(int(conf)).GetPositions())
            geoms = np.array(geoms)
            geoms = np.abs(geoms - geoms.mean(axis=0))
            cluster_confs[cluster_id] = int(
                conf_ids[int(np.argmin([np.sum(x) for x in geoms]))]
            )  # Conf id with minimum MAE to centroid

        if len(set(cluster_confs.values())) != len(cluster_confs.values()):
            raise ValueError(
                "Duplicate conformer centroids found; reduce cutoff distance and rerun!"
            )

        rep_coords = [
            mol.GetConformer(x).GetPositions() for x in cluster_confs.values()
        ]
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        ase_mols = [
            ase.Atoms(positions=rep_coord, numbers=atoms) for rep_coord in rep_coords
        ]
        for ase_mol in ase_mols:
            ase_mol.calc = mace_mp()
            ase_mol.get_potential_energy()

        to_remove = [x for x in range(1000) if x not in list(cluster_confs.values())]
        for id_ in to_remove:
            mol.RemoveConformer(id_)
        conf_ids = cluster_confs.values()

    else:
        clusters = ClusterData(
            dists.numpy(),
            n_confs,
            cutoff_dist,
            isDistData=True,
            distFunc=None,
            reordering=True,
        )
        rep_coords = [coords[clusters[x][0]].numpy() for x in range(len(clusters))]
        conf_ids = [clusters[x][0] for x in range(len(clusters))]
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        ase_mols = [
            ase.Atoms(positions=rep_coord, numbers=atoms) for rep_coord in rep_coords
        ]
        for ase_mol in ase_mols:
            ase_mol.calc = mace_calc
            ase_mol.get_potential_energy()

        to_remove = [x for x in range(n_confs) if x not in conf_ids]
        for id_ in to_remove:
            mol.RemoveConformer(id_)

    return mol, conf_ids, ase_mols


def write_sdf(
    mol: rdkit.Chem.Mol,
    conf_ids: List,
    ase_mols: List,
    id: str,
    output_dir: os.PathLike | str,
    save_lowest_energy: bool,
) -> None:
    """
    Wrapper function to enable separation of conformer generation
    Params:
        mol: rdkit.Chem.Mol : input mol object
        conf_ids: List : list of valid rdkit conformer ids
        ase_mols: List : list of ASE mol objects
        id: str : molecule uuid
        save_dir: os.PathLike | str : sdf save directory
    Returns:
        None
    """

    writer = Chem.SDWriter(os.path.join(output_dir, id + ".sdf"))
    if save_lowest_energy:
        energies = []
        for conf_id, ase_mol in zip(conf_ids, ase_mols):
            energies.append(ase_mol.get_potential_energy())
        min_conf = conf_ids[energies.index(min(energies))]
        conf = mol.GetConformer(min_conf)
        conf.SetDoubleProp("MACE_ENERGY", min(energies))
        mol.SetDoubleProp("MACE_ENERGY", min(energies))
        mol.SetProp("id", id)
        writer.write(mol, confId=conf_id)
    else:
        for conf_id, ase_mol in zip(conf_ids, ase_mols):
            mpe = ase_mol.get_potential_energy()
            conf = mol.GetConformer(conf_id)
            conf.SetDoubleProp("MACE_ENERGY", mpe)
            mol.SetDoubleProp("MACE_ENERGY", mpe)
            mol.SetProp("id", id)
            writer.write(mol, confId=conf_id)


@click.command()
@click.option("--smi_csv")
@click.option("--output_dir")
@click.option("--save_lowest_energy", default=False)
def run_PE_calc(
    smi_csv: pd.DataFrame, output_dir: os.PathLike | str, save_lowest_energy: bool
):
    params = get_embed_params()
    hardware_opts = get_hardware_opts()
    macemp = get_mace_calc()
    smi_df = read_csv(smi_csv)
    for smi, uuid_ in tqdm(zip(smi_df["smiles"], smi_df["uuid"]), total=len(smi_df)):
        mol, conf_ids, ase_mols = get_mol_PE(
            smi=smi,
            params=params,
            hardware_opts=hardware_opts,
            mace_calc=macemp,
        )
        write_sdf(
            mol=mol,
            conf_ids=conf_ids,
            ase_mols=ase_mols,
            id=uuid_,
            output_dir=output_dir,
            save_lowest_energy=save_lowest_energy,
        )


if __name__ == "__main__":
    run_PE_calc()
