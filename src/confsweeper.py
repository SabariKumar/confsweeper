import contextlib
import os
import random
import uuid
import warnings
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
from nvmolkit import clustering
from nvmolkit.types import HardwareOptions
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3
from rdkit.ML.Cluster.Butina import ClusterData
from tqdm import tqdm

from torsional_sampling import load_ramachandran_grids, sample_constrained_confs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


def get_embed_params_macrocycle() -> rdkit.Chem.rdDistGeom.EmbedParameters:
    """
    ETKDG setup for macrocyclic molecules. ETKDGv3 already enables
    useMacrocycleTorsions and useMacrocycle14config by default; the
    useSmallRingTorsions flag (which would improve embedding for small ring
    systems within the macrocycle scaffold) is intentionally left disabled
    because nvmolkit does not support it and hangs indefinitely in CPU
    preprocessing when it is set.

    Params:
        None
    Returns:
        rdkit.Chem.rdDistGeom.EmbedParameters
    """
    params = ETKDGv3()
    params.useRandomCoords = True
    params.useMacrocycleTorsions = True
    params.useMacrocycle14config = True
    return params


def get_hardware_opts(
    preprocessingThreads: int = 4,
    batch_size: int = 500,
    batchesPerGpu: int = 1,
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


def get_uma_calc(model: str = "uma-s-1", task: str = "omol"):
    """
    Return a FairChem UMA ASE calculator (default: UMA-S, omol task).

    Deprecated: UMA is not actively supported. Use get_mace_calc() instead.

    Params:
        model : FairChem checkpoint name or path (default "uma-s-1")
        task  : task head to use (default "omol")
    Returns:
        ASE calculator compatible with ase_mol.get_potential_energy()
    """
    import warnings

    warnings.warn(
        "get_uma_calc() is not actively supported. Use get_mace_calc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from fairchem.core import FAIRChemCalculator

    return FAIRChemCalculator(checkpoint_path=model, task_name=task)


def get_mace_calc(model: str = "medium", device: str = "cuda"):
    """
    Return a MACE-OFF ASE calculator.  Requires the optional mace extra:
        pip install confsweeper[mace]   or   pixi install -e mace

    Params:
        model  : MACE-OFF model size — "small", "medium", or "large"
        device : torch device string (default "cuda"); use "cpu" on login nodes
                 to pre-cache the model file without a GPU
    Returns:
        ASE calculator compatible with ase_mol.get_potential_energy()
    """
    try:
        from mace.calculators import mace_off  # type: ignore
    except ImportError:
        raise ImportError(
            "mace-torch is not installed. "
            "Install the optional extra: pip install confsweeper[mace] "
            "or pixi install -e mace"
        ) from None
    return mace_off(model=model, device=device, default_dtype="float32")


def get_mol_PE(
    smi: str,
    params,
    hardware_opts,
    calc,
    n_confs: int = 1000,
    cutoff_dist: float = 0.1,
    gpu_clustering: bool = True,
) -> Tuple:
    """
    Embed conformers for a SMILES string, cluster with Butina, and score with an ASE calculator.
    Params:
        smi: str : input smiles string
        params : ETKDG params from get_embed_params
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        calc : ASE calculator (e.g. from get_uma_calc() or get_mace_calc())
        n_confs: int : Number of conformers to embed
        cutoff_dist: float : Distance threshold for Butina clustering
    Returns:
        rdkit.Chem.Mol : mol with Butina-representative conformers
        List : conformer IDs of Butina representatives
        List : potential energies (eV) for each representative
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_confs, hardwareOptions=hardware_opts
    )
    coords = []
    for conf in mol.GetConformers():
        coords.append(conf.GetPositions())
    coords = torch.tensor(np.array(coords))  # (N_CONFS, n_atoms, 3)
    n_atoms = coords.shape[1]
    dists = torch.cdist(
        torch.flatten(coords, start_dim=1), torch.flatten(coords, start_dim=1), p=1.0
    ) / (3 * n_atoms)
    if gpu_clustering:
        # TODO: refactor to support dists on multiple gpus
        # butina returns a tuple of AsyncGpuResult objects when return_centroids=True
        clusters_result, centroids_result = clustering.butina(
            dists.to("cuda:0"), cutoff=cutoff_dist, return_centroids=True
        )
        # centroids is a 1-D array of conformer indices, one per cluster
        centroid_ids = centroids_result.numpy().tolist()
        rep_coords = [mol.GetConformer(x).GetPositions() for x in centroid_ids]
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pe = []

        for conf_coords in rep_coords:
            ase_mol = ase.Atoms(positions=conf_coords, numbers=atoms)
            ase_mol.calc = calc
            pe.append(ase_mol.get_potential_energy())
            del ase_mol
            torch.cuda.empty_cache()

        to_remove = [x for x in range(n_confs) if x not in centroid_ids]
        for id_ in to_remove:
            mol.RemoveConformer(id_)
        return mol, centroid_ids, pe

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
        pe = []
        for ase_mol in ase_mols:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    ase_mol.calc = calc
                    pe.append(ase_mol.get_potential_energy())

        to_remove = [x for x in range(n_confs) if x not in conf_ids]
        for id_ in to_remove:
            mol.RemoveConformer(id_)
        return mol, conf_ids, pe


def get_mol_PE_batched(
    smi: str,
    params,
    hardware_opts,
    calc,
    n_confs: int = 1000,
    cutoff_dist: float = 0.1,
    gpu_clustering: bool = True,
    grids: dict | None = None,
    n_constrained_samples: int = 0,
    torsion_strategy: str = "uniform",
    torsion_seed: int = 0,
) -> Tuple:
    """
    Embed conformers for a SMILES string, cluster with Butina, and score with a
    batched MACE forward pass. Optionally augments the ETKDG pool (Pool A) with a
    second pool of backbone dihedral-constrained conformers (Pool B) before
    deduplication, so the merged pool covers dihedral regions that ETKDG misses.

    Params:
        smi: str : input SMILES string
        params : ETKDG params from get_embed_params or get_embed_params_macrocycle
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        calc : MACECalculator from get_mace_calc() (UMA fallback supported)
        n_confs: int : Pool A conformers to embed via nvmolkit ETKDG
        cutoff_dist: float : Butina clustering cutoff (normalised L1 units)
        gpu_clustering: bool : use nvmolkit GPU Butina (True) or RDKit CPU Butina (False)
        grids: dict | None : Ramachandran grids from load_ramachandran_grids(); if None,
                             torsional sampling (Pool B) is skipped. MACROCYCLIC
                             PEPTIDES ONLY — see torsional_sampling.py module docstring.
        n_constrained_samples: int : Pool B (phi, psi) draws; ignored when grids is None
        torsion_strategy: str : 'uniform' or 'inverse' (see torsional_sampling module)
        torsion_seed: int : RNG seed for Pool B sampling
    Returns:
        rdkit.Chem.Mol : mol with only Butina-representative conformers attached
        List[int] : conformer IDs of cluster representatives
        List[float] : potential energies in eV for each representative
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_confs, hardwareOptions=hardware_opts
    )

    if grids is not None and n_constrained_samples > 0:
        sample_constrained_confs(
            mol,
            grids,
            n_constrained_samples,
            strategy=torsion_strategy,
            seed=torsion_seed,
        )

    # Collect all conformer IDs in iteration order before building the distance
    # matrix.  Pool A IDs are 0-based sequential (nvmolkit); Pool B IDs start
    # from n_pool_a (appended by embed_constrained).  Butina returns 0-based row
    # indices, not IDs, so we need this mapping to recover the actual conf IDs.
    all_conf_ids = [c.GetId() for c in mol.GetConformers()]
    if not all_conf_ids:
        return mol, [], []

    coords = torch.tensor(
        np.array([mol.GetConformer(cid).GetPositions() for cid in all_conf_ids])
    )
    n_atoms = coords.shape[1]
    dists = torch.cdist(
        torch.flatten(coords, start_dim=1), torch.flatten(coords, start_dim=1), p=1.0
    ) / (3 * n_atoms)

    if gpu_clustering:
        _, centroids_result = clustering.butina(
            dists.to("cuda:0"), cutoff=cutoff_dist, return_centroids=True
        )
        centroid_row_ids = centroids_result.numpy().tolist()
    else:
        clusters = ClusterData(
            dists.numpy(),
            len(all_conf_ids),
            cutoff_dist,
            isDistData=True,
            distFunc=None,
            reordering=True,
        )
        centroid_row_ids = [clusters[x][0] for x in range(len(clusters))]

    centroid_ids = [all_conf_ids[i] for i in centroid_row_ids]

    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    ase_mols = [
        ase.Atoms(positions=mol.GetConformer(cid).GetPositions(), numbers=atomic_nums)
        for cid in centroid_ids
    ]

    pe = _mace_batch_energies(calc, ase_mols)

    centroid_set = set(centroid_ids)
    for cid in all_conf_ids:
        if cid not in centroid_set:
            mol.RemoveConformer(cid)

    return mol, centroid_ids, pe


_KCAL_TO_EV = 0.043364  # 1 kcal/mol in eV


def _mace_batch_energies(calc, ase_mols: list) -> list:
    """
    Score multiple ASE Atoms in a single MACE forward pass.

    Builds one PyG Batch from all conformers, runs the model once, and returns
    per-conformer energies in eV. Falls back to sequential scoring on any error
    (e.g. if calc is not a MACECalculator or the private API has changed).

    Params:
        calc : MACECalculator returned by get_mace_calc()
        ase_mols : list[ase.Atoms] conformers to score
    Returns:
        list[float] potential energies in eV, one per conformer
    """
    try:
        from mace.tools.torch_geometric.batch import Batch

        data_list = [calc._atoms_to_batch(m) for m in ase_mols]
        batch = Batch.from_data_list(data_list).to(calc.device)

        with torch.no_grad():
            out = calc.models[0](
                batch,
                training=False,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
            )

        scale = getattr(calc, "energy_units_to_eV", 1.0)
        return (out["energy"].detach().cpu().float().numpy() * scale).tolist()
    except Exception:
        pe = []
        for mol in ase_mols:
            mol.calc = calc
            pe.append(float(mol.get_potential_energy()))
        return pe


def _energy_ranked_dedup(
    coords: torch.Tensor,
    energies: np.ndarray,
    rmsd_threshold: float,
) -> list[int]:
    """
    Pick basin representatives by energy rank with a geometric exclusion radius.

    Differs from Butina: Butina picks dense cluster centres first, which can
    discard the lowest-energy member of a dense cluster. Energy-ranked dedup
    iterates lowest energy first, so each basin's representative is its
    energy minimum.

    Distances use the same normalised L1 metric as get_mol_PE_batched so the
    rmsd_threshold parameter is comparable to its cutoff_dist (default 0.1).

    Params:
        coords: torch.Tensor : conformer coordinates [N, A, 3]
        energies: ndarray [N] : potential energies in eV (any shift; only differences matter)
        rmsd_threshold: float : exclusion radius in normalised L1 units (sum |Δ| / 3·A)
    Returns:
        list[int] : indices into the input arrays of the chosen centroids,
                    ordered by ascending energy
    """
    n = coords.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    n_atoms = coords.shape[1]
    flat = coords.reshape(n, -1)
    order = np.argsort(np.asarray(energies), kind="stable")

    excluded = torch.zeros(n, dtype=torch.bool, device=flat.device)
    centroids: list[int] = []
    for idx in order.tolist():
        if bool(excluded[idx]):
            continue
        centroids.append(idx)
        d = (flat - flat[idx].unsqueeze(0)).abs().sum(dim=1) / (3 * n_atoms)
        excluded |= d < rmsd_threshold
    return centroids


def get_mol_PE_mmff(
    smi: str,
    params,
    hardware_opts,
    n_confs: int = 1000,
    cutoff_dist: float = 0.1,
    gpu_clustering: bool = True,
) -> Tuple:
    """
    Like get_mol_PE but scores conformers with RDKit MMFF94 instead of an ASE calculator.

    Requires no GPU for energy scoring (MMFF94 runs on CPU). GPU is still used for
    conformer embedding and Butina clustering via nvmolkit when gpu_clustering=True.
    Energies are returned in eV (converted from MMFF94 kcal/mol) for compatibility
    with downstream Boltzmann weighting code.

    Params:
        smi: str : input SMILES string
        params : ETKDG params from get_embed_params or get_embed_params_macrocycle
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        n_confs: int : conformers to embed before clustering
        cutoff_dist: float : Butina RMSD clustering cutoff
        gpu_clustering: bool : use nvmolkit GPU Butina (True) or RDKit CPU Butina (False)
    Returns:
        rdkit.Chem.Mol : mol with only Butina-representative conformers attached
        List[int] : conformer IDs of cluster representatives
        List[float] : MMFF94 potential energies in eV for each representative
    """
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_confs, hardwareOptions=hardware_opts
    )

    n_embedded = mol.GetNumConformers()
    if n_embedded == 0:
        return mol, [], []

    coords = torch.tensor(
        np.array([mol.GetConformer(i).GetPositions() for i in range(n_embedded)])
    )
    n_atoms = coords.shape[1]
    dists = torch.cdist(
        torch.flatten(coords, start_dim=1), torch.flatten(coords, start_dim=1), p=1.0
    ) / (3 * n_atoms)

    if gpu_clustering:
        _, centroids_result = clustering.butina(
            dists.to("cuda:0"), cutoff=cutoff_dist, return_centroids=True
        )
        centroid_ids = centroids_result.numpy().tolist()
    else:
        clusters = ClusterData(
            dists.numpy(),
            n_embedded,
            cutoff_dist,
            isDistData=True,
            distFunc=None,
            reordering=True,
        )
        centroid_ids = [clusters[x][0] for x in range(len(clusters))]

    mp = AllChem.MMFFGetMoleculeProperties(mol)
    pe = []
    for cid in centroid_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        energy_kcal = ff.CalcEnergy() if ff is not None else 1e6
        pe.append(energy_kcal * _KCAL_TO_EV)

    to_remove = [x for x in range(n_embedded) if x not in centroid_ids]
    for id_ in to_remove:
        mol.RemoveConformer(id_)

    return mol, centroid_ids, pe


def write_sdf(
    mol: rdkit.Chem.Mol,
    conf_ids: List,
    pe: List,
    id: str,
    output_dir: os.PathLike | str,
    save_lowest_energy: bool,
) -> None:
    """
    Wrapper function to enable separation of conformer generation
    Params:
        mol: rdkit.Chem.Mol : input mol object
        conf_ids: List : list of valid rdkit conformer ids
        pe: List : list of ASE mol objects
        id: str : molecule uuid
        save_dir: os.PathLike | str : sdf save directory
    Returns:
        None
    """

    writer = Chem.SDWriter(os.path.join(output_dir, id + ".sdf"))
    if save_lowest_energy:
        min_conf = conf_ids[pe.index(min(pe))]
        conf = mol.GetConformer(min_conf)
        conf.SetDoubleProp("MACE_ENERGY", min(pe))
        mol.SetDoubleProp("MACE_ENERGY", min(pe))
        mol.SetProp("id", id)
        writer.write(mol, confId=min_conf)
    else:
        for conf_id, pe_ in zip(conf_ids, pe):
            conf = mol.GetConformer(conf_id)
            conf.SetDoubleProp("MACE_ENERGY", pe_)
            mol.SetDoubleProp("MACE_ENERGY", pe_)
            mol.SetProp("id", id)
            writer.write(mol, confId=conf_id)


@click.command()
@click.option("--smi_csv")
@click.option("--output_dir")
@click.option("--save_lowest_energy", default=False)
@click.option(
    "--sampling_mode",
    type=click.Choice(["etkdg", "etkdg+torsional"]),
    default="etkdg",
    show_default=True,
    help="etkdg: standard nvmolkit ETKDGv3, suitable for any molecule type. "
    "etkdg+torsional: adds a second pool of backbone dihedral-constrained conformers "
    "sampled from the CREMP Ramachandran prior — MACROCYCLIC PEPTIDES ONLY. "
    "Automatically switches to macrocycle-specific ETKDG parameters.",
)
@click.option(
    "--n_constrained_samples",
    default=200,
    show_default=True,
    help="Number of (phi, psi) draws for Pool B (etkdg+torsional only).",
)
@click.option(
    "--torsion_strategy",
    type=click.Choice(["uniform", "inverse"]),
    default="uniform",
    show_default=True,
    help="uniform: equal weight to all CREMP-accessible cells. "
    "inverse: oversample rare cells to fill ETKDG gaps (etkdg+torsional only).",
)
@click.option(
    "--ramachandran_grids",
    default="data/processed/cremp/ramachandran_grids.npz",
    show_default=True,
    help="Path to CREMP Ramachandran grids .npz built by data/scripts/build_ramachandran_grids.py.",
)
def run_PE_calc(
    smi_csv: pd.DataFrame,
    output_dir: os.PathLike | str,
    save_lowest_energy: bool,
    sampling_mode: str,
    n_constrained_samples: int,
    torsion_strategy: str,
    ramachandran_grids: str,
):
    hardware_opts = get_hardware_opts()
    mace_calc = get_mace_calc()
    smi_df = read_csv(smi_csv)

    if sampling_mode == "etkdg+torsional":
        params = get_embed_params_macrocycle()
        grids = load_ramachandran_grids(ramachandran_grids)
    else:
        params = get_embed_params()
        grids = None

    for smi, uuid_ in tqdm(zip(smi_df["smiles"], smi_df["uuid"]), total=len(smi_df)):
        if sampling_mode == "etkdg+torsional":
            mol, conf_ids, pe = get_mol_PE_batched(
                smi=smi,
                params=params,
                hardware_opts=hardware_opts,
                calc=mace_calc,
                grids=grids,
                n_constrained_samples=n_constrained_samples,
                torsion_strategy=torsion_strategy,
            )
        else:
            mol, conf_ids, pe = get_mol_PE(
                smi=smi,
                params=params,
                hardware_opts=hardware_opts,
                calc=mace_calc,
            )
        write_sdf(
            mol=mol,
            conf_ids=conf_ids,
            pe=pe,
            id=uuid_,
            output_dir=output_dir,
            save_lowest_energy=save_lowest_energy,
        )


if __name__ == "__main__":
    run_PE_calc()
