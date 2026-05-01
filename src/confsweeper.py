"""
confsweeper.py — generation-to-scoring conformer pipeline.

The public surface is a family of `get_mol_PE*` functions that each take a
SMILES string and return `(mol, conf_ids, energies)`: an RDKit mol with the
representative conformers attached, the integer IDs of those representatives,
and their potential energies in eV.

Three pipelines, each suited to a different sampling regime:

    get_mol_PE             baseline reference. Embeds n_confs conformers via
                           nvmolkit ETKDG, Butina-clusters, scores each
                           representative in a separate ASE calculator call.
                           Slow for large representative sets; use mainly to
                           validate the other pipelines.

    get_mol_PE_batched     production small-N pipeline. Same embed + Butina,
                           but scores all representatives in a single batched
                           MACE forward pass. Optionally adds backbone
                           dihedral-constrained Pool B conformers via
                           torsional_sampling.sample_constrained_confs (this
                           is the only pipeline that supports torsional
                           sampling). Use for general-purpose conformer
                           generation when n_confs ≤ ~1000 is sufficient.

    get_mol_PE_exhaustive  randomized-saturation pipeline for cyclic peptides
                           and other molecules with rich multi-basin Boltzmann
                           ensembles. Embeds thousands of conformers,
                           optionally MMFF-minimises them on GPU
                           (nvmolkit.mmffOptimization), MACE-scores in chunks,
                           applies a 5 kT energy filter, and dedup via the
                           private _energy_ranked_dedup helper (a basin-energy
                           variant of Butina that picks the lowest-energy
                           member of each geometric basin). Saturation-
                           validated defaults are baked in; see the function
                           docstring and docs/exhaustive_etkdg_plan.md.

The three pipelines share the same return contract, so downstream consumers
(SDF writers, fine-tuning conformer caches) swap one function call to upgrade.
"""

import contextlib
import logging
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

from mcmm import BasinMemory, MCMMWalker, ReplicaExchangeMCMMDriver, make_mcmm_proposer
from torsional_sampling import load_ramachandran_grids, sample_constrained_confs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("confsweeper")

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
_KT_EV_298K = 8.617333e-5 * 298.0  # k_B T at 298 K in eV (≈ 0.02568)


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


# RDKit's standard rotatable-bond SMARTS — single bonds, not in any ring,
# between heavy atoms that are themselves not terminal (D=1) and not in
# triple bonds. For cyclic peptides this matches side-chain rotations only;
# the macrocycle backbone is in-ring and therefore excluded.
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def _jitter_rotatable_dihedrals(
    mol: Chem.Mol,
    jitter_deg: float,
    seed: int,
) -> int:
    """
    Add uniform random rotation in [-jitter_deg, +jitter_deg] to each
    rotatable-bond dihedral on every conformer attached to mol (in place).

    Intended as a post-embedding diversifier: ETKDG's torsion-knowledge prior
    biases initial coordinates toward known-favourable dihedrals, which can
    leave it stuck in one basin even with thousands of seeds. A small random
    perturbation per rotatable bond pushes some conformers across nearby
    basin boundaries before MACE rescoring; combined with the energy filter
    and energy-ranked dedup, this can recover basins that pure ETKDG misses.

    Macrocycle backbone bonds are in-ring and not matched by the rotatable-
    bond SMARTS, so this only perturbs side-chain rotamers on cyclic
    peptides. To explore backbone dihedrals on macrocycles, use
    `torsional_sampling.sample_constrained_confs` instead.

    Params:
        mol: rdkit.Chem.Mol : mol with conformers attached (modified in place)
        jitter_deg: float : maximum absolute perturbation per dihedral in degrees;
            sampled uniformly on [-jitter_deg, +jitter_deg] independently per
            (conformer, dihedral) pair
        seed: int : RNG seed; fully determines the perturbation given
            identical input geometry
    Returns:
        int : number of rotatable dihedrals jittered per conformer (constant
            across conformers for the same mol)
    """
    from rdkit.Chem import rdMolTransforms

    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)

    # Build (i, j, k, l) atom-index quadruples once: j-k is the rotatable bond;
    # i and l are the first heavy-atom neighbours other than the bond partner.
    # Heavy-atom-only matters because the SMARTS still matches when Hs are
    # explicit (methyl-CH3 carbons have degree 4, not D1), but rotating around a
    # bond whose only off-partner neighbour is an H is geometrically degenerate
    # — it just spins the H(s). Skipping such bonds keeps `n` aligned with the
    # standard implicit-H rotatable-bond count.
    dihedrals: list[tuple[int, int, int, int]] = []
    for j, k in matches:
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)
        i = next(
            (
                n.GetIdx()
                for n in atom_j.GetNeighbors()
                if n.GetIdx() != k and n.GetAtomicNum() != 1
            ),
            None,
        )
        l = next(  # noqa: E741 — l is the standard fourth-atom name for dihedrals
            (
                n.GetIdx()
                for n in atom_k.GetNeighbors()
                if n.GetIdx() != j and n.GetAtomicNum() != 1
            ),
            None,
        )
        if i is None or l is None:
            continue
        dihedrals.append((i, j, k, l))

    rng = np.random.default_rng(seed)
    for cid in [c.GetId() for c in mol.GetConformers()]:
        conf = mol.GetConformer(cid)
        for i, j, k, l in dihedrals:
            current = rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
            delta = float(rng.uniform(-jitter_deg, jitter_deg))
            rdMolTransforms.SetDihedralDeg(conf, i, j, k, l, current + delta)

    return len(dihedrals)


def _minimize_score_filter_dedup(
    mol: Chem.Mol,
    all_conf_ids: List[int],
    hardware_opts,
    calc,
    score_chunk_size: int,
    e_window_kT: float,
    rmsd_threshold: float,
    minimize: bool,
    mmff_backend: str,
) -> Tuple[Chem.Mol, List[int], List[float]]:
    """
    Shared post-sampling pipeline tail used by every get_mol_PE_* family
    function whose Phase 1 sampler produces a pool of raw conformers attached
    to a single mol.

    Pipeline:
      1. Optional MMFF94 minimization in place. Two backends:
         'gpu' calls nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs in a
         single batched CUDA pass. 'cpu' calls RDKit's serial
         AllChem.MMFFOptimizeMolecule per conformer.
      2. Batched MACE scoring in chunks of score_chunk_size.
      3. Energy filter: drop conformers with (E - E_min) > e_window_kT * kT
         (kT_298K ≈ 26 meV). At least one conformer is always retained even
         when every scored energy is degenerate (e.g. all NaN); the
         argmin-energy conformer is force-kept in that case.
      4. Energy-ranked geometric dedup via _energy_ranked_dedup, keeping the
         lowest-energy member of each geometric basin defined by
         rmsd_threshold (normalised L1 units). Singleton survivors skip
         dedup entirely.
      5. Drop non-centroid conformers from the mol so the returned object
         matches the (mol, conf_ids, energies) contract used across the
         get_mol_PE_* family.

    Callers must guarantee all_conf_ids is non-empty. The empty-pool case is
    handled at the Phase 1 layer where it can be detected before the helper
    is invoked (returning (mol, [], []) without paying for MMFF or MACE).

    Params:
        mol: Chem.Mol : input mol with the raw conformer pool already attached
        all_conf_ids: List[int] : conformer IDs to process (must be non-empty)
        hardware_opts : nvmolkit hardware options (only used when mmff_backend='gpu')
        calc : MACECalculator from get_mace_calc()
        score_chunk_size: int : per-batch MACE forward pass cap
        e_window_kT: float : energy filter window in units of kT_298K
        rmsd_threshold: float : geometric dedup exclusion radius (normalised L1)
        minimize: bool : MMFF94-minimize each conformer before scoring
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA) or 'cpu' (RDKit serial).
            Only consulted when minimize=True.
    Returns:
        rdkit.Chem.Mol : mol with only basin-representative conformers attached
        List[int] : conformer IDs of basin representatives, ordered by ascending energy
        List[float] : potential energies in eV for each representative
    """
    # 1. Optional MMFF94 minimization. Errors leave the conformer at its
    # pre-minimize geometry; we don't track per-conformer success/failure
    # because MACE scoring afterwards naturally surfaces any pathological
    # geometries through extreme energies.
    if minimize:
        if mmff_backend == "gpu":
            # nvmolkit batched CUDA implementation — single call optimises
            # every conformer of `mol` in place. Order-dependent import:
            # nvmolkit.embedMolecules must be loaded first to register some
            # global C++ state, which the module-level import already did.
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([mol], hardwareOptions=hardware_opts)
        elif mmff_backend == "cpu":
            from rdkit.Chem import AllChem

            for cid in all_conf_ids:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        else:
            raise ValueError(
                f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
            )

    # 2. Batched MACE scoring, chunked to bound the GPU forward-pass batch size.
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    energies: List[float] = []
    for start in range(0, len(all_conf_ids), score_chunk_size):
        chunk_ids = all_conf_ids[start : start + score_chunk_size]
        ase_mols = [
            ase.Atoms(
                positions=mol.GetConformer(cid).GetPositions(),
                numbers=atomic_nums,
            )
            for cid in chunk_ids
        ]
        energies.extend(_mace_batch_energies(calc, ase_mols))

    energies_arr = np.asarray(energies, dtype=np.float64)

    # 3. Energy filter: keep conformers within e_window_kT * kT of the minimum.
    e_min = energies_arr.min()
    keep_mask = (energies_arr - e_min) <= e_window_kT * _KT_EV_298K
    if not keep_mask.any():
        # Only reachable on degenerate inputs (e.g. NaN energies). Force the
        # caller's contract: at least one centroid is always returned.
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[int(np.argmin(energies_arr))] = True

    kept_pool_idx = np.where(keep_mask)[0].tolist()
    kept_conf_ids = [all_conf_ids[i] for i in kept_pool_idx]
    kept_energies = energies_arr[kept_pool_idx]

    # 4. Energy-ranked geometric dedup. With a single survivor there is
    # nothing to cluster; otherwise use the basin-energy primitive.
    if len(kept_conf_ids) == 1:
        centroid_ids = list(kept_conf_ids)
        centroid_energies = [float(kept_energies[0])]
    else:
        coords = torch.tensor(
            np.array([mol.GetConformer(cid).GetPositions() for cid in kept_conf_ids])
        )
        centroid_pool_idx = _energy_ranked_dedup(
            coords, kept_energies, rmsd_threshold=rmsd_threshold
        )
        centroid_ids = [kept_conf_ids[i] for i in centroid_pool_idx]
        centroid_energies = [float(kept_energies[i]) for i in centroid_pool_idx]

    # 5. Output: drop non-centroid conformers from the mol so the returned
    # object matches the (mol, ids, energies) contract used across the
    # get_mol_PE_* family.
    centroid_set = set(centroid_ids)
    for cid in all_conf_ids:
        if cid not in centroid_set:
            mol.RemoveConformer(cid)

    return mol, centroid_ids, centroid_energies


def get_mol_PE_exhaustive(
    smi: str,
    params,
    hardware_opts,
    calc,
    n_seeds: int = 10000,
    embed_chunk_size: int = 1000,
    score_chunk_size: int = 500,
    e_window_kT: float = 5.0,
    rmsd_threshold: float = 0.1,
    minimize: bool = True,
    mmff_backend: str = "gpu",
    dihedral_jitter_deg: float = 0.0,
    seed: int = 0,
) -> Tuple[Chem.Mol, List[int], List[float]]:
    """
    Embed many randomized ETKDG conformers, MACE-score them all, energy-filter,
    and dedup geometrically by basin energy minimum.

    The premise is to replace CREST metadynamics with brute-force randomization:
    nvmolkit's GPU embed is two orders of magnitude cheaper than CREST per
    conformer, so we can afford thousands of seeds. The energy filter keeps
    geometric dedup tractable on the massive pre-pool by dropping conformers
    that cannot contribute to a 298 K Boltzmann ensemble regardless of their
    geometric uniqueness.

    When to use vs. get_mol_PE_batched:
      * get_mol_PE_batched — small N, optionally torsional sampling, fast.
      * get_mol_PE_exhaustive — cyclic peptides / molecules with rich
        Boltzmann ensembles where get_mol_PE_batched produces near-one-hot
        weight distributions because ETKDG-100 misses low-energy basins.

    Default values for n_seeds, minimize, and mmff_backend are the
    saturation-validated production settings from
    docs/exhaustive_etkdg_plan.md: across the five representative cyclic
    peptides tested (CREMP + PAMPA, n_heavy 27-103), this configuration
    reproduces CREST-quality Boltzmann distributions on most peptides at
    a fraction of CREST's compute cost. Larger n_seeds keeps helping
    stochastically but with diminishing returns; minimize=True is the
    decisive lever and should rarely be turned off.

    Pipeline:
      1. Embed n_seeds conformers via nvmolkit ETKDG. A single
         EmbedMolecules call is used when n_seeds <= embed_chunk_size; for
         larger n_seeds the call is repeated in chunks with seed offsets so
         the GPU memory footprint stays bounded.
      2. Optional rotatable-bond dihedral jitter (off by default;
         dihedral_jitter_deg > 0 enables). Adds uniform random rotation in
         [-jitter, +jitter] to every rotatable-bond dihedral on every
         conformer to push some conformers across nearby basin boundaries
         before scoring. Excludes ring bonds, so on cyclic peptides this
         only perturbs side-chain rotamers.
      3. Optional MMFF94 minimization in place (off by default). Two backends:
         'gpu' (default) calls nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs
         in a single batched CUDA pass — ~25-50× faster than 'cpu' on cyclic
         peptides. 'cpu' calls RDKit's serial AllChem.MMFFOptimizeMolecule per
         conformer; pick this only if you need bit-exact agreement with RDKit
         reference behaviour or if nvmolkit is unavailable.

         The two backends can disagree on the final geometry of any given
         conformer — different BFGS gradient details push the optimizer into
         neighbouring basins on a few percent of cases. Final-energy Pearson r
         between backends is ~0.95 with ~3-5 kcal/mol per-conformer std on
         cyclic peptides; the basin distribution after dedup is comparable
         but not identical. For exhaustive ETKDG saturation runs the GPU
         backend's stochasticity is acceptable because we care about basin
         coverage in aggregate, not per-conformer reproducibility.

         When both jitter and minimize are on, MMFF runs after jitter and
         will tend to relax perturbed geometries back toward the original
         basin.
      4. MACE-score the full pool in chunks of score_chunk_size.
      5. Drop conformers with (E - E_min) > e_window_kT * kT (kT = 26 meV at
         298 K). At least one conformer (the minimum) is always retained.
      6. Energy-ranked geometric dedup via _energy_ranked_dedup, keeping the
         lowest-energy member of each geometric basin defined by
         rmsd_threshold (normalised L1 units, matching get_mol_PE_batched).

    The randomSeed attribute on `params` is mutated during chunked embedding
    (each chunk uses seed + chunk_index * embed_chunk_size). Bit-exact
    reproducibility across runs is not guaranteed because nvmolkit advances
    internal random state independently of params.randomSeed, but
    distributional reproducibility (sorted-energy and basin-count statistics)
    is preserved.

    Params:
        smi: str : input SMILES string
        params : ETKDG params from get_embed_params or get_embed_params_macrocycle
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        calc : MACECalculator from get_mace_calc()
        n_seeds: int : total ETKDG seeds to embed
        embed_chunk_size: int : per-call cap before chunking; tune to GPU memory
        score_chunk_size: int : per-batch MACE forward pass cap
        e_window_kT: float : energy filter window in units of kT_298K
        rmsd_threshold: float : geometric dedup exclusion radius (normalised L1)
        minimize: bool : MMFF94-minimize each conformer before scoring
        mmff_backend: str : 'gpu' (default; nvmolkit batched CUDA) or 'cpu'
            (RDKit serial). Only consulted when minimize=True. See pipeline
            step 3 above for behaviour notes.
        dihedral_jitter_deg: float : maximum absolute rotation in degrees applied
            uniformly at random to each rotatable-bond dihedral on every
            conformer; 0.0 disables jitter (default)
        seed: int : base ETKDG random seed; chunk i uses seed + i*embed_chunk_size.
            The dihedral jitter RNG is also seeded from this value (with a
            fixed offset) so two runs with identical params and seed produce
            identical jitter patterns.
    Returns:
        rdkit.Chem.Mol : mol with only basin-representative conformers attached
        List[int] : conformer IDs of basin representatives, ordered by ascending energy
        List[float] : potential energies in eV for each representative
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    # 1. Massive embed (Path A single call, or Path B chunked when n_seeds is large)
    if n_seeds <= embed_chunk_size:
        params.randomSeed = seed
        embed.EmbedMolecules(
            [mol], params, confsPerMolecule=n_seeds, hardwareOptions=hardware_opts
        )
    else:
        n_remaining = n_seeds
        chunk_idx = 0
        while n_remaining > 0:
            this_chunk = min(embed_chunk_size, n_remaining)
            params.randomSeed = seed + chunk_idx * embed_chunk_size
            n_before = mol.GetNumConformers()
            embed.EmbedMolecules(
                [mol],
                params,
                confsPerMolecule=this_chunk,
                hardwareOptions=hardware_opts,
            )
            # Stop early if a chunk produced nothing — likely an
            # embedding-incompatible molecule rather than a transient failure.
            if mol.GetNumConformers() == n_before:
                break
            n_remaining -= this_chunk
            chunk_idx += 1

    all_conf_ids = [c.GetId() for c in mol.GetConformers()]
    if not all_conf_ids:
        return mol, [], []

    # 2. Optional rotatable-bond dihedral jitter. Mutates conformer geometry
    # in place; uses a deterministic seed offset so the perturbation is
    # reproducible from the function-level seed.
    if dihedral_jitter_deg > 0.0:
        _jitter_rotatable_dihedrals(
            mol,
            jitter_deg=dihedral_jitter_deg,
            seed=seed
            + 1_000_003,  # large prime offset to avoid clashing with chunked embed seeds
        )

    # 3-7. MMFF + MACE batched score + 5 kT energy filter + energy-ranked
    # dedup + non-centroid prune. Shared with get_mol_PE_pool_b (and any
    # future sampler with the same post-Phase-1 pipeline).
    return _minimize_score_filter_dedup(
        mol,
        all_conf_ids,
        hardware_opts,
        calc,
        score_chunk_size=score_chunk_size,
        e_window_kT=e_window_kT,
        rmsd_threshold=rmsd_threshold,
        minimize=minimize,
        mmff_backend=mmff_backend,
    )


def get_mol_PE_pool_b(
    smi: str,
    grids: dict,
    hardware_opts,
    calc,
    n_samples: int = 10000,
    n_attempts: int = 1,
    tolerance_deg: float = 30.0,
    strategy: str = "inverse",
    score_chunk_size: int = 500,
    e_window_kT: float = 5.0,
    rmsd_threshold: float = 0.1,
    minimize: bool = True,
    mmff_backend: str = "gpu",
    seed: int = 0,
) -> Tuple[Chem.Mol, List[int], List[float]]:
    """
    Sample backbone-dihedral-constrained ('Pool B') conformers, MACE-score them,
    energy-filter, and dedup geometrically. Macrocyclic peptides only.

    Sister function to get_mol_PE_exhaustive: same `(mol, conf_ids, energies)`
    contract and the same post-sampling tail (MMFF → MACE batched scoring → kT
    energy window → _energy_ranked_dedup), but Phase 1 swaps nvmolkit ETKDG for
    sample_constrained_confs, which embeds each conformer with the bounds matrix
    tightened to a CREMP-derived (phi, psi) target. The premise is that
    randomized ETKDG cannot push through to certain low-energy basins on large
    macrocycles (pampa_large) regardless of MMFF; constrained DG samples those
    basins directly and lets the rest of the pipeline judge whether they
    survive scoring.

    Defaults are calibrated for benchmarking against get_mol_PE_exhaustive at a
    matched compute budget:
      * n_samples=10000 matches exhaustive ETKDG's saturation-validated n_seeds.
      * n_attempts=1 (vs. sample_constrained_confs's own default of 5) caps the
        raw conformer count near 10k so wall-clock and basin-coverage curves
        compare cleanly. Bump for tight macrocycles where ring-closure failures
        dominate the feasibility rate.
      * strategy='inverse' weights rare-but-accessible Ramachandran cells, which
        is the design intent: the gaps ETKDG misses are exactly the rare cells
        in the GFN2-xTB distribution. Use 'uniform' for an unbiased sweep over
        the full CREMP-accessible region.

    The post-sampling tail (MMFF → MACE batched scoring → 5 kT energy filter
    → energy-ranked dedup → non-centroid pruning) is shared with
    get_mol_PE_exhaustive via the private _minimize_score_filter_dedup helper.

    Params:
        smi: str : input SMILES string (must be a head-to-tail cyclic peptide;
            sample_constrained_confs requires backbone (phi, psi) atoms)
        grids: dict : CREMP Ramachandran grids from load_ramachandran_grids()
        hardware_opts : nvmolkit hardware options from get_hardware_opts
        calc : MACECalculator from get_mace_calc()
        n_samples: int : number of (phi, psi) draws to attempt (raw budget;
            actual conformer count depends on per-sample feasibility)
        n_attempts: int : ETKDGv3 attempts per (phi, psi) draw (default 1 to
            keep raw pool comparable to get_mol_PE_exhaustive's n_seeds)
        tolerance_deg: float : dihedral constraint half-width in degrees
        strategy: str : 'inverse' (default; oversample rare cells) or 'uniform'
        score_chunk_size: int : per-batch MACE forward pass cap
        e_window_kT: float : energy filter window in units of kT_298K
        rmsd_threshold: float : geometric dedup exclusion radius (normalised L1)
        minimize: bool : MMFF94-minimize each conformer before scoring
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA) or 'cpu' (RDKit serial)
        seed: int : random seed for the (phi, psi) RNG and CPU embed attempts
    Returns:
        rdkit.Chem.Mol : mol with only basin-representative conformers attached
        List[int] : conformer IDs of basin representatives, ordered by ascending energy
        List[float] : potential energies in eV for each representative
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))

    sample_constrained_confs(
        mol,
        grids,
        n_samples=n_samples,
        n_attempts=n_attempts,
        tolerance_deg=tolerance_deg,
        strategy=strategy,
        seed=seed,
    )

    all_conf_ids = [c.GetId() for c in mol.GetConformers()]
    if not all_conf_ids:
        return mol, [], []

    return _minimize_score_filter_dedup(
        mol,
        all_conf_ids,
        hardware_opts,
        calc,
        score_chunk_size=score_chunk_size,
        e_window_kT=e_window_kT,
        rmsd_threshold=rmsd_threshold,
        minimize=minimize,
        mmff_backend=mmff_backend,
    )


def _geometric_temperature_ladder(kt_low: float, kt_high: float, n: int) -> List[float]:
    """
    Build a geometric temperature ladder of length `n` spanning kT_low to
    kT_high inclusive.

    Used by get_mol_PE_mcmm to seed the replica-exchange driver. Geometric
    spacing keeps the swap-acceptance rate roughly uniform across the
    ladder (the standard REMD convention).

    Params:
        kt_low: float : kT at the lowest temperature, in eV
        kt_high: float : kT at the highest temperature, in eV
        n: int : number of temperatures (ladder rungs)
    Returns:
        list[float] : kT values, monotone increasing, length n
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return [float(kt_low)]
    if not (0 < kt_low < kt_high):
        raise ValueError(
            f"require 0 < kt_low < kt_high, got kt_low={kt_low}, kt_high={kt_high}"
        )
    ratio = (kt_high / kt_low) ** (1.0 / (n - 1))
    return [float(kt_low * ratio**i) for i in range(n)]


def get_mol_PE_mcmm(
    smi: str,
    params,
    hardware_opts,
    calc,
    n_walkers_per_temp: int = 8,
    n_temperatures: int = 8,
    kt_low: float = None,
    kt_high: float = None,
    n_steps: int = 200,
    swap_interval: int = 20,
    drive_sigma_rad: float = 0.1,
    closure_tol: float = 0.01,
    score_chunk_size: int = 500,
    e_window_kT: float = 5.0,
    rmsd_threshold: float = 0.1,
    minimize: bool = True,
    mmff_backend: str = "gpu",
    n_init_confs: int = 1,
    seed: int = 0,
) -> Tuple[Chem.Mol, List[int], List[float]]:
    """
    Multiple Minimum Monte Carlo sampler with replica exchange and DBT
    concerted-rotation backbone moves. Macrocyclic peptides only.

    Sister function to get_mol_PE_exhaustive and get_mol_PE_pool_b. Same
    `(mol, conf_ids, energies)` contract and the same post-sampling tail
    (MMFF → MACE batched scoring → 5 kT energy filter →
    `_energy_ranked_dedup` → non-centroid pruning) via the shared
    `_minimize_score_filter_dedup` helper. Phase 1 is the new MCMM
    pipeline: ETKDG seed → MMFF minimise → run replica-exchange MC for
    `n_steps` steps with N walkers across the temperature ladder → take
    the basin set from the shared `BasinMemory` as the starting pool
    for the shared tail.

    Sampler premise (issue #11): exhaustive ETKDG saturates near
    one-hot Boltzmann distributions on `pampa_large`-style peptides
    despite the saturation-validated 10000-seed default. The
    hypothesis is that pure randomisation cannot push through to
    certain low-energy basins regardless of seed budget — a
    connectivity problem rather than a sampling-density one. MCMM
    walks adaptively from a known minimum with shared basin memory:
    each move perturbs near a current state, locally minimises, and
    biases against re-discovery via the Saunders 1/√usage factor.
    Replica exchange's high-T replicas cross barriers; the low-T
    replicas refine. See docs/mcmm_plan.md for the complete design.

    Defaults: 8 temperatures geometric 300 K → 600 K, 8 walkers per
    temperature (64 walkers total), 200 steps per walker (12 800
    minimisations total — matched-budget against
    `get_mol_PE_exhaustive`'s 10 000 seeds), swap attempts every 20
    steps. The `make_mcmm_proposer` factory in `src/mcmm.py` builds
    the per-step DBT geometry + batched MMFF + batched MACE pipeline.

    **v0 status note**: `make_mcmm_proposer` is currently a stub that
    returns `success=False` for every proposal, so no MC exploration
    occurs and the basin set ends with only the initial conformer.
    The orchestration around it is fully wired and tested. Step 8b
    swaps in the real DBT + MMFF + MACE proposer to enable
    exploration; see docs/mcmm_plan.md.

    Params:
        smi: str : input SMILES string (must be a head-to-tail cyclic
            peptide; the proposer enumerates backbone windows on it)
        params : ETKDG params for the seed conformer; typically from
            `get_embed_params_macrocycle()` for cyclic peptides
        hardware_opts : nvmolkit hardware options
        calc : MACECalculator from `get_mace_calc()`
        n_walkers_per_temp: int : walkers at each temperature (default 8)
        n_temperatures: int : ladder size (default 8)
        kt_low: float | None : kT at the coldest replica in eV; defaults
            to `_KT_EV_298K` (≈300 K)
        kt_high: float | None : kT at the hottest replica in eV;
            defaults to `2 * _KT_EV_298K` (≈600 K)
        n_steps: int : MC steps per walker (default 200)
        swap_interval: int : steps between swap attempts (default 20)
        drive_sigma_rad: float : Gaussian σ for drive-angle perturbation
            in radians (default 0.1 ≈ 5.7°)
        closure_tol: float : DBT closure tolerance in Å (default 0.01)
        score_chunk_size: int : MACE per-batch forward pass cap
        e_window_kT: float : energy filter window in kT_298K units
        rmsd_threshold: float : geometric dedup exclusion radius in
            normalised L1 units (matches `_energy_ranked_dedup`)
        minimize: bool : MMFF94-minimise the seed and final basins
        mmff_backend: str : 'gpu' (nvmolkit) or 'cpu' (RDKit serial)
        n_init_confs: int : number of distinct ETKDG seed conformers
            (default 1, the original behaviour). With n_init_confs > 1,
            ETKDG embeds that many seeds and walkers are distributed
            round-robin across them — walker `i` starts at seed
            `i % n_init_confs`. Round-robin (vs. block) means each
            temperature stack gets exposure to every seed basin, which
            preserves replica-exchange's mixing behaviour. The basin
            memory ends up pre-populated with up to `n_init_confs`
            distinct basins after walker construction (deduped per
            `rmsd_threshold`), giving MCMM a head start over the
            single-seed default. Lever C9 in docs/mcmm_plan.md.
        seed: int : base seed; derived seeds are produced for each
            walker, the proposer, and the swap RNG by adding fixed
            offsets, so two runs with the same seed are deterministic
    Returns:
        rdkit.Chem.Mol : mol with only basin-representative conformers
            attached (one per centroid surviving the energy filter and
            geometric dedup)
        List[int] : conformer IDs of basin representatives, ordered by
            ascending MACE energy
        List[float] : potential energies in eV for each representative
    """
    if kt_low is None:
        kt_low = _KT_EV_298K
    if kt_high is None:
        kt_high = 2.0 * _KT_EV_298K
    if n_init_confs < 1:
        raise ValueError(f"n_init_confs must be >= 1, got {n_init_confs}")

    # 1. Build seed conformer(s): ETKDG embed + MMFF + MACE-score per
    # seed for its initial energy. With n_init_confs > 1 we embed that
    # many distinct conformers up front; walkers later distribute
    # round-robin across them (lever C9 in docs/mcmm_plan.md).
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params.randomSeed = seed
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_init_confs, hardwareOptions=hardware_opts
    )
    if mol.GetNumConformers() == 0:
        return mol, [], []

    if minimize:
        if mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([mol], hardwareOptions=hardware_opts)
        elif mmff_backend == "cpu":
            from rdkit.Chem import AllChem

            for cid in [c.GetId() for c in mol.GetConformers()]:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        else:
            raise ValueError(
                f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
            )

    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    seed_conf_ids = [c.GetId() for c in mol.GetConformers()]
    n_actual_seeds = len(seed_conf_ids)

    # MACE-score every embedded seed in one batched call. Each walker's
    # initial energy is its assigned seed's energy.
    seed_ase_mols = [
        ase.Atoms(
            positions=mol.GetConformer(cid).GetPositions(),
            numbers=atomic_nums,
        )
        for cid in seed_conf_ids
    ]
    seed_energies = [float(e) for e in _mace_batch_energies(calc, seed_ase_mols)]
    seed_coords_list = [
        torch.tensor(mol.GetConformer(cid).GetPositions(), dtype=torch.float64)
        for cid in seed_conf_ids
    ]

    # 2. Build the shared basin memory.
    memory = BasinMemory(n_atoms=n_atoms, rmsd_threshold=rmsd_threshold)

    # 3. Build the temperature ladder and walkers. Per-walker random_fns
    # use deterministic offsets from `seed` so replicate runs are
    # reproducible without sharing global RNG state. Walkers distribute
    # round-robin across the embedded seeds: walker w (with global index
    # t * n_walkers_per_temp + i) starts at seed `w % n_actual_seeds`.
    # Round-robin (rather than block) gives each temperature stack
    # exposure to every seed basin, preserving REMD's mixing behaviour.
    kts = _geometric_temperature_ladder(kt_low, kt_high, n_temperatures)
    walkers_by_temp: List[List] = []
    for t, kt in enumerate(kts):
        group = []
        for i in range(n_walkers_per_temp):
            walker_idx_global = t * n_walkers_per_temp + i
            seed_idx = walker_idx_global % n_actual_seeds
            walker_rng = np.random.default_rng(seed + 1_000_003 + walker_idx_global)
            walker = MCMMWalker(
                seed_coords_list[seed_idx],
                seed_energies[seed_idx],
                kt=kt,
                memory=memory,
                random_fn=walker_rng.random,
            )
            group.append(walker)
        walkers_by_temp.append(group)

    # 4. Build the batch proposer. Step 8b's real implementation does the
    # DBT + MMFF + MACE work here; the v0 stub rejects every proposal.
    batch_propose_fn = make_mcmm_proposer(
        mol,
        hardware_opts=hardware_opts,
        calc=calc,
        drive_sigma_rad=drive_sigma_rad,
        closure_tol=closure_tol,
        seed=seed + 7_777_777,
    )

    # 5. Build the replica-exchange driver and run.
    swap_rng = np.random.default_rng(seed + 9_999_999)
    driver = ReplicaExchangeMCMMDriver(
        walkers_by_temp,
        batch_propose_fn,
        swap_interval=swap_interval,
        swap_random_fn=swap_rng.random,
    )
    driver.run(n_steps)

    # Diagnostic log so the "1 basin" pathology on small peptides is
    # easy to triage: closure-failure rate near 1.0 means DBT can't find
    # closing geometries (try looser closure_tol or larger
    # drive_sigma_rad); low Metropolis acceptance with healthy closure
    # means moves get rejected on energy alone (rare on small peptides);
    # low swap rate means the temperature ladder is too wide.
    proposer_stats = getattr(batch_propose_fn, "stats", None) or {}
    n_proposed = int(proposer_stats.get("n_proposed", 0))
    n_closure_failures = int(proposer_stats.get("n_closure_failures", 0))
    closure_failure_rate = (
        n_closure_failures / n_proposed if n_proposed > 0 else float("nan")
    )
    metropolis_accept_rate = (
        driver.n_accepted / n_proposed if n_proposed > 0 else float("nan")
    )
    logger.info(
        "MCMM diagnostics for %r: n_proposed=%d, closure_failure_rate=%.3f, "
        "metropolis_accept_rate=%.3f, swap_accept_rate=%.3f, "
        "basins_in_memory=%d (incl. initial)",
        smi,
        n_proposed,
        closure_failure_rate,
        metropolis_accept_rate,
        driver.swap_acceptance_rate,
        memory.n_basins,
    )

    # 6. Extract the basin set as conformers on the mol. Drop the seed
    # conformer first; we re-add the basin representatives below. Each
    # walker's initial state is already in memory (added on construction
    # via the MCMMWalker init's add_basin call), so memory.n_basins
    # reflects every distinct starting basin plus every novel basin
    # accepted during the run.
    mol.RemoveAllConformers()
    all_conf_ids: List[int] = []
    for k in range(memory.n_basins):
        conf = Chem.Conformer(n_atoms)
        coords_np = memory.coords[k].cpu().numpy()
        for atom_idx in range(n_atoms):
            x, y, z = coords_np[atom_idx]
            conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
        cid = mol.AddConformer(conf, assignId=True)
        all_conf_ids.append(cid)

    if not all_conf_ids:
        # Defensive: at least one basin (the initial state) is always
        # added on walker init, so this branch is reachable only if every
        # walker construction failed. Match the empty-pool convention of
        # the get_mol_PE_* family.
        return mol, [], []

    # 7. Pass through the shared MMFF + MACE + filter + dedup tail.
    return _minimize_score_filter_dedup(
        mol,
        all_conf_ids,
        hardware_opts,
        calc,
        score_chunk_size=score_chunk_size,
        e_window_kT=e_window_kT,
        rmsd_threshold=rmsd_threshold,
        minimize=minimize,
        mmff_backend=mmff_backend,
    )


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
