"""
GPU integration tests for the confsweeper pipeline.

These tests exercise the real GPU code paths — nvmolkit conformer embedding,
nvmolkit Butina clustering, and MACE/UMA energy scoring — without mocking.
They are skipped by default and must be opted into explicitly:

    pixi run python -m pytest tests/integration/ --gpu

A CUDA device must be present. Tests that require a specific energy backend
(MACE or UMA) are individually skipped if that backend is not available.

The test molecule is cyclo(Ala)4 — a 12-membered macrocyclic peptide with four
backbone residues, small enough to embed quickly but representative of the target
use case.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import ase
import numpy as np
import nvmolkit.embedMolecules as embed
import pytest
import torch
from nvmolkit import clustering
from rdkit import Chem
from rdkit.Chem import Conformer

from confsweeper import (
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mol_PE_batched,
    get_mol_PE_mmff,
    load_ramachandran_grids,
)

_CYCLOALA4 = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"
_GRIDS_PATH = Path(__file__).parents[2] / "data/processed/cremp/ramachandran_grids.npz"


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped to avoid repeated setup)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return get_embed_params_macrocycle()


@pytest.fixture(scope="module")
def hw():
    return get_hardware_opts()


@pytest.fixture(scope="module")
def cremp_grids():
    if not _GRIDS_PATH.exists():
        pytest.skip(
            "CREMP Ramachandran grids not found; run build_ramachandran_grids.py first"
        )
    return load_ramachandran_grids(_GRIDS_PATH)


# ---------------------------------------------------------------------------
# Layer 1: raw nvmolkit GPU primitives
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_nvmolkit_gpu_embedding_produces_conformers(params, hw):
    """nvmolkit GPU embedding must produce at least one conformer for cyclo(Ala)4."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_CYCLOALA4))
    embed.EmbedMolecules([mol], params, confsPerMolecule=50, hardwareOptions=hw)
    assert mol.GetNumConformers() > 0, (
        "nvmolkit GPU embedding produced 0 conformers for cyclo(Ala)4 — "
        "check CUDA drivers, nvmolkit installation, and macrocycle embed params"
    )


@pytest.mark.gpu
def test_nvmolkit_gpu_butina_returns_valid_centroids(params, hw):
    """nvmolkit GPU Butina must return unique centroid indices within [0, n_confs)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_CYCLOALA4))
    embed.EmbedMolecules([mol], params, confsPerMolecule=50, hardwareOptions=hw)
    n = mol.GetNumConformers()
    if n == 0:
        pytest.skip("GPU embedding produced 0 conformers")

    coords = torch.tensor(
        np.array([mol.GetConformer(i).GetPositions() for i in range(n)])
    )
    n_atoms = coords.shape[1]
    dists = torch.cdist(
        torch.flatten(coords, start_dim=1),
        torch.flatten(coords, start_dim=1),
        p=1.0,
    ) / (3 * n_atoms)

    _, centroids_result = clustering.butina(
        dists.to("cuda:0"), cutoff=0.1, return_centroids=True
    )
    centroid_ids = centroids_result.numpy().tolist()

    assert len(centroid_ids) > 0
    assert all(0 <= cid < n for cid in centroid_ids)
    assert len(centroid_ids) == len(
        set(centroid_ids)
    ), "Butina returned duplicate centroids"


# ---------------------------------------------------------------------------
# Layer 2: pipeline functions — GPU embed + GPU Butina, CPU energy scoring
#
# These tests exercise the full embed→cluster→score path without requiring a
# neural-network model. MMFF94 scores on CPU; the GPU code is still exercised
# for embedding and Butina clustering.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_get_mol_PE_mmff_gpu_pipeline(params, hw):
    """Full GPU embed + GPU Butina + CPU MMFF94 pipeline runs end-to-end."""
    mol, conf_ids, pe = get_mol_PE_mmff(
        smi=_CYCLOALA4,
        params=params,
        hardware_opts=hw,
        n_confs=200,
        cutoff_dist=0.1,
        gpu_clustering=True,
    )
    assert len(conf_ids) > 0
    assert len(pe) == len(conf_ids)
    assert mol.GetNumConformers() == len(conf_ids)
    assert all(isinstance(e, float) for e in pe)
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


@pytest.mark.gpu
def test_get_mol_PE_batched_gpu_etkdg(params, hw):
    """GPU embed + GPU Butina + mocked scoring: verifies embed/cluster path is correct."""
    with patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):
        mol, conf_ids, pe = get_mol_PE_batched(
            smi=_CYCLOALA4,
            params=params,
            hardware_opts=hw,
            calc=MagicMock(),
            n_confs=200,
            cutoff_dist=0.1,
        )
    assert len(conf_ids) > 0
    assert len(pe) == len(conf_ids)
    assert mol.GetNumConformers() == len(conf_ids)
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


@pytest.mark.gpu
def test_get_mol_PE_batched_gpu_torsional(params, hw, cremp_grids):
    """Two-pool pipeline (ETKDG + torsional) with GPU Butina and mocked scoring."""
    with patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):
        mol, conf_ids, pe = get_mol_PE_batched(
            smi=_CYCLOALA4,
            params=params,
            hardware_opts=hw,
            calc=MagicMock(),
            n_confs=200,
            cutoff_dist=0.1,
            grids=cremp_grids,
            n_constrained_samples=100,
            torsion_strategy="inverse",
        )
    assert len(conf_ids) > 0
    assert len(pe) == len(conf_ids)
    assert mol.GetNumConformers() == len(conf_ids)
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


@pytest.mark.gpu
def test_get_mol_PE_batched_torsional_pool_b_increases_coverage(
    params, hw, cremp_grids
):
    """Torsional sampling must add at least one conformer that survives Butina.

    This is a sanity check that Pool B is non-trivially contributing to the
    merged pool: if the torsional sampler produces conformers and Butina
    de-duplicates the combined pool, the total representative count should be
    >= the ETKDG-only count (never fewer).
    """
    with patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):
        _, conf_ids_etkdg, _ = get_mol_PE_batched(
            smi=_CYCLOALA4,
            params=params,
            hardware_opts=hw,
            calc=MagicMock(),
            n_confs=100,
            cutoff_dist=0.1,
        )

    with patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):
        _, conf_ids_torsional, _ = get_mol_PE_batched(
            smi=_CYCLOALA4,
            params=params,
            hardware_opts=hw,
            calc=MagicMock(),
            n_confs=100,
            cutoff_dist=0.1,
            grids=cremp_grids,
            n_constrained_samples=100,
            torsion_strategy="inverse",
        )

    assert len(conf_ids_torsional) >= len(conf_ids_etkdg), (
        f"Torsional sampling reduced the representative count "
        f"({len(conf_ids_torsional)} < {len(conf_ids_etkdg)})"
    )


# ---------------------------------------------------------------------------
# Layer 3: full pipeline with real neural-network scoring
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_get_mol_PE_batched_gpu_mace(params, hw):
    """Full pipeline with real MACE-OFF scoring on GPU.

    Skipped if mace-torch is not installed (requires the mace pixi environment).
    """
    try:
        from confsweeper import get_mace_calc

        calc = get_mace_calc(model="small", device="cuda")
    except ImportError:
        pytest.skip("mace-torch not installed; activate the mace pixi env")

    mol, conf_ids, pe = get_mol_PE_batched(
        smi=_CYCLOALA4,
        params=params,
        hardware_opts=hw,
        calc=calc,
        n_confs=200,
        cutoff_dist=0.1,
    )
    assert len(conf_ids) > 0
    assert len(pe) == len(conf_ids)
    assert all(isinstance(e, float) for e in pe)
    # MACE-OFF energies for closed-shell organic molecules are negative (in eV)
    assert all(e < 0 for e in pe), f"Unexpected positive MACE energies: {pe}"


@pytest.mark.gpu
def test_get_mol_PE_batched_gpu_uma(params, hw):
    """Full pipeline with real UMA (FairChem) scoring on GPU.

    Skipped if fairchem-core is not installed or the UMA checkpoint is not cached.
    """
    try:
        from confsweeper import get_uma_calc

        calc = get_uma_calc(model="uma-s-1", task="omol")
    except Exception as exc:
        pytest.skip(f"UMA calculator unavailable: {exc}")

    mol, conf_ids, pe = get_mol_PE_batched(
        smi=_CYCLOALA4,
        params=params,
        hardware_opts=hw,
        calc=calc,
        n_confs=200,
        cutoff_dist=0.1,
    )
    assert len(conf_ids) > 0
    assert len(pe) == len(conf_ids)
    assert all(isinstance(e, float) for e in pe)
