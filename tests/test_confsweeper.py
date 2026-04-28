"""Unit tests for src/confsweeper.py."""
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import ase
import numpy as np
import pytest
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedParameters

from confsweeper import (
    _mace_batch_energies,
    get_embed_params,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mol_PE,
    get_mol_PE_batched,
    get_mol_PE_mmff,
    read_csv,
    run_PE_calc,
    write_sdf,
)

TESTS_DIR = Path(__file__).parent
PEPTIDES_CSV = TESTS_DIR / "peptides.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed(mols, params, confsPerMolecule, hardwareOptions):
    """CPU-based drop-in for nvmolkit.embedMolecules.EmbedMolecules."""
    for mol in mols:
        AllChem.EmbedMultipleConfs(mol, numConfs=confsPerMolecule, randomSeed=42)


# ---------------------------------------------------------------------------
# get_embed_params
# ---------------------------------------------------------------------------


def test_get_embed_params_type():
    params = get_embed_params()
    assert isinstance(params, EmbedParameters)


def test_get_embed_params_random_coords():
    params = get_embed_params()
    assert params.useRandomCoords is True


# ---------------------------------------------------------------------------
# get_embed_params_macrocycle
# ---------------------------------------------------------------------------


def test_get_embed_params_macrocycle_type():
    params = get_embed_params_macrocycle()
    assert isinstance(params, EmbedParameters)


def test_get_embed_params_macrocycle_random_coords():
    params = get_embed_params_macrocycle()
    assert params.useRandomCoords is True


def test_get_embed_params_macrocycle_torsions_enabled():
    params = get_embed_params_macrocycle()
    assert params.useMacrocycleTorsions is True


def test_get_embed_params_macrocycle_14config_enabled():
    params = get_embed_params_macrocycle()
    assert params.useMacrocycle14config is True


def test_get_embed_params_macrocycle_small_ring_torsions_disabled():
    """useSmallRingTorsions is intentionally NOT enabled even though it
    would help small rings inside the macrocycle scaffold: nvmolkit hangs
    indefinitely in CPU preprocessing when the flag is set. See the
    docstring of get_embed_params_macrocycle for the rationale."""
    params = get_embed_params_macrocycle()
    assert params.useSmallRingTorsions is False


def test_get_embed_params_macrocycle_enables_macrocycle_torsion_flags():
    """The macrocycle helper differs from the default in turning on the two
    macrocycle-specific torsion flags. useSmallRingTorsions is left off in
    both (see test_get_embed_params_macrocycle_small_ring_torsions_disabled)."""
    default = get_embed_params()
    macro = get_embed_params_macrocycle()
    assert macro.useMacrocycleTorsions is True
    assert macro.useMacrocycle14config is True
    # useMacrocycleTorsions defaults to True in ETKDGv3 too, so the helper's
    # only behavioural difference vs get_embed_params is the explicit setting
    # of these two flags (and the explicit useRandomCoords=True).
    assert macro.useRandomCoords == default.useRandomCoords


# ---------------------------------------------------------------------------
# get_hardware_opts
# ---------------------------------------------------------------------------


def test_get_hardware_opts_defaults():
    opts = get_hardware_opts()
    assert opts.batchSize == 500
    assert opts.preprocessingThreads == 4
    assert opts.batchesPerGpu == 1
    assert opts.gpuIds == [0]


def test_get_hardware_opts_custom():
    opts = get_hardware_opts(
        preprocessingThreads=4,
        batch_size=128,
        batchesPerGpu=2,
        gpuIds=[1, 2],
    )
    assert opts.preprocessingThreads == 4
    assert opts.batchSize == 128
    assert opts.batchesPerGpu == 2
    assert opts.gpuIds == [1, 2]


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------


def test_read_csv_assigns_uuid_column(tmp_path):
    csv_copy = tmp_path / "peptides.csv"
    shutil.copy(PEPTIDES_CSV, csv_copy)
    df = read_csv(csv_copy, write_uuids=True)
    assert "uuid" in df.columns
    assert df["uuid"].notna().all()
    assert len(df["uuid"]) == 6


def test_read_csv_writes_uuids_file(tmp_path):
    csv_copy = tmp_path / "peptides.csv"
    shutil.copy(PEPTIDES_CSV, csv_copy)
    read_csv(csv_copy, write_uuids=True)
    assert (tmp_path / "peptides_uuids.csv").exists()


def test_read_csv_no_write_uuids(tmp_path):
    csv_copy = tmp_path / "peptides.csv"
    shutil.copy(PEPTIDES_CSV, csv_copy)
    read_csv(csv_copy, write_uuids=False)
    assert not (tmp_path / "peptides_uuids.csv").exists()


# ---------------------------------------------------------------------------
# get_mol_PE
# ---------------------------------------------------------------------------

ETHANE_SMILES = "CC"
TEST_UUID = "test-uuid-0001"


@pytest.fixture
def mol_pe_mocks():
    """Patches for GPU embedding and MACE energy calculation."""
    with patch(
        "confsweeper.embed.EmbedMolecules", side_effect=_mock_embed
    ), patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):
        yield


def test_get_mol_PE_creates_sdf(tmp_path, mol_pe_mocks):
    params = get_embed_params()
    hardware_opts = get_hardware_opts()
    mock_calc = MagicMock()
    mock_calc.get_potential_energy = MagicMock(return_value=-100.0)

    mol, conf_ids, pe = get_mol_PE(
        smi=ETHANE_SMILES,
        params=params,
        hardware_opts=hardware_opts,
        calc=mock_calc,
        n_confs=5,
    )
    write_sdf(mol, conf_ids, pe, TEST_UUID, tmp_path, save_lowest_energy=False)

    sdf_path = tmp_path / f"{TEST_UUID}.sdf"
    assert sdf_path.exists()

    from rdkit.Chem import SDMolSupplier

    suppl = list(SDMolSupplier(str(sdf_path), removeHs=False))
    assert len(suppl) >= 1
    for mol in suppl:
        assert mol is not None
        assert mol.HasProp("MACE_ENERGY")


def test_get_mol_PE_save_lowest_energy(tmp_path, mol_pe_mocks):
    params = get_embed_params()
    hardware_opts = get_hardware_opts()
    mock_calc = MagicMock()
    mock_calc.get_potential_energy = MagicMock(return_value=-100.0)

    mol, conf_ids, pe = get_mol_PE(
        smi=ETHANE_SMILES,
        params=params,
        hardware_opts=hardware_opts,
        calc=mock_calc,
        n_confs=5,
    )
    write_sdf(mol, conf_ids, pe, TEST_UUID, tmp_path, save_lowest_energy=True)

    sdf_path = tmp_path / f"{TEST_UUID}.sdf"
    assert sdf_path.exists()

    from rdkit.Chem import SDMolSupplier

    suppl = [m for m in SDMolSupplier(str(sdf_path), removeHs=False) if m is not None]
    assert len(suppl) == 1


# ---------------------------------------------------------------------------
# run_PE_calc CLI
# ---------------------------------------------------------------------------


def test_run_PE_calc_cli(tmp_path):
    csv_copy = tmp_path / "peptides.csv"
    shutil.copy(PEPTIDES_CSV, csv_copy)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("confsweeper.get_uma_calc", return_value=MagicMock()), patch(
        "confsweeper.embed.EmbedMolecules", side_effect=_mock_embed
    ), patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0):

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            run_PE_calc,
            ["--smi_csv", str(csv_copy), "--output_dir", str(output_dir)],
        )

    assert result.exit_code == 0, result.output
    sdf_files = list(output_dir.glob("*.sdf"))
    assert len(sdf_files) == 6


# ---------------------------------------------------------------------------
# get_mol_PE — CPU clustering path
# ---------------------------------------------------------------------------


def test_get_mol_PE_cpu_clustering(mol_pe_mocks):
    mol, conf_ids, pe = get_mol_PE(
        smi=ETHANE_SMILES,
        params=get_embed_params(),
        hardware_opts=get_hardware_opts(),
        calc=MagicMock(),
        n_confs=5,
        gpu_clustering=False,
    )
    assert isinstance(conf_ids, list)
    assert len(pe) == len(conf_ids)
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


# ---------------------------------------------------------------------------
# Helpers shared by get_mol_PE_batched and get_mol_PE_mmff tests
# ---------------------------------------------------------------------------

_MOCK_GRIDS = {
    "L": np.ones((36, 36)) / (36 * 36),
    "D": np.ones((36, 36)) / (36 * 36),
    "NMe": np.ones((36, 36)) / (36 * 36),
    "Gly": np.ones((36, 36)) / (36 * 36),
    "bin_centers": np.linspace(-175, 175, 36),
    "bin_edges": np.linspace(-180, 180, 37),
    "n_bins": np.int64(36),
}

_GRIDS_PATH = Path(__file__).parents[1] / "data/processed/cremp/ramachandran_grids.npz"


def _mock_butina_gpu(dists, cutoff, return_centroids):
    """Return row index 0 as the sole centroid."""
    mock_centroids = MagicMock()
    mock_centroids.numpy.return_value = np.array([0])
    return MagicMock(), mock_centroids


def _mock_butina_last(dists, cutoff, return_centroids):
    """Return the last row index as the sole centroid."""
    n = dists.shape[0]
    mock_centroids = MagicMock()
    mock_centroids.numpy.return_value = np.array([n - 1])
    return MagicMock(), mock_centroids


@pytest.fixture
def batched_mocks():
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_gpu),
        patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0),
    ):
        yield


# ---------------------------------------------------------------------------
# get_mol_PE_batched
# ---------------------------------------------------------------------------


def test_get_mol_PE_batched_returns_triple(batched_mocks):
    mol, conf_ids, pe = get_mol_PE_batched(
        smi=ETHANE_SMILES,
        params=get_embed_params(),
        hardware_opts=get_hardware_opts(),
        calc=MagicMock(),
        n_confs=5,
    )
    assert isinstance(conf_ids, list)
    assert len(pe) == len(conf_ids)
    assert all(isinstance(cid, int) for cid in conf_ids)


def test_get_mol_PE_batched_centroid_ids_valid(batched_mocks):
    mol, conf_ids, pe = get_mol_PE_batched(
        smi=ETHANE_SMILES,
        params=get_embed_params(),
        hardware_opts=get_hardware_opts(),
        calc=MagicMock(),
        n_confs=5,
    )
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


def test_get_mol_PE_batched_only_centroids_remain(batched_mocks):
    mol, conf_ids, pe = get_mol_PE_batched(
        smi=ETHANE_SMILES,
        params=get_embed_params(),
        hardware_opts=get_hardware_opts(),
        calc=MagicMock(),
        n_confs=5,
    )
    assert mol.GetNumConformers() == len(conf_ids)


def test_get_mol_PE_batched_empty_pool():
    with patch("confsweeper.embed.EmbedMolecules"):  # no-op: embeds nothing
        mol, conf_ids, pe = get_mol_PE_batched(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            calc=MagicMock(),
            n_confs=5,
        )
    assert conf_ids == []
    assert pe == []


def test_get_mol_PE_batched_cpu_clustering(batched_mocks):
    mol, conf_ids, pe = get_mol_PE_batched(
        smi=ETHANE_SMILES,
        params=get_embed_params(),
        hardware_opts=get_hardware_opts(),
        calc=MagicMock(),
        n_confs=5,
        gpu_clustering=False,
    )
    assert isinstance(conf_ids, list)
    assert len(pe) == len(conf_ids)
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    for cid in conf_ids:
        assert cid in valid_ids


def test_get_mol_PE_batched_torsional_sample_called(batched_mocks):
    """sample_constrained_confs is invoked when grids and n_constrained_samples are set."""
    with patch("confsweeper.sample_constrained_confs", return_value=[]) as mock_sc:
        get_mol_PE_batched(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            calc=MagicMock(),
            n_confs=5,
            grids=_MOCK_GRIDS,
            n_constrained_samples=10,
            torsion_strategy="inverse",
        )
    mock_sc.assert_called_once()


def test_get_mol_PE_batched_torsional_not_called_without_grids(batched_mocks):
    """sample_constrained_confs is NOT invoked when grids=None."""
    with patch("confsweeper.sample_constrained_confs") as mock_sc:
        get_mol_PE_batched(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            calc=MagicMock(),
            n_confs=5,
            grids=None,
            n_constrained_samples=10,
        )
    mock_sc.assert_not_called()


def test_get_mol_PE_batched_pool_b_non_centroids_removed():
    """Pool B conformers that are not Butina centroids must be pruned from mol.

    This is a regression test for the pool-merge bug: the old to_remove loop
    only iterated range(n_confs), leaving Pool B conformer IDs (which start at
    n_pool_a) orphaned on the mol.
    """

    def _add_pool_b(mol, grids, n_samples, **kwargs):
        AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=99)
        return [c.GetId() for c in mol.GetConformers()][-2:]

    # Butina selects only row 0 (Pool A ID 0); Pool B IDs 3 and 4 must be removed.
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_gpu),
        patch("confsweeper.sample_constrained_confs", side_effect=_add_pool_b),
        patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0),
    ):
        mol, conf_ids, pe = get_mol_PE_batched(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            calc=MagicMock(),
            n_confs=3,
            grids=_MOCK_GRIDS,
            n_constrained_samples=2,
        )

    assert mol.GetNumConformers() == 1
    assert mol.GetConformers()[0].GetId() == conf_ids[0]


def test_get_mol_PE_batched_pool_b_centroid_id_correct():
    """When Butina selects a Pool B conformer, its RDKit conf ID is returned correctly.

    Pool B conformer IDs start at n_pool_a, so the Butina row index (n_pool_a + k)
    must be mapped back through all_conf_ids to the actual conformer ID.
    """
    pool_b_ids: list[int] = []

    def _add_pool_b(mol, grids, n_samples, **kwargs):
        # Add a conformer directly rather than via ETKDG — ethane is too simple for
        # EmbedMultipleConfs to reliably produce a new conformer when one already exists.
        from rdkit.Chem import Conformer

        n_atoms = mol.GetNumAtoms()
        conf = Conformer(n_atoms)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        new_id = mol.AddConformer(conf, assignId=True)
        pool_b_ids.append(new_id)
        return [new_id]

    # _mock_butina_last selects the last row — which must be the Pool B conformer.
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_last),
        patch("confsweeper.sample_constrained_confs", side_effect=_add_pool_b),
        patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0),
    ):
        mol, conf_ids, pe = get_mol_PE_batched(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            calc=MagicMock(),
            n_confs=3,
            grids=_MOCK_GRIDS,
            n_constrained_samples=1,
        )

    assert len(conf_ids) == 1
    valid_ids = {c.GetId() for c in mol.GetConformers()}
    assert conf_ids[0] in valid_ids
    # The selected centroid must be one of the Pool B conformers we tracked.
    assert pool_b_ids, "Pool B must have produced at least one conformer"
    assert conf_ids[0] in pool_b_ids


# ---------------------------------------------------------------------------
# _mace_batch_energies
# ---------------------------------------------------------------------------


def test_mace_batch_energies_fallback():
    """When the MACE batch API is unavailable, fall back to sequential scoring."""
    calc = MagicMock()
    calc._atoms_to_batch.side_effect = AttributeError("no batch API")

    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC")
    atoms = [a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms()]
    n = len(atoms)
    ase_mols = [ase.Atoms(numbers=atoms, positions=np.zeros((n, 3))) for _ in range(3)]

    with patch.object(ase.Atoms, "get_potential_energy", return_value=-50.0):
        energies = _mace_batch_energies(calc, ase_mols)

    assert len(energies) == 3
    assert all(e == -50.0 for e in energies)


# ---------------------------------------------------------------------------
# get_mol_PE_mmff
# ---------------------------------------------------------------------------


def test_get_mol_PE_mmff_returns_triple():
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_gpu),
    ):
        mol, conf_ids, pe = get_mol_PE_mmff(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            n_confs=5,
        )
    assert isinstance(conf_ids, list)
    assert len(pe) == len(conf_ids)
    assert all(isinstance(e, float) for e in pe)


def test_get_mol_PE_mmff_only_centroids_remain():
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_gpu),
    ):
        mol, conf_ids, pe = get_mol_PE_mmff(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            n_confs=5,
        )
    assert mol.GetNumConformers() == len(conf_ids)


def test_get_mol_PE_mmff_empty_embed():
    with patch("confsweeper.embed.EmbedMolecules"):  # no-op
        mol, conf_ids, pe = get_mol_PE_mmff(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            n_confs=5,
        )
    assert conf_ids == []
    assert pe == []


def test_get_mol_PE_mmff_cpu_clustering():
    with (patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),):
        mol, conf_ids, pe = get_mol_PE_mmff(
            smi=ETHANE_SMILES,
            params=get_embed_params(),
            hardware_opts=get_hardware_opts(),
            n_confs=5,
            gpu_clustering=False,
        )
    assert isinstance(conf_ids, list)
    assert len(pe) == len(conf_ids)


# ---------------------------------------------------------------------------
# run_PE_calc CLI — etkdg+torsional mode
# ---------------------------------------------------------------------------


def test_run_PE_calc_cli_torsional_mode(tmp_path):
    if not _GRIDS_PATH.exists():
        pytest.skip(
            "CREMP Ramachandran grids not found; run build_ramachandran_grids.py first"
        )

    csv_copy = tmp_path / "peptides.csv"
    shutil.copy(PEPTIDES_CSV, csv_copy)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with (
        patch("confsweeper.get_uma_calc", return_value=MagicMock()),
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed),
        patch("confsweeper.sample_constrained_confs", return_value=[]),
        patch("confsweeper.clustering.butina", side_effect=_mock_butina_gpu),
        patch.object(ase.Atoms, "get_potential_energy", return_value=-100.0),
    ):
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            run_PE_calc,
            [
                "--smi_csv",
                str(csv_copy),
                "--output_dir",
                str(output_dir),
                "--sampling_mode",
                "etkdg+torsional",
                "--n_constrained_samples",
                "5",
                "--ramachandran_grids",
                str(_GRIDS_PATH),
            ],
        )

    assert result.exit_code == 0, result.output
    sdf_files = list(output_dir.glob("*.sdf"))
    assert len(sdf_files) == 6
