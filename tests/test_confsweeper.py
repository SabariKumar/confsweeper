"""Unit tests for src/confsweeper.py."""
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import ase
import pytest
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedParameters

from confsweeper import (
    get_embed_params,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mol_PE,
    read_csv,
    run_PE_calc,
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


def test_get_embed_params_macrocycle_small_ring_torsions_enabled():
    params = get_embed_params_macrocycle()
    assert params.useSmallRingTorsions is True


def test_get_embed_params_macrocycle_differs_from_default():
    # useSmallRingTorsions is the key addition — off by default in ETKDGv3
    default = get_embed_params()
    macro = get_embed_params_macrocycle()
    assert macro.useSmallRingTorsions != default.useSmallRingTorsions


# ---------------------------------------------------------------------------
# get_hardware_opts
# ---------------------------------------------------------------------------


def test_get_hardware_opts_defaults():
    opts = get_hardware_opts()
    assert opts.batchSize == 500
    assert opts.preprocessingThreads == 16
    assert opts.batchesPerGpu == 16
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

    get_mol_PE(
        smi=ETHANE_SMILES,
        uuid=TEST_UUID,
        output_dir=tmp_path,
        params=params,
        hardware_opts=hardware_opts,
        mace_calc=mock_calc,
        n_confs=5,
    )

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

    get_mol_PE(
        smi=ETHANE_SMILES,
        uuid=TEST_UUID,
        output_dir=tmp_path,
        params=params,
        hardware_opts=hardware_opts,
        mace_calc=mock_calc,
        n_confs=5,
        save_lowest_energy=True,
    )

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

    with patch("confsweeper.get_mace_calc", return_value=MagicMock()), patch(
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
