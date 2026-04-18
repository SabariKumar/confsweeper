"""Unit tests for src/validation/cremp_coverage.py."""
import csv
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner
from rdkit import Chem
from rdkit.Chem import AllChem

from validation.cremp_coverage import (
    ERROR_COLUMNS,
    OUTPUT_COLUMNS,
    _append_row,
    _load_checkpoint,
    run_coverage_benchmark,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mol(smiles: str, n_confs: int, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed)
    return mol


ETHANOL = "CCO"


def _write_pickle(path, smiles=ETHANOL, n_confs=3):
    mol = _make_mol(smiles, n_confs)
    data = {
        "rd_mol": mol,
        "smiles": smiles,
        "uniqueconfs": n_confs,
        "lowestenergy": -50.0,
        "charge": 0,
        "conformers": [],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return mol


def _make_subset_csv(tmp_path, sequences, smiles=ETHANOL):
    rows = [
        {
            "sequence": s,
            "smiles": smiles,
            "topology": "all-L",
            "atom_bin": "small",
            "num_monomers": 4,
            "num_heavy_atoms": 3,
        }
        for s in sequences
    ]
    path = tmp_path / "subset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# _load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_empty_when_file_missing(self, tmp_path):
        done = _load_checkpoint(str(tmp_path / "nonexistent.csv"))
        assert done == set()

    def test_loads_existing_pairs(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            w.writeheader()
            w.writerow(
                {c: "" for c in OUTPUT_COLUMNS}
                | {"sequence": "A.V.G.L", "n_confs": "500"}
            )
            w.writerow(
                {c: "" for c in OUTPUT_COLUMNS}
                | {"sequence": "a.V.G.L", "n_confs": "1000"}
            )
        done = _load_checkpoint(str(csv_path))
        assert ("A.V.G.L", 500, "etkdg") in done
        assert ("a.V.G.L", 1000, "etkdg") in done

    def test_n_confs_parsed_as_int(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            w.writeheader()
            w.writerow(
                {c: "" for c in OUTPUT_COLUMNS} | {"sequence": "X", "n_confs": "2000"}
            )
        done = _load_checkpoint(str(csv_path))
        assert ("X", 2000, "etkdg") in done
        assert ("X", "2000", "etkdg") not in done  # must be int


# ---------------------------------------------------------------------------
# _append_row
# ---------------------------------------------------------------------------


class TestAppendRow:
    def test_creates_file_with_header(self, tmp_path):
        path = str(tmp_path / "out.csv")
        _append_row(path, {"a": 1, "b": 2}, ["a", "b"])
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 1
        assert rows[0] == {"a": "1", "b": "2"}

    def test_appends_without_duplicate_header(self, tmp_path):
        path = str(tmp_path / "out.csv")
        _append_row(path, {"a": 1, "b": 2}, ["a", "b"])
        _append_row(path, {"a": 3, "b": 4}, ["a", "b"])
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 2

    def test_does_not_overwrite_existing(self, tmp_path):
        path = str(tmp_path / "out.csv")
        _append_row(path, {"a": 1}, ["a"])
        _append_row(path, {"a": 2}, ["a"])
        values = [r["a"] for r in csv.DictReader(open(path))]
        assert values == ["1", "2"]


# ---------------------------------------------------------------------------
# run_coverage_benchmark CLI
# ---------------------------------------------------------------------------


def _mock_get_mol_PE(
    smi, params, hardware_opts, calc, n_confs, cutoff_dist, gpu_clustering
):
    """Returns a mol with n_confs conformers and dummy PE values."""
    mol = _make_mol(smi, min(n_confs, 3))
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    pe = [0.0] * len(conf_ids)
    return mol, conf_ids, pe


class TestRunCoverageBenchmark:
    def _run(self, tmp_path, sequences=("A.V.G.L",), n_confs="1000", extra_args=None):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        for seq in sequences:
            _write_pickle(pkl_dir / f"{seq}.pickle")

        subset_csv = _make_subset_csv(tmp_path, sequences)
        output_csv = str(tmp_path / "coverage.csv")
        errors_csv = str(tmp_path / "errors.csv")

        runner = CliRunner()
        args = [
            "--subset_csv",
            str(subset_csv),
            "--pickle_dir",
            str(pkl_dir),
            "--output_csv",
            output_csv,
            "--errors_csv",
            errors_csv,
            "--n_confs",
            n_confs,
            "--rmsd_cutoff",
            "1.0",
        ]
        if extra_args:
            args += extra_args

        with patch("validation.cremp_coverage.get_embed_params_macrocycle"), patch(
            "validation.cremp_coverage.get_hardware_opts"
        ), patch("validation.cremp_coverage.get_uma_calc"), patch(
            "validation.cremp_coverage.get_mol_PE", side_effect=_mock_get_mol_PE
        ):
            result = runner.invoke(run_coverage_benchmark, args)

        return result, output_csv, errors_csv

    def test_creates_output_csv(self, tmp_path):
        result, output_csv, _ = self._run(tmp_path)
        assert result.exit_code == 0, result.output
        assert Path(output_csv).exists()

    def test_output_has_correct_columns(self, tmp_path):
        _, output_csv, _ = self._run(tmp_path)
        df = pd.read_csv(output_csv)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns

    def test_one_row_per_molecule_per_n_confs(self, tmp_path):
        _, output_csv, _ = self._run(
            tmp_path, sequences=("A.V.G.L", "a.V.G.L"), n_confs="500,1000"
        )
        df = pd.read_csv(output_csv)
        assert len(df) == 4  # 2 molecules × 2 n_confs values

    def test_coverage_between_zero_and_one(self, tmp_path):
        _, output_csv, _ = self._run(tmp_path)
        df = pd.read_csv(output_csv)
        assert ((df["coverage"] >= 0.0) & (df["coverage"] <= 1.0)).all()

    def test_n_confs_column_matches_requested(self, tmp_path):
        _, output_csv, _ = self._run(tmp_path, n_confs="500,2000")
        df = pd.read_csv(output_csv)
        assert set(df["n_confs"].tolist()) == {500, 2000}

    def test_checkpoint_skips_done_pairs(self, tmp_path):
        # Pre-populate output CSV with one (sequence, n_confs) pair
        output_csv = str(tmp_path / "coverage.csv")
        pre_row = {c: "" for c in OUTPUT_COLUMNS}
        pre_row.update({"sequence": "A.V.G.L", "n_confs": "1000", "coverage": "0.99"})
        _append_row(output_csv, pre_row, OUTPUT_COLUMNS)

        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        _write_pickle(pkl_dir / "A.V.G.L.pickle")
        subset_csv = _make_subset_csv(tmp_path, ["A.V.G.L"])
        errors_csv = str(tmp_path / "errors.csv")

        call_count = {"n": 0}

        def counting_mock(*args, **kwargs):
            call_count["n"] += 1
            return _mock_get_mol_PE(*args, **kwargs)

        runner = CliRunner()
        with patch("validation.cremp_coverage.get_embed_params_macrocycle"), patch(
            "validation.cremp_coverage.get_hardware_opts"
        ), patch("validation.cremp_coverage.get_uma_calc"), patch(
            "validation.cremp_coverage.get_mol_PE", side_effect=counting_mock
        ):
            runner.invoke(
                run_coverage_benchmark,
                [
                    "--subset_csv",
                    str(subset_csv),
                    "--pickle_dir",
                    str(pkl_dir),
                    "--output_csv",
                    output_csv,
                    "--errors_csv",
                    errors_csv,
                    "--n_confs",
                    "1000",
                ],
            )

        # Already done — get_mol_PE should not have been called
        assert call_count["n"] == 0

    def test_errors_logged_on_failure(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        _write_pickle(pkl_dir / "A.V.G.L.pickle")
        subset_csv = _make_subset_csv(tmp_path, ["A.V.G.L"])
        output_csv = str(tmp_path / "coverage.csv")
        errors_csv = str(tmp_path / "errors.csv")

        runner = CliRunner()
        with patch("validation.cremp_coverage.get_embed_params_macrocycle"), patch(
            "validation.cremp_coverage.get_hardware_opts"
        ), patch("validation.cremp_coverage.get_uma_calc"), patch(
            "validation.cremp_coverage.get_mol_PE", side_effect=RuntimeError("GPU OOM")
        ):
            runner.invoke(
                run_coverage_benchmark,
                [
                    "--subset_csv",
                    str(subset_csv),
                    "--pickle_dir",
                    str(pkl_dir),
                    "--output_csv",
                    output_csv,
                    "--errors_csv",
                    errors_csv,
                    "--n_confs",
                    "1000",
                ],
            )

        assert Path(errors_csv).exists()
        errors = pd.read_csv(errors_csv)
        assert len(errors) == 1
        assert errors.iloc[0]["sequence"] == "A.V.G.L"
        assert "GPU OOM" in errors.iloc[0]["error"]

    def test_errors_do_not_halt_remaining_molecules(self, tmp_path):
        pkl_dir = tmp_path / "pickle"
        pkl_dir.mkdir()
        for seq in ["A.V.G.L", "a.V.G.L"]:
            _write_pickle(pkl_dir / f"{seq}.pickle")
        subset_csv = _make_subset_csv(tmp_path, ["A.V.G.L", "a.V.G.L"])
        output_csv = str(tmp_path / "coverage.csv")
        errors_csv = str(tmp_path / "errors.csv")

        call_count = {"n": 0}

        def fail_first(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first molecule fails")
            return _mock_get_mol_PE(*args, **kwargs)

        runner = CliRunner()
        with patch("validation.cremp_coverage.get_embed_params_macrocycle"), patch(
            "validation.cremp_coverage.get_hardware_opts"
        ), patch("validation.cremp_coverage.get_uma_calc"), patch(
            "validation.cremp_coverage.get_mol_PE", side_effect=fail_first
        ):
            result = runner.invoke(
                run_coverage_benchmark,
                [
                    "--subset_csv",
                    str(subset_csv),
                    "--pickle_dir",
                    str(pkl_dir),
                    "--output_csv",
                    output_csv,
                    "--errors_csv",
                    errors_csv,
                    "--n_confs",
                    "1000",
                ],
            )

        assert result.exit_code == 0
        # Second molecule should still be in output
        df = pd.read_csv(output_csv)
        assert len(df) == 1
        assert df.iloc[0]["sequence"] == "a.V.G.L"
