"""Unit tests for src/validation/make_validation_sets_cremp.py."""
import numpy as np
import pandas as pd
from click.testing import CliRunner

from validation.make_validation_sets_cremp import (
    assign_atom_bin,
    main,
    parse_topology,
    sample_subset,
)

# ---------------------------------------------------------------------------
# parse_topology
# ---------------------------------------------------------------------------


class TestParseTopology:
    def test_all_L(self):
        assert parse_topology("A.V.G.L") == "all-L"

    def test_d_only(self):
        # single lowercase residue → D-only
        assert parse_topology("A.v.G.L") == "D-only"

    def test_d_only_all_lower(self):
        assert parse_topology("a.v.g.l") == "D-only"

    def test_nme_only(self):
        # MeA, MeV → N-methylated L-residues
        assert parse_topology("MeA.MeV.G.L") == "NMe-only"

    def test_d_plus_nme_via_methylated_d(self):
        # 'Mei' = N-methylated D-isoleucine (lowercase after 'Me')
        assert parse_topology("MeA.Mei.G.L") == "D+NMe"

    def test_d_plus_nme_separate(self):
        # separate D residue and separate NMe residue
        assert parse_topology("a.MeV.G.L") == "D+NMe"

    def test_single_monomer_L(self):
        assert parse_topology("G") == "all-L"

    def test_single_monomer_D(self):
        assert parse_topology("g") == "D-only"

    def test_single_monomer_NMe(self):
        assert parse_topology("MeG") == "NMe-only"

    def test_single_monomer_NMe_D(self):
        assert parse_topology("Meg") == "D+NMe"

    def test_real_cremp_sequences(self):
        assert parse_topology("N.A.P.A") == "all-L"
        assert parse_topology("f.I.N.G") == "D-only"
        assert parse_topology("MeV.MeT.MeV.Q") == "NMe-only"
        assert parse_topology("MeA.Mei.F.S") == "D+NMe"


# ---------------------------------------------------------------------------
# assign_atom_bin
# ---------------------------------------------------------------------------


def _make_df(num_monomers_list, num_heavy_atoms_list):
    return pd.DataFrame(
        {"num_monomers": num_monomers_list, "num_heavy_atoms": num_heavy_atoms_list}
    )


class TestAssignAtomBin:
    def test_three_bins_produced(self):
        df = _make_df([4] * 9, list(range(9)))
        bins = assign_atom_bin(df)
        assert set(bins.unique()) == {"small", "medium", "large"}

    def test_bins_computed_within_monomer_class(self):
        # 4-mers: atoms 10-12, 5-mers: atoms 50-52 — bins should be independent
        df = _make_df(
            [4, 4, 4, 5, 5, 5],
            [10, 11, 12, 50, 51, 52],
        )
        bins = assign_atom_bin(df)
        # Both classes should have one of each bin
        assert sorted(bins[df["num_monomers"] == 4].tolist()) == [
            "large",
            "medium",
            "small",
        ]
        assert sorted(bins[df["num_monomers"] == 5].tolist()) == [
            "large",
            "medium",
            "small",
        ]

    def test_returns_series_with_same_index(self):
        df = _make_df([4] * 6, [10, 20, 30, 40, 50, 60])
        bins = assign_atom_bin(df)
        assert list(bins.index) == list(df.index)


# ---------------------------------------------------------------------------
# sample_subset
# ---------------------------------------------------------------------------

_SEQS = {
    "all-L": "A.V.G.L",
    "D-only": "a.V.G.L",
    "NMe-only": "MeA.V.G.L",
    "D+NMe": "MeA.v.G.L",
}


def _make_summary_csv(tmp_path, n_per_cell=20):
    """
    Builds a minimal summary CSV covering all 36 strata.
    Heavy atoms are spread across 3 tertile ranges per monomer class
    so that assign_atom_bin produces all three bins.
    """
    rows = []
    # Use distinct num_heavy_atoms ranges per monomer count to guarantee all 3 bins
    atom_ranges = {4: (20, 40), 5: (35, 55), 6: (50, 70)}
    for n_mon in [4, 5, 6]:
        lo, hi = atom_ranges[n_mon]
        atoms = np.linspace(lo, hi, n_per_cell * 4, dtype=int)
        for seq in _SEQS.values():
            for a in atoms:
                rows.append(
                    {
                        "sequence": seq,
                        "smiles": "C",  # placeholder
                        "num_monomers": n_mon,
                        "num_atoms": a + 10,
                        "num_heavy_atoms": int(a),
                        "totalconfs": 100,
                        "uniqueconfs": 50,
                        "lowestenergy": -80.0,
                        "poplowestpct": 20.0,
                        "temperature": 298.15,
                        "ensembleenergy": 0.5,
                        "ensembleentropy": 7.0,
                        "ensemblefreeenergy": -2.0,
                    }
                )
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class TestSampleSubset:
    def test_returns_dataframe(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        result = sample_subset(csv, n_per_stratum=5, seed=0)
        assert isinstance(result, pd.DataFrame)

    def test_has_topology_and_atom_bin_columns(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        result = sample_subset(csv, n_per_stratum=5, seed=0)
        assert "topology" in result.columns
        assert "atom_bin" in result.columns

    def test_all_36_strata_present(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        result = sample_subset(csv, n_per_stratum=5, seed=0)
        strata = result.groupby(["topology", "num_monomers", "atom_bin"]).size()
        assert len(strata) == 36

    def test_n_per_stratum_respected(self, tmp_path):
        csv = _make_summary_csv(tmp_path, n_per_cell=20)
        n = 7
        result = sample_subset(csv, n_per_stratum=n, seed=0)
        strata = result.groupby(["topology", "num_monomers", "atom_bin"]).size()
        assert (strata <= n).all()
        assert (strata == n).all()  # all cells have >= 7 members

    def test_reproducible_with_same_seed(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        r1 = sample_subset(csv, n_per_stratum=5, seed=99)
        r2 = sample_subset(csv, n_per_stratum=5, seed=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_give_different_results(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        r1 = sample_subset(csv, n_per_stratum=5, seed=1)
        r2 = sample_subset(csv, n_per_stratum=5, seed=2)
        # num_heavy_atoms varies within strata, so different seeds should pick different rows
        assert not r1["num_heavy_atoms"].equals(r2["num_heavy_atoms"])

    def test_small_stratum_included_in_full(self, tmp_path):
        # Build a CSV where one stratum has only 3 members
        rows = []
        atom_ranges = {4: (20, 40), 5: (35, 55), 6: (50, 70)}
        for n_mon in [4, 5, 6]:
            lo, hi = atom_ranges[n_mon]
            atoms = np.linspace(lo, hi, 20 * 4, dtype=int)
            for topo, seq in _SEQS.items():
                n_rows = 3 if (n_mon == 6 and topo == "D-only") else 20
                for a in atoms[:n_rows]:
                    rows.append(
                        {
                            "sequence": seq,
                            "smiles": "C",
                            "num_monomers": n_mon,
                            "num_atoms": int(a) + 10,
                            "num_heavy_atoms": int(a),
                            "totalconfs": 100,
                            "uniqueconfs": 50,
                            "lowestenergy": -80.0,
                            "poplowestpct": 20.0,
                            "temperature": 298.15,
                            "ensembleenergy": 0.5,
                            "ensembleentropy": 7.0,
                            "ensemblefreeenergy": -2.0,
                        }
                    )
        csv_path = tmp_path / "summary_small.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        result = sample_subset(csv_path, n_per_stratum=10, seed=0)
        # The small stratum (6-mer, D-only) should have ≤3 rows, not be dropped
        small = result[(result["num_monomers"] == 6) & (result["topology"] == "D-only")]
        assert len(small) > 0


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_creates_output_csv(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        out = tmp_path / "out" / "subset.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--summary_csv",
                str(csv),
                "--output_csv",
                str(out),
                "--n_per_stratum",
                "5",
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) > 0

    def test_output_contains_expected_columns(self, tmp_path):
        csv = _make_summary_csv(tmp_path)
        out = tmp_path / "subset.csv"
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "--summary_csv",
                str(csv),
                "--output_csv",
                str(out),
                "--n_per_stratum",
                "5",
                "--seed",
                "0",
            ],
        )
        df = pd.read_csv(out)
        for col in ["sequence", "smiles", "topology", "atom_bin", "num_monomers"]:
            assert col in df.columns
