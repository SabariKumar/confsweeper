"""
Saturation experiment for get_mol_PE_exhaustive: how do BW uniformity and
basin count scale with the number of ETKDG seeds, and at what N do the
curves flatten?

This is the Phase 2 diagnostic for picking a production default for
`n_seeds`. Results are written one row at a time to a CSV so a long sweep
can be resumed if interrupted.

Selection of representative peptides (5 by default):
  * 2 from CREMP (validation subset) — one near the median CREMP
    `poplowestpct` (~25 %, "typical rich ensemble") and one in the high tail
    (~75–90 %, "sharper landscape than usual"). CREMP gives us CREST-
    derived ground truth (`poplowestpct`, `uniqueconfs`) for the same
    peptides we benchmark, so the saturation curve can be sanity-checked
    against CREST.
  * 3 from PAMPA (CycPeptMPDB-deduped) — one small (~50 heavy atoms),
    one medium (~70), one large (~100+). PAMPA is the actual fine-tuning
    consumer, so saturating the curve here is the binding goal.

Per (peptide, n_seeds, minimize) row, the script reports:
    n_pool          number of conformers actually embedded
    n_basins        cluster representatives surviving energy filter + dedup
    max_bw          weight of the dominant conformer in the basin set
    eff_n           1 / Σ w_i² over the basin set
    entropy         -Σ w_i log w_i over the basin set
    n_within_kT     basins with E - E_min ≤ kT (≈ 26 meV at 298 K)
    n_within_3kT    basins with E - E_min ≤ 3·kT
    e_min_eV        minimum MACE energy seen in the pool
    time_*          wall-clock per stage in seconds

Usage (in the confsweeper pixi environment):

    pixi run python scripts/saturation_etkdg.py \\
        --cremp_csv data/processed/cremp/validation_subset.csv \\
        --pampa_csv /home/sabari/peptide_electrostatics/data/fine_tune/CycPeptMPDB_PAMPA_deduped.csv \\
        --out_csv  results/saturation_etkdg.csv \\
        --n_seeds_grid 100,500,1000,5000,10000,50000

A `--minimize_at_largest` flag re-runs the largest n_seeds with MMFF94
minimization on each peptide, so the ablation lands in the same CSV.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

# fmt: off
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rdkit.Chem.rdDistGeom import ETKDG  # noqa: E402

from confsweeper import (  # noqa: E402
    _KT_EV_298K,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE_exhaustive,
)

# fmt: on


def _build_params(mode: str):
    """
    Build ETKDG parameters for the given saturation mode.

    Two modes are supported:
        etkdgv3_macrocycle — confsweeper's default macrocycle-aware ETKDGv3
            (useMacrocycleTorsions / useMacrocycle14config on). Same params as
            the production fine-tuning pipeline; this is the baseline.
        etkdg_original — RDKit's original 2015 ETKDG. No macrocycle-specific
            sampling bias and weaker torsion-knowledge prior, intended to test
            whether ETKDGv3's torsion bias is causing oversampling of the
            global minimum at the expense of diverse low-energy basins.

    Empirically verified: nvmolkit accepts all three of ETKDG, ETKDGv2,
    ETKDGv3 without hanging or producing zero conformers (small linear
    peptide, 10 conformers each).

    Params:
        mode: str : one of 'etkdgv3_macrocycle' or 'etkdg_original'
    Returns:
        rdkit.Chem.rdDistGeom.EmbedParameters : parameter object ready for nvmolkit
    """
    if mode == "etkdgv3_macrocycle":
        return get_embed_params_macrocycle()
    if mode == "etkdg_original":
        params = ETKDG()
        params.useRandomCoords = True
        return params
    raise ValueError(f"unknown params_mode: {mode}")


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("saturation")

OUTPUT_COLUMNS = [
    "peptide_id",
    "source",
    "n_heavy",
    "smiles",
    "params_mode",
    "n_seeds",
    "minimize",
    "dihedral_jitter_deg",
    "n_pool",
    "n_basins",
    "max_bw",
    "eff_n",
    "entropy",
    "n_within_kT",
    "n_within_3kT",
    "e_min_eV",
    "time_embed_s",
    "time_score_s",
    "time_filter_dedup_s",
    "time_total_s",
    "ground_truth_max_bw",  # CREMP poplowestpct/100, NaN for PAMPA
    "ground_truth_n_confs",  # CREMP uniqueconfs, NaN for PAMPA
]


def select_cremp_peptides(cremp_csv: Path, n: int = 2) -> list[dict]:
    """
    Pick CREMP peptides spanning the `poplowestpct` distribution.

    Picks the peptide closest to the median `poplowestpct` and the peptide
    closest to the 90th-percentile `poplowestpct` from the validation subset.
    Ties broken by lowest heavy-atom count (cheaper saturation runs first).

    Params:
        cremp_csv: Path : CREMP validation subset CSV
        n: int : number of peptides to return (only n=2 is currently supported)
    Returns:
        list[dict] : peptide rows with keys peptide_id, source, smiles, n_heavy,
            ground_truth_max_bw, ground_truth_n_confs
    """
    if n != 2:
        raise NotImplementedError("Only n=2 CREMP picks supported today")
    df = pd.read_csv(cremp_csv)

    targets = [
        ("cremp_typical", df["poplowestpct"].quantile(0.50)),
        ("cremp_sharp", df["poplowestpct"].quantile(0.90)),
    ]
    picks: list[dict] = []
    for label, target_pct in targets:
        df_sorted = df.assign(
            _delta=(df["poplowestpct"] - target_pct).abs()
        ).sort_values(["_delta", "num_heavy_atoms"])
        row = df_sorted.iloc[0]
        picks.append(
            {
                "peptide_id": f"{label}:{row['sequence']}",
                "source": "cremp",
                "smiles": row["smiles"],
                "n_heavy": int(row["num_heavy_atoms"]),
                "ground_truth_max_bw": float(row["poplowestpct"]) / 100.0,
                "ground_truth_n_confs": int(row["uniqueconfs"]),
            }
        )
    return picks


def select_pampa_peptides(pampa_csv: Path, smiles_col: str = "SMILES") -> list[dict]:
    """
    Pick three PAMPA peptides: small / medium / large heavy-atom count.

    Targets the medians of three heavy-atom buckets: small (≤55), medium
    (60–75), large (≥95). Each bucket contributes the row with median
    `num_heavy_atoms` (ties broken by first occurrence). PAMPA has no
    CREST ground truth, so the ground-truth columns are NaN.

    Params:
        pampa_csv: Path : CycPeptMPDB-deduped CSV
        smiles_col: str : SMILES column name in the CSV
    Returns:
        list[dict] : three peptide rows with keys peptide_id, source, smiles,
            n_heavy, ground_truth_max_bw (NaN), ground_truth_n_confs (NaN)
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    df = pd.read_csv(pampa_csv)
    df["n_heavy"] = df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
        if Chem.MolFromSmiles(s)
        else 0
    )

    buckets = [
        ("pampa_small", df[df["n_heavy"] <= 55]),
        ("pampa_medium", df[(df["n_heavy"] >= 60) & (df["n_heavy"] <= 75)]),
        ("pampa_large", df[df["n_heavy"] >= 95]),
    ]
    picks: list[dict] = []
    for label, sub in buckets:
        if sub.empty:
            logger.warning("PAMPA bucket %s is empty — skipping", label)
            continue
        median_n = int(sub["n_heavy"].median())
        row = (
            sub.assign(_delta=(sub["n_heavy"] - median_n).abs())
            .sort_values(["_delta", "n_heavy"])
            .iloc[0]
        )
        picks.append(
            {
                "peptide_id": label,
                "source": "pampa",
                "smiles": row[smiles_col],
                "n_heavy": int(row["n_heavy"]),
                "ground_truth_max_bw": float("nan"),
                "ground_truth_n_confs": float("nan"),
            }
        )
    return picks


def _read_done_set(out_csv: Path) -> set[tuple]:
    """
    Return the set of (peptide_id, params_mode, n_seeds, minimize,
    dihedral_jitter_deg) tuples already written so the runner can skip them
    on resume.

    Older CSV files written before `params_mode` or `dihedral_jitter_deg`
    were added are read with sensible defaults — `params_mode` defaults to
    'etkdgv3_macrocycle' (the only mode the original sweep ran in) and
    `dihedral_jitter_deg` defaults to 0.0 (jitter off).

    Params:
        out_csv: Path : output CSV path
    Returns:
        set[tuple[str, str, int, bool, float]] : already-completed runs
    """
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if "params_mode" not in df.columns:
        df["params_mode"] = "etkdgv3_macrocycle"
    if "dihedral_jitter_deg" not in df.columns:
        df["dihedral_jitter_deg"] = 0.0
    return set(
        zip(
            df["peptide_id"].astype(str),
            df["params_mode"].astype(str),
            df["n_seeds"].astype(int),
            df["minimize"].astype(bool),
            df["dihedral_jitter_deg"].astype(float),
        )
    )


def _append_row(out_csv: Path, row: dict) -> None:
    """
    Append one result row to the CSV, writing the header on first call.

    Params:
        out_csv: Path : output CSV path
        row: dict : keys must be a subset of OUTPUT_COLUMNS
    Returns:
        None
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _bw_metrics(energies_eV: list[float]) -> dict:
    """
    Compute Boltzmann-weight statistics and energy-window counts for a basin set.

    Params:
        energies_eV: list[float] : MACE energies in eV for the basin centroids
    Returns:
        dict with keys n_basins, max_bw, eff_n, entropy, n_within_kT,
            n_within_3kT, e_min_eV
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size == 0:
        return {
            "n_basins": 0,
            "max_bw": float("nan"),
            "eff_n": float("nan"),
            "entropy": float("nan"),
            "n_within_kT": 0,
            "n_within_3kT": 0,
            "e_min_eV": float("nan"),
        }
    de = e - e.min()
    w = np.exp(-de / _KT_EV_298K)
    w = w / w.sum()
    return {
        "n_basins": int(e.size),
        "max_bw": float(w.max()),
        "eff_n": float(1.0 / (w**2).sum()),
        "entropy": float(-(w * np.log(w.clip(1e-45))).sum()),
        "n_within_kT": int((de <= _KT_EV_298K).sum()),
        "n_within_3kT": int((de <= 3 * _KT_EV_298K).sum()),
        "e_min_eV": float(e.min()),
    }


def run_one(
    peptide: dict,
    n_seeds: int,
    minimize: bool,
    params_mode: str,
    params,
    hardware_opts,
    calc,
    embed_chunk_size: int,
    score_chunk_size: int,
    e_window_kt: float,
    rmsd_threshold: float,
    dihedral_jitter_deg: float,
) -> dict:
    """
    Run get_mol_PE_exhaustive for one (peptide, n_seeds, minimize, params_mode,
    dihedral_jitter_deg) point and compute saturation metrics.

    Params:
        peptide: dict : selected-peptide row (peptide_id, source, smiles, n_heavy, ground_truth_*)
        n_seeds: int : ETKDG seed count for this run
        minimize: bool : whether to MMFF94-minimize each conformer before scoring
        params_mode: str : ETKDG variant label (recorded into the output row)
        params : ETKDG params instance constructed from params_mode
        hardware_opts : nvmolkit hardware options
        calc : MACE calculator
        embed_chunk_size: int : per-call embed cap
        score_chunk_size: int : per-batch MACE forward pass cap
        e_window_kt: float : energy filter window in units of kT_298K
        rmsd_threshold: float : geometric dedup exclusion radius
        dihedral_jitter_deg: float : maximum random rotation in degrees applied
            to each rotatable-bond dihedral after embedding (0 disables jitter)
    Returns:
        dict with all OUTPUT_COLUMNS populated for this run
    """
    t_embed_start = time.perf_counter()
    _, _, energies_eV = get_mol_PE_exhaustive(
        peptide["smiles"],
        params,
        hardware_opts,
        calc,
        n_seeds=n_seeds,
        embed_chunk_size=embed_chunk_size,
        score_chunk_size=score_chunk_size,
        e_window_kT=e_window_kt,
        rmsd_threshold=rmsd_threshold,
        minimize=minimize,
        dihedral_jitter_deg=dihedral_jitter_deg,
    )
    t_total = time.perf_counter() - t_embed_start

    metrics = _bw_metrics(energies_eV)
    # The exhaustive function does not return per-stage timings; record the
    # total elapsed and leave per-stage NaN. Phase 2 follow-up may extend the
    # function's return contract if profiling becomes important.
    return {
        "peptide_id": peptide["peptide_id"],
        "source": peptide["source"],
        "n_heavy": peptide["n_heavy"],
        "smiles": peptide["smiles"],
        "params_mode": params_mode,
        "n_seeds": n_seeds,
        "minimize": minimize,
        "dihedral_jitter_deg": dihedral_jitter_deg,
        "n_pool": n_seeds,
        **metrics,
        "time_embed_s": float("nan"),
        "time_score_s": float("nan"),
        "time_filter_dedup_s": float("nan"),
        "time_total_s": t_total,
        "ground_truth_max_bw": peptide["ground_truth_max_bw"],
        "ground_truth_n_confs": peptide["ground_truth_n_confs"],
    }


@click.command()
@click.option(
    "--cremp_csv",
    required=True,
    type=Path,
    help="CREMP validation subset CSV (sequence,smiles,...,poplowestpct,uniqueconfs)",
)
@click.option(
    "--pampa_csv",
    required=True,
    type=Path,
    help="CycPeptMPDB-deduped PAMPA CSV with a SMILES column",
)
@click.option(
    "--out_csv",
    required=True,
    type=Path,
    help="Output saturation results CSV (one row per (peptide, n_seeds, minimize))",
)
@click.option(
    "--n_seeds_grid",
    default="100,500,1000,5000,10000,50000",
    help="Comma-separated ETKDG seed counts to sweep",
)
@click.option("--embed_chunk_size", default=1000, type=int)
@click.option("--score_chunk_size", default=500, type=int)
@click.option("--e_window_kT", default=5.0, type=float)
@click.option("--rmsd_threshold", default=0.1, type=float)
@click.option(
    "--params_mode",
    type=click.Choice(["etkdgv3_macrocycle", "etkdg_original"]),
    default="etkdgv3_macrocycle",
    help="ETKDG variant to embed with. 'etkdgv3_macrocycle' is the "
    "production default; 'etkdg_original' drops the macrocycle "
    "torsion knowledge to test whether v3's bias suppresses "
    "low-energy basin diversity.",
)
@click.option(
    "--dihedral_jitter_deg",
    default=0.0,
    type=float,
    help="If >0, after embedding apply a uniform random rotation in "
    "[-jitter, +jitter] degrees to each rotatable-bond dihedral "
    "on every conformer. Used to push side-chain rotamer "
    "exploration past ETKDG's torsion-knowledge bias.",
)
@click.option(
    "--minimize_at_largest",
    is_flag=True,
    help="Also run the largest n_seeds value with minimize=True per peptide",
)
@click.option(
    "--smiles_col", default="SMILES", help="SMILES column name in the PAMPA CSV"
)
def main(
    cremp_csv: Path,
    pampa_csv: Path,
    out_csv: Path,
    n_seeds_grid: str,
    embed_chunk_size: int,
    score_chunk_size: int,
    e_window_kt: float,
    rmsd_threshold: float,
    params_mode: str,
    dihedral_jitter_deg: float,
    minimize_at_largest: bool,
    smiles_col: str,
) -> None:
    """
    Run the saturation sweep on the chosen representative peptides.

    Params:
        cremp_csv: Path : CREMP validation subset CSV
        pampa_csv: Path : CycPeptMPDB-deduped PAMPA CSV
        out_csv: Path : output CSV path (resume-aware)
        n_seeds_grid: str : comma-separated seed counts
        embed_chunk_size: int : per-call embed cap
        score_chunk_size: int : per-batch MACE forward pass cap
        e_window_kt: float : energy filter window in units of kT_298K (Click
            lowercases option names, so the CLI flag is --e_window_kT but the
            Python kwarg is e_window_kt)
        rmsd_threshold: float : dedup exclusion radius
        params_mode: str : ETKDG variant ('etkdgv3_macrocycle' or 'etkdg_original')
        dihedral_jitter_deg: float : maximum random rotation per rotatable bond
            (0 disables; same value applied to every n_seeds in this invocation)
        minimize_at_largest: bool : run minimize=True at largest n_seeds per peptide
        smiles_col: str : SMILES column name in PAMPA CSV
    Returns:
        None
    """
    seeds = sorted(int(x.strip()) for x in n_seeds_grid.split(",") if x.strip())
    largest = max(seeds)

    peptides = select_cremp_peptides(cremp_csv, n=2) + select_pampa_peptides(
        pampa_csv, smiles_col=smiles_col
    )
    logger.info("Selected %d peptides:", len(peptides))
    for p in peptides:
        logger.info(
            "  %-22s  source=%s  n_heavy=%d  ground_truth_max_bw=%s",
            p["peptide_id"],
            p["source"],
            p["n_heavy"],
            f"{p['ground_truth_max_bw']:.3f}"
            if not np.isnan(p["ground_truth_max_bw"])
            else "n/a",
        )
    logger.info(
        "params_mode=%s  dihedral_jitter_deg=%g", params_mode, dihedral_jitter_deg
    )

    done = _read_done_set(out_csv)
    if done:
        logger.info(
            "Resuming: %d (peptide, params_mode, n_seeds, minimize, dihedral_jitter_deg) tuples already done",
            len(done),
        )

    # GPU-bound shared resources: build once, reuse across runs.
    params = _build_params(params_mode)
    hw = get_hardware_opts()
    calc = get_mace_calc()

    runs: list[tuple[dict, int, bool]] = []
    for peptide in peptides:
        for n_seeds in seeds:
            runs.append((peptide, n_seeds, False))
        if minimize_at_largest:
            runs.append((peptide, largest, True))

    for peptide, n_seeds, minimize in runs:
        key = (
            peptide["peptide_id"],
            params_mode,
            n_seeds,
            minimize,
            dihedral_jitter_deg,
        )
        if key in done:
            logger.info(
                "skip %s mode=%s n=%d minimize=%s jitter=%g (already done)",
                peptide["peptide_id"],
                params_mode,
                n_seeds,
                minimize,
                dihedral_jitter_deg,
            )
            continue

        logger.info(
            "run  %s mode=%s n=%d minimize=%s jitter=%g",
            peptide["peptide_id"],
            params_mode,
            n_seeds,
            minimize,
            dihedral_jitter_deg,
        )
        try:
            row = run_one(
                peptide,
                n_seeds,
                minimize,
                params_mode,
                params,
                hw,
                calc,
                embed_chunk_size=embed_chunk_size,
                score_chunk_size=score_chunk_size,
                e_window_kt=e_window_kt,
                rmsd_threshold=rmsd_threshold,
                dihedral_jitter_deg=dihedral_jitter_deg,
            )
        except Exception as exc:
            logger.exception(
                "Failed %s mode=%s n=%d minimize=%s jitter=%g: %s",
                peptide["peptide_id"],
                params_mode,
                n_seeds,
                minimize,
                dihedral_jitter_deg,
                exc,
            )
            continue

        _append_row(out_csv, row)
        logger.info(
            "    -> n_basins=%d  max_bw=%.3f  n_within_3kT=%d  total=%.1fs",
            row["n_basins"],
            row["max_bw"],
            row["n_within_3kT"],
            row["time_total_s"],
        )
        # Free GPU memory between runs so the next embed has headroom.
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
