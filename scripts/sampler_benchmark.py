"""
Head-to-head benchmark of conformer samplers against the same MACE-OFF
scoring path on a fixed peptide library.

Companion to scripts/saturation_etkdg.py. The saturation script sweeps
n_seeds for a single sampler (get_mol_PE_exhaustive) to find the
diminishing-returns elbow; this script sweeps *samplers* at a matched
compute budget to compare CREST-quality basin coverage achieved per
GPU-hour.

Background — see GitHub issue #10. get_mol_PE_exhaustive matches CREST
on small/medium peptides (cremp_typical, pampa_small/medium) but
collapses to near-one-hot distributions on pampa_large in 3 of 4 runs,
suggesting a randomization-+-MMFF ceiling that no amount of n_seeds
will break through. The candidate samplers differ structurally from
exhaustive ETKDG (constrained DG, MD, basin-memory MC, metadynamics)
and the question is which of them clears that ceiling without paying
CREST's full per-conformer cost.

Sampler dispatch table — currently:
    exhaustive_etkdg  baseline (get_mol_PE_exhaustive, saturation-validated defaults)
    pool_b            backbone-dihedral-constrained DG (get_mol_PE_pool_b),
                      strategy='inverse', n_attempts=1, otherwise matched
                      defaults to exhaustive_etkdg
    mcmm              Multiple Minimum Monte Carlo with replica exchange
                      (get_mol_PE_mcmm). 8 temps × 8 walkers (300 K → 600 K)
                      with n_steps derived from n_seeds so total MMFF work
                      matches exhaustive_etkdg's at the same --n_seeds.
Future entries (CREST-fast, REMD) plug in as new functions plus a single
new dispatch-table key; the benchmark protocol stays unchanged.

Per (peptide, sampler) row, the script reports the same Boltzmann-weight
metrics as saturation_etkdg.py (n_basins, max_bw, eff_n, n_within_3kT,
e_min_eV) plus the wall-clock total. CREMP peptides carry the CREST
ground-truth `poplowestpct` and `uniqueconfs` for direct comparison;
PAMPA peptides have no ground truth (NaN columns).

Usage (in the confsweeper pixi mace environment):

    pixi run -e mace python scripts/sampler_benchmark.py \\
        --cremp_csv data/processed/cremp/validation_subset.csv \\
        --pampa_csv /home/sabari/peptide_electrostatics/data/fine_tune/CycPeptMPDB_PAMPA_deduped.csv \\
        --out_csv  results/sampler_benchmark.csv \\
        --samplers exhaustive_etkdg,pool_b,mcmm \\
        --n_seeds  10000

The output CSV is resume-aware on (peptide_id, sampler, n_seeds): re-running
after an interruption skips already-completed rows. Failed runs are
logged and skipped; the runner moves on to the next (peptide, sampler).
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

# Reuse the saturation script's peptide selection and metrics primitives so
# both benchmarks operate on the same peptide library and produce
# directly-comparable rows.
from saturation_etkdg import (  # noqa: E402
    _bw_metrics,
    select_cremp_peptides,
    select_pampa_peptides,
)

from confsweeper import (  # noqa: E402
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE_exhaustive,
    get_mol_PE_mcmm,
    get_mol_PE_pool_b,
)
from torsional_sampling import load_ramachandran_grids  # noqa: E402

# fmt: on


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("sampler_benchmark")


OUTPUT_COLUMNS = [
    "peptide_id",
    "source",
    "n_heavy",
    "smiles",
    "sampler",
    "n_seeds",
    "n_basins",
    "max_bw",
    "eff_n",
    "entropy",
    "n_within_kT",
    "n_within_3kT",
    "e_min_eV",
    "time_total_s",
    "ground_truth_max_bw",
    "ground_truth_n_confs",
]


def _run_exhaustive_etkdg(
    peptide: dict, n_seeds: int, hardware_opts, calc, grids: dict | None
) -> list[float]:
    """
    Adapter: run get_mol_PE_exhaustive at saturation-validated defaults.

    The `grids` argument is unused (the exhaustive sampler does not need the
    Ramachandran prior); accepting it keeps every adapter's signature
    identical so the dispatch loop stays uniform.

    Params:
        peptide: dict : peptide row from select_*_peptides
        n_seeds: int : ETKDG seed budget
        hardware_opts : nvmolkit hardware options
        calc : MACE calculator
        grids: dict | None : ignored
    Returns:
        list[float] : MACE energies in eV for the basin-representative conformers
    """
    del grids  # unused by this adapter
    params = get_embed_params_macrocycle()
    _, _, energies_eV = get_mol_PE_exhaustive(
        peptide["smiles"],
        params,
        hardware_opts,
        calc,
        n_seeds=n_seeds,
    )
    return energies_eV


def _run_pool_b(
    peptide: dict, n_seeds: int, hardware_opts, calc, grids: dict | None
) -> list[float]:
    """
    Adapter: run get_mol_PE_pool_b with strategy='inverse' and n_attempts=1.

    n_seeds is forwarded as n_samples, the (phi, psi) draw budget. With
    n_attempts=1 the raw conformer count is bounded above by n_samples
    (ring-closure failures bring it down); this is the matched-budget
    semantic for direct comparison against exhaustive_etkdg.

    Params:
        peptide: dict : peptide row from select_*_peptides
        n_seeds: int : (phi, psi) draw budget (forwarded as n_samples)
        hardware_opts : nvmolkit hardware options
        calc : MACE calculator
        grids: dict | None : CREMP Ramachandran grids; required by this adapter
    Returns:
        list[float] : MACE energies in eV for the basin-representative conformers
    """
    if grids is None:
        raise ValueError(
            "pool_b sampler requires Ramachandran grids; pass --ramachandran_grids"
        )
    _, _, energies_eV = get_mol_PE_pool_b(
        peptide["smiles"],
        grids,
        hardware_opts,
        calc,
        n_samples=n_seeds,
    )
    return energies_eV


def _run_mcmm(
    peptide: dict, n_seeds: int, hardware_opts, calc, grids: dict | None
) -> list[float]:
    """
    Adapter: run get_mol_PE_mcmm with the issue-#11 default temperature
    ladder (8 temps × 8 walkers = 64 walkers, 300 K → 600 K geometric)
    and `n_steps` derived from `n_seeds` so the total MMFF minimisation
    budget matches `exhaustive_etkdg`'s for the same `--n_seeds`.

    Mapping: MCMM costs roughly one MMFF call per walker per step. At
    the default 64 walkers, `n_steps = n_seeds // 64` keeps total MMFF
    work proportional to `n_seeds`. For the saturation-validated
    `n_seeds=10000` this gives 156 steps per walker (12 480 total
    minimisations, within ~25 % of exhaustive ETKDG's headline budget).

    The `grids` argument is unused — MCMM does not consume the
    Ramachandran prior.

    Params:
        peptide: dict : peptide row from select_*_peptides
        n_seeds: int : matched-budget MMFF call budget; converted to
            n_steps = max(1, n_seeds // 64) per walker
        hardware_opts : nvmolkit hardware options
        calc : MACE calculator
        grids: dict | None : ignored
    Returns:
        list[float] : MACE energies in eV for the basin-representative conformers
    """
    del grids  # unused by this adapter
    params = get_embed_params_macrocycle()
    n_walkers_per_temp = 8
    n_temperatures = 8
    n_walkers = n_walkers_per_temp * n_temperatures
    n_steps = max(1, n_seeds // n_walkers)
    _, _, energies_eV = get_mol_PE_mcmm(
        peptide["smiles"],
        params,
        hardware_opts,
        calc,
        n_walkers_per_temp=n_walkers_per_temp,
        n_temperatures=n_temperatures,
        n_steps=n_steps,
    )
    return energies_eV


SAMPLERS: dict[str, callable] = {
    "exhaustive_etkdg": _run_exhaustive_etkdg,
    "pool_b": _run_pool_b,
    "mcmm": _run_mcmm,
}


def _read_done_set(out_csv: Path) -> set[tuple]:
    """
    Return the set of (peptide_id, sampler, n_seeds) tuples already written.

    Params:
        out_csv: Path : output CSV path
    Returns:
        set[tuple[str, str, int]] : completed (peptide_id, sampler, n_seeds)
    """
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    return set(
        zip(
            df["peptide_id"].astype(str),
            df["sampler"].astype(str),
            df["n_seeds"].astype(int),
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


def run_one(
    peptide: dict,
    sampler: str,
    n_seeds: int,
    hardware_opts,
    calc,
    grids: dict | None,
) -> dict:
    """
    Run one sampler on one peptide, compute basin metrics, return a row dict.

    Params:
        peptide: dict : peptide row from select_*_peptides
        sampler: str : key into SAMPLERS dispatch table
        n_seeds: int : sampler budget (interpretation is sampler-specific)
        hardware_opts : nvmolkit hardware options
        calc : MACE calculator
        grids: dict | None : CREMP Ramachandran grids (required by pool_b)
    Returns:
        dict with all OUTPUT_COLUMNS populated for this run
    """
    runner = SAMPLERS[sampler]
    t0 = time.perf_counter()
    energies_eV = runner(peptide, n_seeds, hardware_opts, calc, grids)
    t_total = time.perf_counter() - t0

    metrics = _bw_metrics(energies_eV)
    return {
        "peptide_id": peptide["peptide_id"],
        "source": peptide["source"],
        "n_heavy": peptide["n_heavy"],
        "smiles": peptide["smiles"],
        "sampler": sampler,
        "n_seeds": n_seeds,
        **metrics,
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
    help="Output benchmark CSV (one row per (peptide, sampler, n_seeds))",
)
@click.option(
    "--samplers",
    default=",".join(SAMPLERS.keys()),
    show_default=True,
    help="Comma-separated sampler names from the dispatch table.",
)
@click.option(
    "--n_seeds",
    default=10000,
    type=int,
    show_default=True,
    help="Sampler budget (n_seeds for ETKDG, n_samples for Pool B).",
)
@click.option(
    "--ramachandran_grids",
    default="data/processed/cremp/ramachandran_grids.npz",
    show_default=True,
    type=Path,
    help="CREMP Ramachandran grids .npz (required when 'pool_b' is in --samplers).",
)
@click.option(
    "--smiles_col", default="SMILES", help="SMILES column name in the PAMPA CSV"
)
def main(
    cremp_csv: Path,
    pampa_csv: Path,
    out_csv: Path,
    samplers: str,
    n_seeds: int,
    ramachandran_grids: Path,
    smiles_col: str,
) -> None:
    """
    Benchmark each requested sampler against the same five peptides at a
    matched compute budget, writing one CSV row per (peptide, sampler) cell.

    Params:
        cremp_csv: Path : CREMP validation subset CSV
        pampa_csv: Path : CycPeptMPDB-deduped PAMPA CSV
        out_csv: Path : output CSV path (resume-aware)
        samplers: str : comma-separated sampler names
        n_seeds: int : sampler budget passed to every adapter
        ramachandran_grids: Path : CREMP Ramachandran grids file
        smiles_col: str : SMILES column name in PAMPA CSV
    Returns:
        None
    """
    sampler_list = [s.strip() for s in samplers.split(",") if s.strip()]
    unknown = [s for s in sampler_list if s not in SAMPLERS]
    if unknown:
        raise click.BadParameter(
            f"unknown sampler(s) {unknown!r}; available: {list(SAMPLERS)}"
        )

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
    logger.info("samplers=%s  n_seeds=%d", sampler_list, n_seeds)

    done = _read_done_set(out_csv)
    if done:
        logger.info(
            "Resuming: %d (peptide, sampler, n_seeds) tuples already done", len(done)
        )

    # Shared GPU resources: build once, reuse across runs.
    hw = get_hardware_opts()
    calc = get_mace_calc()
    grids = (
        load_ramachandran_grids(ramachandran_grids)
        if "pool_b" in sampler_list
        else None
    )

    for peptide in peptides:
        for sampler in sampler_list:
            key = (peptide["peptide_id"], sampler, n_seeds)
            if key in done:
                logger.info(
                    "skip %s sampler=%s n=%d (already done)",
                    peptide["peptide_id"],
                    sampler,
                    n_seeds,
                )
                continue

            logger.info(
                "run  %s sampler=%s n=%d",
                peptide["peptide_id"],
                sampler,
                n_seeds,
            )
            try:
                row = run_one(peptide, sampler, n_seeds, hw, calc, grids)
            except Exception as exc:
                logger.exception(
                    "Failed %s sampler=%s n=%d: %s",
                    peptide["peptide_id"],
                    sampler,
                    n_seeds,
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
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
