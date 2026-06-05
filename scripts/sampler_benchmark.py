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
from rdkit import Chem

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
    "dedup_mode",
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


def _maybe_dump_sdf(
    mol: Chem.Mol,
    conf_ids: list,
    energies_eV: list,
    peptide_id: str,
    sampler: str,
    dump_sdf_dir,
) -> None:
    """
    Write the basin centroids of a single (peptide, sampler) run to an SDF
    file, one conformer per centroid, with per-conformer `MACE_ENERGY`
    in eV. No-op if `dump_sdf_dir` is None.

    Output path is `<dump_sdf_dir>/<safe_peptide_id>_<sampler>.sdf` where
    `safe_peptide_id` strips ':' and '/' to make a filesystem-safe name.

    Used for diagnostic visualisation — load these in PyMOL or RDKit's
    conformer viewer to inspect ring topology, side-chain rotamers, and
    spread of the basin set. The deferred Step 18 (DBT analytical
    polynomial branches) is gated on observing whether MCMM runs ever
    produce ring-topology changes; SDF dumps are how we'll find out.

    Params:
        mol: Chem.Mol : the mol returned by `get_mol_PE_*`, with basin
            centroids attached as conformers.
        conf_ids: list[int] : conformer IDs to write, in the order
            produced by the sampler (ascending energy by convention).
        energies_eV: list[float] : per-conformer MACE energies, same
            order as conf_ids; written as the `MACE_ENERGY` SDF prop.
        peptide_id: str : peptide identifier, used in the filename.
        sampler: str : sampler name, used in the filename.
        dump_sdf_dir: Path | None : directory to write into. None
            disables the dump entirely.
    Returns:
        None
    """
    if dump_sdf_dir is None:
        return
    dump_sdf_dir = Path(dump_sdf_dir)
    dump_sdf_dir.mkdir(parents=True, exist_ok=True)
    safe_id = peptide_id.replace(":", "_").replace("/", "_")
    out_path = dump_sdf_dir / f"{safe_id}_{sampler}.sdf"
    writer = Chem.SDWriter(str(out_path))
    try:
        mol.SetProp("peptide_id", peptide_id)
        mol.SetProp("sampler", sampler)
        for cid, energy in zip(conf_ids, energies_eV):
            mol.SetDoubleProp("MACE_ENERGY", float(energy))
            writer.write(mol, confId=cid)
    finally:
        writer.close()
    logger.info(
        "  → dumped %d basin centroids to %s",
        len(conf_ids),
        out_path,
    )


def _run_exhaustive_etkdg(
    peptide: dict,
    n_seeds: int,
    hardware_opts,
    calc,
    grids: dict | None,
    dump_sdf_dir=None,
    dedup_mode: str = "kabsch",
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
        dump_sdf_dir: Path | None : if set, dump basin centroids
        dedup_mode: str : 'kabsch' (default) or 'crest' for the
            three-criteria AND-test (Step 17 of docs/mcmm_plan.md)
    Returns:
        list[float] : MACE energies in eV for the basin-representative conformers
    """
    del grids  # unused by this adapter
    params = get_embed_params_macrocycle()
    mol, conf_ids, energies_eV = get_mol_PE_exhaustive(
        peptide["smiles"],
        params,
        hardware_opts,
        calc,
        n_seeds=n_seeds,
        dedup_mode=dedup_mode,
    )
    _maybe_dump_sdf(
        mol,
        conf_ids,
        energies_eV,
        peptide["peptide_id"],
        "exhaustive_etkdg",
        dump_sdf_dir,
    )
    return energies_eV


def _run_pool_b(
    peptide: dict,
    n_seeds: int,
    hardware_opts,
    calc,
    grids: dict | None,
    dump_sdf_dir=None,
    dedup_mode: str = "kabsch",
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
        dump_sdf_dir: Path | None : if set, dump basin centroids
        dedup_mode: str : 'kabsch' (default) or 'crest'
    Returns:
        list[float] : MACE energies in eV for the basin-representative conformers
    """
    if grids is None:
        raise ValueError(
            "pool_b sampler requires Ramachandran grids; pass --ramachandran_grids"
        )
    mol, conf_ids, energies_eV = get_mol_PE_pool_b(
        peptide["smiles"],
        grids,
        hardware_opts,
        calc,
        n_samples=n_seeds,
        dedup_mode=dedup_mode,
    )
    _maybe_dump_sdf(
        mol,
        conf_ids,
        energies_eV,
        peptide["peptide_id"],
        "pool_b",
        dump_sdf_dir,
    )
    return energies_eV


def _run_mcmm(
    peptide: dict,
    n_seeds: int,
    hardware_opts,
    calc,
    grids: dict | None,
    dump_sdf_dir=None,
    dedup_mode: str = "kabsch",
    cartesian_weight: float = 0.0,
    e_window_kT: float = 5.0,
    saunders_exponent: float = 0.5,
) -> list[float]:
    """
    Adapter: run get_mol_PE_mcmm with the issue-#11 default temperature
    ladder (8 temps × 8 walkers = 64 walkers) and `n_steps` derived from
    `n_seeds` so the total MMFF minimisation budget matches
    `exhaustive_etkdg`'s for the same `--n_seeds`.

    Mapping: MCMM costs roughly one MMFF call per walker per step. At
    64 walkers, `n_steps = n_seeds // 64` keeps total MMFF work
    proportional to `n_seeds`. For the saturation-validated
    `n_seeds=10000` this gives 156 steps per walker (12 480 total
    minimisations, within ~25 % of exhaustive ETKDG's headline budget).

    **Tuning experiment (cremp_typical diagnosis, 2026-04-29).** The
    default `drive_sigma_rad=0.1` / `closure_tol=0.01` / `kt_high=2 ×
    kT_298K` produced 1 basin on cremp_typical (basin-collapse pattern:
    160 Metropolis-accepted moves all within `rmsd_threshold=0.1` of
    the seed; swap_accept_rate=0.96 confirming all replicas in the
    same basin). Hardcoding more aggressive values here:
      * `drive_sigma_rad=0.3`  : ~17° backbone moves vs. ~5.7° default;
        bigger perturbations push past the seed's MMFF gradient.
      * `closure_tol=0.05`     : pairs with the larger drive; stays
        within MMFF's basin-recovery range per the module docstring.
      * `kt_high=8 × kT_298K`  : ~2400 K hot end of the ladder, eight
        times the cold end. Bumped from 4× after the multi-seed run
        showed structurally-distinct basin pairs at 1.4 Å / 3 kT
        on cremp_typical without bridging. Tests whether wider
        replica spread improves discovery on truly-distinct basins.
      * `n_init_confs=8`       : multi-seed initialisation (lever C9 in
        docs/mcmm_plan.md). Embeds 8 distinct ETKDG seed conformers
        instead of 1; walkers distribute round-robin across them so
        each temperature stack gets exposure to every starting basin.
        Directly addresses the structural problem that all 64 walkers
        previously started in the same basin.
    Roll back to defaults once benchmark data confirms the diagnosis.

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
    # Hardcoded tuning experiment — see docstring above.
    from confsweeper import _KT_EV_298K

    mol, conf_ids, energies_eV = get_mol_PE_mcmm(
        peptide["smiles"],
        params,
        hardware_opts,
        calc,
        n_walkers_per_temp=n_walkers_per_temp,
        n_temperatures=n_temperatures,
        n_steps=n_steps,
        drive_sigma_rad=0.3,
        closure_tol=0.05,
        kt_high=8.0 * _KT_EV_298K,
        n_init_confs=8,
        dedup_mode=dedup_mode,
        cartesian_weight=cartesian_weight,
        e_window_kT=e_window_kT,
        saunders_exponent=saunders_exponent,
    )
    _maybe_dump_sdf(
        mol,
        conf_ids,
        energies_eV,
        peptide["peptide_id"],
        "mcmm",
        dump_sdf_dir,
    )
    return energies_eV


SAMPLERS: dict[str, callable] = {
    "exhaustive_etkdg": _run_exhaustive_etkdg,
    "pool_b": _run_pool_b,
    "mcmm": _run_mcmm,
}


def select_peptide_list_csv(peptide_list_csv: Path) -> list[dict]:
    """
    Build a peptide list from a CSV (e.g. the output of
    `sample_cremp_peptides.py`). Required columns: `sequence`, `smiles`.
    Optional columns are passed through as-is when present:
    `num_heavy_atoms`, `poplowestpct` (CREMP max-Boltzmann-weight
    equivalent), `uniqueconfs` (CREMP basin-count ground truth).

    Each row becomes one peptide dict matching the contract
    `select_cremp_peptides` / `select_pampa_peptides` produce.

    Params:
        peptide_list_csv: Path : input CSV
    Returns:
        list[dict] : peptide rows with peptide_id, source, smiles,
            n_heavy, ground_truth_max_bw, ground_truth_n_confs
    """
    df = pd.read_csv(peptide_list_csv)
    required = {"sequence", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise click.BadParameter(
            f"{peptide_list_csv} missing required columns {sorted(missing)}"
        )
    out: list[dict] = []
    for row in df.itertuples(index=False):
        out.append(
            {
                "peptide_id": f"cremp:{row.sequence}",
                "source": "cremp_sample",
                "smiles": row.smiles,
                "n_heavy": (
                    int(row.num_heavy_atoms)
                    if hasattr(row, "num_heavy_atoms")
                    and not np.isnan(row.num_heavy_atoms)
                    else 0
                ),
                "ground_truth_max_bw": (
                    float(row.poplowestpct) / 100.0
                    if hasattr(row, "poplowestpct") and not np.isnan(row.poplowestpct)
                    else float("nan")
                ),
                "ground_truth_n_confs": (
                    int(row.uniqueconfs)
                    if hasattr(row, "uniqueconfs") and not np.isnan(row.uniqueconfs)
                    else 0
                ),
            }
        )
    return out


def _read_done_set(out_csv: Path) -> set[tuple]:
    """
    Return the set of (peptide_id, sampler, n_seeds, dedup_mode) tuples
    already written. CSVs from before Step 17 lacked the `dedup_mode`
    column — those rows are treated as `'kabsch'` for resume purposes.

    Params:
        out_csv: Path : output CSV path
    Returns:
        set[tuple[str, str, int, str]] : completed
            (peptide_id, sampler, n_seeds, dedup_mode)
    """
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if "dedup_mode" in df.columns:
        modes = df["dedup_mode"].astype(str)
    else:
        modes = pd.Series(["kabsch"] * len(df))
    return set(
        zip(
            df["peptide_id"].astype(str),
            df["sampler"].astype(str),
            df["n_seeds"].astype(int),
            modes,
        )
    )


def _check_header_matches(out_csv: Path) -> None:
    """
    Verify the existing CSV's header matches `OUTPUT_COLUMNS`.

    Schema-evolution guard: if a column was added between runs (e.g.
    `dedup_mode` introduced in Step 17), appending new rows under the
    new schema to a file with the old header silently misaligns
    columns — the extra value lands in a neighbour's slot. This check
    catches that at the start of `main` so the user can migrate
    explicitly rather than discovering it in pandas later.

    Params:
        out_csv: Path : path to an existing CSV (caller checks existence)
    Returns:
        None
    Raises:
        click.ClickException: if the header does not match OUTPUT_COLUMNS
    """
    with out_csv.open() as f:
        header = next(csv.reader(f), None)
    if header is None:
        return
    if header != OUTPUT_COLUMNS:
        raise click.ClickException(
            f"Existing CSV {out_csv} has header columns that do not match the "
            f"current OUTPUT_COLUMNS schema. Either delete it, write to a new "
            f"path, or migrate it (insert new columns with appropriate "
            f"backfill values).\n"
            f"  current header: {header}\n"
            f"  expected:       {OUTPUT_COLUMNS}"
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
    dump_sdf_dir=None,
    dedup_mode: str = "kabsch",
    cartesian_weight: float = 0.0,
    e_window_kT: float = 5.0,
    saunders_exponent: float = 0.5,
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
        dump_sdf_dir: Path | None : if set, the adapter writes an SDF of
            the basin centroids to this directory; one file per
            (peptide, sampler) pair.
        dedup_mode: str : 'kabsch' (default) or 'crest'. Recorded in
            the output row so multi-mode runs can be joined cleanly.
        cartesian_weight: float : routing weight for the GOAT-style
            Cartesian-kick proposer in MCMM. 0 = pure DBT (legacy);
            0.5 = 50/50 mix per walker per step. Ignored by non-MCMM
            samplers. Step 12 of docs/mcmm_plan.md.
    Returns:
        dict with all OUTPUT_COLUMNS populated for this run
    """
    runner = SAMPLERS[sampler]
    t0 = time.perf_counter()
    runner_kwargs = {"dump_sdf_dir": dump_sdf_dir, "dedup_mode": dedup_mode}
    if sampler == "mcmm":
        runner_kwargs["cartesian_weight"] = cartesian_weight
        runner_kwargs["e_window_kT"] = e_window_kT
        runner_kwargs["saunders_exponent"] = saunders_exponent
    energies_eV = runner(
        peptide,
        n_seeds,
        hardware_opts,
        calc,
        grids,
        **runner_kwargs,
    )
    t_total = time.perf_counter() - t0

    metrics = _bw_metrics(energies_eV)
    return {
        "peptide_id": peptide["peptide_id"],
        "source": peptide["source"],
        "n_heavy": peptide["n_heavy"],
        "smiles": peptide["smiles"],
        "sampler": sampler,
        "dedup_mode": dedup_mode,
        "n_seeds": n_seeds,
        **metrics,
        "time_total_s": t_total,
        "ground_truth_max_bw": peptide["ground_truth_max_bw"],
        "ground_truth_n_confs": peptide["ground_truth_n_confs"],
    }


@click.command()
@click.option(
    "--cremp_csv",
    type=Path,
    default=None,
    help="CREMP validation subset CSV (sequence,smiles,...,poplowestpct,uniqueconfs). "
    "Required unless --peptide_list_csv is given.",
)
@click.option(
    "--pampa_csv",
    type=Path,
    default=None,
    help="CycPeptMPDB-deduped PAMPA CSV with a SMILES column. "
    "Required unless --peptide_list_csv is given.",
)
@click.option(
    "--peptide_list_csv",
    type=Path,
    default=None,
    help="Alternative peptide source: any CSV with `sequence` and `smiles` columns "
    "(e.g. the output of `sample_cremp_peptides.py`). When provided, "
    "--cremp_csv / --pampa_csv are ignored and the entire CSV becomes the "
    "peptide list. Optional `num_heavy_atoms`, `poplowestpct`, `uniqueconfs` "
    "columns are passed through to the output rows.",
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
@click.option(
    "--dump_sdf_dir",
    default=None,
    type=Path,
    help="If set, write basin centroids to <dir>/<peptide_id>_<sampler>.sdf "
    "for each (peptide, sampler) cell. MACE_ENERGY is set per conformer "
    "in eV. Diagnostic; PyMOL / RDKit conformer viewer can load these to "
    "inspect basin geometry, ring topology, and side-chain rotamers.",
)
@click.option(
    "--dedup_mode",
    type=click.Choice(["kabsch", "crest", "both"]),
    default="kabsch",
    show_default=True,
    help="Basin-dedup criterion. 'kabsch' (default, chemical-basin scale), "
    "'crest' (CREMP-comparable three-criteria AND-test), or 'both' "
    "(run each peptide × sampler in both modes and emit two rows for "
    "side-by-side reporting). See Step 17 of docs/mcmm_plan.md.",
)
@click.option(
    "--cartesian_weight",
    type=float,
    default=0.0,
    show_default=True,
    help="MCMM proposer mix: routing weight for the GOAT-style "
    "Cartesian-kick proposer alongside DBT. 0 = pure DBT (legacy); "
    "0.5 = 50/50 mix per walker per step. Ignored by non-MCMM "
    "samplers. Step 12 of docs/mcmm_plan.md.",
)
@click.option(
    "--e_window_kT",
    "e_window_kT",
    type=float,
    default=5.0,
    show_default=True,
    help="Post-MCMM energy filter window in units of kT_298K. Bump "
    "to 10 when Cartesian kicks find minima 0.4+ eV deeper than "
    "DBT alone — the relative window then keeps a stratum of "
    "basins that 5 kT excludes.",
)
@click.option(
    "--saunders_exponent",
    type=float,
    default=0.5,
    show_default=True,
    help="Exponent in the Saunders 1/usage^p bias. 0.5 = original "
    "Saunders 1990; 1.0 = stronger decay, lever C15 of "
    "docs/mcmm_plan.md (pushes walkers out of deep wells faster, "
    "useful when Cartesian-kick discovers traps that 1/√usage "
    "doesn't escape).",
)
def main(
    cremp_csv: Path | None,
    pampa_csv: Path | None,
    peptide_list_csv: Path | None,
    out_csv: Path,
    samplers: str,
    n_seeds: int,
    ramachandran_grids: Path,
    smiles_col: str,
    dump_sdf_dir: Path | None,
    dedup_mode: str,
    cartesian_weight: float,
    e_window_kT: float,
    saunders_exponent: float,
) -> None:
    """
    Benchmark each requested sampler against the same five peptides at a
    matched compute budget, writing one CSV row per (peptide, sampler,
    dedup_mode) cell.

    Params:
        cremp_csv: Path : CREMP validation subset CSV
        pampa_csv: Path : CycPeptMPDB-deduped PAMPA CSV
        out_csv: Path : output CSV path (resume-aware)
        samplers: str : comma-separated sampler names
        n_seeds: int : sampler budget passed to every adapter
        ramachandran_grids: Path : CREMP Ramachandran grids file
        smiles_col: str : SMILES column name in PAMPA CSV
        dedup_mode: str : 'kabsch', 'crest', or 'both'
    Returns:
        None
    """
    sampler_list = [s.strip() for s in samplers.split(",") if s.strip()]
    unknown = [s for s in sampler_list if s not in SAMPLERS]
    if unknown:
        raise click.BadParameter(
            f"unknown sampler(s) {unknown!r}; available: {list(SAMPLERS)}"
        )
    mode_list = ["kabsch", "crest"] if dedup_mode == "both" else [dedup_mode]

    # Two peptide-source modes:
    #   - default (5-peptide test set): --cremp_csv + --pampa_csv with the
    #     hardcoded select_cremp_peptides / select_pampa_peptides selectors.
    #   - at-scale (any N): --peptide_list_csv pointing at a CSV with
    #     `sequence` and `smiles` columns (e.g. the output of
    #     sample_cremp_peptides.py). Bypasses the selectors entirely.
    if peptide_list_csv is not None:
        peptides = select_peptide_list_csv(peptide_list_csv)
    else:
        if cremp_csv is None or pampa_csv is None:
            raise click.BadParameter(
                "must provide either --peptide_list_csv, or both "
                "--cremp_csv and --pampa_csv"
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
    logger.info(
        "samplers=%s  n_seeds=%d  dedup_modes=%s  cartesian_weight=%.2f  "
        "e_window_kT=%.2f  saunders_exponent=%.2f",
        sampler_list,
        n_seeds,
        mode_list,
        cartesian_weight,
        e_window_kT,
        saunders_exponent,
    )

    if out_csv.exists():
        _check_header_matches(out_csv)
    done = _read_done_set(out_csv)
    if done:
        logger.info(
            "Resuming: %d (peptide, sampler, n_seeds, dedup_mode) tuples already done",
            len(done),
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
            for mode in mode_list:
                key = (peptide["peptide_id"], sampler, n_seeds, mode)
                if key in done:
                    logger.info(
                        "skip %s sampler=%s n=%d dedup=%s (already done)",
                        peptide["peptide_id"],
                        sampler,
                        n_seeds,
                        mode,
                    )
                    continue

                logger.info(
                    "run  %s sampler=%s n=%d dedup=%s",
                    peptide["peptide_id"],
                    sampler,
                    n_seeds,
                    mode,
                )
                try:
                    row = run_one(
                        peptide,
                        sampler,
                        n_seeds,
                        hw,
                        calc,
                        grids,
                        dump_sdf_dir=dump_sdf_dir,
                        dedup_mode=mode,
                        cartesian_weight=cartesian_weight,
                        e_window_kT=e_window_kT,
                        saunders_exponent=saunders_exponent,
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed %s sampler=%s n=%d dedup=%s: %s",
                        peptide["peptide_id"],
                        sampler,
                        n_seeds,
                        mode,
                        exc,
                    )
                    continue

                _append_row(out_csv, row)
                logger.info(
                    "    -> dedup=%s  n_basins=%d  max_bw=%.3f  n_within_3kT=%d  total=%.1fs",
                    mode,
                    row["n_basins"],
                    row["max_bw"],
                    row["n_within_3kT"],
                    row["time_total_s"],
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
