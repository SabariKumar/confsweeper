"""
Checkpointed benchmark CLI: measures confsweeper conformer coverage against CREMP.

For each molecule in the validation subset, generates conformers with confsweeper
and measures what fraction of CREMP reference conformers are covered (symmetric
RMSD <= rmsd_cutoff).

Supports sweeping n_confs: pass multiple values via --n_confs to process each
molecule at all requested conformer counts in a single pass (one pickle load per
molecule regardless of how many n_confs values are swept).

Checkpoint behavior: if output_csv already exists, any (sequence, n_confs) pair
already present is skipped. Interrupted runs resume automatically.

Usage:
    python cremp_coverage.py \\
        --subset_csv   data/processed/cremp/validation_subset.csv \\
        --pickle_dir   data/raw/cremp/pickle \\
        --output_csv   data/processed/cremp/coverage.csv \\
        --n_confs      500,1000,2000 \\
        --butina_thresh 0.1 \\
        --rmsd_cutoff  1.0
"""

import csv
import logging
import os
import sys
import warnings
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

from confsweeper import (
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE,
)
from validation.cremp import calc_coverage, iter_validation_mols

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "sequence",
    "smiles",
    "topology",
    "atom_bin",
    "num_monomers",
    "num_heavy_atoms",
    "n_confs",
    "n_generated_confs",
    "n_ref_confs",
    "coverage",
    "mean_min_rmsd",
    "median_min_rmsd",
]

ERROR_COLUMNS = ["sequence", "n_confs", "error"]


def _load_checkpoint(output_csv: str) -> set[tuple[str, int]]:
    """Returns set of (sequence, n_confs) pairs already in output_csv."""
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["sequence"], int(row["n_confs"])))
    return done


def _append_row(path: str, row: dict, columns: list[str]) -> None:
    """Appends a single row to a CSV, writing the header if the file is new."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@click.command()
@click.option(
    "--subset_csv",
    default="data/processed/cremp/validation_subset.csv",
    show_default=True,
    help="Path to validation subset CSV (from make_validation_sets_cremp.py)",
)
@click.option(
    "--pickle_dir",
    default="data/raw/cremp/pickle",
    show_default=True,
    help="Path to CREMP pickle/ directory",
)
@click.option(
    "--output_csv",
    default="data/processed/cremp/coverage.csv",
    show_default=True,
    help="Path to write coverage results (appended to if it exists)",
)
@click.option(
    "--errors_csv",
    default="data/processed/cremp/coverage_errors.csv",
    show_default=True,
    help="Path to write per-molecule errors",
)
@click.option(
    "--n_confs",
    default="1000",
    show_default=True,
    help="Comma-separated conformer counts to sweep, e.g. '500,1000,2000'",
)
@click.option(
    "--butina_thresh",
    default=0.1,
    show_default=True,
    type=float,
    help="Butina clustering distance threshold",
)
@click.option(
    "--rmsd_cutoff",
    default=1.0,
    show_default=True,
    type=float,
    help="RMSD threshold (Å) for counting a reference conformer as covered",
)
@click.option(
    "--filter_factor",
    default=2.0,
    show_default=True,
    type=float,
    help="Pre-filter multiplier: tensor RMSD cutoff = rmsd_cutoff * filter_factor",
)
def run_coverage_benchmark(
    subset_csv,
    pickle_dir,
    output_csv,
    errors_csv,
    n_confs,
    butina_thresh,
    rmsd_cutoff,
    filter_factor,
):
    """Benchmark confsweeper conformer coverage against CREMP reference conformers."""
    n_confs_values = [int(x.strip()) for x in n_confs.split(",")]
    logger.info("n_confs sweep: %s", n_confs_values)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(errors_csv) or ".", exist_ok=True)

    done = _load_checkpoint(output_csv)
    if done:
        logger.info(
            "Resuming: %d (sequence, n_confs) pairs already processed", len(done)
        )

    params = get_embed_params_macrocycle()
    hardware_opts = get_hardware_opts(
        preprocessingThreads=8, batch_size=1000, batchesPerGpu=2
    )
    mace_calc = get_mace_calc()

    mol_iter = iter_validation_mols(subset_csv, pickle_dir)

    for sequence, smiles, ref_mol, meta in tqdm(mol_iter, desc="molecules"):
        n_ref_confs = ref_mol.GetNumConformers()

        for n in n_confs_values:
            if (sequence, n) in done:
                continue

            try:
                gen_mol, gen_conf_ids, _pe = get_mol_PE(
                    smi=smiles,
                    params=params,
                    hardware_opts=hardware_opts,
                    mace_calc=mace_calc,
                    n_confs=n,
                    cutoff_dist=butina_thresh,
                    gpu_clustering=True,
                )
                gen_conf_ids = list(gen_conf_ids)

                if len(gen_conf_ids) == 0:
                    raise ValueError("EmbedMolecules produced 0 conformers")

                coverage, min_rmsds = calc_coverage(
                    ref_mol=ref_mol,
                    gen_mol=gen_mol,
                    gen_conf_ids=gen_conf_ids,
                    rmsd_cutoff=rmsd_cutoff,
                    filter_factor=filter_factor,
                )

                finite_rmsds = [r for r in min_rmsds if r != float("inf")]
                mean_min_rmsd = (
                    float(np.mean(finite_rmsds)) if finite_rmsds else float("nan")
                )
                median_min_rmsd = (
                    float(np.median(finite_rmsds)) if finite_rmsds else float("nan")
                )

                row = {
                    "sequence": sequence,
                    "smiles": smiles,
                    "topology": meta["topology"],
                    "atom_bin": meta["atom_bin"],
                    "num_monomers": meta["num_monomers"],
                    "num_heavy_atoms": meta["num_heavy_atoms"],
                    "n_confs": n,
                    "n_generated_confs": len(gen_conf_ids),
                    "n_ref_confs": n_ref_confs,
                    "coverage": round(coverage, 6),
                    "mean_min_rmsd": round(mean_min_rmsd, 4),
                    "median_min_rmsd": round(median_min_rmsd, 4),
                }
                _append_row(output_csv, row, OUTPUT_COLUMNS)
                done.add((sequence, n))

            except Exception as e:
                logger.warning("Error processing %s at n_confs=%d: %s", sequence, n, e)
                _append_row(
                    errors_csv,
                    {"sequence": sequence, "n_confs": n, "error": str(e)},
                    ERROR_COLUMNS,
                )


if __name__ == "__main__":
    run_coverage_benchmark()
