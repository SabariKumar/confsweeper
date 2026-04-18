"""
Checkpointed benchmark CLI: measures confsweeper conformer coverage against CREMP.

For each molecule in the validation subset, generates conformers with confsweeper
and measures what fraction of CREMP reference conformers are covered (symmetric
RMSD <= rmsd_cutoff).

Supports sweeping n_confs: pass multiple values via --n_confs to process each
molecule at all requested conformer counts in a single pass (one pickle load per
molecule regardless of how many n_confs values are swept).

Sampling modes
--------------
etkdg (default)
    Standard nvmolkit ETKDGv3 → GPU Butina → MACE scoring.

etkdg+torsional
    Pool A: same as etkdg.
    Pool B: backbone dihedral-constrained DG (CPU ETKDGv3), (phi, psi) targets
            sampled from the CREMP Ramachandran prior.
    Merge:  Pool B conformers are MACE-scored and filtered to energies within
            mean(A) + 2*std(A).  A single Butina pass de-duplicates the combined
            A+B pool before coverage is measured.

Checkpoint behavior: if output_csv already exists, any (sequence, n_confs,
sampling_mode) triple already present is skipped. Interrupted runs resume
automatically.

Usage:
    # ETKDG only
    python cremp_coverage.py \\
        --subset_csv   data/processed/cremp/validation_subset.csv \\
        --pickle_dir   data/raw/cremp/pickle \\
        --output_csv   data/processed/cremp/coverage.csv \\
        --n_confs      500,1000,2000

    # ETKDG + torsional sampling
    python cremp_coverage.py \\
        --subset_csv   data/processed/cremp/validation_subset.csv \\
        --pickle_dir   data/raw/cremp/pickle \\
        --output_csv   data/processed/cremp/coverage.csv \\
        --n_confs      1000 \\
        --torsional_sampling \\
        --torsional_n_samples 200
"""

import csv
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ase
import click
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

from nvmolkit import clustering

from confsweeper import (
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mol_PE,
    get_uma_calc,
)
from torsional_sampling import load_ramachandran_grids, sample_constrained_confs
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
    "sampling_mode",
    "n_generated_confs",
    "n_ref_confs",
    "coverage",
    "mean_min_rmsd",
    "median_min_rmsd",
    "time_gpu_s",
    "time_coverage_s",
]

ERROR_COLUMNS = ["sequence", "n_confs", "sampling_mode", "error"]


def _load_checkpoint(output_csv: str) -> set[tuple[str, int, str]]:
    """Returns set of (sequence, n_confs, sampling_mode) triples already in output_csv."""
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            for row in csv.DictReader(f):
                mode = (
                    row.get("sampling_mode") or "etkdg"
                )  # back-compat: missing or empty → etkdg
                done.add((row["sequence"], int(row["n_confs"]), mode))
    return done


def _append_row(path: str, row: dict, columns: list[str]) -> None:
    """Appends a single row to a CSV, writing the header if the file is new."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _mace_score_conf_ids(mol, conf_ids: list[int], mace_calc) -> list[float]:
    """Return MACE potential energies for a list of conformer IDs."""
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    energies = []
    for cid in conf_ids:
        coords = mol.GetConformer(cid).GetPositions()
        ase_mol = ase.Atoms(positions=coords, numbers=atoms)
        ase_mol.calc = mace_calc
        energies.append(ase_mol.get_potential_energy())
        del ase_mol
        torch.cuda.empty_cache()
    return energies


def _butina_on_conf_ids(mol, conf_ids: list[int], cutoff: float) -> list[int]:
    """Run GPU Butina on a subset of conformers; return centroid conformer IDs."""
    coords = torch.tensor(
        np.array([mol.GetConformer(cid).GetPositions() for cid in conf_ids])
    )
    n_atoms = coords.shape[1]
    dists = torch.cdist(torch.flatten(coords, 1), torch.flatten(coords, 1), p=1.0) / (
        3 * n_atoms
    )
    _, centroids_result = clustering.butina(
        dists.to("cuda:0"), cutoff=cutoff, return_centroids=True
    )
    centroid_positions = centroids_result.numpy().tolist()
    return [conf_ids[p] for p in centroid_positions]


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
@click.option(
    "--torsional_sampling",
    is_flag=True,
    default=False,
    help="Enable torsional sampling (Pool B) in addition to standard ETKDG (Pool A)",
)
@click.option(
    "--torsional_n_samples",
    default=200,
    show_default=True,
    type=int,
    help="Number of (phi, psi) samples to draw for Pool B per molecule",
)
@click.option(
    "--torsional_strategy",
    default="uniform",
    show_default=True,
    type=click.Choice(["uniform", "inverse"]),
    help="Ramachandran sampling strategy: 'uniform' (all accessible cells equally) "
    "or 'inverse' (oversample rare cells)",
)
@click.option(
    "--torsional_energy_sigma",
    default=2.0,
    show_default=True,
    type=float,
    help="Pool B energy filter: keep conformers with energy < mean(A) + k*std(A)",
)
@click.option(
    "--ramachandran_grids",
    default="data/processed/cremp/ramachandran_grids.npz",
    show_default=True,
    help="Path to CREMP Ramachandran grids .npz (required with --torsional_sampling)",
)
@click.option(
    "--max_workers",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel worker threads",
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
    torsional_sampling,
    torsional_n_samples,
    torsional_strategy,
    torsional_energy_sigma,
    ramachandran_grids,
    max_workers,
):
    """Benchmark confsweeper conformer coverage against CREMP reference conformers."""
    n_confs_values = [int(x.strip()) for x in n_confs.split(",")]
    sampling_mode = "etkdg+torsional" if torsional_sampling else "etkdg"
    logger.info("n_confs sweep: %s  sampling_mode: %s", n_confs_values, sampling_mode)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(errors_csv) or ".", exist_ok=True)

    done = _load_checkpoint(output_csv)
    if done:
        logger.info(
            "Resuming: %d (sequence, n_confs, sampling_mode) triples already processed",
            len(done),
        )

    grids = None
    if torsional_sampling:
        grids = load_ramachandran_grids(ramachandran_grids)
        logger.info("Loaded Ramachandran grids from %s", ramachandran_grids)

    params = get_embed_params_macrocycle()
    hardware_opts = get_hardware_opts(
        preprocessingThreads=8, batch_size=1000, batchesPerGpu=2
    )
    mace_calc = get_uma_calc()

    # _gpu_lock: nvmolkit and MACE are not thread-safe — serialise all GPU work.
    # _write_lock: serialise CSV writes and done-set updates.
    _gpu_lock = threading.Lock()
    _write_lock = threading.Lock()

    def _compute_and_write(
        sequence, smiles, ref_mol, meta, n, gen_mol, gen_conf_ids, time_gpu_s
    ):
        """CPU-bound half: spyrmsd coverage then write result. Runs in parallel."""
        t0 = time.perf_counter()
        coverage, min_rmsds = calc_coverage(
            ref_mol=ref_mol,
            gen_mol=gen_mol,
            gen_conf_ids=gen_conf_ids,
            rmsd_cutoff=rmsd_cutoff,
            filter_factor=filter_factor,
        )
        time_coverage_s = time.perf_counter() - t0

        finite_rmsds = [r for r in min_rmsds if r != float("inf")]
        mean_min_rmsd = float(np.mean(finite_rmsds)) if finite_rmsds else float("nan")
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
            "n_ref_confs": ref_mol.GetNumConformers(),
            "coverage": round(coverage, 6),
            "mean_min_rmsd": round(mean_min_rmsd, 4),
            "median_min_rmsd": round(median_min_rmsd, 4),
            "time_gpu_s": round(time_gpu_s, 3),
            "time_coverage_s": round(time_coverage_s, 3),
        }
        with _write_lock:
            _append_row(output_csv, row, OUTPUT_COLUMNS)
            done.add((sequence, n))

    coverage_futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        mol_iter = iter_validation_mols(subset_csv, pickle_dir)
        for sequence, smiles, ref_mol, meta in tqdm(mol_iter, desc="molecules"):
            for n in n_confs_values:
                if (sequence, n) in done:
                    continue
                try:
                    # GPU work: serialised so nvmolkit/MACE are never called concurrently.
                    t0 = time.perf_counter()
                    with _gpu_lock:
                        gen_mol, gen_conf_ids, _ = get_mol_PE(
                            smi=smiles,
                            params=params,
                            hardware_opts=hardware_opts,
                            mace_calc=mace_calc,
                            n_confs=n,
                            cutoff_dist=butina_thresh,
                            gpu_clustering=True,
                        )
                    time_gpu_s = time.perf_counter() - t0
                    gen_conf_ids = list(gen_conf_ids)

                    if len(gen_conf_ids) == 0:
                        raise ValueError("EmbedMolecules produced 0 conformers")

                    # CPU work: submit spyrmsd coverage to thread pool, overlapping
                    # with the next GPU job.
                    future = executor.submit(
                        _compute_and_write,
                        sequence,
                        smiles,
                        ref_mol,
                        meta,
                        n,
                        gen_mol,
                        gen_conf_ids,
                        time_gpu_s,
                    )
                    coverage_futures.append(future)

                except Exception as e:
                    logger.warning(
                        "Error processing %s at n_confs=%d: %s", sequence, n, e
                    )
                    with _write_lock:
                        _append_row(
                            errors_csv,
                            {"sequence": sequence, "n_confs": n, "error": str(e)},
                            ERROR_COLUMNS,
                        )

        # Drain remaining coverage futures after the molecule loop finishes.
        for future in tqdm(
            as_completed(coverage_futures),
            total=len(coverage_futures),
            desc="coverage (draining)",
        ):
            try:
                future.result()
            except Exception as e:
                logger.error("Unexpected error in coverage worker: %s", e)


if __name__ == "__main__":
    run_coverage_benchmark()
