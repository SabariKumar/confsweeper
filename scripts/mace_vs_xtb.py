"""
MACE-OFF23 vs xtb GFN2 single-point energy comparison on the same conformer pool.

Purpose: rule MACE out (or in) as the cause of the near-one-hot Boltzmann weight
distributions seen in the saturation experiments. If both backends produce
similar BW vectors on the same conformer set, sampling is the bottleneck and
the energy model is fine. If MACE concentrates BW while xtb spreads it (or
vice-versa), the energy model itself is suspect and the production pipeline's
backend choice matters.

Pipeline (one peptide per invocation):
    1. Embed n_seeds conformers via nvmolkit ETKDG (etkdgv3_macrocycle params).
    2. Score the full pool with MACE batched on GPU (single forward pass).
    3. Score the same pool with xtb GFN2 single-point in parallel CPU workers.
    4. Compute BW vectors under each backend and report:
         - max_bw, eff_n, n_within_kT, n_within_3kT (per backend)
         - Pearson r of (E_MACE - mean) vs (E_xtb - mean)
         - Spearman rank correlation (rank stability across backends)
         - max disagreement: highest-BW conformer under each backend
                             and how the other backend scores it
    5. Optionally write a per-conformer CSV with both energies for follow-up.

xtb is CPU-only and runs as multiple subprocess workers (default 8). MACE
runs on GPU. The two stages are sequential but can coexist with a running
GPU sweep — the script does its GPU work first (embed + MACE) and then
hands off to CPU xtb workers.

Usage (in the confsweeper pixi environment):

    pixi run python scripts/mace_vs_xtb.py \\
        --smiles "..." \\
        --n_seeds 1000 \\
        --xtb_workers 8 \\
        --out_csv /tmp/mace_vs_xtb_pampa_large.csv
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
import numpy as np

# fmt: off
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nvmolkit.embedMolecules as embed  # noqa: E402  (must come after sys.path tweak)

from confsweeper import (  # noqa: E402
    _KT_EV_298K,
    _mace_batch_energies,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
)

# fmt: on


_HARTREE_TO_EV = 27.211386245988


def _bw(energies_eV: np.ndarray) -> np.ndarray:
    """
    Compute normalised Boltzmann weights at 298 K from energies in eV.

    Params:
        energies_eV: ndarray [N] : per-conformer energies in eV
    Returns:
        ndarray [N] float64 of weights summing to 1
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    w = np.exp(-(e - e.min()) / _KT_EV_298K)
    return w / w.sum()


def _bw_summary(energies_eV: np.ndarray) -> dict:
    """
    Boltzmann-weight summary stats for a single energy backend.

    Params:
        energies_eV: ndarray [N] : per-conformer energies in eV
    Returns:
        dict with keys n, max_bw, eff_n, entropy, n_within_kT, n_within_3kT,
            e_min_eV, e_range_eV
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size == 0:
        return {"n": 0}
    w = _bw(e)
    de = e - e.min()
    return {
        "n": int(e.size),
        "max_bw": float(w.max()),
        "eff_n": float(1.0 / (w**2).sum()),
        "entropy": float(-(w * np.log(w.clip(1e-45))).sum()),
        "n_within_kT": int((de <= _KT_EV_298K).sum()),
        "n_within_3kT": int((de <= 3 * _KT_EV_298K).sum()),
        "e_min_eV": float(e.min()),
        "e_range_eV": float(de.max()),
    }


def _write_xyz(symbols: list[str], coords: np.ndarray, path: str) -> None:
    """
    Write an XYZ file (n-line header + element + coords).

    Params:
        symbols: list[str] : atomic element symbols, length N
        coords: ndarray [N, 3] float : Cartesian positions in Å
        path: str : output file path
    Returns:
        None
    """
    n = len(symbols)
    lines = [str(n), ""]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    Path(path).write_text("\n".join(lines) + "\n")


def _xtb_one(args: tuple[str, list[str], np.ndarray]) -> float:
    """
    Run xtb GFN2 single-point on one conformer and return the energy in eV.

    Designed for use in a multiprocessing.Pool — accepts a single tuple arg.
    Sets OMP_NUM_THREADS=1 in the worker environment so the pool's parallelism
    is across conformers, not within each xtb call.

    Params:
        args: tuple : (label, symbols, coords [N, 3])
    Returns:
        float : MACE-comparable energy in eV (sign-aligned), NaN if xtb failed
    """
    label, symbols, coords = args
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as td:
        xyz = os.path.join(td, f"{label}.xyz")
        _write_xyz(symbols, coords, xyz)
        result = subprocess.run(
            ["xtb", xyz, "--gfn", "2", "--sp", "--silent", "--norestart"],
            cwd=td,
            capture_output=True,
            text=True,
            env=env,
        )
    if result.returncode != 0:
        return float("nan")
    for line in result.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            # Format: "| TOTAL ENERGY              -172.0109957 Eh   |"
            try:
                hartree = float(line.split()[3])
                return hartree * _HARTREE_TO_EV
            except (IndexError, ValueError):
                return float("nan")
    return float("nan")


def _embed_pool(smi: str, n_seeds: int, hardware_opts) -> tuple[list[str], np.ndarray]:
    """
    Embed `n_seeds` ETKDG conformers via nvmolkit using macrocycle params.

    Params:
        smi: str : input SMILES
        n_seeds: int : number of conformer attempts
        hardware_opts : nvmolkit hardware options
    Returns:
        tuple (symbols, coords) where symbols is a list of element symbols
        (length N atoms) and coords is an ndarray [n_conformers, N, 3]
    """
    from rdkit import Chem

    params = get_embed_params_macrocycle()
    params.randomSeed = 0
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    embed.EmbedMolecules(
        [mol], params, confsPerMolecule=n_seeds, hardwareOptions=hardware_opts
    )
    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        return [], np.zeros((0, 0, 3), dtype=np.float64)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.stack(
        [
            np.asarray(mol.GetConformer(c.GetId()).GetPositions(), dtype=np.float64)
            for c in mol.GetConformers()
        ]
    )
    return symbols, coords


def _score_mace(
    symbols: list[str], coords: np.ndarray, calc, score_chunk_size: int
) -> np.ndarray:
    """
    MACE-score every conformer in `coords` in chunks of `score_chunk_size`.

    Params:
        symbols: list[str] : per-atom element symbols
        coords: ndarray [n_conf, n_atoms, 3] : conformer geometries
        calc : MACECalculator from get_mace_calc()
        score_chunk_size: int : per-batch MACE forward pass cap
    Returns:
        ndarray [n_conf] float : MACE energies in eV
    """
    import ase

    n = coords.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, score_chunk_size):
        chunk = coords[start : start + score_chunk_size]
        ase_mols = [ase.Atoms(positions=c, symbols=symbols) for c in chunk]
        out[start : start + len(chunk)] = _mace_batch_energies(calc, ase_mols)
    return out


def _score_xtb(symbols: list[str], coords: np.ndarray, n_workers: int) -> np.ndarray:
    """
    xtb GFN2 single-point energies for every conformer in `coords`, in parallel.

    Each worker runs an independent xtb subprocess with OMP_NUM_THREADS=1.

    Params:
        symbols: list[str] : per-atom element symbols
        coords: ndarray [n_conf, n_atoms, 3] : conformer geometries
        n_workers: int : number of parallel xtb subprocesses
    Returns:
        ndarray [n_conf] float : xtb energies in eV (NaN on failure)
    """
    args_list = [(f"c{i}", symbols, coords[i]) for i in range(coords.shape[0])]
    if n_workers <= 1:
        return np.asarray([_xtb_one(a) for a in args_list], dtype=np.float64)
    with mp.Pool(processes=n_workers) as pool:
        return np.asarray(pool.map(_xtb_one, args_list), dtype=np.float64)


def _print_summary(label: str, summary: dict) -> None:
    """
    Print a single-backend BW summary block.

    Params:
        label: str : header label (e.g. 'MACE' or 'xtb')
        summary: dict : output of _bw_summary
    Returns:
        None
    """
    print(f"  ── {label}")
    print(f"      n_conformers:  {summary['n']}")
    print(f"      max_bw:        {summary['max_bw']:.4f}")
    print(f"      eff_n:         {summary['eff_n']:.2f}")
    print(f"      entropy:       {summary['entropy']:.4f}")
    print(f"      n<=kT:         {summary['n_within_kT']}")
    print(f"      n<=3kT:        {summary['n_within_3kT']}")
    print(f"      e_min  (eV):   {summary['e_min_eV']:.4f}")
    print(f"      e_range (eV):  {summary['e_range_eV']:.4f}")


def _correlation_block(e_mace: np.ndarray, e_xtb: np.ndarray) -> None:
    """
    Print Pearson and Spearman correlations + max-BW disagreement diagnostics.

    Params:
        e_mace: ndarray [n] float : MACE energies in eV
        e_xtb: ndarray [n] float : xtb energies in eV (may contain NaN)
    Returns:
        None
    """
    mask = ~(np.isnan(e_mace) | np.isnan(e_xtb))
    em, ex = e_mace[mask], e_xtb[mask]
    n = mask.sum()
    print(f"  ── cross-backend (n={n} valid pairs)")
    if n < 2:
        print("      not enough valid pairs to compute correlations")
        return
    # Pearson on relative energies
    pearson = float(np.corrcoef(em - em.mean(), ex - ex.mean())[0, 1])
    # Spearman on ranks
    rm = np.argsort(np.argsort(em))
    rx = np.argsort(np.argsort(ex))
    spearman = float(np.corrcoef(rm, rx)[0, 1])
    print(f"      Pearson r (eV-shifted):  {pearson:+.4f}")
    print(f"      Spearman r (rank):       {spearman:+.4f}")
    # Disagreement: pick top-1 BW conformer under each backend, see what the
    # other backend says about it.
    bw_m, bw_x = _bw(em), _bw(ex)
    top_m, top_x = int(np.argmax(bw_m)), int(np.argmax(bw_x))
    if top_m == top_x:
        print(
            f"      top-BW conformer agrees: idx={top_m}  (BW_MACE={bw_m[top_m]:.4f}, BW_xtb={bw_x[top_x]:.4f})"
        )
    else:
        print(f"      top-BW disagreement:")
        print(
            f"        MACE picks conformer {top_m}: BW_MACE={bw_m[top_m]:.4f}, BW_xtb={bw_x[top_m]:.4f}"
        )
        print(
            f"        xtb  picks conformer {top_x}: BW_MACE={bw_m[top_x]:.4f}, BW_xtb={bw_x[top_x]:.4f}"
        )


@click.command()
@click.option("--smiles", required=True, help="SMILES of the peptide to benchmark")
@click.option(
    "--label", default="peptide", help="Human-readable label for output headers"
)
@click.option("--n_seeds", default=1000, type=int)
@click.option(
    "--xtb_workers",
    default=8,
    type=int,
    help="Number of parallel xtb subprocesses (each pinned to OMP_NUM_THREADS=1)",
)
@click.option(
    "--score_chunk_size", default=500, type=int, help="MACE batched forward-pass cap"
)
@click.option(
    "--out_csv",
    default=None,
    type=Path,
    help="Optional per-conformer CSV with both energies",
)
def main(
    smiles: str,
    label: str,
    n_seeds: int,
    xtb_workers: int,
    score_chunk_size: int,
    out_csv: Path | None,
) -> None:
    """
    Embed one peptide pool, score it under MACE and xtb, report comparison.

    Params:
        smiles: str : SMILES of the peptide
        label: str : output label
        n_seeds: int : number of ETKDG seeds to embed
        xtb_workers: int : parallel xtb workers
        score_chunk_size: int : MACE forward-pass cap
        out_csv: Path | None : optional per-conformer CSV with both energies
    Returns:
        None
    """
    print(f"=== {label} ===  smiles={smiles[:60]}{'...' if len(smiles) > 60 else ''}")
    print(f"n_seeds={n_seeds}  xtb_workers={xtb_workers}")

    hw = get_hardware_opts()
    t = time.perf_counter()
    symbols, coords = _embed_pool(smiles, n_seeds, hw)
    t_embed = time.perf_counter() - t
    n_conf, n_atoms = coords.shape[0], coords.shape[1]
    print(
        f"\n[1/3] embed:  {n_conf}/{n_seeds} conformers, {n_atoms} atoms,  {t_embed:.1f}s"
    )
    if n_conf == 0:
        print("ABORT: no conformers embedded")
        return

    print(f"\n[2/3] MACE scoring  (batched, GPU)")
    calc = get_mace_calc()
    t = time.perf_counter()
    e_mace = _score_mace(symbols, coords, calc, score_chunk_size=score_chunk_size)
    t_mace = time.perf_counter() - t
    print(f"      done: {t_mace:.1f}s  ({1000 * t_mace / n_conf:.1f} ms/conf)")

    print(f"\n[3/3] xtb scoring  ({xtb_workers} workers, 1 thread each)")
    t = time.perf_counter()
    e_xtb = _score_xtb(symbols, coords, n_workers=xtb_workers)
    t_xtb = time.perf_counter() - t
    n_xtb_ok = int((~np.isnan(e_xtb)).sum())
    print(
        f"      done: {t_xtb:.1f}s ({1000 * t_xtb / n_conf:.1f} ms/conf wall, "
        f"{1000 * t_xtb * xtb_workers / n_conf:.1f} ms/conf CPU)  "
        f"{n_xtb_ok}/{n_conf} succeeded"
    )

    print()
    print("===  Boltzmann-weight comparison  ===")
    print()
    _print_summary("MACE", _bw_summary(e_mace))
    print()
    valid = ~np.isnan(e_xtb)
    if valid.sum() > 0:
        _print_summary("xtb (GFN2)", _bw_summary(e_xtb[valid]))
    print()
    _correlation_block(e_mace, e_xtb)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conf_idx", "e_mace_eV", "e_xtb_eV"])
            for i, (em, ex) in enumerate(zip(e_mace, e_xtb)):
                w.writerow([i, f"{em:.6f}", f"{ex:.6f}" if not np.isnan(ex) else ""])
        print(f"\nper-conformer CSV written to {out_csv}")


if __name__ == "__main__":
    main()
