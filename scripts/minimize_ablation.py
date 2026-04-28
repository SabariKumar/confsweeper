"""
Controlled MMFF-minimization ablation: same input pool, only the minimize
flag varies.

Motivation: the n=10k saturation results showed minimize=True regressing on
the larger PAMPA peptides (pampa_medium, pampa_large) but improving smaller
ones. Confound: the no-minimize and minimize runs in the saturation sweep
used different nvmolkit ETKDG seeds and therefore different starting pools,
so we cannot attribute the regression to MMFF rather than ETKDG draw luck.

This script removes the confound by embedding once and running pipeline A
(MACE → filter → dedup → BW) and pipeline B (MMFF → MACE → filter → dedup
→ BW) on the **same starting pool**. Side-by-side BW statistics tell us
whether MMFF itself causes the regression on large peptides or whether the
saturation-sweep regression was sampling luck.

Pipeline:
    1. Embed n_seeds conformers via nvmolkit ETKDG (etkdgv3_macrocycle).
    2. Deepcopy the mol so both pipelines start from identical geometries.
    3. Pipeline A (no minimize): MACE-score → energy filter → dedup → BW.
    4. Pipeline B (minimize=True): GPU MMFF in place → MACE-score → energy
       filter → dedup → BW.
    5. Print BW summaries side-by-side, plus per-conformer correlations and
       basin-membership comparison.

Designed to run on multiple peptides in one invocation by passing
--peptides comma-separated. Defaults to pampa_large (the most extreme
regression case from the saturation sweep).

Usage:

    pixi run python scripts/minimize_ablation.py \\
        --peptides pampa_large,pampa_medium \\
        --n_seeds 10000 \\
        --out_csv /home/sabari/confsweeper/data/processed/saturation/minimize_ablation.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import click
import numpy as np

# fmt: off
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nvmolkit.embedMolecules as embed  # noqa: E402

from confsweeper import (  # noqa: E402
    _KT_EV_298K,
    _energy_ranked_dedup,
    _mace_batch_energies,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
)

# fmt: on


# Built-in peptide library — sourced from the saturation sweep so the
# comparison is on the exact same molecules.
PEPTIDE_LIBRARY: dict[str, dict] = {
    "cremp_typical": {
        "smiles": "OCC(N)C(=O)NC(C)C(=O)N1CCCC1C(=O)NCC(=O)N",  # placeholder; replaced from CSV at startup
        "n_heavy_expected": 27,
    },
    "cremp_sharp": {
        "smiles": None,
        "n_heavy_expected": 50,
    },
    "pampa_small": {
        "smiles": None,
        "n_heavy_expected": 51,
    },
    "pampa_medium": {
        "smiles": None,
        "n_heavy_expected": 70,
    },
    "pampa_large": {
        "smiles": None,
        "n_heavy_expected": 103,
    },
}

OUTPUT_COLUMNS = [
    "peptide_id",
    "n_heavy",
    "n_seeds",
    "n_pool",
    "pipeline",
    "n_basins",
    "max_bw",
    "eff_n",
    "entropy",
    "n_within_kT",
    "n_within_3kT",
    "e_min_eV",
    "e_range_eV",
    "time_total_s",
]


def _load_peptide_smiles(saturation_csv: Path) -> dict[str, dict]:
    """
    Look up SMILES strings for the saturation peptide library by reading the
    saturation results CSV (which already contains them in the `smiles`
    column, recorded per row).

    Params:
        saturation_csv: Path : saturation_etkdg.csv (output of saturation sweep)
    Returns:
        dict[str, dict] : peptide library keyed by peptide_id base label
    """
    import pandas as pd

    df = pd.read_csv(saturation_csv)
    library: dict[str, dict] = {}
    for _, row in df.drop_duplicates("peptide_id").iterrows():
        # Strip any "label:sequence" suffix the CREMP picks use, then keep
        # the base label.
        base = row["peptide_id"].split(":")[0]
        if base not in library:
            library[base] = {
                "peptide_id": row["peptide_id"],
                "smiles": row["smiles"],
                "n_heavy": int(row["n_heavy"]),
            }
    return library


def _bw_summary(energies_eV: np.ndarray) -> dict:
    """
    Boltzmann-weight summary stats for a single basin set.

    Params:
        energies_eV: ndarray [N] : per-basin energies in eV
    Returns:
        dict with keys n_basins, max_bw, eff_n, entropy, n_within_kT,
            n_within_3kT, e_min_eV, e_range_eV
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size == 0:
        return dict(
            n_basins=0,
            max_bw=float("nan"),
            eff_n=float("nan"),
            entropy=float("nan"),
            n_within_kT=0,
            n_within_3kT=0,
            e_min_eV=float("nan"),
            e_range_eV=float("nan"),
        )
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
        "e_range_eV": float(de.max()),
    }


def _embed_pool(smi: str, n_seeds: int, hardware_opts, embed_chunk_size: int):
    """
    Embed `n_seeds` conformers via nvmolkit ETKDG (etkdgv3_macrocycle params).
    Chunked when n_seeds > embed_chunk_size to bound GPU memory.

    Params:
        smi: str : SMILES
        n_seeds: int : total seeds
        hardware_opts : nvmolkit hardware options
        embed_chunk_size: int : per-call cap
    Returns:
        rdkit.Chem.Mol with conformers attached
    """
    from rdkit import Chem

    params = get_embed_params_macrocycle()
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if n_seeds <= embed_chunk_size:
        params.randomSeed = 0
        embed.EmbedMolecules(
            [mol], params, confsPerMolecule=n_seeds, hardwareOptions=hardware_opts
        )
    else:
        n_remaining = n_seeds
        chunk_idx = 0
        while n_remaining > 0:
            this_chunk = min(embed_chunk_size, n_remaining)
            params.randomSeed = chunk_idx * embed_chunk_size
            n_before = mol.GetNumConformers()
            embed.EmbedMolecules(
                [mol],
                params,
                confsPerMolecule=this_chunk,
                hardwareOptions=hardware_opts,
            )
            if mol.GetNumConformers() == n_before:
                break
            n_remaining -= this_chunk
            chunk_idx += 1
    return mol


def _score_pool_mace(mol, calc, score_chunk_size: int) -> np.ndarray:
    """
    MACE-score every conformer attached to `mol` in chunks. Returns ndarray [N].

    Params:
        mol : RDKit mol with conformers
        calc : MACE calculator
        score_chunk_size: int : per-batch cap
    Returns:
        ndarray [N] float64 of energies in eV, in conformer ID order
    """
    import ase

    conf_ids = [c.GetId() for c in mol.GetConformers()]
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    out = np.empty(len(conf_ids), dtype=np.float64)
    for start in range(0, len(conf_ids), score_chunk_size):
        chunk_ids = conf_ids[start : start + score_chunk_size]
        ase_mols = [
            ase.Atoms(
                positions=mol.GetConformer(cid).GetPositions(), numbers=atomic_nums
            )
            for cid in chunk_ids
        ]
        out[start : start + len(chunk_ids)] = _mace_batch_energies(calc, ase_mols)
    return out


def _filter_and_dedup(
    mol,
    energies_eV: np.ndarray,
    e_window_kT: float,
    rmsd_threshold: float,
) -> tuple[list[int], np.ndarray]:
    """
    Apply 5 kT energy filter then energy-ranked geometric dedup.

    Params:
        mol : RDKit mol with conformers (used to fetch positions for dedup)
        energies_eV: ndarray [N] : per-conformer energies in eV
        e_window_kT: float : energy filter window in kT_298K units
        rmsd_threshold: float : geometric dedup exclusion radius
    Returns:
        tuple (kept_conf_ids, kept_energies) of basin centroids
    """
    import torch

    conf_ids = [c.GetId() for c in mol.GetConformers()]
    e = np.asarray(energies_eV, dtype=np.float64)
    e_min = e.min()
    keep = (e - e_min) <= e_window_kT * _KT_EV_298K
    if not keep.any():
        keep = np.zeros_like(keep)
        keep[int(np.argmin(e))] = True
    kept_idx = np.where(keep)[0].tolist()
    kept_ids = [conf_ids[i] for i in kept_idx]
    kept_e = e[kept_idx]
    if len(kept_ids) == 1:
        return kept_ids, kept_e
    coords = torch.tensor(
        np.array([mol.GetConformer(cid).GetPositions() for cid in kept_ids])
    )
    centroid_idx = _energy_ranked_dedup(coords, kept_e, rmsd_threshold=rmsd_threshold)
    return [kept_ids[i] for i in centroid_idx], np.asarray(
        [kept_e[i] for i in centroid_idx], dtype=np.float64
    )


def _print_summary(pipeline_label: str, summary: dict, t_total: float) -> None:
    """
    Print one pipeline's BW summary block.

    Params:
        pipeline_label: str : display label ('A: no-minimize' or 'B: minimize=True')
        summary: dict : output of _bw_summary
        t_total: float : wall-clock seconds for this pipeline
    Returns:
        None
    """
    print(f"  ── {pipeline_label}  ({t_total:.1f}s)")
    print(f"      n_basins:      {summary['n_basins']}")
    print(f"      max_bw:        {summary['max_bw']:.4f}")
    print(f"      eff_n:         {summary['eff_n']:.2f}")
    print(f"      entropy:       {summary['entropy']:.4f}")
    print(f"      n<=kT:         {summary['n_within_kT']}")
    print(f"      n<=3kT:        {summary['n_within_3kT']}")
    print(f"      e_range (eV):  {summary['e_range_eV']:.4f}")


@click.command()
@click.option(
    "--peptides",
    default="pampa_large",
    help="Comma-separated peptide labels from the saturation library "
    "(any of cremp_typical, cremp_sharp, pampa_small, pampa_medium, pampa_large)",
)
@click.option(
    "--saturation_csv",
    default=Path(
        "/home/sabari/confsweeper/data/processed/saturation/saturation_etkdg.csv"
    ),
    type=Path,
    help="CSV used to look up SMILES + n_heavy for each peptide label",
)
@click.option("--n_seeds", default=10000, type=int)
@click.option("--embed_chunk_size", default=1000, type=int)
@click.option("--score_chunk_size", default=500, type=int)
@click.option("--e_window_kT", default=5.0, type=float)
@click.option("--rmsd_threshold", default=0.1, type=float)
@click.option(
    "--out_csv",
    default=None,
    type=Path,
    help="Optional output CSV with one row per (peptide, pipeline)",
)
def main(
    peptides: str,
    saturation_csv: Path,
    n_seeds: int,
    embed_chunk_size: int,
    score_chunk_size: int,
    e_window_kt: float,
    rmsd_threshold: float,
    out_csv: Path | None,
) -> None:
    """
    Run the controlled minimize=True vs minimize=False ablation on each
    requested peptide, embedding once and scoring twice.

    Params:
        peptides: str : comma-separated peptide labels
        saturation_csv: Path : source of SMILES + n_heavy lookups
        n_seeds: int : ETKDG seeds per peptide
        embed_chunk_size: int : per-call embed cap
        score_chunk_size: int : per-batch MACE forward-pass cap
        e_window_kt: float : energy filter window in kT_298K units (Click
            lowercases the option name)
        rmsd_threshold: float : geometric dedup exclusion radius
        out_csv: Path | None : optional output CSV
    Returns:
        None
    """
    library = _load_peptide_smiles(saturation_csv)
    labels = [s.strip() for s in peptides.split(",") if s.strip()]
    missing = [s for s in labels if s not in library]
    if missing:
        raise click.BadParameter(
            f"unknown peptide labels: {missing}; available: {sorted(library)}"
        )

    print(
        f"=== Minimize ablation  n_seeds={n_seeds}  e_window_kT={e_window_kt}  rmsd_threshold={rmsd_threshold} ===\n"
    )

    hw = get_hardware_opts()
    calc = get_mace_calc()
    rows: list[dict] = []

    for label in labels:
        pep = library[label]
        print(f"\n*** {label}  ({pep['n_heavy']} heavy atoms)")
        print(
            f"    smiles: {pep['smiles'][:80]}{'...' if len(pep['smiles']) > 80 else ''}"
        )

        # 1. Embed once.
        t = time.perf_counter()
        mol_seed = _embed_pool(pep["smiles"], n_seeds, hw, embed_chunk_size)
        n_pool = mol_seed.GetNumConformers()
        t_embed = time.perf_counter() - t
        print(f"    [embed] {n_pool}/{n_seeds} conformers, {t_embed:.1f}s")
        if n_pool == 0:
            print("    ABORT: nothing embedded")
            continue

        from rdkit import Chem

        # 2. Pipeline A — no minimize. Deepcopy so MMFF in pipeline B can't
        # mutate the geometry pipeline A reads.
        mol_a = Chem.Mol(mol_seed)
        t = time.perf_counter()
        e_a = _score_pool_mace(mol_a, calc, score_chunk_size=score_chunk_size)
        ids_a, basin_e_a = _filter_and_dedup(
            mol_a,
            e_a,
            e_window_kT=e_window_kt,
            rmsd_threshold=rmsd_threshold,
        )
        t_a = time.perf_counter() - t
        sa = _bw_summary(basin_e_a)

        # 3. Pipeline B — GPU MMFF then identical scoring. Same starting
        # geometry as pipeline A.
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

        mol_b = Chem.Mol(mol_seed)
        t = time.perf_counter()
        MMFFOptimizeMoleculesConfs([mol_b], hardwareOptions=hw)
        e_b = _score_pool_mace(mol_b, calc, score_chunk_size=score_chunk_size)
        ids_b, basin_e_b = _filter_and_dedup(
            mol_b,
            e_b,
            e_window_kT=e_window_kt,
            rmsd_threshold=rmsd_threshold,
        )
        t_b = time.perf_counter() - t
        sb = _bw_summary(basin_e_b)

        # 4. Report.
        print()
        _print_summary("A: no minimize", sa, t_a)
        print()
        _print_summary("B: minimize=True (GPU MMFF)", sb, t_b)
        print()
        # MACE-energy rank correlation pre/post-MMFF — does the per-conformer
        # ranking change much, or does MMFF mostly translate the landscape?
        if e_a.size > 1:
            rank_a = np.argsort(np.argsort(e_a))
            rank_b = np.argsort(np.argsort(e_b))
            spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
            print(
                f"  ── Spearman r of per-conformer MACE energies (pre vs post MMFF): {spearman:+.4f}"
            )

        for pl_label, summary, t_total in [
            ("no_minimize", sa, t_a),
            ("minimize_gpu", sb, t_b),
        ]:
            rows.append(
                {
                    "peptide_id": pep["peptide_id"],
                    "n_heavy": pep["n_heavy"],
                    "n_seeds": n_seeds,
                    "n_pool": n_pool,
                    "pipeline": pl_label,
                    **summary,
                    "time_total_s": t_total,
                }
            )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
