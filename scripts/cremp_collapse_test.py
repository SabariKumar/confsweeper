"""
CREMP basin-collapse sanity check (Step 16 of docs/mcmm_plan.md).

Feeds CREMP's own GFN2-xTB-relaxed conformer sets through our
`MMFF + MACE + Kabsch-dedup` pipeline and counts survivors at each
stage. Discriminates between two interpretations of the
`n_basins ≪ uniqueconfs` benchmark gap:

  - "our sampling is deficient" (we never visit those geometries), vs.
  - "CREMP overcounts" (CREST + GFN2-xTB classifies sub-Å wobble as
    distinct, our MMFF + Kabsch-0.125 Å pipeline collapses them).

Per peptide reports counts at each stage and e_min under both xtb and
MACE so the verdict can be read off directly. See Step 16's decision
tree in docs/mcmm_plan.md.

Usage:
    pixi run python scripts/cremp_collapse_test.py \\
        --pickle_dir data/raw/cremp/pickle \\
        --peptides   t.I.G.N S.S.N.MeW.MeA.MeN \\
        --out_csv    results/cremp_collapse_test.csv
"""

from __future__ import annotations

import csv
import logging
import pickle
import sys
from pathlib import Path

import ase
import click
import numpy as np
import torch
from rdkit import Chem

# Match sampler_benchmark.py's import shape so this script picks up the
# in-tree confsweeper rather than any installed copy.
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from confsweeper import (  # noqa: E402
    _KT_EV_298K,
    _energy_ranked_dedup,
    _mace_batch_energies,
    get_hardware_opts,
    get_mace_calc,
)

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    """
    Return atom indices of all non-hydrogen atoms.

    Params:
        mol: Chem.Mol : input mol
    Returns:
        list[int] : indices into mol's atom list, heavy atoms only
    """
    return [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() != 1]


def _coords_tensor(mol: Chem.Mol) -> torch.Tensor:
    """
    Stack all conformer coordinates of `mol` into a single tensor.

    Params:
        mol: Chem.Mol : mol with N conformers attached
    Returns:
        torch.Tensor : (N, n_atoms, 3) float64 coordinate tensor
    """
    return torch.tensor(
        np.array([c.GetPositions() for c in mol.GetConformers()]),
        dtype=torch.float64,
    )


def _mace_score_all(mol: Chem.Mol, calc, score_chunk_size: int = 500) -> np.ndarray:
    """
    MACE-score every conformer attached to `mol`, batched in chunks.

    Params:
        mol: Chem.Mol : mol with N conformers (Hs included)
        calc : MACECalculator from `get_mace_calc()`
        score_chunk_size: int : per-batch forward-pass cap
    Returns:
        ndarray (N,) : potential energies in eV, in conformer-id order
    """
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    energies: list[float] = []
    for start in range(0, len(conf_ids), score_chunk_size):
        chunk_ids = conf_ids[start : start + score_chunk_size]
        ase_mols = [
            ase.Atoms(
                positions=mol.GetConformer(cid).GetPositions(),
                numbers=atomic_nums,
            )
            for cid in chunk_ids
        ]
        energies.extend(_mace_batch_energies(calc, ase_mols))
    return np.asarray(energies, dtype=np.float64)


def _filter_then_dedup(
    coords: torch.Tensor,
    energies: np.ndarray,
    heavy_atom_indices: list[int],
    rmsd_threshold: float,
    e_window_kT: float = 5.0,
    *,
    dedup_mode: str = "kabsch",
    energy_threshold_eV: float = 0.05,
    rotconst_anisotropy_threshold: float = 0.01,
    atomic_numbers=None,
) -> int:
    """
    Apply the standard 5 kT energy filter then Kabsch (or CREST-style
    three-criteria) dedup at the given threshold; return the number of
    survivors.

    Mirrors stages 3+4 of `_minimize_score_filter_dedup` so the
    counts we report here are directly comparable to what
    `get_mol_PE_mcmm` would report for the same coords.

    Params:
        coords: torch.Tensor (N, n_atoms, 3) : already-relaxed coords
        energies: ndarray (N,) : potential energies in eV
        heavy_atom_indices: list[int] : atom subset for the metric
        rmsd_threshold: float : Kabsch RMSD threshold in Å
        e_window_kT: float : energy filter window in kT_298K units
        dedup_mode: str : 'kabsch' (default) or 'crest'
        energy_threshold_eV: float : crest-mode energy criterion in eV
        rotconst_anisotropy_threshold: float : crest-mode rotational
            anisotropy threshold (relative)
        atomic_numbers: list[int] | None : required for crest mode
    Returns:
        int : number of basin survivors
    """
    e_min = energies.min()
    keep_mask = (energies - e_min) <= e_window_kT * _KT_EV_298K
    if not keep_mask.any():
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[int(np.argmin(energies))] = True
    kept_idx = np.where(keep_mask)[0]
    kept_coords = coords[kept_idx]
    kept_energies = energies[kept_idx]
    if len(kept_coords) == 1:
        return 1
    centroids = _energy_ranked_dedup(
        kept_coords,
        kept_energies,
        rmsd_threshold=rmsd_threshold,
        heavy_atom_indices=heavy_atom_indices,
        dedup_mode=dedup_mode,
        energy_threshold_eV=energy_threshold_eV,
        rotconst_anisotropy_threshold=rotconst_anisotropy_threshold,
        atomic_numbers=atomic_numbers,
    )
    return len(centroids)


def _run_one_peptide(
    sequence: str,
    pickle_path: Path,
    calc,
    hardware_opts,
    score_chunk_size: int = 500,
) -> dict:
    """
    Run the full collapse-test pipeline on one CREMP peptide.

    Params:
        sequence: str : CREMP sequence id (e.g. "t.I.G.N")
        pickle_path: Path : path to <sequence>.pickle
        calc : MACECalculator from `get_mace_calc()`
        hardware_opts : nvmolkit hardware options for batched MMFF
        score_chunk_size: int : MACE per-batch forward-pass cap
    Returns:
        dict : one row of the output CSV (see OUTPUT_COLUMNS below)
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    mol = Chem.Mol(data["rd_mol"])
    n_confs = mol.GetNumConformers()
    if n_confs != data["uniqueconfs"]:
        logger.warning(
            "uniqueconfs (%d) != rd_mol.GetNumConformers() (%d) for %s",
            data["uniqueconfs"],
            n_confs,
            sequence,
        )
    heavy = _heavy_atom_indices(mol)
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    xtb_energies_kcal = np.array(
        [c["totalenergy"] for c in data["conformers"]], dtype=np.float64
    )
    # CREST's energy criterion is 0.05 kcal/mol = ~0.00217 eV. xtb is
    # precise enough for that threshold; we use it pre-MMFF where the
    # xtb-relaxed energies are the natural ranking signal.
    KCAL_PER_MOL_TO_EV = 0.04336
    xtb_energies_eV = xtb_energies_kcal * KCAL_PER_MOL_TO_EV
    CREST_ENERGY_THRESHOLD_EV = 0.05 * KCAL_PER_MOL_TO_EV  # CREST default

    logger.info(
        "  %s: loaded %d confs, %d heavy atoms, e_min(xtb)=%.5f kcal/mol",
        sequence,
        n_confs,
        len(heavy),
        float(xtb_energies_kcal.min()),
    )

    # Stage 1: pre-MMFF dedup, both kabsch (pure RMSD) and crest
    # (RMSD AND xtb-energy AND rotational-constant). Energy ranks come
    # from xtb so the cluster centre at each threshold is the
    # lowest-xtb-energy conformer in its basin — matches CREMP's own
    # ranking logic.
    raw_coords = _coords_tensor(mol)
    pre_mmff_kabsch_0125 = len(
        _energy_ranked_dedup(
            raw_coords,
            xtb_energies_kcal,
            rmsd_threshold=0.125,
            heavy_atom_indices=heavy,
        )
    )
    pre_mmff_kabsch_05 = len(
        _energy_ranked_dedup(
            raw_coords,
            xtb_energies_kcal,
            rmsd_threshold=0.5,
            heavy_atom_indices=heavy,
        )
    )
    pre_mmff_crest_0125 = len(
        _energy_ranked_dedup(
            raw_coords,
            xtb_energies_eV,
            rmsd_threshold=0.125,
            heavy_atom_indices=heavy,
            dedup_mode="crest",
            energy_threshold_eV=CREST_ENERGY_THRESHOLD_EV,
            rotconst_anisotropy_threshold=0.01,
            atomic_numbers=atomic_nums,
        )
    )
    logger.info(
        "  %s: pre-MMFF dedup → kabsch %d @ 0.125 Å,  %d @ 0.5 Å;  crest %d @ 0.125 Å",
        sequence,
        pre_mmff_kabsch_0125,
        pre_mmff_kabsch_05,
        pre_mmff_crest_0125,
    )

    # Stage 2: pre-MMFF MACE scoring (sanity check that MACE assigns
    # sensible energies to xtb-relaxed geometries).
    pre_mmff_mace_e = _mace_score_all(mol, calc, score_chunk_size=score_chunk_size)
    e_min_mace_pre = float(pre_mmff_mace_e.min())
    logger.info("  %s: pre-MMFF MACE e_min = %.4f eV", sequence, e_min_mace_pre)

    # Stage 3: MMFF-relax all conformers in place, batched.
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

    MMFFOptimizeMoleculesConfs([mol], hardwareOptions=hardware_opts)

    # Stage 4: post-MMFF MACE scoring + counts at each threshold under
    # the standard 5 kT energy filter, both kabsch and crest. Crest
    # uses the looser 0.05 eV energy threshold here (MACE float32 noise
    # floor); see Step 17 in docs/mcmm_plan.md.
    post_mmff_coords = _coords_tensor(mol)
    post_mmff_mace_e = _mace_score_all(mol, calc, score_chunk_size=score_chunk_size)
    e_min_mace_post = float(post_mmff_mace_e.min())

    e_min_post = post_mmff_mace_e.min()
    n_within_5kt = int(((post_mmff_mace_e - e_min_post) <= 5.0 * _KT_EV_298K).sum())
    post_mmff_kabsch_0125 = _filter_then_dedup(
        post_mmff_coords, post_mmff_mace_e, heavy, rmsd_threshold=0.125
    )
    post_mmff_kabsch_05 = _filter_then_dedup(
        post_mmff_coords, post_mmff_mace_e, heavy, rmsd_threshold=0.5
    )
    post_mmff_crest_0125 = _filter_then_dedup(
        post_mmff_coords,
        post_mmff_mace_e,
        heavy,
        rmsd_threshold=0.125,
        dedup_mode="crest",
        energy_threshold_eV=0.05,
        rotconst_anisotropy_threshold=0.01,
        atomic_numbers=atomic_nums,
    )
    logger.info(
        "  %s: post-MMFF n_within_5kt=%d  dedup → kabsch %d @ 0.125 Å,  %d @ 0.5 Å;  crest %d @ 0.125 Å",
        sequence,
        n_within_5kt,
        post_mmff_kabsch_0125,
        post_mmff_kabsch_05,
        post_mmff_crest_0125,
    )
    logger.info(
        "  %s: post-MMFF MACE e_min = %.4f eV  (Δ vs pre-MMFF = %.4f eV)",
        sequence,
        e_min_mace_post,
        e_min_mace_post - e_min_mace_pre,
    )

    return {
        "sequence": sequence,
        "uniqueconfs_cremp": int(data["uniqueconfs"]),
        "pre_mmff_kabsch_0125": pre_mmff_kabsch_0125,
        "pre_mmff_kabsch_05": pre_mmff_kabsch_05,
        "pre_mmff_crest_0125": pre_mmff_crest_0125,
        "post_mmff_within_5kt": n_within_5kt,
        "post_mmff_kabsch_0125": post_mmff_kabsch_0125,
        "post_mmff_kabsch_05": post_mmff_kabsch_05,
        "post_mmff_crest_0125": post_mmff_crest_0125,
        "e_min_xtb_kcal_per_mol": float(xtb_energies_kcal.min()),
        "e_min_mace_pre_mmff_eV": e_min_mace_pre,
        "e_min_mace_post_mmff_eV": e_min_mace_post,
    }


OUTPUT_COLUMNS = [
    "sequence",
    "uniqueconfs_cremp",
    "pre_mmff_kabsch_0125",
    "pre_mmff_kabsch_05",
    "pre_mmff_crest_0125",
    "post_mmff_within_5kt",
    "post_mmff_kabsch_0125",
    "post_mmff_kabsch_05",
    "post_mmff_crest_0125",
    "e_min_xtb_kcal_per_mol",
    "e_min_mace_pre_mmff_eV",
    "e_min_mace_post_mmff_eV",
    # Feature columns copied from the sampler's output CSV when available.
    # Empty string when the input CSV doesn't carry them.
    "topology",
    "has_proline",
    "has_glycine",
    "num_monomers",
]

# Feature columns optionally carried in from the sampler's output. We pass them
# straight through so the `summarize` subcommand can stratify without re-deriving.
_FEATURE_COLUMNS = ["topology", "has_proline", "has_glycine", "num_monomers"]


def _read_peptide_list(peptide_list_csv: Path) -> list[dict]:
    """
    Load the peptide list and any per-peptide feature columns the sampler
    emitted. Each returned entry has at least `sequence`, plus whatever
    feature columns are present in the source CSV.

    Params:
        peptide_list_csv: Path : CSV with a `sequence` column (the
            sampler's output, or any CSV with one column named `sequence`)
    Returns:
        list[dict] : one dict per peptide; keys are CSV column names
    """
    import pandas as pd

    df = pd.read_csv(peptide_list_csv)
    if "sequence" not in df.columns:
        raise click.ClickException(f"{peptide_list_csv} must have a `sequence` column")
    keep_cols = ["sequence"] + [c for c in _FEATURE_COLUMNS if c in df.columns]
    return df[keep_cols].to_dict(orient="records")


def _read_done_set(out_csv: Path) -> set[str]:
    """
    Return the set of sequences already present in `out_csv`. Used to
    skip completed work on resumed runs. Returns an empty set when the
    file doesn't exist.

    Params:
        out_csv: Path : path to the running output CSV
    Returns:
        set[str] : sequence ids already written
    """
    if not out_csv.exists():
        return set()
    import pandas as pd

    try:
        df = pd.read_csv(out_csv)
    except Exception as e:
        logger.warning("could not read existing %s: %s", out_csv, e)
        return set()
    if "sequence" not in df.columns:
        return set()
    return set(df["sequence"].astype(str).tolist())


def _append_row(out_csv: Path, row: dict) -> None:
    """
    Append one row to `out_csv`, writing the header on first call. Extra
    keys in `row` not in `OUTPUT_COLUMNS` are ignored.

    Params:
        out_csv: Path : output CSV path
        row: dict : row data
    Returns:
        None
    """
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@click.group()
def cli() -> None:
    """CREMP basin-collapse benchmark and summary tool (Step 19)."""


@cli.command("run")
@click.option(
    "--pickle_dir",
    required=True,
    type=Path,
    help="Path to the CREMP pickle/ directory (one <sequence>.pickle per peptide)",
)
@click.option(
    "--peptide_list_csv",
    required=True,
    type=Path,
    help="CSV with a `sequence` column. Optional `topology`, `has_proline`, "
    "`has_glycine`, `num_monomers` columns are passed through to the output "
    "for downstream stratification (the sampler emits them).",
)
@click.option(
    "--out_csv",
    required=True,
    type=Path,
    help="Output CSV path. Resumed automatically when the file already exists "
    "— sequences already present are skipped.",
)
@click.option(
    "--score_chunk_size",
    type=int,
    default=500,
    help="MACE per-batch forward pass cap (default 500)",
)
@click.option(
    "--mace_model",
    type=str,
    default="medium",
    help="MACE-OFF model size passed to get_mace_calc()",
)
def run_cmd(
    pickle_dir: Path,
    peptide_list_csv: Path,
    out_csv: Path,
    score_chunk_size: int,
    mace_model: str,
) -> None:
    """
    Run the CREMP basin-collapse sanity check on every peptide in
    `peptide_list_csv` and append rows to `out_csv` as each completes.
    Resumes by skipping sequences already in `out_csv`. Wraps each
    peptide's pipeline in try/except so a single bad pickle doesn't
    abort the run.

    Params:
        pickle_dir: Path : CREMP pickle directory
        peptide_list_csv: Path : input CSV with `sequence` column
        out_csv: Path : output CSV path (resume-aware)
        score_chunk_size: int : MACE per-batch cap
        mace_model: str : MACE-OFF size token (medium / large / etc.)
    Returns:
        None : appends rows to `out_csv` as each peptide completes
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    peptides = _read_peptide_list(peptide_list_csv)
    done = _read_done_set(out_csv)
    if done:
        logger.info("resuming: %d sequences already in %s", len(done), out_csv)

    todo = [p for p in peptides if p["sequence"] not in done]
    logger.info("processing %d peptides (%d already done)", len(todo), len(done))

    hardware_opts = get_hardware_opts()
    calc = get_mace_calc(model=mace_model)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for entry in todo:
        sequence = entry["sequence"]
        pickle_path = pickle_dir / f"{sequence}.pickle"
        if not pickle_path.exists():
            logger.warning("missing pickle for %s, skipping", sequence)
            n_skipped += 1
            continue
        logger.info(
            "[%d/%d] running %s", n_done + n_skipped + n_failed + 1, len(todo), sequence
        )
        try:
            row = _run_one_peptide(
                sequence,
                pickle_path,
                calc,
                hardware_opts,
                score_chunk_size=score_chunk_size,
            )
        except Exception as e:
            logger.exception("failed %s: %s", sequence, e)
            n_failed += 1
            continue
        # Pass through any feature columns the sampler attached.
        for col in _FEATURE_COLUMNS:
            if col in entry:
                row[col] = entry[col]
        _append_row(out_csv, row)
        n_done += 1

    logger.info(
        "done: %d succeeded, %d skipped (missing pickle), %d failed",
        n_done,
        n_skipped,
        n_failed,
    )


# Stratification axes the summary stratifies on.
_STRATIFICATION_KEYS = ["topology", "has_proline", "has_glycine"]


def _collapse_ratios(df) -> "pd.DataFrame":
    """
    Add per-peptide collapse-ratio columns to a copy of the collapse-test
    DataFrame:

      - `pre_mmff_kabsch_collapse` : `uniqueconfs_cremp / pre_mmff_kabsch_0125`
      - `pre_mmff_crest_collapse`  : `uniqueconfs_cremp / pre_mmff_crest_0125`
      - `post_mmff_kabsch_collapse`: `uniqueconfs_cremp / post_mmff_kabsch_0125`
      - `post_mmff_crest_collapse` : `uniqueconfs_cremp / post_mmff_crest_0125`

    Params:
        df: pd.DataFrame : output rows from `cremp_collapse_test.py run`
    Returns:
        pd.DataFrame : same rows with collapse-ratio columns appended
    """
    out = df.copy()
    out["pre_mmff_kabsch_collapse"] = out["uniqueconfs_cremp"] / out[
        "pre_mmff_kabsch_0125"
    ].clip(lower=1)
    out["pre_mmff_crest_collapse"] = out["uniqueconfs_cremp"] / out[
        "pre_mmff_crest_0125"
    ].clip(lower=1)
    out["post_mmff_kabsch_collapse"] = out["uniqueconfs_cremp"] / out[
        "post_mmff_kabsch_0125"
    ].clip(lower=1)
    out["post_mmff_crest_collapse"] = out["uniqueconfs_cremp"] / out[
        "post_mmff_crest_0125"
    ].clip(lower=1)
    return out


@cli.command("summarize")
@click.option(
    "--in_csv",
    required=True,
    type=Path,
    help="Collapse-test CSV (output of `cremp_collapse_test.py run`).",
)
@click.option(
    "--out_csv",
    required=True,
    type=Path,
    help="Output CSV path: one row per stratum with N + median/quartile "
    "collapse ratios. Suitable as input to `cremp_overlap_figure.py`.",
)
def summarize_cmd(in_csv: Path, out_csv: Path) -> None:
    """
    Read a completed collapse-test CSV and emit per-stratum summary
    statistics (N, median, IQR of the collapse ratios) for each cell of
    the (topology × has_proline × has_glycine) grid plus marginal
    breakdowns by `num_monomers`. The output is plot-ready: one row per
    stratum.

    Params:
        in_csv: Path : input collapse-test CSV
        out_csv: Path : per-stratum summary CSV
    Returns:
        None
    """
    import pandas as pd

    df = pd.read_csv(in_csv)
    missing = [c for c in _STRATIFICATION_KEYS if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"in_csv missing stratification columns {missing}; rerun the "
            "collapse test with a peptide_list_csv that carries them "
            "(e.g. the sampler's output)."
        )

    df = _collapse_ratios(df)
    ratio_cols = [
        "pre_mmff_kabsch_collapse",
        "pre_mmff_crest_collapse",
        "post_mmff_kabsch_collapse",
        "post_mmff_crest_collapse",
    ]
    n_kabsch_lt_10pct = df["post_mmff_kabsch_0125"] < df["uniqueconfs_cremp"] / 10

    rows: list[dict] = []
    # Per-cell rows: topology × has_proline × has_glycine.
    for keys, group in df.groupby(_STRATIFICATION_KEYS):
        row = {
            "stratum_kind": "cell",
            "topology": keys[0],
            "has_proline": keys[1],
            "has_glycine": keys[2],
            "num_monomers": "",
            "n_peptides": len(group),
            "frac_post_mmff_kabsch_lt_10pct": float(
                n_kabsch_lt_10pct.loc[group.index].mean()
            ),
        }
        for col in ratio_cols:
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_q25"] = float(group[col].quantile(0.25))
            row[f"{col}_q75"] = float(group[col].quantile(0.75))
        rows.append(row)

    # Marginal rows by num_monomers (post-hoc length axis).
    if "num_monomers" in df.columns and df["num_monomers"].notna().any():
        for nm, group in df.groupby("num_monomers"):
            row = {
                "stratum_kind": "num_monomers",
                "topology": "",
                "has_proline": "",
                "has_glycine": "",
                "num_monomers": int(nm),
                "n_peptides": len(group),
                "frac_post_mmff_kabsch_lt_10pct": float(
                    n_kabsch_lt_10pct.loc[group.index].mean()
                ),
            }
            for col in ratio_cols:
                row[f"{col}_median"] = float(group[col].median())
                row[f"{col}_q25"] = float(group[col].quantile(0.25))
                row[f"{col}_q75"] = float(group[col].quantile(0.75))
            rows.append(row)

    # Aggregate row across the whole sample.
    agg = {
        "stratum_kind": "all",
        "topology": "",
        "has_proline": "",
        "has_glycine": "",
        "num_monomers": "",
        "n_peptides": len(df),
        "frac_post_mmff_kabsch_lt_10pct": float(n_kabsch_lt_10pct.mean()),
    }
    for col in ratio_cols:
        agg[f"{col}_median"] = float(df[col].median())
        agg[f"{col}_q25"] = float(df[col].quantile(0.25))
        agg[f"{col}_q75"] = float(df[col].quantile(0.75))
    rows.append(agg)

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("wrote %d strata rows to %s", len(out_df), out_csv)


if __name__ == "__main__":
    cli()
