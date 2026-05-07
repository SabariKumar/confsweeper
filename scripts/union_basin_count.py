"""
Post-hoc union analysis of basin SDFs from two sampler runs.

Per peptide, combines the dumped basin centroids from a DBT-only run
and a DBT+Cart run, applies our standard energy filter + Kabsch dedup
on the merged set, and reports:

  - per-method basin counts (already in each run's CSV; surfaced here
    for one-table comparison)
  - union basin counts after re-dedup at the configurable threshold,
    both with the standard 5 kT energy filter and *without*
    (the latter being the "discovery diversity" metric — independent
    of where the global e_min lands and so unaffected by Cart's
    deeper-minimum-pulls-the-window-down regression)
  - per-method "unique" / "overlap" splits: how many union basins were
    only found by one method, only by the other, or by both
  - coverage % vs the CREMP-rescored ceiling from
    `cremp_collapse_test.py` (Step 16) when a sequence match exists

Energies come from the SDF's `MACE_ENERGY` per-conformer property —
no GPU needed. Heavy atoms inferred from the SMILES on the first
loaded mol per peptide.

Usage:
    pixi run python scripts/union_basin_count.py \\
        --dbt_sdf_dir       results/mcmm_basin_sdfs_kabsch_0125 \\
        --cart_sdf_dir      results/mcmm_basin_sdfs_with_cartesian \\
        --cremp_collapse_csv results/cremp_collapse_test_dual.csv \\
        --out_csv           results/union_basin_coverage.csv
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

# Match other scripts' import shape so this picks up the in-tree package.
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from confsweeper import _KT_EV_298K, _energy_ranked_dedup  # noqa: E402
from mcmm import _inertia_eigvals, _kabsch_rmsd_pairwise  # noqa: E402

# Default thresholds for crest-mode union analysis. The 0.05 eV energy
# threshold matches what `_minimize_score_filter_dedup` uses post-MMFF
# (set to absorb the MACE float32 ~0.01–0.05 eV noise floor) and what
# `cremp_collapse_test.py` uses for its post-MMFF crest count — so the
# coverage-vs-CREMP-ceiling comparison is at the same metric.
DEFAULT_CREST_ENERGY_THRESHOLD_EV = 0.05
DEFAULT_CREST_ROTCONST_THRESHOLD = 0.01

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def _load_basin_sdf(sdf_path: Path) -> tuple[Chem.Mol | None, np.ndarray]:
    """
    Load all conformers from a basin-dump SDF and the per-conformer
    `MACE_ENERGY` (eV).

    Params:
        sdf_path: Path : path to one of `_maybe_dump_sdf`'s outputs
    Returns:
        tuple[Chem.Mol | None, ndarray] : (mol with all conformers
            attached, shape (n,) MACE energies). `None` mol when the
            file is missing or has no conformers.
    """
    if not sdf_path.exists():
        return None, np.array([], dtype=np.float64)
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    confs = [m for m in suppl if m is not None]
    if not confs:
        return None, np.array([], dtype=np.float64)
    template = Chem.Mol(confs[0])
    template.RemoveAllConformers()
    for c in confs:
        template.AddConformer(c.GetConformer(), assignId=True)
    energies = []
    for c in confs:
        if c.HasProp("MACE_ENERGY"):
            energies.append(float(c.GetProp("MACE_ENERGY")))
        else:
            energies.append(float("nan"))
    return template, np.array(energies, dtype=np.float64)


def _heavy_indices(mol: Chem.Mol) -> list[int]:
    """Return atom indices of all non-hydrogen atoms in `mol`."""
    return [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() != 1]


def _peptide_id_from_sdf(sdf_path: Path) -> str:
    """
    Strip the `_mcmm.sdf` suffix from the filename to recover the
    peptide_id used by `sampler_benchmark.py`.

    Params:
        sdf_path: Path : SDF file path
    Returns:
        str : peptide_id (e.g. 'cremp_typical_t.I.G.N')
    """
    return sdf_path.stem.removesuffix("_mcmm")


def _match_cremp_sequence(peptide_id: str, cremp_df: pd.DataFrame) -> str | None:
    """
    Find the CREMP sequence whose name appears in `peptide_id`. Returns
    None if no match (typical for PAMPA peptides).

    Params:
        peptide_id: str : SDF-derived peptide id
        cremp_df: DataFrame : rows from `cremp_collapse_test.py`'s output
    Returns:
        str | None : matched sequence value or None
    """
    for seq in cremp_df["sequence"]:
        if str(seq) in peptide_id:
            return str(seq)
    return None


def _classify_per_method(
    union_coords: torch.Tensor,
    union_energies: np.ndarray,
    union_methods: np.ndarray,
    centroid_indices: list[int],
    heavy: list[int],
    rmsd_threshold: float,
    *,
    dedup_mode: str = "kabsch",
    energy_threshold_eV: float = DEFAULT_CREST_ENERGY_THRESHOLD_EV,
    rotconst_anisotropy_threshold: float = DEFAULT_CREST_ROTCONST_THRESHOLD,
    masses: torch.Tensor | None = None,
) -> dict:
    """
    For each centroid in the union, decide whether DBT-only conformers,
    Cart-only conformers, or both contributed to it.

    Two centroids are "supported by method M" iff at least one method-M
    conformer in the union satisfies the active dedup criterion against
    the centroid:

      - `dedup_mode='kabsch'`: Kabsch RMSD < `rmsd_threshold`
      - `dedup_mode='crest'`: that AND |ΔE| < `energy_threshold_eV` AND
        rotational-constant max relative diff < `rotconst_anisotropy_threshold`
        (the same AND-test `_energy_ranked_dedup` applies under crest).

    Params:
        union_coords: torch.Tensor : (N, n_atoms, 3) all coords
        union_energies: ndarray (N,) : per-conformer energies in eV
        union_methods: ndarray (N,) of str : 'dbt' or 'cart' per coord
        centroid_indices: list[int] : indices into the union picked by
            `_energy_ranked_dedup`
        heavy: list[int] : heavy-atom indices for the metric
        rmsd_threshold: float : Kabsch RMSD threshold in Å
        dedup_mode: str : 'kabsch' (default) or 'crest'
        energy_threshold_eV: float : crest-mode energy criterion in eV
        rotconst_anisotropy_threshold: float : crest-mode rotational-
            anisotropy criterion (relative)
        masses: torch.Tensor | None : (n_atoms,) atomic masses in Da;
            required when `dedup_mode='crest'`
    Returns:
        dict with keys 'n_dbt_only', 'n_cart_only', 'n_overlap'
    """
    if len(centroid_indices) == 0:
        return {"n_dbt_only": 0, "n_cart_only": 0, "n_overlap": 0}
    heavy_idx = torch.tensor(heavy, dtype=torch.int64)
    union_heavy = union_coords.index_select(dim=1, index=heavy_idx)
    centroid_heavy = union_heavy[centroid_indices]  # (K, n_heavy, 3)
    distances = _kabsch_rmsd_pairwise(centroid_heavy, union_heavy)  # (K, N)
    close = distances < rmsd_threshold  # (K, N)

    if dedup_mode == "crest":
        if masses is None:
            raise ValueError("masses required for dedup_mode='crest' classification")
        # Energy AND-test
        e_t = torch.as_tensor(union_energies, dtype=torch.float64)
        de = (e_t.unsqueeze(0) - e_t[centroid_indices].unsqueeze(1)).abs()  # (K, N)
        de_close = de < energy_threshold_eV
        # Rotational-constant AND-test (max relative diff between eigvals).
        all_rotconsts = _inertia_eigvals(union_coords, masses)  # (N, 3)
        centroid_rotconsts = all_rotconsts[centroid_indices]  # (K, 3)
        diff = (
            all_rotconsts.unsqueeze(0) - centroid_rotconsts.unsqueeze(1)
        ).abs()  # (K, N, 3)
        denom = torch.maximum(
            all_rotconsts.unsqueeze(0).abs(),
            centroid_rotconsts.unsqueeze(1).abs(),
        ).clamp(min=1e-12)
        rot_diff = (diff / denom).max(dim=-1).values  # (K, N)
        rot_close = rot_diff < rotconst_anisotropy_threshold
        close = close & de_close & rot_close

    is_dbt = torch.tensor(union_methods == "dbt")
    is_cart = torch.tensor(union_methods == "cart")
    has_dbt = (close & is_dbt.unsqueeze(0)).any(dim=1)  # (K,)
    has_cart = (close & is_cart.unsqueeze(0)).any(dim=1)  # (K,)
    n_overlap = int((has_dbt & has_cart).sum().item())
    n_dbt_only = int((has_dbt & ~has_cart).sum().item())
    n_cart_only = int((has_cart & ~has_dbt).sum().item())
    return {
        "n_dbt_only": n_dbt_only,
        "n_cart_only": n_cart_only,
        "n_overlap": n_overlap,
    }


def _process_peptide(
    peptide_id: str,
    dbt_sdf: Path,
    cart_sdf: Path,
    rmsd_threshold: float,
    e_window_kT: float,
    cremp_row: pd.Series | None,
    dedup_mode: str,
    energy_threshold_eV: float,
    rotconst_anisotropy_threshold: float,
) -> dict | None:
    """
    Compute the union analysis for one peptide under one dedup mode.
    Returns None when neither SDF is loadable.

    Params:
        peptide_id: str : SDF-derived peptide id
        dbt_sdf: Path : DBT-only SDF path
        cart_sdf: Path : DBT+Cart SDF path
        rmsd_threshold: float : Kabsch RMSD threshold in Å
        e_window_kT: float : energy filter window in kT_298K
        cremp_row: Series | None : matching CREMP-collapse row, or
            None if peptide isn't in CREMP
        dedup_mode: str : 'kabsch' or 'crest'
        energy_threshold_eV: float : crest-mode energy criterion in eV
        rotconst_anisotropy_threshold: float : crest-mode rotational-
            anisotropy criterion
    Returns:
        dict : one row of the output CSV
    """
    dbt_mol, dbt_e = _load_basin_sdf(dbt_sdf)
    cart_mol, cart_e = _load_basin_sdf(cart_sdf)
    if dbt_mol is None and cart_mol is None:
        logger.warning("no SDFs loadable for %s, skipping", peptide_id)
        return None
    template = dbt_mol or cart_mol
    heavy = _heavy_indices(template)
    atomic_nums = [a.GetAtomicNum() for a in template.GetAtoms()]
    if dedup_mode == "crest":
        from rdkit.Chem import GetPeriodicTable

        pt = GetPeriodicTable()
        masses = torch.tensor(
            [pt.GetAtomicWeight(int(z)) for z in atomic_nums],
            dtype=torch.float64,
        )
    else:
        masses = None

    dbt_n = int(dbt_mol.GetNumConformers()) if dbt_mol is not None else 0
    cart_n = int(cart_mol.GetNumConformers()) if cart_mol is not None else 0

    # Build the union conformer set.
    union_coords_list: list[torch.Tensor] = []
    union_energies_list: list[float] = []
    union_methods_list: list[str] = []
    if dbt_mol is not None:
        for c, e in zip(dbt_mol.GetConformers(), dbt_e):
            union_coords_list.append(
                torch.tensor(c.GetPositions(), dtype=torch.float64)
            )
            union_energies_list.append(float(e))
            union_methods_list.append("dbt")
    if cart_mol is not None:
        for c, e in zip(cart_mol.GetConformers(), cart_e):
            union_coords_list.append(
                torch.tensor(c.GetPositions(), dtype=torch.float64)
            )
            union_energies_list.append(float(e))
            union_methods_list.append("cart")

    union_coords = torch.stack(union_coords_list)  # (N, n_atoms, 3)
    union_energies = np.array(union_energies_list, dtype=np.float64)
    union_methods = np.array(union_methods_list)
    n_total = len(union_methods)

    dedup_kwargs = dict(
        rmsd_threshold=rmsd_threshold,
        heavy_atom_indices=heavy,
        dedup_mode=dedup_mode,
        energy_threshold_eV=energy_threshold_eV,
        rotconst_anisotropy_threshold=rotconst_anisotropy_threshold,
        atomic_numbers=atomic_nums if dedup_mode == "crest" else None,
    )
    classify_kwargs = dict(
        dedup_mode=dedup_mode,
        energy_threshold_eV=energy_threshold_eV,
        rotconst_anisotropy_threshold=rotconst_anisotropy_threshold,
        masses=masses,
    )

    # Discovery-diversity: dedup the full union with no energy filter.
    discovery_indices = _energy_ranked_dedup(
        union_coords, union_energies, **dedup_kwargs
    )
    n_union_all = len(discovery_indices)

    # Filtered union: energy-filter relative to union's e_min, then dedup.
    e_min = float(union_energies.min())
    keep_mask = (union_energies - e_min) <= e_window_kT * _KT_EV_298K
    if not keep_mask.any():
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[int(np.argmin(union_energies))] = True
    kept_pool_idx = np.where(keep_mask)[0]
    kept_coords = union_coords[kept_pool_idx]
    kept_energies = union_energies[kept_pool_idx]
    if len(kept_coords) == 1:
        filtered_centroid_pool_idx = [0]
    else:
        filtered_centroid_pool_idx = _energy_ranked_dedup(
            kept_coords, kept_energies, **dedup_kwargs
        )
    filtered_centroid_union_idx = [
        int(kept_pool_idx[i]) for i in filtered_centroid_pool_idx
    ]
    n_union_filtered = len(filtered_centroid_union_idx)

    filtered_split = _classify_per_method(
        union_coords,
        union_energies,
        union_methods,
        filtered_centroid_union_idx,
        heavy,
        rmsd_threshold,
        **classify_kwargs,
    )
    discovery_split = _classify_per_method(
        union_coords,
        union_energies,
        union_methods,
        discovery_indices,
        heavy,
        rmsd_threshold,
        **classify_kwargs,
    )

    # Coverage vs CREMP-rescored ceiling.
    if cremp_row is not None:
        cremp_ceiling_kabsch = int(cremp_row["post_mmff_kabsch_0125"])
        cremp_ceiling_crest = int(cremp_row.get("post_mmff_crest_0125", float("nan")))
    else:
        cremp_ceiling_kabsch = None
        cremp_ceiling_crest = None

    return {
        "peptide_id": peptide_id,
        "dedup_mode": dedup_mode,
        "n_dbt": dbt_n,
        "n_cart": cart_n,
        "n_total_union_pool": n_total,
        "n_union_filtered_5kT": n_union_filtered,
        "n_union_all": n_union_all,
        "n_dbt_only_filtered": filtered_split["n_dbt_only"],
        "n_cart_only_filtered": filtered_split["n_cart_only"],
        "n_overlap_filtered": filtered_split["n_overlap"],
        "n_dbt_only_all": discovery_split["n_dbt_only"],
        "n_cart_only_all": discovery_split["n_cart_only"],
        "n_overlap_all": discovery_split["n_overlap"],
        "e_min_union_eV": e_min,
        "cremp_ceiling_kabsch_0125": (
            cremp_ceiling_kabsch if cremp_ceiling_kabsch is not None else ""
        ),
        "cremp_ceiling_crest_0125": (
            cremp_ceiling_crest if cremp_ceiling_crest is not None else ""
        ),
        "coverage_union_filtered_vs_kabsch": (
            n_union_filtered / cremp_ceiling_kabsch if cremp_ceiling_kabsch else ""
        ),
        "coverage_union_filtered_vs_crest": (
            n_union_filtered / cremp_ceiling_crest if cremp_ceiling_crest else ""
        ),
        "coverage_union_all_vs_kabsch": (
            n_union_all / cremp_ceiling_kabsch if cremp_ceiling_kabsch else ""
        ),
        "coverage_union_all_vs_crest": (
            n_union_all / cremp_ceiling_crest if cremp_ceiling_crest else ""
        ),
    }


OUTPUT_COLUMNS = [
    "peptide_id",
    "dedup_mode",
    "n_dbt",
    "n_cart",
    "n_total_union_pool",
    "n_union_filtered_5kT",
    "n_union_all",
    "n_dbt_only_filtered",
    "n_cart_only_filtered",
    "n_overlap_filtered",
    "n_dbt_only_all",
    "n_cart_only_all",
    "n_overlap_all",
    "e_min_union_eV",
    "cremp_ceiling_kabsch_0125",
    "cremp_ceiling_crest_0125",
    "coverage_union_filtered_vs_kabsch",
    "coverage_union_filtered_vs_crest",
    "coverage_union_all_vs_kabsch",
    "coverage_union_all_vs_crest",
]


@click.command()
@click.option(
    "--dbt_sdf_dir",
    required=True,
    type=Path,
    help="Directory of <peptide_id>_mcmm.sdf files dumped by the DBT-only "
    "sampler run.",
)
@click.option(
    "--cart_sdf_dir",
    required=True,
    type=Path,
    help="Directory of <peptide_id>_mcmm.sdf files dumped by the DBT+Cart run.",
)
@click.option(
    "--cremp_collapse_csv",
    type=Path,
    default=None,
    help="Optional output of `cremp_collapse_test.py`. When provided, "
    "coverage % is computed against the post-MMFF Kabsch and CREST "
    "ceilings for any peptide whose CREMP sequence matches the SDF "
    "filename.",
)
@click.option(
    "--rmsd_threshold",
    type=float,
    default=0.125,
    show_default=True,
    help="Kabsch heavy-atom RMSD threshold for the union dedup.",
)
@click.option(
    "--e_window_kT",
    "e_window_kT",
    type=float,
    default=5.0,
    show_default=True,
    help="Energy filter window (kT_298K) for the filtered union count.",
)
@click.option(
    "--dedup_mode",
    type=click.Choice(["kabsch", "crest", "both"]),
    default="kabsch",
    show_default=True,
    help="Basin-dedup criterion. 'kabsch' (default), 'crest' (CREMP-"
    "comparable AND-test), or 'both' (one row per peptide per mode).",
)
@click.option(
    "--energy_threshold_eV",
    "energy_threshold_eV",
    type=float,
    default=DEFAULT_CREST_ENERGY_THRESHOLD_EV,
    show_default=True,
    help="Crest-mode energy criterion in eV. Default 0.05 eV matches "
    "what `cremp_collapse_test.py`'s post-MMFF stage uses, so the "
    "coverage column is at the same metric as the CREMP ceiling.",
)
@click.option(
    "--rotconst_anisotropy_threshold",
    type=float,
    default=DEFAULT_CREST_ROTCONST_THRESHOLD,
    show_default=True,
    help="Crest-mode rotational-constant anisotropy threshold "
    "(relative). Default 0.01 matches CREST's middle of the "
    "documented 1–2.5% range.",
)
@click.option(
    "--out_csv",
    required=True,
    type=Path,
    help="Output CSV path",
)
def main(
    dbt_sdf_dir: Path,
    cart_sdf_dir: Path,
    cremp_collapse_csv: Path | None,
    rmsd_threshold: float,
    e_window_kT: float,
    dedup_mode: str,
    energy_threshold_eV: float,
    rotconst_anisotropy_threshold: float,
    out_csv: Path,
) -> None:
    """
    Compute the post-hoc union of basin sets from a DBT-only run and a
    DBT+Cart run, with optional coverage % against the CREMP-rescored
    ceiling.

    Params:
        dbt_sdf_dir: Path : DBT-only SDF directory
        cart_sdf_dir: Path : DBT+Cart SDF directory
        cremp_collapse_csv: Path | None : optional CREMP-rescore CSV
        rmsd_threshold: float : Kabsch threshold in Å
        e_window_kT: float : energy filter window
        dedup_mode: str : 'kabsch' (default), 'crest', or 'both'
        energy_threshold_eV: float : crest-mode energy threshold
        rotconst_anisotropy_threshold: float : crest-mode rotation
            anisotropy threshold
        out_csv: Path : output CSV path
    Returns:
        None
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cremp_df = (
        pd.read_csv(cremp_collapse_csv) if cremp_collapse_csv is not None else None
    )
    mode_list = ["kabsch", "crest"] if dedup_mode == "both" else [dedup_mode]

    sdfs = sorted(dbt_sdf_dir.glob("*_mcmm.sdf"))
    rows: list[dict] = []
    for dbt_sdf in sdfs:
        peptide_id = _peptide_id_from_sdf(dbt_sdf)
        cart_sdf = cart_sdf_dir / dbt_sdf.name
        cremp_row = None
        if cremp_df is not None:
            seq = _match_cremp_sequence(peptide_id, cremp_df)
            if seq is not None:
                cremp_row = cremp_df.loc[cremp_df["sequence"] == seq].iloc[0]

        for mode in mode_list:
            logger.info("processing %s (dedup=%s)", peptide_id, mode)
            row = _process_peptide(
                peptide_id,
                dbt_sdf,
                cart_sdf,
                rmsd_threshold,
                e_window_kT,
                cremp_row,
                dedup_mode=mode,
                energy_threshold_eV=energy_threshold_eV,
                rotconst_anisotropy_threshold=rotconst_anisotropy_threshold,
            )
            if row is None:
                continue
            rows.append(row)
            logger.info(
                "  %s [%s]: dbt=%d  cart=%d  | union_all=%d "
                "(dbt_only=%d cart_only=%d overlap=%d) | union_filtered=%d "
                "(dbt_only=%d cart_only=%d overlap=%d)",
                peptide_id,
                mode,
                row["n_dbt"],
                row["n_cart"],
                row["n_union_all"],
                row["n_dbt_only_all"],
                row["n_cart_only_all"],
                row["n_overlap_all"],
                row["n_union_filtered_5kT"],
                row["n_dbt_only_filtered"],
                row["n_cart_only_filtered"],
                row["n_overlap_filtered"],
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("wrote %d rows to %s", len(rows), out_csv)


if __name__ == "__main__":
    main()
