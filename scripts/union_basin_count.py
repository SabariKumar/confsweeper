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
from rdkit.Geometry import Point3D

# Match other scripts' import shape so this picks up the in-tree package.
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from confsweeper import _KT_EV_298K, _energy_ranked_dedup  # noqa: E402
from mcmm import _inertia_eigvals, _kabsch_rmsd_pairwise  # noqa: E402
from validation.cremp import calc_coverage  # noqa: E402

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


def _boltzmann_weights(energies_eV: np.ndarray, kT: float) -> np.ndarray:
    """
    Normalised Boltzmann weights at thermal energy `kT` (eV) over a set of
    basin energies. Mirrors the weight computation in
    `saturation_etkdg._bw_metrics`.

    Params:
        energies_eV: ndarray (K,) : basin energies in eV
        kT: float : thermal energy in eV
    Returns:
        ndarray (K,) : weights summing to 1.0 (empty array if input empty)
    """
    e = np.asarray(energies_eV, dtype=np.float64)
    if e.size == 0:
        return e
    w = np.exp(-(e - e.min()) / kT)
    return w / w.sum()


def _mol_from_coords(template: Chem.Mol, coords: torch.Tensor) -> Chem.Mol:
    """
    Build a mol from `template`'s topology / atom ordering carrying one
    conformer per row of `coords`. Used to wrap the sampler's union basin
    geometries (a coordinate tensor) in an RDKit mol so spyrmsd symmetric
    RMSD can compare them against the CREMP ceiling mol — which uses a
    different atom ordering, so raw index-aligned RMSD would be invalid.

    Params:
        template: Chem.Mol : provides topology + atom ordering (Hs included)
        coords: torch.Tensor (K, n_atoms, 3) : basin coordinates
    Returns:
        Chem.Mol : copy of `template` carrying K conformers (ids 0..K-1)
    """
    m = Chem.Mol(template)
    m.RemoveAllConformers()
    arr = coords.cpu().numpy()
    n_atoms = m.GetNumAtoms()
    for k in range(arr.shape[0]):
        conf = Chem.Conformer(n_atoms)
        for i in range(n_atoms):
            x, y, z = arr[k, i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
    return m


def _boltzmann_coverage(
    template: Chem.Mol,
    union_coords: torch.Tensor,
    union_energies: np.ndarray,
    heavy: list[int],
    ceiling_mol: Chem.Mol,
    ceiling_e: np.ndarray,
    kT: float,
    match_rmsd: float,
    basin_rmsd: float,
) -> dict:
    """
    Boltzmann-weighted coverage of the CREMP-rescored ceiling by the sampler's
    union basin set, plus joint-reference masses and a new-basin (discovery)
    metric. Physically: missing a low-energy (high-weight) ceiling basin costs
    more than missing a high-energy one.

    The sampler basin set is the Kabsch-deduped discovery set (geometrically
    distinct basins, no energy filter) — independent of the row's dedup_mode, so
    the coverage is a pure geometric+thermodynamic property. Matching uses
    spyrmsd symmetric, heavy-atom RMSD (`calc_coverage(..., strip=True)`) because
    the ceiling and sampler mols carry different atom orderings. Energies are
    MACE for both sides ⇒ same absolute scale ⇒ a joint Boltzmann distribution
    is valid.

    Params:
        template: Chem.Mol : sampler topology / atom ordering
        union_coords: torch.Tensor (N, n_atoms, 3) : raw union basin coords
        union_energies: ndarray (N,) : union basin MACE energies (eV)
        heavy: list[int] : heavy-atom indices on the sampler mol
        ceiling_mol: Chem.Mol : ceiling basins (CREMP-rescored, MACE-scored)
        ceiling_e: ndarray (M,) : ceiling MACE energies (eV)
        kT: float : thermal energy in eV for the Boltzmann weights
        match_rmsd: float : cross-method symmetric-RMSD tolerance (Å) used to
            decide whether a sampler basin matches a ceiling basin. Looser than
            `basin_rmsd` because MMFF (sampler) and GFN2-xTB (ceiling) relax the
            same conformational basin to geometries that typically differ by a
            few tenths of an Å.
        basin_rmsd: float : within-method Kabsch threshold (Å) used to dedup the
            sampler union into distinct basins (the 0.125 Å basin-identity
            convention). Fixed independent of `match_rmsd` so the sampler basin
            set is stable while the cross-method match tolerance varies.
    Returns:
        dict : the Boltzmann-coverage / joint / discovery columns
    """
    # Sampler basin set = Kabsch-deduped discovery set at the within-method
    # basin threshold (NOT match_rmsd), so the distinct-basin count is stable
    # while the cross-method match tolerance τ varies.
    s_idx = _energy_ranked_dedup(
        union_coords,
        union_energies,
        rmsd_threshold=basin_rmsd,
        heavy_atom_indices=heavy,
        dedup_mode="kabsch",
    )
    s_coords = union_coords[s_idx]
    s_e = union_energies[np.asarray(s_idx)]
    sampler_mol = _mol_from_coords(template, s_coords)
    sampler_conf_ids = [c.GetId() for c in sampler_mol.GetConformers()]
    ceiling_conf_ids = [c.GetId() for c in ceiling_mol.GetConformers()]

    # Basin sets are tiny (tens), and the ceiling (GFN2-xTB frame) and sampler
    # (MMFF frame) sit in arbitrary rotational frames. calc_coverage's tensor
    # pre-filter centers translation but does NOT align rotation, so a large
    # filter_factor effectively disables it — every pair is handed to the
    # rotation- and symmetry-minimizing spyrmsd, which is what actually decides.
    _NO_PREFILTER = 1.0e6

    # ceiling basin i covered by the sampler?  (heavy-atom symmetric RMSD)
    _, ceiling_min_rmsds = calc_coverage(
        ceiling_mol,
        sampler_mol,
        sampler_conf_ids,
        rmsd_cutoff=match_rmsd,
        filter_factor=_NO_PREFILTER,
        strip=True,
    )
    covered_C = np.array([r <= match_rmsd for r in ceiling_min_rmsds], dtype=bool)

    # sampler basin j absent from the ceiling?  → a discovery
    _, sampler_min_rmsds = calc_coverage(
        sampler_mol,
        ceiling_mol,
        ceiling_conf_ids,
        rmsd_cutoff=match_rmsd,
        filter_factor=_NO_PREFILTER,
        strip=True,
    )
    new_S = np.array([r > match_rmsd for r in sampler_min_rmsds], dtype=bool)

    # 1. Ceiling-only recovered Boltzmann mass (headline).
    p = _boltzmann_weights(ceiling_e, kT)
    coverage_bw_ceiling = float(p[covered_C].sum()) if covered_C.any() else 0.0
    coverage_count_matched = float(covered_C.mean()) if covered_C.size else 0.0
    max_missed_bw = float(p[~covered_C].max()) if (~covered_C).any() else 0.0

    # 2. Joint reference (ceiling ∪ new sampler basins).
    new_idx = np.where(new_S)[0]
    joint_e = np.concatenate([ceiling_e, s_e[new_idx]])
    q = _boltzmann_weights(joint_e, kT)
    n_c = len(ceiling_e)
    q_ceiling, q_new = q[:n_c], q[n_c:]
    ceiling_mass_joint = float(q_ceiling.sum())
    new_basin_mass_joint = float(q_new.sum())
    missed_ceiling_mass_joint = (
        float(q_ceiling[~covered_C].sum()) if (~covered_C).any() else 0.0
    )
    sampler_mass_joint = float(q_ceiling[covered_C].sum() + q_new.sum())

    # 3. New-basins / discovery.
    n_new_basins = int(new_S.sum())
    e_min_ceiling = float(ceiling_e.min())
    if n_new_basins > 0:
        e_min_new = float(s_e[new_idx].min())
        delta_emin = e_min_new - e_min_ceiling
        found_new_global_min = bool(e_min_new < e_min_ceiling - 1e-3)
    else:
        e_min_new = ""
        delta_emin = ""
        found_new_global_min = False

    return {
        "n_ceiling_basins": n_c,
        "n_sampler_basins": len(s_idx),
        "coverage_bw_ceiling": coverage_bw_ceiling,
        "coverage_count_matched": coverage_count_matched,
        "max_missed_bw": max_missed_bw,
        "sampler_mass_joint": sampler_mass_joint,
        "ceiling_mass_joint": ceiling_mass_joint,
        "missed_ceiling_mass_joint": missed_ceiling_mass_joint,
        "n_new_basins": n_new_basins,
        "new_basin_mass_joint": new_basin_mass_joint,
        "e_min_new_eV": e_min_new,
        "delta_emin_vs_ceiling": delta_emin,
        "found_new_global_min": found_new_global_min,
    }


# Boltzmann-coverage columns (blank for peptides without a CREMP ceiling).
_BW_COVERAGE_COLUMNS = [
    "n_ceiling_basins",
    "n_sampler_basins",
    "coverage_bw_ceiling",
    "coverage_count_matched",
    "max_missed_bw",
    "sampler_mass_joint",
    "ceiling_mass_joint",
    "missed_ceiling_mass_joint",
    "n_new_basins",
    "new_basin_mass_joint",
    "e_min_new_eV",
    "delta_emin_vs_ceiling",
    "found_new_global_min",
]


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
    ceiling_mol: Chem.Mol | None = None,
    ceiling_e: np.ndarray | None = None,
    coverage_kT: float = _KT_EV_298K,
    match_rmsd: float = 0.125,
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
        ceiling_mol: Chem.Mol | None : ceiling basin mol (from the
            `--dump_ceiling_sdf_dir` SDF); enables the Boltzmann-coverage
            columns. None for peptides with no CREMP ceiling.
        ceiling_e: ndarray | None : ceiling MACE energies (eV)
        coverage_kT: float : thermal energy (eV) for the Boltzmann weights
        match_rmsd: float : heavy-atom symmetric-RMSD match threshold (Å)
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

    # Coverage vs CREMP-rescored ceiling (count ratio — kept for continuity).
    if cremp_row is not None:
        cremp_ceiling_kabsch = int(cremp_row["post_mmff_kabsch_0125"])
        cremp_ceiling_crest = int(cremp_row.get("post_mmff_crest_0125", float("nan")))
    else:
        cremp_ceiling_kabsch = None
        cremp_ceiling_crest = None

    # Boltzmann-weighted coverage (geometric matching against the dumped
    # ceiling basins). dedup-mode-independent: identical on the kabsch and
    # crest rows. Blank columns when no ceiling SDF was provided.
    if ceiling_mol is not None and ceiling_e is not None and len(ceiling_e) > 0:
        bw_cols = _boltzmann_coverage(
            template,
            union_coords,
            union_energies,
            heavy,
            ceiling_mol,
            ceiling_e,
            coverage_kT,
            match_rmsd,
            basin_rmsd=rmsd_threshold,
        )
    else:
        bw_cols = {c: "" for c in _BW_COVERAGE_COLUMNS}

    row = {
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
    row.update(bw_cols)
    return row


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
] + _BW_COVERAGE_COLUMNS


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
    "--ceiling_sdf_dir",
    type=Path,
    default=None,
    help="Directory of <sequence>.sdf ceiling basins dumped by "
    "`cremp_collapse_test.py run --dump_ceiling_sdf_dir`. When provided, the "
    "Boltzmann-weighted coverage columns (coverage_bw_ceiling, joint masses, "
    "new-basin discovery) are computed for any peptide whose CREMP sequence "
    "matches the SDF filename.",
)
@click.option(
    "--coverage_kT",
    "coverage_kT",
    type=float,
    default=_KT_EV_298K,
    show_default=True,
    help="Thermal energy (eV) for the Boltzmann coverage weights (default 298 K).",
)
@click.option(
    "--match_rmsd",
    type=float,
    default=0.5,
    show_default=True,
    help="Heavy-atom symmetric-RMSD tolerance (Å) for matching sampler basins "
    "to ceiling basins in the Boltzmann coverage analysis. Note this is "
    "*looser* than the 0.125 Å within-method dedup threshold (`--rmsd_threshold`) "
    "because MMFF (sampler) and GFN2-xTB (ceiling) relax the same basin to "
    "geometries that typically differ by a few tenths of an Å. 0.5 Å is the "
    "convention used by `src/validation/cremp_coverage.py`.",
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
    ceiling_sdf_dir: Path | None,
    coverage_kT: float,
    match_rmsd: float,
    out_csv: Path,
) -> None:
    """
    Compute the post-hoc union of basin sets from a DBT-only run and a
    DBT+Cart run, with optional count-ratio coverage against the
    CREMP-rescored ceiling and (when `--ceiling_sdf_dir` is given)
    Boltzmann-weighted coverage of the ceiling distribution.

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
        ceiling_sdf_dir: Path | None : ceiling-basin SDF directory for
            Boltzmann coverage; None disables those columns
        coverage_kT: float : thermal energy (eV) for the coverage weights
        match_rmsd: float : heavy-atom symmetric-RMSD match threshold (Å)
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
        seq = None
        if cremp_df is not None:
            seq = _match_cremp_sequence(peptide_id, cremp_df)
            if seq is not None:
                cremp_row = cremp_df.loc[cremp_df["sequence"] == seq].iloc[0]

        # Load the ceiling basins for the Boltzmann-coverage analysis, if a
        # dump dir was given and this peptide maps to a CREMP sequence.
        ceiling_mol, ceiling_e = None, None
        if ceiling_sdf_dir is not None and seq is not None:
            ceiling_mol, ceiling_e = _load_basin_sdf(ceiling_sdf_dir / f"{seq}.sdf")
            if ceiling_mol is None:
                logger.warning(
                    "no ceiling SDF for %s at %s; Boltzmann coverage skipped",
                    seq,
                    ceiling_sdf_dir / f"{seq}.sdf",
                )

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
                ceiling_mol=ceiling_mol,
                ceiling_e=ceiling_e,
                coverage_kT=coverage_kT,
                match_rmsd=match_rmsd,
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
