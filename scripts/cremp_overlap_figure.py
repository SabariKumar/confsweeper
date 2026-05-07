"""
3-panel paper figure for the CREMP overlap-statistics benchmark
(Step 19 of docs/mcmm_plan.md).

Reads a collapse-test CSV (output of `cremp_collapse_test.py run`,
carrying `topology`, `has_proline`, `has_glycine`, `num_monomers`
feature columns the sampler emitted) and renders:

  - Panel A: pre-MMFF Kabsch collapse ratio (`uniqueconfs / pre_mmff_kabsch_0125`)
    by topology. Tests whether NMe-rich peptides have more CREST overcounting
    than canonical L.
  - Panel B: post-MMFF Kabsch collapse ratio by Pro/Gly bucket
    (`neither` / `Pro-only` / `Gly-only` / `both`). Tests whether the
    MMFF-vs-xtb relaxer mismatch concentrates where Pro's ring-pucker
    or Gly's permissive φ-ψ matters most.
  - Panel C: heatmap of post-MMFF Kabsch median collapse ratio across the
    full 16-cell (topology × has_proline × has_glycine) grid, with
    cell counts overlaid.

Usage:
    pixi run python scripts/cremp_overlap_figure.py \\
        --collapse_csv  results/cremp_overlap_collapse.csv \\
        --out_svg       results/cremp_overlap_figure.svg \\
        [--out_pdf      results/cremp_overlap_figure.pdf] \\
        [--out_png      results/cremp_overlap_figure.png]

SVG is the default and required output (vector, editable in Illustrator
/ Inkscape, lossless for the paper). PDF and PNG are optional secondary
outputs for direct paste into LaTeX / slides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


_TOPOLOGY_ORDER = ["all-L", "D-only", "NMe-only", "D+NMe"]
_PROGLY_BUCKETS = [
    ("neither", lambda r: not r["has_proline"] and not r["has_glycine"]),
    ("Pro-only", lambda r: r["has_proline"] and not r["has_glycine"]),
    ("Gly-only", lambda r: not r["has_proline"] and r["has_glycine"]),
    ("both", lambda r: r["has_proline"] and r["has_glycine"]),
]


def _add_collapse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach per-peptide collapse-ratio columns. Mirrors the helper in
    `cremp_collapse_test.summarize_cmd` so this script is self-contained
    and can run on a raw collapse CSV without going through summarize first.

    Params:
        df: pd.DataFrame : per-peptide collapse-test rows
    Returns:
        pd.DataFrame : same rows with `pre_mmff_kabsch_collapse` and
            `post_mmff_kabsch_collapse` columns appended
    """
    out = df.copy()
    out["pre_mmff_kabsch_collapse"] = out["uniqueconfs_cremp"] / out[
        "pre_mmff_kabsch_0125"
    ].clip(lower=1)
    out["post_mmff_kabsch_collapse"] = out["uniqueconfs_cremp"] / out[
        "post_mmff_kabsch_0125"
    ].clip(lower=1)
    return out


def _bucket_label(row: pd.Series) -> str:
    """
    Pro/Gly bucket label for one peptide row. Returns 'neither',
    'Pro-only', 'Gly-only', or 'both' depending on the boolean flags.

    Params:
        row: pd.Series : one row of the collapse-test DataFrame
    Returns:
        str : bucket label (one of `_PROGLY_BUCKETS`)
    """
    for label, pred in _PROGLY_BUCKETS:
        if pred(row):
            return label
    raise ValueError(f"could not bucket row: {row!r}")


def _panel_a(ax, df: pd.DataFrame) -> None:
    """
    Panel A: pre-MMFF Kabsch collapse ratio boxplot, grouped by topology.

    Params:
        ax: matplotlib Axes
        df: pd.DataFrame : with `topology` and `pre_mmff_kabsch_collapse`
    Returns:
        None
    """
    data = [
        df.loc[df["topology"] == t, "pre_mmff_kabsch_collapse"].dropna().values
        for t in _TOPOLOGY_ORDER
    ]
    counts = [len(d) for d in data]
    bp = ax.boxplot(
        data,
        tick_labels=[f"{t}\n(N={n})" for t, n in zip(_TOPOLOGY_ORDER, counts)],
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfd8dc")
        patch.set_edgecolor("#37474f")
    ax.axhline(1.0, ls="--", color="grey", lw=0.8, alpha=0.6)
    ax.set_ylabel("uniqueconfs / pre-MMFF Kabsch n_basins")
    ax.set_title("A. Pre-MMFF Kabsch collapse by topology")
    ax.tick_params(axis="x", labelsize=9)


def _panel_b(ax, df: pd.DataFrame) -> None:
    """
    Panel B: post-MMFF Kabsch collapse ratio boxplot, grouped by
    Pro/Gly bucket.

    Params:
        ax: matplotlib Axes
        df: pd.DataFrame : with `has_proline`, `has_glycine`,
            `post_mmff_kabsch_collapse`
    Returns:
        None
    """
    df = df.copy()
    df["progly_bucket"] = df.apply(_bucket_label, axis=1)
    bucket_names = [b[0] for b in _PROGLY_BUCKETS]
    data = [
        df.loc[df["progly_bucket"] == b, "post_mmff_kabsch_collapse"].dropna().values
        for b in bucket_names
    ]
    counts = [len(d) for d in data]
    bp = ax.boxplot(
        data,
        tick_labels=[f"{b}\n(N={n})" for b, n in zip(bucket_names, counts)],
        showfliers=False,
        patch_artist=True,
    )
    for patch, n in zip(bp["boxes"], counts):
        # Highlight cells where the relaxer-mismatch hypothesis predicts
        # the largest collapse (Pro and Gly).
        patch.set_facecolor("#ffcdd2" if n > 0 else "#eeeeee")
        patch.set_edgecolor("#b71c1c")
    ax.axhline(1.0, ls="--", color="grey", lw=0.8, alpha=0.6)
    ax.set_ylabel("uniqueconfs / post-MMFF Kabsch n_basins")
    ax.set_title("B. Post-MMFF Kabsch collapse by Pro/Gly")
    ax.tick_params(axis="x", labelsize=9)


def _panel_c(ax, df: pd.DataFrame) -> None:
    """
    Panel C: heatmap of post-MMFF Kabsch median collapse ratio across
    the (topology × has_proline × has_glycine) grid. Cell counts are
    overlaid in each cell. The 4 (has_proline, has_glycine)
    combinations are stacked vertically to keep the figure square.

    Params:
        ax: matplotlib Axes
        df: pd.DataFrame : with stratification columns and
            `post_mmff_kabsch_collapse`
    Returns:
        None
    """
    bucket_names = [b[0] for b in _PROGLY_BUCKETS]
    medians = np.full((len(bucket_names), len(_TOPOLOGY_ORDER)), np.nan)
    counts = np.zeros((len(bucket_names), len(_TOPOLOGY_ORDER)), dtype=int)

    for i, (bucket, pred) in enumerate(_PROGLY_BUCKETS):
        bucket_mask = df.apply(pred, axis=1)
        for j, topo in enumerate(_TOPOLOGY_ORDER):
            cell_mask = bucket_mask & (df["topology"] == topo)
            cell_vals = df.loc[cell_mask, "post_mmff_kabsch_collapse"].dropna()
            counts[i, j] = len(cell_vals)
            if len(cell_vals) > 0:
                medians[i, j] = float(cell_vals.median())

    finite_max = float(np.nanmax(medians)) if np.isfinite(medians).any() else 1.0
    im = ax.imshow(
        medians,
        cmap="OrRd",
        aspect="auto",
        vmin=1.0,
        vmax=max(finite_max, 1.0),
    )
    ax.set_xticks(range(len(_TOPOLOGY_ORDER)))
    ax.set_xticklabels(_TOPOLOGY_ORDER, rotation=20)
    ax.set_yticks(range(len(bucket_names)))
    ax.set_yticklabels(bucket_names)
    ax.set_title("C. Post-MMFF Kabsch median collapse  (cell N below)")

    # Overlay median + N in each cell.
    for i in range(len(bucket_names)):
        for j in range(len(_TOPOLOGY_ORDER)):
            if counts[i, j] == 0:
                ax.text(j, i, "n/a", ha="center", va="center", color="grey", fontsize=9)
                continue
            txt = f"{medians[i, j]:.1f}\nN={counts[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("median collapse ratio")


@click.command()
@click.option(
    "--collapse_csv",
    required=True,
    type=Path,
    help="Per-peptide collapse-test CSV from `cremp_collapse_test.py run`.",
)
@click.option(
    "--out_svg",
    required=True,
    type=Path,
    help="Output SVG path (required default — vector, editable, lossless).",
)
@click.option(
    "--out_pdf",
    type=Path,
    default=None,
    help="Optional PDF output path (in addition to SVG).",
)
@click.option(
    "--out_png",
    type=Path,
    default=None,
    help="Optional PNG output path (in addition to SVG).",
)
def main(
    collapse_csv: Path,
    out_svg: Path,
    out_pdf: Path | None,
    out_png: Path | None,
) -> None:
    """
    Render the 3-panel CREMP overlap figure from a collapse-test CSV.
    SVG is the primary, required output; PDF and PNG are optional.

    Params:
        collapse_csv: Path : per-peptide collapse-test CSV
        out_svg: Path : SVG output path (required)
        out_pdf: Path | None : optional PDF path
        out_png: Path | None : optional PNG path
    Returns:
        None
    """
    df = pd.read_csv(collapse_csv)
    required = {
        "topology",
        "has_proline",
        "has_glycine",
        "uniqueconfs_cremp",
        "pre_mmff_kabsch_0125",
        "post_mmff_kabsch_0125",
    }
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(
            f"collapse_csv missing required columns: {sorted(missing)}"
        )
    df = _add_collapse_columns(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _panel_a(axes[0], df)
    _panel_b(axes[1], df)
    _panel_c(axes[2], df)
    fig.suptitle(
        f"CREMP overlap statistics  (N={len(df)} peptides)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    logger.info("wrote SVG to %s", out_svg)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        logger.info("wrote PDF to %s", out_pdf)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        logger.info("wrote PNG to %s", out_png)
    plt.close(fig)


if __name__ == "__main__":
    main()
