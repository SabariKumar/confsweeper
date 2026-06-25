"""
2-panel paper figure for the ETKDG saturation benchmark — the coverage
problem that motivated the Monte Carlo samplers (docs/mcmm_plan.md, Phase 1).

Reads the saturation sweep CSV (output of `saturation_etkdg.py`) and renders,
for the original randomized confsweeper sampler (`params_mode == etkdg_original`,
no minimization, no dihedral jitter):

  - Panel A: max Boltzmann weight of the recovered ensemble vs seed budget.
    The original sampler hugs 1.0 (one dominant conformer carries all the
    population — a "one-hot" ensemble), while the CREMP ground-truth ceiling
    spreads mass far lower (dashed reference lines). More seeds do not close
    the gap: this is a connectivity failure, not a sampling-density failure.
  - Panel B: effective number of distinct conformers (eff_n = 1 / sum p_i^2)
    vs seed budget. The original sampler stays pinned near 1 regardless of
    compute; the CREMP reference holds hundreds of conformers (annotated).

Only the two CREMP peptides carry a ground-truth ceiling, so the figure is
restricted to them.

Usage:
    pixi run python scripts/etkdg_saturation_figure.py \\
        --saturation_csv  data/processed/saturation/saturation_etkdg.csv \\
        --out_svg         results/etkdg_saturation_figure.svg \\
        [--out_png        results/etkdg_saturation_figure.png]

SVG is the default and required output (vector, editable, lossless for the
paper). PNG is an optional secondary output for slides / quick preview.
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


# The two CREMP peptides carry a ground-truth ceiling; map id -> display label + colour.
_PEPTIDES = [
    ("cremp_typical:t.I.G.N", "cremp_typical (t.I.G.N)", "#1f77b4"),
    ("cremp_sharp:S.S.N.MeW.MeA.MeN", "cremp_sharp (S.S.N.MeW.MeA.MeN)", "#d62728"),
]


def _load_original_etkdg(saturation_csv: Path) -> pd.DataFrame:
    """
    Load the saturation sweep and isolate the original randomized ETKDG setting.

    Params:
        saturation_csv: Path : path to the saturation_etkdg sweep CSV
    Returns:
        pd.DataFrame : rows for params_mode == 'etkdg_original', minimize False,
                       dihedral_jitter_deg == 0, sorted by peptide and seed budget
    """
    df = pd.read_csv(saturation_csv)
    mask = (
        (df["params_mode"] == "etkdg_original")
        & (~df["minimize"].astype(bool))
        & (df["dihedral_jitter_deg"] == 0.0)
    )
    sub = df.loc[mask].sort_values(["peptide_id", "n_seeds"]).reset_index(drop=True)
    logger.info(
        "Selected %d original-ETKDG rows across %d peptides",
        len(sub),
        sub["peptide_id"].nunique(),
    )
    return sub


def _render(df: pd.DataFrame, out_svg: Path, out_png: Path | None) -> None:
    """
    Render the 2-panel saturation figure and write SVG (+ optional PNG).

    Params:
        df: pd.DataFrame : original-ETKDG rows from _load_original_etkdg
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    fig, (ax_bw, ax_eff) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- Panel A: max Boltzmann weight (one-hot collapse) ----
    ax_bw.axhspan(0.9, 1.0, color="0.85", zorder=0)
    ax_bw.text(
        130,
        0.905,
        "one-hot collapse\n(all population on one conformer)",
        fontsize=8,
        va="bottom",
        ha="left",
        color="0.35",
    )

    for pep_id, label, colour in _PEPTIDES:
        d = df[df["peptide_id"] == pep_id]
        if d.empty:
            continue
        ax_bw.plot(
            d["n_seeds"],
            d["max_bw"],
            marker="o",
            color=colour,
            label=label,
            zorder=3,
        )
        ceiling = d["ground_truth_max_bw"].iloc[0]
        ax_bw.axhline(ceiling, ls="--", color=colour, alpha=0.7, lw=1.2, zorder=2)
        ax_bw.text(
            d["n_seeds"].max(),
            ceiling,
            f"  CREMP ceiling = {ceiling:.2f}",
            fontsize=8,
            va="center",
            ha="left",
            color=colour,
        )

    ax_bw.set_xscale("log")
    ax_bw.set_xlabel("ETKDG seed budget (n_seeds)")
    ax_bw.set_ylabel("max Boltzmann weight of recovered ensemble")
    ax_bw.set_ylim(0.0, 1.05)
    ax_bw.set_title("A. Population collapses onto one conformer")
    ax_bw.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ---- Panel B: effective number of conformers ----
    for pep_id, label, colour in _PEPTIDES:
        d = df[df["peptide_id"] == pep_id]
        if d.empty:
            continue
        ax_eff.plot(
            d["n_seeds"], d["eff_n"], marker="o", color=colour, label=label, zorder=3
        )

    ax_eff.axhline(1.0, ls=":", color="0.4", lw=1.0, zorder=1)
    ax_eff.text(
        130, 1.02, "eff_n = 1  (single conformer)", fontsize=8, va="bottom", color="0.4"
    )

    # Annotate the reference conformer counts (raw CREMP uniqueconfs).
    ref_lines = [
        f"{label.split(' ')[0]}: {int(d['ground_truth_n_confs'].iloc[0])} CREMP conformers"
        for pep_id, label, _ in _PEPTIDES
        if not (d := df[df["peptide_id"] == pep_id]).empty
        and pd.notna(d["ground_truth_n_confs"].iloc[0])
    ]
    ax_eff.text(
        0.97,
        0.97,
        "CREMP reference holds:\n" + "\n".join(ref_lines),
        transform=ax_eff.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    ax_eff.set_xscale("log")
    ax_eff.set_xlabel("ETKDG seed budget (n_seeds)")
    ax_eff.set_ylabel("effective # conformers  (1 / Σ p$_i^2$)")
    ax_eff.set_ylim(0.5, max(2.5, df["eff_n"].max() * 1.1))
    ax_eff.set_title("B. More seeds do not broaden the ensemble")
    ax_eff.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Original randomized ETKDG saturates far from the CREMP ceiling",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg")
    logger.info("Wrote %s", out_svg)
    if out_png is not None:
        fig.savefig(out_png, format="png", dpi=150)
        logger.info("Wrote %s", out_png)
    plt.close(fig)


@click.command()
@click.option(
    "--saturation_csv",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/saturation/saturation_etkdg.csv"),
    show_default=True,
    help="Saturation sweep CSV (output of saturation_etkdg.py).",
)
@click.option(
    "--out_svg",
    type=click.Path(path_type=Path),
    default=Path("results/etkdg_saturation_figure.svg"),
    show_default=True,
    help="Required SVG output path.",
)
@click.option(
    "--out_png",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional PNG output path (secondary).",
)
def main(saturation_csv: Path, out_svg: Path, out_png: Path | None) -> None:
    """
    Build the ETKDG saturation / coverage-problem figure.

    Params:
        saturation_csv: Path : input saturation sweep CSV
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    df = _load_original_etkdg(saturation_csv)
    _render(df, out_svg, out_png)


if __name__ == "__main__":
    main()
