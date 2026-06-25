"""
2-panel paper figure for the v0.2 aromatic-aware + MMFF-ablation result
(docs/dihedral_kick_v0_2_plan.md, Phase 0.2 / issue #15).

Companion to scripts/etkdg_saturation_figure.py: that figure shows the original
randomized ETKDG sampler collapsing to a one-hot ensemble far from the CREMP
ceiling; this one shows where the Monte Carlo sampler stands after the v0.2
2x2 ablation (aromatic-aware rotamer wells x skip-MMFF relax).

Reads the four v0.2 coverage CSVs (one per ablation cell, output of the
cremp coverage benchmark) and renders, for the two CREMP peptides:

  - Panel A: Boltzmann coverage of the CREMP ceiling across the 2x2 cells.
    cremp_typical climbs to ~1.0 at cell B.4 (aromatic + skip-MMFF), the
    synergistic win — both fixes together beat either alone. cremp_sharp
    stays flat at 0.0: its dominant ceiling basin is never reached.
  - Panel B: the cremp_sharp residual wall. The new (sampler-discovered,
    thermodynamically real) basin mass climbs ~3000x from B.1 to B.4, yet
    the dominant ceiling basin (max_missed_bw = 0.72) is still missed by
    every cell — motivating the v0.3 concerted multi-dihedral moves.

Usage:
    pixi run python scripts/v0_2_coverage_figure.py \\
        --out_svg  results/v0_2_coverage_figure.svg \\
        [--out_png results/v0_2_coverage_figure.png]

SVG is the default and required output; PNG is an optional secondary output.
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


# The four v0.2 ablation cells: (label, on/off tick caption, coverage CSV).
_CELLS = [
    ("B.1", "–\n–", "results/sweep_v0_2_step4_coverage_aromatic_off.csv"),
    ("B.2", "aromatic\n–", "results/sweep_v0_2_step4_coverage_aromatic_on.csv"),
    (
        "B.3",
        "–\nskip-MMFF",
        "results/sweep_v0_2_step7_coverage_B3_aromatic_off_skip_mmff_on.csv",
    ),
    (
        "B.4",
        "aromatic\nskip-MMFF",
        "results/sweep_v0_2_step7_coverage_B4_aromatic_on_skip_mmff_on.csv",
    ),
]

# Peptide id in the coverage CSVs -> (display label, colour). Matches etkdg_saturation_figure.
_PEPTIDES = [
    ("cremp_t.I.G.N", "cremp_typical (t.I.G.N)", "#1f77b4"),
    ("cremp_S.S.N.MeW.MeA.MeN", "cremp_sharp (S.S.N.MeW.MeA.MeN)", "#d62728"),
]

_MASS_FLOOR = 1e-8  # log-axis floor for near-zero new-basin masses


def _load_cells(repo_root: Path) -> pd.DataFrame:
    """
    Load the four v0.2 ablation CSVs into a long table keyed by cell and peptide.

    Params:
        repo_root: Path : repository root, used to resolve the CSV paths
    Returns:
        pd.DataFrame : columns [cell, caption, peptide_id, coverage_bw_ceiling,
                       new_basin_mass_joint, max_missed_bw]; kabsch dedup only
    """
    rows = []
    for cell, caption, rel in _CELLS:
        df = pd.read_csv(repo_root / rel)
        df = df[df["dedup_mode"] == "kabsch"]
        for pep_id, _, _ in _PEPTIDES:
            r = df[df["peptide_id"] == pep_id].iloc[0]
            rows.append(
                {
                    "cell": cell,
                    "caption": caption,
                    "peptide_id": pep_id,
                    "coverage_bw_ceiling": r["coverage_bw_ceiling"],
                    "new_basin_mass_joint": r["new_basin_mass_joint"],
                    "max_missed_bw": r["max_missed_bw"],
                }
            )
    out = pd.DataFrame(rows)
    logger.info("Loaded %d cell x peptide rows", len(out))
    return out


def _render(df: pd.DataFrame, out_svg: Path, out_png: Path | None) -> None:
    """
    Render the 2-panel v0.2 coverage figure and write SVG (+ optional PNG).

    Params:
        df: pd.DataFrame : long table from _load_cells
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    cells = [c for c, _, _ in _CELLS]
    captions = [f"{c}\n{cap}" for c, cap, _ in _CELLS]
    x = np.arange(len(cells))
    width = 0.38

    fig, (ax_cov, ax_res) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- Panel A: Boltzmann coverage of the CREMP ceiling, 2x2 cells ----
    for i, (pep_id, label, colour) in enumerate(_PEPTIDES):
        vals = [
            df[(df.cell == c) & (df.peptide_id == pep_id)]["coverage_bw_ceiling"].iloc[
                0
            ]
            for c in cells
        ]
        bars = ax_cov.bar(x + (i - 0.5) * width, vals, width, label=label, color=colour)
        # Value labels inside the top of each cremp_typical bar (the only tall series).
        if pep_id == "cremp_t.I.G.N":
            for c, b, v in zip(cells, bars, vals):
                ax_cov.text(
                    b.get_x() + b.get_width() / 2,
                    v - 0.02,
                    f"{v:.3f}",
                    fontsize=8.5,
                    color="white",
                    ha="center",
                    va="top",
                    fontweight="bold" if c == "B.4" else "normal",
                )

    ax_cov.axhline(1.0, ls="--", color="0.4", lw=1.0)
    ax_cov.text(-0.45, 1.005, "full coverage", fontsize=8, color="0.4", va="bottom")
    ax_cov.text(
        3 - 0.5 * width,
        1.02,
        "synergy\nwin",
        fontsize=8,
        color="#1f77b4",
        ha="center",
        va="bottom",
    )
    ax_cov.text(
        3,
        0.03,
        "cremp_sharp\nstays at 0",
        fontsize=8,
        color="#d62728",
        ha="center",
        va="bottom",
    )

    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(captions, fontsize=8)
    ax_cov.set_ylabel("Boltzmann coverage of CREMP ceiling")
    ax_cov.set_xlabel("v0.2 ablation cell  (aromatic wells / skip-MMFF)")
    ax_cov.set_ylim(0.0, 1.08)
    ax_cov.set_title("A. Aromatic + skip-MMFF synergise on cremp_typical")
    ax_cov.legend(loc="center left", fontsize=8, framealpha=0.9)

    # ---- Panel B: cremp_sharp residual wall ----
    sharp = "cremp_S.S.N.MeW.MeA.MeN"
    new_mass = [
        max(
            df[(df.cell == c) & (df.peptide_id == sharp)]["new_basin_mass_joint"].iloc[
                0
            ],
            _MASS_FLOOR,
        )
        for c in cells
    ]
    missed = df[(df.cell == "B.4") & (df.peptide_id == sharp)]["max_missed_bw"].iloc[0]

    ax_res.bar(
        x, new_mass, 0.6, color="#d62728", alpha=0.85, label="new real basin mass found"
    )
    ax_res.axhline(missed, ls="--", color="0.25", lw=1.3)
    ax_res.text(
        0.0,
        missed * 0.62,
        f"dominant ceiling basin (mass {missed:.2f}) — NEVER reached by any cell",
        fontsize=8,
        color="0.25",
        ha="left",
        va="top",
    )
    ax_res.annotate(
        "~3000x rise\nB.1 → B.4",
        xy=(3, new_mass[3]),
        xytext=(1.3, new_mass[3] * 0.15),
        fontsize=8.5,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728"),
    )

    ax_res.set_yscale("log")
    ax_res.set_ylim(_MASS_FLOOR, 1.5)
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(captions, fontsize=8)
    ax_res.set_ylabel("Boltzmann mass (log scale)")
    ax_res.set_xlabel("v0.2 ablation cell  (aromatic wells / skip-MMFF)")
    ax_res.set_title("B. cremp_sharp: real basins found, dominant basin walled")
    ax_res.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "After v0.2 Monte Carlo sampling: cremp_typical solved, cremp_sharp wall remains",
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
    "--repo_root",
    type=click.Path(exists=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Repository root used to resolve the v0.2 coverage CSV paths.",
)
@click.option(
    "--out_svg",
    type=click.Path(path_type=Path),
    default=Path("results/v0_2_coverage_figure.svg"),
    show_default=True,
    help="Required SVG output path.",
)
@click.option(
    "--out_png",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional PNG output path (secondary).",
)
def main(repo_root: Path, out_svg: Path, out_png: Path | None) -> None:
    """
    Build the v0.2 Monte Carlo coverage figure.

    Params:
        repo_root: Path : repository root for resolving input CSVs
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    df = _load_cells(repo_root)
    _render(df, out_svg, out_png)


if __name__ == "__main__":
    main()
