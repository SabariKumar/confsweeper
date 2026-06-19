"""
Before/after coverage figure: original randomized ETKDG vs the v0.2 Monte
Carlo sampler, scored through the *same* Boltzmann-coverage harness.

The two earlier figures live on different axes (seed budget vs ablation cell)
and cannot be overlaid. This one puts them on one footing: it scores an
exhaustive-ETKDG basin set and the v0.2 B.4 (aromatic + skip-MMFF) basin set
through union_basin_count.py against the identical CREMP ceiling, then plots
Boltzmann coverage of that ceiling side by side for both CREMP peptides.

Message: on cremp_typical the MC sampler lifts ceiling coverage from ~0 (the
one-hot ETKDG collapse) to ~1.0; on cremp_sharp both methods sit at 0 — the
residual structural wall that motivates v0.3 concerted moves.

Usage:
    pixi run python scripts/before_after_coverage_figure.py \\
        --etkdg_csv  results/etkdg_baseline_coverage.csv \\
        --mc_csv     results/sweep_v0_2_step7_coverage_B4_aromatic_on_skip_mmff_on.csv \\
        --out_svg    results/before_after_coverage_figure.svg \\
        [--out_png   results/before_after_coverage_figure.png]

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


# (display label, colour, substring that identifies the peptide in any peptide_id).
_PEPTIDES = [
    ("cremp_typical (t.I.G.N)", "#1f77b4", "t.I.G.N"),
    ("cremp_sharp (S.S.N.MeW.MeA.MeN)", "#d62728", "S.S.N.MeW.MeA.MeN"),
]


def _coverage_for(df: pd.DataFrame, key: str) -> float:
    """
    Pull kabsch coverage_bw_ceiling for the peptide whose id contains `key`.

    Params:
        df: pd.DataFrame : a coverage CSV from union_basin_count.py
        key: str : substring identifying the peptide (e.g. 't.I.G.N')
    Returns:
        float : coverage_bw_ceiling for that peptide under kabsch dedup
    """
    d = df[df["dedup_mode"] == "kabsch"] if "dedup_mode" in df.columns else df
    hit = d[d["peptide_id"].str.contains(key, regex=False)]
    if hit.empty:
        raise ValueError(f"no peptide_id containing {key!r} in coverage CSV")
    return float(hit["coverage_bw_ceiling"].iloc[0])


def _render(etkdg_csv: Path, mc_csv: Path, out_svg: Path, out_png: Path | None) -> None:
    """
    Render the before/after grouped-bar coverage figure and write outputs.

    Params:
        etkdg_csv: Path : ETKDG-baseline coverage CSV
        mc_csv: Path : v0.2 B.4 Monte Carlo coverage CSV
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    etkdg = pd.read_csv(etkdg_csv)
    mc = pd.read_csv(mc_csv)

    x = np.arange(len(_PEPTIDES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for i, (csv_df, method, hatch) in enumerate(
        [(etkdg, "original ETKDG (no MC)", None), (mc, "v0.2 Monte Carlo (B.4)", None)]
    ):
        offset = (i - 0.5) * width
        for j, (label, colour, key) in enumerate(_PEPTIDES):
            cov = _coverage_for(csv_df, key)
            # ETKDG bars use a desaturated grey edge of the peptide colour to
            # read as "before"; MC bars are the saturated peptide colour.
            face = "0.75" if i == 0 else colour
            bar = ax.bar(
                x[j] + offset,
                cov,
                width,
                color=face,
                edgecolor=colour,
                linewidth=1.4,
                label=method if j == 0 else None,
            )
            ax.text(
                x[j] + offset,
                cov + 0.015,
                f"{cov:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if cov > 0.5 else "normal",
                color=colour if i == 1 else "0.35",
            )

    ax.axhline(1.0, ls="--", color="0.4", lw=1.0)
    ax.text(-0.45, 1.005, "full coverage", fontsize=8, color="0.4", va="bottom")

    # Narrate the two outcomes.
    ax.annotate(
        "MC closes the gap",
        xy=(0 + 0.5 * width, 0.99),
        xytext=(0.0, 0.55),
        fontsize=9,
        color="#1f77b4",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    ax.text(
        1,
        0.07,
        "both still 0\n(residual wall → v0.3)",
        fontsize=9,
        color="#d62728",
        ha="center",
        va="bottom",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in _PEPTIDES], fontsize=9)
    ax.set_ylabel("Boltzmann coverage of CREMP ceiling")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Original ETKDG vs v0.2 Monte Carlo — same coverage harness")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg")
    logger.info("Wrote %s", out_svg)
    if out_png is not None:
        fig.savefig(out_png, format="png", dpi=150)
        logger.info("Wrote %s", out_png)
    plt.close(fig)


@click.command()
@click.option(
    "--etkdg_csv",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/etkdg_baseline_coverage.csv"),
    show_default=True,
    help="ETKDG-baseline coverage CSV (union_basin_count.py output).",
)
@click.option(
    "--mc_csv",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/sweep_v0_2_step7_coverage_B4_aromatic_on_skip_mmff_on.csv"),
    show_default=True,
    help="v0.2 B.4 Monte Carlo coverage CSV.",
)
@click.option(
    "--out_svg",
    type=click.Path(path_type=Path),
    default=Path("results/before_after_coverage_figure.svg"),
    show_default=True,
    help="Required SVG output path.",
)
@click.option(
    "--out_png",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional PNG output path (secondary).",
)
def main(etkdg_csv: Path, mc_csv: Path, out_svg: Path, out_png: Path | None) -> None:
    """
    Build the before/after ETKDG-vs-MC coverage figure.

    Params:
        etkdg_csv: Path : ETKDG-baseline coverage CSV
        mc_csv: Path : v0.2 B.4 Monte Carlo coverage CSV
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    _render(etkdg_csv, mc_csv, out_svg, out_png)


if __name__ == "__main__":
    main()
