"""
Broadened ETKDG-vs-MC ceiling-coverage figure (2026-06-19 finding follow-up).

Replaces the single-run 2-peptide before/after (scripts/before_after_coverage_figure.py,
whose "MC closes the gap" premise the 2026-06-19 finding falsified). Scores
exhaustive ETKDG against the v0.2 B.4 Monte Carlo sampler through the identical
coverage harness across a stratified CREMP subset, and draws a dumbbell plot:
one row per peptide, an ETKDG dot and an MC dot joined by a line, so where MC
helps (MC dot to the right), ties (dots coincide), or hurts (MC dot to the left)
is read at a glance.

Combines the broaden-sweep peptides with the two original peptides (which already
have coverage CSVs from the finding run).

Usage:
    pixi run python scripts/broaden_coverage_figure.py \\
        --out_svg results/broaden_coverage_figure.svg \\
        --out_png results/broaden_coverage_figure.png
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

# ETKDG and MC coverage CSVs to union: (etkdg_csv, mc_csv). Both the broaden
# sweep and the original 2-peptide finding run.
_SOURCES = [
    ("results/broaden_etkdg_coverage.csv", "results/broaden_mc_coverage.csv"),
    (
        "results/etkdg_baseline_coverage.csv",
        "results/sweep_v0_2_step7_coverage_B4_aromatic_on_skip_mmff_on.csv",
    ),
]

_ETKDG_COLOR = "#888888"
_MC_COLOR = "#1f77b4"


def _seq_from_pid(pid: str) -> str:
    """
    Recover the CREMP sequence label from a coverage-CSV peptide_id.

    Params:
        pid: str : peptide_id (e.g. 'cremp_C.F.G.S' or 'cremp_t.I.G.N')
    Returns:
        str : sequence (the peptide_id minus a leading 'cremp_')
    """
    return pid[len("cremp_") :] if pid.startswith("cremp_") else pid


def _load_coverage(path: Path) -> pd.DataFrame:
    """
    Load one coverage CSV (kabsch rows) keyed by recovered sequence.

    Params:
        path: Path : union_basin_count coverage CSV
    Returns:
        pd.DataFrame : columns [seq, coverage_bw_ceiling]
    """
    df = pd.read_csv(path)
    df = df[df["dedup_mode"] == "kabsch"].copy()
    df["seq"] = df["peptide_id"].map(_seq_from_pid)
    return df[["seq", "coverage_bw_ceiling"]]


def _assemble() -> pd.DataFrame:
    """
    Merge ETKDG and MC coverage across all sources, attach peptide metadata.

    Params: None
    Returns:
        pd.DataFrame : one row per peptide with etkdg, mc coverage + topology, arom
    """
    etk = pd.concat([_load_coverage(Path(e)) for e, _ in _SOURCES], ignore_index=True)
    mc = pd.concat([_load_coverage(Path(m)) for _, m in _SOURCES], ignore_index=True)
    etk = etk.rename(columns={"coverage_bw_ceiling": "etkdg"}).drop_duplicates("seq")
    mc = mc.rename(columns={"coverage_bw_ceiling": "mc"}).drop_duplicates("seq")
    out = etk.merge(mc, on="seq", how="inner")

    # metadata: topology + aromatic flag from the broaden list + the 2 originals
    meta = pd.read_csv("data/processed/cremp/broaden_peptides.csv")[
        ["sequence", "topology"]
    ]
    meta = meta.rename(columns={"sequence": "seq"})
    extra = pd.DataFrame(
        {"seq": ["t.I.G.N", "S.S.N.MeW.MeA.MeN"], "topology": ["D-only", "NMe-only"]}
    )
    meta = pd.concat([meta, extra], ignore_index=True).drop_duplicates("seq")
    out = out.merge(meta, on="seq", how="left")

    arom = {"f", "w", "y", "h"}
    out["arom"] = out["seq"].apply(
        lambda s: any(
            (t[2:] if t.startswith("Me") else t).lower() in arom for t in s.split(".")
        )
    )
    return out


def _render(df: pd.DataFrame, out_svg: Path, out_png: Path | None) -> None:
    """
    Render the dumbbell figure and write SVG (+ optional PNG).

    Params:
        df: pd.DataFrame : assembled per-peptide coverage table
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    df = df.sort_values(["mc", "etkdg"]).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9.0, 0.5 * len(df) + 2.0))

    for i, r in df.iterrows():
        # connecting line coloured by who wins
        gap = r["mc"] - r["etkdg"]
        lc = "#2ca02c" if gap > 0.01 else ("#d62728" if gap < -0.01 else "#bbbbbb")
        ax.plot([r["etkdg"], r["mc"]], [i, i], color=lc, lw=2.0, zorder=1)
    ax.scatter(
        df["etkdg"],
        y,
        s=55,
        color=_ETKDG_COLOR,
        label="exhaustive ETKDG (no MC)",
        zorder=3,
    )
    ax.scatter(
        df["mc"], y, s=55, color=_MC_COLOR, label="v0.2 Monte Carlo (B.4)", zorder=3
    )

    labels = [
        f"{r.seq}  ({r.topology}{', arom' if r.arom else ''})" for r in df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Boltzmann coverage of CREMP ceiling")
    ax.set_xlim(-0.03, 1.05)
    ax.axvline(1.0, ls="--", color="0.6", lw=0.8)

    n_help = int((df["mc"] - df["etkdg"] > 0.01).sum())
    n_hurt = int((df["mc"] - df["etkdg"] < -0.01).sum())
    n_tie = len(df) - n_help - n_hurt
    ax.set_title(
        f"ETKDG vs Monte Carlo ceiling coverage — {len(df)} CREMP peptides, matched 10k-seed budget\n"
        f"MC higher: {n_help}   tie: {n_tie}   MC lower: {n_hurt}   (single run each)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
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
    "--out_svg",
    type=click.Path(path_type=Path),
    default=Path("results/broaden_coverage_figure.svg"),
    show_default=True,
)
@click.option("--out_png", type=click.Path(path_type=Path), default=None)
def main(out_svg: Path, out_png: Path | None) -> None:
    """
    Assemble the broadened coverage table and render the dumbbell figure.

    Params:
        out_svg: Path : required SVG output path
        out_png: Path | None : optional PNG output path
    Returns:
        None
    """
    df = _assemble()
    logger.info("assembled %d peptides", len(df))
    _render(df, out_svg, out_png)


if __name__ == "__main__":
    main()
