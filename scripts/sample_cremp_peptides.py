"""
Stratified sampler over the CREMP `summary.csv` for the at-scale collapse
benchmark (Step 19 of docs/mcmm_plan.md).

Strata: 4 topology × 2 has_proline × 2 has_glycine = 16 cells.

Topology axis matches `make_validation_sets_cremp.parse_topology` so the
labels here agree with `validation_subset.csv`. Pro and Gly axes are
binary presence flags derived directly from each sequence's residue
tokens (CREMP convention: `[A-Za-z]` or `Me[A-Za-z]`, where the
case of the residue letter denotes L/D and the `Me` prefix denotes
N-methylation; we strip the `Me` prefix before checking for `P`/`p`
or `G`/`g`).

Output CSV mirrors `validation_subset.csv`'s schema (sequence, smiles,
num_monomers, num_atoms, num_heavy_atoms, uniqueconfs, lowestenergy,
topology) plus the new feature columns (`has_proline`, `has_glycine`)
so downstream scripts can group by them without re-deriving.

Usage:
    pixi run python scripts/sample_cremp_peptides.py \\
        --summary_csv      data/raw/cremp/summary.csv \\
        --out_csv          data/processed/cremp/overlap_benchmark_sample.csv \\
        --n_per_cell       100 \\
        --min_per_cell     30 \\
        --seed             42
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd

# Reuse the existing topology parser so labels agree with validation_subset.csv.
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from validation.make_validation_sets_cremp import parse_topology  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


def _residue_letters(sequence: str) -> list[str]:
    """
    Return the upper-case residue letter for each token in a CREMP
    sequence, stripping the `Me` prefix when present.

    Example: 't.I.G.N' → ['T', 'I', 'G', 'N']
             'MeV.MeT.Mev.Q' → ['V', 'T', 'V', 'Q']

    Params:
        sequence: str : dot-separated CREMP sequence string
    Returns:
        list[str] : upper-case single-letter residue codes
    """
    letters: list[str] = []
    for token in sequence.split("."):
        residue = token[2:] if token.startswith("Me") else token
        if not residue:
            continue
        letters.append(residue[0].upper())
    return letters


def _has_residue(sequence: str, code: str) -> bool:
    """
    Whether `sequence` contains at least one residue whose letter (after
    stripping any `Me` prefix and case-folding to upper) equals `code`.

    Catches Pro / Gly across both isomers and N-methylation states:
    `P`, `p`, `MeP`, `Mep` all count as proline; `G`/`g`/`MeG`/`Meg`
    all count as glycine.

    Params:
        sequence: str : CREMP sequence string
        code: str : single-letter upper-case amino-acid code
    Returns:
        bool
    """
    return code.upper() in _residue_letters(sequence)


def _annotate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four columns the stratification consumes — `topology`,
    `has_proline`, `has_glycine`, `cell` — to a copy of the
    summary-csv-shaped DataFrame.

    Params:
        df: pd.DataFrame : rows from summary.csv (must contain `sequence`)
    Returns:
        pd.DataFrame : same rows with feature columns appended
    """
    out = df.copy()
    out["topology"] = out["sequence"].apply(parse_topology)
    out["has_proline"] = out["sequence"].apply(lambda s: _has_residue(s, "P"))
    out["has_glycine"] = out["sequence"].apply(lambda s: _has_residue(s, "G"))
    out["cell"] = (
        out["topology"]
        + "_"
        + out["has_proline"].map({True: "Pro", False: "noPro"})
        + "_"
        + out["has_glycine"].map({True: "Gly", False: "noGly"})
    )
    return out


def _sample_stratified(
    df: pd.DataFrame,
    n_per_cell: int,
    min_per_cell: int,
    seed: int,
) -> pd.DataFrame:
    """
    Pull a stratified sample over `(topology, has_proline, has_glycine)`.

    For each cell, sample up to `n_per_cell` rows. Cells with fewer than
    `n_per_cell` peptides naturally are taken in full. Cells with fewer
    than `min_per_cell` peptides are dropped with a logged warning so
    the downstream summary stats are guaranteed to have meaningful N.

    Params:
        df: pd.DataFrame : summary rows with feature columns appended
        n_per_cell: int : target per-cell sample size
        min_per_cell: int : floor below which a cell is dropped from the sample
        seed: int : pandas RNG seed for reproducibility
    Returns:
        pd.DataFrame : the sampled rows, with `cell` retained
    """
    chunks: list[pd.DataFrame] = []
    for cell, group in df.groupby("cell"):
        n_avail = len(group)
        if n_avail < min_per_cell:
            logger.warning(
                "cell %s has %d peptides (< min_per_cell=%d); dropping from sample",
                cell,
                n_avail,
                min_per_cell,
            )
            continue
        n_take = min(n_per_cell, n_avail)
        chunks.append(group.sample(n=n_take, random_state=seed))
        logger.info(
            "  cell %-30s : sampled %d / %d (target %d)",
            cell,
            n_take,
            n_avail,
            n_per_cell,
        )
    if not chunks:
        raise RuntimeError(
            "no cells met the min_per_cell floor; check the summary CSV "
            "and stratification logic"
        )
    return pd.concat(chunks, ignore_index=True)


@click.command()
@click.option(
    "--summary_csv",
    required=True,
    type=Path,
    help="CREMP summary.csv (one row per peptide, with `sequence` and `smiles`).",
)
@click.option(
    "--out_csv",
    required=True,
    type=Path,
    help="Output CSV path for the stratified sample.",
)
@click.option(
    "--n_per_cell",
    type=int,
    default=100,
    show_default=True,
    help="Target sample size per (topology × has_proline × has_glycine) cell.",
)
@click.option(
    "--min_per_cell",
    type=int,
    default=30,
    show_default=True,
    help="Floor below which a cell is dropped from the sample with a warning.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Deterministic RNG seed for the per-cell sample.",
)
def main(
    summary_csv: Path,
    out_csv: Path,
    n_per_cell: int,
    min_per_cell: int,
    seed: int,
) -> None:
    """
    Read CREMP `summary.csv`, derive topology + Pro/Gly feature flags,
    pull a stratified sample over the 16-cell grid, and write the
    selected peptides to `out_csv`.

    Params:
        summary_csv: Path : full CREMP summary CSV
        out_csv: Path : where to write the sampled subset
        n_per_cell: int : target per-cell sample size
        min_per_cell: int : floor below which cells are dropped
        seed: int : sampling seed
    Returns:
        None
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    logger.info("loaded %d peptides from %s", len(summary), summary_csv)

    annotated = _annotate_features(summary)
    cell_counts = annotated["cell"].value_counts().sort_index()
    logger.info("natural cell distribution (%d cells):", len(cell_counts))
    for cell, n in cell_counts.items():
        logger.info("  %-30s : %d", cell, n)

    sample = _sample_stratified(annotated, n_per_cell, min_per_cell, seed)
    sample = sample.sort_values(["topology", "has_proline", "has_glycine", "sequence"])
    sample.to_csv(out_csv, index=False)
    logger.info("wrote %d sampled peptides to %s", len(sample), out_csv)


if __name__ == "__main__":
    main()
