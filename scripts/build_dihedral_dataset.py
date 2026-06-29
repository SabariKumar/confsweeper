"""Build the dihedral-predictor training dataset from CREMP (issue #20).

Extracts per-residue features + dominant-conformer binned (phi/psi/omega)
targets for every CREMP peptide and saves a peptide-split dataset pickle.

Usage:
    pixi run python scripts/build_dihedral_dataset.py \\
        --summary data/raw/cremp/summary.csv \\
        --pickle_dir data/raw/cremp/pickle \\
        --out data/processed/dihedral_predictor/dataset.pkl
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.data import build_dataset  # noqa: E402


@click.command()
@click.option("--summary", default="data/raw/cremp/summary.csv")
@click.option("--pickle_dir", default="data/raw/cremp/pickle")
@click.option("--out", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--seed", default=42)
@click.option("--limit", default=None, type=int)
def main(summary, pickle_dir, out, seed, limit):
    build_dataset(summary, pickle_dir, out, seed=seed, limit=limit)


if __name__ == "__main__":
    main()
