"""
Builds a stratified validation subset from the CREMP summary CSV.

Stratification axes:
  - topology  : all-L | D-only | NMe-only | D+NMe  (derived from sequence)
  - num_monomers : 4 | 5 | 6
  - atom_bin  : small | medium | large  (tertiles of num_heavy_atoms within
                                         each monomer class)

Usage:
    python make_validation_sets_cremp.py \
        --summary_csv data/raw/cremp/summary.csv \
        --output_csv  data/processed/cremp/validation_subset.csv \
        --n_per_stratum 14 \
        --seed 42
"""

import os

import click
import numpy as np
import pandas as pd


def parse_topology(sequence: str) -> str:
    """
    Derives topology label from a CREMP dot-separated sequence string.

    Rules:
        - Lowercase residue name (after optional 'Me' prefix) → D-amino acid
        - 'Me' prefix → N-methylated backbone nitrogen
    Returns one of: 'all-L', 'D-only', 'NMe-only', 'D+NMe'
    """
    monomers = sequence.split(".")
    has_d = False
    has_nme = False
    for m in monomers:
        if m.startswith("Me"):
            has_nme = True
            # e.g. 'Mei', 'Mef' → D-amino acid under NMe
            if len(m) > 2 and m[2].islower():
                has_d = True
        else:
            if m[0].islower():
                has_d = True
    if has_d and has_nme:
        return "D+NMe"
    elif has_d:
        return "D-only"
    elif has_nme:
        return "NMe-only"
    else:
        return "all-L"


def assign_atom_bin(df: pd.DataFrame) -> pd.Series:
    """
    Assigns small/medium/large tertile bin for num_heavy_atoms *within*
    each num_monomers class, so that ring-size diversity is preserved
    independently of peptide length.
    """
    bins = pd.Series(index=df.index, dtype=str)
    for n in df["num_monomers"].unique():
        mask = df["num_monomers"] == n
        tertiles = df.loc[mask, "num_heavy_atoms"].quantile([1 / 3, 2 / 3]).values
        labels = pd.cut(
            df.loc[mask, "num_heavy_atoms"],
            bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
            labels=["small", "medium", "large"],
        )
        bins[mask] = labels.astype(str)
    return bins


def sample_subset(
    summary_csv: os.PathLike | str,
    n_per_stratum: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Reads the CREMP summary CSV and returns a stratified sample.

    Strata: topology × num_monomers × atom_bin  (4 × 3 × 3 = 36 strata).
    Molecules with fewer than n_per_stratum members are included in full.

    Params:
        summary_csv    : path to CREMP summary.csv
        n_per_stratum  : target samples per stratum
        seed           : random seed for reproducibility
    Returns:
        pd.DataFrame with columns from summary.csv plus 'topology' and 'atom_bin'
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(summary_csv)
    df["topology"] = df["sequence"].apply(parse_topology)
    df["atom_bin"] = assign_atom_bin(df)

    strata_keys = ["num_monomers", "topology", "atom_bin"]
    sampled = []
    for keys, group in df.groupby(strata_keys):
        n = min(n_per_stratum, len(group))
        chosen = group.sample(n=n, random_state=int(rng.integers(1 << 31)))
        sampled.append(chosen)

    result = pd.concat(sampled).reset_index(drop=True)
    return result


@click.command()
@click.option(
    "--summary_csv",
    default="data/raw/cremp/summary.csv",
    show_default=True,
    help="Path to CREMP summary.csv",
)
@click.option(
    "--output_csv",
    default="data/processed/cremp/validation_subset.csv",
    show_default=True,
    help="Path to write the sampled subset CSV",
)
@click.option(
    "--n_per_stratum",
    default=28,
    show_default=True,
    help="Target number of molecules per stratum (topology × monomers × atom_bin)",
)
@click.option("--seed", default=42, show_default=True, help="Random seed")
def main(summary_csv, output_csv, n_per_stratum, seed):
    """Build a stratified CREMP validation subset."""
    subset = sample_subset(summary_csv, n_per_stratum=n_per_stratum, seed=seed)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    subset.to_csv(output_csv, index=False)

    total = len(subset)
    print(f"Saved {total} molecules to {output_csv}")
    print()
    print("Stratum counts (topology × num_monomers × atom_bin):")
    counts = (
        subset.groupby(["topology", "num_monomers", "atom_bin"])
        .size()
        .rename("n")
        .reset_index()
    )
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
