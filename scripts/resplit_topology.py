"""Step 8 (issue #20): composition-based re-split for an honest generalisation test.

The random peptide split can leak: CREMP contains cyclic permutations of the same
cyclic peptide (same molecule read from a different start residue) and linear
permutations, which a random split scatters across train/test — inflating the
apparent generalisation. This re-splits the existing dataset so that all peptides
sharing a residue *composition* (the sorted multiset of residue tokens, which
groups cyclic and linear permutations together) land in the same split.

Writes a sibling dataset pkl with the new split; everything else (features,
targets) is unchanged, so no re-extraction is needed.
"""

import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pickle  # noqa: E402

from dihedral_predictor.data import load_dataset  # noqa: E402


def _composition_key(seq: str) -> tuple:
    return tuple(sorted(seq.split(".")))


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--out", default="data/processed/dihedral_predictor/dataset_topo.pkl")
@click.option("--val_frac", default=0.1)
@click.option("--test_frac", default=0.1)
@click.option("--seed", default=42)
def main(dataset, out, val_frac, test_frac, seed):
    data = load_dataset(dataset)
    seqs = data["seqs"]
    # group peptide indices by composition
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(seqs):
        groups.setdefault(_composition_key(s), []).append(i)
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(seqs)
    n_test_target, n_val_target = int(n * test_frac), int(n * val_frac)
    split = np.empty(n, dtype=object)
    n_test = n_val = 0
    for k in keys:
        idxs = groups[k]
        if n_test < n_test_target:
            tag = "test"
            n_test += len(idxs)
        elif n_val < n_val_target:
            tag = "val"
            n_val += len(idxs)
        else:
            tag = "train"
        for i in idxs:
            split[i] = tag

    new = dict(data)
    new["split"] = split
    with open(out, "wb") as fh:
        pickle.dump(new, fh)
    n_groups = len(keys)
    multi = sum(1 for k in keys if len(groups[k]) > 1)
    print(
        f"composition groups: {n_groups} ({multi} with >1 peptide — permutation/near-dup clusters)"
    )
    print(
        f"  split: train {(split=='train').sum()} / val {(split=='val').sum()} / test {(split=='test').sum()} -> {out}"
    )


if __name__ == "__main__":
    main()
