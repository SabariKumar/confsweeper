"""Step 8 (issue #20): aggregate coverage lift from learned seeding over a set of
INVERTED TEST-split peptides — to size the gap before improving side-chain precision.

For each inverted test peptide: build the raw-CREST ceiling (CREST-Boltzmann weights)
on both all-heavy and backbone atom sets, run the real MCMM de-novo (baseline) and with
learned backbone+chi seeds (no-relax path), and measure cov_bw_ceil @0.75 Å for both
atom sets. Reports per-peptide rows and the mean lift, separating how much of the
inversion is backbone-reachable de-novo vs side-chain-limited.

Test split only. Resume-aware CSV.
"""

import csv
import os
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from confsweeper import get_hardware_opts, get_mace_calc  # noqa: E402
from dihedral_predictor.data import load_dataset  # noqa: E402
from dihedral_predictor.residues import residue_atoms  # noqa: E402
from dihedral_predictor.seed import (  # noqa: E402
    load_chi_model,
    load_model,
    seed_conformers,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_seeding_coverage import _run, build_crest_ceiling, coverage  # noqa: E402


def _inverted_dmmff(cmol, w, rng, max_confs=60, thresh=2.0):
    """Subsampled relaxed dMMFF of the dominant; True if inverted (> thresh)."""
    cids = [c.GetId() for c in cmol.GetConformers()]
    dom_id = cids[int(w.argmax())]
    if len(cids) > max_confs:
        keep = set(
            rng.choice([c for c in cids if c != dom_id], max_confs - 1, replace=False)
        ) | {dom_id}
    else:
        keep = set(cids)
    m = Chem.Mol(cmol)
    for cid in cids:
        if cid not in keep:
            m.RemoveConformer(cid)
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=400, numThreads=0)
    p = AllChem.MMFFGetMoleculeProperties(m)
    e = {
        c.GetId(): AllChem.MMFFGetMoleculeForceField(
            m, p, confId=c.GetId()
        ).CalcEnergy()
        for c in m.GetConformers()
    }
    return (e[dom_id] - min(e.values())) > thresh


def _cov(mol, cids, basin, weights, idx, thr=0.75):
    if not cids:
        return 0.0
    sh = torch.tensor(
        np.stack([mol.GetConformer(c).GetPositions()[idx] for c in cids]),
        dtype=torch.float64,
    )
    return coverage(sh, basin, weights, thr)


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--pickle_dir", default="data/raw/cremp/pickle")
@click.option("--ckpt", default="data/processed/dihedral_predictor/ckpt_w2_d256_l6.pt")
@click.option(
    "--chi_ckpt", default="data/processed/dihedral_predictor/chi_checkpoint.pt"
)
@click.option(
    "--out", default="data/processed/dihedral_predictor/aggregate_coverage.csv"
)
@click.option("--n_inverted", default=15)
@click.option("--n_seeds", default=4800)
@click.option("--n_conf", default=16)
def main(dataset, pickle_dir, ckpt, chi_ckpt, out, n_inverted, n_seeds, n_conf):
    data = load_dataset(dataset)
    test_seqs = [s for s, sp in zip(data["seqs"], data["split"]) if sp == "test"]
    rng = np.random.default_rng(0)
    rng.shuffle(test_seqs)
    hw = get_hardware_opts()
    calc = get_mace_calc()
    model, window = load_model(ckpt)
    chi_model, chi_window = load_chi_model(chi_ckpt)

    done = set()
    if os.path.exists(out):
        done = {r["sequence"] for r in csv.DictReader(open(out))}
    write_header = not os.path.exists(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "a", newline="")
    w_csv = csv.writer(fh)
    if write_header:
        w_csv.writerow(
            [
                "sequence",
                "dom_w_heavy",
                "dom_w_bb",
                "base_heavy",
                "seed_heavy",
                "base_bb",
                "seed_bb",
            ]
        )

    n_done = len(done)
    for seq in test_seqs:
        if n_done >= n_inverted:
            break
        if seq in done:
            continue
        try:
            d = pickle.load(open(f"{pickle_dir}/{seq}.pickle", "rb"))
            cmol = d["rd_mol"]
            wt = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            if not _inverted_dmmff(cmol, wt, rng):
                continue
            smi = d["smiles"]
            ms = Chem.AddHs(Chem.MolFromSmiles(smi))
            match = cmol.GetSubstructMatch(ms, useChirality=True)
            heavy = [a.GetIdx() for a in ms.GetAtoms() if a.GetAtomicNum() > 1]
            bb = [a for n, ca, c in residue_atoms(ms) for a in (n, ca, c)]
            bh_h, w_h, _ = build_crest_ceiling(cmol, wt, ms, match, heavy)
            bh_b, w_b, _ = build_crest_ceiling(cmol, wt, ms, match, bb)

            seed_src = Chem.Mol(ms)
            seed_src.RemoveAllConformers()
            sids = seed_conformers(
                seed_src,
                model,
                window=window,
                n_attempts=n_conf,
                chi_model=chi_model,
                chi_window=chi_window,
            )
            learned = [seed_src.GetConformer(c).GetPositions() for c in sids]

            row = {}
            for label, extra, relax in [("base", None, True), ("seed", learned, False)]:
                mol, cids, _ = _run(
                    smi, hw, calc, extra, n_seeds, False, relax_seeds=relax
                )
                row[f"{label}_heavy"] = _cov(mol, cids, bh_h, w_h, heavy)
                row[f"{label}_bb"] = _cov(mol, cids, bh_b, w_b, bb)
            w_csv.writerow(
                [
                    seq,
                    f"{w_h.max():.3f}",
                    f"{w_b.max():.3f}",
                    f"{row['base_heavy']:.3f}",
                    f"{row['seed_heavy']:.3f}",
                    f"{row['base_bb']:.3f}",
                    f"{row['seed_bb']:.3f}",
                ]
            )
            fh.flush()
            n_done += 1
            print(
                f"  [{n_done}/{n_inverted}] {seq}: heavy base={row['base_heavy']:.2f} "
                f"seed={row['seed_heavy']:.2f} | bb base={row['base_bb']:.2f} "
                f"seed={row['seed_bb']:.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {seq}: {exc}", flush=True)
    fh.close()

    rows = list(csv.DictReader(open(out)))

    def col(k):
        return np.array([float(r[k]) for r in rows])

    print(
        f"\n=== aggregate over {len(rows)} inverted test peptides (cov_bw_ceil @0.75) ==="
    )
    print(
        f"ALL-ATOM:  baseline mean {col('base_heavy').mean():.3f} | seeded {col('seed_heavy').mean():.3f} "
        f"| lift {(col('seed_heavy')-col('base_heavy')).mean():+.3f} | seeded>base {100*np.mean(col('seed_heavy')>col('base_heavy')):.0f}%"
    )
    print(
        f"BACKBONE:  baseline mean {col('base_bb').mean():.3f} | seeded {col('seed_bb').mean():.3f} "
        f"| lift {(col('seed_bb')-col('base_bb')).mean():+.3f} | seeded>base {100*np.mean(col('seed_bb')>col('base_bb')):.0f}%"
    )


if __name__ == "__main__":
    main()
