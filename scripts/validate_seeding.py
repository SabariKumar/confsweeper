"""Validate learned-dihedral seeding end-to-end (issue #20).

The classification metrics (phi/psi/omega bin accuracy) are a proxy. The real
test: does constrained-DG seeding from *predicted* dihedrals land the CREST-
dominant basin for INVERTED peptides — the ones where de-novo sampling fails?

For each inverted test-split peptide (relaxed dMMFF > 2), compare two ways of
generating conformers de novo and measure the minimum heavy-atom RMSD each
achieves to the CREST-dominant conformer:
  - seeded:   predicted dihedrals -> constrained DG -> MMFF relax
  - baseline: unconstrained macrocycle ETKDG        -> MMFF relax
If seeding reaches a much lower min-RMSD-to-dominant than unconstrained ETKDG,
the learned predictor recovers basins the de-novo explorer cannot.

Test split only (no training leakage). CPU model + RDKit.
"""

import sys
from pathlib import Path

import click
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pickle  # noqa: E402

from confsweeper import get_embed_params_macrocycle  # noqa: E402
from dihedral_predictor.data import load_dataset  # noqa: E402
from dihedral_predictor.residues import residue_atoms  # noqa: E402
from dihedral_predictor.seed import load_model, seed_conformers  # noqa: E402
from mcmm import _kabsch_rmsd_pairwise  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _backbone_idx(mol):
    """Backbone atom indices (N, Ca, C per residue) — the fold-defining atoms.

    The dominant basin is defined by its backbone fold; flexible side chains
    embedded by DG won't match CREST and would inflate a heavy-atom RMSD, so we
    measure proximity to the dominant on backbone atoms only.
    """
    idx = []
    for n, ca, c in residue_atoms(mol):
        idx.extend([n, ca, c])
    return idx


def _coords(mol, conf_id, idx):
    return mol.GetConformer(conf_id).GetPositions()[idx]


def _min_rmsd(mol_confs, dom_bb, idx):
    """Min backbone Kabsch RMSD from any conformer of mol_confs to dom_bb."""
    if mol_confs.GetNumConformers() == 0:
        return np.inf
    refs = np.stack([c.GetPositions()[idx] for c in mol_confs.GetConformers()])
    rmsds = _kabsch_rmsd_pairwise(torch.from_numpy(dom_bb), torch.from_numpy(refs))
    return float(rmsds.min())


def _both_min_rmsd(mol_confs, dom_bb, idx):
    """Return (unrelaxed, MMFF-relaxed) min backbone RMSD to dom_bb.

    The unrelaxed RMSD isolates whether prediction + constrained DG reaches the
    dominant fold; the relaxed RMSD reflects what survives the sampler's MMFF step.
    """
    if mol_confs.GetNumConformers() == 0:
        return np.inf, np.inf
    pre = _min_rmsd(mol_confs, dom_bb, idx)
    AllChem.MMFFOptimizeMoleculeConfs(mol_confs, maxIters=500, numThreads=0)
    post = _min_rmsd(mol_confs, dom_bb, idx)
    return pre, post


def _relaxed_dmmff_dominant(mol, bw, max_confs=60, rng=None):
    """Relaxed dMMFF of the dominant conformer, estimated on a conformer subsample.

    CREMP peptides carry ~460 conformers on average; relaxing all of them just to
    decide whether a peptide is inverted is the run's bottleneck. We relax the
    dominant plus a random subsample (the low-energy basin is well-populated, so a
    subsample captures the MMFF min) — a fast, adequate inverted/not filter.
    """
    dom = int(bw.argmax())
    cids = [c.GetId() for c in mol.GetConformers()]
    dom_id = cids[dom]
    if rng is not None and len(cids) > max_confs:
        others = [c for c in cids if c != dom_id]
        keep = set(rng.choice(others, max_confs - 1, replace=False)) | {dom_id}
    else:
        keep = set(cids)
    m = Chem.Mol(mol)
    for cid in cids:
        if cid not in keep:
            m.RemoveConformer(cid)
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=400, numThreads=0)
    props = AllChem.MMFFGetMoleculeProperties(m)
    e = {
        c.GetId(): AllChem.MMFFGetMoleculeForceField(
            m, props, confId=c.GetId()
        ).CalcEnergy()
        for c in m.GetConformers()
    }
    return float(e[dom_id] - min(e.values()))


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--ckpt", default="data/processed/dihedral_predictor/checkpoint.pt")
@click.option("--pickle_dir", default="data/raw/cremp/pickle")
@click.option("--n_inverted", default=120, help="target #inverted peptides to test")
@click.option("--n_conf", default=24, help="seeded attempts / baseline confs")
@click.option("--inv_thresh", default=2.0)
@click.option(
    "--tol", default=60.0, help="constrained-DG tolerance (deg) for phi/psi/omega"
)
def main(dataset, ckpt, pickle_dir, n_inverted, n_conf, inv_thresh, tol):
    data = load_dataset(dataset)
    test_seqs = [s for s, sp in zip(data["seqs"], data["split"]) if sp == "test"]
    rng = np.random.default_rng(0)
    rng.shuffle(test_seqs)
    model, window = load_model(ckpt, device="cpu")

    seeded_pre, seeded_post, base_pre, base_post, n_seed_confs = [], [], [], [], []
    n_done = 0
    n_seen = 0
    for seq in test_seqs:
        if n_done >= n_inverted:
            break
        try:
            d = pickle.load(open(f"{pickle_dir}/{seq}.pickle", "rb"))
            mol = d["rd_mol"]
            bw = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            n_seen += 1
            if n_seen % 20 == 0:
                print(
                    f"  ...scanned {n_seen} peptides, {n_done} inverted found",
                    flush=True,
                )
            if _relaxed_dmmff_dominant(mol, bw, rng=rng) <= inv_thresh:
                continue  # not inverted
            idx = _backbone_idx(mol)
            dom_bb = _coords(mol, mol.GetConformers()[int(bw.argmax())].GetId(), idx)

            seed_mol = Chem.Mol(mol)
            seed_mol.RemoveAllConformers()
            seed_conformers(
                seed_mol,
                model,
                window=window,
                n_attempts=n_conf,
                tol_phi_psi=tol,
                tol_omega=tol,
            )
            n_seed_confs.append(seed_mol.GetNumConformers())
            pre, post = _both_min_rmsd(seed_mol, dom_bb, idx)
            seeded_pre.append(pre)
            seeded_post.append(post)

            base_mol = Chem.Mol(mol)
            base_mol.RemoveAllConformers()
            params = get_embed_params_macrocycle()
            params.randomSeed = 0
            rdDistGeom.EmbedMultipleConfs(base_mol, numConfs=n_conf, params=params)
            bpre, bpost = _both_min_rmsd(base_mol, dom_bb, idx)
            base_pre.append(bpre)
            base_post.append(bpost)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {seq}: {exc}")

    sp, spo = np.array(seeded_pre), np.array(seeded_post)
    bp, bpo = np.array(base_pre), np.array(base_post)
    print(f"\n=== seeding validation: {len(sp)} inverted test peptides ===")
    print(
        f"mean seeded confs/peptide: {np.mean(n_seed_confs):.1f} "
        f"({100*np.mean([c==0 for c in n_seed_confs]):.0f}% infeasible)"
    )
    print("\nmedian min BACKBONE RMSD-to-dominant (Å):")
    print(f"  UNRELAXED:  seeded {np.median(sp):.2f} | baseline {np.median(bp):.2f}")
    print(
        f"  MMFF-relaxed: seeded {np.median(spo):.2f} | baseline {np.median(bpo):.2f}"
    )
    print(
        "\nfraction reaching dominant (UNRELAXED seed — does prediction+DG reach the geometry):"
    )
    for thr in (0.5, 1.0, 1.5, 2.0):
        print(
            f"  within {thr} Å:  seeded {100*(sp<=thr).mean():.0f}% | baseline {100*(bp<=thr).mean():.0f}%"
        )
    print(
        "\nfraction reaching dominant (MMFF-relaxed — what survives the sampler's relax):"
    )
    for thr in (0.5, 1.0, 1.5, 2.0):
        print(
            f"  within {thr} Å:  seeded {100*(spo<=thr).mean():.0f}% | baseline {100*(bpo<=thr).mean():.0f}%"
        )
    print(
        f"\nseeded strictly closer than baseline: "
        f"unrelaxed {100*(sp<bp).mean():.0f}% | relaxed {100*(spo<bpo).mean():.0f}% of peptides"
    )


if __name__ == "__main__":
    main()
