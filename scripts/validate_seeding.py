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
from dihedral_predictor.seed import load_model, seed_conformers  # noqa: E402
from mcmm import _kabsch_rmsd_pairwise  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _heavy_coords(mol, conf_id):
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return mol.GetConformer(conf_id).GetPositions()[idx]


def _relaxed_min_rmsd(mol_confs, dom_heavy):
    """MMFF-relax all confs of mol_confs, return min heavy-atom RMSD to dom_heavy."""
    if mol_confs.GetNumConformers() == 0:
        return np.inf
    AllChem.MMFFOptimizeMoleculeConfs(mol_confs, maxIters=500, numThreads=0)
    idx = [a.GetIdx() for a in mol_confs.GetAtoms() if a.GetAtomicNum() > 1]
    refs = np.stack(
        [c.GetPositions()[idx] for c in mol_confs.GetConformers()]
    )  # (K, n, 3)
    rmsds = _kabsch_rmsd_pairwise(torch.from_numpy(dom_heavy), torch.from_numpy(refs))
    return float(rmsds.min())


def _relaxed_dmmff_dominant(mol, bw):
    m = Chem.Mol(mol)
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=1000, numThreads=0)
    props = AllChem.MMFFGetMoleculeProperties(m)
    e = np.array(
        [
            AllChem.MMFFGetMoleculeForceField(m, props, confId=c.GetId()).CalcEnergy()
            for c in m.GetConformers()
        ]
    )
    return float(e[int(bw.argmax())] - e.min())


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--ckpt", default="data/processed/dihedral_predictor/checkpoint.pt")
@click.option("--pickle_dir", default="data/raw/cremp/pickle")
@click.option("--n_inverted", default=120, help="target #inverted peptides to test")
@click.option("--n_conf", default=24, help="seeded attempts / baseline confs")
@click.option("--inv_thresh", default=2.0)
def main(dataset, ckpt, pickle_dir, n_inverted, n_conf, inv_thresh):
    data = load_dataset(dataset)
    test_seqs = [s for s, sp in zip(data["seqs"], data["split"]) if sp == "test"]
    rng = np.random.default_rng(0)
    rng.shuffle(test_seqs)
    model, window = load_model(ckpt, device="cpu")

    seeded_rmsd, base_rmsd, n_seed_confs = [], [], []
    n_done = 0
    for seq in test_seqs:
        if n_done >= n_inverted:
            break
        try:
            d = pickle.load(open(f"{pickle_dir}/{seq}.pickle", "rb"))
            mol = d["rd_mol"]
            bw = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            if _relaxed_dmmff_dominant(mol, bw) <= inv_thresh:
                continue  # not inverted
            dom_heavy = _heavy_coords(
                mol, mol.GetConformers()[int(bw.argmax())].GetId()
            )

            seed_mol = Chem.Mol(mol)
            seed_mol.RemoveAllConformers()
            seed_conformers(seed_mol, model, window=window, n_attempts=n_conf)
            n_seed_confs.append(seed_mol.GetNumConformers())
            seeded_rmsd.append(_relaxed_min_rmsd(seed_mol, dom_heavy))

            base_mol = Chem.Mol(mol)
            base_mol.RemoveAllConformers()
            params = get_embed_params_macrocycle()
            params.randomSeed = 0
            rdDistGeom.EmbedMultipleConfs(base_mol, numConfs=n_conf, params=params)
            base_rmsd.append(_relaxed_min_rmsd(base_mol, dom_heavy))
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {seq}: {exc}")

    s = np.array(seeded_rmsd)
    b = np.array(base_rmsd)
    print(f"\n=== seeding validation: {len(s)} inverted test peptides ===")
    print(
        f"min RMSD-to-dominant (Å):  seeded median {np.median(s):.2f} | baseline median {np.median(b):.2f}"
    )
    print(
        f"mean seeded confs/peptide: {np.mean(n_seed_confs):.1f} "
        f"({100*np.mean([c==0 for c in n_seed_confs]):.0f}% infeasible)"
    )
    for thr in (0.5, 1.0, 1.5, 2.0):
        print(
            f"  reached dominant within {thr} Å:  seeded {100*(s<=thr).mean():.0f}% | "
            f"baseline {100*(b<=thr).mean():.0f}%"
        )
    win = (s < b).mean()
    print(f"seeded strictly closer than baseline: {100*win:.0f}% of peptides")


if __name__ == "__main__":
    main()
