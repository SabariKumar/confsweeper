"""Add the backbone model's PREDICTED phi/psi/omega bins to the dataset (issue #20).

Conditioning the chi model on TRUE backbone helps offline (chi_peptide_ok 0.12 -> 0.26)
but barely at inference (0.145) — a teacher-forcing gap, because at seeding the chi model
sees the backbone model's *predicted* backbone. This precomputes those predictions for
every peptide so the chi model can be trained on predicted backbone (train == inference).

Writes a sibling dataset with phi_bin_pred / psi_bin_pred / omega_bin_pred added.
"""

import pickle
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.data import load_dataset  # noqa: E402
from dihedral_predictor.residues import neighbor_augment  # noqa: E402
from dihedral_predictor.seed import load_model  # noqa: E402


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--ckpt", default="data/processed/dihedral_predictor/ckpt_w2_d256_l6.pt")
@click.option("--out", default="data/processed/dihedral_predictor/dataset_pred.pkl")
@click.option("--device", default="cuda")
def main(dataset, ckpt, out, device):
    data = load_dataset(dataset)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, window = load_model(ckpt, device=str(dev))
    feats = data["feats"]
    phi_p, psi_p, om_p = [], [], []
    with torch.no_grad():
        for f in feats:
            x = (
                torch.from_numpy(neighbor_augment(f, window=window))
                .unsqueeze(0)
                .to(dev)
            )
            mask = torch.ones(1, f.shape[0], dtype=torch.bool, device=dev)
            pl, sl, ol = model(x, mask)
            phi_p.append(pl.argmax(-1)[0].cpu().numpy().astype(np.int64))
            psi_p.append(sl.argmax(-1)[0].cpu().numpy().astype(np.int64))
            om_p.append(ol.argmax(-1)[0].cpu().numpy().astype(np.int64))
    new = dict(data)
    new["phi_bin_pred"] = phi_p
    new["psi_bin_pred"] = psi_p
    new["omega_bin_pred"] = om_p
    new["backbone_ckpt"] = ckpt
    with open(out, "wb") as fh:
        pickle.dump(new, fh)
    print(f"added predicted backbone for {len(feats)} peptides -> {out}")


if __name__ == "__main__":
    main()
