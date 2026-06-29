"""
Training loop and evaluation for the dihedral predictor.

Loss is masked cross-entropy over phi/psi/omega bins (summed). Metrics report
per-residue exact-bin and within-1-bin accuracy (within-1-bin ≈ ±22°, inside the
constrained-DG seeding tolerance), plus the key seeding proxy: the fraction of
peptides whose *every* backbone dihedral is predicted within tolerance (phi & psi
within 1 bin and omega correct) — i.e. peptides the predictor could seed end-to-end.
A majority baseline (most-common phi/psi bin, all-trans omega) is reported for context.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DihedralDataset, collate, load_dataset
from .model import DihedralPredictor
from .residues import PHI_PSI_BINS


def masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over real (unmasked) residues."""
    b, n_len, c = logits.shape
    loss = F.cross_entropy(
        logits.reshape(-1, c), target.reshape(-1), reduction="none"
    ).reshape(b, n_len)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _bin_dist(pred: torch.Tensor, target: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Circular distance (in bins) between predicted and target bin indices."""
    d = (pred - target).abs()
    return torch.minimum(d, n_bins - d)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Compute accuracy metrics over a loader.

    Params:
        model: DihedralPredictor : the model (or None for majority baseline)
        loader: DataLoader : eval data
        device: torch.device : compute device
    Returns:
        dict of metrics
    """
    if model is not None:
        model.eval()
    n_res = 0
    n_pep = 0
    phi_exact = psi_exact = om_acc = 0
    phi_w1 = psi_w1 = 0
    pep_ok = 0
    for b in loader:
        x, mask = b["x"].to(device), b["mask"].to(device)
        phi_t, psi_t, om_t = (
            b["phi"].to(device),
            b["psi"].to(device),
            b["omega"].to(device),
        )
        if model is not None:
            pl, sl, ol = model(x, mask)
            phi_p, psi_p, om_p = pl.argmax(-1), sl.argmax(-1), ol.argmax(-1)
        else:  # majority baseline: most-common bin (computed below), all-trans omega
            phi_p = torch.full_like(phi_t, loader.majority_phi)
            psi_p = torch.full_like(psi_t, loader.majority_psi)
            om_p = torch.ones_like(om_t)  # trans
        m = mask
        pe = (phi_p == phi_t) & m
        se = (psi_p == psi_t) & m
        oa = (om_p == om_t) & m
        pw = (_bin_dist(phi_p, phi_t, PHI_PSI_BINS) <= 1) & m
        sw = (_bin_dist(psi_p, psi_t, PHI_PSI_BINS) <= 1) & m
        phi_exact += pe.sum().item()
        psi_exact += se.sum().item()
        om_acc += oa.sum().item()
        phi_w1 += pw.sum().item()
        psi_w1 += sw.sum().item()
        n_res += m.sum().item()
        # per-peptide: all real residues within tolerance (phi&psi within 1 bin, omega correct)
        per_res_ok = pw & sw & oa | ~m  # masked positions count as ok
        pep_ok += per_res_ok.all(dim=1).sum().item()
        n_pep += x.shape[0]
    return {
        "phi_exact": phi_exact / n_res,
        "psi_exact": psi_exact / n_res,
        "omega_acc": om_acc / n_res,
        "phi_within1": phi_w1 / n_res,
        "psi_within1": psi_w1 / n_res,
        "peptide_all_ok": pep_ok / n_pep,
    }


def _majority_bins(train_ds) -> tuple[int, int]:
    """Most-common phi and psi bin over the training set (for the baseline)."""
    phi = torch.cat([torch.from_numpy(p) for p in train_ds.phi])
    psi = torch.cat([torch.from_numpy(p) for p in train_ds.psi])
    return int(phi.bincount(minlength=PHI_PSI_BINS).argmax()), int(
        psi.bincount(minlength=PHI_PSI_BINS).argmax()
    )


def train(
    dataset_path: str,
    out_ckpt: str,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_layers: int = 3,
    window: int = 1,
    device: str = "cuda",
) -> dict:
    """
    Train the dihedral predictor and checkpoint the best (by peptide_all_ok) model.

    Params:
        dataset_path: str : dataset pickle from build_dataset
        out_ckpt: str : output checkpoint path
        epochs: int : training epochs
        batch_size: int : batch size
        lr: float : Adam learning rate
        d_model: int : model width
        n_layers: int : transformer layers
        window: int : neighbour augmentation half-width
        device: str : 'cuda' or 'cpu'
    Returns:
        dict of best validation metrics
    """
    import os

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    data = load_dataset(dataset_path)
    tr = DihedralDataset(data, "train", window=window)
    va = DihedralDataset(data, "val", window=window)
    in_features = tr[0]["x"].shape[1]

    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=batch_size, collate_fn=collate)
    mphi, mpsi = _majority_bins(tr)
    vl.majority_phi, vl.majority_psi = mphi, mpsi
    print(
        "majority baseline (val):",
        {k: round(v, 3) for k, v in evaluate(None, vl, dev).items()},
    )

    model = DihedralPredictor(in_features, d_model=d_model, n_layers=n_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = {"peptide_all_ok": -1.0}
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in tl:
            x, mask = b["x"].to(dev), b["mask"].to(dev)
            pl, sl, ol = model(x, mask)
            loss = (
                masked_ce(pl, b["phi"].to(dev), mask)
                + masked_ce(sl, b["psi"].to(dev), mask)
                + masked_ce(ol, b["omega"].to(dev), mask)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        m = evaluate(model, vl, dev)
        flag = ""
        if m["peptide_all_ok"] > best["peptide_all_ok"]:
            best = m
            os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_features": in_features,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "window": window,
                    "metrics": m,
                },
                out_ckpt,
            )
            flag = " *"
        if ep % 2 == 0 or flag:
            print(
                f"ep{ep:3d} loss {tot/len(tl):.3f} | "
                f"phi {m['phi_within1']:.2f} psi {m['psi_within1']:.2f} "
                f"om {m['omega_acc']:.2f} | pep_ok {m['peptide_all_ok']:.3f}{flag}"
            )
    print("best val:", {k: round(v, 3) for k, v in best.items()})
    return best
