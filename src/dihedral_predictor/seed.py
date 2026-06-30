"""
Inference and constrained-DG seeding from predicted backbone dihedrals.

Loads a trained predictor, predicts the dominant conformer's per-residue
(phi, psi) bin centers and omega cis/trans, then builds a bounds matrix
constraining all three and embeds conformers via the existing constrained-DG
machinery. These seed conformers are added to the sampler pool so the MC walk
can start from (and retain) the otherwise MMFF-inaccessible dominant basin.

The existing `make_constrained_bounds` only constrains phi/psi; here we build the
bounds directly so omega (cis/trans) is constrained too — the Step-5b probe
showed omega matters for a meaningful minority of inverted peptides.

Tolerance note: the constraint half-width defaults to 60°, not the 30° used for
Pool-B Ramachandran sampling. Tight bounds make from-scratch distance geometry
thrash (measured: ~90 s/embed at 30° with omega vs ~2-5 s at 60°), and ±60° is
the right precision anyway — rotamer states sit ~120° apart, so ±60° windows just
separate them, and the prediction itself is only accurate to ~±15-22°. MMFF
relaxation downstream snaps each seed to the nearest basin minimum.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms

from torsional_sampling import embed_constrained, set_dihedral_bounds

from .model import ChiPredictor, DihedralPredictor
from .residues import (
    backbone_feature_block,
    bin_to_center,
    neighbor_augment,
    omega_bin_to_center,
    omega_quads,
    ordered_backbone_dihedrals,
    residue_features,
    sidechain_chi_quads,
)


def load_model(ckpt_path: str, device: str = "cpu") -> tuple[DihedralPredictor, int]:
    """
    Load a trained predictor from a checkpoint.

    Params:
        ckpt_path: str : checkpoint path from train()
        device: str : torch device string
    Returns:
        tuple (model in eval mode, neighbour window used at training time)
    """
    ck = torch.load(ckpt_path, map_location=device)
    model = DihedralPredictor(
        ck["in_features"], d_model=ck["d_model"], n_layers=ck["n_layers"]
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["window"]


def load_chi_model(ckpt_path: str, device: str = "cpu") -> tuple[ChiPredictor, int]:
    """
    Load the separate side-chain chi predictor from a checkpoint.

    Params:
        ckpt_path: str : checkpoint path from train_chi()
        device: str : torch device string
    Returns:
        tuple (chi model in eval mode, neighbour window used at training time)
    """
    ck = torch.load(ckpt_path, map_location=device)
    model = ChiPredictor(
        ck["in_features"], d_model=ck["d_model"], n_layers=ck["n_layers"]
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model.chi_cond = ck.get("chi_cond", False)  # backbone-conditioned?
    return model, ck["window"]


@torch.no_grad()
def predict_chi(
    mol: Chem.Mol,
    chi_model: ChiPredictor,
    window: int = 2,
    device: str = "cpu",
    backbone: "tuple | None" = None,
    sample: bool = False,
    temperature: float = 1.0,
) -> list[list[float]]:
    """
    Predict per-residue side-chain chi angles (degrees), one inner list per residue
    in ring order, each of length = that residue's chi count (sidechain_chi_quads).

    With `sample=False` (default) returns the argmax chi. With `sample=True` draws
    each chi bin from the model's softmax (temperature-scaled) — used to generate
    DIVERSE rotamer sets across seeds, so a downstream MACE relaxation can settle
    whichever rotamer set is closest to the true side-chain basin (the chi model is
    used as a rotamer *prior*, not a point estimate — refinement-based side chains).

    Params:
        mol: Chem.Mol : peptide with explicit Hs
        chi_model: ChiPredictor : trained chi model
        window: int : neighbour augmentation half-width (must match training)
        device: str : torch device
        backbone: tuple | None : (phi, psi, omega) degree lists (ring order) to
            condition on when the model is backbone-conditioned (chi_model.chi_cond)
        sample: bool : sample chi bins from the softmax instead of argmax
        temperature: float : softmax temperature for sampling (higher = more diverse)
    Returns:
        list (per residue) of chi target angles in degrees
    """
    feats = residue_features(mol)
    if getattr(chi_model, "chi_cond", False):
        if backbone is None:
            raise ValueError(
                "backbone-conditioned chi model needs backbone=(phi,psi,omega)"
            )
        phi, psi, omega = backbone
        omega_trans = [
            0 if abs(((o + 180.0) % 360.0) - 180.0) < 90.0 else 1 for o in omega
        ]
        feats = np.concatenate(
            [feats, backbone_feature_block(phi, psi, omega_trans)], axis=1
        )
    x = neighbor_augment(feats, window=window)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=device)
    logits = chi_model(xt, mask)[0]  # (n_res, MAX_CHI, bins)
    if sample:
        probs = torch.softmax(logits / temperature, dim=-1)
        n_res, n_chi, n_bins = probs.shape
        bins = torch.multinomial(probs.reshape(-1, n_bins), 1).reshape(n_res, n_chi)
    else:
        bins = logits.argmax(-1)  # (n_res, MAX_CHI)
    quads = sidechain_chi_quads(mol)
    return [
        [bin_to_center(int(bins[i, k])) for k in range(len(qs))]
        for i, qs in enumerate(quads)
    ]


def apply_chi(mol: Chem.Mol, conf_id: int, chi_angles: list[list[float]]) -> None:
    """
    Set predicted side-chain chi on one conformer via SetDihedralDeg.

    Params:
        mol: Chem.Mol : peptide with the conformer (smi-built; same chi quads as prediction)
        conf_id: int : conformer id to modify in place
        chi_angles: list[list[float]] : per-residue chi angles from predict_chi
    Returns:
        None
    """
    conf = mol.GetConformer(conf_id)
    for qs, angs in zip(sidechain_chi_quads(mol), chi_angles):
        for q, a in zip(qs, angs):
            rdMolTransforms.SetDihedralDeg(conf, *q, a)


@torch.no_grad()
def predict_dihedrals(
    mol: Chem.Mol, model: DihedralPredictor, window: int = 1, device: str = "cpu"
) -> tuple[list[float], list[float], list[float]]:
    """
    Predict per-residue (phi, psi, omega) target angles (degrees) for the dominant
    conformer, in ring order (matching residue_features / ordered_backbone_dihedrals).

    Params:
        mol: Chem.Mol : peptide with explicit Hs
        model: DihedralPredictor : trained model
        window: int : neighbour augmentation half-width (must match training)
        device: str : torch device
    Returns:
        tuple (phi, psi, omega) lists of target angles in degrees
    """
    x = neighbor_augment(residue_features(mol), window=window)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=device)
    pl, sl, ol = model(xt, mask)
    phi_b, psi_b, om_b = pl.argmax(-1)[0], sl.argmax(-1)[0], ol.argmax(-1)[0]
    phi = [bin_to_center(int(b)) for b in phi_b]
    psi = [bin_to_center(int(b)) for b in psi_b]
    omega = [omega_bin_to_center(int(b)) for b in om_b]
    return phi, psi, omega


def make_bounds_phi_psi_omega(
    mol: Chem.Mol,
    phi: list[float],
    psi: list[float],
    omega: list[float],
    tol_phi_psi: float = 60.0,
    tol_omega: float = 60.0,
) -> "np.ndarray | None":
    """
    Build a bounds matrix constraining phi, psi, and omega per residue.

    Params:
        mol: Chem.Mol : peptide with explicit Hs
        phi, psi, omega: list[float] : target angles (deg), per residue
        tol_phi_psi: float : phi/psi constraint half-width (deg)
        tol_omega: float : omega constraint half-width (deg)
    Returns:
        bounds matrix, or None if any constraint is infeasible
    """
    defs = ordered_backbone_dihedrals(mol)
    oq = omega_quads(mol)
    if not (len(defs) == len(phi) == len(psi) == len(omega) == len(oq)):
        raise ValueError("dihedral list lengths must match the residue count")
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    for (phi_atoms, psi_atoms), p, s in zip(defs, phi, psi):
        bounds = set_dihedral_bounds(bounds, *phi_atoms, p, tol_phi_psi)
        if bounds is None:
            return None
        bounds = set_dihedral_bounds(bounds, *psi_atoms, s, tol_phi_psi)
        if bounds is None:
            return None
    for quad, o in zip(oq, omega):
        bounds = set_dihedral_bounds(bounds, *quad, o, tol_omega)
        if bounds is None:
            return None
    return bounds


def seed_conformers(
    mol: Chem.Mol,
    model: DihedralPredictor,
    window: int = 1,
    n_attempts: int = 20,
    tol_phi_psi: float = 60.0,
    tol_omega: float = 60.0,
    seed: int = 0,
    device: str = "cpu",
    chi_model: "ChiPredictor | None" = None,
    chi_window: int = 2,
    chi_sample: bool = False,
    chi_temperature: float = 1.0,
) -> list[int]:
    """
    Predict the dominant dihedrals and embed seed conformers via constrained DG.
    Optionally also set predicted side-chain chi on each seed (separate chi model).

    Conformers are added to `mol` in place. Returns [] if the predicted dihedral
    set is geometrically infeasible (free ring-closure rejection).

    Params:
        mol: Chem.Mol : peptide with explicit Hs (modified in place)
        model: DihedralPredictor : trained backbone model
        window: int : backbone neighbour augmentation half-width (must match training)
        n_attempts: int : ETKDGv3 embedding attempts
        tol_phi_psi: float : phi/psi constraint half-width (deg)
        tol_omega: float : omega constraint half-width (deg)
        seed: int : RNG seed
        device: str : torch device
        chi_model: ChiPredictor | None : optional separate chi predictor; when given,
            its predicted chi are set on every embedded seed via SetDihedralDeg
            (side-chain bonds are not ring-closure-constrained, so this is safe)
        chi_window: int : neighbour window for the chi model (must match its training)
        chi_sample: bool : if True, sample a DIFFERENT rotamer set per seed from the
            chi model's softmax (chi as a prior, for refinement-based side chains);
            if False, apply the single argmax chi to every seed
        chi_temperature: float : softmax temperature for chi sampling
    Returns:
        list of conformer IDs added to mol
    """
    phi, psi, omega = predict_dihedrals(mol, model, window=window, device=device)
    bounds = make_bounds_phi_psi_omega(mol, phi, psi, omega, tol_phi_psi, tol_omega)
    if bounds is None:
        return []
    conf_ids = embed_constrained(mol, bounds, n_attempts=n_attempts, seed=seed)
    if chi_model is not None and conf_ids:
        bb = (phi, psi, omega)
        if chi_sample:
            for cid in conf_ids:  # a fresh sampled rotamer set per seed
                apply_chi(
                    mol,
                    cid,
                    predict_chi(
                        mol,
                        chi_model,
                        window=chi_window,
                        device=device,
                        backbone=bb,
                        sample=True,
                        temperature=chi_temperature,
                    ),
                )
        else:
            chi_angles = predict_chi(
                mol, chi_model, window=chi_window, device=device, backbone=bb
            )
            for cid in conf_ids:
                apply_chi(mol, cid, chi_angles)
    return conf_ids
