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
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from torsional_sampling import (
    embed_constrained,
    get_backbone_dihedrals,
    set_dihedral_bounds,
)

from .model import DihedralPredictor
from .residues import (
    bin_to_center,
    neighbor_augment,
    omega_bin_to_center,
    omega_quads,
    residue_features,
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


@torch.no_grad()
def predict_dihedrals(
    mol: Chem.Mol, model: DihedralPredictor, window: int = 1, device: str = "cpu"
) -> tuple[list[float], list[float], list[float]]:
    """
    Predict per-residue (phi, psi, omega) target angles (degrees) for the dominant
    conformer, in get_backbone_dihedrals order.

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
    tol_phi_psi: float = 30.0,
    tol_omega: float = 30.0,
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
    defs = get_backbone_dihedrals(mol)
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
    tol_phi_psi: float = 30.0,
    tol_omega: float = 30.0,
    seed: int = 0,
    device: str = "cpu",
) -> list[int]:
    """
    Predict the dominant dihedrals and embed seed conformers via constrained DG.

    Conformers are added to `mol` in place. Returns [] if the predicted dihedral
    set is geometrically infeasible (free ring-closure rejection).

    Params:
        mol: Chem.Mol : peptide with explicit Hs (modified in place)
        model: DihedralPredictor : trained model
        window: int : neighbour augmentation half-width (must match training)
        n_attempts: int : ETKDGv3 embedding attempts
        tol_phi_psi: float : phi/psi constraint half-width (deg)
        tol_omega: float : omega constraint half-width (deg)
        seed: int : RNG seed
        device: str : torch device
    Returns:
        list of conformer IDs added to mol
    """
    phi, psi, omega = predict_dihedrals(mol, model, window=window, device=device)
    bounds = make_bounds_phi_psi_omega(mol, phi, psi, omega, tol_phi_psi, tol_omega)
    if bounds is None:
        return []
    return embed_constrained(mol, bounds, n_attempts=n_attempts, seed=seed)
