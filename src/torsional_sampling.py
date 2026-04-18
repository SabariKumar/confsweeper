"""
Backbone dihedral-constrained conformer generation for macrocyclic peptides.

Provides tools to inject target (phi, psi) backbone dihedral angles as 1,4-distance
constraints into an RDKit bounds matrix, then embed conformers satisfying those
constraints via ETKDGv3.  Intended as Phase 2 of the confsweeper pipeline:

    Phase 1 (nvmolkit ETKDG) → pool A   (force-field-favored conformers)
    Phase 2 (constrained DG) → pool B   (dihedral-targeted conformers)
    Merge A + B → GPU Butina → MACE scoring

The key idea is that ETKDGv3's distance geometry can be guided to embed conformers
in specific (phi, psi) regions by tightening the 1,4-distance bounds for the
relevant backbone atoms.  Ring-closure failures are free: ETKDGv3 returns no
conformer for geometrically impossible combinations.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdDistGeom import ETKDGv3

logger = logging.getLogger(__name__)

# SMARTS: matches one residue backbone unit and its flanking amide bonds.
# Tuple layout: (C_prev, O, N, Ca, C, O, N_next)
_BACKBONE_SMARTS = Chem.MolFromSmarts("[C:1](=O)[N:2][CX4:3][C:4](=O)[N:5]")

_RESIDUE_CLASSES = ("L", "D", "NMe", "Gly")


# ---------------------------------------------------------------------------
# Dihedral → 1,4-distance bounds
# ---------------------------------------------------------------------------


def _d14(
    r12: float,
    r23: float,
    r34: float,
    cos_t123: float,
    sin_t123: float,
    cos_t234: float,
    sin_t234: float,
    phi_rad: float,
) -> float:
    """
    1,4-distance for a four-atom chain A-B-C-D given bond lengths, bond angles,
    and dihedral angle phi.

    Derived from the law of cosines applied iteratively through the chain:
        d14² = r12² + r23² + r34²
               - 2·r12·r23·cos(θ_ABC)
               - 2·r23·r34·cos(θ_BCD)
               + 2·r12·r34·(cos(θ_ABC)·cos(θ_BCD) - sin(θ_ABC)·sin(θ_BCD)·cos(φ))
    """
    d2 = (
        r12**2
        + r23**2
        + r34**2
        - 2 * r12 * r23 * cos_t123
        - 2 * r23 * r34 * cos_t234
        + 2 * r12 * r34 * (cos_t123 * cos_t234 - sin_t123 * sin_t234 * np.cos(phi_rad))
    )
    return float(np.sqrt(max(d2, 0.0)))


def set_dihedral_bounds(
    bounds_mat: np.ndarray,
    i: int,
    j: int,
    k: int,
    l: int,
    angle_deg: float,
    tolerance_deg: float = 30.0,
) -> Optional[np.ndarray]:
    """
    Return a copy of bounds_mat with the 1,4-distance entry for atoms (i, l)
    tightened to be consistent with dihedral (i,j,k,l) = angle_deg ± tolerance_deg.

    Uses the four-atom 1,4-distance formula, deriving bond lengths and bond angles
    from the existing bounds matrix (midpoints of the lower/upper ranges).

    Returns None if the computed range is geometrically inconsistent (lower > upper
    after intersecting with the existing bounds) — the caller should discard
    this (phi, psi) sample.

    Params:
        bounds_mat    : RDKit bounds matrix (upper in upper triangle, lower in lower)
        i, j, k, l   : atom indices of the four-atom dihedral
        angle_deg     : target dihedral angle in degrees
        tolerance_deg : half-width of the allowed range (default 30°)
    Returns:
        Modified copy of bounds_mat, or None if the constraint is infeasible.
    """
    n = bounds_mat.shape[0]
    assert all(0 <= idx < n for idx in (i, j, k, l)), "Atom index out of range"

    def _mid(a, b):
        """Midpoint of the lower/upper bounds for pair (a, b)."""
        lo, hi = (
            (bounds_mat[b, a], bounds_mat[a, b])
            if a < b
            else (bounds_mat[a, b], bounds_mat[b, a])
        )
        return (lo + hi) / 2.0

    # Bond lengths and 1,3 distances from existing bounds (midpoints)
    r12 = _mid(i, j)
    r23 = _mid(j, k)
    r34 = _mid(k, l)
    r13 = _mid(i, k)  # 1,3 distance spanning bond angle at j
    r24 = _mid(j, l)  # 1,3 distance spanning bond angle at k

    # Bond angles at j and k via law of cosines on the 1,3 distances
    cos_t123 = np.clip((r12**2 + r23**2 - r13**2) / (2 * r12 * r23), -1.0, 1.0)
    cos_t234 = np.clip((r23**2 + r34**2 - r24**2) / (2 * r23 * r34), -1.0, 1.0)
    sin_t123 = float(np.sqrt(1.0 - cos_t123**2))
    sin_t234 = float(np.sqrt(1.0 - cos_t234**2))

    # d14(phi) = sqrt(A - B*cos(phi)), which is monotone in cos(phi).
    # The extrema over [phi_center - tol, phi_center + tol] occur at the two
    # endpoints plus at phi=0 and phi=±pi if they fall inside the range.
    phi_c = float(np.deg2rad(angle_deg))
    phi_t = float(np.deg2rad(tolerance_deg))

    candidate_phis = [phi_c - phi_t, phi_c + phi_t]
    # Include cos(phi) extrema (0 and ±pi) if they fall within the range
    for extremum in [0.0, np.pi, -np.pi]:
        # Normalise extremum to be within 2pi of phi_c
        diff = (extremum - phi_c + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) <= phi_t:
            candidate_phis.append(phi_c + diff)

    d14_vals = [
        _d14(r12, r23, r34, cos_t123, sin_t123, cos_t234, sin_t234, p)
        for p in candidate_phis
    ]
    d14_min = min(d14_vals)
    d14_max = max(d14_vals)

    # Tighten existing bounds: take intersection
    a, b = (i, l) if i < l else (l, i)  # ensure a < b for upper-triangle access
    existing_upper = bounds_mat[a, b]
    existing_lower = bounds_mat[b, a]

    new_upper = min(existing_upper, d14_max)
    new_lower = max(existing_lower, d14_min)

    if new_lower > new_upper + 1e-6:
        logger.debug(
            "Dihedral constraint i=%d j=%d k=%d l=%d phi=%.1f° is infeasible: "
            "[%.3f, %.3f] does not intersect existing [%.3f, %.3f]",
            i,
            j,
            k,
            l,
            angle_deg,
            d14_min,
            d14_max,
            existing_lower,
            existing_upper,
        )
        return None

    new_bounds = bounds_mat.copy()
    new_bounds[a, b] = new_upper
    new_bounds[b, a] = new_lower
    return new_bounds


# ---------------------------------------------------------------------------
# Residue classification
# ---------------------------------------------------------------------------


def _h_neighbors(atom) -> int:
    """Count explicit H-atom neighbours (CREMP mols store Hs as explicit atoms)."""
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)


def _classify_residue(mol: Chem.Mol, n_idx: int, ca_idx: int) -> str:
    """
    Classify a single backbone residue as L, D, NMe, or Gly.

    Priority order: NMe > Gly > D > L.

    Requires that Chem.AssignStereochemistry has already been called on mol.
    Works with both implicit-H and explicit-H molecules; the NMe and Gly checks
    use explicit H counting, which returns 0 for implicit-H mols — in that case
    NMe and Gly classification falls back to L/D only.
    """
    n_atom = mol.GetAtomWithIdx(n_idx)
    ca_atom = mol.GetAtomWithIdx(ca_idx)

    # NMe: amide N carries a methyl carbon (3 explicit H-neighbours, 1 heavy-atom
    # neighbour other than Cα). GetDegree() counts H-bonds when Hs are explicit,
    # so we count heavy-atom neighbours explicitly to avoid false negatives.
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 6 and nb.GetIdx() != ca_idx:
            heavy_nbs = sum(1 for nb2 in nb.GetNeighbors() if nb2.GetAtomicNum() != 1)
            if _h_neighbors(nb) == 3 and heavy_nbs == 1:
                return "NMe"

    # Gly: alpha carbon carries 2 explicit H-atom neighbours (no side chain).
    if _h_neighbors(ca_atom) >= 2:
        return "Gly"

    # D vs L via CIP stereo label on Cα.
    cip = ca_atom.GetPropsAsDict().get("_CIPCode", "")
    return "D" if cip == "R" else "L"


def classify_backbone_residues(mol: Chem.Mol) -> list[str]:
    """
    Return the residue class (L, D, NMe, or Gly) for each backbone residue,
    in the same order as get_backbone_dihedrals().

    Calls AssignStereochemistry internally so the mol does not need to be
    pre-processed by the caller.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    classes = []
    seen_n: set[int] = set()
    for match in mol.GetSubstructMatches(_BACKBONE_SMARTS):
        _, _, n, ca, *_ = match
        if n in seen_n:
            continue
        seen_n.add(n)
        classes.append(_classify_residue(mol, n, ca))
    return classes


# ---------------------------------------------------------------------------
# Backbone dihedral definitions
# ---------------------------------------------------------------------------


def get_backbone_dihedrals(mol: Chem.Mol) -> list[tuple]:
    """
    Return one (phi_atoms, psi_atoms) tuple per backbone residue.

    phi_atoms = (C_prev, N, Ca, C)
    psi_atoms = (N, Ca, C, N_next)

    Each is a 4-tuple of atom indices suitable for passing to set_dihedral_bounds
    or rdMolTransforms.GetDihedralDeg.
    """
    defs = []
    seen_n = set()
    for match in mol.GetSubstructMatches(_BACKBONE_SMARTS):
        c_prev, _, n, ca, c, _, n_next = match
        if n in seen_n:
            continue
        seen_n.add(n)
        defs.append(((c_prev, n, ca, c), (n, ca, c, n_next)))
    return defs


# ---------------------------------------------------------------------------
# Constrained embedding
# ---------------------------------------------------------------------------


def make_constrained_bounds(
    mol: Chem.Mol,
    phi_psi_values: list[tuple[float, float]],
    tolerance_deg: float = 30.0,
) -> Optional[np.ndarray]:
    """
    Build a bounds matrix for mol with backbone dihedral constraints applied.

    phi_psi_values: list of (phi_deg, psi_deg) pairs, one per backbone residue
    in the order returned by get_backbone_dihedrals().

    Returns the modified bounds matrix, or None if any constraint is infeasible.
    """
    defs = get_backbone_dihedrals(mol)
    if len(phi_psi_values) != len(defs):
        raise ValueError(
            f"phi_psi_values has {len(phi_psi_values)} entries "
            f"but mol has {len(defs)} backbone residues"
        )

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)

    for (phi_atoms, psi_atoms), (phi_deg, psi_deg) in zip(defs, phi_psi_values):
        bounds = set_dihedral_bounds(bounds, *phi_atoms, phi_deg, tolerance_deg)
        if bounds is None:
            return None
        bounds = set_dihedral_bounds(bounds, *psi_atoms, psi_deg, tolerance_deg)
        if bounds is None:
            return None

    return bounds


def embed_constrained(
    mol: Chem.Mol,
    bounds: np.ndarray,
    n_attempts: int = 10,
    seed: int = 0,
) -> list[int]:
    """
    Embed conformers for mol using a custom bounds matrix (CPU ETKDGv3).

    CPU embedding is used here — not nvmolkit — because per-conformer bounds
    constraints require separate EmbedParameters objects that cannot be batched
    into a single nvmolkit call.  The constrained DG phase typically generates
    one conformer per (phi, psi) sample, so GPU batching provides little benefit
    at this step; GPU time is preserved for Butina clustering and MACE scoring.

    Returns list of conformer IDs successfully embedded.
    """
    params = ETKDGv3()
    params.useRandomCoords = True
    params.useMacrocycleTorsions = True
    params.useMacrocycle14config = True
    params.SetBoundsMat(bounds)
    params.randomSeed = seed

    # EmbedMolecule clears all existing conformers before embedding (it calls
    # EmbedMultipleConfs internally).  To accumulate conformers across attempts
    # without erasing previous ones, embed into a disposable copy and transfer.
    conf_ids = []
    for attempt in range(n_attempts):
        params.randomSeed = seed + attempt
        tmp = Chem.RWMol(mol)
        for cid in [c.GetId() for c in tmp.GetConformers()]:
            tmp.RemoveConformer(cid)
        tmp_cid = rdDistGeom.EmbedMolecule(tmp, params)
        if tmp_cid >= 0:
            new_cid = mol.AddConformer(tmp.GetConformer(tmp_cid), assignId=True)
            conf_ids.append(new_cid)
    return conf_ids


# ---------------------------------------------------------------------------
# Ramachandran-prior sampling
# ---------------------------------------------------------------------------


def load_ramachandran_grids(
    path: Union[str, Path] = "data/processed/cremp/ramachandran_grids.npz",
) -> dict:
    """
    Load the CREMP Ramachandran grids from an .npz file.

    Returns a dict with keys: 'L', 'D', 'NMe', 'Gly' (each (36, 36) float64),
    'bin_centers' ((36,) degrees), 'bin_edges' ((37,) degrees), 'n_bins'.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _sample_from_grid(
    grid: np.ndarray,
    bin_centers: np.ndarray,
    n_samples: int,
    strategy: str,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """
    Draw n_samples (phi_deg, psi_deg) pairs from a 2D Ramachandran grid.

    strategy='uniform'  — equal weight for every non-zero cell (samples from the
                          full CREMP-accessible region without probability bias)
    strategy='inverse'  — weight ∝ 1/p, so rare-but-accessible cells are
                          oversampled relative to the CREMP distribution, targeting
                          the gaps that standard ETKDG tends to miss
    """
    nonzero_idx = np.argwhere(grid > 0)  # (K, 2)

    if len(nonzero_idx) == 0:
        # No data for this residue class — fall back to uniform over all cells
        nonzero_idx = np.argwhere(np.ones_like(grid, dtype=bool))
        weights = None
    elif strategy == "uniform":
        weights = None
    elif strategy == "inverse":
        probs = np.array([grid[i, j] for i, j in nonzero_idx], dtype=np.float64)
        weights = 1.0 / probs
        weights /= weights.sum()
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose 'uniform' or 'inverse'"
        )

    chosen = rng.choice(len(nonzero_idx), size=n_samples, replace=True, p=weights)
    return [
        (float(bin_centers[nonzero_idx[c, 0]]), float(bin_centers[nonzero_idx[c, 1]]))
        for c in chosen
    ]


def sample_constrained_confs(
    mol: Chem.Mol,
    grids: dict,
    n_samples: int,
    n_attempts: int = 5,
    tolerance_deg: float = 30.0,
    strategy: str = "uniform",
    seed: int = 0,
) -> list[int]:
    """
    Generate Pool B conformers by sampling (phi, psi) targets from the CREMP
    Ramachandran prior and embedding each with constrained DG (CPU ETKDGv3).

    For each of n_samples draws:
      1. Each backbone residue gets an independent (phi, psi) sampled from the
         CREMP grid for its residue class (L/D/NMe/Gly).
      2. make_constrained_bounds builds a bounds matrix for that assignment.
      3. embed_constrained tries up to n_attempts embeddings.

    Infeasible samples (ring-closure failure) are silently skipped — the caller
    receives whatever conformers were successfully embedded.

    Params:
        mol           : RDKit mol with explicit Hs (modified in-place with new confs)
        grids         : dict from load_ramachandran_grids()
        n_samples     : number of (phi, psi) draws to attempt
        n_attempts    : ETKDGv3 attempts per draw (default 5)
        tolerance_deg : dihedral constraint half-width in degrees (default 30°)
        strategy      : 'uniform' or 'inverse' (see _sample_from_grid)
        seed          : random seed for reproducibility

    Returns:
        List of conformer IDs added to mol.
    """
    residue_classes = classify_backbone_residues(mol)
    if not residue_classes:
        logger.warning("No backbone residues found; returning empty conformer list")
        return []

    bin_centers = grids["bin_centers"]
    rng = np.random.default_rng(seed)

    # Pre-draw all (phi, psi) targets: shape (n_samples, n_residues, 2)
    # Each residue draws independently from its own class grid.
    per_residue_draws: list[list[tuple[float, float]]] = []
    for cls in residue_classes:
        grid = grids.get(cls, grids["L"])  # fall back to L if class missing
        draws = _sample_from_grid(grid, bin_centers, n_samples, strategy, rng)
        per_residue_draws.append(draws)

    # Transpose: per_residue_draws[residue][sample] → targets[sample][residue]
    targets = [
        [(per_residue_draws[r][s]) for r in range(len(residue_classes))]
        for s in range(n_samples)
    ]

    all_conf_ids: list[int] = []
    n_feasible = 0
    for s, phi_psi_values in enumerate(targets):
        bounds = make_constrained_bounds(mol, phi_psi_values, tolerance_deg)
        if bounds is None:
            continue
        n_feasible += 1
        conf_ids = embed_constrained(mol, bounds, n_attempts=n_attempts, seed=seed + s)
        all_conf_ids.extend(conf_ids)

    logger.debug(
        "sample_constrained_confs: %d/%d samples feasible, %d conformers embedded",
        n_feasible,
        n_samples,
        len(all_conf_ids),
    )
    return all_conf_ids
