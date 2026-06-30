"""
Per-residue feature and dihedral-target extraction for the dihedral predictor.

Everything is derived from the RDKit mol in `get_backbone_dihedrals` order, so
per-residue input features and (phi, psi, omega) targets are aligned by
construction — no fragile sequence-string ↔ atom-index matching. Features are
mol-derived physico-chemical descriptors (residue class + side-chain chemistry),
which also generalise to non-standard residues, plus cyclic ±window neighbour
descriptors (neighbouring residues strongly drive backbone dihedral propensities:
pre-proline effects, flanking N-methylation, etc.).

Dihedral targets are binned for classification: phi/psi into `PHI_PSI_BINS`
equal-width circular bins over (-180, 180], omega into 2 bins (cis / trans).
Classification (rather than regression) is robust to the multimodal, circular
nature of backbone dihedrals and aligns with the ±tolerance window used by the
downstream constrained-DG seeding.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from torsional_sampling import get_backbone_dihedrals

# --- binning -------------------------------------------------------------

PHI_PSI_BINS = 24  # 15° bins over (-180, 180]
OMEGA_BINS = 2  # 0 = cis (center 0°), 1 = trans (center 180°)
OMEGA_CIS_CUTOFF = 90.0  # |omega| < cutoff → cis


def angle_to_bin(angle_deg: float, n_bins: int = PHI_PSI_BINS) -> int:
    """
    Map a circular angle in degrees to a bin index in [0, n_bins).

    Params:
        angle_deg: float : angle in degrees (any range; wrapped to (-180, 180])
        n_bins: int : number of equal-width circular bins
    Returns:
        int bin index
    """
    width = 360.0 / n_bins
    return int(((angle_deg + 180.0) % 360.0) // width) % n_bins


def bin_to_center(bin_idx: int, n_bins: int = PHI_PSI_BINS) -> float:
    """
    Return the center angle (degrees) of a circular bin.

    Params:
        bin_idx: int : bin index in [0, n_bins)
        n_bins: int : number of equal-width circular bins
    Returns:
        float center angle in (-180, 180]
    """
    width = 360.0 / n_bins
    return -180.0 + (bin_idx + 0.5) * width


def omega_to_bin(omega_deg: float) -> int:
    """
    Classify an omega dihedral as cis (0) or trans (1).

    Params:
        omega_deg: float : omega dihedral in degrees
    Returns:
        int : 0 if cis (|omega| < OMEGA_CIS_CUTOFF), else 1
    """
    return 0 if abs(((omega_deg + 180.0) % 360.0) - 180.0) < OMEGA_CIS_CUTOFF else 1


def omega_bin_to_center(bin_idx: int) -> float:
    """
    Return the representative omega angle (degrees) for a cis/trans bin.

    Params:
        bin_idx: int : 0 (cis) or 1 (trans)
    Returns:
        float : 0.0 for cis, 180.0 for trans
    """
    return 0.0 if bin_idx == 0 else 180.0


# --- backbone atom bookkeeping ------------------------------------------


def ordered_backbone_dihedrals(mol: Chem.Mol) -> list[tuple]:
    """
    Return get_backbone_dihedrals defs reordered into ring-connectivity order.

    `get_backbone_dihedrals` returns residues in SMARTS-match (atom-index) order,
    which is NOT ring-sequential for many macrocycles (notably cyclic
    tetrapeptides). Ring order matters here: omega links consecutive residues and
    the cyclic neighbour augmentation assumes adjacency. Each residue's psi tuple
    carries n_next (the next residue's amide N), so we chain residues by matching
    n_next → the residue whose N equals it. Falls back to the original order if the
    chain cannot be completed (atypical connectivity).

    Params:
        mol: Chem.Mol : peptide with explicit Hs
    Returns:
        list of (phi_atoms, psi_atoms) tuples in ring order
    """
    defs = get_backbone_dihedrals(mol)
    n_to_res = {psi[0]: i for i, (_phi, psi) in enumerate(defs)}
    n_next = [psi[3] for (_phi, psi) in defs]
    order, seen, cur = [0], {0}, 0
    for _ in range(len(defs) - 1):
        nxt = n_to_res.get(n_next[cur])
        if nxt is None or nxt in seen:
            return defs  # fall back to original order
        order.append(nxt)
        seen.add(nxt)
        cur = nxt
    return [defs[i] for i in order]


def residue_atoms(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """
    Return (N, Ca, C) backbone atom indices per residue, in ring order.

    Params:
        mol: Chem.Mol : peptide with explicit Hs
    Returns:
        list of (n_idx, ca_idx, c_idx) tuples
    """
    out = []
    for _phi, psi in ordered_backbone_dihedrals(mol):
        n, ca, c, _n_next = psi
        out.append((n, ca, c))
    return out


def omega_quads(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """
    Return omega dihedral atom-quads (Ca_i, C_i, N_{i+1}, Ca_{i+1}) per residue,
    aligned with get_backbone_dihedrals order (cyclic: last links to first).

    Built from consecutive residues' atoms so it is aligned by construction.

    Params:
        mol: Chem.Mol : peptide with explicit Hs
    Returns:
        list of 4-tuples of atom indices, one per residue
    """
    defs = ordered_backbone_dihedrals(mol)
    n_res = len(defs)
    quads = []
    for i in range(n_res):
        # residue i: psi = (N_i, Ca_i, C_i, N_{i+1})
        _n_i, ca_i, c_i, _n_next = defs[i][1]
        # next residue's phi = (C_prev=c_i, N_{i+1}, Ca_{i+1}, C_{i+1})
        nxt_phi = defs[(i + 1) % n_res][0]
        n_next, ca_next = nxt_phi[1], nxt_phi[2]
        quads.append((ca_i, c_i, n_next, ca_next))
    return quads


def backbone_dihedral_values(
    mol: Chem.Mol, conf_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (phi, psi, omega) in degrees for every residue of one conformer.

    Params:
        mol: Chem.Mol : peptide with the conformer
        conf_id: int : conformer id
    Returns:
        tuple (phi, psi, omega), each np.ndarray of shape (n_res,) in degrees
    """
    conf = mol.GetConformer(conf_id)
    defs = ordered_backbone_dihedrals(mol)
    oq = omega_quads(mol)
    phi = np.array([rdMolTransforms.GetDihedralDeg(conf, *d[0]) for d in defs])
    psi = np.array([rdMolTransforms.GetDihedralDeg(conf, *d[1]) for d in defs])
    omega = np.array([rdMolTransforms.GetDihedralDeg(conf, *q) for q in oq])
    return phi, psi, omega


# --- side-chain chi dihedrals -------------------------------------------

MAX_CHI = 4  # chi slots per residue (chi1..chi4); covers all standard side chains


def sidechain_chi_quads(mol: Chem.Mol, max_chi: int = MAX_CHI) -> list[list[tuple]]:
    """
    Per-residue list of side-chain chi dihedral atom-quads (chi1, chi2, ...).

    Walks the side chain from Cβ outward, defining chi_k as the dihedral whose
    central (rotated) bond is the k-th side-chain bond: chi1 = (N, Cα, Cβ, Cγ),
    chi2 = (Cα, Cβ, Cγ, Cδ), ... A chi is emitted only if its rotated bond is NOT
    in a ring (so the dihedral that rotates a side chain *into* an aromatic ring —
    e.g. Trp chi2, central bond Cβ-Cγ — is kept, but bonds within the ring are not).
    Branches are resolved deterministically (lowest atom index). Each chi is
    settable via rdMolTransforms.SetDihedralDeg, so predicted chi can seed the
    side chain directly (side-chain bonds are not ring-closure-constrained).

    Params:
        mol: Chem.Mol : peptide with explicit Hs
        max_chi: int : maximum chi slots per residue
    Returns:
        list (per residue, ring order) of lists of 4-tuples (0 to max_chi each)
    """
    out = []
    for n, ca, c in residue_atoms(mol):
        cb = next(
            (
                nb.GetIdx()
                for nb in mol.GetAtomWithIdx(ca).GetNeighbors()
                if nb.GetAtomicNum() > 1 and nb.GetIdx() not in (n, c)
            ),
            None,
        )
        if cb is None:  # glycine
            out.append([])
            continue
        path = [n, ca, cb]
        while len(path) < max_chi + 3:
            bb, aa = path[-1], path[-2]
            nbrs = sorted(
                nb.GetIdx()
                for nb in mol.GetAtomWithIdx(bb).GetNeighbors()
                if nb.GetAtomicNum() > 1 and nb.GetIdx() not in path
            )
            if not nbrs:
                break
            q = nbrs[0]
            path.append(q)
            if mol.GetBondBetweenAtoms(bb, q).IsInRing():
                break  # include one ring atom as reference, then stop
        chis = []
        for k in range(1, len(path) - 2):
            if not mol.GetBondBetweenAtoms(path[k], path[k + 1]).IsInRing():
                chis.append((path[k - 1], path[k], path[k + 1], path[k + 2]))
            if len(chis) >= max_chi:
                break
        out.append(chis)
    return out


def sidechain_chi_values(mol: Chem.Mol, conf_id: int, max_chi: int = MAX_CHI):
    """
    Side-chain chi dihedral values (degrees) and a presence mask per residue.

    Params:
        mol: Chem.Mol : peptide with the conformer
        conf_id: int : conformer id
        max_chi: int : chi slots per residue
    Returns:
        tuple (chi (n_res, max_chi) float NaN-padded, mask (n_res, max_chi) bool)
    """
    conf = mol.GetConformer(conf_id)
    quads = sidechain_chi_quads(mol, max_chi=max_chi)
    n_res = len(quads)
    chi = np.full((n_res, max_chi), np.nan, dtype=np.float64)
    mask = np.zeros((n_res, max_chi), dtype=bool)
    for i, qs in enumerate(quads):
        for k, q in enumerate(qs):
            chi[i, k] = rdMolTransforms.GetDihedralDeg(conf, *q)
            mask[i, k] = True
    return chi, mask


# --- per-residue features ------------------------------------------------

FEATURE_NAMES = [
    "is_nme",
    "is_gly",
    "is_d",
    "is_pro",
    "sc_heavy",
    "sc_aromatic",
    "sc_hbond_don",
    "sc_hbond_acc",
]
N_BASE_FEATURES = len(FEATURE_NAMES)


def _h_neighbors(atom) -> int:
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)


def _sidechain_atoms(mol: Chem.Mol, n_idx: int, ca_idx: int, c_idx: int) -> set[int]:
    """BFS from Ca into the side chain, excluding backbone N/C and all H atoms."""
    backbone = {n_idx, c_idx}
    sc: set[int] = set()
    stack = [
        nb.GetIdx()
        for nb in mol.GetAtomWithIdx(ca_idx).GetNeighbors()
        if nb.GetIdx() not in backbone and nb.GetAtomicNum() != 1
    ]
    while stack:
        a = stack.pop()
        if a in sc or a == ca_idx:
            continue
        sc.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j not in sc and j != ca_idx and nb.GetAtomicNum() != 1:
                stack.append(j)
    return sc


def _residue_descriptor(
    mol: Chem.Mol, n_idx: int, ca_idx: int, c_idx: int
) -> list[float]:
    """Physico-chemical descriptor vector for one residue (FEATURE_NAMES order)."""
    n_atom = mol.GetAtomWithIdx(n_idx)
    ca_atom = mol.GetAtomWithIdx(ca_idx)

    is_nme = 0.0
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 6 and nb.GetIdx() != ca_idx:
            heavy = sum(1 for x in nb.GetNeighbors() if x.GetAtomicNum() != 1)
            if _h_neighbors(nb) == 3 and heavy == 1:
                is_nme = 1.0
    is_gly = 1.0 if _h_neighbors(ca_atom) >= 2 else 0.0
    is_d = 1.0 if ca_atom.GetPropsAsDict().get("_CIPCode", "") == "R" else 0.0

    sc = _sidechain_atoms(mol, n_idx, ca_idx, c_idx)
    is_pro = (
        1.0
        if n_idx
        in {nb.GetIdx() for a in sc for nb in mol.GetAtomWithIdx(a).GetNeighbors()}
        else 0.0
    )
    sc_aromatic = 1.0 if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in sc) else 0.0
    sc_don, sc_acc = 0.0, 0.0
    for a in sc:
        at = mol.GetAtomWithIdx(a)
        if at.GetAtomicNum() in (7, 8):
            sc_acc += 1.0
            if _h_neighbors(at) > 0:
                sc_don += 1.0
    return [is_nme, is_gly, is_d, is_pro, float(len(sc)), sc_aromatic, sc_don, sc_acc]


def residue_features(mol: Chem.Mol) -> np.ndarray:
    """
    Per-residue base feature matrix in get_backbone_dihedrals order.

    Calls AssignStereochemistry so D/L (CIP) is available.

    Params:
        mol: Chem.Mol : peptide with explicit Hs
    Returns:
        np.ndarray of shape (n_res, N_BASE_FEATURES), dtype float32
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rows = [_residue_descriptor(mol, n, ca, c) for (n, ca, c) in residue_atoms(mol)]
    return np.asarray(rows, dtype=np.float32)


def neighbor_augment(feats: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Concatenate cyclic ±window neighbour descriptors onto each residue's features.

    For a macrocycle the sequence wraps, so neighbours are taken cyclically. The
    output for residue i is [f_{i-w}, ..., f_{i-1}, f_i, f_{i+1}, ..., f_{i+w}].

    Params:
        feats: np.ndarray : (n_res, F) base per-residue features
        window: int : number of neighbours on each side (default 1)
    Returns:
        np.ndarray of shape (n_res, F*(2*window+1)), dtype float32
    """
    parts = [
        np.roll(feats, shift=-offset, axis=0) for offset in range(-window, window + 1)
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)
