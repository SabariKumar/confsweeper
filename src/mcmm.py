"""
Multiple Minimum Monte Carlo (MCMM) sampler for cyclic peptide conformer
search. Issue #11; companion module to src/concerted_rotation.py, which
holds the move geometry.

This module is being built up incrementally per docs/mcmm_plan.md. Steps
3, 4, and 5 (current) implement backbone window enumeration, the shared
basin-memory data structure, and the single-walker MCMM driver. Future
steps:

    Step 6: parallel walkers (batched MMFF)
    Step 7: replica exchange across temperatures
    Step 8: get_mol_PE_mcmm entry point in src/confsweeper.py

The DBT-style concerted-rotation move geometry — chain rebuild, closure
solver, Wu-Deem Jacobian — lives in src/concerted_rotation.py. This
module orchestrates the moves around a real RDKit mol: enumerates valid
backbone windows, picks one per move, applies the move, and (later)
runs MMFF + accept/reject.
"""

import math

import ase
import numpy as np
import torch
from rdkit import Chem

# 7 consecutive backbone atoms = 4 inner dihedrals = the chain shape
# expected by concerted_rotation.propose_move.
WINDOW_SIZE = 7

# Relaxed backbone SMARTS for MCMM's window enumeration. Matches both
# amide (peptide, [N]) and ester (depsipeptide, [O]) backbone linker
# positions — broader than torsional_sampling's strict
# `_BACKBONE_SMARTS`, which is amide-only and is used by
# `classify_backbone_residues` for L/D/NMe/Gly classification (a
# concept that only applies to amide backbones).
#
# DBT concerted-rotation geometry treats the linker atom kinematically:
# any backbone atom in the chain works, whether it's N or O. Allowing
# both at the [:2] and [:5] positions lets us find the macrocycle on
# depsipeptides (peptides with one or more amide bonds replaced by
# ester linkages — common in natural cyclic peptides like beauveriolide,
# valinomycin, and didemnins). The old strict SMARTS produced 2M fewer
# matches per peptide with M ester bonds, dead-ending the C→next walk.
_MCMM_BACKBONE_SMARTS = Chem.MolFromSmarts("[C:1](=O)[N,O:2][CX4:3][C:4](=O)[N,O:5]")


def _ordered_macrocycle_atoms(mol: Chem.Mol) -> list:
    """
    Return atom indices of the macrocycle (largest ring), in cyclic order.

    Uses RDKit's ring perception (`mol.GetRingInfo().AtomRings()`,
    Smallest-Set-of-Smallest-Rings) and selects the longest ring. For
    typical cyclic peptide and depsipeptide macrocycles this is the
    backbone ring; for fused-ring or multi-cycle molecules the largest
    ring is still the most-encompassing cycle and the most useful
    starting point for DBT moves.

    Why this is the primary path (over the SMARTS-based residue
    enumeration in `_ordered_backbone_residues`): the SMARTS approach
    requires the macrocycle to fit a `C(=O)-linker-Cα-C(=O)-linker`
    repetitive pattern, which fails on real peptides with
    structural irregularities — depsipeptide ester linkers, Cα-Cα
    direct bonds (some natural products), side-chain-to-backbone
    bridges, modified residues. Ring perception just needs a ring;
    DBT geometry just needs ring atoms in cyclic order. Both work
    on any macrocycle topology, including peptides whose chemistry
    falls outside the strict amide-residue model.

    The returned order is RDKit's ring-walk order, which is a valid
    cyclic traversal. The starting atom and direction are not stable
    across mols; window enumeration emits one window per starting
    atom anyway, so the cyclic shift doesn't matter for sampling.

    Params:
        mol: Chem.Mol : input molecule
    Returns:
        list[int] : atom indices in cyclic ring order. Empty if the
            molecule has no rings or the largest ring has fewer than
            `WINDOW_SIZE` atoms.
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return []
    largest = max(atom_rings, key=len)
    if len(largest) < WINDOW_SIZE:
        return []
    return list(largest)


def enumerate_backbone_windows(mol: Chem.Mol) -> list:
    """
    Return every 7-atom backbone window in a cyclic macromolecule.

    Walks the macrocycle (largest ring per RDKit's ring perception) and
    emits one cyclic window per starting ring atom. For a K-atom ring
    there are K windows.

    Each window is a tuple of 7 atom indices in the order they appear
    around the ring. The window's atom layout matches what
    `concerted_rotation.propose_move` expects: r0..r6 are 7 consecutive
    ring atoms with bonds r0-r1, r1-r2, ..., r5-r6. The 4 inner
    dihedrals (around bonds (1,2)..(4,5)) are what the move perturbs.

    Switched in Step 8b from a SMARTS-based residue enumeration to
    direct RDKit ring perception, so it handles arbitrary cyclic
    macromolecules: standard peptides, depsipeptides (with ester
    linkers), Cα-Cα-bonded peptides, and other natural-product
    backbones the SMARTS pattern doesn't fit.

    Params:
        mol: Chem.Mol : a cyclic molecule with a macrocycle of ≥ 7 ring
            atoms; explicit Hs are optional. Side chains are ignored
            (only ring atoms participate in the windows).
    Returns:
        list of 7-tuples of atom indices in cyclic ring order. Empty if
        the molecule has no ring of ≥ 7 atoms.
    """
    ring_atoms = _ordered_macrocycle_atoms(mol)
    n = len(ring_atoms)
    if n < WINDOW_SIZE:
        return []
    return [
        tuple(ring_atoms[(start + i) % n] for i in range(WINDOW_SIZE))
        for start in range(n)
    ]


def _ordered_backbone_residues(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """
    Return (linker, Cα, C) atom indices per residue, in cyclic order
    around the macrocycle.

    The "linker" atom is the backbone N (peptide bond) or O (ester /
    depsipeptide bond) — see `_MCMM_BACKBONE_SMARTS`. The C → next-linker
    bond graph over all SMARTS-matched residues is walked starting from
    each residue in turn; the longest closed cycle is returned.

    Most peptides yield exactly one closed cycle covering every detected
    residue. Two cases need special handling:
      1. Macrocycles with side-chain-to-backbone bridges (e.g.
         head-to-side-chain lactams) trigger spurious extra residue
         matches off the main ring. The longest-cycle rule discards
         them.
      2. Depsipeptides with ester linkers: the relaxed
         `_MCMM_BACKBONE_SMARTS` matches ester positions in addition to
         amide ones, so all backbone residues appear in the residue list
         regardless of linker type.

    Params:
        mol: Chem.Mol : input molecule
    Returns:
        list of (linker_idx, Cα_idx, C_idx) tuples, ordered cyclically.
        `linker_idx` is N for amide residues, O for ester residues.
        Empty if no backbone is found.
    Raises:
        ValueError: if no closed cycle exists among the SMARTS-matched
            residues (input is not a head-to-tail cyclic peptide or
            depsipeptide).
    """
    # Match backbone with the relaxed SMARTS (amide + ester linkers).
    # Each match is one residue, dedup'd by linker atom index. The match
    # tuple shape is (C_prev, =O, linker, Cα, C, =O, linker_next).
    residues: list = []
    seen_linker: set = set()
    for match in mol.GetSubstructMatches(_MCMM_BACKBONE_SMARTS):
        _, _, linker, ca, c, _, _ = match
        if linker in seen_linker:
            continue
        seen_linker.add(linker)
        residues.append((linker, ca, c))

    if not residues:
        return []

    linker_to_res = {linker: (linker, ca, c) for linker, ca, c in residues}

    # For each residue's C(=O), find the next residue's linker (the N or
    # O bonded to this C via the peptide / ester bond). The C atom has
    # three neighbours: its own =O, its own Cα, and the next residue's
    # linker. The linker is uniquely identified by being a key in
    # `linker_to_res`; the carbonyl O and own-residue Cα are not.
    next_linker_for: dict[int, int] = {}
    for linker_idx, _, c_idx in residues:
        c_atom = mol.GetAtomWithIdx(c_idx)
        for nb in c_atom.GetNeighbors():
            if nb.GetIdx() in linker_to_res:
                next_linker_for[linker_idx] = nb.GetIdx()
                break

    # Try every residue as a starting point; return the longest closed
    # cycle.
    best_cycle: list | None = None
    for start_residue in residues:
        start_linker = start_residue[0]
        cycle: list = []
        visited: set = set()
        current_linker: int | None = start_linker
        ring_closed = False
        while current_linker is not None and current_linker not in visited:
            visited.add(current_linker)
            cycle.append(linker_to_res[current_linker])
            next_linker = next_linker_for.get(current_linker)
            if next_linker == start_linker:
                ring_closed = True
                break
            current_linker = next_linker
        if ring_closed and (best_cycle is None or len(cycle) > len(best_cycle)):
            best_cycle = cycle

    if best_cycle is None:
        smi = Chem.MolToSmiles(mol)
        missing_next = [
            linker_idx
            for linker_idx, _, _ in residues
            if next_linker_for.get(linker_idx) is None
        ]
        raise ValueError(
            f"Backbone ring did not close: no closed cycle found among "
            f"{len(residues)} residue matches. Input must be a head-to-tail "
            f"cyclic peptide or depsipeptide. SMILES: {smi}. Residue linker "
            f"atoms with no peptide-bond successor (next_linker_for missing): "
            f"{missing_next or 'none'}."
        )

    return best_cycle


# ---------------------------------------------------------------------------
# Side-chain enumeration — used by the MCMM proposer for full-mol moves
# ---------------------------------------------------------------------------


def _backbone_atom_set(mol: Chem.Mol) -> set:
    """
    Return the set of atom indices that lie on the macrocycle backbone.

    Backbone atoms are the atoms on the largest ring as identified by
    RDKit ring perception (see `_ordered_macrocycle_atoms`). Used as
    the "stop set" for the side-chain BFS in `_side_chain_group` —
    atoms outside this set are side-chain candidates; atoms inside
    are part of the ring and must not be crossed during the BFS.

    Params:
        mol: Chem.Mol : a cyclic molecule with a macrocycle ring
    Returns:
        set[int] : ring atom indices; empty if no qualifying ring exists
    """
    return set(_ordered_macrocycle_atoms(mol))


def _side_chain_group(mol: Chem.Mol, atom_idx: int, backbone_atom_set: set) -> set:
    """
    BFS from `atom_idx` through non-backbone bonds.

    Returns the set of atom indices reachable from `atom_idx` without
    crossing any backbone atom (the starting atom itself is excluded
    from the result; only its non-backbone neighbours and beyond are
    included). For a backbone atom in a cyclic peptide, this returns
    the side chain attached at that residue — Hα, Cβ, side-chain
    branches, etc. — without leaking into adjacent residues' side
    chains via the macrocycle.

    For a non-backbone starting atom, the result is the connected
    non-backbone component containing it (minus the starting atom),
    which is rarely what callers want; pass a backbone atom.

    Params:
        mol: Chem.Mol : input molecule
        atom_idx: int : starting atom (typically a backbone atom)
        backbone_atom_set: set[int] : atoms forming the macrocycle ring;
            traversal does not cross these
    Returns:
        set[int] : reachable atom indices, excluding `atom_idx` itself
    """
    side_chain: set = set()
    queue: list = []
    for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nb_idx = nb.GetIdx()
        if nb_idx not in backbone_atom_set:
            queue.append(nb_idx)
    while queue:
        idx = queue.pop()
        if idx in side_chain:
            continue
        side_chain.add(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in backbone_atom_set or nb_idx in side_chain:
                continue
            queue.append(nb_idx)
    return side_chain


def _compute_window_downstream_sets(
    mol: Chem.Mol, window: tuple, backbone_atom_set: set
) -> list:
    """
    Compute the full-mol atom indices that should rotate per dihedral
    when a DBT move acts on `window`.

    For dihedral k around bond (window[k+1], window[k+2]):
      - Window backbone atoms strictly downstream: window[k+3..6].
      - Side chains of window[k+2..6] (the pivot atom k+2's side chain
        rotates with the local frame at the pivot; downstream backbone
        atoms' side chains rigidly follow their parents).

    The pivot atom window[k+2] itself stays on the rotation axis and
    does not move. Side chains of window[0..k+1] are upstream of the
    bond and do not rotate.

    Params:
        mol: Chem.Mol : input molecule
        window: tuple[int, ...] : 7 atom indices, in chain order
        backbone_atom_set: set[int] : from `_backbone_atom_set(mol)`,
            passed in to avoid recomputation across windows
    Returns:
        list of 4 frozenset[int] : per-dihedral rotation set, suitable
            for `concerted_rotation.apply_dihedral_changes_full_mol`
    """
    side_chains = {
        atom_idx: _side_chain_group(mol, atom_idx, backbone_atom_set)
        for atom_idx in window
    }
    downstream_sets: list = []
    for k in range(4):
        rotated: set = set()
        rotated.update(side_chains[window[k + 2]])
        for j in range(k + 3, 7):
            rotated.add(window[j])
            rotated.update(side_chains[window[j]])
        downstream_sets.append(frozenset(rotated))
    return downstream_sets


# ---------------------------------------------------------------------------
# Kabsch heavy-atom RMSD — shared dedup primitive
# ---------------------------------------------------------------------------


def _kabsch_rmsd_pairwise(
    queries: torch.Tensor,
    refs: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise Kabsch-aligned RMSDs between every query and every reference.

    Both inputs should already be sliced to whatever atom subset defines
    a basin (typically heavy atoms only). The function applies the
    standard Kabsch algorithm: centre per-conformer, build the
    cross-covariance H = r.T @ q for each (query, ref) pair, take the
    SVD, correct for reflections via the determinant sign of U @ Vh,
    rotate the references onto each query, and report the per-pair
    RMSD. Translation- and rotation-invariant; no atom-permutation
    symmetry handling.

    Params:
        queries: torch.Tensor : query coordinates, shape (..., n, 3).
            Leading dims are broadcast as a "batch of queries"; passing
            a single (n, 3) query gives an output of shape (K,).
        refs: torch.Tensor : reference coordinates, shape (K, n, 3).
    Returns:
        torch.Tensor : RMSD per (query, ref) pair, shape (..., K), in
            the input length units (typically Ångström).
    """
    if refs.dim() != 3:
        raise ValueError(f"refs must be (K, n, 3), got {tuple(refs.shape)}")
    if queries.dim() < 2 or queries.shape[-2:] != refs.shape[-2:]:
        raise ValueError(
            f"queries shape {tuple(queries.shape)} incompatible with "
            f"refs shape {tuple(refs.shape)}"
        )
    n = refs.shape[-2]
    if n < 2:
        raise ValueError(f"need at least 2 atoms for Kabsch, got {n}")

    q = (queries - queries.mean(dim=-2, keepdim=True)).to(torch.float64)
    r = (refs - refs.mean(dim=-2, keepdim=True)).to(torch.float64)

    # H[..., k, i, j] = sum_a r[k, a, i] * q[..., a, j] = (r[k]^T @ q[...])_{ij}
    H = torch.einsum("kai,...aj->...kij", r, q)

    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(torch.matmul(U, Vh)))  # (..., K)
    diag = torch.ones(d.shape + (3,), dtype=torch.float64, device=q.device)
    diag[..., 2] = d
    D = torch.diag_embed(diag)
    # Standard Kabsch with row-vec coords: R_align = V @ D @ U.T = Vh.T @ D @ U.T.
    # aligned_r = r @ R_align.T → einsum below contracts r's atom-coordinate
    # axis 'i' against R_align's axis 'i' (the last axis after .T on the matrix).
    R = Vh.transpose(-2, -1) @ D @ U.transpose(-2, -1)
    aligned_r = torch.einsum("kni,...kji->...knj", r, R)

    err = aligned_r - q.unsqueeze(-3)
    rmsd = torch.sqrt((err**2).sum(dim=(-2, -1)) / n)
    return rmsd


# ---------------------------------------------------------------------------
# Inertia-tensor eigenvalues — CREST-style rotational dedup criterion
# ---------------------------------------------------------------------------


def _inertia_eigvals(
    coords: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    """
    Sorted eigenvalues of the inertia tensor for one or more conformers.

    Used by the CREST-style three-criteria dedup (Step 17 of
    docs/mcmm_plan.md): two conformers are distinguishable by overall
    shape — independent of translation, rotation, and atom-permutation
    symmetry — when their inertia eigenvalues differ by more than the
    rotational-constant anisotropy threshold.

    The inertia tensor is:
        I_{ab} = Σ_a m_a (||r_a||² δ_{ab} - r_{a,a} r_{a,b})
    computed in the centre-of-mass frame. Eigenvalues are real and
    non-negative; sorting ascending makes pairwise comparison
    well-defined.

    Params:
        coords: torch.Tensor : (..., n_atoms, 3) atomic coordinates
        masses: torch.Tensor : (n_atoms,) atomic masses (units cancel
            in relative-difference comparisons; Da is conventional)
    Returns:
        torch.Tensor : (..., 3) ascending eigenvalues of the inertia
            tensor
    """
    if coords.shape[-1] != 3:
        raise ValueError(f"coords last dim must be 3, got {tuple(coords.shape)}")
    if masses.dim() != 1 or masses.shape[0] != coords.shape[-2]:
        raise ValueError(
            f"masses shape {tuple(masses.shape)} must be ({coords.shape[-2]},)"
        )
    coords64 = coords.to(torch.float64)
    masses64 = masses.to(torch.float64)

    # Centre of mass.
    M_total = masses64.sum()
    com = (masses64.unsqueeze(-1) * coords64).sum(dim=-2, keepdim=True) / M_total
    r = coords64 - com  # (..., n, 3)

    # Trace term: Σ_a m_a |r_a|^2
    r_sq = (r**2).sum(dim=-1)  # (..., n)
    trace_term = (masses64 * r_sq).sum(dim=-1)  # (...,)

    # Outer-product term: Σ_a m_a r_a r_aᵀ → (..., 3, 3)
    weighted_r = masses64.unsqueeze(-1) * r
    outer = torch.matmul(weighted_r.transpose(-2, -1), r)

    eye = torch.eye(3, dtype=torch.float64, device=coords.device)
    inertia = trace_term.unsqueeze(-1).unsqueeze(-1) * eye - outer
    return torch.linalg.eigvalsh(inertia)  # (..., 3) ascending


def _max_relative_eig_diff(
    query: torch.Tensor,
    stored: torch.Tensor,
) -> torch.Tensor:
    """
    Per-row max relative difference between a single query eigenvalue
    triple and a stored batch.

    For each row k of `stored`, returns the largest of
    `|stored[k, i] - query[i]| / max(|stored[k, i]|, |query[i]|)` over
    i ∈ {0, 1, 2}. This is the natural CREST-style "rotational
    constant anisotropy" comparison: invariant to the absolute scale
    of the inertia eigenvalues, captures distortion in any of the
    three principal axes.

    Params:
        query: torch.Tensor : (3,) eigenvalues of the query conformer
        stored: torch.Tensor : (K, 3) eigenvalues of K stored basins
    Returns:
        torch.Tensor : (K,) max relative differences in [0, 1)
    """
    diff = (stored - query.unsqueeze(0)).abs()  # (K, 3)
    denom = torch.maximum(stored.abs(), query.unsqueeze(0).abs()).clamp(min=1e-12)
    return (diff / denom).max(dim=-1).values  # (K,)


# ---------------------------------------------------------------------------
# Basin memory — shared across MCMM walkers
# ---------------------------------------------------------------------------


# Default Kabsch heavy-atom RMSD (Å) below which two conformers are
# considered the same basin. 0.125 Å matches CREMP / CREST / GOAT's
# conformer-uniqueness contract, which is what `n_basins` should be
# directly comparable to in the issue-#10 benchmark. Override to
# 0.5 Å (the chemical "basin" scale) when looking for coarser
# clustering; raise toward 1.0 Å to merge near-degenerate sub-basins
# aggressively.
DEFAULT_RMSD_THRESHOLD = 0.125


class BasinMemory:
    """
    Shared basin memory for MCMM walkers.

    Stores one representative conformer (coordinates + MMFF energy) per
    discovered basin and tracks a per-basin visit counter for the
    Saunders 1/√usage acceptance bias. Basins are distinguished by
    Kabsch-aligned RMSD over an optional atom subset (typically the
    molecule's heavy atoms). Two conformers are in the same basin if
    `kabsch_rmsd < rmsd_threshold` (strict `<`, matching
    `_energy_ranked_dedup`'s dedup convention so thresholds are
    interchangeable).

    Driver flow per accepted move:

        idx, _ = memory.query_novelty(coords)
        if idx is None:
            memory.add_basin(coords, energy)
        else:
            memory.record_visit(idx)

    The Saunders bias is queried *before* the accept/reject decision:

        bias = memory.acceptance_bias(idx)   # 1.0 if idx is None
        p_accept = min(1, exp(-ΔE / kT) * bias)

    The stored representative is whichever conformer was first observed
    in each basin — re-visits do not update coordinates or energy. For
    the `get_mol_PE_mcmm` final output, the post-MCMM
    `_minimize_score_filter_dedup` re-deduplicates at the MACE level,
    so first-found vs. lowest-energy representative does not change the
    final basin set materially.

    Tensors live on `device` (CPU by default). Move to GPU only if
    profiling shows the per-step novelty query becomes hot — for
    typical macrocyclic peptides K stays in the low hundreds and the
    CPU query is microseconds. The Kabsch SVD dominates per-query cost
    but is still O(K) 3x3 SVDs, ~tens of microseconds for K=200.

    Two dedup modes are available:

    - `dedup_mode='kabsch'` (default): two conformers are the same
      basin iff their Kabsch heavy-atom RMSD is below `rmsd_threshold`.
      Translation- and rotation-invariant; no atom-permutation
      symmetry. Backward-compatible with all pre-Step-17 callers.
    - `dedup_mode='crest'`: CREST/CREMP-style three-criteria dedup.
      Two conformers are the same basin only when **all three** hold:
      Kabsch RMSD < `rmsd_threshold` AND |ΔE| <
      `energy_threshold_eV` AND inertia-eigenvalue max relative
      diff < `rotconst_anisotropy_threshold`. Any one criterion
      saying "different" keeps them as separate basins. Used for
      paper-comparable basin counts vs CREMP `uniqueconfs`. Requires
      `atomic_numbers` to compute inertia tensors. Energy criterion
      uses MACE-on-MMFF-relaxed energies (already in `_energies`),
      no extra compute.

    Params:
        n_atoms: int : number of atoms per conformer (must match every
            coords tensor passed to add_basin / query_novelty). Stored
            coords always include all atoms so the proposer can apply
            moves to the full molecule.
        rmsd_threshold: float : basin-distinguishing distance in Å
            (Kabsch-RMSD units); default 0.125 Å, matching CREMP /
            CREST / GOAT's `uniqueconfs` contract for direct
            benchmark comparability. Override to 0.5 Å for coarser
            "chemical basin" clustering.
        heavy_atom_indices: list[int] | None : optional atom subset
            used for distance comparisons; full coords are still
            stored. Default None means all atoms (H included), which is
            mostly useful for synthetic test fixtures — production
            callers should pass the heavy-atom subset extracted from
            the mol so the metric matches CREST's heavy-atom convention.
        dedup_mode: str : 'kabsch' (default) or 'crest'.
        energy_threshold_eV: float : energy criterion threshold in eV
            for `dedup_mode='crest'`. Default 0.05 eV (≈ 1.2 kcal/mol)
            is loose relative to CREST's xtb-based 0.05 kcal/mol — the
            MACE float32 noise floor (~0.01–0.05 eV per the model's
            ~1e-6 relative precision on 10⁴–10⁵ eV totals) sets a
            practical floor below which the criterion would just see
            noise. Still 2× tighter than thermal kT_298 (0.026 eV) so
            it has discriminatory power.
        rotconst_anisotropy_threshold: float : rotational-constant
            criterion for `dedup_mode='crest'`. Default 0.01 (1%) is
            CREST's middle of the documented 1–2.5% range. Compared
            via max relative difference of the three inertia-tensor
            eigenvalues.
        atomic_numbers: list[int] | None : per-atom Z used to derive
            masses for the inertia tensor. Required when
            `dedup_mode='crest'`; ignored otherwise. Masses come from
            RDKit's PeriodicTable.
        saunders_exponent: float : exponent `p` in the Saunders bias
            `1 / usage^p`. Default 0.5 matches Saunders 1990 (slow
            decay, gentle re-visit penalty); 1.0 implements lever C15
            (faster decay, pushes walkers out of deep wells harder).
        device: torch.device | str : device for stored tensors (default 'cpu')
    """

    NOVEL = -1  # sentinel for the batched query API

    def __init__(
        self,
        n_atoms: int,
        rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
        heavy_atom_indices=None,
        *,
        dedup_mode: str = "kabsch",
        energy_threshold_eV: float = 0.05,
        rotconst_anisotropy_threshold: float = 0.01,
        atomic_numbers=None,
        saunders_exponent: float = 0.5,
        device="cpu",
    ):
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if rmsd_threshold < 0:
            raise ValueError(
                f"rmsd_threshold must be non-negative, got {rmsd_threshold}"
            )
        if dedup_mode not in ("kabsch", "crest"):
            raise ValueError(
                f"dedup_mode must be 'kabsch' or 'crest', got {dedup_mode!r}"
            )
        if energy_threshold_eV < 0:
            raise ValueError(
                f"energy_threshold_eV must be non-negative, got {energy_threshold_eV}"
            )
        if rotconst_anisotropy_threshold < 0:
            raise ValueError(
                f"rotconst_anisotropy_threshold must be non-negative, got "
                f"{rotconst_anisotropy_threshold}"
            )
        if dedup_mode == "crest" and atomic_numbers is None:
            raise ValueError(
                "atomic_numbers is required when dedup_mode='crest' "
                "(masses are needed to build the inertia tensor)"
            )
        if saunders_exponent <= 0:
            raise ValueError(
                f"saunders_exponent must be positive, got {saunders_exponent}"
            )
        self.n_atoms = n_atoms
        self.rmsd_threshold = float(rmsd_threshold)
        self.dedup_mode = dedup_mode
        self.energy_threshold_eV = float(energy_threshold_eV)
        self.rotconst_anisotropy_threshold = float(rotconst_anisotropy_threshold)
        self.saunders_exponent = float(saunders_exponent)
        self.device = torch.device(device)
        if heavy_atom_indices is None:
            self._heavy_atom_indices = None
        else:
            heavy = list(heavy_atom_indices)
            if not heavy:
                raise ValueError("heavy_atom_indices must be non-empty when provided")
            if len(heavy) < 2:
                raise ValueError(
                    "heavy_atom_indices must contain at least 2 atoms for Kabsch RMSD"
                )
            if any(i < 0 or i >= n_atoms for i in heavy):
                raise ValueError(
                    f"heavy_atom_indices out of range [0, {n_atoms}): {heavy}"
                )
            self._heavy_atom_indices = torch.tensor(
                heavy, dtype=torch.int64, device=self.device
            )
        # Atomic-number-derived masses for the inertia tensor. Stored
        # only when dedup_mode='crest'; otherwise None to keep the
        # default-mode footprint identical to pre-Step-17 behaviour.
        if atomic_numbers is None:
            self._masses = None
        else:
            atomic_numbers = list(atomic_numbers)
            if len(atomic_numbers) != n_atoms:
                raise ValueError(
                    f"atomic_numbers length ({len(atomic_numbers)}) must match "
                    f"n_atoms ({n_atoms})"
                )
            from rdkit.Chem import GetPeriodicTable

            pt = GetPeriodicTable()
            self._masses = torch.tensor(
                [pt.GetAtomicWeight(int(z)) for z in atomic_numbers],
                dtype=torch.float64,
                device=self.device,
            )
        self._coords = torch.zeros(
            (0, n_atoms, 3), dtype=torch.float64, device=self.device
        )
        self._energies = torch.zeros((0,), dtype=torch.float64, device=self.device)
        self._usages = torch.zeros((0,), dtype=torch.int64, device=self.device)
        # Per-basin inertia eigenvalues for crest mode. Empty in kabsch mode.
        self._rotconsts = torch.zeros((0, 3), dtype=torch.float64, device=self.device)

    def _select(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Slice coords to the basin-distinguishing atom subset.

        Params:
            coords: torch.Tensor : (..., n_atoms, 3) — full-coord tensor
                or batch thereof
        Returns:
            torch.Tensor : (..., n_select, 3) where n_select == len(
                heavy_atom_indices) when configured, else n_atoms.
        """
        if self._heavy_atom_indices is None:
            return coords
        return coords.index_select(dim=-2, index=self._heavy_atom_indices)

    @property
    def n_basins(self) -> int:
        """Number of basins currently stored."""
        return int(self._coords.shape[0])

    @property
    def coords(self) -> torch.Tensor:
        """Stored basin coordinates, shape (K, n_atoms, 3)."""
        return self._coords

    @property
    def energies(self) -> torch.Tensor:
        """Stored per-basin energies (eV by convention), shape (K,)."""
        return self._energies

    @property
    def usages(self) -> torch.Tensor:
        """Per-basin visit counts (int64), shape (K,)."""
        return self._usages

    def add_basin(self, coords: torch.Tensor, energy: float) -> int:
        """
        Append a new basin and return its index.

        Initialises the visit counter to 1 (the discovery counts as one
        visit). Caller is responsible for ensuring `coords` is novel
        relative to existing basins; if it falls within `rmsd_threshold`
        of a stored basin, this still appends a duplicate. Use
        `query_novelty` first when in doubt.

        Params:
            coords: torch.Tensor (n_atoms, 3) : conformer coordinates
            energy: float : MMFF (or other) energy of this conformer
        Returns:
            int : the new basin's index in [0, K)
        """
        coords = self._validate_coords(coords)
        coords_dev = coords.to(self.device, dtype=torch.float64).unsqueeze(0)
        self._coords = torch.cat([self._coords, coords_dev], dim=0)
        self._energies = torch.cat(
            [
                self._energies,
                torch.tensor([float(energy)], dtype=torch.float64, device=self.device),
            ]
        )
        self._usages = torch.cat(
            [
                self._usages,
                torch.tensor([1], dtype=torch.int64, device=self.device),
            ]
        )
        if self.dedup_mode == "crest":
            rot = _inertia_eigvals(coords_dev[0], self._masses).unsqueeze(0)  # (1, 3)
            self._rotconsts = torch.cat([self._rotconsts, rot], dim=0)
        return self.n_basins - 1

    def query_novelty(self, coords: torch.Tensor, energy=None) -> tuple:
        """
        Find the closest stored basin and report whether it is within
        the basin-distinguishing threshold.

        In `dedup_mode='kabsch'`, two conformers are the same basin iff
        their Kabsch RMSD is below `rmsd_threshold`. In
        `dedup_mode='crest'`, three concurrent criteria must all hold;
        the closest stored basin (by RMSD) among the same-basin
        candidates is returned, or None if no candidate satisfies all
        three.

        Params:
            coords: torch.Tensor (n_atoms, 3) : conformer coordinates
            energy: float | None : conformer energy (eV) — required
                when `dedup_mode='crest'`, ignored when 'kabsch'.
        Returns:
            tuple[int | None, float] : (idx, distance). `idx` is None
                when no stored basin satisfies the active dedup
                criteria; otherwise it is the index of the closest
                basin (by RMSD) among the matches. `distance` is the
                closest Kabsch RMSD regardless of threshold; `inf`
                when memory is empty.
        """
        if self.dedup_mode == "crest" and energy is None:
            raise ValueError(
                "energy is required for query_novelty when dedup_mode='crest'"
            )
        coords = self._validate_coords(coords)
        if self.n_basins == 0:
            return None, math.inf
        coords_dev = coords.to(self.device, dtype=torch.float64)
        distances = _kabsch_rmsd_pairwise(
            self._select(coords_dev),
            self._select(self._coords),
        )  # (K,)
        rmsd_close = distances < self.rmsd_threshold

        if self.dedup_mode == "crest":
            de_close = (self._energies - float(energy)).abs() < self.energy_threshold_eV
            query_rot = _inertia_eigvals(coords_dev, self._masses)
            rot_diff = _max_relative_eig_diff(query_rot, self._rotconsts)
            rot_close = rot_diff < self.rotconst_anisotropy_threshold
            same_basin = rmsd_close & de_close & rot_close
        else:
            same_basin = rmsd_close

        if same_basin.any():
            # Among the matches, return the closest by RMSD.
            masked = torch.where(
                same_basin,
                distances,
                torch.full_like(distances, float("inf")),
            )
            min_dist, min_idx = masked.min(dim=0)
            return int(min_idx.item()), float(min_dist.item())
        return None, float(distances.min().item())

    def query_novelty_batch(self, coords_batch: torch.Tensor, energies=None) -> tuple:
        """
        Batched novelty query for many candidate conformers in one pass.

        For a batch of B candidate coordinates, returns the closest
        stored-basin index per candidate (`NOVEL = -1` for novel ones)
        and the closest-distance tensor. Used by the parallel-walkers
        driver (Step 6) where many proposals are screened per MC step.

        Params:
            coords_batch: torch.Tensor (B, n_atoms, 3) : candidate coords
            energies: torch.Tensor | array-like | None : (B,) per-candidate
                energies in eV. Required when `dedup_mode='crest'`,
                ignored when 'kabsch'.
        Returns:
            tuple[torch.Tensor, torch.Tensor] : (indices, distances).
                `indices` is an int64 tensor of shape (B,) where -1
                marks candidates that fail the active dedup criteria
                against every stored basin. `distances` is a float64
                tensor of shape (B,) with the closest Kabsch RMSD per
                candidate; `inf` for every entry when memory is empty.
        """
        if coords_batch.dim() != 3 or coords_batch.shape[1:] != (self.n_atoms, 3):
            raise ValueError(
                f"coords_batch must be (B, {self.n_atoms}, 3), got {tuple(coords_batch.shape)}"
            )
        if self.dedup_mode == "crest" and energies is None:
            raise ValueError(
                "energies is required for query_novelty_batch when dedup_mode='crest'"
            )
        b = int(coords_batch.shape[0])
        if self.n_basins == 0:
            return (
                torch.full((b,), self.NOVEL, dtype=torch.int64, device=self.device),
                torch.full((b,), math.inf, dtype=torch.float64, device=self.device),
            )
        coords_dev = coords_batch.to(self.device, dtype=torch.float64)
        distances = _kabsch_rmsd_pairwise(
            self._select(coords_dev),  # (B, n_select, 3)
            self._select(self._coords),  # (K, n_select, 3)
        )  # (B, K)
        rmsd_close = distances < self.rmsd_threshold

        if self.dedup_mode == "crest":
            e_t = torch.as_tensor(energies, dtype=torch.float64, device=self.device)
            if e_t.shape != (b,):
                raise ValueError(
                    f"energies must have shape ({b},), got {tuple(e_t.shape)}"
                )
            de = (self._energies.unsqueeze(0) - e_t.unsqueeze(1)).abs()  # (B, K)
            de_close = de < self.energy_threshold_eV
            query_rot = _inertia_eigvals(coords_dev, self._masses)  # (B, 3)
            # Per (b, k): max-rel-diff between query_rot[b] and self._rotconsts[k].
            diff = (
                self._rotconsts.unsqueeze(0) - query_rot.unsqueeze(1)
            ).abs()  # (B, K, 3)
            denom = torch.maximum(
                self._rotconsts.unsqueeze(0).abs(), query_rot.unsqueeze(1).abs()
            ).clamp(min=1e-12)
            rot_diff = (diff / denom).max(dim=-1).values  # (B, K)
            rot_close = rot_diff < self.rotconst_anisotropy_threshold
            same_basin = rmsd_close & de_close & rot_close
        else:
            same_basin = rmsd_close

        masked = torch.where(
            same_basin,
            distances,
            torch.full_like(distances, float("inf")),
        )
        min_masked, min_match_idx = masked.min(dim=1)
        any_match = ~torch.isinf(min_masked)
        min_dist = distances.min(dim=1).values
        indices = torch.where(
            any_match,
            min_match_idx,
            torch.tensor(self.NOVEL, dtype=torch.int64, device=self.device),
        )
        return indices, min_dist

    def record_visit(self, idx: int) -> None:
        """
        Increment the visit counter for basin `idx`.

        Params:
            idx: int : basin index in [0, n_basins)
        Returns:
            None
        Raises:
            IndexError: if idx is out of range.
        """
        if not (0 <= idx < self.n_basins):
            raise IndexError(f"basin index {idx} out of range [0, {self.n_basins})")
        self._usages[idx] += 1

    def acceptance_bias(self, idx) -> float:
        """
        Saunders 1/usage^p acceptance-bias factor for a proposed basin.

        Returns 1.0 (no bias) when `idx` is None — a novel basin is
        always at "first visit" relative to memory, so we don't suppress
        its acceptance. For known basins, returns
        `1 / usage[idx]^saunders_exponent`. Saunders 1990 used p=0.5
        (the default here, decays slowly so re-discovered basins
        remain reachable). Lever C15 of docs/mcmm_plan.md raises p
        toward 1 to push walkers out of deep wells faster — useful
        when Cartesian-kicks (Step 12) keep finding deep traps that
        the default 1/√usage doesn't escape.

        Params:
            idx: int | None : basin index from `query_novelty`, or None
                for a novel basin
        Returns:
            float : multiplicative acceptance-probability factor in (0, 1]
        """
        if idx is None:
            return 1.0
        if not (0 <= idx < self.n_basins):
            raise IndexError(f"basin index {idx} out of range [0, {self.n_basins})")
        usage = int(self._usages[idx].item())
        return 1.0 / (usage**self.saunders_exponent)

    def _validate_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Shape-check single-conformer coordinates."""
        if coords.shape != (self.n_atoms, 3):
            raise ValueError(
                f"coords must be ({self.n_atoms}, 3), got {tuple(coords.shape)}"
            )
        return coords


# ---------------------------------------------------------------------------
# Single-walker MCMM driver
# ---------------------------------------------------------------------------


class MCMMWalker:
    """
    Sequential single-walker MCMM driver: Metropolis accept/reject with
    Saunders 1/√usage bias and Wu-Deem detailed-balance correction.

    Generic over the proposal mechanism. The walker tracks `(coords,
    energy)` and asks an injected `propose_fn` for new states. The
    proposer is responsible for the geometry move, MMFF minimisation,
    and energy scoring; the walker is responsible only for acceptance
    decisions and memory bookkeeping. This separation keeps the MC
    logic unit-testable without an RDKit mol or MMFF backend — the
    real RDKit-coupled proposer wires in at Step 5b / 6.

    Per `step()`:

        new_coords, new_energy, det_j, success = propose_fn(self.coords)
        if not success: return False  # geometry rejection
        idx, _ = memory.query_novelty(new_coords)
        bias = memory.acceptance_bias(idx)
        p = min(1, exp(-ΔE / kT) * bias * det_j)
        if random_fn() < p:
            accept; update memory (add or record_visit)
            return True
        return False

    Special temperature limits:
      * `kt = 0`   pure greedy descent. Uphill always rejects; downhill
                   always accepts (the Saunders bias is dominated by
                   the infinite energy term in the limit).
      * `kt = inf` energy term collapses to 1; bias × det_j is the only
                   gate. Approaches uniform sampling over basins as
                   bias → 1 (i.e. on novel basins).

    Initial state handling: on construction, the walker queries memory
    for the initial coordinates. If novel (the typical fresh-memory
    case), the basin is added with usage=1. If the memory already
    contains the initial basin (e.g. shared across walkers), the
    walker latches onto the existing index without incrementing
    usage — so passing the same `BasinMemory` to N walkers does not
    artificially inflate the discovery basin's count.

    Params:
        coords: torch.Tensor (n_atoms, 3) : initial conformer
            coordinates. Stored as float64 on the memory's device.
        energy: float : initial conformer energy (eV by convention)
        kt: float : Boltzmann temperature in energy units (kT in eV
            for MACE-OFF compatibility, or any consistent unit). Use
            0 for greedy descent and `math.inf` for uniform sampling.
        memory: BasinMemory : shared basin memory; mutated by accept
        random_fn: callable | None : zero-argument callable returning
            a uniform random float in [0, 1). Defaults to
            `np.random.default_rng().random`. Tests inject a fixed
            value to make accept/reject deterministic.
    """

    def __init__(
        self,
        coords: torch.Tensor,
        energy: float,
        kt: float,
        memory: BasinMemory,
        random_fn=None,
    ):
        if kt < 0:
            raise ValueError(f"kt must be non-negative, got {kt}")
        self.coords = coords.to(memory.device, dtype=torch.float64).clone()
        self.energy = float(energy)
        self.kt = float(kt)
        self.memory = memory
        self._random = (
            random_fn if random_fn is not None else np.random.default_rng().random
        )
        self.n_steps = 0
        self.n_accepted = 0
        # Latch onto the initial basin in memory. Add only if it is novel —
        # avoids double-counting when a shared memory already contains it.
        # `energy` is forwarded so crest-mode memory can apply its energy
        # criterion; kabsch-mode ignores it.
        idx, _ = memory.query_novelty(self.coords, energy=self.energy)
        if idx is None:
            self.current_basin_idx = memory.add_basin(self.coords, self.energy)
        else:
            self.current_basin_idx = idx

    @property
    def acceptance_rate(self) -> float:
        """Fraction of attempted steps that were accepted."""
        return self.n_accepted / max(self.n_steps, 1)

    def step(self, propose_fn) -> bool:
        """
        Run one MC step.

        Convenience wrapper around `apply_proposal` for the single-walker
        case where the proposal is generated on demand from the walker's
        current coordinates. The parallel driver bypasses this and calls
        `apply_proposal` directly with proposals batched across walkers.

        Params:
            propose_fn: callable : `propose_fn(coords) -> (new_coords,
                new_energy, det_j, success)`. `new_coords` is a torch
                tensor of shape (n_atoms, 3). `det_j > 0` is the
                Wu-Deem Jacobian factor. `success=False` signals
                geometric infeasibility (the move is rejected at the
                geometry stage, before energy scoring).
        Returns:
            bool : True iff the move was accepted and the walker state
                advanced.
        """
        new_coords, new_energy, det_j, success = propose_fn(self.coords)
        return self.apply_proposal(new_coords, new_energy, det_j, success)

    def apply_proposal(
        self,
        new_coords: torch.Tensor,
        new_energy: float,
        det_j: float,
        success: bool,
    ) -> bool:
        """
        Apply a precomputed proposal: query memory, run Metropolis
        accept/reject, update state on accept.

        Separated from `step` so the parallel driver can batch the
        proposal-generation stage (DBT moves + MMFF + MACE on GPU
        across N walkers) and dispatch the per-walker accept/reject
        decisions in a loop. Single-walker callers normally use
        `step(propose_fn)`; this is the underlying primitive.

        Params:
            new_coords: torch.Tensor (n_atoms, 3) : proposed coords
            new_energy: float : proposed energy in the same units as
                `self.energy`
            det_j: float : Wu-Deem Jacobian factor (must be > 0 for
                accept; 0 forces rejection)
            success: bool : whether the geometry move closed the ring
                successfully (False means rejection without consulting
                memory or energy)
        Returns:
            bool : True iff the move was accepted.
        """
        self.n_steps += 1
        if not success:
            return False

        proposed_idx, _ = self.memory.query_novelty(
            new_coords, energy=float(new_energy)
        )
        bias = self.memory.acceptance_bias(proposed_idx)
        delta_e = float(new_energy) - self.energy
        p_accept = self._acceptance_prob(delta_e, bias, float(det_j))

        if self._random() < p_accept:
            self.coords = new_coords.to(self.memory.device, dtype=torch.float64).clone()
            self.energy = float(new_energy)
            if proposed_idx is None:
                self.current_basin_idx = self.memory.add_basin(self.coords, self.energy)
            else:
                self.memory.record_visit(proposed_idx)
                self.current_basin_idx = proposed_idx
            self.n_accepted += 1
            return True
        return False

    def run(self, n_steps: int, propose_fn) -> int:
        """
        Run `n_steps` MC steps.

        Params:
            n_steps: int : number of step() calls to make
            propose_fn: callable : same as `step`
        Returns:
            int : number of accepted moves during this call (not
                cumulative — for the cumulative count use
                `walker.n_accepted`).
        """
        n_accepted_before = self.n_accepted
        for _ in range(n_steps):
            self.step(propose_fn)
        return self.n_accepted - n_accepted_before

    def _acceptance_prob(self, delta_e: float, bias: float, det_j: float) -> float:
        """
        Metropolis-Hastings acceptance probability with Saunders bias and
        Wu-Deem Jacobian.

        Handles the kt = 0 and kt = inf limits explicitly to avoid
        overflow / NaN from `exp(-ΔE / kT)` at the boundary.
        """
        if self.kt == 0.0:
            if delta_e > 0.0:
                return 0.0
            if delta_e < 0.0:
                # Energy factor → ∞ at T=0 for downhill moves; the
                # min(1, ∞ × bias × det_j) clamps to 1 since both bias
                # and det_j are strictly positive.
                return 1.0
            # delta_e == 0: energy factor is exactly 1 in the limit.
            return min(1.0, bias * det_j)
        if math.isinf(self.kt):
            energy_factor = 1.0
        else:
            # np.exp returns inf on overflow rather than raising; keeps
            # the boundary case (very downhill move at small kt)
            # numerically clean.
            energy_factor = float(np.exp(-delta_e / self.kt))
        return min(1.0, energy_factor * bias * det_j)


# ---------------------------------------------------------------------------
# Parallel walkers — batched MMFF/MACE across N walkers
# ---------------------------------------------------------------------------


class ParallelMCMMDriver:
    """
    Coordinates N MCMM walkers sharing one BasinMemory, batching the
    GPU-bound stages (MMFF minimisation + MACE scoring) across walkers
    so each MC step costs roughly one GPU call rather than N.

    Per `step()`:

      1. Collect every walker's current coordinates.
      2. Hand them to a single `batch_propose_fn` call. The batch
         proposer is responsible for: per-walker DBT move generation
         (CPU, sequential), batched MMFF on the resulting N candidate
         conformers (one GPU call), batched MACE scoring (one GPU
         call), and per-walker Wu-Deem Jacobian estimates. It returns
         a list of N (new_coords, new_energy, det_j, success) tuples.
      3. Walk the list in order, dispatching each tuple to the
         corresponding walker's `apply_proposal`.

    Step 3 is sequential by design: walker `i` sees memory updates
    from walker `j < i` made earlier in the same step. This matters
    when two walkers propose into the same novel basin: the first to
    accept creates the basin (with usage = 1); the second sees it as
    a known basin and increments instead of duplicating. The
    alternative — buffering all proposals and committing to memory
    atomically at end of step — would let two walkers create
    duplicate basins for the same conformation. Sequential dispatch
    is the standard MCMM-with-shared-memory convention.

    The walker list defines the dispatch order. For deterministic
    behaviour across runs, pre-seed each walker's `random_fn`
    explicitly.

    Real-use note: at Step 8, `batch_propose_fn` will be a closure
    over the shared RDKit mol that stages all N walker conformers as
    distinct conformer IDs on one mol, runs
    `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` in one
    pass, and dispatches MACE scoring through the existing
    `_mace_batch_energies` path. For unit testing, a synthetic
    `batch_propose_fn` mock is enough to verify the orchestration
    contract (this module's job).

    Params:
        walkers: list[MCMMWalker] : the walkers to drive. Must share
            the same `BasinMemory` instance — the driver does not
            check this; the test for shared-state behaviour is the
            caller's responsibility.
        batch_propose_fn: callable : `batch_propose_fn(coords_list)
            -> list[(new_coords, new_energy, det_j, success)]`. The
            list lengths must match (N in, N out); otherwise
            `step()` raises ValueError.
    """

    def __init__(self, walkers: list, batch_propose_fn):
        if not walkers:
            raise ValueError("walkers list must be non-empty")
        self.walkers = list(walkers)
        self.batch_propose_fn = batch_propose_fn
        self.n_steps = 0

    @property
    def n_walkers(self) -> int:
        return len(self.walkers)

    @property
    def n_accepted(self) -> int:
        """Cumulative accepts across all walkers (this driver's lifetime)."""
        return sum(w.n_accepted for w in self.walkers)

    def step(self) -> list:
        """
        Run one MC step across every walker.

        Returns:
            list[bool] : per-walker accept/reject result, in walker order.
        Raises:
            ValueError: if the batch proposer returns a list of the
                wrong length.
        """
        coords_list = [w.coords for w in self.walkers]
        proposals = self.batch_propose_fn(coords_list)
        if len(proposals) != self.n_walkers:
            raise ValueError(
                f"batch_propose_fn returned {len(proposals)} proposals; "
                f"expected {self.n_walkers}"
            )
        self.n_steps += 1
        return [
            walker.apply_proposal(*proposal)
            for walker, proposal in zip(self.walkers, proposals)
        ]

    def run(self, n_steps: int) -> int:
        """
        Run `n_steps` MC steps across every walker.

        Params:
            n_steps: int : number of step() calls to make
        Returns:
            int : total accepts across all walkers during this call
                (not cumulative; for cumulative use `self.n_accepted`).
        """
        n_accepted_before = self.n_accepted
        for _ in range(n_steps):
            self.step()
        return self.n_accepted - n_accepted_before


# ---------------------------------------------------------------------------
# Replica exchange — temperature ladder over the parallel driver
# ---------------------------------------------------------------------------


def _swap_walker_configs(walker_a: MCMMWalker, walker_b: MCMMWalker) -> None:
    """
    Exchange `(coords, energy, current_basin_idx)` between two walkers.

    Per-walker counters (`n_steps`, `n_accepted`) and temperature (`kt`)
    stay with the walker — only the configuration moves. This preserves
    "per-temperature provenance": after many swaps, walker at slot
    (temp_idx, walker_idx) still represents the trajectory at
    `kts[temp_idx]`, even though individual configurations have hopped
    across temperatures.

    Memory is unaffected: the swapped basin indices reference the same
    shared `BasinMemory`, so the `current_basin_idx` field stays valid
    after the swap. No usage counters are incremented (a swap is not a
    re-discovery).
    """
    walker_a.coords, walker_b.coords = walker_b.coords, walker_a.coords
    walker_a.energy, walker_b.energy = walker_b.energy, walker_a.energy
    walker_a.current_basin_idx, walker_b.current_basin_idx = (
        walker_b.current_basin_idx,
        walker_a.current_basin_idx,
    )


class ReplicaExchangeMCMMDriver:
    """
    Replica-exchange MCMM driver. Wraps a fixed temperature ladder with
    one or more walkers per temperature, all sharing one BasinMemory,
    and periodically attempts configuration swaps between adjacent
    temperatures via the standard REMD Metropolis criterion.

    Architecture:
      * `walkers_by_temp[t][i]` is the i-th walker at temperature index
        t. Temperatures must be strictly increasing (kt[0] < kt[1] < …),
        and every walker at temperature t must have `kt == kts[t]`.
      * Per `step()`: every walker proposes via `batch_propose_fn` (one
        GPU call across all walkers, same as `ParallelMCMMDriver`), then
        if `n_steps % swap_interval == 0` the driver attempts a swap
        between every adjacent (t, t+1) temperature pair, paired by
        within-temperature walker index. With M walkers per temperature
        and T temperatures, this is (T - 1) × M swap attempts per swap
        event.
      * Swap criterion: `p_swap = min(1, exp((β_{t+1} - β_t) × (E_{t+1}
        - E_t)))`. When the high-T configuration has higher energy than
        the low-T one (the typical regime), the argument is negative and
        the swap accepts conditionally; when reversed (favorable swap),
        always accept.
      * Swap formulation: configurations exchange between slots while
        slot temperatures and per-walker counters stay put — see
        `_swap_walker_configs`.

    Step 8 will configure this driver with 8 temperatures geometric
    300 K → 600 K and N=8 walkers per temperature (64 total walkers,
    matched compute budget against `get_mol_PE_exhaustive`'s 10 000
    seeds at 200 steps per walker).

    Params:
        walkers_by_temp: list[list[MCMMWalker]] : walkers grouped by
            temperature, low-to-high. All groups must have the same
            length M (same number of walkers per temperature). All
            walkers in group t must share the same `kt` value, and
            `kt` must be strictly increasing across groups.
        batch_propose_fn: callable : same contract as
            `ParallelMCMMDriver.batch_propose_fn` — takes the flat list
            of all walkers' coords (low-to-high temp, then by
            within-temp index) and returns one proposal per walker.
        swap_interval: int : number of `step()` calls between swap
            attempts. Default 20 matches the literature convention.
        swap_random_fn: callable | None : zero-argument callable
            returning a uniform random float in [0, 1) for swap
            accept/reject. Defaults to a fresh
            `np.random.default_rng().random`. Tests inject a fixed
            value to make swap decisions deterministic.
    """

    def __init__(
        self,
        walkers_by_temp: list,
        batch_propose_fn,
        swap_interval: int = 20,
        swap_random_fn=None,
    ):
        if not walkers_by_temp:
            raise ValueError("walkers_by_temp must be non-empty")
        n_per_temp = len(walkers_by_temp[0])
        if n_per_temp == 0:
            raise ValueError("each temperature group must have at least one walker")
        for t, group in enumerate(walkers_by_temp):
            if len(group) != n_per_temp:
                raise ValueError(
                    "every temperature group must have the same number of "
                    f"walkers; group {t} has {len(group)}, expected {n_per_temp}"
                )
        self.kts = [group[0].kt for group in walkers_by_temp]
        for t, group in enumerate(walkers_by_temp):
            for w in group:
                if w.kt != self.kts[t]:
                    raise ValueError(
                        f"walkers in temperature group {t} have inconsistent kt: "
                        f"expected {self.kts[t]}, found {w.kt}"
                    )
        for t in range(len(self.kts) - 1):
            if not (self.kts[t] < self.kts[t + 1]):
                raise ValueError(
                    "kts must be strictly increasing across temperature groups; "
                    f"kts[{t}]={self.kts[t]} ≥ kts[{t+1}]={self.kts[t+1]}"
                )
        if swap_interval < 1:
            raise ValueError(f"swap_interval must be >= 1, got {swap_interval}")

        self.walkers_by_temp = [list(group) for group in walkers_by_temp]
        self.batch_propose_fn = batch_propose_fn
        self.swap_interval = swap_interval
        self._swap_random = (
            swap_random_fn
            if swap_random_fn is not None
            else np.random.default_rng().random
        )
        self.n_steps = 0
        self.n_swap_attempts = 0
        self.n_swap_accepted = 0
        self._flat_walkers = [w for group in self.walkers_by_temp for w in group]

    @property
    def n_temperatures(self) -> int:
        return len(self.walkers_by_temp)

    @property
    def n_walkers_per_temp(self) -> int:
        return len(self.walkers_by_temp[0])

    @property
    def n_walkers(self) -> int:
        return len(self._flat_walkers)

    @property
    def n_accepted(self) -> int:
        """Cumulative accepts across all walkers at all temperatures."""
        return sum(w.n_accepted for w in self._flat_walkers)

    @property
    def swap_acceptance_rate(self) -> float:
        """Fraction of attempted swaps that succeeded."""
        return self.n_swap_accepted / max(self.n_swap_attempts, 1)

    def step(self) -> list:
        """
        Run one MC step across every walker, attempting swaps if the
        step count is a multiple of `swap_interval`.

        Returns:
            list[bool] : per-walker accept/reject result, in flat order
                (low-temp first, then by within-temp walker index).
        Raises:
            ValueError: if the batch proposer returns the wrong number
                of proposals.
        """
        coords_list = [w.coords for w in self._flat_walkers]
        proposals = self.batch_propose_fn(coords_list)
        if len(proposals) != self.n_walkers:
            raise ValueError(
                f"batch_propose_fn returned {len(proposals)} proposals; "
                f"expected {self.n_walkers}"
            )
        results = [
            walker.apply_proposal(*proposal)
            for walker, proposal in zip(self._flat_walkers, proposals)
        ]
        self.n_steps += 1
        if self.n_steps % self.swap_interval == 0:
            self.attempt_swaps()
        return results

    def attempt_swaps(self) -> list:
        """
        Attempt one swap between every adjacent (t, t+1) temperature
        pair, paired by within-temperature walker index.

        Returns:
            list[bool] : per-attempt swap result, in scan order
                (t = 0..T-2, then i = 0..M-1). Length is (T-1) × M.
        """
        results: list = []
        for t in range(self.n_temperatures - 1):
            kt_low, kt_high = self.kts[t], self.kts[t + 1]
            for i in range(self.n_walkers_per_temp):
                w_low = self.walkers_by_temp[t][i]
                w_high = self.walkers_by_temp[t + 1][i]
                p_swap = self._swap_acceptance_prob(w_low, w_high, kt_low, kt_high)
                self.n_swap_attempts += 1
                if self._swap_random() < p_swap:
                    _swap_walker_configs(w_low, w_high)
                    self.n_swap_accepted += 1
                    results.append(True)
                else:
                    results.append(False)
        return results

    def run(self, n_steps: int) -> int:
        """
        Run `n_steps` MC steps across the ladder.

        Params:
            n_steps: int : number of step() calls (each may trigger
                swap attempts based on `swap_interval`).
        Returns:
            int : total accepts across all walkers during this call
                (not cumulative — for cumulative use `self.n_accepted`).
        """
        n_accepted_before = self.n_accepted
        for _ in range(n_steps):
            self.step()
        return self.n_accepted - n_accepted_before

    def _swap_acceptance_prob(
        self,
        w_low: MCMMWalker,
        w_high: MCMMWalker,
        kt_low: float,
        kt_high: float,
    ) -> float:
        """
        Standard REMD swap acceptance probability:
        `p = min(1, exp((β_high − β_low)(E_high − E_low)))`.

        kt_low and kt_high must both be strictly positive and finite;
        the constructor enforces this via the strictly-increasing check
        on the temperature ladder.
        """
        beta_low = 1.0 / kt_low
        beta_high = 1.0 / kt_high
        delta_beta = beta_high - beta_low
        delta_e = w_high.energy - w_low.energy
        arg = delta_beta * delta_e
        if arg >= 0.0:
            return 1.0
        return float(np.exp(arg))


# ---------------------------------------------------------------------------
# Real-mol proposer factory
# ---------------------------------------------------------------------------


def make_mcmm_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    drive_sigma_rad: float = 0.1,
    closure_tol: float = 0.01,
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` for `ReplicaExchangeMCMMDriver` that
    proposes DBT moves on the backbone windows of `mol`, batches MMFF +
    MACE across walkers per call, and returns per-walker
    `(new_coords, new_energy, det_j, success)` tuples.

    Per-call pipeline:

      1. **Per-walker move generation** (CPU, sequential): pick a random
         backbone window, drive dihedral, and `drive_delta ~ N(0,
         drive_sigma_rad²)`. Run `concerted_rotation.propose_move` on
         the 7-atom backbone window positions to solve for the closure
         deltas. Walkers whose closure fails are flagged.
      2. **Full-mol coordinate update** (CPU, sequential): for
         successful walkers, replay the per-dihedral rotations on the
         full atom array via
         `concerted_rotation.apply_dihedral_changes_full_mol`,
         transporting side-chain atoms rigidly with their backbone
         parents according to the precomputed
         `_compute_window_downstream_sets` for the chosen window.
      3. **Batched MMFF** (GPU, one call): stage every successful
         candidate as a conformer on a shared throwaway mol and run
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`
         (`mmff_backend='gpu'`) or RDKit's serial MMFF
         (`mmff_backend='cpu'`) for in-place minimisation.
      4. **Batched MACE** (GPU, chunked): score every minimised
         candidate via `_mace_batch_energies` in chunks of
         `score_chunk_size`.
      5. **Return** `(coords_tensor, energy_float, det_j_float,
         success_bool)` per walker, in walker order. Failed walkers
         pass through with their pre-move coords and `success=False`
         so the driver's `apply_proposal` rejects without further work.

    Topology — backbone windows, side-chain groups, the throwaway-mol
    template — is captured at factory build time and reused per call,
    keeping the per-step CPU overhead bounded by the move-generation
    loop and the conformer-staging step.

    Lazy import of `_mace_batch_energies` from `confsweeper` avoids the
    confsweeper → mcmm circular dependency at module load time.

    Params:
        mol: Chem.Mol : a head-to-tail cyclic peptide with explicit Hs.
            Topology is captured at factory-build time; the mol must
            not be mutated structurally afterwards (conformer additions
            and edits are fine).
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only consulted when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        drive_sigma_rad: float : Gaussian standard deviation for the
            drive-angle perturbation in radians (default 0.1 ≈ 5.7°).
            Larger values give bigger moves at lower closure-success
            rate; couples to closure_tol per docs/mcmm_plan.md.
        closure_tol: float : passed through to `propose_move` as the
            maximum r5+r6 displacement-norm tolerated as ring-closed.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500, matches `_minimize_score_filter_dedup`).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA, default) or
            'cpu' (RDKit serial). MMFF runs on the throwaway mol.
        seed: int : base seed for the move-RNG; deterministic across
            replicate runs.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]` matching
            the contract expected by `ParallelMCMMDriver` and
            `ReplicaExchangeMCMMDriver`.
    Raises:
        ValueError: if `mol` has no enumerable backbone windows (input
            is not a cyclic peptide of ≥ 3 residues).
    """
    from concerted_rotation import (
        N_DIHEDRALS,
        apply_dihedral_changes_full_mol,
        propose_move,
    )

    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )

    windows = enumerate_backbone_windows(mol)
    if not windows:
        raise ValueError(
            "mol has no enumerable backbone windows; "
            "check that it is a head-to-tail cyclic peptide of at least 3 residues"
        )
    backbone_atoms = _backbone_atom_set(mol)
    window_downstream_sets = [
        _compute_window_downstream_sets(mol, w, backbone_atoms) for w in windows
    ]
    n_windows = len(windows)
    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    # Throwaway-mol template: structure-only, no conformers. Cloning per
    # call is required because nvmolkit MMFF mutates conformers in place
    # and we don't want to corrupt walker state across step() calls.
    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    rng = np.random.default_rng(seed)

    # Per-call cumulative diagnostic counters. Attached to the returned
    # proposer function as `.stats` so callers (`get_mol_PE_mcmm`, tests)
    # can read them after the run to diagnose acceptance regressions —
    # especially the "1 basin" pathology on small peptides where DBT
    # closure fails on most moves.
    stats = {
        "n_proposed": 0,
        "n_closure_failures": 0,
        "n_closure_successes": 0,
    }

    def batch_propose_fn(coords_list):
        # Lazy import: confsweeper imports from mcmm at module load time;
        # importing _mace_batch_energies here defers resolution until the
        # closure is actually called, breaking the circular dependency.
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        stats["n_proposed"] += n_walkers

        # Stage 1: per-walker DBT closure on the backbone window.
        # Successful walkers contribute a (new_full_coords, det_j) entry.
        successful_meta: list = []
        success_walker_indices: list = []
        for w_idx, coords in enumerate(coords_list):
            window_idx = int(rng.integers(n_windows))
            window = windows[window_idx]
            drive_idx = int(rng.integers(N_DIHEDRALS))
            drive_delta = float(rng.normal(0.0, drive_sigma_rad))

            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            window_pos = coords_np[list(window)]
            result = propose_move(
                window_pos, drive_idx, drive_delta, closure_tol=closure_tol
            )
            if not result.success:
                continue
            new_full = apply_dihedral_changes_full_mol(
                coords_np,
                list(window),
                result.deltas,
                window_downstream_sets[window_idx],
            )
            successful_meta.append(
                {"new_full": new_full, "det_j": float(result.det_jacobian)}
            )
            success_walker_indices.append(w_idx)

        stats["n_closure_successes"] += len(successful_meta)
        stats["n_closure_failures"] += n_walkers - len(successful_meta)

        # Stage 2: short-circuit if every walker failed.
        if not successful_meta:
            return [(coords_list[i], 0.0, 0.0, False) for i in range(n_walkers)]

        # Stage 3: stage successful candidates as conformers on a fresh
        # throwaway mol, run batched MMFF.
        throwaway = Chem.Mol(template_mol)
        for meta in successful_meta:
            conf = Chem.Conformer(n_atoms)
            new_full = meta["new_full"]
            for a_idx in range(n_atoms):
                x, y, z = new_full[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)

        if mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in [c.GetId() for c in throwaway.GetConformers()]:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 4: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 5: assemble per-walker proposals in walker order.
        proposals: list = [None] * n_walkers
        for slot, w_idx in enumerate(success_walker_indices):
            cid = post_mmff_conf_ids[slot]
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            proposals[w_idx] = (
                new_coords,
                float(energies[slot]),
                successful_meta[slot]["det_j"],
                True,
            )
        # Failed walkers get a no-op proposal in their original slot.
        for w_idx in range(n_walkers):
            if proposals[w_idx] is None:
                proposals[w_idx] = (coords_list[w_idx], 0.0, 0.0, False)

        return proposals

    # Expose cumulative stats so the orchestrator (get_mol_PE_mcmm) can
    # read closure-failure rates and diagnose "1 basin" regressions.
    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Cartesian-kick proposer — GOAT-style topology-preserving move
# ---------------------------------------------------------------------------


def make_cartesian_kick_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    sigma_kick_a: float = 0.1,
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that applies an isotropic Gaussian kick
    to all atom positions, MMFF-relaxes, and MACE-scores. The
    GOAT-flavour move type: complements DBT's dihedral-space
    parameterisation by reaching geometries that small dihedral
    perturbations can't (side-chain rotamer flips, depsipeptide ester
    rearrangements, etc.).

    Per-call pipeline:

      1. **Per-walker kick** (CPU, vectorised): add `N(0, sigma_kick_a²)`
         noise independently to every atom-coordinate of every walker.
         No closure step or backbone parameterisation; rely on MMFF to
         pull bond lengths and ring sp² angles back to equilibrium.
      2. **Batched MMFF** (GPU, one call): stage every kicked candidate
         as a conformer on a shared throwaway mol and run
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`.
      3. **Batched MACE** (GPU, chunked) via `_mace_batch_energies`.
      4. **Return** `(coords_tensor, energy_float, det_j=1.0,
         success=True)` per walker. The Gaussian kick is symmetric in
         coordinate space so detailed balance needs no Jacobian
         correction; the post-MMFF state is treated as the proposal
         outcome regardless of the relaxation trajectory (the same
         convention `make_mcmm_proposer` uses).

    Topology preservation is approximate, not strict: GOAT freezes
    bonds and ring sp² angles via constraints during the uphill push,
    while we let MMFF relax them. For peptide cyclisations and N-Me
    bonds this is fine — MMFF's bond-stretch term has a steep gradient
    that pulls covalent bonds back to equilibrium even from large
    kicks. For sigma_kick_a beyond ~0.3 Å, expect occasional MMFF
    non-convergence or bond-breaking; tune accordingly.

    Params:
        mol: Chem.Mol : reference molecule with explicit Hs. Topology
            captured at factory build time; do not mutate structurally.
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only used when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        sigma_kick_a: float : Gaussian standard deviation in Å applied
            independently to every atom-coordinate (default 0.1 Å).
            Treat as the move-magnitude analogue of DBT's
            `drive_sigma_rad`.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA) or 'cpu'
            (RDKit serial).
        seed: int : RNG seed for the per-step kicks.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`
            matching the `ParallelMCMMDriver` /
            `ReplicaExchangeMCMMDriver` proposer contract.
    Raises:
        ValueError: on unknown `mmff_backend`.
    """
    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )

    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    rng = np.random.default_rng(seed)
    stats = {"n_proposed": 0, "n_relax_failures": 0, "n_relax_successes": 0}

    def batch_propose_fn(coords_list):
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        if n_walkers == 0:
            return []
        stats["n_proposed"] += n_walkers

        # Stage 1: per-walker isotropic Gaussian kick.
        kicked_coords: list = []
        for coords in coords_list:
            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            kick = rng.normal(0.0, sigma_kick_a, size=coords_np.shape)
            kicked_coords.append(coords_np + kick)

        # Stage 2: batched MMFF on a fresh throwaway mol.
        throwaway = Chem.Mol(template_mol)
        for kc in kicked_coords:
            conf = Chem.Conformer(n_atoms)
            for a_idx in range(n_atoms):
                x, y, z = kc[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)

        if mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in [c.GetId() for c in throwaway.GetConformers()]:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 3: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 4: assemble per-walker proposals. det_j=1.0 for the
        # symmetric Gaussian kick. Walkers whose post-MMFF energy is
        # non-finite (MMFF blow-up — rare but possible at large
        # sigma_kick_a) get success=False so the driver rejects.
        proposals: list = []
        for slot, cid in enumerate(post_mmff_conf_ids):
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            e = float(energies[slot])
            if not np.isfinite(e):
                stats["n_relax_failures"] += 1
                proposals.append((coords_list[slot], 0.0, 0.0, False))
            else:
                stats["n_relax_successes"] += 1
                proposals.append((new_coords, e, 1.0, True))
        return proposals

    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Composite proposer — randomly route walkers to one of several sub-proposers
# ---------------------------------------------------------------------------


def make_composite_proposer(
    proposers,
    weights=None,
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that routes each walker to one of N
    sub-proposers per step, sampled by `weights`. Lets DBT and
    Cartesian-kick (or future move types) coexist in one MCMM run.

    Routing happens at the walker level, not the step level — so a
    single REMD step can have some walkers proposing DBT moves and
    others proposing Cartesian kicks. Walkers are partitioned by
    chosen proposer, each sub-proposer is invoked on its subset
    (preserving its internal batching), and results are reassembled
    in walker order.

    Each sub-proposer's `.stats` dict is preserved on the composite
    return value as `.stats[i]` so callers can inspect per-proposer
    diagnostics. The composite itself does not aggregate counters.

    Params:
        proposers: list[callable] : sub-proposers, each matching the
            `batch_propose_fn(coords_list) -> list[tuple]` contract.
        weights: list[float] | None : sampling weight per proposer.
            Default None means uniform. Normalised internally.
        seed: int : routing RNG seed; deterministic across replicate
            runs.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`.
            Carries `.stats = [p.stats for p in proposers]` (list,
            indexed by proposer position).
    Raises:
        ValueError: empty `proposers`, weight/proposer count mismatch,
            or any non-positive weight.
    """
    if not proposers:
        raise ValueError("proposers must be non-empty")
    if weights is None:
        weights_arr = np.full(len(proposers), 1.0 / len(proposers))
    else:
        if len(weights) != len(proposers):
            raise ValueError(
                f"weights ({len(weights)}) must match proposers ({len(proposers)})"
            )
        if any(w <= 0 for w in weights):
            raise ValueError(f"weights must be positive, got {weights}")
        w_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = w_arr / w_arr.sum()

    rng = np.random.default_rng(seed)

    def batch_propose_fn(coords_list):
        n = len(coords_list)
        if n == 0:
            return []
        choices = rng.choice(len(proposers), size=n, p=weights_arr)

        results: list = [None] * n
        for p_idx, propose_fn in enumerate(proposers):
            walker_idx = [w for w in range(n) if choices[w] == p_idx]
            if not walker_idx:
                continue
            sub_coords = [coords_list[w] for w in walker_idx]
            sub_results = propose_fn(sub_coords)
            if len(sub_results) != len(sub_coords):
                raise RuntimeError(
                    f"proposer {p_idx} returned {len(sub_results)} results for "
                    f"{len(sub_coords)} walkers"
                )
            for w, r in zip(walker_idx, sub_results):
                results[w] = r
        return results

    batch_propose_fn.stats = [p.stats for p in proposers if hasattr(p, "stats")]
    return batch_propose_fn
