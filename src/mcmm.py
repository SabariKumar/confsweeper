"""
Multiple Minimum Monte Carlo (MCMM) sampler for cyclic peptide conformer
search. Issue #11; companion module to src/concerted_rotation.py, which
holds the move geometry.

This module is being built up incrementally per docs/mcmm_plan.md. Step
3 (current) implements backbone window enumeration. Future steps:

    Step 4: BasinMemory data structure (shared across walkers)
    Step 5: single-walker MCMM driver
    Step 6: parallel walkers (batched MMFF)
    Step 7: replica exchange across temperatures
    Step 8: get_mol_PE_mcmm entry point in src/confsweeper.py

The DBT-style concerted-rotation move geometry — chain rebuild, closure
solver, Wu-Deem Jacobian — lives in src/concerted_rotation.py. This
module orchestrates the moves around a real RDKit mol: enumerates valid
backbone windows, picks one per move, applies the move, and (later)
runs MMFF + accept/reject.
"""

from rdkit import Chem

from torsional_sampling import get_backbone_dihedrals

# 7 consecutive backbone atoms = 4 inner dihedrals = the chain shape
# expected by concerted_rotation.propose_move.
WINDOW_SIZE = 7


def enumerate_backbone_windows(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """
    Return every 7-atom backbone window in a head-to-tail cyclic peptide.

    Walks the macrocycle ring N → Cα → C → N → Cα → C → ... and emits
    one cyclic window per starting backbone atom. For a cyclic peptide
    of K residues there are 3K backbone atoms and 3K windows.

    Each window is a tuple of 7 atom indices in the order they appear
    around the ring. The window's atom layout matches what
    `concerted_rotation.propose_move` expects: r0..r6 are 7 consecutive
    backbone atoms with bonds r0-r1, r1-r2, ..., r5-r6 in the macrocycle.
    The 4 inner dihedrals (around bonds (1,2)..(4,5)) are what the move
    perturbs.

    The MCMM driver picks one window per move uniformly at random from
    this list. The driver may also choose to enumerate in both ring
    directions (a window read backwards is a different move); v0 only
    emits one direction.

    Params:
        mol: Chem.Mol : a head-to-tail cyclic peptide; explicit Hs are
            optional. Side chains are ignored.
    Returns:
        list of 7-tuples of atom indices. Empty if the molecule has fewer
        than 7 backbone atoms.
    Raises:
        ValueError: if the C → N walk fails to close the ring (input
            is not a head-to-tail cyclic peptide).
    """
    residues = _ordered_backbone_residues(mol)
    if not residues:
        return []

    backbone: list[int] = []
    for n_idx, ca_idx, c_idx in residues:
        backbone.extend([n_idx, ca_idx, c_idx])

    n_atoms = len(backbone)
    if n_atoms < WINDOW_SIZE:
        return []

    return [
        tuple(backbone[(start + i) % n_atoms] for i in range(WINDOW_SIZE))
        for start in range(n_atoms)
    ]


def _ordered_backbone_residues(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """
    Return (N, Cα, C) atom indices per residue, in cyclic order around
    the macrocycle.

    Order is established by walking C → N peptide bonds starting from an
    arbitrary residue (the first one returned by get_backbone_dihedrals).
    The starting residue depends on RDKit's substructure-match order, so
    the cyclic shift is not stable across different mols, but a given
    call returns the same cyclic order for the same mol.

    Params:
        mol: Chem.Mol : input molecule
    Returns:
        list of (N_idx, Cα_idx, C_idx) tuples, ordered cyclically. Empty
        if no backbone is found.
    Raises:
        ValueError: if the C → N walk fails to visit every detected
            residue (input is not a closed head-to-tail cyclic peptide).
    """
    residues = [(phi[1], phi[2], phi[3]) for phi, _ in get_backbone_dihedrals(mol)]
    if not residues:
        return []

    n_to_res = {n: (n, ca, c) for n, ca, c in residues}

    # For each residue's amide C, find the next residue's N (the one
    # bonded to this C via the peptide bond). The C atom has three
    # neighbours: the amide O (double bond), this residue's Cα, and the
    # next residue's N — we pick the N that's also a backbone N.
    next_n_for: dict[int, int] = {}
    for n_idx, _, c_idx in residues:
        c_atom = mol.GetAtomWithIdx(c_idx)
        for nb in c_atom.GetNeighbors():
            if nb.GetAtomicNum() == 7 and nb.GetIdx() in n_to_res:
                next_n_for[n_idx] = nb.GetIdx()
                break

    ordered: list[tuple[int, int, int]] = []
    visited: set[int] = set()
    start_n = residues[0][0]
    current_n: int | None = start_n
    ring_closed = False
    while current_n is not None and current_n not in visited:
        visited.add(current_n)
        ordered.append(n_to_res[current_n])
        next_n = next_n_for.get(current_n)
        if next_n == start_n:
            ring_closed = True
            break
        current_n = next_n

    # The walk "completes" under three conditions: (a) ring closure (next
    # is start_n), (b) dead end (next is None — no peptide bond out of
    # this residue), (c) revisit of an already-walked residue. Only (a)
    # is a valid head-to-tail cycle. (b) is a linear peptide; (c) would
    # indicate a branching backbone (not currently produced by the SMARTS
    # but possible in principle).
    if not ring_closed:
        raise ValueError(
            f"Backbone ring did not close: walked {len(ordered)} of "
            f"{len(residues)} residues without returning to the start. "
            "Input must be a head-to-tail cyclic peptide."
        )

    return ordered
