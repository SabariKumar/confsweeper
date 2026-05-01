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
# Basin memory — shared across MCMM walkers
# ---------------------------------------------------------------------------


# Default normalised-L1 distance below which two conformers are considered
# the same basin. Matches the default rmsd_threshold of
# get_mol_PE_exhaustive's _energy_ranked_dedup so basin definitions are
# consistent across the get_mol_PE_* family.
DEFAULT_RMSD_THRESHOLD = 0.1


class BasinMemory:
    """
    Shared basin memory for MCMM walkers.

    Stores one representative conformer (coordinates + MMFF energy) per
    discovered basin and tracks a per-basin visit counter for the
    Saunders 1/√usage acceptance bias. Basins are distinguished by the
    same normalised-L1 metric used by `_energy_ranked_dedup`:
    `d = Σ|Δr| / (3 * n_atoms)`. Two conformers are in the same basin
    if `d < rmsd_threshold` (strict `<`, matching the dedup primitive
    so that thresholds are interchangeable).

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
    CPU query is microseconds.

    Params:
        n_atoms: int : number of atoms per conformer (must match every
            coords tensor passed to add_basin / query_novelty)
        rmsd_threshold: float : basin-distinguishing distance, normalised
            L1 units (default 0.1, matching _energy_ranked_dedup)
        device: torch.device | str : device for stored tensors (default 'cpu')
    """

    NOVEL = -1  # sentinel for the batched query API

    def __init__(
        self,
        n_atoms: int,
        rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
        device="cpu",
    ):
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if rmsd_threshold < 0:
            raise ValueError(
                f"rmsd_threshold must be non-negative, got {rmsd_threshold}"
            )
        self.n_atoms = n_atoms
        self.rmsd_threshold = float(rmsd_threshold)
        self.device = torch.device(device)
        self._coords = torch.zeros(
            (0, n_atoms, 3), dtype=torch.float64, device=self.device
        )
        self._energies = torch.zeros((0,), dtype=torch.float64, device=self.device)
        self._usages = torch.zeros((0,), dtype=torch.int64, device=self.device)

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
        return self.n_basins - 1

    def query_novelty(self, coords: torch.Tensor) -> tuple:
        """
        Find the closest stored basin and report whether it is within
        the basin-distinguishing threshold.

        Params:
            coords: torch.Tensor (n_atoms, 3) : conformer coordinates
        Returns:
            tuple[int | None, float] : (idx, distance). `idx` is None
                when no stored basin is closer than `rmsd_threshold`;
                otherwise it is the index of the closest basin.
                `distance` is the closest distance regardless of
                threshold; `inf` when memory is empty.
        """
        coords = self._validate_coords(coords)
        if self.n_basins == 0:
            return None, math.inf
        coords_dev = coords.to(self.device, dtype=torch.float64).unsqueeze(0)
        diffs = (self._coords - coords_dev).abs()  # (K, n_atoms, 3)
        distances = diffs.sum(dim=(1, 2)) / (3 * self.n_atoms)  # (K,)
        min_dist, min_idx = distances.min(dim=0)
        min_dist_f = float(min_dist.item())
        if min_dist_f < self.rmsd_threshold:
            return int(min_idx.item()), min_dist_f
        return None, min_dist_f

    def query_novelty_batch(self, coords_batch: torch.Tensor) -> tuple:
        """
        Batched novelty query for many candidate conformers in one pass.

        For a batch of B candidate coordinates, returns the closest
        stored-basin index per candidate (`NOVEL = -1` for novel ones)
        and the closest-distance tensor. Used by the parallel-walkers
        driver (Step 6) where many proposals are screened per MC step.

        Params:
            coords_batch: torch.Tensor (B, n_atoms, 3) : candidate coords
        Returns:
            tuple[torch.Tensor, torch.Tensor] : (indices, distances).
                `indices` is an int64 tensor of shape (B,) where -1
                marks candidates outside `rmsd_threshold` of every
                stored basin. `distances` is a float64 tensor of shape
                (B,) with the closest distance per candidate; `inf` for
                every entry when memory is empty.
        """
        if coords_batch.dim() != 3 or coords_batch.shape[1:] != (self.n_atoms, 3):
            raise ValueError(
                f"coords_batch must be (B, {self.n_atoms}, 3), got {tuple(coords_batch.shape)}"
            )
        b = int(coords_batch.shape[0])
        if self.n_basins == 0:
            return (
                torch.full((b,), self.NOVEL, dtype=torch.int64, device=self.device),
                torch.full((b,), math.inf, dtype=torch.float64, device=self.device),
            )
        coords_dev = coords_batch.to(self.device, dtype=torch.float64)
        # (B, 1, n_atoms, 3) - (1, K, n_atoms, 3) → (B, K, n_atoms, 3)
        diffs = (coords_dev.unsqueeze(1) - self._coords.unsqueeze(0)).abs()
        distances = diffs.sum(dim=(2, 3)) / (3 * self.n_atoms)  # (B, K)
        min_dist, min_idx = distances.min(dim=1)  # (B,), (B,)
        novel = min_dist >= self.rmsd_threshold
        indices = torch.where(
            novel,
            torch.tensor(self.NOVEL, dtype=torch.int64, device=self.device),
            min_idx,
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
        Saunders 1/√usage acceptance-bias factor for a proposed basin.

        Returns 1.0 (no bias) when `idx` is None — a novel basin is
        always at "first visit" relative to memory, so we don't suppress
        its acceptance. For known basins, returns 1/√usage[idx], which
        decays slowly enough that re-discovered basins remain reachable
        but progressively penalised.

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
        return 1.0 / math.sqrt(int(self._usages[idx].item()))

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
        idx, _ = memory.query_novelty(self.coords)
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

        proposed_idx, _ = self.memory.query_novelty(new_coords)
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
