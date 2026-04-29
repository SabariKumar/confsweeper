"""
Multiple Minimum Monte Carlo (MCMM) sampler for cyclic peptide conformer
search. Issue #11; companion module to src/concerted_rotation.py, which
holds the move geometry.

This module is being built up incrementally per docs/mcmm_plan.md. Steps
3 and 4 (current) implement backbone window enumeration and the shared
basin-memory data structure. Future steps:

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

import math

import torch
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
        if rmsd_threshold <= 0:
            raise ValueError(f"rmsd_threshold must be positive, got {rmsd_threshold}")
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
