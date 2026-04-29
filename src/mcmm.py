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

import numpy as np
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
        self.n_steps += 1
        new_coords, new_energy, det_j, success = propose_fn(self.coords)
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
