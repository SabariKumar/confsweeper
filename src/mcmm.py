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
