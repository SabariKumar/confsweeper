"""
Numerical concerted-rotation backbone moves for cyclic peptide MC.

Implements an approximation to the Dodd-Boone-Theodorou (1993) and
Coutsias (2004) concerted-rotation move set used in macrocyclic peptide
Monte Carlo. Given a 7-atom window of consecutive backbone atoms with
positions r0..r6, the move perturbs one of the four inner dihedrals — τk
for k in {0..3} where τk is the dihedral around bond (k+1, k+2) — by a
'drive' amount Δθ, then numerically adjusts the other three dihedrals
to restore the positions of the boundary atoms r5 and r6. Atoms r0, r1
remain fixed by construction (they form the upstream frame).

The original DBT/Coutsias algorithms solve the closure constraint
analytically via a degree-16 polynomial in cos τ, enabling enumeration
of up to 16 distinct closure branches per move. This module uses
scipy.optimize.least_squares instead, which finds a single closure
solution in the same homotopy branch as the current geometry. See
docs/mcmm_plan.md for the rationale and the conditions under which the
analytical version (Option A, deferred) becomes worth implementing.

Module surface:
    rotation_matrix(axis, angle_rad)              Rodrigues rotation
    dihedral_angle(p0, p1, p2, p3)                standard dihedral, radians
    apply_dihedral_changes(positions, deltas)     rigid chain rebuild
    closure_residual(positions, drive_idx, ...)   6-vector residual
    propose_move(positions, drive_idx, drive_delta)   top-level entry point

The MCMM driver calls `propose_move`. It returns
`(new_positions, det_jacobian, success)`: when `success` is False the
closure solver could not drive the residual below tolerance and the move
should be rejected geometrically (before MMFF) by the caller.
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import least_squares


class MoveProposal(NamedTuple):
    """
    Outcome of a single DBT concerted-rotation closure attempt.

    Returned by `propose_move`. Tuple-unpacking compatible (the previous
    3-tuple return was `(new_positions, det_jacobian, success)`); the
    `deltas` field was added in Step 8b so the caller can replay the
    same per-dihedral rotations on a full-mol coordinate array (with
    side-chain coupling) via `apply_dihedral_changes_full_mol`.

    Fields:
        new_positions: np.ndarray (7, 3) : new 7-atom backbone window
            geometry. Equals the input positions when `success=False`.
        det_jacobian: float : Wu-Deem |det J| at the closure solution
            (or 0.0 on failure).
        deltas: np.ndarray (4,) : the four dihedral changes applied
            (drive_delta at drive_idx; the three closure-solver
            outputs at the other indices). All zeros on failure.
        success: bool : True iff the closure residual norm fell below
            the tolerance.
    """

    new_positions: np.ndarray
    det_jacobian: float
    deltas: np.ndarray
    success: bool


# Closure tolerance: 6-residual norm (sum of squared r5 and r6 displacements)
# below this counts as a successful close. The system is generically
# over-determined (6 residuals vs. 3 free unknowns when one of the four
# dihedrals is the drive), so exact zero closure is not achievable for
# arbitrary Δθ. 1e-2 Å keeps the per-move drift well below typical MMFF
# bond-stretch tolerances and below thermal RMSD scales (~0.1 Å); MMFF
# in the MCMM inner loop relaxes the residual distortion.
#
# **Tuning lever for coverage.** Relaxing closure_tol increases the
# fraction of moves that pass the geometry check and lets larger Δθ
# values close, which directly improves basin coverage — but only up
# to a sweet spot:
#   * ≪ 0.01 Å  : only tiny Δθ closes; coverage capped by move size.
#   * ~0.01-0.1 Å : balanced — MMFF reliably recovers the targeted basin.
#   * > 0.1 Å    : MMFF may drift to a different basin than the move
#                  targeted; concerted rotation degrades toward random
#                  perturbation + MMFF basin search.
#   * ≫ 1 Å     : effectively random Cartesian noise; advantage gone.
# The lever couples to Δθ amplitude: relax both together to get larger
# directed moves; relax only closure_tol to get the same moves at
# higher acceptance with looser ring closure. Instrument both in
# benchmark logs.
DEFAULT_CLOSURE_TOL = 1e-2

# Number of inner dihedrals in a 7-atom window (atoms 0..6, bonds 0..5,
# inner bonds 1..4 each carrying one rotatable dihedral).
N_DIHEDRALS = 4


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rodrigues rotation matrix for a rotation by `angle_rad` around `axis`.

    Params:
        axis: np.ndarray (3,) : unit-norm rotation axis (caller responsibility)
        angle_rad: float : rotation angle in radians, right-hand rule
    Returns:
        np.ndarray (3, 3) : rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def dihedral_angle(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """
    Dihedral angle of four points around the bond (p1, p2), in radians.

    The result is in (-π, π]. The sign convention is internally consistent
    with `apply_dihedral_changes`: applying delta to dihedral k via that
    function changes this function's measured value of the same dihedral
    by exactly delta. The absolute sign is not standardised against any
    particular external convention (e.g. RDKit, OpenMM), only against the
    rotation primitive in this module.

    Params:
        p0, p1, p2, p3: np.ndarray (3,) : four atom positions
    Returns:
        float : dihedral angle in radians
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_hat)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def apply_dihedral_changes(positions: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """
    Apply incremental dihedral changes to a W-atom chain (W >= 4).

    Atoms r0, r1 stay fixed (the upstream-frame convention). Atoms
    r2..r(W-1) may move. A W-atom chain has W-3 inner dihedrals;
    `deltas[k]` (k in 0..W-4) is the change to the dihedral around bond
    (k+1, k+2), applied as a rigid rotation of all atoms strictly
    downstream of that bond, around the bond axis. W=7 (the DBT case,
    4 dihedrals) and W=10 (the ω-flip case, 7 dihedrals) are both used.

    The order matters: deltas are applied k=0 first, then k=1, ..., k=3.
    Each rotation acts on the geometry produced by the previous rotations,
    so subsequent bond axes are computed from the updated positions. This
    matches the natural "rebuild forward" semantics: changing τ0 first
    repositions atoms downstream of bond (1, 2), and then τ1 acts on the
    new bond (2, 3) axis, and so on.

    Bond lengths and bond angles are preserved by every step (rigid
    rotations preserve internal distances). Dihedrals τj for j != k are
    preserved when their defining atom set is entirely within the rotated
    group (j > k) or entirely outside it (j < k); the boundary case is
    j = k, where τk changes by exactly deltas[k].

    Params:
        positions: np.ndarray (W, 3) : starting atom positions, W >= 4
        deltas: np.ndarray (W-3,) : dihedral changes in radians
    Returns:
        np.ndarray (W, 3) : new atom positions
    """
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 4:
        raise ValueError(f"positions must be (W, 3) with W >= 4, got {positions.shape}")
    window_size = positions.shape[0]
    n_dih = window_size - 3
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (n_dih,):
        raise ValueError(
            f"deltas must be ({n_dih},) for a {window_size}-atom window, "
            f"got {deltas.shape}"
        )

    pos = positions.copy().astype(float)
    for k in range(n_dih):
        delta = float(deltas[k])
        if delta == 0.0:
            continue
        # Rotation axis: from atom (k+2) to atom (k+1). The reversed direction
        # (vs. the bond direction k+1→k+2) is what makes the right-hand-rule
        # rotation consistent with the dihedral_angle sign convention, i.e.
        # rotating by +delta increases τk by exactly +delta.
        axis_vec = pos[k + 1] - pos[k + 2]
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1e-12:
            # Degenerate bond — should never happen on real chemistry but
            # guard against NaN propagation into the solver.
            continue
        axis = axis_vec / axis_norm
        R = rotation_matrix(axis, delta)
        pivot = pos[k + 2]
        for idx in range(k + 3, window_size):
            pos[idx] = pivot + R @ (pos[idx] - pivot)
    return pos


def apply_dihedral_changes_full_mol(
    positions: np.ndarray,
    window: list,
    deltas: np.ndarray,
    downstream_sets: list,
) -> np.ndarray:
    """
    Apply DBT dihedral changes to a full-molecule coordinate array,
    transporting side-chain atoms rigidly with their backbone parents.

    Generalises `apply_dihedral_changes` from a self-contained 7-atom
    chain to an arbitrary atom count. The 7-atom chain is identified by
    indices into `positions` via `window`; per-dihedral rotation sets
    are passed explicitly via `downstream_sets`.

    For dihedral k (around the axis from positions[window[k+2]] to
    positions[window[k+1]]), every atom in `downstream_sets[k]` rotates
    rigidly by `deltas[k]` around the axis, with the pivot at
    `positions[window[k+2]]`. Atoms not in any downstream set are left
    untouched.

    For a cyclic peptide window the appropriate `downstream_sets[k]` is
    the union of the window's strictly-downstream backbone atoms
    (window[k+3..6]) and the side-chain groups of window[k+2..6] —
    see `mcmm._compute_window_downstream_sets`. The MCMM proposer
    precomputes these sets once per window at factory-build time.

    The 7-atom subset of positions[window] is updated identically to
    `apply_dihedral_changes`, plus full-mol downstream atoms get the
    same rigid rotation; bond lengths and bond angles are preserved
    everywhere by construction (rigid rotations).

    Params:
        positions: np.ndarray (n_atoms, 3) : full-mol starting positions
        window: list[int] : 7 atom indices into `positions`, in chain order
        deltas: np.ndarray (4,) : dihedral changes in radians
        downstream_sets: list of 4 iterables of int : full-mol atom
            indices to rotate per dihedral. Order matches `deltas`.
    Returns:
        np.ndarray (n_atoms, 3) : new full-mol positions
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3), got {positions.shape}")
    if len(window) != 7:
        raise ValueError(f"window must have 7 atom indices, got {len(window)}")
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (N_DIHEDRALS,):
        raise ValueError(f"deltas must be ({N_DIHEDRALS},), got {deltas.shape}")
    if len(downstream_sets) != N_DIHEDRALS:
        raise ValueError(
            f"downstream_sets must have {N_DIHEDRALS} entries, "
            f"got {len(downstream_sets)}"
        )

    pos = positions.copy().astype(float)
    for k in range(N_DIHEDRALS):
        delta = float(deltas[k])
        if delta == 0.0:
            continue
        a = window[k + 1]
        b = window[k + 2]
        axis_vec = pos[a] - pos[b]
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1e-12:
            continue
        axis = axis_vec / axis_norm
        R = rotation_matrix(axis, delta)
        pivot = pos[b]
        for idx in downstream_sets[k]:
            pos[idx] = pivot + R @ (pos[idx] - pivot)
    return pos


def _expand_deltas(
    drive_idx: int, drive_delta: float, free_deltas: np.ndarray
) -> np.ndarray:
    """
    Build the full (4,) deltas vector from drive + 3 non-drive components.

    Params:
        drive_idx: int : index of drive dihedral
        drive_delta: float : drive perturbation in radians
        free_deltas: np.ndarray (n_dih-1,) : the non-drive deltas in order
    Returns:
        np.ndarray (n_dih,) : full deltas vector, where n_dih =
            len(free_deltas) + 1 (one slot for the drive)
    """
    n_dih = len(free_deltas) + 1
    deltas = np.empty(n_dih)
    deltas[drive_idx] = drive_delta
    free_iter = iter(free_deltas)
    for k in range(n_dih):
        if k != drive_idx:
            deltas[k] = next(free_iter)
    return deltas


def _wrap_to_pi(angle_rad: float) -> float:
    """
    Wrap an angle to the half-open interval [-π, π).

    Used by `propose_omega_flip` to express the trans↔cis ω change as the
    shortest signed rotation from the current ω to the target. The branch
    cut at ±π is immaterial for ω flips: a +π and a −π drive both reach
    the same cis (or trans) state.

    Params:
        angle_rad: float : angle in radians
    Returns:
        float : equivalent angle in [-π, π)
    """
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def closure_residual(
    positions: np.ndarray,
    drive_idx: int,
    drive_delta: float,
    free_deltas: np.ndarray,
) -> np.ndarray:
    """
    Apply all dihedral changes and return the displacement of the last two
    window atoms (r(W-2), r(W-1)) from their original positions.

    The closure problem solved by `propose_move` / `propose_omega_flip` is
    finding `free_deltas` such that this residual norm falls below the
    tolerance. Exposed for testing and for callers that want to integrate
    the closure into a different solver. Window size W is inferred from
    `positions`; W=7 (DBT) fixes r5, r6 and W=10 (ω flip) fixes r8, r9.

    Params:
        positions: np.ndarray (W, 3) : starting atom positions
        drive_idx: int : index of drive dihedral in [0, W-4]
        drive_delta: float : drive perturbation in radians
        free_deltas: np.ndarray (W-4,) : the non-drive deltas
    Returns:
        np.ndarray (6,) : [r(W-2)_dx, dy, dz, r(W-1)_dx, dy, dz]
    """
    deltas = _expand_deltas(drive_idx, drive_delta, free_deltas)
    new_pos = apply_dihedral_changes(positions, deltas)
    last2 = positions.shape[0] - 2
    return np.concatenate(
        [new_pos[last2] - positions[last2], new_pos[last2 + 1] - positions[last2 + 1]]
    )


def propose_move(
    positions: np.ndarray,
    drive_idx: int,
    drive_delta: float,
    closure_tol: float = DEFAULT_CLOSURE_TOL,
    max_solver_iter: int = 50,
) -> MoveProposal:
    """
    Propose a concerted-rotation move with the given drive perturbation.

    Pipeline:
      1. Solve numerically for the 3 non-drive dihedrals that minimise
         the displacement of atoms r5 and r6 from their original
         positions. The system has 6 residuals (3 components each for
         r5 and r6) and 3 unknowns; least_squares handles the
         over-constraint gracefully.
      2. Check closure: if the residual norm exceeds closure_tol, the
         move is geometrically infeasible. Return a MoveProposal with
         `success=False` so the caller can reject without paying for
         MMFF.
      3. Compute |det J| via finite differences (Wu-Deem 1999 detailed
         balance correction).

    Params:
        positions: np.ndarray (7, 3) : starting atom positions
        drive_idx: int : index of drive dihedral in [0, 3]
        drive_delta: float : drive perturbation in radians
        closure_tol: float : maximum acceptable residual norm in Å
        max_solver_iter: int : least_squares iteration cap
    Returns:
        MoveProposal : NamedTuple with fields (new_positions,
            det_jacobian, deltas, success). Tuple-unpackable as a
            4-tuple. The `deltas` field is the (4,) array of dihedral
            changes applied — drive_delta at drive_idx, closure-solver
            outputs at the other three positions. All zeros on failure.
    """
    if not (0 <= drive_idx < N_DIHEDRALS):
        raise ValueError(
            f"drive_idx must be in [0, {N_DIHEDRALS - 1}], got {drive_idx}"
        )

    initial_guess = np.zeros(3)

    def _residual(free_deltas):
        return closure_residual(positions, drive_idx, drive_delta, free_deltas)

    result = least_squares(
        _residual,
        initial_guess,
        method="lm",
        max_nfev=max_solver_iter,
    )

    residual_norm = float(np.linalg.norm(result.fun))
    if residual_norm > closure_tol:
        return MoveProposal(
            new_positions=positions.copy(),
            det_jacobian=0.0,
            deltas=np.zeros(N_DIHEDRALS),
            success=False,
        )

    deltas = _expand_deltas(drive_idx, drive_delta, result.x)
    new_positions = apply_dihedral_changes(positions, deltas)
    det_j = _finite_difference_det_jacobian(
        positions, drive_idx, drive_delta, result.x, closure_tol
    )

    return MoveProposal(
        new_positions=new_positions,
        det_jacobian=det_j,
        deltas=deltas,
        success=True,
    )


def _finite_difference_det_jacobian(
    positions: np.ndarray,
    drive_idx: int,
    drive_delta: float,
    free_deltas_at_solution: np.ndarray,
    closure_tol: float,
    eps: float = 1e-4,
    solver_method: str = "lm",
) -> float:
    """
    Estimate a Wu-Deem-style Jacobian magnitude via finite differences.

    For Wu-Deem detailed balance, the move-acceptance probability is
    weighted by |det J|, where J is the Jacobian of the closure manifold's
    parametric description by the drive angle. For our v0 numerical
    closure, J is a (W-4)×1 column ∂(free_deltas)/∂(drive_delta) at the
    solution. We return ‖J‖ (the column's L2 norm) as a scalar proxy for
    the determinant — exact for the 1-d drive case, approximate when the
    underlying constraint geometry has more structure.

    TODO (Option A upgrade): replace with the analytical Jacobian from
    DBT 1993 / Coutsias 2004 once the polynomial closure solver lands.

    Params:
        positions: np.ndarray (W, 3)
        drive_idx: int
        drive_delta: float : drive value at which the Jacobian is evaluated
        free_deltas_at_solution: np.ndarray (W-4,) : closure solution at this drive
        closure_tol: float : passed through to the perturbed re-solve
        eps: float : finite-difference step in radians
        solver_method: str : least_squares method for the perturbed
            re-solves ('lm' for the over-determined DBT W=7 closure;
            'trf' for the exactly/under-determined ω-flip W>=10 closure,
            which 'lm' rejects).
    Returns:
        float : |J| proxy (non-negative). Returns 1.0 (neutral weight) if
            the perturbed solves fail.
    """

    def _solve_at(drive_value):
        result = least_squares(
            lambda x: closure_residual(positions, drive_idx, drive_value, x),
            free_deltas_at_solution,
            method=solver_method,
        )
        if np.linalg.norm(result.fun) > closure_tol:
            return None
        return result.x

    plus = _solve_at(drive_delta + eps)
    minus = _solve_at(drive_delta - eps)
    if plus is None or minus is None:
        # Perturbed re-solve failed near the boundary of the closure manifold.
        # Falling back to a neutral weight introduces a small detailed-balance
        # bias for that specific (geometry, drive) pair; acceptable in v0.
        return 1.0

    jacobian_col = (plus - minus) / (2.0 * eps)
    return float(np.linalg.norm(jacobian_col))


def propose_omega_flip(
    positions: np.ndarray,
    drive_idx: int,
    target_omega_rad: float,
    closure_tol: float = DEFAULT_CLOSURE_TOL,
    n_continuation_steps: int = 12,
    max_solver_iter: int = 50,
) -> MoveProposal:
    """
    Propose a cis↔trans ω-isomerization move on a backbone window.

    Move B (v0.3): unlike `propose_move`, which drives an inner dihedral
    by a small Gaussian Δθ, this drives the ω amide dihedral (the inner
    dihedral at `drive_idx`, around bond (drive_idx+1, drive_idx+2)) all
    the way to a target value — typically 0 (cis) or ±π (trans) — and
    re-solves the remaining inner dihedrals to restore the last two window
    atoms r(W-2), r(W-1). The ω bond is in the macrocycle ring, so a naive
    open-tree rotation would break ring closure; the closure solver is
    what keeps the ring intact (B.1a-widened locked design — see
    docs/concerted_moves_v0_3_plan.md).

    Because each inner dihedral τk ends at exactly `original + deltas[k]`
    independent of the others (see `apply_dihedral_changes`), a
    successful flip lands ω **exactly** on `target_omega_rad`; the free
    dihedrals only absorb the ring-closure displacement.

    **Window size matters.** The closure fixes the last two atoms (6
    residuals) against W-4 free dihedrals. For W=7 (the DBT window) that
    is only 3 free dihedrals — over-determined, so a large flip leaves a
    multi-Å residual and fails to close. For W=10 (6 free dihedrals) the
    system is exactly determined and a full trans→cis flip closes to ~0 Å
    on real macrocycle geometry (Findings 2026-06-22). The ω-flip proposer
    therefore builds W=10 windows; W=7 is for DBT only.

    A ω flip is a much larger drive (up to π) than a DBT move (~0.1 rad),
    so a single least_squares solve from a zero guess rarely converges.
    This routine ramps the drive from 0 to the full Δω in
    `n_continuation_steps` increments, warm-starting each closure solve
    from the previous step's free dihedrals — a homotopy continuation
    that tracks the closure manifold through the large rotation. The
    solver uses `trf` (not `lm`): with W>=10 the closure is exactly- or
    under-determined and `lm` rejects those shapes.

    Params:
        positions: np.ndarray (W, 3) : W-atom backbone window (W>=10 for a
            closable ω flip). The ω amide C–N bond must be the inner bond
            (drive_idx+1, drive_idx+2); the ω dihedral is measured over
            atoms (drive_idx, drive_idx+1, drive_idx+2, drive_idx+3).
        drive_idx: int : index of the ω dihedral in [0, W-4].
        target_omega_rad: float : target ω value in radians (0 for cis,
            ±π for trans).
        closure_tol: float : maximum acceptable displacement norm of the
            last two window atoms (Å) for the flip to count as ring-closed.
        n_continuation_steps: int : number of warm-started closure solves
            used to ramp the drive from 0 to the full Δω. More steps cost
            more solver time but improve the chance of tracking the
            closure manifold through the large rotation.
        max_solver_iter: int : least_squares iteration cap per step.
    Returns:
        MoveProposal : NamedTuple (new_positions, det_jacobian, deltas,
            success). On failure: input positions copied, det_jacobian
            0.0, deltas all zero, success False.
    """
    window_size = positions.shape[0]
    n_dih = window_size - 3
    if not (0 <= drive_idx < n_dih):
        raise ValueError(
            f"drive_idx must be in [0, {n_dih - 1}] for a {window_size}-atom "
            f"window, got {drive_idx}"
        )
    if n_continuation_steps < 1:
        raise ValueError(
            f"n_continuation_steps must be >= 1, got {n_continuation_steps}"
        )

    current_omega = dihedral_angle(
        positions[drive_idx],
        positions[drive_idx + 1],
        positions[drive_idx + 2],
        positions[drive_idx + 3],
    )
    total_drive = _wrap_to_pi(target_omega_rad - current_omega)

    # Homotopy continuation: ramp the drive from 0 to total_drive, warm-
    # starting each closure solve from the previous step's free dihedrals.
    # 'trf' handles the exactly/under-determined W>=10 shapes that 'lm' rejects.
    free = np.zeros(n_dih - 1)
    for step in range(1, n_continuation_steps + 1):
        drive_value = total_drive * step / n_continuation_steps
        result = least_squares(
            lambda x, d=drive_value: closure_residual(positions, drive_idx, d, x),
            free,
            method="trf",
            max_nfev=max_solver_iter,
        )
        free = result.x

    final_residual = float(
        np.linalg.norm(closure_residual(positions, drive_idx, total_drive, free))
    )
    if final_residual > closure_tol:
        return MoveProposal(
            new_positions=positions.copy(),
            det_jacobian=0.0,
            deltas=np.zeros(n_dih),
            success=False,
        )

    deltas = _expand_deltas(drive_idx, total_drive, free)
    new_positions = apply_dihedral_changes(positions, deltas)
    det_j = _finite_difference_det_jacobian(
        positions, drive_idx, total_drive, free, closure_tol, solver_method="trf"
    )

    return MoveProposal(
        new_positions=new_positions,
        det_jacobian=det_j,
        deltas=deltas,
        success=True,
    )
