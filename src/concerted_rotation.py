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
    Apply incremental dihedral changes to a 7-atom chain.

    Atoms r0, r1 stay fixed (the upstream-frame convention). Atoms r2..r6
    may move. `deltas[k]` (k in 0..3) is the change to the dihedral around
    bond (k+1, k+2), applied as a rigid rotation of all atoms strictly
    downstream of that bond, around the bond axis.

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
        positions: np.ndarray (7, 3) : starting atom positions
        deltas: np.ndarray (4,) : dihedral changes in radians
    Returns:
        np.ndarray (7, 3) : new atom positions
    """
    if positions.shape != (7, 3):
        raise ValueError(f"positions must be (7, 3), got {positions.shape}")
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (N_DIHEDRALS,):
        raise ValueError(f"deltas must be ({N_DIHEDRALS},), got {deltas.shape}")

    pos = positions.copy().astype(float)
    for k in range(N_DIHEDRALS):
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
        for idx in range(k + 3, 7):
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
        drive_idx: int : index of drive dihedral in [0, 3]
        drive_delta: float : drive perturbation in radians
        free_deltas: np.ndarray (3,) : the three non-drive deltas in order
    Returns:
        np.ndarray (4,) : full deltas vector
    """
    deltas = np.empty(N_DIHEDRALS)
    deltas[drive_idx] = drive_delta
    free_iter = iter(free_deltas)
    for k in range(N_DIHEDRALS):
        if k != drive_idx:
            deltas[k] = next(free_iter)
    return deltas


def closure_residual(
    positions: np.ndarray,
    drive_idx: int,
    drive_delta: float,
    free_deltas: np.ndarray,
) -> np.ndarray:
    """
    Apply all dihedral changes and return the displacement of atoms r5, r6
    from their original positions.

    The closure problem solved by `propose_move` is finding `free_deltas`
    such that this residual norm falls below the tolerance. Exposed for
    testing and for callers that want to integrate the closure into a
    different solver.

    Params:
        positions: np.ndarray (7, 3) : starting atom positions
        drive_idx: int : index of drive dihedral in [0, 3]
        drive_delta: float : drive perturbation in radians
        free_deltas: np.ndarray (3,) : the three non-drive deltas
    Returns:
        np.ndarray (6,) : [r5_dx, r5_dy, r5_dz, r6_dx, r6_dy, r6_dz]
    """
    deltas = _expand_deltas(drive_idx, drive_delta, free_deltas)
    new_pos = apply_dihedral_changes(positions, deltas)
    return np.concatenate([new_pos[5] - positions[5], new_pos[6] - positions[6]])


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
) -> float:
    """
    Estimate a Wu-Deem-style Jacobian magnitude via finite differences.

    For Wu-Deem detailed balance, the move-acceptance probability is
    weighted by |det J|, where J is the Jacobian of the closure manifold's
    parametric description by the drive angle. For our v0 numerical
    closure, J is a 3×1 column ∂(free_deltas)/∂(drive_delta) at the
    solution. We return ‖J‖ (the column's L2 norm) as a scalar proxy for
    the determinant — exact for the 1-d drive case, approximate when the
    underlying constraint geometry has more structure.

    TODO (Option A upgrade): replace with the analytical Jacobian from
    DBT 1993 / Coutsias 2004 once the polynomial closure solver lands.

    Params:
        positions: np.ndarray (7, 3)
        drive_idx: int
        drive_delta: float : drive value at which the Jacobian is evaluated
        free_deltas_at_solution: np.ndarray (3,) : closure solution at this drive
        closure_tol: float : passed through to the perturbed re-solve
        eps: float : finite-difference step in radians
    Returns:
        float : |J| proxy (non-negative). Returns 1.0 (neutral weight) if
            the perturbed solves fail.
    """

    def _solve_at(drive_value):
        result = least_squares(
            lambda x: closure_residual(positions, drive_idx, drive_value, x),
            free_deltas_at_solution,
            method="lm",
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
