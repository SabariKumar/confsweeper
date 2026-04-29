"""Unit tests for src/concerted_rotation.py.

Tests are layered: geometry primitives (rotation_matrix, dihedral_angle)
first, then the chain-rebuild self-consistency (apply_dihedral_changes
preserves bond lengths, bond angles, and changes the targeted dihedral
by exactly the requested delta), then the closure solver and its
edge cases. No RDKit dependency — fixtures are synthetic 7-atom chains
built directly from numpy primitives.
"""

import numpy as np
import pytest

from concerted_rotation import (
    DEFAULT_CLOSURE_TOL,
    N_DIHEDRALS,
    apply_dihedral_changes,
    closure_residual,
    dihedral_angle,
    propose_move,
    rotation_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _planar_zigzag(
    bond_length: float = 1.5, bond_angle_deg: float = 120.0
) -> np.ndarray:
    """
    Planar 7-atom zigzag with constant bond length and bond angle. All
    inner dihedrals are π (trans) since every atom lies in the same plane.

    This is the simplest reference geometry for testing the chain-rebuild
    primitives. Closure tests should use _twisted_chain instead, since
    perturbing a planar geometry off-plane is a degenerate edge case for
    the closure solver.

    Params:
        bond_length: float : Å
        bond_angle_deg: float : degrees, applied at every interior atom
    Returns:
        np.ndarray (7, 3) : atom positions
    """
    bond_angle = np.deg2rad(bond_angle_deg)
    pos = np.zeros((7, 3))
    pos[0] = [0.0, 0.0, 0.0]
    pos[1] = [bond_length, 0.0, 0.0]
    # Alternate +y and 0 components to produce a planar zigzag.
    direction_a = np.array([-np.cos(bond_angle), np.sin(bond_angle), 0.0])
    direction_b = np.array([1.0, 0.0, 0.0])
    for i in range(2, 7):
        d = direction_a if (i % 2 == 0) else direction_b
        pos[i] = pos[i - 1] + bond_length * d
    return pos


def _twisted_chain(seed: int = 0) -> np.ndarray:
    """
    7-atom chain with a generic non-planar geometry, built by applying
    random dihedral perturbations to a planar zigzag. Bond lengths and
    bond angles match the zigzag fixture.

    Used for closure-solver tests: a twisted chain has full 3D structure,
    so the closure manifold is not on a degenerate edge.
    """
    rng = np.random.default_rng(seed)
    base = _planar_zigzag()
    deltas = rng.uniform(-1.0, 1.0, size=N_DIHEDRALS)
    return apply_dihedral_changes(base, deltas)


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------


def test_rotation_matrix_zero_angle_is_identity():
    R = rotation_matrix(np.array([1.0, 0.0, 0.0]), 0.0)
    assert np.allclose(R, np.eye(3))


def test_rotation_matrix_pi_around_x_negates_yz():
    R = rotation_matrix(np.array([1.0, 0.0, 0.0]), np.pi)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert np.allclose(R, expected, atol=1e-12)


def test_rotation_matrix_orthogonal():
    """A rotation matrix is orthogonal: R @ R.T = I and det R = +1."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix(axis, angle)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# dihedral_angle
# ---------------------------------------------------------------------------


def test_dihedral_angle_planar_zigzag_is_pi():
    """All inner dihedrals of a planar zigzag are π (trans)."""
    pos = _planar_zigzag()
    for k in range(N_DIHEDRALS):
        d = dihedral_angle(pos[k], pos[k + 1], pos[k + 2], pos[k + 3])
        assert abs(abs(d) - np.pi) < 1e-10


def test_dihedral_angle_cis_is_zero():
    """
    Cis configuration (atoms 0 and 3 on the same side of the bond axis,
    in the same plane) gives dihedral = 0. Tests the basic rotation
    convention without committing to an absolute right-hand-rule sign,
    which the docstring explicitly does not standardise.
    """
    p0 = np.array([0.0, 1.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, 1.0, 0.0])  # on the same +y side as p0
    d = dihedral_angle(p0, p1, p2, p3)
    assert abs(d) < 1e-10


# ---------------------------------------------------------------------------
# apply_dihedral_changes — geometry preservation
# ---------------------------------------------------------------------------


def test_apply_zero_deltas_is_identity():
    pos = _planar_zigzag()
    new_pos = apply_dihedral_changes(pos, np.zeros(N_DIHEDRALS))
    assert np.allclose(new_pos, pos, atol=1e-12)


def test_apply_preserves_bond_lengths():
    pos = _twisted_chain(seed=1)
    deltas = np.array([0.1, -0.2, 0.3, -0.4])
    new_pos = apply_dihedral_changes(pos, deltas)
    for i in range(6):
        original = np.linalg.norm(pos[i + 1] - pos[i])
        new = np.linalg.norm(new_pos[i + 1] - new_pos[i])
        assert abs(new - original) < 1e-10, f"bond ({i},{i+1}): {original} → {new}"


def test_apply_preserves_bond_angles():
    pos = _twisted_chain(seed=2)
    deltas = np.array([0.1, -0.2, 0.3, -0.4])
    new_pos = apply_dihedral_changes(pos, deltas)
    for i in range(1, 6):
        v1 = pos[i - 1] - pos[i]
        v2 = pos[i + 1] - pos[i]
        cos_old = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        v1n = new_pos[i - 1] - new_pos[i]
        v2n = new_pos[i + 1] - new_pos[i]
        cos_new = np.dot(v1n, v2n) / (np.linalg.norm(v1n) * np.linalg.norm(v2n))
        assert (
            abs(cos_new - cos_old) < 1e-10
        ), f"angle at atom {i}: cos changed by {cos_new - cos_old}"


def test_apply_keeps_first_two_atoms_fixed():
    pos = _twisted_chain(seed=3)
    deltas = np.array([0.5, -0.3, 0.2, 0.4])
    new_pos = apply_dihedral_changes(pos, deltas)
    assert np.allclose(new_pos[0], pos[0], atol=1e-12)
    assert np.allclose(new_pos[1], pos[1], atol=1e-12)


def test_apply_invalid_shape_raises():
    with pytest.raises(ValueError, match="positions must be"):
        apply_dihedral_changes(np.zeros((6, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="deltas must be"):
        apply_dihedral_changes(np.zeros((7, 3)), np.zeros(3))


# ---------------------------------------------------------------------------
# apply_dihedral_changes — sign-convention self-consistency
# ---------------------------------------------------------------------------


def test_apply_changes_targeted_dihedral_by_exactly_delta():
    """
    The load-bearing self-consistency check: applying a non-zero delta to
    one dihedral changes its measured value by exactly that delta. This
    locks the sign convention between rotation_matrix and dihedral_angle.
    """
    pos = _twisted_chain(seed=4)
    delta = 0.3  # ~17°, large enough to disambiguate sign

    for k in range(N_DIHEDRALS):
        deltas = np.zeros(N_DIHEDRALS)
        deltas[k] = delta
        new_pos = apply_dihedral_changes(pos, deltas)
        old = dihedral_angle(pos[k], pos[k + 1], pos[k + 2], pos[k + 3])
        new = dihedral_angle(new_pos[k], new_pos[k + 1], new_pos[k + 2], new_pos[k + 3])
        diff = ((new - old) + np.pi) % (2.0 * np.pi) - np.pi
        assert abs(diff - delta) < 1e-10, f"k={k}: expected change {delta}, got {diff}"


def test_apply_preserves_other_dihedrals_when_changing_one():
    """
    Rigid rotation of atoms downstream of bond k preserves dihedrals τj
    for j != k whose defining atom set is entirely outside the rotated
    region (j < k) or entirely inside it (j > k). The boundary case j = k
    is handled by the test above.
    """
    pos = _twisted_chain(seed=5)
    delta = 0.4

    for k in range(N_DIHEDRALS):
        deltas = np.zeros(N_DIHEDRALS)
        deltas[k] = delta
        new_pos = apply_dihedral_changes(pos, deltas)
        for j in range(N_DIHEDRALS):
            if j == k:
                continue
            old = dihedral_angle(pos[j], pos[j + 1], pos[j + 2], pos[j + 3])
            new = dihedral_angle(
                new_pos[j], new_pos[j + 1], new_pos[j + 2], new_pos[j + 3]
            )
            diff = ((new - old) + np.pi) % (2.0 * np.pi) - np.pi
            assert (
                abs(diff) < 1e-10
            ), f"applying delta to k={k} corrupted τ{j} by {diff}"


# ---------------------------------------------------------------------------
# closure_residual
# ---------------------------------------------------------------------------


def test_closure_residual_zero_for_zero_perturbation():
    pos = _twisted_chain(seed=6)
    for drive_idx in range(N_DIHEDRALS):
        residual = closure_residual(pos, drive_idx, 0.0, np.zeros(3))
        assert np.allclose(residual, 0.0, atol=1e-12)


def test_closure_residual_nonzero_when_drive_perturbed_alone():
    """
    Perturbing only the drive (with zero free_deltas) generally moves
    atoms r5 and r6, so the residual is non-zero. This is the starting
    point of the closure search.
    """
    pos = _twisted_chain(seed=7)
    residual = closure_residual(
        pos, drive_idx=1, drive_delta=0.2, free_deltas=np.zeros(3)
    )
    assert np.linalg.norm(residual) > 1e-3


# ---------------------------------------------------------------------------
# propose_move
# ---------------------------------------------------------------------------


def test_propose_move_zero_perturbation_is_identity():
    pos = _twisted_chain(seed=8)
    new_pos, det_j, success = propose_move(pos, drive_idx=1, drive_delta=0.0)
    assert success
    assert np.allclose(new_pos, pos, atol=DEFAULT_CLOSURE_TOL)
    assert np.isfinite(det_j)


def test_propose_move_outer_atoms_preserved():
    """
    On a successful closure the combined r5 + r6 displacement norm is below
    closure_tol (the contract of `propose_move`). Tested across a sweep of
    drive choices and small Δθ values where closure is expected to succeed.

    The system is over-determined (6 residuals, 3 unknowns), so per-atom
    displacement is not zero — only the joint norm is bounded. Each atom's
    displacement is bounded by the joint norm.
    """
    pos = _twisted_chain(seed=9)
    succeeded_at_least_once = False
    for drive_idx in range(N_DIHEDRALS):
        for drive_delta in [0.05, -0.05, 0.02, -0.1]:
            new_pos, _, success = propose_move(pos, drive_idx, drive_delta)
            if not success:
                continue
            succeeded_at_least_once = True
            joint_residual = np.linalg.norm(
                np.concatenate([new_pos[5] - pos[5], new_pos[6] - pos[6]])
            )
            assert joint_residual < DEFAULT_CLOSURE_TOL
    assert succeeded_at_least_once, "No closure succeeded across the sweep"


def test_propose_move_drive_dihedral_changed_by_delta():
    """The returned geometry has the drive dihedral perturbed by exactly Δθ."""
    pos = _twisted_chain(seed=10)
    drive_delta = 0.05

    for drive_idx in range(N_DIHEDRALS):
        new_pos, _, success = propose_move(pos, drive_idx, drive_delta)
        if not success:
            continue
        old = dihedral_angle(
            pos[drive_idx],
            pos[drive_idx + 1],
            pos[drive_idx + 2],
            pos[drive_idx + 3],
        )
        new = dihedral_angle(
            new_pos[drive_idx],
            new_pos[drive_idx + 1],
            new_pos[drive_idx + 2],
            new_pos[drive_idx + 3],
        )
        diff = ((new - old) + np.pi) % (2.0 * np.pi) - np.pi
        # The closure solver may shift things slightly to satisfy the
        # constraint, so allow ~1e-4 rad slack rather than 1e-10.
        assert (
            abs(diff - drive_delta) < 1e-4
        ), f"drive_idx={drive_idx}: Δτ={diff}, expected {drive_delta}"


def test_propose_move_invalid_drive_idx_raises():
    pos = _twisted_chain(seed=11)
    with pytest.raises(ValueError, match="drive_idx must be"):
        propose_move(pos, drive_idx=4, drive_delta=0.1)
    with pytest.raises(ValueError, match="drive_idx must be"):
        propose_move(pos, drive_idx=-1, drive_delta=0.1)


def test_propose_move_returns_finite_jacobian():
    pos = _twisted_chain(seed=12)
    _, det_j, success = propose_move(pos, drive_idx=1, drive_delta=0.05)
    if success:
        assert np.isfinite(det_j)
        assert det_j >= 0


def test_propose_move_failure_returns_input_positions():
    """
    When closure fails the returned positions equal the input (no spurious
    geometry update). Used by the caller as the rejection signal alongside
    success=False.

    A reliably-failing case is hard to construct geometrically (the solver
    is robust on small perturbations), but we can fake it by setting
    closure_tol absurdly tight so any rounding error counts as failure.
    """
    pos = _twisted_chain(seed=13)
    impossible_tol = 1e-30
    new_pos, det_j, success = propose_move(
        pos, drive_idx=1, drive_delta=0.1, closure_tol=impossible_tol
    )
    assert not success
    assert det_j == 0.0
    assert np.allclose(new_pos, pos)
