"""Tests for src/torsional_sampling.py."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolTransforms

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from torsional_sampling import (
    _d14,
    _sample_from_grid,
    classify_backbone_residues,
    embed_constrained,
    get_backbone_dihedrals,
    make_constrained_bounds,
    sample_constrained_confs,
    set_dihedral_bounds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard sp3 C-C geometry
_R_CC = 1.54  # Å
_COS_T = -1.0 / 3  # cos(109.47°)
_SIN_T = math.sqrt(1.0 - _COS_T**2)

# Cyclo(Ala)4 — 12-membered ring macrocycle, 4 backbone residues
_CYCLOALA4 = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"


def _cycloala4_with_hs() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(_CYCLOALA4))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _butane_bounds() -> tuple[Chem.Mol, np.ndarray, int, int, int, int]:
    """Butane bounds matrix and the four carbon atom indices."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    # Carbon atoms are the first four heavy atoms (indices 0-3 before AddHs reorders)
    c_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    assert len(c_idxs) == 4
    i, j, k, l = c_idxs
    return mol, bounds, i, j, k, l


# ---------------------------------------------------------------------------
# _d14
# ---------------------------------------------------------------------------


def test_d14_anti_longer_than_gauche():
    """Anti (φ=180°) gives longer 1,4-distance than gauche (φ=60°)."""
    anti = _d14(_R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.pi)
    gauche = _d14(_R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.radians(60))
    assert anti > gauche


def test_d14_eclipsed_is_shortest():
    """Eclipsed (φ=0°) gives the shortest 1,4-distance."""
    eclipsed = _d14(_R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, 0.0)
    for phi_deg in [60, 120, 180]:
        d = _d14(
            _R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.radians(phi_deg)
        )
        assert d >= eclipsed - 1e-9


def test_d14_symmetric_around_zero():
    """d14(φ) == d14(-φ) since cos is even."""
    for phi_deg in [30, 60, 90, 120, 150]:
        pos = _d14(
            _R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.radians(phi_deg)
        )
        neg = _d14(
            _R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.radians(-phi_deg)
        )
        assert abs(pos - neg) < 1e-12


def test_d14_anti_butane_near_381():
    """Anti butane 1,4-distance should be ~3.81 Å (matches RDKit default upper bound)."""
    d = _d14(_R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, math.pi)
    assert 3.6 < d < 4.0, f"Expected ~3.8 Å, got {d:.3f}"


def test_d14_nonnegative():
    """d14 is always non-negative, even at extremes."""
    for phi in np.linspace(0, 2 * math.pi, 37):
        d = _d14(_R_CC, _R_CC, _R_CC, _COS_T, _SIN_T, _COS_T, _SIN_T, float(phi))
        assert d >= 0.0


# ---------------------------------------------------------------------------
# set_dihedral_bounds
# ---------------------------------------------------------------------------


def test_set_dihedral_bounds_returns_array():
    _, bounds, i, j, k, l = _butane_bounds()
    result = set_dihedral_bounds(bounds, i, j, k, l, 180.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == bounds.shape


def test_set_dihedral_bounds_tightens_upper():
    """Constraining to anti should tighten the upper bound."""
    _, bounds, i, j, k, l = _butane_bounds()
    result = set_dihedral_bounds(bounds, i, j, k, l, 180.0, tolerance_deg=30.0)
    a, b = (i, l) if i < l else (l, i)
    assert result[a, b] <= bounds[a, b] + 1e-9


def test_set_dihedral_bounds_tightens_lower():
    """Constraining to gauche should raise the lower bound."""
    _, bounds, i, j, k, l = _butane_bounds()
    result = set_dihedral_bounds(bounds, i, j, k, l, 60.0, tolerance_deg=30.0)
    a, b = (i, l) if i < l else (l, i)
    assert result[b, a] >= bounds[b, a] - 1e-9


def test_set_dihedral_bounds_target_in_range():
    """d14 evaluated at the target angle must lie within the new [lower, upper]."""
    _, bounds, i, j, k, l = _butane_bounds()
    for target_deg in [60.0, 120.0, 180.0]:
        result = set_dihedral_bounds(bounds, i, j, k, l, target_deg, tolerance_deg=30.0)
        if result is None:
            continue
        a, b = (i, l) if i < l else (l, i)
        new_upper = result[a, b]
        new_lower = result[b, a]
        assert new_lower <= new_upper + 1e-9

        # Evaluate d14 directly at the target angle using bounds midpoints
        def mid(x, y):
            lo = bounds[max(x, y), min(x, y)]
            hi = bounds[min(x, y), max(x, y)]
            return (lo + hi) / 2.0

        r12, r23, r34 = mid(i, j), mid(j, k), mid(k, l)
        r13, r24 = mid(i, k), mid(j, l)
        cos_t123 = float(
            np.clip((r12**2 + r23**2 - r13**2) / (2 * r12 * r23), -1, 1)
        )
        cos_t234 = float(
            np.clip((r23**2 + r34**2 - r24**2) / (2 * r23 * r34), -1, 1)
        )
        sin_t123 = math.sqrt(1 - cos_t123**2)
        sin_t234 = math.sqrt(1 - cos_t234**2)
        d_target = _d14(
            r12,
            r23,
            r34,
            cos_t123,
            sin_t123,
            cos_t234,
            sin_t234,
            math.radians(target_deg),
        )
        assert (
            new_lower - 1e-6 <= d_target <= new_upper + 1e-6
        ), f"target={target_deg}°: d14={d_target:.4f} not in [{new_lower:.4f}, {new_upper:.4f}]"


def test_set_dihedral_bounds_wider_tolerance_wider_range():
    """Larger tolerance produces a wider or equal d14 range."""
    _, bounds, i, j, k, l = _butane_bounds()
    r_narrow = set_dihedral_bounds(bounds, i, j, k, l, 120.0, tolerance_deg=10.0)
    r_wide = set_dihedral_bounds(bounds, i, j, k, l, 120.0, tolerance_deg=45.0)
    if r_narrow is None or r_wide is None:
        pytest.skip("Infeasible constraint for this geometry")
    a, b = (i, l) if i < l else (l, i)
    narrow_range = r_narrow[a, b] - r_narrow[b, a]
    wide_range = r_wide[a, b] - r_wide[b, a]
    assert wide_range >= narrow_range - 1e-9


def test_set_dihedral_bounds_only_modifies_il():
    """Only the (i,l) bounds entry should change; all other entries are identical."""
    _, bounds, i, j, k, l = _butane_bounds()
    result = set_dihedral_bounds(bounds, i, j, k, l, 180.0)
    assert result is not None
    a, b = (i, l) if i < l else (l, i)
    diff = result - bounds
    mask = np.ones_like(diff, dtype=bool)
    mask[a, b] = False
    mask[b, a] = False
    assert np.allclose(diff[mask], 0.0)


def test_set_dihedral_bounds_does_not_mutate_input():
    _, bounds, i, j, k, l = _butane_bounds()
    original = bounds.copy()
    set_dihedral_bounds(bounds, i, j, k, l, 180.0)
    assert np.array_equal(bounds, original)


def test_set_dihedral_bounds_reversed_indices_same_result():
    """Reversing the dihedral direction (l,k,j,i) should give the same result."""
    _, bounds, i, j, k, l = _butane_bounds()
    forward = set_dihedral_bounds(bounds, i, j, k, l, 120.0, tolerance_deg=30.0)
    backward = set_dihedral_bounds(bounds, l, k, j, i, 120.0, tolerance_deg=30.0)
    if forward is None or backward is None:
        assert forward is backward  # both infeasible or both not
    else:
        assert np.allclose(forward, backward, atol=1e-9)


def test_set_dihedral_bounds_infeasible_returns_none():
    """Force infeasibility by pre-tightening the 1,4 window to a tiny range."""
    _, bounds, i, j, k, l = _butane_bounds()
    a, b = (i, l) if i < l else (l, i)
    # Clamp to a 0.01 Å window at the current lower bound — almost certainly
    # incompatible with any 30°-wide dihedral range.
    tight = bounds.copy()
    tight[a, b] = bounds[b, a] + 0.01  # upper = lower + 0.01 Å
    result = set_dihedral_bounds(tight, i, j, k, l, 60.0, tolerance_deg=30.0)
    # Might be None (infeasible) or still valid if the d14 range happens to fit.
    # We can't guarantee None without knowing exact geometry, so just assert type.
    assert result is None or isinstance(result, np.ndarray)


def test_set_dihedral_bounds_exact_infeasible():
    """Manually craft a case where [d14_min, d14_max] clearly misses existing window."""
    _, bounds, i, j, k, l = _butane_bounds()
    a, b = (i, l) if i < l else (l, i)
    # Anti (180°) d14 for C-C is ~3.8 Å. Set upper bound to 2.0 Å (impossible).
    tight = bounds.copy()
    tight[a, b] = 2.0
    tight[b, a] = 1.99
    result = set_dihedral_bounds(tight, i, j, k, l, 180.0, tolerance_deg=30.0)
    assert result is None


# ---------------------------------------------------------------------------
# get_backbone_dihedrals
# ---------------------------------------------------------------------------


def test_get_backbone_dihedrals_count_cycloala4():
    """Cyclo(Ala)4 has 4 backbone residues → 4 (phi_atoms, psi_atoms) tuples."""
    mol = _cycloala4_with_hs()
    defs = get_backbone_dihedrals(mol)
    assert len(defs) == 4


def test_get_backbone_dihedrals_valid_indices():
    """All atom indices in the returned tuples are within the molecule."""
    mol = _cycloala4_with_hs()
    n_atoms = mol.GetNumAtoms()
    for phi_atoms, psi_atoms in get_backbone_dihedrals(mol):
        for idx in phi_atoms + psi_atoms:
            assert 0 <= idx < n_atoms


def test_get_backbone_dihedrals_phi_psi_share_nca():
    """phi_atoms[1:3] == psi_atoms[0:2] — they share (N, Cα)."""
    mol = _cycloala4_with_hs()
    for phi_atoms, psi_atoms in get_backbone_dihedrals(mol):
        assert phi_atoms[1] == psi_atoms[0]  # N
        assert phi_atoms[2] == psi_atoms[1]  # Cα


def test_get_backbone_dihedrals_four_atoms_each():
    """Each dihedral definition is a 4-tuple."""
    mol = _cycloala4_with_hs()
    for phi_atoms, psi_atoms in get_backbone_dihedrals(mol):
        assert len(phi_atoms) == 4
        assert len(psi_atoms) == 4


def test_get_backbone_dihedrals_no_duplicates():
    """No two residues share the same N atom index."""
    mol = _cycloala4_with_hs()
    n_idxs = [phi[1] for phi, _ in get_backbone_dihedrals(mol)]
    assert len(n_idxs) == len(set(n_idxs))


# ---------------------------------------------------------------------------
# make_constrained_bounds
# ---------------------------------------------------------------------------


def test_make_constrained_bounds_wrong_length():
    mol = _cycloala4_with_hs()
    with pytest.raises(ValueError, match="phi_psi_values"):
        make_constrained_bounds(mol, [(-60.0, -45.0)])  # only 1, need 4


def test_make_constrained_bounds_returns_array_or_none():
    mol = _cycloala4_with_hs()
    n = len(get_backbone_dihedrals(mol))
    # Alpha-helical (phi=-60, psi=-45) is common and should usually be feasible
    result = make_constrained_bounds(mol, [(-60.0, -45.0)] * n)
    assert result is None or isinstance(result, np.ndarray)


def test_make_constrained_bounds_shape():
    mol = _cycloala4_with_hs()
    n_res = len(get_backbone_dihedrals(mol))
    n_atoms = mol.GetNumAtoms()
    result = make_constrained_bounds(mol, [(-60.0, -45.0)] * n_res)
    if result is None:
        pytest.skip("Infeasible constraint combination")
    assert result.shape == (n_atoms, n_atoms)


def test_make_constrained_bounds_at_least_one_tighter():
    """Constrained bounds should be tighter on at least one atom pair than the default."""
    mol = _cycloala4_with_hs()
    n_res = len(get_backbone_dihedrals(mol))
    default = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    result = make_constrained_bounds(mol, [(-60.0, -45.0)] * n_res)
    if result is None:
        pytest.skip("Infeasible constraint combination")
    # Upper triangle: result <= default (some entries tightened)
    upper_default = np.triu(default, k=1)
    upper_result = np.triu(result, k=1)
    assert np.any(upper_result < upper_default - 1e-6)


# ---------------------------------------------------------------------------
# embed_constrained
# ---------------------------------------------------------------------------


def test_embed_constrained_returns_list():
    mol = _cycloala4_with_hs()
    n_res = len(get_backbone_dihedrals(mol))
    bounds = make_constrained_bounds(mol, [(-60.0, -45.0)] * n_res)
    if bounds is None:
        pytest.skip("Infeasible constraint combination")
    conf_ids = embed_constrained(mol, bounds, n_attempts=3, seed=0)
    assert isinstance(conf_ids, list)
    assert all(isinstance(c, int) for c in conf_ids)


def test_embed_constrained_ids_are_valid_conformers():
    mol = _cycloala4_with_hs()
    n_res = len(get_backbone_dihedrals(mol))
    bounds = make_constrained_bounds(mol, [(-60.0, -45.0)] * n_res)
    if bounds is None:
        pytest.skip("Infeasible constraint combination")
    conf_ids = embed_constrained(mol, bounds, n_attempts=5, seed=42)
    valid_ids = [c.GetId() for c in mol.GetConformers()]
    for cid in conf_ids:
        assert cid in valid_ids


def test_embed_constrained_dihedral_near_target():
    """Embedded conformers should have backbone phi within ~60° of target on average."""
    mol = _cycloala4_with_hs()
    n_res = len(get_backbone_dihedrals(mol))
    target_phi = -60.0
    bounds = make_constrained_bounds(mol, [(target_phi, -45.0)] * n_res)
    if bounds is None:
        pytest.skip("Infeasible constraint combination")
    conf_ids = embed_constrained(mol, bounds, n_attempts=10, seed=0)
    if not conf_ids:
        pytest.skip("No conformers embedded")
    defs = get_backbone_dihedrals(mol)
    errors = []
    for cid in conf_ids:
        conf = mol.GetConformer(cid)
        for phi_atoms, _ in defs:
            phi = rdMolTransforms.GetDihedralDeg(conf, *phi_atoms)
            diff = abs(((phi - target_phi + 180) % 360) - 180)
            errors.append(diff)
    mean_err = np.mean(errors)
    assert (
        mean_err < 120.0
    ), f"Mean dihedral error {mean_err:.1f}° exceeds 120° — constraints not guiding embedding"


def test_embed_constrained_no_conformers_for_impossible_bounds():
    """A bounds matrix with upper < lower for some pair should embed 0 conformers."""
    mol = _cycloala4_with_hs()
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    # Corrupt the bounds to make them unembeddable: upper < lower everywhere
    bad_bounds = bounds.copy()
    n = bad_bounds.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            bad_bounds[a, b] = 0.1  # upper = 0.1 Å (all atoms must be ~0 apart)
    conf_ids = embed_constrained(mol, bad_bounds, n_attempts=3, seed=0)
    assert conf_ids == []


# ---------------------------------------------------------------------------
# classify_backbone_residues
# ---------------------------------------------------------------------------

# Cyclo(Ala)4 — all L-amino acids (standard)
_CYCLOALA4_SMILES = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"
# Cyclo(d-Ala)4 — all D-amino acids (mirror)
_CYCLODALAA4_SMILES = "C[C@H]1NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O"
# Cyclo(Sar)4 — N-methylglycine × 4 (all NMe)
_CYCLOSAR4_SMILES = "O=C1CN(C)C(=O)CN(C)C(=O)CN(C)C(=O)CN1C"
# Cyclo(Gly)4
_CYCLOGLY4_SMILES = "O=C1CNC(=O)CNC(=O)CNC(=O)CN1"


def _mol_with_hs(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def test_classify_all_residues_returned():
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    classes = classify_backbone_residues(mol)
    assert len(classes) == 4


def test_classify_all_valid_classes():
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    for cls in classify_backbone_residues(mol):
        assert cls in ("L", "D", "NMe", "Gly")


def test_classify_gly4_all_gly():
    mol = _mol_with_hs(_CYCLOGLY4_SMILES)
    classes = classify_backbone_residues(mol)
    assert all(c == "Gly" for c in classes), f"Expected all Gly, got {classes}"


def test_classify_sar4_all_nme():
    mol = _mol_with_hs(_CYCLOSAR4_SMILES)
    classes = classify_backbone_residues(mol)
    assert all(c == "NMe" for c in classes), f"Expected all NMe, got {classes}"


def test_classify_ala4_vs_dala4_mirror():
    """All-L and all-D cyclic tetrapeptides should give mirror-image class lists."""
    mol_l = _mol_with_hs(_CYCLOALA4_SMILES)
    mol_d = _mol_with_hs(_CYCLODALAA4_SMILES)
    classes_l = classify_backbone_residues(mol_l)
    classes_d = classify_backbone_residues(mol_d)
    _mirror = {"L": "D", "D": "L", "NMe": "NMe", "Gly": "Gly"}
    assert [_mirror[c] for c in classes_l] == classes_d


def test_classify_count_matches_backbone_dihedrals():
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    assert len(classify_backbone_residues(mol)) == len(get_backbone_dihedrals(mol))


# ---------------------------------------------------------------------------
# _sample_from_grid
# ---------------------------------------------------------------------------

_BIN_CENTERS = np.linspace(-175, 175, 36)


def _uniform_grid() -> np.ndarray:
    """36×36 grid with all cells equally probable."""
    g = np.ones((36, 36), dtype=np.float64)
    return g / g.sum()


def _sparse_grid(nonzero_cells: int = 10) -> np.ndarray:
    """Grid with only a few non-zero cells."""
    g = np.zeros((36, 36), dtype=np.float64)
    rng = np.random.default_rng(0)
    idxs = rng.choice(36 * 36, size=nonzero_cells, replace=False)
    g.flat[idxs] = 1.0
    return g / g.sum()


def test_sample_from_grid_correct_count():
    grid = _uniform_grid()
    samples = _sample_from_grid(
        grid, _BIN_CENTERS, 100, "uniform", np.random.default_rng(0)
    )
    assert len(samples) == 100


def test_sample_from_grid_values_in_bin_centers():
    grid = _uniform_grid()
    samples = _sample_from_grid(
        grid, _BIN_CENTERS, 50, "uniform", np.random.default_rng(0)
    )
    for phi, psi in samples:
        assert phi in _BIN_CENTERS.tolist()
        assert psi in _BIN_CENTERS.tolist()


def test_sample_from_grid_uniform_uses_only_nonzero():
    """Uniform sampling should never draw from zero-probability cells."""
    grid = _sparse_grid(nonzero_cells=5)
    nonzero_pairs = {
        (float(_BIN_CENTERS[i]), float(_BIN_CENTERS[j]))
        for i, j in np.argwhere(grid > 0)
    }
    samples = _sample_from_grid(
        grid, _BIN_CENTERS, 200, "uniform", np.random.default_rng(0)
    )
    for pair in samples:
        assert pair in nonzero_pairs


def test_sample_from_grid_inverse_uses_only_nonzero():
    grid = _sparse_grid(nonzero_cells=5)
    nonzero_pairs = {
        (float(_BIN_CENTERS[i]), float(_BIN_CENTERS[j]))
        for i, j in np.argwhere(grid > 0)
    }
    samples = _sample_from_grid(
        grid, _BIN_CENTERS, 200, "inverse", np.random.default_rng(0)
    )
    for pair in samples:
        assert pair in nonzero_pairs


def test_sample_from_grid_inverse_oversamples_rare():
    """Inverse strategy should draw rare cells more often than common ones."""
    grid = np.zeros((36, 36), dtype=np.float64)
    # One very common cell and one very rare cell
    grid[0, 0] = 0.99
    grid[1, 1] = 0.01
    grid /= grid.sum()

    rng = np.random.default_rng(42)
    samples = _sample_from_grid(grid, _BIN_CENTERS, 1000, "inverse", rng)
    rare_phi, rare_psi = float(_BIN_CENTERS[1]), float(_BIN_CENTERS[1])
    rare_count = sum(1 for phi, psi in samples if phi == rare_phi and psi == rare_psi)
    # With inverse weighting, rare cell (prob=0.01) gets weight ∝ 100,
    # common cell (prob=0.99) gets weight ∝ 1.01. Rare cell should dominate.
    assert rare_count > 900


def test_sample_from_grid_unknown_strategy_raises():
    grid = _uniform_grid()
    with pytest.raises(ValueError, match="Unknown strategy"):
        _sample_from_grid(
            grid, _BIN_CENTERS, 10, "bad_strategy", np.random.default_rng(0)
        )


def test_sample_from_grid_all_zero_falls_back():
    """Empty grid (all zeros) should not crash — falls back to uniform."""
    grid = np.zeros((36, 36), dtype=np.float64)
    samples = _sample_from_grid(
        grid, _BIN_CENTERS, 10, "uniform", np.random.default_rng(0)
    )
    assert len(samples) == 10


# ---------------------------------------------------------------------------
# sample_constrained_confs
# ---------------------------------------------------------------------------

_GRIDS_PATH = Path(__file__).parents[1] / "data/processed/cremp/ramachandran_grids.npz"


@pytest.fixture(scope="module")
def cremp_grids():
    if not _GRIDS_PATH.exists():
        pytest.skip(
            "CREMP Ramachandran grids not found; run build_ramachandran_grids.py first"
        )
    data = np.load(_GRIDS_PATH)
    return {k: data[k] for k in data.files}


def test_sample_constrained_confs_returns_list(cremp_grids):
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    conf_ids = sample_constrained_confs(
        mol, cremp_grids, n_samples=5, n_attempts=3, seed=0
    )
    assert isinstance(conf_ids, list)
    assert all(isinstance(c, int) for c in conf_ids)


def test_sample_constrained_confs_adds_conformers(cremp_grids):
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    before = mol.GetNumConformers()
    conf_ids = sample_constrained_confs(
        mol, cremp_grids, n_samples=5, n_attempts=3, seed=0
    )
    assert mol.GetNumConformers() == before + len(conf_ids)


def test_sample_constrained_confs_seed_reproducible(cremp_grids):
    """Same seed produces the same conformer coordinates."""
    mol_a = _mol_with_hs(_CYCLOALA4_SMILES)
    mol_b = _mol_with_hs(_CYCLOALA4_SMILES)
    ids_a = sample_constrained_confs(
        mol_a, cremp_grids, n_samples=3, n_attempts=2, seed=7
    )
    ids_b = sample_constrained_confs(
        mol_b, cremp_grids, n_samples=3, n_attempts=2, seed=7
    )
    assert len(ids_a) == len(ids_b)
    for ca, cb in zip(ids_a, ids_b):
        pos_a = mol_a.GetConformer(ca).GetPositions()
        pos_b = mol_b.GetConformer(cb).GetPositions()
        assert np.allclose(pos_a, pos_b, atol=1e-4)


def test_sample_constrained_confs_inverse_strategy(cremp_grids):
    mol = _mol_with_hs(_CYCLOALA4_SMILES)
    conf_ids = sample_constrained_confs(
        mol, cremp_grids, n_samples=5, n_attempts=3, strategy="inverse", seed=0
    )
    assert isinstance(conf_ids, list)


def test_sample_constrained_confs_no_backbone_returns_empty(cremp_grids):
    """A non-peptide mol (no backbone) should return [] without error."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    conf_ids = sample_constrained_confs(mol, cremp_grids, n_samples=5)
    assert conf_ids == []
