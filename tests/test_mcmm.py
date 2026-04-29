"""Unit tests for src/mcmm.py.

Steps 3 and 4 of the issue-#11 implementation: backbone window
enumeration and the BasinMemory data structure. Fixtures for window
tests are cyclo(Ala)N peptides; basin-memory tests use synthetic
torch tensors to isolate the metric and bookkeeping invariants.
"""

import math

import pytest
import torch
from rdkit import Chem

from mcmm import (
    DEFAULT_RMSD_THRESHOLD,
    WINDOW_SIZE,
    BasinMemory,
    _ordered_backbone_residues,
    enumerate_backbone_windows,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Cyclo(Ala)N — head-to-tail cyclic homopolymer of L-alanine. The pattern is
# one [C@@H](C)NC(=O) per residue with ring closure via the final NC1=O.
_CYCLOALA4 = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"
_CYCLOALA6 = (
    "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)"
    "[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"
)


def _cycloala_mol(n_residues: int) -> Chem.Mol:
    """Build a cyclo(Ala)N mol with explicit Hs."""
    smiles_map = {4: _CYCLOALA4, 6: _CYCLOALA6}
    mol = Chem.MolFromSmiles(smiles_map[n_residues])
    return Chem.AddHs(mol)


# ---------------------------------------------------------------------------
# enumerate_backbone_windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_residues, expected_windows",
    [(4, 12), (6, 18)],
)
def test_window_count_matches_backbone_atom_count(n_residues, expected_windows):
    """Cyclo(Ala)N has 3N backbone atoms and 3N cyclic 7-atom windows."""
    mol = _cycloala_mol(n_residues)
    windows = enumerate_backbone_windows(mol)
    assert len(windows) == expected_windows


def test_each_window_has_seven_distinct_atoms():
    """For peptides of ≥ 7 backbone atoms, no atom repeats inside a single
    window."""
    mol = _cycloala_mol(4)
    windows = enumerate_backbone_windows(mol)
    for window in windows:
        assert len(window) == WINDOW_SIZE
        assert len(set(window)) == WINDOW_SIZE


def test_window_atoms_are_sequentially_bonded():
    """Consecutive atoms inside a window are bonded in the molecule — the
    window traces an actual path through the backbone ring."""
    mol = _cycloala_mol(4)
    windows = enumerate_backbone_windows(mol)
    for window in windows:
        for i in range(WINDOW_SIZE - 1):
            bond = mol.GetBondBetweenAtoms(window[i], window[i + 1])
            assert (
                bond is not None
            ), f"window {window}: atoms {window[i]} and {window[i+1]} not bonded"


def test_windows_cover_every_backbone_atom():
    """The union of atoms across all 3K windows equals the full set of
    3K backbone atoms — no gaps in the ring walk."""
    mol = _cycloala_mol(4)
    windows = enumerate_backbone_windows(mol)
    union = set()
    for window in windows:
        union.update(window)
    assert len(union) == 12  # 4 residues × (N, Cα, C)


def test_windows_are_distinct_cyclic_shifts():
    """Each window starts at a different backbone atom; the set of starting
    atoms equals the set of all backbone atoms."""
    mol = _cycloala_mol(6)
    windows = enumerate_backbone_windows(mol)
    starts = [w[0] for w in windows]
    assert len(set(starts)) == len(starts) == 18


def test_returns_empty_for_non_peptide():
    """A mol with no recognised peptide backbone (cyclohexane) returns []."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    assert enumerate_backbone_windows(mol) == []


def test_raises_for_linear_peptide():
    """A linear (non-cyclic) peptide raises ValueError because the C→N walk
    cannot close the ring."""
    # Linear tripeptide H-Ala-Ala-Ala-OH
    linear = "C[C@@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"
    mol = Chem.AddHs(Chem.MolFromSmiles(linear))
    with pytest.raises(ValueError, match="ring did not close"):
        enumerate_backbone_windows(mol)


# ---------------------------------------------------------------------------
# _ordered_backbone_residues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_residues", [4, 6])
def test_ordered_residues_count_matches_input(n_residues):
    mol = _cycloala_mol(n_residues)
    residues = _ordered_backbone_residues(mol)
    assert len(residues) == n_residues


def test_ordered_residues_each_has_n_ca_c():
    """Each residue tuple is (N, Cα, C) — N is nitrogen, Cα and C are
    distinct carbons."""
    mol = _cycloala_mol(4)
    residues = _ordered_backbone_residues(mol)
    for n_idx, ca_idx, c_idx in residues:
        assert n_idx != ca_idx and ca_idx != c_idx and n_idx != c_idx
        assert mol.GetAtomWithIdx(n_idx).GetAtomicNum() == 7
        assert mol.GetAtomWithIdx(ca_idx).GetAtomicNum() == 6
        assert mol.GetAtomWithIdx(c_idx).GetAtomicNum() == 6


def test_ordered_residues_form_cyclic_peptide_chain():
    """For each consecutive pair (i, i+1), the C of residue i is bonded
    to the N of residue i+1 — including the wrap-around (last residue
    back to first)."""
    mol = _cycloala_mol(4)
    residues = _ordered_backbone_residues(mol)
    n_res = len(residues)
    for i in range(n_res):
        _, _, c = residues[i]
        n_next, _, _ = residues[(i + 1) % n_res]
        assert (
            mol.GetBondBetweenAtoms(c, n_next) is not None
        ), f"peptide bond missing between residue {i} and residue {(i+1) % n_res}"


def test_ordered_residues_within_residue_bonds():
    """Within each residue, N is bonded to Cα and Cα is bonded to C."""
    mol = _cycloala_mol(4)
    residues = _ordered_backbone_residues(mol)
    for n_idx, ca_idx, c_idx in residues:
        assert mol.GetBondBetweenAtoms(n_idx, ca_idx) is not None
        assert mol.GetBondBetweenAtoms(ca_idx, c_idx) is not None


# ---------------------------------------------------------------------------
# BasinMemory — fixtures and helpers
# ---------------------------------------------------------------------------

# Synthetic 4-atom conformer fixture: a rigid line along the x-axis, shifted
# uniformly. This mirrors the _line_coords pattern used in
# tests/test_exhaustive_etkdg.py so basin distances are easy to reason about
# (translation by Δx → normalised L1 distance = |Δx|).
_TEST_N_ATOMS = 4


def _line_conformer(x_offset: float) -> torch.Tensor:
    """4-atom line shifted by `x_offset` along x. Pairwise distance between
    two such conformers (offsets x1, x2) is |x1 - x2|."""
    base = torch.zeros(_TEST_N_ATOMS, 3, dtype=torch.float64)
    base[:, 0] = torch.arange(_TEST_N_ATOMS, dtype=torch.float64)
    base[:, 0] += x_offset
    return base


# ---------------------------------------------------------------------------
# BasinMemory — construction and shape contract
# ---------------------------------------------------------------------------


def test_basin_memory_starts_empty():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    assert mem.n_basins == 0
    assert mem.coords.shape == (0, _TEST_N_ATOMS, 3)
    assert mem.energies.shape == (0,)
    assert mem.usages.shape == (0,)


def test_basin_memory_invalid_constructor_raises():
    with pytest.raises(ValueError, match="n_atoms must be positive"):
        BasinMemory(n_atoms=0)
    with pytest.raises(ValueError, match="rmsd_threshold must be positive"):
        BasinMemory(n_atoms=4, rmsd_threshold=0.0)


def test_basin_memory_default_threshold_matches_dedup():
    """Default threshold matches _energy_ranked_dedup's default rmsd_threshold
    so basin definitions are interchangeable across the get_mol_PE_* family."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    assert mem.rmsd_threshold == DEFAULT_RMSD_THRESHOLD == 0.1


# ---------------------------------------------------------------------------
# add_basin
# ---------------------------------------------------------------------------


def test_add_basin_appends_and_returns_index():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    idx0 = mem.add_basin(_line_conformer(0.0), energy=-1.5)
    idx1 = mem.add_basin(_line_conformer(5.0), energy=-1.2)
    assert idx0 == 0 and idx1 == 1
    assert mem.n_basins == 2
    assert mem.energies.tolist() == pytest.approx([-1.5, -1.2])
    assert mem.usages.tolist() == [1, 1]


def test_add_basin_invalid_shape_raises():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    with pytest.raises(ValueError, match="coords must be"):
        mem.add_basin(torch.zeros(3, 3), energy=0.0)
    with pytest.raises(ValueError, match="coords must be"):
        mem.add_basin(torch.zeros(_TEST_N_ATOMS, 2), energy=0.0)


# ---------------------------------------------------------------------------
# query_novelty — single
# ---------------------------------------------------------------------------


def test_query_novelty_empty_returns_none_and_inf():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    idx, dist = mem.query_novelty(_line_conformer(0.0))
    assert idx is None
    assert dist == math.inf


def test_query_novelty_exact_match_returns_zero_distance():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    idx, dist = mem.query_novelty(_line_conformer(0.0))
    assert idx == 0
    assert dist == pytest.approx(0.0, abs=1e-12)


def test_query_novelty_far_returns_none_with_finite_distance():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.1)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    idx, dist = mem.query_novelty(_line_conformer(5.0))
    assert idx is None
    # Pairwise normalised L1 between line conformers at offsets 0 and 5:
    # 4 atoms × 5 / (3 × 4) = 5/3 ≈ 1.667
    assert dist == pytest.approx(5.0 / 3.0, abs=1e-12)


def test_query_novelty_threshold_boundary_strictly_less_than():
    """Strict `<` matches `_energy_ranked_dedup`: distance EQUAL to the
    threshold is treated as a different basin."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    # Offset 1.5 → normalised L1 = 4×1.5 / (3×4) = 0.5 — exactly the threshold.
    idx, dist = mem.query_novelty(_line_conformer(1.5))
    assert idx is None  # strictly < threshold required for same basin
    assert dist == pytest.approx(0.5, abs=1e-12)


def test_query_novelty_picks_closest_when_multiple_match():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=1.0)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    mem.add_basin(_line_conformer(0.5), energy=0.0)
    # Query at 0.4: distances are 0.4/3 ≈ 0.133 and 0.1/3 ≈ 0.033.
    # Both within threshold; closest is basin 1.
    idx, dist = mem.query_novelty(_line_conformer(0.4))
    assert idx == 1
    assert dist == pytest.approx(0.1 / 3.0, abs=1e-12)


# ---------------------------------------------------------------------------
# query_novelty_batch
# ---------------------------------------------------------------------------


def test_query_batch_empty_memory_returns_novel_for_all():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    batch = torch.stack([_line_conformer(x) for x in [0.0, 1.0, 2.0]])
    indices, distances = mem.query_novelty_batch(batch)
    assert (indices == BasinMemory.NOVEL).all()
    assert (distances == math.inf).all()


def test_query_batch_matches_individual_queries():
    """Batched query produces the same per-row results as B individual
    `query_novelty` calls. Locks the broadcast geometry."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    mem.add_basin(_line_conformer(5.0), energy=0.0)

    queries = [0.0, 0.4, 1.5, 5.05, 10.0]
    batch = torch.stack([_line_conformer(x) for x in queries])
    batch_indices, batch_dists = mem.query_novelty_batch(batch)

    for i, x in enumerate(queries):
        ref_idx, ref_dist = mem.query_novelty(_line_conformer(x))
        expected_idx = ref_idx if ref_idx is not None else BasinMemory.NOVEL
        assert int(batch_indices[i].item()) == expected_idx
        assert float(batch_dists[i].item()) == pytest.approx(ref_dist, abs=1e-12)


def test_query_batch_invalid_shape_raises():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    with pytest.raises(ValueError, match="coords_batch must be"):
        mem.query_novelty_batch(torch.zeros(_TEST_N_ATOMS, 3))  # missing batch dim
    with pytest.raises(ValueError, match="coords_batch must be"):
        mem.query_novelty_batch(torch.zeros(2, 3, 3))  # wrong n_atoms


# ---------------------------------------------------------------------------
# record_visit / usages
# ---------------------------------------------------------------------------


def test_record_visit_increments_usage():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    idx = mem.add_basin(_line_conformer(0.0), energy=0.0)
    assert int(mem.usages[idx].item()) == 1
    mem.record_visit(idx)
    mem.record_visit(idx)
    assert int(mem.usages[idx].item()) == 3


def test_record_visit_invalid_index_raises():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    with pytest.raises(IndexError, match="out of range"):
        mem.record_visit(0)  # empty memory
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    with pytest.raises(IndexError, match="out of range"):
        mem.record_visit(5)
    with pytest.raises(IndexError, match="out of range"):
        mem.record_visit(-1)


# ---------------------------------------------------------------------------
# acceptance_bias
# ---------------------------------------------------------------------------


def test_acceptance_bias_novel_returns_one():
    """A None idx means "novel basin" — no bias against acceptance."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    assert mem.acceptance_bias(None) == 1.0


def test_acceptance_bias_decays_as_one_over_sqrt_usage():
    """Saunders 1/√usage form. After the discovery (usage=1) the bias is 1;
    after one re-visit (usage=2) it is 1/√2; etc."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    idx = mem.add_basin(_line_conformer(0.0), energy=0.0)
    assert mem.acceptance_bias(idx) == pytest.approx(1.0)
    mem.record_visit(idx)
    assert mem.acceptance_bias(idx) == pytest.approx(1.0 / math.sqrt(2))
    mem.record_visit(idx)
    assert mem.acceptance_bias(idx) == pytest.approx(1.0 / math.sqrt(3))


def test_acceptance_bias_invalid_index_raises():
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS)
    mem.add_basin(_line_conformer(0.0), energy=0.0)
    with pytest.raises(IndexError, match="out of range"):
        mem.acceptance_bias(5)


# ---------------------------------------------------------------------------
# Driver-flow integration
# ---------------------------------------------------------------------------


def test_basin_memory_driver_loop_smoke():
    """End-to-end shape: discover three distinct basins, re-visit one,
    verify usage counts and that the bias decays correctly. This is the
    pattern the MCMM driver (Step 5) will use per accepted move."""
    mem = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)

    # Three proposals at distant offsets — all novel.
    proposals = [
        (_line_conformer(0.0), -2.0),
        (_line_conformer(5.0), -1.5),
        (_line_conformer(10.0), -1.0),
    ]
    for coords, e in proposals:
        idx, _ = mem.query_novelty(coords)
        if idx is None:
            mem.add_basin(coords, energy=e)
        else:
            mem.record_visit(idx)

    assert mem.n_basins == 3
    assert mem.usages.tolist() == [1, 1, 1]

    # Re-visit basin 1 (offset 5.0) twice via slightly perturbed coords.
    for x in [5.05, 4.97]:
        idx, _ = mem.query_novelty(_line_conformer(x))
        assert idx == 1
        mem.record_visit(idx)

    assert mem.n_basins == 3
    assert mem.usages.tolist() == [1, 3, 1]
    # Saunders bias against re-visiting basin 1 should be 1/√3.
    assert mem.acceptance_bias(1) == pytest.approx(1.0 / math.sqrt(3))
