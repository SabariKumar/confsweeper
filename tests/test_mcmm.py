"""Unit tests for src/mcmm.py.

Steps 3 and 4 of the issue-#11 implementation: backbone window
enumeration and the BasinMemory data structure. Fixtures for window
tests are cyclo(Ala)N peptides; basin-memory tests use synthetic
torch tensors to isolate the metric and bookkeeping invariants.
"""

import math

import numpy as np
import pytest
import torch
from rdkit import Chem

from mcmm import (
    DEFAULT_RMSD_THRESHOLD,
    WINDOW_SIZE,
    BasinMemory,
    MCMMWalker,
    ParallelMCMMDriver,
    ReplicaExchangeMCMMDriver,
    _backbone_atom_set,
    _compute_window_downstream_sets,
    _ordered_backbone_residues,
    _side_chain_group,
    _swap_walker_configs,
    enumerate_backbone_windows,
    make_mcmm_proposer,
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


def test_returns_empty_for_linear_peptide():
    """A linear (non-cyclic) peptide has no rings, so ring-info-based
    enumeration returns []. The MCMM proposer factory then raises a
    clearer 'no enumerable backbone windows' error at construction
    time."""
    # Linear tripeptide H-Ala-Ala-Ala-OH
    linear = "C[C@@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"
    mol = Chem.AddHs(Chem.MolFromSmiles(linear))
    assert enumerate_backbone_windows(mol) == []


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
    with pytest.raises(ValueError, match="rmsd_threshold must be non-negative"):
        BasinMemory(n_atoms=4, rmsd_threshold=-0.1)


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


# ---------------------------------------------------------------------------
# MCMMWalker — proposer mocks and helpers
# ---------------------------------------------------------------------------


def _fixed_proposer(
    target_x: float, target_e: float, det_j: float = 1.0, success: bool = True
):
    """
    Build a propose_fn that always proposes a specific (coords, energy).

    Returns a callable matching the MCMMWalker propose_fn contract:
        (current_coords) -> (new_coords, new_energy, det_j, success).
    """

    def fn(_):
        return _line_conformer(target_x), target_e, det_j, success

    return fn


def _make_walker(
    initial_x: float = 0.0,
    initial_e: float = 0.0,
    kt: float = 1.0,
    threshold: float = 0.5,
    random_fn=None,
    memory=None,
):
    """Construct an MCMMWalker over a fresh BasinMemory unless one is provided."""
    coords = _line_conformer(initial_x)
    if memory is None:
        memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=threshold)
    return MCMMWalker(coords, initial_e, kt=kt, memory=memory, random_fn=random_fn)


# ---------------------------------------------------------------------------
# MCMMWalker — construction
# ---------------------------------------------------------------------------


def test_walker_initial_state_added_to_fresh_memory():
    walker = _make_walker(initial_x=0.0, initial_e=-2.5)
    assert walker.energy == -2.5
    assert walker.memory.n_basins == 1
    assert walker.current_basin_idx == 0
    assert walker.n_steps == 0 and walker.n_accepted == 0


def test_walker_initial_state_latches_to_existing_memory_basin():
    """If the memory already contains the initial basin (e.g. shared with
    other walkers), the walker latches onto the existing index without
    incrementing its usage count."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    memory.add_basin(_line_conformer(0.0), energy=-1.0)  # usage = 1
    assert memory.n_basins == 1 and int(memory.usages[0].item()) == 1

    walker = MCMMWalker(_line_conformer(0.0), 0.0, kt=1.0, memory=memory)
    assert walker.current_basin_idx == 0
    assert memory.n_basins == 1  # no duplicate added
    assert int(memory.usages[0].item()) == 1  # not incremented


def test_walker_negative_kt_raises():
    with pytest.raises(ValueError, match="kt must be non-negative"):
        _make_walker(kt=-1.0)


# ---------------------------------------------------------------------------
# MCMMWalker — temperature limits
# ---------------------------------------------------------------------------


def test_walker_kt_zero_rejects_uphill():
    """Greedy descent: a worse-energy proposal is always rejected at T=0."""
    walker = _make_walker(initial_e=0.0, kt=0.0, random_fn=lambda: 0.5)
    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=10.0))
    assert not accepted
    assert walker.energy == 0.0  # state unchanged


def test_walker_kt_zero_accepts_downhill():
    """Greedy descent: a better-energy proposal is always accepted at T=0,
    regardless of bias × det_J (the energy term dominates in the limit)."""
    walker = _make_walker(initial_e=0.0, kt=0.0, random_fn=lambda: 0.999)
    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=-10.0))
    assert accepted
    assert walker.energy == -10.0


def test_walker_kt_inf_accepts_iso_energy_iso_bias():
    """At T=∞ with bias=1 and det_J=1, every proposal is accepted regardless
    of the energy difference."""
    walker = _make_walker(kt=math.inf, random_fn=lambda: 0.999)
    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=1000.0))
    assert accepted
    assert walker.energy == 1000.0


# ---------------------------------------------------------------------------
# MCMMWalker — geometry rejection
# ---------------------------------------------------------------------------


def test_walker_rejects_geometrically_infeasible_proposal():
    """When the proposer signals success=False, the walker rejects without
    inspecting energy or memory."""
    walker = _make_walker()
    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=-10.0, success=False))
    assert not accepted
    assert walker.energy == 0.0
    assert walker.memory.n_basins == 1  # no new basin recorded
    assert walker.n_steps == 1


# ---------------------------------------------------------------------------
# MCMMWalker — memory bookkeeping
# ---------------------------------------------------------------------------


def test_walker_memory_grows_monotonically_under_always_accept():
    """Visiting a sequence of distinct basins under always-accept mode
    grows the basin set by exactly one per accept."""
    walker = _make_walker(kt=math.inf, random_fn=lambda: 0.0)
    for x in [5.0, 10.0, 15.0, 20.0, 25.0]:
        walker.step(_fixed_proposer(target_x=x, target_e=0.0))
    # Initial basin + 5 distinct novel basins
    assert walker.memory.n_basins == 6
    assert walker.memory.usages.tolist() == [1, 1, 1, 1, 1, 1]
    assert walker.n_accepted == 5
    assert walker.acceptance_rate == 1.0


def test_walker_revisit_increments_existing_basin_usage():
    """Returning to a previously-visited basin increments its usage rather
    than adding a duplicate."""
    walker = _make_walker(kt=math.inf, random_fn=lambda: 0.0)
    walker.step(_fixed_proposer(target_x=5.0, target_e=0.0))  # add basin 1
    walker.step(_fixed_proposer(target_x=0.0, target_e=0.0))  # back to 0; usage[0]=2
    walker.step(_fixed_proposer(target_x=5.0, target_e=0.0))  # back to 1; usage[1]=2
    assert walker.memory.n_basins == 2
    assert walker.memory.usages.tolist() == [2, 2]


def test_walker_rejected_step_leaves_memory_unchanged():
    """Memory is mutated only on accept. A rejected proposal does not
    increment any counter or add a basin."""
    walker = _make_walker(initial_e=0.0, kt=0.0, random_fn=lambda: 0.5)
    n_basins_before = walker.memory.n_basins
    walker.step(_fixed_proposer(target_x=5.0, target_e=10.0))  # uphill, T=0 → reject
    assert walker.memory.n_basins == n_basins_before
    assert walker.memory.usages.tolist() == [1]


# ---------------------------------------------------------------------------
# MCMMWalker — Saunders bias
# ---------------------------------------------------------------------------


def test_walker_saunders_bias_eventually_rejects_revisit():
    """With deterministic random_fn = 0.5 and ΔE = 0, acceptance is equivalent
    to bias × det_J ≥ 0.5. The 1/√k Saunders form drops below 0.5 at k=4
    (1/√4 = 0.5, fails the strict `<` check), so re-visit attempts to a
    basin already at usage=4 are rejected."""
    threshold = 10.0  # large so all `_line_conformer` offsets within 5.0 hit memory
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=threshold)
    # Pre-populate basin B (at offset 5) with usage=4 so the next attempted
    # entry has bias = 1/√4 = 0.5.
    memory.add_basin(_line_conformer(0.0), energy=0.0)
    memory.add_basin(_line_conformer(5.0), energy=0.0)
    memory.record_visit(1)
    memory.record_visit(1)
    memory.record_visit(1)
    assert int(memory.usages[1].item()) == 4

    walker = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=1.0, memory=memory, random_fn=lambda: 0.5
    )

    # Propose entry into basin B (ΔE=0, bias=1/√4=0.5). With rng=0.5 the
    # strict `<` rejects.
    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=0.0))
    assert not accepted
    assert int(memory.usages[1].item()) == 4  # unchanged


def test_walker_saunders_bias_admits_revisit_below_threshold():
    """Below the bias threshold (usage ≤ 3 → bias ≥ 1/√3 ≈ 0.577 > 0.5),
    re-visits with rng = 0.5 and ΔE = 0 are accepted."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=10.0)
    memory.add_basin(_line_conformer(0.0), energy=0.0)
    memory.add_basin(_line_conformer(5.0), energy=0.0)
    memory.record_visit(1)
    memory.record_visit(1)
    assert int(memory.usages[1].item()) == 3  # bias = 1/√3 ≈ 0.577

    walker = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=1.0, memory=memory, random_fn=lambda: 0.5
    )

    accepted = walker.step(_fixed_proposer(target_x=5.0, target_e=0.0))
    assert accepted
    assert int(memory.usages[1].item()) == 4  # incremented after accept


# ---------------------------------------------------------------------------
# MCMMWalker — run loop
# ---------------------------------------------------------------------------


def test_walker_run_returns_accept_count_for_this_call():
    """`run(N)` returns the number of accepts during this call only —
    cumulative count is `walker.n_accepted`."""
    walker = _make_walker(kt=math.inf, random_fn=lambda: 0.0)
    n1 = walker.run(3, _fixed_proposer(target_x=10.0, target_e=0.0))
    n2 = walker.run(2, _fixed_proposer(target_x=20.0, target_e=0.0))
    # First run: initial basin (1) → step at x=10 (new basin), then 2 more
    # to the same basin. n1 = 3 accepts (all proposals accepted under T=∞).
    # Second run: 2 more accepts to a third basin.
    assert n1 == 3
    assert n2 == 2
    assert walker.n_accepted == 5
    assert walker.n_steps == 5


# ---------------------------------------------------------------------------
# ParallelMCMMDriver
# ---------------------------------------------------------------------------


def _scripted_batch_propose_fn(per_walker_proposals: list):
    """
    Deterministic batch proposer that emits one fixed proposal per call,
    cycling through `per_walker_proposals` for each step. Each entry is
    a list of (target_x, target_e, det_j, success) — one tuple per walker.

    Returns a callable matching `batch_propose_fn(coords_list) → list[...]`
    that ignores the input coords and returns the next scripted batch.
    """
    state = {"step": 0}

    def fn(coords_list):
        proposals_for_this_step = per_walker_proposals[state["step"]]
        state["step"] += 1
        out = []
        for target_x, target_e, det_j, success in proposals_for_this_step:
            out.append((_line_conformer(target_x), target_e, det_j, success))
        del coords_list  # ignored — fully scripted
        return out

    return fn


def test_parallel_driver_constructor_rejects_empty_walkers():
    with pytest.raises(ValueError, match="walkers list must be non-empty"):
        ParallelMCMMDriver(walkers=[], batch_propose_fn=lambda _: [])


def test_parallel_driver_n1_matches_single_walker_step():
    """N=1 ParallelMCMMDriver should produce identical state to a single
    MCMMWalker.step run with the equivalent propose_fn — same proposal,
    same RNG, same memory updates."""
    # Reference: single walker
    memory_ref = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    walker_ref = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=1.0, memory=memory_ref, random_fn=lambda: 0.1
    )
    walker_ref.step(_fixed_proposer(target_x=5.0, target_e=-1.0))

    # Parallel: N=1 driver with the same proposal
    memory_par = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    walker_par = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=1.0, memory=memory_par, random_fn=lambda: 0.1
    )
    batch_fn = _scripted_batch_propose_fn([[(5.0, -1.0, 1.0, True)]])
    driver = ParallelMCMMDriver([walker_par], batch_fn)
    driver.step()

    assert walker_par.coords.equal(walker_ref.coords)
    assert walker_par.energy == walker_ref.energy
    assert walker_par.n_accepted == walker_ref.n_accepted
    assert memory_par.n_basins == memory_ref.n_basins
    assert memory_par.usages.tolist() == memory_ref.usages.tolist()


def test_parallel_driver_step_returns_per_walker_results():
    """`step()` returns a list of accept/reject booleans, one per walker,
    in the walker-list order."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w0 = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=0.0, memory=memory, random_fn=lambda: 0.5
    )
    w1 = MCMMWalker(
        _line_conformer(20.0), 0.0, kt=0.0, memory=memory, random_fn=lambda: 0.5
    )
    # w0 proposes uphill (rejected at T=0); w1 proposes downhill (accepted).
    batch_fn = _scripted_batch_propose_fn(
        [
            [
                (10.0, 5.0, 1.0, True),  # uphill for w0 → reject
                (25.0, -5.0, 1.0, True),  # downhill for w1 → accept
            ]
        ]
    )
    driver = ParallelMCMMDriver([w0, w1], batch_fn)
    results = driver.step()
    assert results == [False, True]


def test_parallel_driver_disjoint_basins_independent_acceptance():
    """Two walkers proposing into completely disjoint novel basins each
    update memory once — no interference, both basins added."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w0 = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    w1 = MCMMWalker(
        _line_conformer(50.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    # Initial state: 2 basins.
    assert memory.n_basins == 2

    # w0 → x=10 (novel), w1 → x=60 (novel).
    batch_fn = _scripted_batch_propose_fn(
        [
            [(10.0, 0.0, 1.0, True), (60.0, 0.0, 1.0, True)],
        ]
    )
    driver = ParallelMCMMDriver([w0, w1], batch_fn)
    results = driver.step()
    assert results == [True, True]
    assert memory.n_basins == 4  # 2 initial + 2 new
    assert memory.usages.tolist() == [1, 1, 1, 1]


def test_parallel_driver_shared_basin_serializes_through_memory():
    """When two walkers in the SAME step propose into the same novel basin,
    the first to dispatch creates it (usage=1) and the second sees it as a
    known basin and increments instead of duplicating. This is the
    sequential-dispatch semantic."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w0 = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    w1 = MCMMWalker(
        _line_conformer(50.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    n_basins_initial = memory.n_basins  # 2

    # Both propose to the SAME conformer (x=100). The first to dispatch
    # adds it; the second finds it as known and increments.
    batch_fn = _scripted_batch_propose_fn(
        [
            [(100.0, 0.0, 1.0, True), (100.0, 0.0, 1.0, True)],
        ]
    )
    driver = ParallelMCMMDriver([w0, w1], batch_fn)
    driver.step()

    assert memory.n_basins == n_basins_initial + 1  # only one new basin, not two
    new_basin_idx = n_basins_initial
    assert int(memory.usages[new_basin_idx].item()) == 2  # 1 (add) + 1 (re-visit)


def test_parallel_driver_proposal_count_mismatch_raises():
    """If the batch proposer returns a list of the wrong length, step()
    raises before mutating any walker."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w0 = MCMMWalker(_line_conformer(0.0), 0.0, kt=1.0, memory=memory)
    w1 = MCMMWalker(_line_conformer(10.0), 0.0, kt=1.0, memory=memory)

    def wrong_count_fn(coords_list):
        return [(coords_list[0], 0.0, 1.0, True)]  # only 1 proposal for 2 walkers

    driver = ParallelMCMMDriver([w0, w1], wrong_count_fn)
    with pytest.raises(ValueError, match="returned 1 proposals; expected 2"):
        driver.step()


def test_parallel_driver_run_total_accept_count():
    """run(N) returns total accepts across all walkers and steps for this
    call only."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    walkers = [
        MCMMWalker(
            _line_conformer(x),
            0.0,
            kt=math.inf,
            memory=memory,
            random_fn=lambda: 0.0,
        )
        for x in [0.0, 50.0, 100.0]
    ]
    # 2 steps, 3 walkers each, all distinct novel basins, all accepts.
    batch_fn = _scripted_batch_propose_fn(
        [
            [(10.0, 0.0, 1.0, True), (60.0, 0.0, 1.0, True), (110.0, 0.0, 1.0, True)],
            [(20.0, 0.0, 1.0, True), (70.0, 0.0, 1.0, True), (120.0, 0.0, 1.0, True)],
        ]
    )
    driver = ParallelMCMMDriver(walkers, batch_fn)
    n_accepted = driver.run(2)
    assert n_accepted == 6  # 3 walkers × 2 steps
    assert driver.n_accepted == 6  # cumulative property
    assert driver.n_steps == 2


def test_parallel_driver_n_accepted_property_aggregates_walkers():
    """`driver.n_accepted` reflects the live accept counter sum across all
    walkers — even if walkers are stepped through other paths."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w0 = MCMMWalker(
        _line_conformer(0.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    w1 = MCMMWalker(
        _line_conformer(50.0), 0.0, kt=math.inf, memory=memory, random_fn=lambda: 0.0
    )
    driver = ParallelMCMMDriver(
        [w0, w1],
        _scripted_batch_propose_fn([[(5.0, 0.0, 1.0, True), (55.0, 0.0, 1.0, True)]]),
    )
    assert driver.n_accepted == 0
    driver.step()
    assert driver.n_accepted == 2
    # Stepping a walker outside the driver also reflects in n_accepted.
    w0.step(_fixed_proposer(target_x=100.0, target_e=0.0))
    assert driver.n_accepted == 3


# ---------------------------------------------------------------------------
# _swap_walker_configs
# ---------------------------------------------------------------------------


def test_swap_walker_configs_exchanges_state_in_place():
    """`_swap_walker_configs` swaps coords + energy + current_basin_idx
    while leaving kt and per-walker counters with the walker."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    w_low = MCMMWalker(_line_conformer(0.0), -1.0, kt=1.0, memory=memory)
    w_high = MCMMWalker(_line_conformer(50.0), 5.0, kt=2.0, memory=memory)
    # Step them so n_steps differ and we can verify counters don't swap.
    w_low.step(_fixed_proposer(target_x=0.0, target_e=-1.0, success=False))
    w_low.step(_fixed_proposer(target_x=0.0, target_e=-1.0, success=False))
    assert w_low.n_steps == 2 and w_high.n_steps == 0

    coords_low_before = w_low.coords.clone()
    coords_high_before = w_high.coords.clone()
    idx_low_before = w_low.current_basin_idx
    idx_high_before = w_high.current_basin_idx

    _swap_walker_configs(w_low, w_high)

    # Configs swapped:
    assert w_low.coords.equal(coords_high_before)
    assert w_high.coords.equal(coords_low_before)
    assert w_low.energy == 5.0
    assert w_high.energy == -1.0
    assert w_low.current_basin_idx == idx_high_before
    assert w_high.current_basin_idx == idx_low_before
    # kt and counters did NOT swap:
    assert w_low.kt == 1.0
    assert w_high.kt == 2.0
    assert w_low.n_steps == 2
    assert w_high.n_steps == 0


# ---------------------------------------------------------------------------
# ReplicaExchangeMCMMDriver — construction validation
# ---------------------------------------------------------------------------


def _make_remd_walkers(
    kts: list, n_per_temp: int = 1, threshold: float = 0.5, memory=None
):
    """Build a `walkers_by_temp` list with one walker per (temp, slot)
    pair. Each walker starts at a distant offset so initial basins are
    pairwise distinct (gap of 10 along the line, distance 10/3 ≈ 3.3
    well above the default threshold of 0.5)."""
    if memory is None:
        memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=threshold)
    groups = []
    for t, kt in enumerate(kts):
        group = []
        for i in range(n_per_temp):
            offset = 100 * t + 10 * i
            group.append(
                MCMMWalker(
                    _line_conformer(float(offset)),
                    energy=0.0,
                    kt=kt,
                    memory=memory,
                    random_fn=lambda: 0.5,
                )
            )
        groups.append(group)
    return groups, memory


def test_remd_constructor_rejects_empty_walkers():
    with pytest.raises(ValueError, match="must be non-empty"):
        ReplicaExchangeMCMMDriver([], lambda c: [])


def test_remd_constructor_rejects_empty_temperature_group():
    with pytest.raises(ValueError, match="at least one walker"):
        ReplicaExchangeMCMMDriver([[]], lambda c: [])


def test_remd_constructor_rejects_uneven_group_sizes():
    """All temperature groups must have the same number of walkers
    (paired-swap convention)."""
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    groups = [
        [MCMMWalker(_line_conformer(0.0), 0.0, kt=1.0, memory=memory)],
        [
            MCMMWalker(_line_conformer(50.0), 0.0, kt=2.0, memory=memory),
            MCMMWalker(_line_conformer(60.0), 0.0, kt=2.0, memory=memory),
        ],
    ]
    with pytest.raises(ValueError, match="same number of"):
        ReplicaExchangeMCMMDriver(groups, lambda c: [])


def test_remd_constructor_rejects_mixed_kt_within_group():
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    groups = [
        [
            MCMMWalker(_line_conformer(0.0), 0.0, kt=1.0, memory=memory),
            MCMMWalker(_line_conformer(10.0), 0.0, kt=1.5, memory=memory),  # mismatch
        ],
        [
            MCMMWalker(_line_conformer(50.0), 0.0, kt=2.0, memory=memory),
            MCMMWalker(_line_conformer(60.0), 0.0, kt=2.0, memory=memory),
        ],
    ]
    with pytest.raises(ValueError, match="inconsistent kt"):
        ReplicaExchangeMCMMDriver(groups, lambda c: [])


def test_remd_constructor_rejects_non_monotonic_temperatures():
    memory = BasinMemory(n_atoms=_TEST_N_ATOMS, rmsd_threshold=0.5)
    groups = [
        [MCMMWalker(_line_conformer(0.0), 0.0, kt=2.0, memory=memory)],
        [MCMMWalker(_line_conformer(50.0), 0.0, kt=1.0, memory=memory)],  # decreasing
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        ReplicaExchangeMCMMDriver(groups, lambda c: [])


def test_remd_constructor_rejects_invalid_swap_interval():
    groups, _ = _make_remd_walkers([1.0, 2.0])
    with pytest.raises(ValueError, match="swap_interval must be >= 1"):
        ReplicaExchangeMCMMDriver(groups, lambda c: [], swap_interval=0)


# ---------------------------------------------------------------------------
# ReplicaExchangeMCMMDriver — swap acceptance formula
# ---------------------------------------------------------------------------


def test_remd_swap_probability_matches_exp_delta_beta_delta_e():
    """`_swap_acceptance_prob` returns `exp((β_high − β_low)(E_high − E_low))`
    when the argument is negative (the conditional regime)."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(groups, lambda c: [])
    w_low, w_high = groups[0][0], groups[1][0]

    w_low.energy = 0.0
    w_high.energy = 5.0
    # delta_beta = 1/2 - 1/1 = -0.5; delta_e = 5; arg = -2.5
    p = driver._swap_acceptance_prob(w_low, w_high, kt_low=1.0, kt_high=2.0)
    assert p == pytest.approx(math.exp(-2.5), rel=1e-9)


def test_remd_swap_probability_one_when_high_temp_lower_energy():
    """When the high-T configuration has lower energy than the low-T one,
    the swap is always favorable (the low-energy state belongs at low T)."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(groups, lambda c: [])
    w_low, w_high = groups[0][0], groups[1][0]

    w_low.energy = 5.0
    w_high.energy = 0.0  # high-T has lower energy
    p = driver._swap_acceptance_prob(w_low, w_high, kt_low=1.0, kt_high=2.0)
    assert p == 1.0


def test_remd_swap_probability_one_when_energies_equal():
    """ΔE = 0 → arg = 0 → exp(0) = 1, and the `arg >= 0` branch returns 1."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(groups, lambda c: [])
    w_low, w_high = groups[0][0], groups[1][0]
    w_low.energy = 3.0
    w_high.energy = 3.0
    p = driver._swap_acceptance_prob(w_low, w_high, kt_low=1.0, kt_high=2.0)
    assert p == 1.0


# ---------------------------------------------------------------------------
# ReplicaExchangeMCMMDriver — swap mechanics
# ---------------------------------------------------------------------------


def test_remd_attempt_swaps_exchanges_configs_at_p_one():
    """When p_swap = 1 (favorable swap, e.g. high-T has lower energy),
    `attempt_swaps` always exchanges configurations between the paired
    slots."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(
        groups, lambda c: [], swap_random_fn=lambda: 0.999
    )
    w_low, w_high = groups[0][0], groups[1][0]
    # Force favorable swap: high-T energy < low-T energy.
    w_low.energy = 5.0
    w_high.energy = 0.0
    coords_low_before = w_low.coords.clone()
    coords_high_before = w_high.coords.clone()

    results = driver.attempt_swaps()
    assert results == [True]
    assert w_low.coords.equal(coords_high_before)
    assert w_high.coords.equal(coords_low_before)
    assert driver.n_swap_attempts == 1
    assert driver.n_swap_accepted == 1


def test_remd_attempt_swaps_skips_when_random_above_threshold():
    """When the swap probability is < 1 (typical regime) and random_fn ≥ p,
    no swap occurs."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(
        groups, lambda c: [], swap_random_fn=lambda: 0.999
    )
    w_low, w_high = groups[0][0], groups[1][0]
    # Unfavorable swap: high-T has higher energy. p_swap = exp(-2.5) ≈ 0.082.
    w_low.energy = 0.0
    w_high.energy = 5.0
    coords_low_before = w_low.coords.clone()
    coords_high_before = w_high.coords.clone()

    results = driver.attempt_swaps()
    assert results == [False]
    assert w_low.coords.equal(coords_low_before)
    assert w_high.coords.equal(coords_high_before)
    assert driver.n_swap_attempts == 1
    assert driver.n_swap_accepted == 0


def test_remd_attempt_swaps_iterates_all_adjacent_pairs():
    """With T temperatures and M walkers per temp, attempt_swaps performs
    (T - 1) × M swap attempts, in scan order (low-temp first, then by
    within-temp index)."""
    groups, _ = _make_remd_walkers([1.0, 2.0, 3.0], n_per_temp=2)
    driver = ReplicaExchangeMCMMDriver(
        groups, lambda c: [], swap_random_fn=lambda: 0.999
    )
    results = driver.attempt_swaps()
    assert len(results) == (3 - 1) * 2
    assert driver.n_swap_attempts == 4


# ---------------------------------------------------------------------------
# ReplicaExchangeMCMMDriver — step() and swap_interval semantics
# ---------------------------------------------------------------------------


def _identity_batch(coords_list):
    """Mock batch_propose_fn that proposes each walker's current coords
    (no movement) with a small downhill bump so the move is always
    accepted under finite kt. Useful for testing swap timing without
    having walkers diverge to new basins."""
    return [(c.clone(), 0.0, 1.0, True) for c in coords_list]


def test_remd_step_returns_per_walker_results_in_flat_order():
    """`step()` returns a list of accept/reject results in the order
    [temp_0_walker_0, temp_0_walker_1, ..., temp_T-1_walker_M-1]."""
    groups, _ = _make_remd_walkers([1.0, 2.0], n_per_temp=2)
    driver = ReplicaExchangeMCMMDriver(
        groups, _identity_batch, swap_interval=100, swap_random_fn=lambda: 0.999
    )
    results = driver.step()
    assert len(results) == 4  # 2 temps × 2 walkers


def test_remd_step_does_not_swap_below_interval():
    """`step()` only triggers swap attempts every `swap_interval` steps."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(
        groups,
        _identity_batch,
        swap_interval=5,
        swap_random_fn=lambda: 0.999,
    )
    # 4 steps: no swap attempts
    for _ in range(4):
        driver.step()
    assert driver.n_swap_attempts == 0
    # 5th step: triggers exactly one swap attempt (one adjacent pair × 1 walker per temp)
    driver.step()
    assert driver.n_swap_attempts == 1


def test_remd_step_swaps_at_each_interval_multiple():
    """Swap attempts trigger at each multiple of `swap_interval`."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(
        groups, _identity_batch, swap_interval=2, swap_random_fn=lambda: 0.999
    )
    for _ in range(6):
        driver.step()
    # Steps 2, 4, 6 each trigger one attempt → 3 total
    assert driver.n_swap_attempts == 3


# ---------------------------------------------------------------------------
# ReplicaExchangeMCMMDriver — properties and statistics
# ---------------------------------------------------------------------------


def test_remd_walker_count_properties():
    groups, _ = _make_remd_walkers([0.5, 1.0, 2.0, 4.0], n_per_temp=3)
    driver = ReplicaExchangeMCMMDriver(groups, lambda c: [])
    assert driver.n_temperatures == 4
    assert driver.n_walkers_per_temp == 3
    assert driver.n_walkers == 12


def test_remd_swap_acceptance_rate_property():
    """`swap_acceptance_rate` = n_swap_accepted / n_swap_attempts."""
    groups, _ = _make_remd_walkers([1.0, 2.0])
    driver = ReplicaExchangeMCMMDriver(
        groups, lambda c: [], swap_random_fn=lambda: 0.999
    )
    assert driver.swap_acceptance_rate == 0.0
    # Force one favorable swap
    groups[0][0].energy = 5.0
    groups[1][0].energy = 0.0
    driver.attempt_swaps()
    assert driver.swap_acceptance_rate == 1.0
    # Now an unfavorable one — rejected with random=0.999
    groups[0][0].energy = 0.0
    groups[1][0].energy = 5.0
    driver.attempt_swaps()
    assert driver.swap_acceptance_rate == 0.5  # 1 / 2


def test_remd_run_accumulates_accepts_across_walkers_and_steps():
    """`run(N)` returns the total accepts across walkers and steps for
    this call."""
    groups, _ = _make_remd_walkers([1.0, 2.0], n_per_temp=2)
    driver = ReplicaExchangeMCMMDriver(
        groups, _identity_batch, swap_interval=100, swap_random_fn=lambda: 0.999
    )
    # _identity_batch always proposes the walker's current coords (no move),
    # so apply_proposal sees ΔE = 0 - current_e. With initial e=0, ΔE=0.
    # bias = 1/√1 = 1 (each walker is at its own initial basin), det_j = 1,
    # so p_accept = min(1, 1*1*1) = 1. random_fn = lambda: 0.5 < 1.
    n_accepted = driver.run(3)
    assert n_accepted == 4 * 3  # 4 walkers × 3 steps


# ---------------------------------------------------------------------------
# make_mcmm_proposer (v0 stub behaviour)
# ---------------------------------------------------------------------------


def test_make_mcmm_proposer_returns_callable_for_cyclic_peptide():
    """The factory accepts a real cyclic peptide mol and returns a
    `batch_propose_fn(coords_list) → list` callable."""
    mol = _cycloala_mol(4)
    proposer = make_mcmm_proposer(
        mol, hardware_opts=None, calc=None, drive_sigma_rad=0.1, seed=0
    )
    assert callable(proposer)


def test_make_mcmm_proposer_rejects_for_non_cyclic_input():
    """The factory raises `ValueError` for a mol with no enumerable
    backbone windows (caught at build time, before any moves are
    proposed)."""
    cyclohexane = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    with pytest.raises(ValueError, match="no enumerable backbone windows"):
        make_mcmm_proposer(cyclohexane, hardware_opts=None, calc=None, seed=0)


def test_make_mcmm_proposer_invalid_mmff_backend_raises():
    """An unrecognised mmff_backend value raises at build time."""
    mol = _cycloala_mol(4)
    with pytest.raises(ValueError, match="unknown mmff_backend"):
        make_mcmm_proposer(
            mol, hardware_opts=None, calc=None, mmff_backend="other", seed=0
        )


def _make_real_proposer_with_mocks(
    mol, drive_sigma_rad: float = 0.05, closure_tol: float = 0.01, seed: int = 0
):
    """Build a real `make_mcmm_proposer` callable while patching the GPU
    stages — `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` as
    a no-op and `confsweeper._mace_batch_energies` as a sequential
    monotone mock. Returns an `(proposer, mock_mace, mock_mmff)` tuple
    plus a context manager that activates the patches; the caller
    enters the context before calling the proposer."""
    from contextlib import contextmanager
    from unittest.mock import patch as _patch

    proposer = make_mcmm_proposer(
        mol,
        hardware_opts=None,
        calc=None,
        drive_sigma_rad=drive_sigma_rad,
        closure_tol=closure_tol,
        seed=seed,
    )

    counter = [0]

    def _mock_mace(_calc, ase_mols):
        out = [(counter[0] + i) * 0.01 for i in range(len(ase_mols))]
        counter[0] += len(ase_mols)
        return out

    @contextmanager
    def patched():
        with (
            _patch("confsweeper._mace_batch_energies", side_effect=_mock_mace),
            _patch(
                "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
                return_value=[[]],
            ),
        ):
            yield

    return proposer, patched


def _seed_full_mol_coords(mol) -> torch.Tensor:
    """ETKDG-embed one conformer on `mol` and return its coords as a
    `(n_atoms, 3)` float64 torch tensor. Used as a starting state for
    walkers when calling the real proposer."""
    from rdkit.Chem import AllChem

    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    AllChem.EmbedMolecule(mol_copy, randomSeed=42)
    return torch.tensor(mol_copy.GetConformer(0).GetPositions(), dtype=torch.float64)


def test_make_mcmm_proposer_returns_proposal_per_walker():
    """The proposer respects the input list length: N input coords → N
    output proposals, each a 4-tuple `(coords, energy, det_j, success)`."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(mol, seed=42)
    seed_coords = _seed_full_mol_coords(mol)
    for n in [1, 3, 8]:
        coords_list = [seed_coords.clone() for _ in range(n)]
        with patched():
            proposals = proposer(coords_list)
        assert len(proposals) == n
        for prop in proposals:
            assert len(prop) == 4
            new_coords, new_energy, det_j, success = prop
            assert isinstance(new_coords, torch.Tensor)
            assert new_coords.shape == seed_coords.shape


def test_make_mcmm_proposer_can_accept_a_move():
    """For a small cyclic peptide with a small drive_sigma, at least one
    walker out of a batch should produce `success=True` — i.e. the
    closure solver finds a feasible move and the full-mol pipeline
    flows through MMFF/MACE without error."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(
        mol, drive_sigma_rad=0.05, seed=0
    )
    seed_coords = _seed_full_mol_coords(mol)
    coords_list = [seed_coords.clone() for _ in range(8)]
    with patched():
        proposals = proposer(coords_list)
    assert any(
        p[3] for p in proposals
    ), "Expected at least one closure success across 8 walkers; got 0"


def test_make_mcmm_proposer_successful_proposal_has_finite_energy_and_jacobian():
    """Successful proposals carry a finite energy from the MACE mock and
    a non-negative Wu-Deem |det J| from the closure solver."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(mol, seed=1)
    seed_coords = _seed_full_mol_coords(mol)
    coords_list = [seed_coords.clone() for _ in range(8)]
    with patched():
        proposals = proposer(coords_list)
    for new_coords, new_energy, det_j, success in proposals:
        if not success:
            continue
        assert np.isfinite(new_energy)
        assert det_j >= 0.0


def test_make_mcmm_proposer_failed_proposal_passes_through_input_coords():
    """Failed proposals (closure tolerance not met) pass through the
    walker's original coords with success=False, so the driver's
    `apply_proposal` rejects without further work."""
    mol = _cycloala_mol(4)
    # Tight closure_tol forces failure on most moves
    proposer, patched = _make_real_proposer_with_mocks(
        mol, drive_sigma_rad=0.5, closure_tol=1e-30, seed=99
    )
    seed_coords = _seed_full_mol_coords(mol)
    coords_list = [seed_coords.clone() for _ in range(4)]
    with patched():
        proposals = proposer(coords_list)
    # All should fail with this absurdly-tight tolerance
    for i, (new_coords, new_energy, det_j, success) in enumerate(proposals):
        assert not success
        assert det_j == 0.0
        assert torch.allclose(new_coords, coords_list[i])


def test_make_mcmm_proposer_does_not_mutate_input_coords():
    """The proposer treats input coords as immutable — even on accepted
    proposals, the input tensor is not modified in place."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(mol, seed=2)
    seed_coords = _seed_full_mol_coords(mol)
    coords_before = seed_coords.clone()
    coords_list = [seed_coords]
    with patched():
        proposer(coords_list)
    # The original tensor's contents are unchanged
    assert torch.allclose(seed_coords, coords_before)


# ---------------------------------------------------------------------------
# Side-chain helpers — _backbone_atom_set, _side_chain_group, _compute_window_downstream_sets
# ---------------------------------------------------------------------------


def test_backbone_atom_set_size_for_cyclic_peptide():
    """A K-residue cyclic peptide has 3K backbone atoms."""
    mol4 = _cycloala_mol(4)
    mol6 = _cycloala_mol(6)
    assert len(_backbone_atom_set(mol4)) == 12
    assert len(_backbone_atom_set(mol6)) == 18


def test_backbone_atom_set_excludes_hydrogens_and_side_chains():
    """Atoms in the backbone-atom set are all heavy atoms (N, Cα, C);
    Hs and side-chain methyls (Cβ for Ala) are not."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    for idx in backbone:
        atom = mol.GetAtomWithIdx(idx)
        assert atom.GetAtomicNum() in (
            6,
            7,
        ), f"backbone atom {idx} is element {atom.GetAtomicNum()}, expected C or N"


def test_side_chain_group_for_alanine_ca():
    """For an Ala Cα atom, the side-chain group is the Cβ methyl carbon
    plus its three Hs plus the Hα. With explicit Hs on cyclo(Ala)4
    the Cα has 4 non-backbone neighbours (= side chain), and the
    methyl Cβ has 3 H neighbours that join via BFS."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    residues = _ordered_backbone_residues(mol)
    # Ala: Cα → Cβ + Hα + 3 methyl Hs = 5 side-chain atoms total
    for n_idx, ca_idx, c_idx in residues:
        sc = _side_chain_group(mol, ca_idx, backbone)
        assert (
            len(sc) == 5
        ), f"Ala Cα {ca_idx} side chain has {len(sc)} atoms, expected 5"
        # Side-chain atoms are all non-backbone
        assert sc.isdisjoint(backbone)


def test_side_chain_group_for_amide_n_is_just_h():
    """For an Ala backbone amide N (non-NMe), the side chain is the
    single NH hydrogen."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    residues = _ordered_backbone_residues(mol)
    for n_idx, _, _ in residues:
        sc = _side_chain_group(mol, n_idx, backbone)
        assert len(sc) == 1
        h_idx = next(iter(sc))
        assert mol.GetAtomWithIdx(h_idx).GetAtomicNum() == 1


def test_side_chain_group_for_carbonyl_c_is_just_o():
    """For an amide C, the side chain is the single carbonyl oxygen."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    residues = _ordered_backbone_residues(mol)
    for _, _, c_idx in residues:
        sc = _side_chain_group(mol, c_idx, backbone)
        assert len(sc) == 1
        o_idx = next(iter(sc))
        assert mol.GetAtomWithIdx(o_idx).GetAtomicNum() == 8


def test_side_chain_group_does_not_cross_macrocycle():
    """Starting from one backbone atom, the BFS doesn't reach atoms
    attached to other backbone atoms — the macrocycle cycle is
    blocked at every other backbone atom."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    residues = _ordered_backbone_residues(mol)
    # Each residue's Cα has its own 5-atom side chain, all disjoint
    # from other residues' side chains.
    side_chains = [
        _side_chain_group(mol, ca_idx, backbone) for _, ca_idx, _ in residues
    ]
    union = set().union(*side_chains)
    # Sum of individual sizes equals the union size → all disjoint.
    assert sum(len(sc) for sc in side_chains) == len(union)


def test_compute_window_downstream_sets_length_and_disjoint_from_upstream():
    """Each of the 4 downstream sets should be a subset of the full mol's
    atoms and should NOT contain any of the upstream window backbone
    atoms (window[0..k+1])."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    windows = enumerate_backbone_windows(mol)
    window = windows[0]
    sets = _compute_window_downstream_sets(mol, window, backbone)
    assert len(sets) == 4
    for k, ds in enumerate(sets):
        upstream_window_atoms = set(window[: k + 2])  # window[0..k+1]
        assert ds.isdisjoint(
            upstream_window_atoms
        ), f"dihedral {k}: downstream set leaks into upstream window atoms"


def test_compute_window_downstream_sets_pivot_not_included():
    """The pivot atom window[k+2] sits ON the rotation axis and does
    not move — it must NOT be in downstream_sets[k]. (Its side-chain
    atoms DO appear, however, since they rotate with the local frame
    at the pivot.)"""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    windows = enumerate_backbone_windows(mol)
    sets = _compute_window_downstream_sets(mol, windows[0], backbone)
    for k, ds in enumerate(sets):
        pivot = windows[0][k + 2]
        assert pivot not in ds


def test_compute_window_downstream_sets_includes_pivot_side_chain():
    """The side-chain group of the pivot atom window[k+2] (e.g. Cα's
    Cβ + Hα + methyl Hs) IS included in the downstream set, because
    it rotates rigidly with the local frame at the pivot when the
    dihedral around the bond ending at the pivot changes."""
    mol = _cycloala_mol(4)
    backbone = _backbone_atom_set(mol)
    windows = enumerate_backbone_windows(mol)
    window = windows[0]
    sets = _compute_window_downstream_sets(mol, window, backbone)
    for k, ds in enumerate(sets):
        pivot_sc = _side_chain_group(mol, window[k + 2], backbone)
        assert pivot_sc.issubset(
            ds
        ), f"dihedral {k}: pivot's side chain not in downstream set"


def test_compute_window_downstream_sets_monotone_in_dihedral_index():
    """Downstream-set sizes shrink as k increases: dihedral 0 affects
    the most atoms (pivot side chain + 4 downstream backbone atoms +
    their side chains); dihedral 3 affects the fewest (pivot side
    chain + just window[6] + its side chain)."""
    mol = _cycloala_mol(6)  # use a longer ring so all 4 are non-trivial
    backbone = _backbone_atom_set(mol)
    windows = enumerate_backbone_windows(mol)
    sets = _compute_window_downstream_sets(mol, windows[0], backbone)
    sizes = [len(s) for s in sets]
    for k in range(3):
        assert (
            sizes[k] >= sizes[k + 1]
        ), f"size[{k}]={sizes[k]} should be >= size[{k+1}]={sizes[k+1]}"


# ---------------------------------------------------------------------------
# Regression: real cyclic peptide that triggered the original close-walk bug
# ---------------------------------------------------------------------------

# Reported in an end-to-end run of scripts/sampler_benchmark.py: this is a
# head-to-tail cyclic peptide containing several NMe residues, one Pro,
# and one Phe — a real entry from the PAMPA benchmark set. Under the
# original strict single-start C→N walk it raised "Backbone ring did not
# close: walked 8 of 8 residues" because the SMARTS produced extra
# matches outside the main ring (or the start residue was on a non-cycle
# spur). The longest-cycle-wins algorithm in `_ordered_backbone_residues`
# recovers the macrocycle correctly.
_PROBLEMATIC_REAL_PEPTIDE = (
    "CC(C)C[C@@H]1C(=O)N(C)[C@@H](CC(C)C)C(=O)N2CCC[C@@H]2C(=O)N(C)"
    "[C@@H](CC(C)C)C(=O)N[C@@H](Cc2ccccc2)C(=O)N(C)[C@@H](C)C(=O)N1C"
)


def test_ordered_residues_handles_problematic_real_peptide():
    """Real peptide that originally crashed `_ordered_backbone_residues`
    under the strict single-start walk. The robust longest-cycle-wins
    algorithm must recover a non-empty cyclic ordering with proper
    peptide-bond connectivity."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_PROBLEMATIC_REAL_PEPTIDE))
    residues = _ordered_backbone_residues(mol)
    # The macrocycle has at least 6 backbone residues; the algorithm
    # should recover all of them (extra SMARTS matches that aren't on
    # the main ring are silently dropped via the longest-cycle rule).
    assert len(residues) >= 6
    # Each (N, Cα, C) tuple has 3 distinct atom indices
    for n_idx, ca_idx, c_idx in residues:
        assert len({n_idx, ca_idx, c_idx}) == 3
    # Peptide-bond connectivity: residue i's C bonds to residue (i+1)'s N,
    # including the wrap-around from last to first.
    n_res = len(residues)
    for i in range(n_res):
        _, _, c = residues[i]
        n_next, _, _ = residues[(i + 1) % n_res]
        assert (
            mol.GetBondBetweenAtoms(c, n_next) is not None
        ), f"peptide bond missing between recovered residue {i} and {(i+1) % n_res}"


def test_enumerate_windows_for_problematic_real_peptide():
    """End-to-end: `enumerate_backbone_windows` returns 7-atom windows for
    every backbone atom of the recovered cycle on the same peptide that
    originally crashed."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_PROBLEMATIC_REAL_PEPTIDE))
    windows = enumerate_backbone_windows(mol)
    # 3K windows for K-residue cycle; K >= 6 → at least 18 windows
    assert len(windows) >= 18
    for window in windows:
        assert len(window) == WINDOW_SIZE
        assert len(set(window)) == WINDOW_SIZE


# ---------------------------------------------------------------------------
# make_mcmm_proposer — diagnostic stats counter
# ---------------------------------------------------------------------------


def test_make_mcmm_proposer_stats_initialized_to_zero():
    """The proposer factory attaches a `.stats` dict initialised to zeros
    (no calls yet means no proposals counted)."""
    mol = _cycloala_mol(4)
    proposer = make_mcmm_proposer(mol, hardware_opts=None, calc=None, seed=0)
    assert hasattr(proposer, "stats")
    assert proposer.stats == {
        "n_proposed": 0,
        "n_closure_failures": 0,
        "n_closure_successes": 0,
    }


def test_make_mcmm_proposer_stats_increment_per_call():
    """After a batched call, `n_proposed` equals the number of walkers and
    `n_closure_successes + n_closure_failures` matches `n_proposed`."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(mol, seed=42)
    seed_coords = _seed_full_mol_coords(mol)

    # First call: 5 walkers
    coords_list = [seed_coords.clone() for _ in range(5)]
    with patched():
        proposer(coords_list)
    assert proposer.stats["n_proposed"] == 5
    closures_so_far = (
        proposer.stats["n_closure_successes"] + proposer.stats["n_closure_failures"]
    )
    assert closures_so_far == 5

    # Second call: 3 walkers — counters accumulate.
    coords_list = [seed_coords.clone() for _ in range(3)]
    with patched():
        proposer(coords_list)
    assert proposer.stats["n_proposed"] == 8
    closures_so_far = (
        proposer.stats["n_closure_successes"] + proposer.stats["n_closure_failures"]
    )
    assert closures_so_far == 8


def test_make_mcmm_proposer_stats_match_returned_success_flags():
    """Cumulative closure-success count equals the number of returned
    proposals with success=True. Locks the bookkeeping against the
    per-walker proposal contract."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(mol, seed=7)
    seed_coords = _seed_full_mol_coords(mol)
    coords_list = [seed_coords.clone() for _ in range(8)]
    with patched():
        proposals = proposer(coords_list)
    n_success_returned = sum(1 for p in proposals if p[3])
    assert proposer.stats["n_closure_successes"] == n_success_returned
    assert (
        proposer.stats["n_closure_failures"]
        == proposer.stats["n_proposed"] - n_success_returned
    )


def test_make_mcmm_proposer_stats_record_failure_when_tol_unreachable():
    """With an absurdly tight closure_tol, every proposal fails. Stats
    reflect this: n_closure_failures = n_proposed, n_closure_successes = 0."""
    mol = _cycloala_mol(4)
    proposer, patched = _make_real_proposer_with_mocks(
        mol, drive_sigma_rad=0.5, closure_tol=1e-30, seed=99
    )
    seed_coords = _seed_full_mol_coords(mol)
    coords_list = [seed_coords.clone() for _ in range(4)]
    with patched():
        proposals = proposer(coords_list)
    assert all(not p[3] for p in proposals)
    assert proposer.stats["n_closure_successes"] == 0
    assert proposer.stats["n_closure_failures"] == 4
    assert proposer.stats["n_proposed"] == 4


# ---------------------------------------------------------------------------
# Regression: depsipeptide (peptide with ester linker in the macrocycle ring)
# ---------------------------------------------------------------------------

# Real PAMPA peptide that originally crashed `_ordered_backbone_residues`
# because the strict amide-only SMARTS missed the residues adjacent to
# the ester `OC(=O)` linker. The relaxed `_MCMM_BACKBONE_SMARTS` matches
# both N and O at the linker positions, so the macrocycle closes
# correctly.
_FAILING_DEPSIPEPTIDE = (
    "[H]N1C(=O)[C@]([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])"
    "N([H])C(=O)[C@]([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])"
    "N([H])C(=O)[C@]([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])"
    "N([H])C(=O)[C@@]([H])(N([H])C(=O)[C@@]([H])(N(C(=O)[C@]2([H])"
    "N(C(=O)C([H])([H])[H])C([H])([H])C([H])([H])C2([H])[H])C([H])([H])[H])"
    "C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])"
    "[C@@]([H])(C([H])([H])[H])OC(=O)[C@]2([H])"
    "N(C(=O)[C@]([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])"
    "N([H])C(=O)[C@]1([H])C([H])([H])[H])C([H])([H])C([H])([H])C2([H])[H]"
)


def test_enumerate_windows_for_depsipeptide():
    """Real PAMPA peptide with an `OC(=O)` ester linker AND a Cα-Cα bond
    closing the macrocycle (canonical SMILES shows ring 1 closing as
    `[C@@H]1...[C@@H]1`). Both features fall outside the strict
    amide-residue SMARTS pattern. The ring-info-based
    `enumerate_backbone_windows` walks the macrocycle directly and
    handles arbitrary cyclic-backbone topologies."""
    mol = Chem.MolFromSmiles(_FAILING_DEPSIPEPTIDE)
    assert mol is not None, "test SMILES failed to parse"
    windows = enumerate_backbone_windows(mol)
    # The macrocycle is the largest ring; for this PAMPA peptide it is
    # well above the 7-atom minimum.
    assert len(windows) >= 7
    # Each window: 7 distinct atoms, sequentially bonded around the ring.
    for window in windows:
        assert len(window) == WINDOW_SIZE
        assert len(set(window)) == WINDOW_SIZE
        for i in range(WINDOW_SIZE - 1):
            assert mol.GetBondBetweenAtoms(window[i], window[i + 1]) is not None


def test_enumerate_windows_handles_macrocycle_with_ester_linker():
    """The macrocycle of the failing depsipeptide includes at least one
    oxygen ring atom (the ester O linker). Ring-info enumeration treats
    it just like any other ring atom, so it can appear at any position
    in a 7-atom window."""
    mol = Chem.MolFromSmiles(_FAILING_DEPSIPEPTIDE)
    ring_atom_set = set()
    for window in enumerate_backbone_windows(mol):
        ring_atom_set.update(window)
    ring_atomic_nums = {mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring_atom_set}
    # Macrocycle backbone contains C and N (peptide), plus O (ester).
    assert 8 in ring_atomic_nums, (
        "expected at least one O atom in the macrocycle ring "
        f"(ester linker); got element atomic numbers {ring_atomic_nums}"
    )
