"""Unit tests for src/mcmm.py.

Step 3 of the issue-#11 implementation: backbone window enumeration.
Fixtures are cyclo(Ala)N peptides for varying N. Tests verify the
window count, that each window is 7 consecutively-bonded backbone atoms,
and that the C→N walk handles the non-cyclic and non-peptide cases.
"""

import pytest
from rdkit import Chem

from mcmm import WINDOW_SIZE, _ordered_backbone_residues, enumerate_backbone_windows

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
