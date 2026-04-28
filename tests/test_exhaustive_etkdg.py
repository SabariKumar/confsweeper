"""Unit tests for the exhaustive ETKDG sampling primitives in src/confsweeper.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from rdkit.Chem import AllChem

from confsweeper import (
    _KT_EV_298K,
    _energy_ranked_dedup,
    _jitter_rotatable_dihedrals,
    get_embed_params,
    get_mol_PE_exhaustive,
)


def _line_coords(positions: list[float], n_atoms: int = 4) -> torch.Tensor:
    """
    Build [N, n_atoms, 3] coords where conformer i is a rigid translation of a
    fixed atom layout by `positions[i]` along x. Pairwise normalised L1 distance
    between conformer i and j equals |positions[i] - positions[j]|.

    Params:
        positions: list[float] : x-translation per conformer
        n_atoms: int : number of atoms (constant across conformers)
    Returns:
        torch.Tensor [N, n_atoms, 3] of conformer coordinates
    """
    base = torch.zeros(n_atoms, 3)
    base[:, 0] = torch.arange(n_atoms, dtype=torch.float32)
    coords = torch.stack([base + torch.tensor([p, 0.0, 0.0]) for p in positions])
    return coords


# ---------------------------------------------------------------------------
# _energy_ranked_dedup
# ---------------------------------------------------------------------------


def test_energy_ranked_dedup_empty():
    coords = torch.zeros(0, 4, 3)
    energies = np.zeros(0)
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1) == []


def test_energy_ranked_dedup_single():
    coords = torch.zeros(1, 4, 3)
    energies = np.array([0.0])
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1) == [0]


def test_energy_ranked_dedup_keeps_all_distinct():
    """Three conformers spaced far apart in geometry should all survive dedup."""
    coords = _line_coords([0.0, 5.0, 10.0])
    energies = np.array([0.0, 1.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert sorted(centroids) == [0, 1, 2]


def test_energy_ranked_dedup_collapses_all_close():
    """Three conformers all within the RMSD threshold collapse to the lowest-energy."""
    # All translations are 0.0, so pairwise normalised L1 distance = 0
    coords = _line_coords([0.0, 0.0, 0.0])
    energies = np.array([2.0, 0.5, 1.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [1]  # index of the lowest energy


def test_energy_ranked_dedup_picks_lowest_energy_in_basin():
    """The lowest-energy conformer of a basin must be the one returned, even
    when a higher-energy conformer is geometrically nearer the centroid of
    other already-selected representatives."""
    # conformers 0 and 1 are geometric near-duplicates; conformer 0 has the
    # higher energy, so dedup must drop conformer 0 and keep conformer 1.
    coords = _line_coords([0.0, 0.05, 5.0])
    energies = np.array([1.0, 0.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    # Lowest-energy basin rep (conformer 1) and the geometrically distant
    # conformer 2 survive; the higher-energy near-duplicate (conformer 0) drops.
    assert sorted(centroids) == [1, 2]


def test_energy_ranked_dedup_centroid_order_is_energy_ascending():
    """Returned centroid order must match ascending energy."""
    coords = _line_coords([0.0, 5.0, 10.0])
    energies = np.array([2.0, 0.5, 1.5])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [1, 2, 0]


def test_energy_ranked_dedup_stable_on_energy_ties():
    """When two conformers tie on energy, np.argsort(kind='stable') breaks the
    tie by original index. The lower-index conformer is selected first and
    excludes its near-duplicate twin."""
    # conformers 0 and 1 are near-duplicates with identical energy
    coords = _line_coords([0.0, 0.05, 5.0])
    energies = np.array([1.0, 1.0, 2.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    assert centroids == [0, 2]


def test_energy_ranked_dedup_threshold_boundary():
    """A pair right at the threshold is treated as overlapping (strict < not <=)
    only when distance is *less than* the threshold. Distance exactly at the
    threshold is considered distinct."""
    # On _line_coords, normalised L1 distance between conformer 0 (offset 0)
    # and conformer 1 (offset 0.4) over n_atoms=4 atoms is
    # (4 * 0.4) / (3 * 4) = 0.1333...
    coords = _line_coords([0.0, 0.4])
    energies = np.array([0.0, 1.0])
    # Threshold 0.2 > 0.133, so the second is excluded
    assert _energy_ranked_dedup(coords, energies, rmsd_threshold=0.2) == [0]
    # Threshold 0.1 < 0.133, so both survive
    assert sorted(_energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)) == [0, 1]


@pytest.mark.parametrize("device", ["cpu"])
def test_energy_ranked_dedup_device_independent(device):
    """Algorithm output is identical regardless of which device coords live on.
    GPU coverage is exercised in integration tests; this parametrisation makes
    it easy to add 'cuda' once the helper is wired into the full pipeline."""
    coords = _line_coords([0.0, 5.0, 0.05]).to(device)
    energies = np.array([1.0, 2.0, 0.0])
    centroids = _energy_ranked_dedup(coords, energies, rmsd_threshold=0.1)
    # Lowest energy is conformer 2 (offset 0.05, near conformer 0). It excludes
    # conformer 0; conformer 1 is far away and survives.
    assert sorted(centroids) == [1, 2]


# ---------------------------------------------------------------------------
# get_mol_PE_exhaustive
# ---------------------------------------------------------------------------

# Tripeptide cyclic-ish backbone — small enough that CPU ETKDG produces
# distinct conformers reliably, large enough that the geometric dedup actually
# discriminates basins.
TEST_SMILES = "C[C@@H](N)C(=O)NCC(=O)NCC(=O)O"


def _mock_embed_seed_aware(mols, params, confsPerMolecule, hardwareOptions):
    """
    CPU drop-in for nvmolkit.embedMolecules.EmbedMolecules.

    Matches two behaviours of the real call that the chunked-embed loop relies
    on: respects params.randomSeed (so chunked calls produce distinct
    conformers) and appends to any existing conformer set rather than
    replacing it (verified empirically against nvmolkit on a small peptide).
    """
    for m in mols:
        AllChem.EmbedMultipleConfs(
            m,
            numConfs=confsPerMolecule,
            randomSeed=params.randomSeed,
            clearConfs=False,
        )


def _make_seq_mock_mace():
    """
    Return a `_mace_batch_energies` mock that yields a global monotone-increasing
    energy sequence across calls. Conformer i (counted across all chunks) is
    assigned energy i * 0.01 eV — about 0.39 kT_298K per step. This makes the
    energy filter and dedup behaviour deterministic and easy to assert against.
    """
    counter = [0]

    def _mock(_calc, ase_mols):
        out = [(counter[0] + i) * 0.01 for i in range(len(ase_mols))]
        counter[0] += len(ase_mols)
        return out

    return _mock


@pytest.fixture
def exhaustive_mocks():
    """
    Patch nvmolkit embed, MACE scoring, and GPU MMFF with deterministic CPU
    mocks so integration tests run without a GPU and without depending on
    nvmolkit's real CUDA paths.

    GPU MMFF is patched as a no-op (it would otherwise be called from
    `get_mol_PE_exhaustive` whenever minimize=True, which is now the default).
    Tests that specifically exercise the MMFF dispatch path patch
    `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` themselves.
    """
    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_embed_seed_aware),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
    ):
        yield


def test_exhaustive_returns_centroids_and_energies(exhaustive_mocks):
    """Smoke: pipeline returns aligned centroids + energies in ascending order."""
    mol, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=20,
        embed_chunk_size=20,
        score_chunk_size=20,
        rmsd_threshold=0.0,  # disable geometric dedup so all energy-filtered survivors are returned
    )
    assert len(centroid_ids) > 0
    assert len(centroid_ids) == len(energies)
    # Energies are in ascending order (energy-ranked dedup guarantees this).
    assert energies == sorted(energies)
    # Mol contains exactly the centroid conformers — non-centroids are removed.
    assert mol.GetNumConformers() == len(centroid_ids)


def test_exhaustive_chunked_path_accumulates_full_pool(exhaustive_mocks):
    """When n_seeds > embed_chunk_size, the chunked loop accumulates the full
    requested pool size (modulo CPU embed failures, which our mock does not
    induce). This validates the chunked-embed mechanics, not nvmolkit's
    cross-call diversity (the semantics experiment in docs/ covers that)."""
    n_seeds = 24
    embed_chunk_size = 8
    _, centroid_ids, _ = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=n_seeds,
        embed_chunk_size=embed_chunk_size,
        score_chunk_size=64,
        rmsd_threshold=0.0,
        e_window_kT=1e9,  # disable energy filter so we can count the full pool
    )
    # With rmsd_threshold=0 and the energy filter disabled, every successfully-
    # embedded conformer is its own centroid.
    assert len(centroid_ids) == n_seeds


def test_exhaustive_energy_filter_drops_high_energy(exhaustive_mocks):
    """Conformers more than e_window_kT * kT above the minimum are dropped.

    With monotone energies E_i = i * 0.01 eV and kT = 0.02568 eV at 298 K,
    a window of e_window_kT=2.0 keeps E_i ≤ 2 * 0.02568 ≈ 0.0514 eV, i.e.
    the first 6 conformers (i = 0..5).
    """
    n_seeds = 20
    e_window_kT = 2.0
    expected_kept = int(np.floor(e_window_kT * _KT_EV_298K / 0.01)) + 1

    _, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=n_seeds,
        embed_chunk_size=n_seeds,
        score_chunk_size=n_seeds,
        e_window_kT=e_window_kT,
        rmsd_threshold=0.0,  # isolate the energy filter from dedup
    )
    assert len(centroid_ids) == expected_kept
    assert max(energies) - min(energies) <= e_window_kT * _KT_EV_298K + 1e-12


def test_exhaustive_zero_conformers_safe():
    """When embed returns 0 conformers, contract is (mol, [], []) — no crash,
    no MACE call, no dedup attempt."""

    def _embed_nothing(mols, params, confsPerMolecule, hardwareOptions):
        return  # don't add any conformers

    mock_mace = MagicMock()
    with patch("confsweeper.embed.EmbedMolecules", side_effect=_embed_nothing), patch(
        "confsweeper._mace_batch_energies", side_effect=mock_mace
    ):
        mol, centroid_ids, energies = get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=20,
            embed_chunk_size=20,
            score_chunk_size=20,
        )
    assert centroid_ids == []
    assert energies == []
    assert mol.GetNumConformers() == 0
    mock_mace.assert_not_called()


def test_exhaustive_single_conformer_skips_dedup(exhaustive_mocks):
    """n_seeds=1 returns one centroid without invoking the dedup primitive
    (which is unnecessary on a singleton pool)."""
    mol, centroid_ids, energies = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=1,
        embed_chunk_size=1,
        score_chunk_size=1,
    )
    assert len(centroid_ids) == 1
    assert len(energies) == 1
    assert mol.GetNumConformers() == 1


def test_exhaustive_minimize_cpu_invokes_per_conformer_mmff(exhaustive_mocks):
    """mmff_backend='cpu' triggers AllChem.MMFFOptimizeMolecule once per
    conformer in the pool (called before energy scoring)."""
    n_seeds = 6
    with patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", return_value=0) as mock_mmff:
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=n_seeds,
            embed_chunk_size=n_seeds,
            score_chunk_size=n_seeds,
            minimize=True,
            mmff_backend="cpu",
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    # One MMFF call per embedded conformer. Mock embed always succeeds.
    assert mock_mmff.call_count == n_seeds


def test_exhaustive_minimize_gpu_invokes_batched_mmff(exhaustive_mocks):
    """mmff_backend='gpu' calls nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs
    exactly once with the full conformer set."""
    n_seeds = 6
    with patch(
        "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs", return_value=[[]]
    ) as mock_mmff:
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=n_seeds,
            embed_chunk_size=n_seeds,
            score_chunk_size=n_seeds,
            minimize=True,
            mmff_backend="gpu",
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    # Single batched call regardless of n_seeds.
    mock_mmff.assert_called_once()
    args, kwargs = mock_mmff.call_args
    # First positional arg is the list of mols (length 1, our test mol).
    assert len(args[0]) == 1
    assert args[0][0].GetNumConformers() == n_seeds


def test_exhaustive_minimize_unknown_backend_raises(exhaustive_mocks):
    """An unrecognised mmff_backend value raises ValueError."""
    with pytest.raises(ValueError, match="unknown mmff_backend"):
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=2,
            embed_chunk_size=2,
            score_chunk_size=2,
            minimize=True,
            mmff_backend="something_else",
        )


def test_exhaustive_minimize_false_skips_both_backends(exhaustive_mocks):
    """minimize=False must not call MMFF on either backend, regardless of what
    mmff_backend is set to. The mmff_backend value is only consulted when
    minimize is True."""
    n_seeds = 6
    with (
        patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", return_value=0) as mock_cpu,
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ) as mock_gpu,
    ):
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=n_seeds,
            embed_chunk_size=n_seeds,
            score_chunk_size=n_seeds,
            minimize=False,
            mmff_backend="gpu",  # ignored when minimize=False
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    mock_cpu.assert_not_called()
    mock_gpu.assert_not_called()


def test_exhaustive_energy_filter_force_keeps_minimum(exhaustive_mocks):
    """If every conformer's energy comparison evaluates to False (e.g. NaN
    energies), the energy-filter guard force-retains a single conformer at
    `argmin(energies)` so the contract "at least one centroid" holds.

    This exercises the `if not keep_mask.any()` fallback in get_mol_PE_exhaustive.
    """
    n_seeds = 5

    def _all_nan_mace(_calc, ase_mols):
        # Every conformer scored as NaN. (energies - e_min) <= window then
        # evaluates to all-False, triggering the forced-keep-minimum guard.
        return [float("nan")] * len(ase_mols)

    with patch("confsweeper._mace_batch_energies", side_effect=_all_nan_mace):
        mol, centroid_ids, energies = get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=n_seeds,
            embed_chunk_size=n_seeds,
            score_chunk_size=n_seeds,
            minimize=False,
            rmsd_threshold=0.0,
            e_window_kT=5.0,
        )
    # Forced-keep-minimum guarantees exactly one centroid even though no
    # conformer passes the energy filter. The returned mol holds exactly that
    # conformer and energies has length one.
    assert len(centroid_ids) == 1
    assert len(energies) == 1
    assert mol.GetNumConformers() == 1


def test_exhaustive_returned_mol_matches_centroid_ids(exhaustive_mocks):
    """The returned mol contains exactly the conformers identified by
    centroid_ids — no leftover non-centroid conformers and no centroid
    pruned out. This is the cleanup contract for the final stage of the
    pipeline."""
    n_seeds = 12
    mol, centroid_ids, _ = get_mol_PE_exhaustive(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_seeds=n_seeds,
        embed_chunk_size=n_seeds,
        score_chunk_size=n_seeds,
        minimize=False,
        rmsd_threshold=0.0,
        e_window_kT=1e9,
    )
    mol_conf_ids = sorted(c.GetId() for c in mol.GetConformers())
    assert mol_conf_ids == sorted(centroid_ids)
    # Every centroid_id must resolve to a real conformer on the mol.
    for cid in centroid_ids:
        mol.GetConformer(cid)  # raises if missing


# ---------------------------------------------------------------------------
# _jitter_rotatable_dihedrals
# ---------------------------------------------------------------------------


def _embed_butane(n_confs: int = 3, seed: int = 0):
    """
    Build n-butane (CCCC) with explicit Hs and CPU-embed `n_confs` conformers.
    n-butane has exactly one rotatable dihedral (the central C-C bond), so the
    output of `_jitter_rotatable_dihedrals` is unambiguous.

    Params:
        n_confs: int : number of conformers to embed
        seed: int : RNG seed for ETKDG
    Returns:
        rdkit.Chem.Mol : n-butane mol with explicit Hs and `n_confs` conformers
    """
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed)
    return mol


def test_jitter_rotatable_dihedrals_returns_dihedral_count():
    """n-butane has exactly one rotatable backbone dihedral."""
    mol = _embed_butane(n_confs=2)
    n = _jitter_rotatable_dihedrals(mol, jitter_deg=10.0, seed=0)
    assert n == 1


def test_jitter_rotatable_dihedrals_changes_geometry():
    """A non-zero jitter actually moves the dihedral; magnitude is bounded by
    jitter_deg and direction is random per (conformer, dihedral)."""
    from rdkit.Chem import rdMolTransforms

    mol = _embed_butane(n_confs=4, seed=1)
    # Capture original dihedrals (atom indices for n-butane: 0-1-2-3 are the carbons).
    before = [
        rdMolTransforms.GetDihedralDeg(mol.GetConformer(c), 0, 1, 2, 3)
        for c in range(mol.GetNumConformers())
    ]
    _jitter_rotatable_dihedrals(mol, jitter_deg=15.0, seed=42)
    after = [
        rdMolTransforms.GetDihedralDeg(mol.GetConformer(c), 0, 1, 2, 3)
        for c in range(mol.GetNumConformers())
    ]
    deltas = [(a - b + 180) % 360 - 180 for a, b in zip(after, before)]
    # Every conformer moved (uniform sampling at jitter=15° draws zero with
    # probability ≪ 1e-9 across 4 conformers).
    assert all(abs(d) > 1e-3 for d in deltas)
    # Every move stays inside the [-jitter, +jitter] envelope (with tiny tolerance
    # for floating-point in the trig conversions).
    assert all(abs(d) <= 15.0 + 1e-6 for d in deltas)


def test_jitter_rotatable_dihedrals_seed_is_deterministic():
    """Two jitter calls on identical inputs with the same seed produce
    identical dihedrals; with different seeds they differ."""
    from rdkit.Chem import rdMolTransforms

    def _jitter_and_read(seed: int) -> list[float]:
        mol = _embed_butane(n_confs=3, seed=7)
        _jitter_rotatable_dihedrals(mol, jitter_deg=20.0, seed=seed)
        return [
            rdMolTransforms.GetDihedralDeg(mol.GetConformer(c), 0, 1, 2, 3)
            for c in range(mol.GetNumConformers())
        ]

    a1 = _jitter_and_read(seed=99)
    a2 = _jitter_and_read(seed=99)
    b = _jitter_and_read(seed=100)
    assert a1 == a2
    assert a1 != b


def test_jitter_rotatable_dihedrals_skips_ring_bonds():
    """The rotatable-bond SMARTS uses '!@' to exclude ring bonds. Cyclohexane
    therefore has zero rotatable bonds and the helper is a no-op."""
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=0)
    n = _jitter_rotatable_dihedrals(mol, jitter_deg=30.0, seed=0)
    assert n == 0


def test_get_mol_PE_exhaustive_jitter_zero_is_noop(exhaustive_mocks):
    """dihedral_jitter_deg=0 must not invoke the jitter helper. The default
    pipeline behaviour stays unchanged."""
    with patch(
        "confsweeper._jitter_rotatable_dihedrals", return_value=0
    ) as mock_jitter:
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=6,
            embed_chunk_size=6,
            score_chunk_size=6,
            dihedral_jitter_deg=0.0,
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    mock_jitter.assert_not_called()


def test_get_mol_PE_exhaustive_jitter_invokes_helper(exhaustive_mocks):
    """dihedral_jitter_deg>0 invokes the jitter helper exactly once with the
    matching jitter angle."""
    with patch(
        "confsweeper._jitter_rotatable_dihedrals", return_value=1
    ) as mock_jitter:
        get_mol_PE_exhaustive(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_seeds=6,
            embed_chunk_size=6,
            score_chunk_size=6,
            dihedral_jitter_deg=12.5,
            rmsd_threshold=0.0,
            e_window_kT=1e9,
        )
    mock_jitter.assert_called_once()
    _, kwargs = mock_jitter.call_args
    assert kwargs["jitter_deg"] == 12.5
