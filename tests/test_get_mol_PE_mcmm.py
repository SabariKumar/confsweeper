"""Unit tests for get_mol_PE_mcmm in src/confsweeper.py.

Mirrors the structure of test_pool_b.py: mocks the GPU stages
(ETKDG embed, MMFF, MACE) and the proposer factory so the orchestration
of get_mol_PE_mcmm is verifiable without a GPU. The Step-8 stub
proposer never accepts moves, so the basin set ends at the initial
conformer; tests that need exploration inject a custom proposer mock
via `patch("confsweeper.make_mcmm_proposer")`.
"""

from unittest.mock import MagicMock, patch

import pytest
from rdkit.Chem import AllChem

from confsweeper import (
    _KT_EV_298K,
    _geometric_temperature_ladder,
    get_embed_params,
    get_mol_PE_mcmm,
)

# Cyclic tetrapeptide. The MCMM proposer enumerates backbone windows on
# this mol, so it must be a head-to-tail cyclic peptide. cyclo(Ala)4
# has 12 backbone atoms and 30 atoms total with explicit Hs.
TEST_SMILES = "C[C@@H]1NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@@H](C)NC1=O"


def _mock_etkdg_embed(mols, params, confsPerMolecule, hardwareOptions):
    """CPU drop-in for nvmolkit ETKDG: embed `confsPerMolecule` conformers
    via RDKit so tests don't need nvmolkit/CUDA."""
    for m in mols:
        AllChem.EmbedMultipleConfs(
            m,
            numConfs=confsPerMolecule,
            randomSeed=params.randomSeed,
            clearConfs=False,
        )


def _make_seq_mock_mace():
    """Mock for `_mace_batch_energies` that yields globally monotone-
    increasing energies (i * 0.01 eV ≈ 0.39 kT_298K per step). Same
    pattern as test_pool_b's mock — keeps energy filter / dedup
    behaviour deterministic."""
    counter = [0]

    def _mock(_calc, ase_mols):
        out = [(counter[0] + i) * 0.01 for i in range(len(ase_mols))]
        counter[0] += len(ase_mols)
        return out

    return _mock


def _stub_proposer_factory(mol, hardware_opts, calc, **kwargs):
    """Replacement for `make_mcmm_proposer` that always rejects every
    proposal. Identical to the v0 stub behaviour but isolated from the
    real factory's import-time topology probe."""
    del mol, hardware_opts, calc, kwargs

    def fn(coords_list):
        return [(c, 0.0, 0.0, False) for c in coords_list]

    return fn


@pytest.fixture
def mcmm_mocks():
    """Patch ETKDG embed, MACE scoring, GPU MMFF, and the proposer
    factory with CPU-only drop-ins. Tests that need a more interesting
    proposer (e.g. one that accepts moves) override
    `confsweeper.make_mcmm_proposer` themselves."""
    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch("confsweeper.make_mcmm_proposer", side_effect=_stub_proposer_factory),
    ):
        yield


# ---------------------------------------------------------------------------
# _geometric_temperature_ladder
# ---------------------------------------------------------------------------


def test_geometric_ladder_endpoints_match():
    """Ladder hits the endpoints exactly: kts[0] = kt_low, kts[-1] = kt_high."""
    kts = _geometric_temperature_ladder(0.025, 0.05, 8)
    assert kts[0] == pytest.approx(0.025)
    assert kts[-1] == pytest.approx(0.05)
    assert len(kts) == 8


def test_geometric_ladder_strictly_increasing():
    kts = _geometric_temperature_ladder(0.025, 0.05, 8)
    for i in range(len(kts) - 1):
        assert kts[i] < kts[i + 1]


def test_geometric_ladder_geometric_spacing():
    """Ratios between adjacent ladder rungs are constant."""
    kts = _geometric_temperature_ladder(0.025, 0.05, 8)
    expected_ratio = (0.05 / 0.025) ** (1.0 / 7)
    for i in range(len(kts) - 1):
        assert kts[i + 1] / kts[i] == pytest.approx(expected_ratio)


def test_geometric_ladder_length_one():
    """n=1 returns a single-element list at kt_low."""
    kts = _geometric_temperature_ladder(0.025, 0.05, 1)
    assert kts == [0.025]


def test_geometric_ladder_invalid_inputs_raise():
    with pytest.raises(ValueError, match="n must be positive"):
        _geometric_temperature_ladder(0.025, 0.05, 0)
    with pytest.raises(ValueError, match="0 < kt_low < kt_high"):
        _geometric_temperature_ladder(0.05, 0.025, 4)  # reversed
    with pytest.raises(ValueError, match="0 < kt_low < kt_high"):
        _geometric_temperature_ladder(0.0, 0.05, 4)  # zero
    with pytest.raises(ValueError, match="0 < kt_low < kt_high"):
        _geometric_temperature_ladder(0.025, 0.025, 4)  # equal


# ---------------------------------------------------------------------------
# get_mol_PE_mcmm — contract under the no-exploration stub
# ---------------------------------------------------------------------------


def test_mcmm_returns_centroids_and_energies(mcmm_mocks):
    """End-to-end smoke: pipeline returns aligned centroids + energies in
    ascending order, mol contains exactly the centroid conformers. Under
    the stub proposer no moves are accepted, so the basin set is just the
    initial conformer (rescored and deduped by the shared tail)."""
    mol, centroid_ids, energies = get_mol_PE_mcmm(
        TEST_SMILES,
        get_embed_params(),
        hardware_opts=None,
        calc=MagicMock(),
        n_walkers_per_temp=2,
        n_temperatures=2,
        n_steps=4,
        swap_interval=2,
        rmsd_threshold=0.0,
    )
    assert len(centroid_ids) > 0
    assert len(centroid_ids) == len(energies)
    assert energies == sorted(energies)
    assert mol.GetNumConformers() == len(centroid_ids)


def test_mcmm_zero_conformers_safe():
    """If ETKDG fails to embed any seed conformer, the contract is
    `(mol, [], [])` with no MMFF, MACE, or proposer calls."""

    def _embed_nothing(mols, params, confsPerMolecule, hardwareOptions):
        return  # no conformers added

    mock_mace = MagicMock()
    mock_mmff = MagicMock()
    mock_proposer_factory = MagicMock()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_embed_nothing),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            side_effect=mock_mmff,
        ),
        patch("confsweeper.make_mcmm_proposer", side_effect=mock_proposer_factory),
    ):
        mol, centroid_ids, energies = get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=2,
            n_temperatures=2,
            n_steps=2,
        )
    assert centroid_ids == []
    assert energies == []
    assert mol.GetNumConformers() == 0
    mock_mace.assert_not_called()
    mock_proposer_factory.assert_not_called()


def test_mcmm_proposer_called_per_step_per_temperature_pair(mcmm_mocks):
    """The replica-exchange driver calls the batch proposer once per
    `step()` (and N steps total). Verify the proposer factory is invoked
    exactly once at setup, and the returned callable is invoked n_steps
    times."""
    proposer_call_count = {"n": 0}

    def _counting_proposer_factory(mol, hardware_opts, calc, **kwargs):
        del mol, hardware_opts, calc, kwargs

        def fn(coords_list):
            proposer_call_count["n"] += 1
            return [(c, 0.0, 0.0, False) for c in coords_list]

        return fn

    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch(
            "confsweeper.make_mcmm_proposer",
            side_effect=_counting_proposer_factory,
        ),
    ):
        get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=2,
            n_temperatures=2,
            n_steps=5,
            swap_interval=100,  # disable swaps so step count is unambiguous
            rmsd_threshold=0.0,
        )
    assert proposer_call_count["n"] == 5


def test_mcmm_basin_memory_grows_with_accepting_proposer():
    """When the proposer accepts moves (mocked here to always succeed
    into a slightly-shifted novel basin), the basin set grows beyond the
    initial single basin — basin memory accumulates discovered basins
    across walkers and steps."""

    def _accepting_proposer_factory(mol, hardware_opts, calc, **kwargs):
        del hardware_opts, calc, kwargs
        n_atoms = mol.GetNumAtoms()
        # Persistent counter so each call moves further from origin —
        # successive proposals visit distinct basins.
        counter = {"n": 0}

        def fn(coords_list):
            proposals = []
            for coords in coords_list:
                counter["n"] += 1
                # Shift x of the first atom by a unique large offset
                # (well above any reasonable rmsd_threshold) — guarantees
                # distinct basins across calls and walkers.
                new_coords = coords.clone()
                new_coords[0, 0] += 100.0 * counter["n"]
                # Energy decreases with each step so moves are downhill
                # and always accepted under finite kT.
                proposals.append((new_coords, -float(counter["n"]), 1.0, True))
            return proposals

        return fn

    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch(
            "confsweeper.make_mcmm_proposer",
            side_effect=_accepting_proposer_factory,
        ),
    ):
        mol, centroid_ids, energies = get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=2,
            n_temperatures=2,
            n_steps=3,
            swap_interval=100,
            rmsd_threshold=0.0,  # never collapse basins in the shared tail
            e_window_kT=1e9,  # don't drop any basins by energy
        )
    # With rmsd_threshold=0, the strict `<` basin-novelty check never
    # matches, so every walker's init adds its own basin (4 basins for
    # 4 walkers, even though they share starting coords) and every one
    # of the 12 accepted proposals (4 walkers × 3 steps) adds another.
    # Total: 4 + 12 = 16 basins, all surviving the rmsd_threshold=0 /
    # e_window=∞ shared tail.
    assert len(centroid_ids) == 16
    assert mol.GetNumConformers() == 16
    assert energies == sorted(energies)


def test_mcmm_minimize_unknown_backend_raises():
    """An unrecognised mmff_backend value raises ValueError, mirroring
    the dispatch in get_mol_PE_pool_b / get_mol_PE_exhaustive."""
    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch("confsweeper.make_mcmm_proposer", side_effect=_stub_proposer_factory),
    ):
        with pytest.raises(ValueError, match="unknown mmff_backend"):
            get_mol_PE_mcmm(
                TEST_SMILES,
                get_embed_params(),
                hardware_opts=None,
                calc=MagicMock(),
                n_walkers_per_temp=1,
                n_temperatures=2,
                n_steps=2,
                minimize=True,
                mmff_backend="something_else",
            )


def test_mcmm_default_temperature_endpoints():
    """Defaults seed the temperature ladder at kT_low = _KT_EV_298K
    (≈300 K) and kT_high = 2 × _KT_EV_298K (≈600 K) — the issue-#11
    target ladder."""
    captured: dict = {}

    def _capturing_proposer_factory(mol, hardware_opts, calc, **kwargs):
        del mol, hardware_opts, calc, kwargs

        def fn(coords_list):
            return [(c, 0.0, 0.0, False) for c in coords_list]

        return fn

    # Wrap the driver to peek at its kts at construction time.
    import confsweeper

    original_remd = confsweeper.ReplicaExchangeMCMMDriver

    class CapturingREMD(original_remd):
        def __init__(self, walkers_by_temp, *args, **kwargs):
            super().__init__(walkers_by_temp, *args, **kwargs)
            captured["kts"] = list(self.kts)

    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch(
            "confsweeper.make_mcmm_proposer",
            side_effect=_capturing_proposer_factory,
        ),
        patch("confsweeper.ReplicaExchangeMCMMDriver", CapturingREMD),
    ):
        get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=1,
            n_temperatures=4,
            n_steps=1,
            swap_interval=100,
        )

    assert captured["kts"][0] == pytest.approx(_KT_EV_298K)
    assert captured["kts"][-1] == pytest.approx(2.0 * _KT_EV_298K)


# ---------------------------------------------------------------------------
# get_mol_PE_mcmm — multi-seed initialisation (lever C9)
# ---------------------------------------------------------------------------


def test_mcmm_multi_seed_embeds_n_init_confs_etkdg_seeds():
    """`n_init_confs=K` makes the function call ETKDG with
    `confsPerMolecule=K`, embedding K distinct seed conformers up
    front. With the no-exploration stub proposer, no MC moves change
    the picture; the resulting basin set after dedup reflects the K
    distinct seeds."""
    embed_call_args = {}

    def _capture_embed(mols, params, confsPerMolecule, hardwareOptions):
        embed_call_args["confsPerMolecule"] = confsPerMolecule
        # Add `confsPerMolecule` random conformers via CPU ETKDG so the
        # rest of the pipeline has something to MMFF/score.
        for m in mols:
            AllChem.EmbedMultipleConfs(
                m,
                numConfs=confsPerMolecule,
                randomSeed=params.randomSeed,
                clearConfs=False,
            )

    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_capture_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch("confsweeper.make_mcmm_proposer", side_effect=_stub_proposer_factory),
    ):
        get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=2,
            n_temperatures=2,
            n_steps=1,
            n_init_confs=8,
            rmsd_threshold=0.0,
        )

    assert embed_call_args["confsPerMolecule"] == 8


def test_mcmm_multi_seed_distributes_walkers_round_robin():
    """With n_init_confs distinct seeds and N walkers, the basin memory
    after construction holds up to n_init_confs basins (one per
    distinct seed). Round-robin distribution: walker w starts at seed
    `w % n_init_confs`. With rmsd_threshold = 0 every seed is unique,
    so basin memory exactly hits n_init_confs after construction."""

    def _embed_distinct_seeds(mols, params, confsPerMolecule, hardwareOptions):
        # Embed distinct conformers via CPU ETKDG. With rmsd_threshold=0
        # downstream, each seed will be its own basin.
        for m in mols:
            AllChem.EmbedMultipleConfs(
                m,
                numConfs=confsPerMolecule,
                randomSeed=params.randomSeed,
                clearConfs=False,
            )

    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_embed_distinct_seeds),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch(
            "nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs",
            return_value=[[]],
        ),
        patch("confsweeper.make_mcmm_proposer", side_effect=_stub_proposer_factory),
    ):
        mol, centroid_ids, energies = get_mol_PE_mcmm(
            TEST_SMILES,
            get_embed_params(),
            hardware_opts=None,
            calc=MagicMock(),
            n_walkers_per_temp=2,
            n_temperatures=2,  # 4 walkers total
            n_steps=1,
            n_init_confs=4,
            rmsd_threshold=0.0,
            e_window_kT=1e9,  # don't drop any basin via energy filter
        )
    # 4 walkers across 4 distinct seeds → 4 distinct basins in memory.
    # The shared tail with rmsd_threshold=0 / e_window=∞ keeps all 4.
    assert len(centroid_ids) == 4
    assert mol.GetNumConformers() == 4


def test_mcmm_multi_seed_invalid_count_raises():
    """`n_init_confs < 1` is rejected at the entry point."""
    mock_mace = _make_seq_mock_mace()
    with (
        patch("confsweeper.embed.EmbedMolecules", side_effect=_mock_etkdg_embed),
        patch("confsweeper._mace_batch_energies", side_effect=mock_mace),
        patch("confsweeper.make_mcmm_proposer", side_effect=_stub_proposer_factory),
    ):
        with pytest.raises(ValueError, match="n_init_confs must be >= 1"):
            get_mol_PE_mcmm(
                TEST_SMILES,
                get_embed_params(),
                hardware_opts=None,
                calc=MagicMock(),
                n_walkers_per_temp=1,
                n_temperatures=2,
                n_steps=1,
                n_init_confs=0,
            )
