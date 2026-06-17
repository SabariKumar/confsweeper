"""
Tests for `scripts/sampler_benchmark.py` helpers. Focused on regression
guards for behaviours discovered in the Step-7 / v0.2 Step-4 sweeps.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rdkit import Chem

# scripts/ isn't on the default sys.path; insert it so the import works
# without depending on PYTHONPATH or pixi env tweaks.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from sampler_benchmark import _maybe_dump_sdf  # noqa: E402

# ---------------------------------------------------------------------------
# _maybe_dump_sdf — zero-basin defensive write skip (v0.2 Step-4 regression)
# ---------------------------------------------------------------------------


def _make_mol_with_two_confs() -> Chem.Mol:
    """Build a trivial mol with two conformers so we have something to
    actually write when conf_ids is non-empty."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    # Two conformers with distinct positions so SDWriter has real data.
    for shift in (0.0, 1.0):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(i) + shift, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)
    return mol


def test_maybe_dump_sdf_writes_file_when_conf_ids_non_empty(tmp_path):
    """Sanity: the normal path writes a valid SDF with the expected
    conformer count."""
    mol = _make_mol_with_two_confs()
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    energies = [-1.234, -2.345]
    _maybe_dump_sdf(
        mol=mol,
        conf_ids=conf_ids,
        energies_eV=energies,
        peptide_id="cremp:t.I.G.N",
        sampler="mcmm",
        dump_sdf_dir=tmp_path,
    )
    out_path = tmp_path / "cremp_t.I.G.N_mcmm.sdf"
    assert out_path.exists(), "SDF must be written when conf_ids is non-empty"
    assert (
        out_path.stat().st_size > 0
    ), "SDF must not be empty when conf_ids is non-empty"
    # Round-trip: SDMolSupplier should read back two conformers without raising.
    suppl = Chem.SDMolSupplier(str(out_path), removeHs=False)
    read_mols = [m for m in suppl if m is not None]
    assert (
        len(read_mols) == 2
    ), f"expected 2 conformers in dumped SDF; got {len(read_mols)}"


def test_maybe_dump_sdf_skips_write_when_conf_ids_empty(tmp_path):
    """The v0.2 Step-4 regression: when a sampler crashes mid-run (e.g.
    CUDA OOM) it returns an empty `energies_eV` list. The old code path
    would open Chem.SDWriter and immediately close it, producing a
    0-byte file that union_basin_count._load_basin_sdf cannot read
    (raises OSError on Chem.SDMolSupplier). Fix: skip the write entirely
    when conf_ids is empty, so downstream globbers simply miss the file
    rather than crash on a malformed one."""
    mol = _make_mol_with_two_confs()
    _maybe_dump_sdf(
        mol=mol,
        conf_ids=[],
        energies_eV=[],
        peptide_id="cremp:S.S.N.MeW.MeA.MeN",
        sampler="mcmm",
        dump_sdf_dir=tmp_path,
    )
    out_path = tmp_path / "cremp_S.S.N.MeW.MeA.MeN_mcmm.sdf"
    assert not out_path.exists(), (
        "SDF must NOT be written when conf_ids is empty — a 0-byte file "
        "crashes downstream tools (union_basin_count._load_basin_sdf "
        "raises OSError on Chem.SDMolSupplier of an empty file)"
    )


def test_maybe_dump_sdf_skips_write_when_dump_sdf_dir_is_none(tmp_path):
    """Sanity for the existing dump-disabled fast path: when the caller
    passes None, the function returns without touching the filesystem
    even on non-empty conf_ids."""
    mol = _make_mol_with_two_confs()
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    _maybe_dump_sdf(
        mol=mol,
        conf_ids=conf_ids,
        energies_eV=[-1.0, -2.0],
        peptide_id="cremp:t.I.G.N",
        sampler="mcmm",
        dump_sdf_dir=None,
    )
    # tmp_path should remain empty since dump_sdf_dir was None.
    assert (
        list(tmp_path.iterdir()) == []
    ), "dump_sdf_dir=None must not touch the filesystem"
