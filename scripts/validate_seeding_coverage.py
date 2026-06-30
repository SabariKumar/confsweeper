"""Step 7 (issue #20): does learned-dihedral seeding lift CREST-distribution
coverage in the *real* MCMM pipeline?

The project goal is to reproduce the CREST conformer distribution with cheap
sampling. So the ceiling is the raw CREST ensemble (from CREMP), clustered into
basins weighted by **CREST Boltzmann population** (not MACE), and coverage is the
CREST-weighted fraction of ceiling basins a sampler reproduces.

Everything is computed in one atom order (the smi-built mol the sampler uses):
CREST conformers are mapped onto it via substructure match, and RMSDs are
heavy-atom Kabsch — single-provenance, so the oracle (true CREST dominant)
registers as covering the dominant basin by construction.

Runs the actual confsweeper MCMM (MC + MACE) with/without learned-dihedral seeds
injected via `get_mol_PE_mcmm(extra_seed_coords=)`. Default peptide cremp_sharp is
in the model's TRAIN split → mechanism proof-of-concept, not generalisation.
"""

import pickle
import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from confsweeper import (  # noqa: E402
    _KT_EV_298K,
    get_embed_params_macrocycle,
    get_hardware_opts,
    get_mace_calc,
    get_mol_PE_mcmm,
)
from dihedral_predictor.seed import load_model, seed_conformers  # noqa: E402
from mcmm import _kabsch_rmsd_pairwise  # noqa: E402


def _mapped_coords(cmol, match, n_atoms):
    """All CREST conformers' coords reordered to the smi-mol atom order."""
    out = []
    for c in cmol.GetConformers():
        p = c.GetPositions()
        out.append(np.stack([p[match[i]] for i in range(n_atoms)]))
    return np.stack(out)  # (N, n_atoms, 3)


def build_crest_ceiling(cmol, weights, ms, match, heavy, basin_rmsd=0.125):
    """Raw-CREST ceiling: cluster CREST conformers (smi-order, heavy-atom Kabsch)
    into basins, weight = summed CREST Boltzmann population. Returns
    (basin_heavy (B,nh,3) torch, basin_weights (B,), dominant_full (n_atoms,3)).

    `match[i]` = the cmol atom index for smi-mol (ms) atom i (from
    cmol.GetSubstructMatch(ms)), so coords are reordered to ms's atom order."""
    full = _mapped_coords(cmol, match, ms.GetNumAtoms())
    Xh = torch.tensor(full[:, heavy], dtype=torch.float64)  # (N, nh, 3)
    order = np.argsort(weights)[::-1]
    rep_idx, rep_w = [], []
    for idx in order:
        if rep_idx:
            d = _kabsch_rmsd_pairwise(
                Xh[idx], torch.stack([Xh[r] for r in rep_idx])
            ).numpy()
            j = int(d.argmin())
            if d[j] <= basin_rmsd:
                rep_w[j] += weights[idx]
                continue
        rep_idx.append(int(idx))
        rep_w.append(float(weights[idx]))
    basin_heavy = torch.stack([Xh[r] for r in rep_idx])
    dom = rep_idx[int(np.argmax(rep_w))]
    return basin_heavy, np.array(rep_w), full[dom]


def coverage(sampler_heavy, basin_heavy, weights, match_rmsd):
    """CREST-weighted fraction of ceiling basins within match_rmsd of a sampler basin."""
    if sampler_heavy.shape[0] == 0:
        return 0.0
    covered = np.zeros(len(weights), dtype=bool)
    for b in range(len(weights)):
        d = _kabsch_rmsd_pairwise(basin_heavy[b], sampler_heavy).numpy()
        covered[b] = d.min() <= match_rmsd
    return float(weights[covered].sum())


def _run(smi, hw, calc, extra, n_seeds, sidechain, relax_seeds=True):
    n_walkers = 64
    sc = (
        dict(
            dihedral_weight=0.2,
            concerted_dihedral_weight=0.2,
            aromatic_wells_deg=(-90.0, 0.0, 90.0, 180.0),
            p_rotamer_jump=0.3,
        )
        if sidechain
        else {}
    )
    return get_mol_PE_mcmm(
        smi,
        get_embed_params_macrocycle(),
        hw,
        calc,
        n_walkers_per_temp=8,
        n_temperatures=8,
        n_steps=max(1, n_seeds // n_walkers),
        drive_sigma_rad=0.3,
        closure_tol=0.05,
        kt_high=8.0 * _KT_EV_298K,
        n_init_confs=8,
        extra_seed_coords=extra,
        relax_seeds=relax_seeds,
        seed=0,
        **sc,
    )


@click.command()
@click.option("--peptide", default="S.S.N.MeW.MeA.MeN")
@click.option("--pickle_dir", default="data/raw/cremp/pickle")
@click.option("--ckpt", default="data/processed/dihedral_predictor/checkpoint.pt")
@click.option("--n_seeds", default=6400)
@click.option("--n_conf", default=12)
@click.option("--sidechain", is_flag=True)
@click.option("--oracle", is_flag=True, help="also run a true-dominant-seeded control")
@click.option(
    "--no_relax_seed",
    is_flag=True,
    help="keep seeds at predicted geometry (no MMFF relax)",
)
def main(peptide, pickle_dir, ckpt, n_seeds, n_conf, sidechain, oracle, no_relax_seed):
    d = pickle.load(open(f"{pickle_dir}/{peptide}.pickle", "rb"))
    smi = d["smiles"]
    cmol = d["rd_mol"]
    w = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
    ms = Chem.AddHs(Chem.MolFromSmiles(smi))
    heavy = [a.GetIdx() for a in ms.GetAtoms() if a.GetAtomicNum() > 1]
    match = cmol.GetSubstructMatch(ms, useChirality=True)

    basin_heavy, weights, dom_full = build_crest_ceiling(cmol, w, ms, match, heavy)
    print(
        f"peptide={peptide}  CREST basins={len(weights)}  dominant CREST weight={weights.max():.3f}",
        flush=True,
    )

    hw = get_hardware_opts()
    calc = get_mace_calc()
    model, window = load_model(ckpt)
    seed_src = Chem.Mol(ms)
    seed_src.RemoveAllConformers()
    seed_ids = seed_conformers(seed_src, model, window=window, n_attempts=n_conf)
    learned = [seed_src.GetConformer(c).GetPositions() for c in seed_ids]
    print(
        f"learned seeds: {len(learned)} | side-chain MC moves: {sidechain}", flush=True
    )

    relax_seeds = not no_relax_seed
    print(f"relax_seeds: {relax_seeds}", flush=True)
    runs = [("baseline", None), ("seeded", learned)]
    if oracle:
        runs.append(("oracle", [dom_full]))
    for label, extra in runs:
        mol, cids, _ = _run(
            smi, hw, calc, extra, n_seeds, sidechain, relax_seeds=relax_seeds
        )
        sh = (
            np.stack([mol.GetConformer(c).GetPositions()[heavy] for c in cids])
            if cids
            else np.empty((0, len(heavy), 3))
        )
        sh = torch.tensor(sh, dtype=torch.float64)
        covs = {
            thr: coverage(sh, basin_heavy, weights, thr) for thr in (0.5, 0.75, 1.0)
        }
        print(
            f"{label:9s}: basins={len(cids):3d}  cov_bw_ceil @0.5={covs[0.5]:.3f} "
            f"@0.75={covs[0.75]:.3f} @1.0={covs[1.0]:.3f}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
