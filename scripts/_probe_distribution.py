"""Step 3 (issue #19): de-novo reachability of conf0 vs seed budget AND sampling
distribution.

Tests the user's "fill in the distribution" idea (Lever 4): ETKDG samples
torsions from a non-uniform CSD-derived prior; the inverted basin (conf0) lives
where that prior is low, so ETKDG rarely embeds there. Compares three embedding
distributions at matched seed budgets and reports the nearest-to-conf0 heavy-atom
(Kabsch) RMSD:

  - etkdgv3_macro : ETKDGv3 + macrocycle priors (CSD-biased; current default)
  - no_exp_tors   : same but useExpTorsionAnglePrefs=False (drop CSD torsion prior)
  - plain_dg      : all torsion knowledge off + random coords (most uniform)

If a less-biased / uniform distribution reaches conf0 (RMSD ↓ toward 0.125 Å)
where ETKDGv3 plateaus, Lever 4 is the de-novo fix.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdDistGeom import ETKDGv3

from confsweeper import embed, get_hardware_opts

CREMP_SHARP = (
    "C[C@H]1C(=O)N(C)[C@@H](CC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)"
    "C(=O)N[C@@H](CC(N)=O)C(=O)N(C)[C@@H](Cc2c[nH]c3ccccc23)C(=O)N1C"
)
CEILING = "results/cremp_ceiling_sdfs/S.S.N.MeW.MeA.MeN.sdf"
CHUNK = 5000
MAX_SEEDS = 30000
BUDGETS = [1000, 10000, 30000]


def kabsch_rmsd_batch(P, Q):
    """Heavy-atom Kabsch RMSD of a batch of conformers P (B,N,3) to ref Q (N,3)."""
    Pc = P - P.mean(axis=1, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = np.einsum("bni,nj->bij", Pc, Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("bij,bjk->bik", U, Vt)))
    sumS = S[:, 0] + S[:, 1] + d * S[:, 2]
    n = P.shape[1]
    msd = (np.square(Pc).sum(axis=(1, 2)) + np.square(Qc).sum() - 2.0 * sumS) / n
    return np.sqrt(np.maximum(msd, 0.0))


def _macro():
    p = ETKDGv3()
    p.useRandomCoords = True
    p.useMacrocycleTorsions = True
    p.useMacrocycle14config = True
    return p


def _no_exp_tors():
    p = _macro()
    p.useExpTorsionAnglePrefs = False
    return p


def _plain_dg():
    p = ETKDGv3()
    p.useRandomCoords = True
    p.useExpTorsionAnglePrefs = False
    p.useBasicKnowledge = False
    p.useMacrocycleTorsions = False
    return p


def reach(params, q, heavy):
    mol = Chem.AddHs(Chem.MolFromSmiles(CREMP_SHARP))
    hw = get_hardware_opts()
    s = 0
    while mol.GetNumConformers() < MAX_SEEDS:
        params.randomSeed = s
        embed.EmbedMolecules([mol], params, confsPerMolecule=CHUNK, hardwareOptions=hw)
        s += CHUNK
        if mol.GetNumConformers() == 0 and s >= 3 * CHUNK:
            break  # distribution can't embed this mol at all
    n_have = mol.GetNumConformers()
    if n_have == 0:
        return n_have, {}
    allpos = np.array(
        [mol.GetConformer(c.GetId()).GetPositions()[heavy] for c in mol.GetConformers()]
    )
    rmsd = kabsch_rmsd_batch(allpos, q)
    out = {}
    for b in BUDGETS:
        if b <= n_have:
            sub = rmsd[:b]
            out[b] = (
                float(sub.min()),
                int((sub <= 0.125).sum()),
                int((sub <= 0.36).sum()),
            )
    return n_have, out


def main():
    ref_mol = Chem.AddHs(Chem.MolFromSmiles(CREMP_SHARP))
    conf0 = list(Chem.SDMolSupplier(CEILING, removeHs=False))[0]
    match = ref_mol.GetSubstructMatch(conf0)
    inv = [0] * ref_mol.GetNumAtoms()
    for c0i, mi in enumerate(match):
        inv[mi] = c0i
    heavy = [a.GetIdx() for a in ref_mol.GetAtoms() if a.GetAtomicNum() != 1]
    c0pos = conf0.GetConformer().GetPositions()
    q = np.array([c0pos[inv[i]] for i in heavy])

    for name, builder in [
        ("etkdgv3_macro", _macro),
        ("no_exp_tors", _no_exp_tors),
        ("plain_dg", _plain_dg),
    ]:
        n_have, out = reach(builder(), q, heavy)
        print(f"\n=== {name} (embedded {n_have}) ===")
        print("  budget | nearest RMSD | n<=0.125 | n<=0.36")
        for b in BUDGETS:
            if b in out:
                near, n012, n036 = out[b]
                print(f"  {b:>6} | {near:6.3f} Å   | {n012:5d}    | {n036:5d}")


if __name__ == "__main__":
    main()
