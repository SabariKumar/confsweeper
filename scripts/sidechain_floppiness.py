"""Step 1 (issue #22): quantify side-chain rotamer floppiness across CREMP.

The chi predictor (issue #20) targets the single dominant conformer's chi and caps at
~0.15 peptide-ok. Part of that is ill-posedness: side chains are often multi-rotamer in
the CREST ensemble, so predicting THE dominant rotamer is inherently noisy. This measures,
per residue / chi slot, the CREST-Boltzmann-weighted rotamer distribution and how
concentrated it is — bounding how much of the chi ceiling is genuine multimodality vs
model capacity, and informing the distribution/top-K target representation (Step 2).

For each (peptide, residue, chi slot): assign every CREST conformer's chi to the nearest
sp3 rotamer well (-60 / +60 / 180), Boltzmann-weight the wells, and report:
  dominant_frac   = weight of the most-populated well  (1.0 = unimodal)
  eff_rotamers    = 1 / sum_k p_k^2                     (1 = unimodal, ->3 = uniform)
"Effectively unimodal" = dominant_frac > 0.8.

Pure CPU. Random sample over CREMP.
"""

import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import rdMolTransforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.residues import (  # noqa: E402
    residue_features,
    sidechain_chi_quads,
)

RDLogger.DisableLog("rdApp.*")

SUMMARY = "data/raw/cremp/summary.csv"
PICKLE_DIR = "data/raw/cremp/pickle"
OUT = "data/processed/sidechain_floppiness.csv"
N_SAMPLE = 1500
SEED = 0
WELLS = np.array([-60.0, 60.0, 180.0])
UNIMODAL_THRESH = 0.8


def _well(chi_deg):
    """Index of the nearest sp3 rotamer well (circular)."""
    d = np.abs(((chi_deg - WELLS + 180.0) % 360.0) - 180.0)
    return int(d.argmin())


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = list(csv.DictReader(open(SUMMARY)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)

    fh = open(OUT, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["sequence", "chi_slot", "sc_aromatic", "dominant_frac", "eff_rotamers"])

    n_ok = 0
    recs = []
    for r in rows:
        if n_ok >= N_SAMPLE:
            break
        seq = r["sequence"]
        try:
            d = pickle.load(open(os.path.join(PICKLE_DIR, f"{seq}.pickle"), "rb"))
            mol = d["rd_mol"]
            bw = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            bw = bw / bw.sum()
            quads = sidechain_chi_quads(mol)
            feats = residue_features(mol)  # aligned with quads (ring order)
            confs = list(mol.GetConformers())
            if len(confs) != len(bw):
                continue
            for ri, qs in enumerate(quads):
                arom = int(feats[ri, 5])  # sc_aromatic
                for slot, q in enumerate(qs):
                    p = np.zeros(3)
                    for ci, conf in enumerate(confs):
                        p[_well(rdMolTransforms.GetDihedralDeg(conf, *q))] += bw[ci]
                    dom = float(p.max())
                    eff = float(1.0 / np.square(p).sum())
                    w.writerow([seq, slot + 1, arom, f"{dom:.3f}", f"{eff:.3f}"])
                    recs.append((slot + 1, arom, dom, eff))
            n_ok += 1
        except Exception:  # noqa: BLE001
            continue
    fh.close()

    a = np.array([(s, ar, dm, ef) for (s, ar, dm, ef) in recs], float)
    slot, arom, dom, eff = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    print(
        f"scored {n_ok} peptides -> {len(recs)} (residue, chi-slot) instances -> {OUT}"
    )
    print(f"\n=== overall ===")
    print(f"  dominant_frac: median {np.median(dom):.2f}  mean {dom.mean():.2f}")
    print(
        f"  effectively unimodal (dom>{UNIMODAL_THRESH}): {100*(dom>UNIMODAL_THRESH).mean():.0f}%"
    )
    print(f"  mean eff_rotamers: {eff.mean():.2f}")
    print(f"\n=== by chi slot ===")
    for s in sorted(set(slot.astype(int))):
        m = slot == s
        print(
            f"  chi{s} (n={int(m.sum())}): unimodal {100*(dom[m]>UNIMODAL_THRESH).mean():.0f}% | "
            f"median dom {np.median(dom[m]):.2f} | mean eff_rot {eff[m].mean():.2f}"
        )
    print(f"\n=== aromatic vs non-aromatic side chains ===")
    for label, m in [("aromatic", arom == 1), ("non-arom", arom == 0)]:
        if m.any():
            print(
                f"  {label} (n={int(m.sum())}): unimodal {100*(dom[m]>UNIMODAL_THRESH).mean():.0f}% | "
                f"mean eff_rot {eff[m].mean():.2f}"
            )


if __name__ == "__main__":
    main()
