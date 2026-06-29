"""Step 5 (issue #19): inversion pre-screen across CREMP.

The cremp_sharp failure is an MMFF↔CREST rank inversion: the CREST-dominant
conformer is MMFF-disfavoured (+2.4 kcal/mol), so the MMFF/MC explorer never
finds it. For CREMP peptides we have the CREST ensemble, so we can detect the
inversion directly and cheaply (MMFF single-points only, no MACE, no sampler
run): MMFF-score every CREST conformer and measure how far the CREST-dominant
sits above the peptide's MMFF-best.

  dMMFF(dominant) = MMFF_sp(CREST-dominant) - min_i MMFF_sp(conf_i)   [kcal/mol]

Large dMMFF -> MMFF disagrees with CREST about the best conformer -> the
MMFF-driven sampler will miss it -> inversion / de-novo-unreliable. This both
(a) quantifies the true inversion prevalence (vs the ~11% "sharp" upper bound)
and (b) shows which cheap sequence features predict it (for flagging novel
peptides without a CREST ensemble).

Pure CPU. Stratified sample over CREMP poplowestpct. Resume-aware CSV.
"""

import csv
import os
import pickle

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

SUMMARY = "data/raw/cremp/summary.csv"
PICKLE_DIR = "data/raw/cremp/pickle"
OUT = "data/processed/inversion_prescreen.csv"
N_SHARP = 300  # poplowestpct >= 53.48 (cremp_sharp level)
N_CONTROL = 300  # random across the rest
SEED = 42
SHARP_THRESH = 53.48


def mmff_singlepoints(mol):
    """MMFF94 single-point energies (kcal/mol) for every conformer of `mol`,
    on the as-given (GFN2-xTB / CREST) geometry — no relaxation.

    Single-points on foreign (GFN2-xTB) geometries carry cross-method strain
    that inflates dMMFF; kept alongside `mmff_relaxed_energies` to document the
    single-point-vs-relaxed disparity (anchor: cremp_sharp 13.5 sp vs 7.1 relaxed).
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    out = []
    for c in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=c.GetId())
        out.append(ff.CalcEnergy())
    return np.array(out)


def mmff_relaxed_energies(mol):
    """MMFF94-relaxed energies (kcal/mol) for every conformer of `mol`.

    Relax with MMFF (the energy surface the sampler actually explores), then take
    the relaxed energy. This is the sampling-relevant metric; single-points are
    the cross-method-strain-inflated comparison.

    Params:
        mol: Chem.Mol : molecule with conformers (CREST geometries, explicit Hs)
    Returns:
        np.ndarray of relaxed MMFF energies (kcal/mol), or None if unparametrised
    """
    if AllChem.MMFFGetMoleculeProperties(mol) is None:
        return None
    m = Chem.Mol(mol)
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=1000, numThreads=0)
    props = AllChem.MMFFGetMoleculeProperties(m)
    out = []
    for c in m.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=c.GetId())
        out.append(ff.CalcEnergy())
    return np.array(out)


def n_nme(seq):
    return sum(1 for tok in seq.split(".") if tok.startswith("Me"))


def has_aromatic(seq):
    return any(
        tok.replace("Me", "") in ("W", "F", "Y", "H", "w", "f", "y", "h")
        for tok in seq.split(".")
    )


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = list(csv.DictReader(open(SUMMARY)))
    rng = np.random.default_rng(SEED)
    sharp = [r for r in rows if float(r["poplowestpct"]) >= SHARP_THRESH]
    rest = [r for r in rows if float(r["poplowestpct"]) < SHARP_THRESH]
    rng.shuffle(sharp)
    rng.shuffle(rest)
    sample = sharp[:N_SHARP] + rest[:N_CONTROL]

    done = set()
    if os.path.exists(OUT):
        done = {r["sequence"] for r in csv.DictReader(open(OUT))}
    write_header = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if write_header:
        w.writerow(
            [
                "sequence",
                "num_monomers",
                "poplowestpct",
                "n_nme",
                "has_aromatic",
                "n_conf",
                "dMMFF_dominant_sp_kcal",
                "dMMFF_dominant_relaxed_kcal",
                "dominant_sp_rank_frac",
                "dominant_relaxed_rank_frac",
            ]
        )

    n_ok = 0
    for r in sample:
        seq = r["sequence"]
        if seq in done:
            continue
        path = os.path.join(PICKLE_DIR, f"{seq}.pickle")
        try:
            d = pickle.load(open(path, "rb"))
            mol = d["rd_mol"]
            bw = np.array([c["boltzmannweight"] for c in d["conformers"]], float)
            e_sp = mmff_singlepoints(mol)
            e_rx = mmff_relaxed_energies(mol)
            if e_sp is None or e_rx is None or len(e_sp) != len(bw) or len(e_sp) < 2:
                continue
            dom = int(bw.argmax())
            dmmff_sp = float(e_sp[dom] - e_sp.min())
            dmmff_rx = float(e_rx[dom] - e_rx.min())
            rank_sp = float((e_sp < e_sp[dom]).sum() / len(e_sp))
            rank_rx = float((e_rx < e_rx[dom]).sum() / len(e_rx))
            w.writerow(
                [
                    seq,
                    r["num_monomers"],
                    r["poplowestpct"],
                    n_nme(seq),
                    int(has_aromatic(seq)),
                    len(e_sp),
                    f"{dmmff_sp:.3f}",
                    f"{dmmff_rx:.3f}",
                    f"{rank_sp:.3f}",
                    f"{rank_rx:.3f}",
                ]
            )
            fh.flush()
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {seq}: {exc}")
    fh.close()
    print(f"scored {n_ok} peptides -> {OUT}")

    # summary
    data = list(csv.DictReader(open(OUT)))
    sp = np.array([float(x["dMMFF_dominant_sp_kcal"]) for x in data])
    rx = np.array([float(x["dMMFF_dominant_relaxed_kcal"]) for x in data])
    pl = np.array([float(x["poplowestpct"]) for x in data])
    nme = np.array([int(x["n_nme"]) for x in data])
    arom = np.array([int(x["has_aromatic"]) for x in data])
    sharp_mask = pl >= SHARP_THRESH
    print(f"\n=== inversion pre-screen ({len(data)} peptides) ===")
    print("dMMFF(dominant) kcal/mol      median   75th   90th    max")
    for label, d in [("single-point", sp), ("MMFF-relaxed", rx)]:
        print(
            f"  {label:12s}             {np.median(d):6.2f} {np.percentile(d,75):6.2f} "
            f"{np.percentile(d,90):6.2f} {d.max():6.2f}"
        )
    print(
        f"  single-point/relaxed inflation (median ratio): "
        f"{np.median(sp)/max(np.median(rx),1e-6):.1f}x"
    )
    print("\nrelaxed-dMMFF threshold (anchors: typical 0.18, sharp 7.1 kcal/mol):")
    for thr in (1.0, 2.0, 3.0):
        print(
            f"  flagged (relaxed dMMFF>{thr}): all {100*(rx>thr).mean():.0f}% | "
            f"sharp {100*(rx[sharp_mask]>thr).mean():.0f}% | "
            f"non-sharp {100*(rx[~sharp_mask]>thr).mean():.0f}%"
        )
    flag = rx > 2.0
    print(
        f"\nfeature means, relaxed-flagged (>2) vs not: "
        f"poplowestpct {pl[flag].mean():.0f}% vs {pl[~flag].mean():.0f}% | "
        f"n_nme {nme[flag].mean():.1f} vs {nme[~flag].mean():.1f} | "
        f"aromatic {100*arom[flag].mean():.0f}% vs {100*arom[~flag].mean():.0f}%"
    )


if __name__ == "__main__":
    main()
