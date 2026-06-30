# Side-chain rotamer sampling — all-atom CREST coverage of inverted peptides

Branch: `22-side-chain-rotamer-sampling`. Implements issue #22 (follow-up to #20/PR #23,
the learned dihedral predictor / backbone-seeding work).

## 20260630

## Background (carried over from #20 / PR #23)

Issue #20 built learned dihedral seeding to reach the MMFF↔CREST-inverted dominant basins
de-novo MMFF/MC misses. The result split along the backbone/side-chain axis:

- **Backbone seeding works and shipped** — a learned backbone (φ/ψ/ω) predictor + constrained-DG
  seeding recovers backbone-driven inversions (+0.21 mean backbone `cov_bw_ceil` lift; 0→1.0
  where de-novo misses the fold). Infra/mechanism validated (the true-geometry oracle
  reproduces the dominant basin end-to-end).
- **All-atom coverage was NOT achieved** — across backbone+χ prediction, backbone-conditioned
  χ, a teacher-forcing fix, and refinement (sampled-χ prior + per-seed MACE-relax), all-atom
  `cov_bw_ceil` of inverted peptides stayed at 0.000 while the oracle covers fully.

Two diagnosed causes of the wall: **(1) compounding error** — all-atom coverage needs the
backbone AND all χ correct at once; realistic χ accuracy caps at ~0.15 per-peptide (backbone
noise upper-bounds the backbone-conditioned χ). **(2) ill-posed target** — side chains are
often floppy (the CREST ensemble samples multiple rotamers; e.g. cremp peptide P.A.Q.I's
all-atom dominant basin is only 0.353 of the ensemble), so predicting *the* single dominant
rotamer is inherently noisy. Part of the χ ceiling is genuine multimodality, not model
weakness.

**Reusable from #20 (`src/dihedral_predictor/`):** χ extraction (`sidechain_chi_quads`,
canonical-rank slot consistency), `ChiPredictor` (separate model), χ seeding via
`SetDihedralDeg` with a sampled-χ prior (`predict_chi(sample=True)`,
`seed_conformers(chi_sample=…)`), the no-relax seed path (`get_mol_PE_mcmm(relax_seeds=…)`),
and the raw-CREST, CREST-Boltzmann-weighted, single-provenance coverage harness
(`scripts/validate_seeding_coverage.py`, `aggregate_seeding_coverage.py`).

## Goals and constraints

- **Goal.** Close the all-atom gap so learned seeding reproduces the CREST *side-chain
  distribution* (not just the backbone fold) on backbone-recovered inverted peptides —
  lifting all-atom `cov_bw_ceil` above the current 0.000.
- **Reproduce the distribution, not one conformer.** The metric and the χ target should
  reflect that the CREST ensemble is multi-rotamer; aim for distribution/top-K coverage.
- **Constraints.** Reuse the #20 infra above; keep the χ model separate from the backbone
  model; MACE is the cheap re-scorer (full-MCMM MACE-relax is infeasible at ~600× MMFF, but
  MACE-relax / MACE-scoring over a handful of seeds is affordable).

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Quantify side-chain floppiness across CREMP | ✓ complete (2026-06-30; only 56% of χ are unimodal — ~44% genuinely multi-rotamer; aromatic χ floppiest at 36%; most of the ~0.15 χ ceiling is genuine multimodality → distribution/top-K target justified) |
| 2 | Reframe the χ target to the rotamer **distribution / top-K** (per-residue rotamer probabilities or top-K rotamer sets) instead of a single argmax; train + evaluate against the multi-rotamer ensemble | pending |
| 3 | **Rotamer-search + MACE-scoring stage** on the seeded (recovered) backbone — prior-guided enumeration/sampling of rotamer combinations, per-residue greedy / branch-and-bound (not the independent sampling prototyped in #20), MACE-scored, keep the best | pending |
| 4 | Higher-accuracy side-chain model — richer side-chain context features and/or explicit rotamer-library priors; revisit as backbone accuracy improves | pending |
| 5 | Re-validate all-atom `cov_bw_ceil` on backbone-recovered inverted peptides (reuse `validate_seeding_coverage.py` / `aggregate_seeding_coverage.py`); compare to the #20 baseline (all-atom 0.000, oracle full) | pending |

## Key questions / unknowns

- How much of the ~0.15 χ-peptide-ok ceiling is genuine rotamer multimodality vs model
  capacity? (Step 1 answers this and sets the target representation.)
- Does predicting the rotamer distribution / top-K + seeding multiple rotamer sets lift
  all-atom coverage where point-χ + MACE-relax did not?
- Is a per-residue rotamer-search + MACE-scoring stage enough to land the ≤0.75 Å all-atom
  match on backbone-recovered peptides?
- What all-atom match tolerance / weighting best reflects "reproducing the CREST distribution"
  for floppy side chains (vs the single-dominant framing)?

## Relationship to other work

- **Upstream:** #20 / PR #23 (backbone seeding + side-chain-wall characterization + χ
  infrastructure). This thread reuses all of it and consumes the same CREMP data + coverage
  harness.
- **Downstream:** closing the all-atom gap makes learned seeding reproduce the full CREST
  distribution, feeding better conformer ensembles to the peptide_electrostatics pipeline.

### Findings 2026-06-30 — Step 1: ~44% of side-chain χ are genuinely multi-rotamer

`scripts/sidechain_floppiness.py` (1500 CREMP peptides, 9641 residue×χ-slot instances).
For each (residue, χ slot), assign every CREST conformer's χ to the nearest sp3 well
(-60/+60/180), Boltzmann-weight, report dominant-well fraction + effective #rotamers
(1/Σp²). "Unimodal" = dominant_frac > 0.8.

| group | unimodal | median dom-frac | mean eff_rotamers |
|---|---|---|---|
| overall | 56% | 0.84 | 1.48 |
| χ1 | 63% | 0.88 | 1.43 |
| χ2 | 42% | 0.75 | 1.59 |
| χ3 | 70% | 0.92 | 1.35 |
| aromatic side chains | 36% | — | 1.66 |
| non-aromatic | 66% | — | 1.39 |

- Only **56%** of χ are effectively unimodal — **~44% are genuinely multi-rotamer** in the
  CREST ensemble (mean ~1.5 wells populated).
- χ1 (near backbone) is more determined (63%) than χ2 (42%); **aromatic side chains are the
  floppiest (36%)** — the NMe-Trp χ in cremp_sharp is exactly this case.
- **This explains the χ ceiling quantitatively:** `chi_peptide_ok` needs every χ right at
  once, but with ~44% of χ multimodal, almost every peptide has ≥1 χ slot where "the
  dominant conformer's rotamer" is ill-defined (≈0.56^(#χ) → a few %). So most of the ~0.15
  ceiling is genuine multimodality, not model weakness.

**Implication:** the single-dominant χ target is fundamentally ill-posed for ~44% of slots.
Reframe to the rotamer **distribution / top-K** (Step 2) and seed multiple rotamer sets — the
goal is to reproduce the CREST rotamer *distribution*, not one conformer.

## Deferred follow-ups

Carried over from `docs/dihedral_predictor_plan.md` (still open after #20):

- **Top-K / multi-modal BACKBONE targets** — predict more than the single dominant conformer
  to seed the whole CREST ensemble's backbone diversity, not just the ceiling basin. (Distinct
  from Step 2's side-chain rotamer top-K, though the multi-modal framing is shared.)
- **Topology-stratified split** — stronger generalisation test than the random peptide split;
  the composition (permutation-aware) split is built (`scripts/resplit_topology.py`) but not
  yet used for a generalisation eval of the backbone or χ models.
