# MMFF↔CREST/MACE inversion — cheap levers to reach MMFF-disfavoured basins

Branch: `19-mmff-crestmace-inversion-investigation`. Implements issue #19 (follow-up to #17).

## 20260625

This thread follows directly from the v0.3 root-cause finding
(`docs/concerted_moves_v0_3_plan.md`): the `cremp_sharp` coverage gap is an
**MMFF↔MACE energy rank inversion**, not a sampling-move deficiency. The
CREST-dominant conformer (conf0) is ranked ~2.4 kcal/mol *above* our best sampled
basin by MMFF (which drives exploration) but ~10 kcal/mol *below* by MACE (which
only re-scores). Because MACE is downstream of MMFF, it can re-rank the basins
MMFF finds but cannot make MMFF *discover* a basin MMFF avoids. The goal here is
to find **cheap, MMFF/MC-only levers** that give the explorer access to
MMFF-disfavoured-but-CREST-real geometries — **without MACE relaxation** (out of
scope; it defeats the low-compute objective).

## Goals and constraints

**Goal.** Improve CREST-ensemble coverage on the inversion subset (canonical case
`cremp_sharp`) at low compute, while not regressing the agreement case
(`cremp_typical`, currently ~0.99). "Coverage" = `cov_bw_ceil` (Boltzmann-weighted
coverage of the CREST ceiling), `match_rmsd = 0.125 Å`.

**Constraints (locked).**
- MMFF/MC drive exploration; MACE is a single-point re-scorer ONLY. No
  MACE-relaxation anywhere in the inner loop or final relax.
- Reproduce CREST as closely as possible at lowest compute — every lever is
  judged on coverage-gained-per-GPU-second.
- Don't regress cremp_typical (≥ ~0.99) or any of the 497-passing test suite.
- Benchmark dyad: `cremp_sharp` (inversion / failure) vs `cremp_typical`
  (agreement / success). Definitions in `docs/concerted_moves_v0_3_plan.md`.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Foundational diagnostic: MMFF-iteration reachability of conf0 (does lighter/no MMFF preserve geometries near conf0 that full MMFF collapses away?) | ✓ complete (2026-06-25; light MMFF RULED OUT — ETKDG never gets near conf0 at any cap) |
| 2 | Lever 2 — partial/light MMFF relaxation (`mmff_max_iters`) | ✗ ruled out by Step 1 (no near-conf0 geometries for MMFF to preserve) |
| 3 | Lever 1 + 4 — de-novo reachability of conf0 vs (a) seed budget and (b) **sampling distribution**: ETKDGv3-macrocycle (CSD-biased) vs torsion-prefs-off vs plain-DG (uniform). Does a less-biased / uniform torsion distribution reach conf0 where ETKDG's prior avoids it? | in progress |
| 4 | Inversion pre-screen: does a cheap MMFF-vs-MACE rank-disagreement metric (or poplowestpct) predict which peptides suffer the inversion? | pending |
| 5 | Document the structural ceiling + benchmark figures; wrap-up | pending |

## Lever menu (cheap, no MACE relaxation)

Priority subset marked ★. Cost/impact are rough first estimates.

- ★ **Lever 2 — lighter / partial MMFF relaxation.** Cap MMFF iterations so a
  geometry that starts near conf0 (from ETKDG or a move) doesn't fully collapse
  into the lower-MMFF neighbour before MACE scores it. Cost: ~free (fewer MMFF
  steps = cheaper). Impact: directly targets the identified mechanism (MMFF pulls
  proposals off conf0). Risk: residual strain degrades MACE single-point accuracy
  — find the iteration count that balances reachability vs scoring noise.
- ★ **Lever 1 — seeding diversity.** Richer/larger ETKDG seed pools, basins
  accumulated across independent runs, or (where a reference exists) direct
  CREMP-seeding (confirmed to close cremp_sharp → 0.724). Cost: linear in pool
  size. Impact: high if a de-novo pool can land in conf0's basin; bounded by how
  narrow that basin is.
- ★ **Lever 4 — uniform / unbiased torsion sampling (the "fill in the
  distribution" idea).** ETKDG samples torsions from a non-uniform,
  CSD-derived prior (`useExpTorsionAnglePrefs`) plus basic-knowledge and
  macrocycle priors; the inverted basin (conf0) lives where that prior assigns
  low probability, so ETKDG rarely embeds there regardless of seed count.
  Replace the biased distribution with a more uniform one (turn off
  `useExpTorsionAnglePrefs`, optionally `useBasicKnowledge` /
  `useMacrocycleTorsions`, with `useRandomCoords`) so under-sampled torsion
  regions get covered. Cost: ~free (same embed, different flags); may need more
  seeds since uniform sampling is less efficient on the easy basins. Impact:
  potentially high — directly attacks the de-novo reachability gap Step 1
  identified. Risk: uniform DG yields looser/invalid geometries (mitigated by
  the downstream MMFF relax + MACE filter, or by keeping basic-knowledge on).
- **Lever 3 — accept + bound the ceiling.** Quantify how often the inversion
  occurs and document cremp_sharp as the canonical case. Cost: analysis only.
  Impact: sets expectations; not a fix.
- **(reserve) skip-MMFF-on-a-fraction.** `skip_mmff_relax` already exists for
  side-chain moves; extend selectively. Risk: strained geometries → unreliable
  MACE. Use only if light MMFF (Lever 2) underperforms.

## Risks to instrument from day one

- **MACE single-point unreliability on under-relaxed geometries.** Lighter MMFF
  leaves strain; MACE energies on strained geometries may be noisy/wrong.
  Diagnostic: track MACE-energy spread of a basin under varying MMFF iterations;
  flag if acceptance/ranking destabilises. Mitigation: a minimum iteration floor.
- **cremp_typical regression.** Any lever that helps cremp_sharp must be checked
  on cremp_typical in the same sweep (the v0.2/v0.3 sweeps sometimes dropped
  typical from cells — do not repeat that; run both peptides every cell).
- **Bursty exhaustive-ETKDG saturation.** Seed-pool quality varies run-to-run
  (seen in v0.3: 1 vs 6 basins at n=10k). Diagnostic: report pool e_min and
  nearest-to-conf0 per run; average over seeds, don't trust a single draw.

## Deferred follow-ups

- Constrained MMFF (freeze the rotated dihedral) as a middle ground between full
  and skipped relaxation — trigger: if `mmff_max_iters` capping (Lever 2) helps
  but leaves too much strain for reliable MACE scoring.
- Graceful-degradation for Move A (concerted) on non-aromatic peptides (carried
  over from v0.3; same class as the ω/large-window fixes) — trigger: if any
  production mix re-enables `concerted_dihedral_weight` across a mixed dataset.

## Findings

(append-only, dated)

### Findings 2026-06-25 — Step 1: light MMFF ruled out; the gap is de-novo ETKDG reachability

Embedded a 300-conformer ETKDG pool (macrocycle params) of cremp_sharp and
measured nearest-to-conf0 heavy-atom RMSD as MMFF relaxation is capped at
{0 (raw), 5, 20, 50, 200, 2000} iterations:

| MMFF cap | nearest RMSD | n ≤ 0.125 Å | n ≤ 0.36 Å |
|---|---|---|---|
| raw (0) | 0.987 Å | 0 | 0 |
| 5 | 0.980 | 0 | 0 |
| 20 | 0.979 | 0 | 0 |
| 50 | 0.972 | 0 | 0 |
| 200 | 0.977 | 0 | 0 |
| 2000 (full) | 1.187 | 0 | 0 |

**Light MMFF (Lever 2) is ruled out.** The pool never gets within ~1 Å of conf0
at *any* MMFF cap — so MMFF is not collapsing near-conf0 geometries (there are
none to preserve). ETKDG simply does not generate conf0's fold de novo at this
pool size. Full MMFF (2000) nudges the pool slightly *farther* (0.99 → 1.19 Å),
consistent with MMFF pulling toward its own basins, but it's moot since nothing
is in range. **The mechanism is upstream of the relaxer: de-novo ETKDG
reachability, not MMFF collapse.**

Seed-budget scaling so far: 300 raw seeds → 0.99 Å nearest; the v0.3 exhaustive
run at 10k seeds → 0.36 Å nearest basin; never < 0.125 Å. → Step 3 quantifies
whether more seeds ever cross 0.125 Å or asymptote short (the de-novo ceiling
for Lever 1).
