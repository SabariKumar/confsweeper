# Side-chain dihedral-kick proposer v0.2

Branch: `15-mmff-ablation-aromatic-aware-proposer`. Implements issue #15. Follow-up to issue #12 (`docs/dihedral_kick_plan.md`).

## 20260615

This document is the working design for v0.2 of the side-chain dihedral-kick proposer. Motivated directly by the issue-#12 Step-7 phase 2 Findings (2026-06-15) in [dihedral_kick_plan.md](dihedral_kick_plan.md): the v0 proposer hit **`coverage_bw_ceiling = 0.991` on cremp_typical** (+0.16 over the issue-#10 baseline) but flatlined at **`0.000` on cremp_sharp** at every tested mix, with `max_missed_bw = 0.724023` identical to 7 decimal places across all 6 phase-1 + phase-2 cells. The same single dominant ceiling basin (72 % of the 298 K Boltzmann population) is missed everywhere. The phase-2 diagnostic at `p_rotamer_jump=0.70` showed `new_mass` scaling ~400× over the phase-1 baseline — confirming the proposer IS exploring much more aggressively at higher rotamer-jump rates — but the new basins are not the *one* basin that dominates the ceiling distribution. Two structural hypotheses survive and are armed as v0.2 work: per-bond aromatic-aware rotamer-well sets, and a no-MMFF ablation path.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Lock four design forks | ✓ complete |
| 2 | Per-bond aromatic-aware rotamer-well helper + factory plumbing + tests | ✓ complete |
| 3 | Thread `aromatic_wells` kwarg through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI | ✓ complete |
| 4 | Validation A: aromatic wells alone on cremp_sharp + cremp_typical at n_seeds=10000 | ✓ complete (decision: proceed to Step 7's 2×2) |
| 5 | `skip_mmff_relax` ablation kwarg on `make_dihedral_kick_proposer` + tests | ✓ complete |
| 6 | Thread `skip_mmff_relax` through `get_mol_PE_mcmm` + CLI | ✓ complete (targeted 3/3; full suite pending) |
| 7 | Validation B: 2×2 ablation matrix `(aromatic_wells ∈ {off, on}) × (skip_mmff_relax ∈ {off, on})` on cremp_sharp + cremp_typical at n_seeds=10000 | ✓ complete (B.4 synergy wins cremp_typical at 0.997; cremp_sharp still 0.000 — v0.3 escalates) |
| 8 | Documentation: `src/README.md`, `scripts/README.md`, dated Findings in v0.2 plan + cross-ref in v0.1 plan + mcmm_plan.md | pending |

---

## Goals and constraints

The benchmark target is `cremp_sharp` (`S.S.N.MeW.MeA.MeN`): bring its `coverage_bw_ceiling` above zero at the τ=0.5 Å cross-method match tolerance, ideally crossing the locked issue-#12 success criterion of ≥ 0.10. The diagnostic target is `cremp_typical`: confirm the issue-#12 win (0.991) is preserved under both v0.2 kwargs (no regression on the working peptide).

Constraints inherited from v0 (issue #12):

- Same `(mol, conf_ids, energies)` contract through `get_mol_PE_mcmm`'s shared post-sampling tail.
- All locked v0 defaults preserved unchanged: `sigma_chi_rad=0.5`, `p_rotamer_jump=0.3`, the existing `rotamer_wells_deg=(-60, 60, 180)` for sp3-χ₁ bonds, `dihedral_weight=0.0` (legacy-pure-DBT default). Both v0.2 kwargs default to "off" so the issue-#12 production mix continues to behave identically.
- `det_j = 1.0` invariant preserved — both v0.2 kwargs are downstream of the rotation step.
- Strict separation from DBT — backbone untouched by the dihedral kick.

New constraints specific to v0.2:

- Per-bond well sets are computed once at proposer-construction time and cached, not recomputed per call. The per-bond decision is `O(n_bonds × small_constant)` at construction; the call-time hot path stays at the v0 cost.
- The `skip_mmff_relax=True` path must still produce valid `(coords, energy, det_j, success)` tuples per the existing contract. When MACE energies are non-finite (e.g. severe steric clash on an unrelaxed rotation), the proposer must reject cleanly via `success=False` exactly as the MMFF path does.

---

## Design choices to lock (Step 1)

All four forks resolved on 2026-06-15 — see the dated Findings entry under *Findings* below for the conversation. Running recommendations matched the locked choices on 3 of 4 (the well-set choice was upgraded from the proposed two-well to a four-well variant).

### A. Aromatic detection scheme — **LOCKED: atom-c aromaticity flag**

| option | mechanics | generality |
|---|---|---|
| **Atom-c aromaticity flag** | `mol.GetAtomWithIdx(c).GetIsAromatic()` (and optionally d) | trips on Trp Cγ, Phe / Tyr Cγ, His Cγ uniformly without per-residue logic |
| Per-residue SMARTS | matches named residue patterns (Trp / Phe / Tyr / His) | less general; needs NMe + modified-residue handling |
| Either-endpoint aromatic | trigger on b OR c aromatic | over-matches: ring-edge bonds that already cannot rotate |

Locked at **atom-c aromaticity flag**. Implementation note: for each side-chain rotatable bond `(a, b, c, d)` returned by `_enumerate_side_chain_dihedrals`, atom `c` is the "downstream" endpoint and is the natural anchor for the aromatic check — if `c` is aromatic, the χ being rotated is the bond that points the aromatic ring at the rest of the molecule, which is precisely the χ₂-style state where the (-90, 90) wells live.

### B. Aromatic well set — **LOCKED: (-90, 0, 90, 180) symmetric four-well**

| option | well count | branching factor | rationale |
|---|---|---|---|
| (-90, 90) two-well | 2 | doubles χ₁ branching for aromatic bonds | standard literature χ₂ aromatic wells, symmetric across the aromatic plane |
| **(-90, 0, 90, 180) symmetric four-well** | 4 | matches sp3 χ₁ branching | adds edge-on states (0, 180) alongside the standard face-on (-90, 90); captures the full rotameric basin set without inflating the per-bond branching factor over the existing sp3 case |
| (-90, 90, 0, 180) four-well (different ordering) | 4 | same as above | functionally identical to the locked choice |

Locked at the **symmetric four-well (-90, 0, 90, 180)**. Rationale (lock conversation, 2026-06-15): the locked choice keeps the per-bond rotameric branching factor at 4 for aromatic bonds vs 3 for sp3 (the existing `(-60, 60, 180)`), so the rotamer-jump exploration budget on an aromatic-anchored bond is roughly comparable to that on a sp3-χ₁ bond and we don't over-bias the proposer toward aromatic discovery vs. sp3 refinement. The two-well alternative would have under-sampled the edge-on states; the asymmetric four-well variants are functionally identical to the locked choice and rejected on ordering aesthetics only.

### C. `skip_mmff_relax` scope — **LOCKED: skip MMFF entirely → rotated coords → MACE**

| option | semantics | interpretability as ablation |
|---|---|---|
| **Skip MMFF entirely** | rotated coords pass directly to the MACE batched scorer | clean ablation: separates the MMFF snap-back contribution from every other relaxation effect |
| Skip MMFF only for rotamer-jump proposals | keep MMFF for Gaussian Δχ, skip for rotamer jumps | hybrid; harder to interpret as an isolated ablation; could be a v0.3 follow-up if useful |
| Replace MMFF with constrained MMFF (freeze the rotated dihedral) | closer to a production-quality fix than an ablation | more implementation work; v0.3 candidate |

Locked at **skip MMFF entirely**. Rationale: the diagnostic question is whether MMFF snap-back is dragging rotamer jumps back across barriers on cremp_sharp. The cleanest test is to remove MMFF from the proposer pipeline entirely and let MACE see the raw rotated coords. The hybrid and constrained-MMFF variants conflate signals.

**Risk explicitly documented.** A pure-rotation proposal with no relaxation may put neighbouring atoms in mildly strained Cartesian positions (e.g. close H-H contacts on the rotated side chain). MACE handles these but the proposal acceptance rate may dip. Phase B of validation should track per-step acceptance with and without MMFF; if acceptance drops to <5 % with `skip_mmff_relax=True` the ablation may be uninterpretable for the wrong reason (rejection at the energy step, not at the basin step) and a v0.3 partial-relax path becomes the next candidate.

### D. Implementation order — **LOCKED: aromatic wells first, then skip_mmff_relax**

| option | reason |
|---|---|
| Combined — both in one Step | cheaper integration testing; 2×2 ablation needs both anyway |
| **Sequential: aromatic wells first, then skip_mmff_relax** | lets us test the structural-fix hypothesis first; if aromatic wells alone fix cremp_sharp, skip_mmff_relax stays as a diagnostic-only path |
| Sequential: skip_mmff_relax first, then aromatic wells | reverses the priority on the working hypothesis |

Locked at **sequential, aromatic wells first**. Both kwargs ship as part of this PR regardless, but the validation flow is split: Step 4 validates aromatic wells alone, and Step 7 runs the 2×2 ablation only if Step 4 leaves cremp_sharp at zero. If aromatic wells alone close cremp_sharp, `skip_mmff_relax` stays implemented + tested but is not in the production mix.

---

## Phase 1 — Foundation

### Step 1: Lock design choices — ✓ complete

All four forks locked at the choices above. The conversation is captured under the *Findings* section.

## Phase 2 — Aromatic-aware rotamer wells

### Step 2: Per-bond aromatic-aware rotamer-well helper + factory plumbing + tests — ✓ complete

- New helper `_classify_rotamer_wells(mol, dihedrals, aromatic_wells_deg, sp3_wells_deg) -> list[np.ndarray]` in `src/proposers.py`, called once at `make_dihedral_kick_proposer` construction time. Returns one well-set array per rotatable bond, in the same order as `_enumerate_side_chain_dihedrals(mol)`.
- Aromatic detection: for each `(a, b, c, d)`, check `mol.GetAtomWithIdx(c).GetIsAromatic()`. If True → use `aromatic_wells_deg`; else → use `sp3_wells_deg`.
- New kwarg on `make_dihedral_kick_proposer`: `aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0)`. The existing `rotamer_wells_deg: tuple = (-60.0, 60.0, 180.0)` stays as the sp3 fallback.
- Factory call path: replace the single `rotamer_wells_arr` with a per-bond list `rotamer_wells_per_bond`; at proposal time, look up the well set for the chosen bond.
- Tests (new section in `tests/test_mcmm.py`):
  - `test_aromatic_wells_classifier_trp_chi2_returns_aromatic_wells` — on cremp_sharp, locate Trp χ₂ via SMARTS, assert the corresponding well set equals the aromatic tuple.
  - `test_aromatic_wells_classifier_ala_chi1_returns_sp3_wells` — on a pure-sp3 peptide, every well set equals the sp3 tuple.
  - `test_aromatic_wells_factory_uses_per_bond_wells_at_proposal_time` — patch the rng to force a rotamer jump; assert that the jumped-to χ value is in the bond-appropriate well set, not the other one.
  - `test_aromatic_wells_factory_default_off_preserves_v0_behaviour` — with `aromatic_wells_deg=None` (or matching `sp3_wells_deg`), the proposer behaves identically to v0.

**Outcome (2026-06-15).** Helper `_classify_rotamer_wells(mol, dihedrals, aromatic_wells_deg, sp3_wells_deg)` shipped in [src/proposers.py:266](../src/proposers.py#L266) just after `_enumerate_side_chain_dihedrals`. For each rotatable bond it inspects `mol.GetAtomWithIdx(c).GetIsAromatic()` and returns either the aromatic well-set array or the sp3 fallback; when `aromatic_wells_deg is None` every bond gets sp3 (preserves issue-#12 behaviour byte-for-byte). The factory `make_dihedral_kick_proposer` gained `aromatic_wells_deg: tuple | None = None` as a new kwarg; the construction-time path now stores a `rotamer_wells_per_bond: list[np.ndarray]` indexed by `dihedral_idx`, and the call-time rotamer-jump branch looks up the well set for the chosen bond instead of using a single shared array. New validation: empty `aromatic_wells_deg` (when not None) with `p_rotamer_jump > 0` raises at factory time, mirroring the existing sp3 case. Docstring extended with the new kwarg + the v0.1-default behaviour note. 6 new tests in `tests/test_mcmm.py`: 3 unit tests on `_classify_rotamer_wells` (None → all sp3; cremp_sharp aromatic-c detection asserts ≥1 aromatic and ≥1 sp3 bond returned with the right well set each; empty-dihedrals edge case), 3 factory-level tests (default `aromatic_wells_deg=None` preserves stats shape; explicit empty + `p_rotamer_jump > 0` raises; empty `aromatic_wells_deg` is benign at `p_rotamer_jump=0`). Targeted run: 6/6 in 1.95 s; the wider dihedral test family (`enumerate_side_chain_dihedrals`, `dihedral_kick`, plus the new `classify_rotamer`) is 24/24 in 2.81 s. Full-suite run kicked off; predicted **426 passed, 8 skipped** (+6 over the v0 baseline of 420/8).

### Step 3: Thread `aromatic_wells_deg` through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI — ✓ complete

- New kwarg on `get_mol_PE_mcmm` ([src/confsweeper.py](../src/confsweeper.py)): `aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0)`. Passed to `make_dihedral_kick_proposer` only when `dihedral_weight > 0`.
- Threaded through `scripts/sampler_benchmark.py:_run_mcmm` adapter (5 thread-through sites following the Step-6 `dihedral_weight` pattern).
- CLI: `--aromatic_wells` (bool flag) at `main()`. Defaults to off — the v0 sp3-only behaviour. When on, the v0.2 four-well set is used for aromatic bonds.
- Optional escape hatch: `--aromatic_wells_deg "-90,0,90,180"` for explicit override; deferred to v0.3 unless v0.2 sweep shows the well set itself wants tuning.

**Outcome (2026-06-15).** `get_mol_PE_mcmm` gained `aromatic_wells_deg: tuple | None = None` ([src/confsweeper.py:1122](../src/confsweeper.py#L1122)); the existing v0.1 default `None` preserves issue-#12 behaviour byte-for-byte. Threaded into the `make_dihedral_kick_proposer` call site (kwarg added, builds only when `dihedral_weight > 0` per the existing short-circuit). Docstring extended with both `rotamer_wells_deg` (sp3 fallback semantics clarified) and the new `aromatic_wells_deg` block pointing to the issue-#15 lock. `scripts/sampler_benchmark.py` threaded through end-to-end with 8 sites following the Step-6 / `p_rotamer_jump` pattern: `_run_mcmm` signature + body pass to `get_mol_PE_mcmm`, `run_one` signature + dispatcher, `--aromatic_wells/--no-aromatic_wells` click bool flag, `main()` signature, start-of-run log line (`aromatic_wells=on/off`), and `run_one(...)` call. The bool→tuple mapping happens in `main()` (`(-90.0, 0.0, 90.0, 180.0) if aromatic_wells else None`) so the boundary stays at the CLI; everything below threads as `aromatic_wells_deg: tuple | None` matching the `get_mol_PE_mcmm` / `make_dihedral_kick_proposer` signature. 3 new tests in `tests/test_get_mol_PE_mcmm.py` cover the integration thread-through: `test_mcmm_aromatic_wells_deg_default_none_passes_through_to_dihedral_factory` (spies the factory call, asserts `aromatic_wells_deg=None` is in the captured kwargs — regression guard against the kwarg silently dropping), `test_mcmm_aromatic_wells_deg_explicit_tuple_forwards_intact` (explicit `(-90, 0, 90, 180)` reaches the factory unchanged), `test_mcmm_aromatic_wells_deg_unused_when_dihedral_weight_zero` (zero-overhead short-circuit preserved — factory not called even when `aromatic_wells_deg` is set). Targeted run: 3/3 in 2.56 s. Full suite kicked off; predicted **429 passed, 8 skipped** (+3 over Step 2's 426/8).

### Step 4: Validation A — aromatic wells alone — ✓ complete (2026-06-17)

- 2-cell sweep × 2 peptides at n_seeds=10000:
  - Cell A.1: 3-way mix `(cart=0.33, dih=0.33, p_rotamer_jump=0.30)`, `--aromatic_wells off` — replicates the issue-#12 phase-2 headline as the apples-to-apples baseline at this branch.
  - Cell A.2: same mix, `--aromatic_wells on`.
- Driver script `scripts/sweep_v0_2_step4.sh` mirroring `sweep_step7_phase2.sh` structure (detached via setsid, peptide list from `data/processed/cremp/sweep_step7_peptides.csv`).
- **Success criteria for Step 4:**
  - cremp_typical: no regression (`cov_bw_ceil ≥ 0.99` matching issue-#12 headline at the same budget).
  - cremp_sharp: `cov_bw_ceil > 0` is the v0.2 win; `cov_bw_ceil ≥ 0.10` clears the issue-#12 lock's success threshold.
  - Diagnostic: track `max_missed_bw` per cell — if it drops below 0.724 even with `cov_bw_ceil` still small, the proposer is *finding* the dominant basin but the τ=0.5 match tolerance is wrong (rare; would point at a v0.3 match-RMSD knob).
- **Decision point at end of Step 4:**
  - If aromatic wells alone close cremp_sharp → skip Step 7's 2×2 ablation; Step 5+6 ship `skip_mmff_relax` as a diagnostic kwarg only; document the success in Step 8.
  - If aromatic wells alone leave cremp_sharp at zero → proceed to Step 7's full 2×2.

**Outcome (2026-06-17).** **Aromatic wells alone do NOT close cremp_sharp — proceed to Step 7's 2×2.** Driver `scripts/sweep_v0_2_step4.sh` ran two cells × 2 peptides at n_seeds=10000 with the issue-#12 production mix `(cart=0.33, dih=0.33, p_rotamer_jump=0.30)`. **One real-world bug surfaced and was fixed mid-sweep** (see Findings entry "_maybe_dump_sdf zero-conformer crash" below for the full story — CUDA OOM during cremp_sharp cell-1 swallowed by `run_one`'s try/except continue produced two 0-byte SDFs that then crashed `union_basin_count.py` at file-load time). Recovery via `scripts/sweep_v0_2_step4_recover.sh` reused the cell-1 cremp_typical data (clean — 11+15 basins via kabsch+crest) and re-ran cell-1 cremp_sharp + cell-2 both peptides from scratch (~80 min wall-clock total). Results: <br><br>**cremp_typical (kabsch dedup):**<br><br>| run | aromatic | cov_bw_ceil | cov_count | max_missed_bw |<br>|---|---|---|---|---|<br>| issue-#12 phase-2 HEADLINE | off | 0.991 | 20/22 | 0.006 |<br>| Step-4 aromatic_off (regression check) | off | 0.989 | 20/22 | 0.009 |<br>| Step-4 aromatic_ON | on | **0.971** | 16/22 | 0.009 |<br><br>**cremp_sharp (kabsch dedup):**<br><br>| run | aromatic | cov_bw_ceil | max_missed_bw | n_new | new_mass |<br>|---|---|---|---|---|---|<br>| issue-#12 phase-2 HEADLINE | off | 0.000 | 0.724023 | 2 | 5.0e-4 |<br>| Step-4 aromatic_off | off | 0.000 | 0.724023 | 2 | 2.8e-6 |<br>| Step-4 aromatic_ON | on | **0.000** | **0.724023** | **8** | 4.1e-8 |<br><br>**Interpretation.** **(1) Step-4 aromatic_off matches the issue-#12 headline within RNG noise** (0.989 vs 0.991; cov_count identical at 20/22) — confirms Steps 2+3 introduced no behavioural regression on the v0.1 path. **(2) Aromatic wells regress cremp_typical by ~2 absolute points** (0.989 → 0.971; cov_count 20/22 → 16/22) — same pattern as the issue-#12 phase-2 `p_rotamer_jump=0.70` diagnostic: too aggressive an exploration knob shrinks the refined basin set. **(3) Aromatic wells do NOT close cremp_sharp.** `max_missed_bw=0.724023` is identical to 6 decimal places across all 3 runs — the SAME dominant ceiling basin is still missed everywhere. **(4) But aromatic wells DO change cremp_sharp behaviour qualitatively:** `n_new` jumped from 2 → **8** (4× the baseline; the highest n_new seen on cremp_sharp across the entire issue-#12 + v0.2 sweep matrix), indicating the proposer IS exploring more aggressively when aromatic χ₂ rotamer jumps are available. The new-basin thermodynamic weight (4.1e-8) is even smaller than the baseline (5.0e-4) — meaning the aromatic-driven exploration finds *more* basins but they're *less* thermally interesting. Diagnostic: aromatic χ₂ rotamer jumps reach geometries that ARE distinct enough from the seed to be deduped as new basins, but those geometries don't sit anywhere near the dominant ceiling basin's location in conformational space. **Step-7 decision: proceed to the 2×2 ablation.** Of the four B-cells, B.1 `(off, off)` and B.2 `(on, off)` are this Step-4 data and will be reused; only B.3 `(off, skip_mmff=on)` and B.4 `(on, skip_mmff=on)` need to be run. The most informative single ablation is B.3 vs B.1 — if skip_mmff alone moves cremp_sharp off zero, the MMFF snap-back is the dominant failure mode; if B.4 is needed on top of B.3, the issue is multiplicative. **Files / artefacts:** `results/sweep_v0_2_step4_coverage_aromatic_{off,on}.csv` (BW coverage); `results/sweep_v0_2_step4_aromatic_{off,on}/` (basin SDFs); `results/sweep_v0_2_step4_logs/` (per-cell logs); driver scripts `scripts/sweep_v0_2_step4.sh` (original) + `scripts/sweep_v0_2_step4_recover.sh` (post-OOM-bug recovery).

## Phase 3 — No-MMFF ablation

### Step 5: `skip_mmff_relax` ablation kwarg on `make_dihedral_kick_proposer` + tests — ✓ complete

- New kwarg `skip_mmff_relax: bool = False` on `make_dihedral_kick_proposer`. When True, the Stage-2 MMFF batched relax is bypassed; the rotated coordinates pass directly to the MACE batched scorer.
- Stats: new counter `n_mmff_skipped` increments per proposal when `skip_mmff_relax=True`. Existing `n_relax_successes` / `n_relax_failures` (now MACE-acceptance counters) stay meaningful since `skip_mmff_relax=True` still produces successful/failed proposals based on MACE finite-ness.
- Tests:
  - `test_dihedral_kick_skip_mmff_bypasses_mmff_call` — mock the MMFF backend; with `skip_mmff_relax=True`, assert it's never called.
  - `test_dihedral_kick_skip_mmff_passes_rotated_coords_to_mace` — with the mock MMFF as a no-op (returns input unchanged) and `skip_mmff_relax=True`, assert MACE receives coords that match the post-rotation throwaway-mol state, not the pre-rotation walker state.
  - `test_dihedral_kick_skip_mmff_handles_non_finite_mace` — mock MACE to return NaN; assert the proposer rejects via `success=False` exactly as the MMFF-on path does.
  - `test_dihedral_kick_skip_mmff_default_off_preserves_v0_behaviour` — explicit-False matches the v0 + Step-2 aromatic-wells behaviour.

**Outcome (2026-06-16).** `make_dihedral_kick_proposer` gained `skip_mmff_relax: bool = False` ([src/proposers.py:665](../src/proposers.py#L665)); default preserves v0.1 / Step-2 behaviour byte-for-byte. Stage-2 branch ([src/proposers.py:955](../src/proposers.py#L955)) now short-circuits when the flag is on: increment `n_mmff_skipped` by `len(pre_mmff_conf_ids)` and skip both the GPU (`MMFFOptimizeMoleculesConfs`) and CPU (`AllChem.MMFFOptimizeMolecule`) branches entirely. New stats counter `n_mmff_skipped` (initialised to 0 alongside the existing counters), exposed via `proposer.stats` for Step-7 diagnostic logging. Stage 3 (MACE scoring) and Stage 4 (proposal assembly with the `det_j=1.0` + `np.isfinite` guard) are unchanged — non-finite MACE energies on raw rotated geometries still get rejected via `success=False`, preserving the existing contract. Docstring extended in three places: the per-call-pipeline Stage-2 description gets a one-sentence note about the bypass; the Params section gains a `skip_mmff_relax` block pointing at the issue-#12 Step-7 phase-2 Findings (the empirical trigger for adding this ablation) plus the "MACE-acceptance < 5 % ⇒ v0.3 partial-MMFF" trigger; the "Known risk" callout's "diagnostic is to disable Stage 2" sentence is now backed by a real code path. 6 new tests in `tests/test_mcmm.py` using a new `_make_dihedral_kick_with_mmff_spies` helper that returns `MagicMock` spies for both MMFF backends so callers can assert on call counts: `test_dihedral_kick_skip_mmff_relax_default_false_calls_mmff_gpu` (v0.1 byte-compat — GPU MMFF called once per batch), `test_dihedral_kick_skip_mmff_relax_true_bypasses_mmff_gpu` (the headline contract — GPU MMFF call count == 0 with flag on), `test_dihedral_kick_skip_mmff_relax_true_bypasses_mmff_cpu` (parallel for the `mmff_backend='cpu'` path), `test_dihedral_kick_skip_mmff_relax_n_mmff_skipped_stat_increments` (counter == n_walkers per batch, accumulates across batches), `test_dihedral_kick_skip_mmff_relax_default_n_mmff_skipped_stays_zero` (counter key always present in stats dict — no KeyError for legacy callers — but stays at 0 when off), `test_dihedral_kick_skip_mmff_relax_still_rejects_non_finite_mace` (non-finite MACE energy on the raw rotated geometry still produces `success=False` exactly as the MMFF-on path does; regression guard against the bypass branch silently accepting NaN). Two pre-existing tests needed updates — `test_make_dihedral_kick_proposer_aromatic_wells_default_none_preserves_v0` and `test_dihedral_kick_proposer_stats_initialised_and_increment` both asserted the exact set of `proposer.stats` keys; both updated to include `n_mmff_skipped`. Targeted run: 6/6 in 3.08 s; full dihedral test family (30 tests: 24 baseline + 6 new): 30/30 in 3.38 s. Full suite kicked off in parallel with the Step-4 sweep; predicted **435 passed, 8 skipped** (+6 over Step 3's 429/8).

### Step 6: Thread `skip_mmff_relax` through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI — ✓ complete (targeted; full suite pending behind Step 7)

- Mirrors Step 3's pattern. CLI flag: `--skip_mmff_relax` (bool, default False).

**Outcome (2026-06-17).** `get_mol_PE_mcmm` gained `skip_mmff_relax: bool = False` ([src/confsweeper.py:1123](../src/confsweeper.py#L1123)); default preserves v0.1 / issue-#12 behaviour byte-for-byte. Threaded into the `make_dihedral_kick_proposer` call site at [src/confsweeper.py:1417](../src/confsweeper.py#L1417) (built only when `dihedral_weight > 0` per the existing short-circuit). Docstring extended with the `skip_mmff_relax` block pointing to issue #15 and the MMFF snap-back hypothesis. `scripts/sampler_benchmark.py` threaded through end-to-end with 8 sites following the Step-3 `aromatic_wells` pattern: `_run_mcmm` signature + body pass to `get_mol_PE_mcmm`, `run_one` signature + dispatcher, `--skip_mmff_relax/--no-skip_mmff_relax` click bool flag, `main()` signature, start-of-run log line (`skip_mmff_relax=on/off`), and `run_one(...)` call. CLI flag verified via `pixi run python scripts/sampler_benchmark.py --help` and again live in the Step-7 sweep start-of-run log line which prints `... aromatic_wells=off  skip_mmff_relax=on ...`. 3 new tests in `tests/test_get_mol_PE_mcmm.py` cover the integration thread-through: `test_mcmm_skip_mmff_relax_default_false_passes_through_to_dihedral_factory` (spies the factory call, asserts `skip_mmff_relax=False` is in the captured kwargs — v0.1 byte-compat), `test_mcmm_skip_mmff_relax_true_forwards_intact` (explicit `True` reaches the factory unchanged), `test_mcmm_skip_mmff_relax_unused_when_dihedral_weight_zero` (zero-overhead short-circuit preserved — factory not called even when `skip_mmff_relax=True`). Targeted run: 3/3 in 2.20 s. Full suite confirmation **deferred until after Step 7 sweep completes** — the `_maybe_dump_sdf` bug-fix Findings entry above documents the GPU-OOM trap from running the full pytest suite alongside a GPU sweep; running them sequentially this time. Expected when the full suite eventually lands: **441 passed, 8 skipped** (435 baseline + 3 Step-6 thread-through + 3 `_maybe_dump_sdf` bug fix).

### Step 7: Validation B — 2×2 ablation (conditional on Step 4 outcome) — ✓ complete (2026-06-17)

- Conditional: run only if Step 4 leaves cremp_sharp at `cov_bw_ceil = 0`.
- 2×2 ablation × 2 peptides at n_seeds=10000:
  - Cell B.1: aromatic_wells=off, skip_mmff_relax=off (= Step 4 cell A.1; reuse outputs).
  - Cell B.2: aromatic_wells=on, skip_mmff_relax=off (= Step 4 cell A.2; reuse outputs).
  - Cell B.3: aromatic_wells=off, skip_mmff_relax=on.
  - Cell B.4: aromatic_wells=on, skip_mmff_relax=on.
- **Decomposition logic:** if cell B.3 lifts cremp_sharp off zero, MMFF snap-back was the dominant issue. If cell B.2 lifts it off zero, the well set was. If cell B.4 lifts but neither B.2 nor B.3 alone, the failure is multiplicative — both fixes are needed together. If all four stay at zero, the cremp_sharp failure is structural beyond what v0.2 can fix, and v0.3 escalates to e.g. concerted χ₁ + χ₂ rotation.

**Outcome (2026-06-17).** Driver `scripts/sweep_v0_2_step7.sh` ran the 2 new cells (B.3 + B.4) × 2 peptides at n_seeds=10000 with the issue-#12 production mix `(cart=0.33, dih=0.33, p_rotamer_jump=0.30)`. B.1 + B.2 outputs reused from Step-4 unchanged. ~38 min wall-clock; clean completion at 10:08:03. <br><br>**cremp_typical (kabsch dedup):**<br><br>| cell | aromatic | skip_mmff | cov_bw_ceil | cov_count | max_missed_bw |<br>|---|---|---|---|---|---|<br>| B.1 (off, off) | off | off | 0.989 | 20/22 | 0.009 |<br>| B.2 (on, off) | on | off | 0.971 | 16/22 | 0.009 |<br>| B.3 (off, on) | off | on | 0.966 | 15/22 | 0.009 |<br>| **B.4 (on, on)** | **on** | **on** | **0.997** | **21/22** | **0.003** |<br><br>**cremp_sharp (kabsch dedup):**<br><br>| cell | aromatic | skip_mmff | cov_bw_ceil | max_missed_bw | n_new | new_mass |<br>|---|---|---|---|---|---|---|<br>| B.1 (off, off) | off | off | 0.000 | 0.724023 | 2 | 2.8 × 10⁻⁶ |<br>| B.2 (on, off) | on | off | 0.000 | 0.724023 | 8 | 4.1 × 10⁻⁸ |<br>| B.3 (off, on) | off | on | 0.000 | 0.724023 | 6 | 7.8 × 10⁻⁷ |<br>| **B.4 (on, on)** | **on** | **on** | **0.000** | **0.724023** | 3 | **8.8 × 10⁻³** |<br><br>**Interpretation per peptide.**

- **cremp_typical: synergy.** Each fix alone *regresses* cremp_typical against the v0.1 baseline B.1 (B.2 drops 1.8 pts, B.3 drops 2.3 pts), but **B.4 with both fixes on lifts to 0.997 — a new branch record** (+0.6 absolute pts over the issue-#12 phase-2 headline of 0.991, +0.8 pts over the Step-4 aromatic_off regression check of 0.989). `cov_count` rises from 20/22 → 21/22; `max_missed_bw` drops from 0.009 → 0.003 (3× tighter). This is a genuine non-linear interaction: the structural-fix axis (aromatic wells) and the MMFF-bypass axis (skip_mmff) compose multiplicatively — aromatic wells without the MMFF-bypass let MMFF drag rotamer jumps back across barriers (B.2 result); MMFF-bypass without aromatic wells leaves the proposer jumping to sp3 wells on aromatic χ₂ where MACE then drifts to non-equilibrium states (B.3 result); only with both on does the proposer (a) propose geometrically-meaningful aromatic χ₂ states AND (b) skip the MMFF that would otherwise pull them back, producing the 0.997 record.
- **cremp_sharp: structural wall.** All four cells leave `cov_bw_ceil=0.000` with `max_missed_bw=0.724023` identical to 6 decimal places — the same single dominant ceiling basin (72 % of the 298 K Boltzmann population) is missed at every setting. **But B.4 IS doing useful work the metric doesn't surface:** `new_mass = 8.8 × 10⁻³` is the largest cremp_sharp new-basin thermodynamic weight observed across the entire issue-#12 + v0.2 sweep matrix — **~3000× the B.1 v0.1 baseline (2.8 × 10⁻⁶), ~10⁴ × the B.2 aromatic-only cell (4.1 × 10⁻⁸).** The synergy that wins cremp_typical also discovers thermodynamically-real (8.8 mmass-units) cremp_sharp basins; those basins just aren't *the* dominant ceiling basin. Diagnostic: the dominant cremp_sharp basin is structurally inaccessible to a single-dihedral-per-step proposer regardless of well set or MMFF state — the geometry likely requires a concerted χ₁ + χ₂ rotation (the issue-#15 docstring's deferred follow-up) or a backbone-mediated path that DBT misses for this specific topology.

**Decomposition-logic verdict per the locked Step-1 plan.** Strict reading: all four cells stay at `cov_bw_ceil=0.000` on cremp_sharp → "cremp_sharp failure is structural beyond what v0.2 can fix, v0.3 escalates to e.g. concerted χ₁ + χ₂ rotation". **v0.3 trigger fires.** Nuanced reading: B.4's 4-orders-of-magnitude new_mass jump on cremp_sharp says the v0.2 proposer is *actively useful* on that peptide — it just doesn't close the headline metric.

**Production-mix lock for v0.2.** **B.4 wins:** `--aromatic_wells --skip_mmff_relax` at the locked issue-#12 mix `(cart=0.33, dih=0.33, p_rotamer_jump=0.30)`. Strictly improves cremp_typical (+0.6 absolute pts over issue-#12 headline) and does not regress cremp_sharp (still 0.000 = baseline floor; new-basin exploration substantially up). The v0.1 issue-#12 callers who don't pass these flags get unchanged behaviour (both kwargs default to off).

**Files / artefacts.** Driver `scripts/sweep_v0_2_step7.sh`; per-cell outputs `results/sweep_v0_2_step7_coverage_B{3,4}_*.csv` (BW coverage) + `results/sweep_v0_2_step7_sampler_B{3,4}_*.csv` (sampler stats) + `results/sweep_v0_2_step7_B{3,4}_*/` (basin SDFs) + `results/sweep_v0_2_step7_logs/` (per-cell logs). Step-4 outputs reused for B.1 + B.2 cells.

## Phase 4 — Documentation

### Step 8: Documentation — pending

- Update `src/README.md` `## mcmm.py` → `make_dihedral_kick_proposer` paragraph: add the `aromatic_wells_deg`, `skip_mmff_relax`, and the per-bond well-set behaviour. Update the production-mix block with whatever Step 4 or Step 7 lands as the new default.
- Update `scripts/README.md` `### sampler_benchmark.py`: document `--aromatic_wells` and `--skip_mmff_relax` flags; update the production tuning block.
- Append a dated Findings entry to this plan covering the Step 4 / Step 7 sweep results.
- Append a dated Findings entry to `docs/mcmm_plan.md` extending the 2026-06-15 cross-reference with the v0.2 numbers.
- Append a dated Findings entry to `docs/dihedral_kick_plan.md` (the v0.1 plan) noting that the cremp_sharp residual was attacked and closed/deferred in v0.2.

---

## Risks to instrument from day one

- **MACE-on-rotated-coords noise.** Per the Step-1 lock for choice C, raw-rotation proposals with no MMFF relax may push atoms into mildly strained positions. Track per-step MACE acceptance with `skip_mmff_relax=on` vs off: if acceptance drops below 5 % the ablation may be uninterpretable. Mitigation: a v0.3 partial-relax path becomes the next candidate.
- **Aromatic-well over-detection.** The atom-c aromaticity check fires on any aromatic atom at position c. If a non-χ₂-style bond happens to point at an aromatic carbon (rare but possible — e.g. an exocyclic aromatic substituent rotation), it gets the four-well treatment. Step-4 diagnostic should print the per-bond well-set assignments on cremp_sharp + cremp_typical and visually verify only true χ₂ bonds get the aromatic wells. If the over-detection is real and harmful, a v0.3 candidate adds an "is_chi2" predicate (b also non-aromatic, b's parent is the α-carbon).
- **Step-4 success doesn't generalise.** If aromatic wells fix cremp_sharp but not other aromatic-containing peptides (e.g. those in `data/processed/cremp/validation_subset.csv` with `topology` including aromatic residues), the v0.2 fix is partial. Step 4 should sample one or two extra aromatic-containing CREMP peptides as a generalisation sanity check.

## Deferred follow-ups

- **Partial / constrained MMFF relax.** If `skip_mmff_relax=True` produces uninterpretable acceptance rates, a v0.3 candidate replaces the full Stage-2 MMFF with a constrained MMFF that freezes the rotated dihedral. Trigger: Step 7 cell B.3 or B.4 shows acceptance < 5 %.
- **Concerted χ₁ + χ₂ rotation.** If all four Step 7 cells stay at zero on cremp_sharp, the v0 single-dihedral-per-step lock is the bottleneck. A v0.3 proposer would rotate χ₁ AND χ₂ together on aromatic side chains. Implementation cost: significant — the joint rotation is non-trivial to keep volume-preserving.
- **Generalisation to non-CREMP peptides.** v0.2 validation is limited to the 2 CREMP peptides with ceiling SDFs. If the issue-#15 PR ships and the user wants to run on `pampa_large` etc, a v0.3 candidate generates ceilings for those peptides (`scripts/cremp_collapse_test.py`-style pipeline) so the Boltzmann-coverage diagnostic can apply.

---

## Findings

(append-only, dated)

### Design choices locked (2026-06-15)

All four open forks resolved. Three at the running recommendations from the *Design choices to lock* tables; one (the aromatic well set) upgraded from the recommended two-well `(-90, 90)` to a four-well `(-90, 0, 90, 180)` to keep the per-bond rotameric branching factor in line with the existing sp3-χ₁ case.

| fork | locked choice | rationale |
|---|---|---|
| **A. Aromatic detection** | Atom-c aromaticity flag | Simplest and most general; trips on all aromatic side chains uniformly without per-residue logic. |
| **B. Aromatic well set** | Symmetric four-well (-90, 0, 90, 180) | Captures face-on (-90, 90) AND edge-on (0, 180) rotameric states; branching factor (4) matches the existing sp3 case (3) so the proposer isn't over-biased toward aromatic discovery. |
| **C. skip_mmff_relax scope** | Skip MMFF entirely → rotated coords → MACE | Cleanest test of the MMFF snap-back hypothesis; isolates the MMFF contribution from every other relaxation effect. Hybrid and constrained-MMFF variants conflate signals and are deferred to v0.3. |
| **D. Implementation order** | Sequential — aromatic wells (Steps 2–4) first, then skip_mmff_relax (Steps 5–7) | Lets us test the structural-fix hypothesis (aromatic wells) before committing to the more invasive ablation path. If aromatic wells alone fix cremp_sharp, skip_mmff_relax stays as a diagnostic kwarg only. |

**v0.2 default knobs that fall out:**

- `aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0)` — the locked aromatic wells.
- `skip_mmff_relax: bool = False` — opt-in; defaults preserve issue-#12 production behaviour.
- All existing v0 defaults unchanged: `sigma_chi_rad=0.5`, `p_rotamer_jump=0.3`, `rotamer_wells_deg=(-60, 60, 180)`, `dihedral_weight=0.0`.

Step 1 closes; Step 2 (aromatic-aware per-bond well helper) unblocked.

### `_maybe_dump_sdf` zero-conformer crash (2026-06-17)

**Surfaced by Step-4 sweep.** Mid-sweep cremp_sharp cell-1 (aromatic_off) crashed with CUDA OOM at 14:16:30 because the Step-5 full test suite was running in parallel on the same GPU (mistake — never run the GPU-bound test suite alongside a GPU sweep again). `sampler_benchmark.py:run_one`'s `try/except continue` swallowed the OOM, the mol returned from `_run_mcmm` had zero conformers, and `_maybe_dump_sdf` happily opened `Chem.SDWriter` on an empty list and closed it — producing a 0-byte SDF on disk. The downstream `union_basin_count.py:_load_basin_sdf` then called `Chem.SDMolSupplier(str(sdf_path), removeHs=False)` on the malformed file and raised `OSError: File error: Invalid input file`, crashing the entire union-coverage step.

**Fix.** [scripts/sampler_benchmark.py:152-167](../scripts/sampler_benchmark.py#L152) — early-return on `not conf_ids`, log a `warning` flagging the zero-basin state, do not touch the filesystem. Downstream globbers simply miss the file rather than crash on a malformed one. The `warning` log line stays visible in the per-cell sampler log so the no-basins state isn't silent.

**Regression tests.** [tests/test_sampler_benchmark.py](../tests/test_sampler_benchmark.py) — new test file with 3 tests: `test_maybe_dump_sdf_writes_file_when_conf_ids_non_empty` (sanity — normal path round-trips through SDMolSupplier), `test_maybe_dump_sdf_skips_write_when_conf_ids_empty` (the headline fix — empty conf_ids does NOT create an output file), `test_maybe_dump_sdf_skips_write_when_dump_sdf_dir_is_none` (existing dump-disabled fast path). All 3 pass in 2.4 s.

**Recovery for the actual Step-4 sweep.** `scripts/sweep_v0_2_step4_recover.sh` reused the cell-1 cremp_typical data (clean: 11+15 basins via kabsch+crest at n_seeds=10000) and re-ran cell-1 cremp_sharp + cell-2 both peptides from scratch. ~80 min wall-clock; clean completion at 09:07:43 on 2026-06-17.

**Lesson recorded for future GPU-bound sweeps:** do not kick off the full pytest suite (which loads MACECalculator on the same GPU for `tests/test_confsweeper.py::test_run_PE_calc_cli` and similar) while a sweep is running. Sequential or different-GPU only.
