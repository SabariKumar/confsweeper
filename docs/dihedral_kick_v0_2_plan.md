# Side-chain dihedral-kick proposer v0.2

Branch: `15-mmff-ablation-aromatic-aware-proposer`. Implements issue #15. Follow-up to issue #12 (`docs/dihedral_kick_plan.md`).

## 20260615

This document is the working design for v0.2 of the side-chain dihedral-kick proposer. Motivated directly by the issue-#12 Step-7 phase 2 Findings (2026-06-15) in [dihedral_kick_plan.md](dihedral_kick_plan.md): the v0 proposer hit **`coverage_bw_ceiling = 0.991` on cremp_typical** (+0.16 over the issue-#10 baseline) but flatlined at **`0.000` on cremp_sharp** at every tested mix, with `max_missed_bw = 0.724023` identical to 7 decimal places across all 6 phase-1 + phase-2 cells. The same single dominant ceiling basin (72 % of the 298 K Boltzmann population) is missed everywhere. The phase-2 diagnostic at `p_rotamer_jump=0.70` showed `new_mass` scaling ~400× over the phase-1 baseline — confirming the proposer IS exploring much more aggressively at higher rotamer-jump rates — but the new basins are not the *one* basin that dominates the ceiling distribution. Two structural hypotheses survive and are armed as v0.2 work: per-bond aromatic-aware rotamer-well sets, and a no-MMFF ablation path.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Lock four design forks | ✓ complete |
| 2 | Per-bond aromatic-aware rotamer-well helper + factory plumbing + tests | pending |
| 3 | Thread `aromatic_wells` kwarg through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI | pending |
| 4 | Validation A: aromatic wells alone on cremp_sharp + cremp_typical at n_seeds=10000 | pending |
| 5 | `skip_mmff_relax` ablation kwarg on `make_dihedral_kick_proposer` + tests | pending |
| 6 | Thread `skip_mmff_relax` through `get_mol_PE_mcmm` + CLI | pending |
| 7 | Validation B: 2×2 ablation matrix `(aromatic_wells ∈ {off, on}) × (skip_mmff_relax ∈ {off, on})` on cremp_sharp + cremp_typical at n_seeds=10000 | pending |
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

### Step 2: Per-bond aromatic-aware rotamer-well helper + factory plumbing + tests — pending

- New helper `_classify_rotamer_wells(mol, dihedrals, aromatic_wells_deg, sp3_wells_deg) -> list[np.ndarray]` in `src/proposers.py`, called once at `make_dihedral_kick_proposer` construction time. Returns one well-set array per rotatable bond, in the same order as `_enumerate_side_chain_dihedrals(mol)`.
- Aromatic detection: for each `(a, b, c, d)`, check `mol.GetAtomWithIdx(c).GetIsAromatic()`. If True → use `aromatic_wells_deg`; else → use `sp3_wells_deg`.
- New kwarg on `make_dihedral_kick_proposer`: `aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0)`. The existing `rotamer_wells_deg: tuple = (-60.0, 60.0, 180.0)` stays as the sp3 fallback.
- Factory call path: replace the single `rotamer_wells_arr` with a per-bond list `rotamer_wells_per_bond`; at proposal time, look up the well set for the chosen bond.
- Tests (new section in `tests/test_mcmm.py`):
  - `test_aromatic_wells_classifier_trp_chi2_returns_aromatic_wells` — on cremp_sharp, locate Trp χ₂ via SMARTS, assert the corresponding well set equals the aromatic tuple.
  - `test_aromatic_wells_classifier_ala_chi1_returns_sp3_wells` — on a pure-sp3 peptide, every well set equals the sp3 tuple.
  - `test_aromatic_wells_factory_uses_per_bond_wells_at_proposal_time` — patch the rng to force a rotamer jump; assert that the jumped-to χ value is in the bond-appropriate well set, not the other one.
  - `test_aromatic_wells_factory_default_off_preserves_v0_behaviour` — with `aromatic_wells_deg=None` (or matching `sp3_wells_deg`), the proposer behaves identically to v0.

### Step 3: Thread `aromatic_wells_deg` through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI — pending

- New kwarg on `get_mol_PE_mcmm` ([src/confsweeper.py](../src/confsweeper.py)): `aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0)`. Passed to `make_dihedral_kick_proposer` only when `dihedral_weight > 0`.
- Threaded through `scripts/sampler_benchmark.py:_run_mcmm` adapter (5 thread-through sites following the Step-6 `dihedral_weight` pattern).
- CLI: `--aromatic_wells` (bool flag) at `main()`. Defaults to off — the v0 sp3-only behaviour. When on, the v0.2 four-well set is used for aromatic bonds.
- Optional escape hatch: `--aromatic_wells_deg "-90,0,90,180"` for explicit override; deferred to v0.3 unless v0.2 sweep shows the well set itself wants tuning.

### Step 4: Validation A — aromatic wells alone — pending

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

## Phase 3 — No-MMFF ablation

### Step 5: `skip_mmff_relax` ablation kwarg on `make_dihedral_kick_proposer` + tests — pending

- New kwarg `skip_mmff_relax: bool = False` on `make_dihedral_kick_proposer`. When True, the Stage-2 MMFF batched relax is bypassed; the rotated coordinates pass directly to the MACE batched scorer.
- Stats: new counter `n_mmff_skipped` increments per proposal when `skip_mmff_relax=True`. Existing `n_relax_successes` / `n_relax_failures` (now MACE-acceptance counters) stay meaningful since `skip_mmff_relax=True` still produces successful/failed proposals based on MACE finite-ness.
- Tests:
  - `test_dihedral_kick_skip_mmff_bypasses_mmff_call` — mock the MMFF backend; with `skip_mmff_relax=True`, assert it's never called.
  - `test_dihedral_kick_skip_mmff_passes_rotated_coords_to_mace` — with the mock MMFF as a no-op (returns input unchanged) and `skip_mmff_relax=True`, assert MACE receives coords that match the post-rotation throwaway-mol state, not the pre-rotation walker state.
  - `test_dihedral_kick_skip_mmff_handles_non_finite_mace` — mock MACE to return NaN; assert the proposer rejects via `success=False` exactly as the MMFF-on path does.
  - `test_dihedral_kick_skip_mmff_default_off_preserves_v0_behaviour` — explicit-False matches the v0 + Step-2 aromatic-wells behaviour.

### Step 6: Thread `skip_mmff_relax` through `get_mol_PE_mcmm` + `scripts/sampler_benchmark.py` CLI — pending

- Mirrors Step 3's pattern. CLI flag: `--skip_mmff_relax` (bool, default False).

### Step 7: Validation B — 2×2 ablation (conditional on Step 4 outcome) — pending

- Conditional: run only if Step 4 leaves cremp_sharp at `cov_bw_ceil = 0`.
- 2×2 ablation × 2 peptides at n_seeds=10000:
  - Cell B.1: aromatic_wells=off, skip_mmff_relax=off (= Step 4 cell A.1; reuse outputs).
  - Cell B.2: aromatic_wells=on, skip_mmff_relax=off (= Step 4 cell A.2; reuse outputs).
  - Cell B.3: aromatic_wells=off, skip_mmff_relax=on.
  - Cell B.4: aromatic_wells=on, skip_mmff_relax=on.
- **Decomposition logic:** if cell B.3 lifts cremp_sharp off zero, MMFF snap-back was the dominant issue. If cell B.2 lifts it off zero, the well set was. If cell B.4 lifts but neither B.2 nor B.3 alone, the failure is multiplicative — both fixes are needed together. If all four stay at zero, the cremp_sharp failure is structural beyond what v0.2 can fix, and v0.3 escalates to e.g. concerted χ₁ + χ₂ rotation.

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
