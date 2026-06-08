# Side-chain dihedral-kick proposer

Branch: `12-side-chain-dihedral-kick-proposer` (proposed). Implements issue #12.

## 20260522

This document is the working design for a third MCMM proposer — a **side-chain dihedral rotation kick** — added alongside DBT concerted rotation and the GOAT-style Cartesian kick. Motivated directly by the 2026-05-21 Boltzmann-coverage Findings entry in [mcmm_plan.md](mcmm_plan.md): on cremp_sharp the MCMM basin set sits geometrically 3+ Å from every CREMP ceiling basin, the dominant ceiling basin holds 72 % of the 298 K Boltzmann population, and `coverage_bw_ceiling = 0` at every tested match tolerance (τ ∈ {0.125, 0.5, 1.0} Å). The failure mode points at NMe-Trp χ₁ / χ₂ rotamers — side-chain dihedral states that DBT (backbone-only) cannot reach and that small Cartesian kicks + MMFF relax cannot cross because the indole-ring chi barriers are ≈10–15 kcal/mol. The remedy is a proposer that explicitly rotates one side-chain dihedral per step, large enough to cross a rotameric barrier when sampled to.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Lock the four open design choices (see *Design choices to lock* below) | ✓ complete |
| 2 | Refactor: extract proposer factories from `src/mcmm.py` to a new `src/proposers.py` | ✓ complete |
| 3 | Side-chain rotatable-bond enumeration (`_enumerate_side_chain_dihedrals`) | ✓ complete |
| 4 | `make_dihedral_kick_proposer` factory + tests | pending |
| 5 | Extend `make_composite_proposer` to n-way routing across (DBT, Cartesian, dihedral) | pending |
| 6 | Wire `get_mol_PE_mcmm` kwargs + `scripts/sampler_benchmark._run_mcmm` defaults | pending |
| 7 | Validation: re-run sampler benchmark on cremp_sharp + cremp_typical; recompute Boltzmann coverage | pending |
| 8 | Documentation (`src/README.md`, `scripts/README.md`, dated Findings entry in mcmm_plan.md) | pending |

---

## Goals and constraints

The benchmark target is **cremp_sharp** (`S.S.N.MeW.MeA.MeN`) — the headline failure case from the 2026-05-21 Findings — and, by extension, any other peptide whose sampler basin set sits geometrically far from a CREMP ceiling for reasons traceable to side-chain rotamers. Specifically: bring cremp_sharp's `coverage_bw_ceiling` above zero at the τ=0.5 Å cross-method match tolerance (the codebase's `validation/cremp_coverage.py` convention) without regressing cremp_typical's current 0.83.

Constraints inherited from issue #10 / #11:

- Same `(mol, conf_ids, energies)` contract through `get_mol_PE_mcmm`'s shared post-sampling tail; the new proposer plugs into `make_composite_proposer` as a third route.
- MMFF as the inner-loop relaxer, MACE as the post-MCMM scorer — same tier separation as DBT and the Cartesian-kick proposer.
- Trivial Jacobian (`det_j = 1`): a single open-tree dihedral rotation is volume-preserving in dihedral space, so no Wu-Deem-style correction is needed (unlike DBT's closed-loop concerted rotation).

New constraints specific to this work:

- Strict separation from DBT — the new proposer must NOT touch backbone dihedrals. Side-chain rotatable bonds only.
- The rotation must apply to the *downstream* atomic subtree only — backbone, ring atoms, and other side chains untouched (handled correctly by `Chem.rdMolTransforms.SetDihedralDeg`).

---

## Design choices to lock (Step 1)

These four forks are intentionally left open in this plan and the GitHub issue. Step 1 of the work IS locking them, with the conversation captured under a dated Findings entry below.

### A. Move shape — Gaussian / discrete rotamer jump / hybrid

| option | mechanics | barrier crossing | acceptance rate | implementation cost |
|---|---|---|---|---|
| **Gaussian Δχ ~ N(0, σ_chi)** | refines within current rotameric well | low (MMFF tends to snap back across small Δχ) | high | ~5 lines |
| **Discrete rotamer jump** | swap to a different well sampled from {−60°, +60°, 180°} for χ₁, {−90°, +90°} for χ₂-aromatic, etc. | high (geometric definition crosses the barrier before MMFF starts relaxing) | low | ~15 lines |
| **Hybrid** | Gaussian by default; rotamer jump with probability `p_rotamer_jump` | controllable via `p_rotamer_jump` | controllable | ~20 lines |

Running recommendation: **hybrid**. Pure Gaussian almost certainly won't fix cremp_sharp (MMFF will undo small perturbations to the indole position), and pure rotamer jumps starve refinement on the cases where the sampler is already close. Default knob: `p_rotamer_jump = 0.3`.

### B. Multi-dihedral per step — single vs concerted

| option | description | cost |
|---|---|---|
| Single side-chain dihedral per walker per step (v0) | mirrors DBT's single-window default; simplest acceptance accounting | ~5 lines |
| Concerted χ₁ + χ₂ per residue | rotates both side-chain chi angles together; useful for aromatic side chains where χ₁ and χ₂ are coupled (Trp, Phe, Tyr, His) | ~50 lines, deferred unless v0 underperforms |

Running recommendation: **single per step for v0**, with concerted multi-dihedral parked in *Deferred follow-ups* below.

### C. Selection over side-chain rotatable bonds

| option | description | targeting on cremp_sharp |
|---|---|---|
| **Uniform random** | each side-chain rotatable bond equally weighted | NMe-Trp's χ₁ gets ~1/N share of attention |
| **Weighted by side-chain heavy-atom count** | bulky side chains (Trp, Phe, Tyr, NMe residues) preferentially sampled | NMe-Trp gets a larger share |
| **Adaptive (low-acceptance bias)** | weight inversely proportional to recent per-bond acceptance rate | requires per-bond statistics in the walker |

Running recommendation: **uniform random for v0**, with a `dihedral_weight_by_atom_count: bool = False` kwarg in place so the heuristic-weighted variant is one-line away.

### D. Composition-weights API in `make_composite_proposer`

The existing factory ([src/mcmm.py:1914](../src/mcmm.py#L1914)) routes per-walker per-step between DBT and Cartesian-kick using a single `cartesian_weight` float (DBT weight = 1 − cartesian_weight). With three proposers we need a 3-way routing parameter set:

| option | API shape | pros / cons |
|---|---|---|
| `weights: tuple[float, float, float] = (0.5, 0.25, 0.25)` for (DBT, Cartesian, dihedral) | tuple replaces both `cartesian_weight` and `dihedral_weight` | composable to n-way; breaks the existing API surface |
| Separate `dihedral_weight: float = 0.0` kwarg alongside `cartesian_weight` | DBT weight = 1 − cartesian_weight − dihedral_weight | backward-compatible; less elegant for n>3 |

Running recommendation: **additive `dihedral_weight` kwarg** — backward-compatible with existing benchmarks; we can refactor to a tuple later if a fourth proposer ever lands.

---

## Phase 1 — Foundation

### Step 1: Lock design choices — ✓ complete

Lock A, B, C, D per the tables above, with the chosen values recorded in a dated Findings entry under the *Findings* section below.

**Outcome (2026-05-22).** All four locked at the running recommendations — see the dated Findings entry below for the chosen values plus the v0 default knobs (σ_chi_rad, p_rotamer_jump, rotamer_wells_deg) that fall out of the lock. No re-opens; Step 2 (the proposer-module refactor) unblocked.

### Step 2: Refactor — extract proposers to `src/proposers.py` — ✓ complete

Mechanical move that lands BEFORE the new dihedral-kick code so it goes in the right place from day one. Locked via the 2026-05-22 layout discussion (single-module layout, lands as the first commit on this branch). The locking conversation is in the dated Findings entry below.

**What moves out of `src/mcmm.py`:**

- `make_mcmm_proposer` (DBT) and its in-file helpers.
- `make_cartesian_kick_proposer`.
- `make_composite_proposer`.
- The side-chain partition helpers consumed by both kinds of proposers: `_side_chain_group`, `_compute_window_downstream_sets`, `_backbone_atom_set`. (`enumerate_backbone_windows` and `_ordered_backbone_residues` stay in `mcmm.py` — they're consumed by `BasinMemory` and the driver as well as the proposers.)
- Anything else that's exclusively a proposer concern (drive-angle sampling helpers, batched MMFF wrappers, etc.). The boundary is "imported only by proposers" vs "imported by BasinMemory/walkers/drivers too" — the latter stays.

**What stays in `src/mcmm.py`:**

`BasinMemory`, `MCMMWalker`, `ParallelMCMMDriver`, `_swap_walker_configs`, `ReplicaExchangeMCMMDriver`, the Kabsch + inertia helpers (`_kabsch_rmsd_pairwise`, `_inertia_eigvals`, `_max_relative_eig_diff`), and the backbone-window enumeration. Expected line count after extraction: ~1000 lines (down from ~1990), with a clear "MCMM state + driving loop" responsibility.

**Mechanics:**

- Create `src/proposers.py` with the same module-level helper imports the proposer code already uses (`torch`, `numpy`, RDKit, `concerted_rotation`, `confsweeper._mace_batch_energies`, nvmolkit MMFF, etc.).
- Move the listed names verbatim from `mcmm.py` into `proposers.py`.
- In `mcmm.py`, replace the moved code with `from proposers import make_mcmm_proposer, make_cartesian_kick_proposer, make_composite_proposer` re-exports so existing `from mcmm import make_...` callers keep working (back-compat). Optionally drop the re-exports in a later commit if the consumer surface is small.
- Update direct callers: `src/confsweeper.py`'s `get_mol_PE_mcmm` swaps its `from mcmm import make_...` to `from proposers import make_...`.
- Update test files: `tests/test_mcmm.py` may need to split, but easiest first pass is to keep the proposer-specific tests in `test_mcmm.py` and just adjust their imports to `from proposers import ...`. A follow-up commit can split them into `tests/test_proposers.py` if desired.
- No public API change. No new dependencies.

**Test invariants:**

- `pixi run python -m pytest tests/ -q` passes with the same 381 / 8 / 0 count as the pre-refactor baseline (this is the regression guard — the refactor is by definition zero-behavior-change).
- `import src.proposers; src.proposers.make_mcmm_proposer` etc. all resolve.
- `import src.mcmm` still surfaces the re-exported proposer factories (if back-compat re-exports are kept).

**Outcome (2026-05-22).** Refactor landed cleanly with zero behaviour change. `src/proposers.py` carries the three factories (`make_mcmm_proposer`, `make_cartesian_kick_proposer`, `make_composite_proposer`) plus the three side-chain partition helpers (`_backbone_atom_set`, `_side_chain_group`, `_compute_window_downstream_sets`); `src/mcmm.py` shrunk from 1989 → 1434 lines (≈28% reduction) and now reads as "MCMM state + drivers" without the proposer factories on top of it. The dead `import ase` on `src/mcmm.py` was removed in the same edit (it was only consumed by the moved Cartesian-kick code).

**Back-compat:** an `from proposers import …` block at the bottom of `mcmm.py` re-exports all six moved names (three helpers + three factories). The block is placed at the bottom so the helpers consumed by `proposers.py` (`_ordered_macrocycle_atoms`, `enumerate_backbone_windows`) are defined first — the partial-import of `mcmm` from `proposers` resolves cleanly. Tests, `confsweeper.py`, and any external callers that imported these names from `mcmm` continue to work without changes. Caught one omission on the first test run (`_backbone_atom_set` wasn't in the re-export list); fixed by extending the import block to include all three helpers. The IDE flags the re-exports as "not accessed" within `mcmm.py` itself — these are false positives (cross-module consumers exist) and are intentional.

**Verification:** `pixi run python -m pytest tests/ -q` → **381 passed, 8 skipped, 3 warnings in 1292.63s (21m32s)** — bit-for-bit match with the pre-refactor baseline of 381 / 8. Step 3 unblocked.

### Step 3: Side-chain rotatable-bond enumeration — ✓ complete

- New helper `_enumerate_side_chain_dihedrals(mol) -> list[tuple[int, int, int, int]]` in [src/mcmm.py](../src/mcmm.py) returning the four-atom tuples `(a, b, c, d)` for each rotatable bond `(b, c)` that is NOT on the backbone.
- Uses RDKit's rotatable-bond SMARTS (`[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]`) to enumerate candidate bonds, then subtracts the backbone dihedrals already enumerated by `enumerate_backbone_windows` ([src/mcmm.py:94](../src/mcmm.py#L94)).
- Flanking atoms `a` (on b's side) and `d` (on c's side) chosen by lowest-index non-H neighbour, falling back to any neighbour for the methyl-edge case.
- Reuses `_backbone_atom_set` ([src/mcmm.py:237](../src/mcmm.py#L237)) and `_side_chain_group` ([src/mcmm.py:255](../src/mcmm.py#L255)) for the backbone / side-chain partition.

**Test invariants:**

- Empty list for a single-amino-acid model with no side-chain rotatable bond.
- For NMe-Trp (standalone or as part of cremp_sharp), the χ₁ dihedral (Cα–Cβ–Cγ–Cδ1) appears in the output.
- No backbone dihedral appears in the output — sentinel: `set(result) ∩ set(enumerate_backbone_windows(mol)) == ∅`.

**Outcome (2026-05-22).** `_enumerate_side_chain_dihedrals(mol)` shipped in `src/proposers.py` along with two small private helpers (`_heavy_degree`, `_pick_flanking_atom`) and the cached module-level `_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")`. Implementation matches the locked design: RDKit canonical rotatable-bond SMARTS → backbone-bond exclusion via `_backbone_atom_set` → heavy-degree-1 filter (catches methyl-type and -OH / -NH₂ terminal-heavy rotations in a single rule) → deterministic flanking-atom pick by lowest-index non-H neighbour. The methyl filter cleanly subsumes the more naive "is methyl?" check by also catching Ser's -OH and Asn's -NH₂ terminal groups, where heavy-atom rotation is H-dominated and adds no conformational diversity. New section `# Side-chain rotatable-bond enumeration — used by the dihedral-kick proposer` between the existing side-chain helpers and the DBT proposer factory.

**Tests:** 4 new tests in `tests/test_mcmm.py` covering the test invariants from the design: cycloAla4 + cycloAla6 → empty (all-methyl-side-chain case); cremp_sharp → contains Trp χ₁ (sp3-sp3-aromatic identified via `[CX4][CX4][c]` SMARTS, which uniquely matches Trp here); no backbone bond leaks (sentinel: NOT both b and c in backbone atoms); output type contract is list of 4-int tuples. New SMILES constant `_CREMP_SHARP_SMILES` and `_cremp_sharp_mol()` helper added near the top of the test file. New code imports directly from `proposers` (`from proposers import _enumerate_side_chain_dihedrals`) — the back-compat re-export through `mcmm` is reserved for pre-refactor callers.

**Verification:** targeted `pixi run python -m pytest tests/test_mcmm.py -q -k enumerate_side_chain_dihedrals` → 4 passed in 1.37 s. Full regression `pixi run python -m pytest tests/ -q` → **385 passed, 8 skipped, 3 warnings in 1287.27s (21m27s)** — exactly the predicted +4 over Step 2's 381 / 8 baseline.

### Step 4: `make_dihedral_kick_proposer` factory + tests — pending

- Factory `make_dihedral_kick_proposer(mol, sigma_chi_rad, p_rotamer_jump, rotamer_wells_deg, ...)` returning a proposer that matches the `(coords_batch, energies_batch) → (proposals, det_j_batch, success_batch)` contract used by `make_cartesian_kick_proposer` ([src/mcmm.py:1762](../src/mcmm.py#L1762)).
- Per call: for each walker, pick one side-chain dihedral from the Step-2 enumeration per the locked selection rule; sample Δχ per the locked move-shape rule; rotate the downstream subtree via `Chem.rdMolTransforms.SetDihedralDeg`; MMFF-relax batched; return as the new walker state.
- `det_j = 1` for every walker (open-tree rotation is volume-preserving).
- Exposes `.stats` dict (`n_proposed`, `n_relax_failures`, `n_relax_successes`, `n_rotamer_jumps` once the move shape is locked).

**Test invariants:**

- `det_j == 1` for every proposal.
- Pre-MMFF rotation touches only the downstream subtree (backbone + other side chains unchanged bit-for-bit).
- Rotation actually perturbs the targeted dihedral: pre-MMFF Δχ matches the sampled value within float tolerance.
- Single-walker and batched calls produce equivalent outputs (the existing batched-vs-individual invariant used for DBT and Cartesian).
- MMFF non-convergence rate on a representative cyclic-peptide test set < 5 % at the chosen σ_chi default.

---

## Phase 2 — Composition + integration

### Step 5: n-way `make_composite_proposer` — pending

- Extend `make_composite_proposer` ([src/mcmm.py:1914](../src/mcmm.py#L1914)) per the locked composition-weights API. Backward compatibility: existing callers passing only `cartesian_weight` keep working; the new `dihedral_weight` defaults to 0.0.
- Each sub-proposer is invoked on its assigned subset of walkers; results reassembled in walker order (same pattern as the current 2-way routing).
- Routing weights must sum to ≤ 1 with the residual going to DBT; raise on invalid combinations.

**Test invariants:**

- Routing weights sum to 1 after DBT residual.
- Zero-weight paths skip the corresponding factory entirely (no overhead).
- Walker-order preservation across all routing combinations.
- Sub-stats from each proposer reachable via the composite's `.stats`.

### Step 6: `get_mol_PE_mcmm` + sampler-benchmark adapter — pending

- New kwargs on `get_mol_PE_mcmm` ([src/confsweeper.py:1103](../src/confsweeper.py#L1103)): `dihedral_weight`, `sigma_chi_rad`, `p_rotamer_jump` (plus whatever else the locked design surfaces). In-code defaults preserve current behaviour (`dihedral_weight=0.0`).
- `scripts/sampler_benchmark.py:_run_mcmm` locks the production tuning for the new proposer alongside the existing Cartesian-kick tuning — same pattern as the 2026-05-06 production-setting decision recorded in mcmm_plan.md.

---

## Phase 3 — Validation + docs

### Step 7: Validation — pending

- Re-run `sampler_benchmark.py` on cremp_sharp and cremp_typical with the new composite (DBT + Cartesian + dihedral) at the production tuning from Step 6.
- Re-run `cremp_collapse_test.py run --dump_ceiling_sdf_dir` on the 2 CREMP peptides (regenerates the ceiling SDFs — cheap, ensures the ceiling and sampler-side runs use the same MMFF stochasticity window).
- Re-run `union_basin_count.py --ceiling_sdf_dir results/cremp_ceiling_sdfs --dedup_mode both` to recompute the Boltzmann coverage columns.

**Success criteria:**

- cremp_sharp: `coverage_bw_ceiling > 0` at τ=0.5 (any non-zero is a meaningful win given the current floor; ≥ 0.10 is the target).
- cremp_typical: `coverage_bw_ceiling ≥ 0.80` at τ=0.5 (no regression from the current 0.83).
- Diagnostic: track the minimum cross-method symmetric RMSD per ceiling basin via the direct pairwise diagnostic (recorded in mcmm_plan.md's 2026-05-21 Findings). Even a partial reduction from 3.1 Å on cremp_sharp is evidence the new proposer is moving in the right direction.

### Step 8: Documentation — pending

- Add the `make_dihedral_kick_proposer` paragraph and the new `get_mol_PE_mcmm` kwargs to `src/README.md`'s `## mcmm.py` section.
- Update `scripts/README.md`'s `sampler_benchmark.py` row if any new CLI options surface; document the new `dihedral_weight` / `sigma_chi_rad` defaults in the production tuning block.
- Append a dated Findings entry to [mcmm_plan.md](mcmm_plan.md) capturing the cremp_sharp + cremp_typical Boltzmann-coverage numbers under the new composite.

---

## Risks to instrument from day one

- **MMFF rotamer snap-back.** Small Δχ may relax back to the starting rotameric well, masking apparent acceptance as "moved" but contributing zero real diversity. Instrument per-step Δχ-before vs Δχ-after-relax. If snap-back rate > 50 %, lean harder on rotamer jumps via a larger `p_rotamer_jump`.
- **Single-dihedral moves are conformationally narrow on aromatic side chains.** Trp's χ₁ and χ₂ are coupled — rotating χ₁ alone may produce geometries that MMFF resolves to a different χ₂ well than the basin actually wants. Track the per-walker χ₂ distribution post-proposer; if a bimodal pattern emerges, that's the trigger for the concerted-multi-dihedral upgrade in Deferred follow-ups.
- **MACE energy noise on side-chain perturbations.** Side-chain motions can produce MACE energy differences below the float32 noise floor (~0.01–0.05 eV). Metropolis acceptance is unaffected (the Saunders novelty bias dominates the small ΔE), but the post-MCMM CREST-mode dedup may merge basins that are physically distinct in chi space. Diagnostic: track the chi-distribution of basins surviving dedup.
- **Backbone leakage.** A bug-class risk: any rotatable-bond classifier that accidentally lets a backbone dihedral through would silently corrupt the macrocycle. Assert in Step 3's tests that the side-chain enumeration is disjoint from `enumerate_backbone_windows`.
- **Composition imbalance.** Choosing the wrong `dihedral_weight` could starve DBT of barrier-crossing work or starve the dihedral proposer of opportunities. Run a small weight-sweep (e.g. dihedral_weight ∈ {0.1, 0.25, 0.4}) as part of Step 7 before locking the production default.

---

## Deferred follow-ups

- **Concerted χ₁ + χ₂ rotation** for aromatic / Trp-like side chains. Trigger: Step-7 validation shows cremp_sharp coverage stays low AND the per-walker χ₂ diagnostic shows a bimodal residual after χ₁ moves. Estimated cost: ~50 lines, mostly reusing the Step-3 enumeration with a per-residue grouping pass.
- **Adaptive per-bond weighting.** Selection weighted by recent per-bond acceptance rate. Trigger: uniform selection wastes steps on already-saturated side chains in the Step-7 diagnostic. ~30 lines.
- **N-methyl group rotation as a dedicated move type.** The N-CH₃ bond is 3-fold symmetric (just methyl rotation), so generally not interesting — but the backbone N's lone-pair direction can interact with χ₁ acceptance on the adjacent side chain. Speculative; skip unless Step-7 diagnostics show evidence.
- **Cross-method match-tolerance sensitivity.** The 0.5 Å default for `union_basin_count.py --match_rmsd` may not be the right answer in every regime; Step 7 should also report cremp_sharp at τ=1.0 to see whether the failure is "totally missing the region" (no coverage at any τ) or "missing the precise basin" (loose-τ coverage > 0).

---

## Decision points before coding

The four design choices in *Design choices to lock* above. None locked yet — that's Step 1, with the locking conversation recorded under *Findings* below.

---

## Findings

(append-only, dated)

### Design choices locked (2026-05-22)

All four open forks resolved at the running recommendations from the *Design choices to lock* tables above. The locking conversation surfaced no new alternatives; each choice is supported by the v0 scope (unblock cremp_sharp without over-engineering for hypothetical future regimes) and an explicit follow-up trigger should the v0 default underperform in Step 7.

| fork | locked choice | rationale | follow-up trigger |
|---|---|---|---|
| **A. Move shape** | Hybrid — Gaussian Δχ ~ N(0, σ_chi_rad) by default; rotamer jump from `rotamer_wells_deg` with probability `p_rotamer_jump` | Pure Gaussian almost certainly won't fix cremp_sharp (MMFF will snap small Δχ back to the starting rotameric well); pure rotamer jumps starve refinement near a basin. Hybrid keeps both. | Step-7 diagnostic: per-step Δχ-before vs Δχ-after-relax. If snap-back rate > 50 %, raise `p_rotamer_jump` toward 0.5 or higher. |
| **B. Multi-dihedral per step** | Single side-chain rotation per walker per step | Mirrors DBT's single-window default. Simplest acceptance accounting and proposer invariants. Defer concerted χ₁+χ₂ to follow-up. | Step-7 diagnostic: per-walker χ₂ distribution post-proposer. If bimodal residual on aromatic side chains, queue the concerted-multi-dihedral upgrade from *Deferred follow-ups*. |
| **C. Selection over side-chain rotatable bonds** | Uniform random | Simplest and unbiased. NMe-Trp's χ₁ gets its 1/N share, which under the size of cremp_sharp's side-chain pool is enough opportunity to test the hypothesis. Plumb `dihedral_weight_by_atom_count: bool = False` so the heuristic-weighted variant is one line away if uniform underperforms. | Step-7 diagnostic: per-bond proposal vs acceptance breakdown. If NMe-Trp χ₁ proposes often but accepts rarely, flip `dihedral_weight_by_atom_count` and rerun. |
| **D. Composition-weights API** | Additive `dihedral_weight: float = 0.0` kwarg alongside the existing `cartesian_weight`. DBT weight = 1 − cartesian_weight − dihedral_weight (residual). | Backward-compatible — `sampler_benchmark`, existing tests, and prior production-tuning records keep working without changes. Validate `cartesian_weight + dihedral_weight ≤ 1.0` at the entry point. | Refactor to a tuple/dict if a fourth proposer ever lands. |

**v0 default knobs that fall out:**

- `sigma_chi_rad: float = 0.5` (≈ 28°) — refinement Gaussian width. Wide enough to leave the starting micro-well within a few steps but well inside a single rotameric well so MMFF doesn't immediately undo the move.
- `p_rotamer_jump: float = 0.3` — fraction of steps that take a discrete rotameric jump instead of a Gaussian step. Mid-of-the-road; biased toward refinement so DBT and Cart still get most of the barrier-crossing work.
- `rotamer_wells_deg: tuple[float, ...] = (-60.0, 60.0, 180.0)` — standard sp3 χ₁ wells, applied uniformly to all rotatable side-chain bonds in v0. Aromatic χ₂ (≈ {−90, +90}) and other per-bond well sets are deferred (Step-7 diagnostic will tell us whether the universal 3-well set is too coarse).
- `dihedral_weight: float = 0.0` (in-code default) on `get_mol_PE_mcmm` and `make_composite_proposer`. Production tuning is locked in `scripts/sampler_benchmark.py:_run_mcmm` per the existing convention (see Step 6).
- `dihedral_weight_by_atom_count: bool = False` — plumbed but off, ready to flip if uniform selection underperforms.

Step 1 closes; Step 2 (proposer-module refactor) is unblocked.

### Proposer-module refactor locked (2026-05-22)

User flagged `src/mcmm.py` (~1990 lines) as unwieldy mid-Step-1 and asked for a refactor that lands together with the new dihedral-kick proposer. Lock: a new **single-module** `src/proposers.py` carrying all four proposer factories (`make_mcmm_proposer`, `make_cartesian_kick_proposer`, `make_composite_proposer`, plus the future `make_dihedral_kick_proposer`) and the side-chain partition helpers (`_side_chain_group`, `_compute_window_downstream_sets`, `_backbone_atom_set`) they share. **Timing:** lands as the first commit on the issue-#12 branch, before Step 3 builds on it — so the new dihedral-kick code goes into the right place from day one rather than landing in `mcmm.py` and being moved later. **Back-compat:** `mcmm.py` re-exports the moved names so any existing `from mcmm import make_mcmm_proposer` consumers keep working without changes. Trade-off accepted: less elegant than a `src/proposers/` package with per-proposer submodules, but matches the existing pattern in this codebase of one large file per concern (e.g. `concerted_rotation.py` for DBT geometry). Rationale for not splitting into a package: the four factories are tightly coupled (they share helpers, all consume the same batched MMFF + MACE infrastructure, and `make_composite_proposer` directly wraps the others), and four-files-for-four-factories is overkill until a fifth is on the horizon. `mcmm.py` expected to shrink from ~1990 to ~1000 lines with a clearer "MCMM state + driving loop" responsibility.

Test invariant gating this Step: `pixi run python -m pytest tests/ -q` reproduces the post-Boltzmann-coverage baseline of **381 passed, 8 skipped** with zero behaviour change.
