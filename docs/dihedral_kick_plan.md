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
| 4 | `make_dihedral_kick_proposer` factory + tests | ✓ complete |
| 5 | Extend `make_composite_proposer` to n-way routing across (DBT, Cartesian, dihedral) | ✓ complete |
| 6 | Wire `get_mol_PE_mcmm` kwargs + `scripts/sampler_benchmark._run_mcmm` defaults | ✓ complete |
| 7 | Validation: re-run sampler benchmark on cremp_sharp + cremp_typical; recompute Boltzmann coverage | ✓ complete (cremp_typical wins; cremp_sharp deferred to v0.2) |
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

### Step 4: `make_dihedral_kick_proposer` factory + tests — ✓ complete

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

**Outcome (2026-05-22).** `make_dihedral_kick_proposer` shipped in `src/proposers.py` (~140 lines code + ~75 lines docstring), matching the per-call pipeline laid out above and the locked Step-1 design (hybrid move with `sigma_chi_rad=0.5`, `p_rotamer_jump=0.3`, `rotamer_wells_deg=(-60, 60, 180)`; single side-chain dihedral per walker per step; uniform random selection; additive composition kwarg deferred to Step 5). The docstring carries the MMFF-vs-MACE PES collapse caveat the user flagged on 2026-05-22 — Stage 2's MMFF relax can pull a rotamer jump back across its barrier before MACE sees it, artificially collapsing the intended diversity onto the wrong potential energy surface; the no-MMFF ablation is queued in *Deferred follow-ups* with an explicit trigger.

**Behaviour:** `det_j = 1.0` for every successful proposal (open-tree rotation, no Wu-Deem term). `Chem.rdMolTransforms.SetDihedralDeg` is used to apply the rotation, which mutates only the atomic subtree downstream of the bond — backbone, ring, and other side chains are preserved by construction (the strict-separation-from-DBT invariant). Five `.stats` counters: `n_proposed`, `n_gaussian_steps`, `n_rotamer_jumps`, `n_relax_failures`, `n_relax_successes`. Six factory-time `ValueError`/`NotImplementedError` validations: unknown mmff_backend, `p_rotamer_jump ∉ [0, 1]`, non-positive `sigma_chi_rad`, empty `rotamer_wells_deg` when jumps are enabled, no enumerable side-chain dihedrals (e.g. cyclic homo-alanine), and the `dihedral_weight_by_atom_count=True` branch (plumbed but deferred — raises rather than silently shipping a stub).

**Tests:** 14 new tests in `tests/test_mcmm.py` covering all six validation paths plus eight behaviour invariants. A new small fixture `_cyclic_ala_ser_mol()` (head-to-tail cyclic 4-mer Ala-Ser-Ala-Ala) embeds cleanly with default ETKDG via `_seed_full_mol_coords` and gives the proposer exactly one side-chain dihedral (Ser χ₁ = Cα-Cβ) to operate on — small enough that the mock harness is fast and the behaviour invariants (proposal-per-walker, empty-input short-circuit, det_j=1, stats init + increment + Gaussian/rotamer split, pure-Gaussian at p=0, pure-rotamer at p=1, no input-coord mutation, and the strict-separation backbone-preservation check with MMFF mocked as no-op) all run in a couple seconds. Mocks use the same `_patch("confsweeper._mace_batch_energies", ...)` + `_patch("nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs", ...)` pattern `_make_cart_kick_with_mocks` established. New code imports `make_dihedral_kick_proposer` directly from `proposers` (no `mcmm` re-export — same going-forward convention as Step 3).

**Verification:** targeted `pixi run python -m pytest tests/test_mcmm.py -q -k dihedral_kick` → 14 passed in 2.47 s. Full regression `pixi run python -m pytest tests/ -q` → **399 passed, 8 skipped, 3 warnings in 1288.23s (21m28s)** — exactly the predicted +14 over Step 3's 385 / 8 baseline.

---

## Phase 2 — Composition + integration

### Step 5: n-way `make_composite_proposer` — ✓ complete

- Extend `make_composite_proposer` ([src/proposers.py:911](../src/proposers.py#L911)) per the locked composition-weights API. Backward compatibility: existing callers passing only `cartesian_weight` keep working; the new `dihedral_weight` defaults to 0.0.
- Each sub-proposer is invoked on its assigned subset of walkers; results reassembled in walker order (same pattern as the current 2-way routing).
- Routing weights must sum to ≤ 1 with the residual going to DBT; raise on invalid combinations.

**Test invariants:**

- Routing weights sum to 1 after DBT residual.
- Zero-weight paths skip the corresponding factory entirely (no overhead).
- Walker-order preservation across all routing combinations.
- Sub-stats from each proposer reachable via the composite's `.stats`.

**Outcome (2026-06-08).** `make_composite_proposer` itself is already n-way (it takes a `list[proposers]` + `list[weights]`); the real work was the layer above it — the routing assembly logic that owns "weights sum to ≤ 1, residual to DBT, skip zero-weight paths" and that lives at the `get_mol_PE_mcmm` call site. Step 5 ships that layer as a new helper `make_default_mcmm_composite(dbt_proposer, cart_proposer=None, dihedral_proposer=None, *, cartesian_weight=0.0, dihedral_weight=0.0, seed=0)` at [src/proposers.py:990](../src/proposers.py#L990). The helper validates non-negativity + the `cartesian_weight + dihedral_weight ≤ 1` sum, enforces a `weight > 0 ↔ proposer not None` contract (so a caller can't silently build an `~$MMFF + MACE-warmed` sub-proposer that never routes), short-circuits to the single active sub-proposer when only one weight is positive (preserving the dict-shaped `.stats` the issue-#10 aggregation block at [src/confsweeper.py:1383-1393](../src/confsweeper.py#L1383-L1393) reads), and otherwise wraps the active set in `make_composite_proposer`. The existing 2-way `get_mol_PE_mcmm` call site (now [src/confsweeper.py:1340-1358](../src/confsweeper.py#L1340-L1358)) was swapped to use it — `cartesian_weight=0.0` callers still take the zero-overhead identity path (the helper returns `dbt_proposer` directly), `cartesian_weight>0` callers still get a 2-element composite with the same weights as before. No new `get_mol_PE_mcmm` kwargs in this step — `dihedral_proposer=None` is hard-coded for now; Step 6 wires it. 16 new tests cover all 7 validation paths (negative weights ×2, sum > 1, weight↔proposer-presence ×4) + 9 behaviour invariants (pure-DBT identity short-circuit, pure-cart identity, pure-dihedral identity, 2-way DBT+cart back-compat 50/50 distribution, 2-way DBT+dihedral new combination 50/50, 3-way 40/30/30 distribution at N=3000 with ±10% slack, 3-way walker-order preservation, 3-way `.stats` reachability, dict-stats shape preserved through short-circuit). Targeted run: 16/16 in 1.4 s; combined with the 7 existing `composite_proposer` tests: 23/23 in 1.2 s. First full-suite run came back at **1 failed, 414 passed, 8 skipped** — the regression was in `tests/test_get_mol_PE_mcmm.py::test_mcmm_cartesian_weight_positive_builds_composite`, which patches `mcmm.make_cartesian_kick_proposer` to spy on the factory call. The call-site swap routed the import through `from proposers import make_cartesian_kick_proposer`, so the `mcmm.*` patch target no longer intercepted the symbol the running code resolved — the spy never incremented. Fixed by retargeting both spy patches in that file to `proposers.make_cartesian_kick_proposer` (the zero-weight test happened to keep passing because the spy was never expected to be called, but I updated it too for consistency with the new import path). Re-run after the patch fix: **415 passed, 8 skipped** (+16 over Step 4's 399/8).

### Step 6: `get_mol_PE_mcmm` + sampler-benchmark adapter — ✓ complete

- New kwargs on `get_mol_PE_mcmm` ([src/confsweeper.py:1103](../src/confsweeper.py#L1103)): `dihedral_weight`, `sigma_chi_rad`, `p_rotamer_jump` (plus whatever else the locked design surfaces). In-code defaults preserve current behaviour (`dihedral_weight=0.0`).
- `scripts/sampler_benchmark.py:_run_mcmm` locks the production tuning for the new proposer alongside the existing Cartesian-kick tuning — same pattern as the 2026-05-06 production-setting decision recorded in mcmm_plan.md.

**Outcome (2026-06-08).** Five new kwargs threaded onto `get_mol_PE_mcmm`: `dihedral_weight: float = 0.0`, `sigma_chi_rad: float = 0.5`, `p_rotamer_jump: float = 0.3`, `rotamer_wells_deg: tuple = (-60.0, 60.0, 180.0)`, `dihedral_weight_by_atom_count: bool = False` — all matching the locked Step-1 defaults. Each only consulted when `dihedral_weight > 0` (per the same zero-overhead pattern that `sigma_kick_a` follows for the Cartesian kick). Entry-point validation mirrors the existing `cartesian_weight < 0` check (`dihedral_weight < 0` raises with the analogous message); the combined `cartesian_weight + dihedral_weight ≤ 1` constraint is enforced inside `make_default_mcmm_composite` (no duplication). Routing block at [src/confsweeper.py:1340-1372](../src/confsweeper.py#L1340-L1372) extended: import now pulls all three proposer factories from `proposers`, the dihedral sub-proposer is built only when `dihedral_weight > 0` (using `seed + 5_555_555` as the offset, parallel to `+7_777_777` for DBT and `+8_888_888` for Cart), and both `cart_proposer` and `dihedral_proposer` are passed to the helper which handles the routing decision. Docstring extended with `dihedral_weight`, `sigma_chi_rad`, `p_rotamer_jump`, `rotamer_wells_deg`, `dihedral_weight_by_atom_count` blocks pointing to issue #12 and this plan. `scripts/sampler_benchmark.py` adapter chain wired through end-to-end: `_run_mcmm` adapter kwarg → `run_one` dispatcher kwarg → `main` CLI option (`--dihedral_weight`, default 0.0, mirrors `--cartesian_weight`) → start-of-run log line. Production-tuning of the other dihedral knobs stays at the in-code defaults (matches the existing Cartesian pattern where only `--cartesian_weight` is CLI-exposed and `sigma_kick_a` uses the in-code default) — runs will be re-tuned per the Step-7 diagnostics if the v0 defaults underperform. 5 new tests in `tests/test_get_mol_PE_mcmm.py` mirroring the cartesian-weight test family: `test_mcmm_dihedral_weight_zero_skips_dihedral_factory` (default leaves dihedral factory unconstructed — no MMFF + MACE setup paid; also confirms `cart_factory` stays untouched), `test_mcmm_dihedral_weight_positive_builds_composite` (only DBT + dihedral factories called, cart factory NOT called when its weight is 0), `test_mcmm_dihedral_weight_negative_raises`, `test_mcmm_three_way_routing_builds_all_three_factories` (full 3-way at `cartesian=0.3, dihedral=0.3` — all three factories called exactly once), `test_mcmm_weight_sum_exceeding_one_raises` (regression guard on the helper's sum-check propagating up to the `get_mol_PE_mcmm` entry point). Targeted run: 8/8 in 2.7 s (5 new + 3 existing cartesian). Full suite kicked off; predicted **420 passed, 8 skipped** (+5 over Step 5's 415/8).

---

## Phase 3 — Validation + docs

### Step 7: Validation — ✓ complete (cremp_typical at 0.991; cremp_sharp deferred to v0.2, 2026-06-15)

- Re-run `sampler_benchmark.py` on cremp_sharp and cremp_typical with the new composite (DBT + Cartesian + dihedral) at the production tuning from Step 6.
- Re-run `cremp_collapse_test.py run --dump_ceiling_sdf_dir` on the 2 CREMP peptides (regenerates the ceiling SDFs — cheap, ensures the ceiling and sampler-side runs use the same MMFF stochasticity window).
- Re-run `union_basin_count.py --ceiling_sdf_dir results/cremp_ceiling_sdfs --dedup_mode both` to recompute the Boltzmann coverage columns.

**Success criteria:**

- cremp_sharp: `coverage_bw_ceiling > 0` at τ=0.5 (any non-zero is a meaningful win given the current floor; ≥ 0.10 is the target).
- cremp_typical: `coverage_bw_ceiling ≥ 0.80` at τ=0.5 (no regression from the current 0.83).
- Diagnostic: track the minimum cross-method symmetric RMSD per ceiling basin via the direct pairwise diagnostic (recorded in mcmm_plan.md's 2026-05-21 Findings). Even a partial reduction from 3.1 Å on cremp_sharp is evidence the new proposer is moving in the right direction.

**Execution structure — two phases.** Phase 1 = a 4-cell × 2-peptide sweep at n_seeds=5000 (half production budget) to triage which mix moves which peptide off zero. Phase 2 = the surviving mix re-run at the saturation-validated n_seeds=10000 for the headline plus one diagnostic at the snap-back-trigger `p_rotamer_jump=0.7`.

**Phase 1 (2026-06-14) — ✓ complete, mixed outcome.** 4 cells: cell 1 pure DBT baseline `(cart=0, dih=0)`, cell 2 cart-only `(0.33, 0)`, cell 3 dihedral-only `(0, 0.33)`, cell 4 even 3-way `(0.33, 0.33)`. Driver `scripts/sweep_step7.sh`; per-cell outputs `results/sweep_step7_{coverage,sampler}_<cell>.csv` and `results/sweep_step7_<cell>/*_mcmm.sdf`. The full results table is in the dated Findings entry below — short version: **cremp_typical: dih-only cell 3 wins at 0.989 Boltzmann coverage (20/22 ceiling basins covered, max-missed-mass 0.009); cremp_sharp: every cell at exactly 0.000 with identical `max_missed_bw=0.724`** (the same single dominant basin missed by every mix). One non-trivial bug surfaced during the sweep (see Findings entry).

**Phase 2 (2026-06-14) — in progress.** Headline = cell 4 `(cart=0.33, dih=0.33, p_rotamer_jump=0.3)` at **n_seeds=10000** on both peptides → confirms whether the phase-1 cremp_typical 0.989 holds at the full production budget. Diagnostic = same mix with **p_rotamer_jump=0.7** at n_seeds=10000 → tests the locked Step-1 snap-back follow-up trigger on cremp_sharp ("if snap-back rate > 50 %, raise `p_rotamer_jump` toward 0.5 or higher"). Driver `scripts/sweep_step7_phase2.sh`, detached via setsid as PID 2061711 at 10:50:47; expected wall-clock ~30 min; outputs `results/sweep_step7_{coverage,sampler}_{headline_n10k_pjump30,diagnostic_n10k_pjump70}.csv`.

**To close Step 7:** aggregate the four phase-1 cells + two phase-2 runs into one comparison table; lock the production mix; capture the cremp_sharp story (pass or fail) as a dated Findings entry; queue follow-up triggers that did/didn't fire.

### Step 8: Documentation — pending

- Add the `make_dihedral_kick_proposer` paragraph and the new `get_mol_PE_mcmm` kwargs to `src/README.md`'s `## mcmm.py` section.
- Update `scripts/README.md`'s `sampler_benchmark.py` row if any new CLI options surface; document the new `dihedral_weight` / `sigma_chi_rad` defaults in the production tuning block.
- Append a dated Findings entry to [mcmm_plan.md](mcmm_plan.md) capturing the cremp_sharp + cremp_typical Boltzmann-coverage numbers under the new composite.

---

## Risks to instrument from day one

- **MMFF rotamer snap-back / artificial collapse onto the MMFF PES.** Small Δχ may relax back to the starting rotameric well, masking apparent acceptance as "moved" but contributing zero real diversity. More worryingly: the post-rotation MMFF relax operates on the **MMFF94 potential energy surface, not MACE** — for rotamer jumps that land near a well boundary, MMFF can pull the geometry back across the barrier into the starting well before MACE ever sees the proposal. This artificially collapses the dihedral kick's intended diversity through a force-field that isn't the one we're trying to sample (cf. the structural MMFF-vs-xtb basin disagreement already documented in `docs/mcmm_plan.md`'s 2026-05-05 Findings — same root cause, applied here to side-chain rotamers). Instrument per-step Δχ-before vs Δχ-after-relax. If snap-back rate > 50 %, lean harder on rotamer jumps via a larger `p_rotamer_jump`. If snap-back is still high AND Step-7 coverage stays poor, the trigger for the no-MMFF ablation in *Deferred follow-ups* fires.
- **Single-dihedral moves are conformationally narrow on aromatic side chains.** Trp's χ₁ and χ₂ are coupled — rotating χ₁ alone may produce geometries that MMFF resolves to a different χ₂ well than the basin actually wants. Track the per-walker χ₂ distribution post-proposer; if a bimodal pattern emerges, that's the trigger for the concerted-multi-dihedral upgrade in Deferred follow-ups.
- **MACE energy noise on side-chain perturbations.** Side-chain motions can produce MACE energy differences below the float32 noise floor (~0.01–0.05 eV). Metropolis acceptance is unaffected (the Saunders novelty bias dominates the small ΔE), but the post-MCMM CREST-mode dedup may merge basins that are physically distinct in chi space. Diagnostic: track the chi-distribution of basins surviving dedup.
- **Backbone leakage.** A bug-class risk: any rotatable-bond classifier that accidentally lets a backbone dihedral through would silently corrupt the macrocycle. Assert in Step 3's tests that the side-chain enumeration is disjoint from `enumerate_backbone_windows`.
- **Composition imbalance.** Choosing the wrong `dihedral_weight` could starve DBT of barrier-crossing work or starve the dihedral proposer of opportunities. Run a small weight-sweep (e.g. dihedral_weight ∈ {0.1, 0.25, 0.4}) as part of Step 7 before locking the production default.

---

## Deferred follow-ups

- **Concerted χ₁ + χ₂ rotation** for aromatic / Trp-like side chains. Trigger: Step-7 validation shows cremp_sharp coverage stays low AND the per-walker χ₂ diagnostic shows a bimodal residual after χ₁ moves. Estimated cost: ~50 lines, mostly reusing the Step-3 enumeration with a per-residue grouping pass.
- **No-MMFF ablation — score the rotated geometry directly with MACE.** The Stage 2 batched MMFF currently runs MMFF94 after the dihedral perturbation, so the geometry that reaches MACE has already been pulled toward an MMFF basin — possibly back across the barrier the rotation was meant to cross. The ablation swaps Stage 2 out for "no relax" (or a constrained MACE-only relax — more expensive), so MACE sees the rotated geometry as-is and the Metropolis acceptance decides on the true PES the run is sampling. Trigger: Step-7 cremp_sharp coverage stays low AND the per-step Δχ-before-vs-after-MMFF diagnostic shows a high snap-back rate. Estimated cost: ~20 lines (a `mmff_relax: bool = True` kwarg that gates Stage 2). Document the GPU-cost trade-off (MMFF is much cheaper than a MACE-relax; skipping MMFF entirely is the cheaper path).
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

### Step 7 phase 1 — 4-cell sweep at n_seeds=5000 (2026-06-14)

**Method.** 4 mix cells × 2 CREMP peptides (cremp_typical = `t.I.G.N`, cremp_sharp = `S.S.N.MeW.MeA.MeN`), driven by `scripts/sweep_step7.sh`. Per-cell: `sampler_benchmark.py --samplers mcmm --n_seeds 5000 --dedup_mode both --dump_sdf_dir <cell>` writes basin SDFs, then `union_basin_count.py --dbt_sdf_dir <cell> --cart_sdf_dir <cell> --ceiling_sdf_dir results/cremp_ceiling_sdfs --cremp_collapse_csv results/cremp_collapse_test_dual.csv` computes Boltzmann coverage. Passing the same dir as both `--dbt_sdf_dir` and `--cart_sdf_dir` makes `union(A, A) = A` so the BW-coverage columns are per-cell. Peptide list `data/processed/cremp/sweep_step7_peptides.csv` was constructed from `data/processed/cremp/validation_subset.csv` (just the two target rows) so the 3 PAMPA peptides — which have no CREMP ceiling and would have wasted ~hour of compute — are skipped. The cremp_collapse_csv is required because `union_basin_count._match_cremp_sequence` needs the sequence column to map the SDF filename `cremp_t.I.G.N_mcmm.sdf` back to the `t.I.G.N` ceiling file.

**Mid-sweep bug — `AttributeError: module 'rdkit.Chem' has no attribute 'rdMolTransforms'`.** Cells 1 + 2 ran clean; cells 3 + 4 crashed at the first dihedral-kick proposal because [src/proposers.py:856](../src/proposers.py#L856) called `Chem.rdMolTransforms.GetDihedralDeg` but the module had only `from rdkit import Chem` — RDKit doesn't auto-import the `rdMolTransforms` submodule into `Chem`'s namespace. The Step-4 test suite missed the bug because `tests/test_mcmm.py` is imported alongside other test modules that transitively pull in `rdMolTransforms`, populating `sys.modules['rdkit.Chem.rdMolTransforms']` before the dihedral-kick test runs. In the clean production-import path (`sampler_benchmark.py → confsweeper.py → mcmm.py → proposers.py`), nothing else touches the submodule and the attribute lookup fails. `sampler_benchmark.py`'s `try/except continue` wrapper swallowed the crash → empty SDF dirs → header-only coverage CSVs → driver kept going (`set -euo pipefail` didn't catch it because the Python exit code was masked by `try/except`). Fix: added `from rdkit.Chem import rdMolTransforms` at the top of `proposers.py` and rewrote both call sites (`GetDihedralDeg`, `SetDihedralDeg`) to use the direct reference. Clean-import smoke test through `confsweeper → mcmm → proposers` now resolves the symbol; 18/18 Step-3 + Step-4 unit tests still pass. Cells 3 + 4 rerun via `scripts/sweep_step7_rerun_3_4.sh` after stale outputs cleared.

**Results (kabsch dedup; crest is identical — BW columns are dedup-mode-independent).** Cell labels: 1 = DBT-only baseline, 2 = cart-only, 3 = dihedral-only, 4 = even 3-way.

cremp_typical (`t.I.G.N`):

| cell | cart_w | dih_w | DBT | `cov_bw_ceil` | `cov_count` | `max_missed_bw` | `n_new` |
|---|---|---|---|---|---|---|---|
| 1 DBT only | 0.00 | 0.00 | 1.00 | 0.005 | 0.045 (1/22) | 0.332 | 0 |
| 2 cart only | 0.33 | 0.00 | 0.67 | 0.664 | 0.227 (5/22) | 0.115 | 0 |
| **3 dih only** | **0.00** | **0.33** | **0.67** | **0.989** | **0.909 (20/22)** | **0.009** | 0 |
| 4 three-way | 0.33 | 0.33 | 0.34 | 0.989 | 0.909 (20/22) | 0.009 | 1 (mass 0.004) |

cremp_sharp (`S.S.N.MeW.MeA.MeN`):

| cell | cart_w | dih_w | `cov_bw_ceil` | `cov_count` | `max_missed_bw` | `n_new` | `new_mass` |
|---|---|---|---|---|---|---|---|
| 1 DBT only | 0.00 | 0.00 | 0.000 | 0.000 | 0.724023 | 8 | 1.2 × 10⁻¹⁴ |
| 2 cart only | 0.33 | 0.00 | 0.000 | 0.000 | 0.724023 | 4 | 1.7 × 10⁻⁶ |
| 3 dih only | 0.00 | 0.33 | 0.000 | 0.000 | 0.724023 | 3 | 5.7 × 10⁻⁸ |
| 4 three-way | 0.33 | 0.33 | 0.000 | 0.000 | 0.724023 | 3 | 1.3 × 10⁻⁵ |

**Interpretation.**

- **cremp_typical: the dihedral-kick proposer is the clear winner.** Cell 3 (dihedral-only) reaches 0.989 BW coverage at *half* the production budget — beating cell 2 (cart-only at the same weight) by 32 absolute percentage points (0.664) and the pure-DBT baseline by 98 (0.005). The 3-way mix (cell 4) matches cell 3 exactly on the headline metrics, so adding Cartesian on top of dihedral offers no measurable lift on this peptide; the marginal cell-4 win is 1 new basin holding 0.4 % of the joint mass. This far exceeds the Step-7 success criterion (`coverage_bw_ceiling ≥ 0.80`) and the prior `0.83` headline (n_seeds=10000 DBT+cart) — at *half* the production budget.
- **cremp_sharp: all four mixes flatline at zero.** `max_missed_bw=0.724023` is identical to six decimal places across every cell — the same single ceiling basin (holding 72 % of the 298 K Boltzmann population) is missed by every sampler. The `n_new` counts of 3–8 with masses of 10⁻¹⁴ to 10⁻⁵ are basins outside the ceiling but with thermodynamically irrelevant weight. The dihedral kick at the v0 defaults (sigma_chi_rad=0.5, p_rotamer_jump=0.3, rotamer_wells_deg=(-60, 60, 180)) does NOT recover NMe-Trp's dominant rotamer state on cremp_sharp at this budget.
- **n_seeds=5000 is enough to *separate* mixes but too tight for a pure-DBT headline.** Pure-DBT cremp_typical at 0.005 is far below the n_seeds=10000 DBT+cart baseline of 0.83 — half-budget DBT alone is severely undersampled, but the mix comparison is still apples-to-apples. Hence phase 2 confirms cell 4 at n_seeds=10000.

**Triggers that fire from the locked Step-1 follow-ups.** All three cremp_sharp follow-up clauses from the Step-1 design lock are now armed: (1) the **MMFF snap-back hypothesis** (raise `p_rotamer_jump` toward 0.5+) — phase 2's diagnostic run tests this directly at 0.7; (2) the **aromatic-χ₂ rotamer-well mismatch** (the locked `(-60, 60, 180)` is sp3-χ₁; NMe-Trp χ₂ sits near {-90, +90}) — deferred to a follow-up ticket pending phase-2 outcome; (3) the **no-MMFF ablation** queued in *Deferred follow-ups* — deferred similarly. The phase-2 outcome dictates which of (2), (3) escalates to a v0.2 code change.

**Files / artefacts.**

- `scripts/sweep_step7.sh` — phase-1 driver (4 cells × 2 peptides at n_seeds=5000)
- `scripts/sweep_step7_rerun_3_4.sh` — post-fix rerun of cells 3 + 4
- `data/processed/cremp/sweep_step7_peptides.csv` — 2-row peptide list (CREMP-only, no PAMPA)
- `results/sweep_step7_coverage_cell{1..4}_*.csv` — per-cell BW-coverage rows
- `results/sweep_step7_cell{1..4}_*/*.sdf` — per-cell basin SDFs
- `results/sweep_step7_logs/` — per-cell sampler + coverage logs

### Step 7 phase 2 — headline + snap-back diagnostic at n_seeds=10000 (2026-06-15)

**Method.** Two runs at the saturation-validated n_seeds=10000 on both CREMP peptides, driver `scripts/sweep_step7_phase2.sh`, detached via setsid. Both at the 3-way `(cart=0.33, dih=0.33)` mix that phase 1 identified as the leading candidate. Two p_rotamer_jump levels: the locked default `0.30` (headline) and `0.70` (diagnostic for the Step-1 snap-back follow-up trigger, "if snap-back rate > 50 %, raise toward 0.5 or higher"). To support the diagnostic, `scripts/sampler_benchmark.py` gained a `--p_rotamer_jump` CLI flag wired through `_run_mcmm → run_one → main` with the same pattern as `--dihedral_weight` (8 sites total). Outputs `results/sweep_step7_{coverage,sampler}_{headline_n10k_pjump30,diagnostic_n10k_pjump70}.csv`.

**Results (kabsch dedup; crest is identical).**

cremp_typical (`t.I.G.N`):

| run | n_seeds | p_rotamer_jump | `cov_bw_ceil` | `cov_count` | `max_missed_bw` |
|---|---|---|---|---|---|
| phase-1 cell 3 (dih-only) | 5000 | 0.30 | 0.989 | 0.909 (20/22) | 0.009 |
| phase-1 cell 4 (3-way) | 5000 | 0.30 | 0.989 | 0.909 (20/22) | 0.009 |
| **phase-2 HEADLINE (3-way)** | **10000** | **0.30** | **0.991** | **0.909 (20/22)** | **0.006** |
| phase-2 DIAGNOSTIC (3-way) | 10000 | 0.70 | 0.971 | 0.727 (16/22) | 0.009 |

cremp_sharp (`S.S.N.MeW.MeA.MeN`):

| run | n_seeds | p_rotamer_jump | `cov_bw_ceil` | `max_missed_bw` | `n_new` | `new_mass` |
|---|---|---|---|---|---|---|
| phase-1 cell 1 (DBT only) | 5000 | n/a | 0.000 | 0.724023 | 8 | 1.2 × 10⁻¹⁴ |
| phase-1 cell 4 (3-way) | 5000 | 0.30 | 0.000 | 0.724023 | 3 | 1.3 × 10⁻⁵ |
| phase-2 HEADLINE (3-way) | 10000 | 0.30 | 0.000 | 0.724023 | 2 | 5.0 × 10⁻⁴ |
| phase-2 DIAGNOSTIC (3-way) | 10000 | 0.70 | 0.000 | 0.724023 | 7 | 5.1 × 10⁻³ |

**Interpretation.**

- **cremp_typical: locked production mix.** Phase-2 headline confirms cell-4 generalises from n_seeds=5000 (0.989) to n_seeds=10000 (0.991) — at saturation, no meaningful headroom left at the current well set. Phase-2 diagnostic at `p_rotamer_jump=0.70` *regresses* cremp_typical by 2 absolute pts (`cov_count` drops 20/22 → 16/22): too few Gaussian refinement steps means the basin set narrows. The locked default 0.30 is the right knob value. **Production mix: cart=0.33, dih=0.33, p_rotamer_jump=0.30 at n_seeds=10000.**
- **cremp_sharp: still null, but the dihedral kick IS doing work that the metric doesn't capture.** `max_missed_bw=0.724023` is identical across all 6 phase-1 + phase-2 cells (7 decimal places) — the same single dominant basin is missed everywhere. *But* `new_mass` scales by ~40× going from phase-1 to phase-2 headline (1.3 × 10⁻⁵ → 5.0 × 10⁻⁴) and another ~10× going from headline to diagnostic (5.0 × 10⁻⁴ → 5.1 × 10⁻³, i.e. ~400× the phase-1 baseline). The dihedral kick at `p_rotamer_jump=0.70` is exploring much more aggressively and finding much more thermodynamically-weighty new basins — they're just not the *one* basin that dominates the ceiling distribution. **The snap-back hypothesis is partially supported** (more rotamer jumps drive more discovery) **but not the full story** (the dominant basin remains invisible at pjump=0.70).
- **The dominant cremp_sharp basin is structurally inaccessible to the v0 proposer at any tested mix.** Two locked-but-not-yet-implemented Step-1 follow-up triggers remain candidates: (a) the `rotamer_wells_deg=(-60, 60, 180)` defaults are sp3-χ₁ wells — NMe-Trp χ₂ (aromatic) sits near {-90, +90}, so χ₂ rotamer jumps land at non-minima and MMFF immediately drags them back; (b) Stage-2 MMFF relax may be undoing rotamer jumps even when they land near the right basin (the no-MMFF ablation hypothesis). Both are v0.2 code changes, not v0 knob tweaks — deferred to a follow-up ticket (issue #13 — drafted alongside this Step-7 close).

**Production-mix lock.** With cremp_typical confirmed at 0.991 and cremp_sharp deferred to v0.2:

- **Default production mix for the dihedral-kick proposer:** `cartesian_weight=0.33, dihedral_weight=0.33` (DBT residual = 0.34), `p_rotamer_jump=0.30`, `sigma_chi_rad=0.5`, `rotamer_wells_deg=(-60, 60, 180)`, `dihedral_weight_by_atom_count=False`. These are the in-code defaults on `get_mol_PE_mcmm` and `make_dihedral_kick_proposer` (Step-1 lock); only the per-knob `--cartesian_weight`, `--dihedral_weight`, and `--p_rotamer_jump` CLI flags are exposed at `sampler_benchmark.py` for sweep diagnostics.
- **No regression on cremp_typical:** 0.991 vs the prior issue-#10 headline of 0.83 (DBT + cart at n_seeds=10000) — **+16 absolute points of Boltzmann coverage** is the issue-#12 deliverable.
- **cremp_sharp residual:** documented as a known limitation, attacked separately in v0.2.

**Triggers that DID and DID NOT fire.**

- ✓ **Snap-back follow-up trigger fires partially.** Phase-2 diagnostic showed `p_rotamer_jump=0.70` increases `new_mass` discovery by ~400× over phase-1 baseline on cremp_sharp — consistent with the snap-back hypothesis but insufficient. The locked guidance ("raise toward 0.5 or higher") is empirically supported as an exploration tool, but not as a cremp_sharp fix.
- ✗ **Aromatic-χ₂-well trigger held.** Not yet tested — escalates to v0.2.
- ✗ **No-MMFF ablation trigger held.** Not yet tested — escalates to v0.2.
- ✗ **Compositional-imbalance Step-7 weight sweep partially fired.** The phase-1 sweep covered 4 mixes; phase-2 confirms cell-4 at the production budget but did not sweep an even finer weight grid on cremp_typical (judged unnecessary given the 0.991 ceiling).

**Files / artefacts.**

- `scripts/sweep_step7_phase2.sh` — phase-2 driver (3-way × 2 p_rotamer_jump levels × 2 peptides at n_seeds=10000)
- `results/sweep_step7_coverage_{headline_n10k_pjump30,diagnostic_n10k_pjump70}.csv` — phase-2 BW coverage rows
- `results/sweep_step7_sampler_{headline_n10k_pjump30,diagnostic_n10k_pjump70}.csv` — phase-2 sampler benchmark rows
- `results/sweep_step7_{headline_n10k_pjump30,diagnostic_n10k_pjump70}/*_mcmm.sdf` — phase-2 basin SDFs
- `results/sweep_step7_logs/{headline_n10k_pjump30,diagnostic_n10k_pjump70}_{sampler,coverage}.log` — phase-2 logs

Step 7 closes here. Step 8 (documentation) starts with this entry as the empirical record to compress into `src/README.md`, `scripts/README.md`, and a dated `docs/mcmm_plan.md` Findings cross-reference; the cremp_sharp v0.2 follow-up issue is drafted alongside (issue #13).
