# MCMM sampler with replica exchange — implementation plan

Branch: `10-benchmark-non-etkdg-samplers-against-exhaustive-etkdg-on-cyclic-peptides`. Implements issue #11 (sub-issue of #10).

This document is the working design for the third sampler in the issue-#10 benchmark: Multiple Minimum Monte Carlo (Saunders 1990) with replica exchange and Dodd-Boone-Theodorou concerted-rotation backbone moves. It captures the implementation order, the testable invariants at each layer, and the open decisions that block coding. Folded into `src/README.md` (or deleted) before the PR ships.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Refactor shared post-sampling tail | ✓ complete |
| 2 | DBT concerted rotation geometry (`src/concerted_rotation.py`) | ✓ complete |
| 3 | Backbone window enumeration (`src/mcmm.py`) | ✓ complete |
| 4 | Basin memory (`src/mcmm.py`) | ✓ complete |
| 5 | Single-walker MCMM driver (`src/mcmm.py`) | ✓ complete |
| 6 | Parallel walkers (batched) (`src/mcmm.py`) | ✓ complete |
| 7 | Replica exchange (`src/mcmm.py`) | ✓ complete |
| 8 | `get_mol_PE_mcmm` entry point + proposer stub (`src/confsweeper.py`, `src/mcmm.py`) | ✓ complete (orchestration) |
| 8b | Real `make_mcmm_proposer` (DBT + side-chain coupling + MMFF + MACE) | ✓ complete |
| 9 | Sampler benchmark wiring (`scripts/sampler_benchmark.py`) | ✓ complete |
| 10 | Documentation | ✓ complete |
| 11 | Kabsch heavy-atom RMSD dedup (replaces normalised-L1) | ✓ complete |
| 12 | Cartesian-kick proposer (alongside DBT) | ✓ complete |
| 13 | REMD vs. independent-T workers ablation | pending |
| 14 | Rotational-constant anisotropy as tertiary dedup gate | pending |
| 15 | Adaptive termination ("no new basin in K sweeps → stop") | pending |
| 16 | CREMP basin-collapse sanity check (diagnostic experiment) | ✓ complete |
| 17 | CREST-style three-criteria dedup as opt-in `dedup_mode='crest'` | ✓ complete |
| 18 | Post-hoc union of basin sets across proposers (`scripts/union_basin_count.py`) | ✓ complete |
| 19 | CREMP overlap statistics at scale (validation_subset, ~1k peptides) | ✓ complete |

---

## Goals and constraints

The benchmark target is `pampa_large` and similar peptides where `get_mol_PE_exhaustive` saturates near one-hot Boltzmann distributions despite the saturation-validated `n_seeds=10000` default. The hypothesis is that pure randomization plus MMFF cannot reach certain low-energy basins regardless of seed budget — a connectivity problem rather than a sampling-density problem. MCMM walks adaptively from a known minimum with basin memory, which directly attacks that failure mode.

Constraints inherited from issue #10:

- Same `(mol, conf_ids, energies)` contract as `get_mol_PE_exhaustive` and `get_mol_PE_pool_b` so the function plugs into `scripts/sampler_benchmark.py`'s `SAMPLERS` dispatch with a single new row.
- Matched compute budget against exhaustive ETKDG's `n_seeds=10000`: 64 walkers × 200 steps = 12 800 minimizations per peptide.
- MMFF as the inner-loop minimizer (cheap), MACE as the final scorer over the basin set. Same tier separation as exhaustive ETKDG.
- Backbone moves only in v0; side-chain rotamer moves deferred.

Constraints inherited from issue #11:

- DBT concerted rotation as the only move type in v0 (not as a v1 upgrade). The reasoning is that replica-exchange high-T replicas need real barrier-crossing moves, and local-pair perturbations with MMFF closure cap move size; without DBT the high-T compute is wasted.
- Wu-Deem 1999 Jacobian correction so detailed balance holds and the chain converges to the correct stationary distribution.
- Replica exchange in v0, not bolted on later.

---

## Phase 1 — Foundation (unblocks everything else)

### Step 1: Refactor shared post-sampling tail — ✓ complete

Extract the MMFF → batched MACE score → 5 kT energy filter → `_energy_ranked_dedup` → non-centroid prune block from `get_mol_PE_exhaustive` and `get_mol_PE_pool_b` into a private `_minimize_score_filter_dedup(mol, calc, hardware_opts, ...)` helper in `src/confsweeper.py`. Both existing functions become thin wrappers around their respective Phase 1 sampler plus the shared tail.

The refactor flag was deliberately deferred from the Pool B PR because two callers did not justify the risk of touching production code. Three callers do. MCMM cannot be cleanly written without it — otherwise the tail becomes a third copy and the next sampler (CREST-fast or REMD) becomes a fourth.

Behavior must be byte-equivalent. Verification: re-run `tests/test_exhaustive_etkdg.py` and `tests/test_pool_b.py` unmodified; all tests pass. Remove the refactor flag from `src/README.md` and from `get_mol_PE_pool_b`'s docstring.

**Outcome.** Helper landed in `src/confsweeper.py`. Both callers became thin wrappers. All 33 existing exhaustive + pool_b tests pass unchanged. Refactor flag removed from `src/README.md`.

### Step 2: DBT concerted rotation geometry — ✓ complete

New self-contained `src/concerted_rotation.py`. No MC, no MACE, no torch — pure numpy. Standalone module so any other macrocycle MC code in this project (or downstream) can pick it up without depending on the MCMM driver.

**v0 closure solver: numerical (Option B).** The original plan called for the analytical degree-16 polynomial of Coutsias 2004 (Option A), but on review the algebraic re-derivation is high-risk for subtle bugs that would silently violate detailed balance, and pure-Python performance of the closure step is irrelevant given MMFF dominates the per-move cost. v0 uses `scipy.optimize.least_squares` to solve the closure constraint numerically, with the current geometry as the initial guess. This finds a single closure branch (the one in the same homotopy class as the current state) rather than enumerating all up to 16 branches.

**Implementation:**

- 7-atom-window parameterization. Inputs: positions of 7 consecutive backbone atoms, choice of drive dihedral, drive perturbation Δθ. Outputs: new positions of the inner atoms (outer atoms r₀, r₁, r₅, r₆ preserved approximately by the closure solver, exactly to within `tol`).
- Numerical closure via `scipy.optimize.least_squares`. The system is generically over-determined when 1 of 4 dihedrals is the drive (6 raw position constraints on the outer atoms r₅, r₆ minus a few automatic from chain rigidity, vs. 3 free dihedrals). Least-squares handles the over-constraint gracefully: when an exact solution exists, it converges to zero residual; when no exact solution exists, residual stays non-zero and the move is rejected as geometrically infeasible. Tunable Δθ amplitude trades acceptance rate against move size.
- Wu-Deem 1999 Jacobian via finite differences. |det J| of the 3 free dihedrals with respect to the drive angle, evaluated at the solution; used to weight the proposal probability for detailed balance. Finite differences are noisier than the analytical form but have correct qualitative behavior.

**Test invariants:**

- Zero perturbation maps to identity (positions unchanged to machine precision).
- Outer atom positions preserved to within `tol` after a successful closure.
- Bond lengths and bond angles in the 7-atom chain preserved by the chain-rebuild primitive (independent of the closure solver).
- The drive dihedral changes by exactly Δθ relative to its original value.
- Numerical Jacobian smoke test: |det J| is finite and well-conditioned across a sweep of geometries.

**Option A as deferred upgrade.** The analytical degree-16 polynomial (Coutsias 2004 / DBT 1993) gives multi-branch enumeration: a single move can jump to a topologically distant ring conformation rather than being trapped in the current homotopy class. This is the entire reason DBT/Coutsias is special vs. naive loop closure. For cyclic peptides with realistic constrained geometry, real basins probably all live in one homotopy class, so multi-branch enumeration mostly buys noise — but if benchmark data shows MCMM stuck in one branch on real peptides, Option A becomes worth the ~400-500 lines of algebraic implementation. Trigger: per-replica branch-jump rate near zero in `pampa_large` runs, or basin coverage saturating below `get_mol_PE_exhaustive`'s.

Pure-Python performance is fine for v0 — the bottleneck of the full pipeline is MMFF, not move proposal. Profile-driven port to torch or C only if profiling later shows the closure step itself is hot.

**Outcome.** `src/concerted_rotation.py` (~280 lines) with `rotation_matrix`, `dihedral_angle`, `apply_dihedral_changes`, `closure_residual`, `propose_move`. 20 tests in `tests/test_concerted_rotation.py` covering rotation primitives, chain-rebuild geometry preservation, sign-convention self-consistency, closure contract, and edge cases. `DEFAULT_CLOSURE_TOL` set to 0.01 Å (relaxed from initial 1e-6 because the system is over-determined: 6 residuals on r5/r6 vs. 3 free dihedrals when one is the drive); coverage-lever guidance documented in the module docstring.

---

## Phase 2 — MCMM algorithm core

All in a new `src/mcmm.py`. Builds on Step 2 (`concerted_rotation.py`) and Step 1 (refactored shared tail).

### Step 3: Backbone window enumeration — ✓ complete

Given a cyclic peptide mol, return all valid 7-atom windows entirely inside the macrocycle. Reuses `torsional_sampling._BACKBONE_SMARTS` and `get_backbone_dihedrals` for the residue-level structure.

Tests on cyclo(Ala)4 and cyclo(Ala)6: verify expected window count and that every returned window's outer atoms are inside the ring.

**Outcome.** `src/mcmm.py` created with `enumerate_backbone_windows(mol)` and the private `_ordered_backbone_residues(mol)` helper. Walks the macrocycle ring via C → N peptide bonds, emits 3K cyclic windows for a K-residue cyclic peptide. Reuses `get_backbone_dihedrals` from `torsional_sampling.py` (rather than the private `_BACKBONE_SMARTS` constant). 13 tests in `tests/test_mcmm.py` covering window count, sequential bonding, cyclic shifts, full backbone coverage, plus error cases (linear peptides raise `ValueError`, non-peptide cyclic mols return `[]`). The initial `len(ordered) != len(residues)` closure check was too weak for linear peptides where only the fully-internal residues match the SMARTS; replaced with an explicit `ring_closed` flag set only when the walk's `next_n` equals `start_n`.

### Step 4: Basin memory — ✓ complete

`class BasinMemory` backed by torch tensors `[K, n_atoms, 3]` for stored basin coordinates and `[K]` for usage counts.

Operations:

- `add_basin(coords, energy)`: append to the tensor, initialize usage = 1.
- `query_novelty(coords) -> (idx_or_None, distance)`: batched normalized-L1 against all K basins (same metric as `_energy_ranked_dedup`); return the closest basin within the threshold or `None`.
- `record_visit(idx)`: increment usage[idx].
- `acceptance_bias(idx) -> float`: 1/√usage[idx] (Saunders form); 1.0 when idx is None (novel basin).

Tests: threshold behavior (boundary cases match `_energy_ranked_dedup`), usage counter monotonicity, batched novelty query against many proposals returns vector of indices in one call.

**Outcome.** `BasinMemory` class added to `src/mcmm.py` with the four operations above plus a batched `query_novelty_batch(coords_batch) → (indices, distances)` for the parallel-walkers driver (Step 6) and read-only properties for `coords`, `energies`, `usages`, `n_basins`. Default `rmsd_threshold=0.1` matches `_energy_ranked_dedup`'s default; the strict `<` boundary convention matches it too. Stored representative is the first conformer found per basin — re-visits don't update coordinates or energy (the post-MCMM `_minimize_score_filter_dedup` re-deduplicates at the MACE level, so this simplification doesn't change the final basin set materially). 19 tests in `tests/test_mcmm.py` covering construction, add/query contracts, threshold-boundary matching `_energy_ranked_dedup`, batched-vs-individual query equivalence, usage monotonicity, Saunders 1/√usage decay, error cases, and a driver-flow integration smoke test.

### Step 5: Single-walker MCMM driver — ✓ complete

Sequential reference implementation. One step is: propose DBT move (random window, random drive dihedral, Gaussian Δθ) → MMFF minimize → query basin memory → Metropolis accept/reject with Saunders 1/√usage bias multiplied into the standard min(1, exp(−ΔE/kT)) factor → update memory and walker state.

Tests:

- T = 0 always rejects worse-energy proposals.
- T = ∞ always accepts (modulo the Wu-Deem Jacobian factor).
- Basin memory grows monotonically across steps.
- Saunders bias suppresses re-discovery: a walker repeatedly visiting the same basin sees acceptance probability decay as 1/√k where k is the visit count.

**Outcome.** `MCMMWalker` class added to `src/mcmm.py`. Architectural decision: the walker is generic over the proposal mechanism — it takes a `propose_fn(coords) → (new_coords, new_energy, det_j, success)` callable rather than constructing the proposal internally. This separates the MC logic (Metropolis + Saunders bias + Wu-Deem Jacobian + memory bookkeeping) from the geometry application (DBT move + side-chain coupling + MMFF), so the MC logic is unit-testable without an RDKit mol or MMFF backend. The real RDKit-coupled proposer is the Step 5b / Step 6 integration target. T=0 and T=∞ limits are special-cased in `_acceptance_prob` to avoid `exp(-ΔE/kT)` overflow at the boundaries; numpy's `exp` handles intermediate overflow gracefully (returns `inf` rather than raising). Initial-state handling: walker queries memory on construction and only adds the basin if novel, so passing one shared `BasinMemory` to N walkers does not artificially inflate the discovery basin's count. 13 walker tests in `tests/test_mcmm.py` covering init contract (fresh and shared memory), kT=0 / kT=∞ limits, geometric rejection, memory growth and re-visit bookkeeping, the Saunders 1/√k decay (deterministic random_fn = 0.5 → bias < 0.5 at usage = 4 rejects, bias ≥ 1/√3 at usage = 3 admits), and the `run(n_steps)` convenience loop's accept counter semantics.

### Step 6: Parallel walkers (batched) — ✓ complete

N walkers proposing concurrently. Each walker contributes one conformer to a shared mol; MMFF runs on the full set in one `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` call. Basin memory is shared across walkers; each walker's accept/reject decision is independent given the post-MMFF energy.

Verification: small-N batched results match the single-walker reference run sequentially with the same RNG seeds.

**Outcome.** `ParallelMCMMDriver` class added to `src/mcmm.py`. Architectural refactor: `MCMMWalker.step` was split into `step` (single-walker convenience that calls a `propose_fn`) and `apply_proposal` (lower-level primitive that takes a precomputed proposal). The parallel driver builds on `apply_proposal` so it can batch the proposal-generation stage across walkers (one GPU call per step) and dispatch accept/reject decisions in a sequential walker loop. Sequential dispatch is intentional: walker `i` sees memory updates from walkers `j < i` made earlier in the same step. This avoids duplicate-basin creation when two walkers propose into the same novel conformation and matches the standard MCMM-with-shared-memory convention. The `batch_propose_fn(coords_list) → list[(coords, energy, det_j, success)]` abstraction is the integration target for Step 8 — at that point a closure over the shared RDKit mol will stage all N walker conformers as distinct conformer IDs and run nvmolkit MMFF + MACE in single batched calls. Eight driver tests in `tests/test_mcmm.py` covering construction validation, N=1 equivalence with single walker, per-walker accept-list ordering, disjoint-basin independence, shared-basin serialisation (the load-bearing test that two walkers proposing into the same novel basin produce one basin with usage=2), proposal-count mismatch error, run-loop accept aggregation, and `n_accepted` property aggregation across walkers.

### Step 7: Replica exchange — ✓ complete

8 temperatures geometric 300 K → 600 K. N walkers per temperature (default 8 → 64 walkers total). Swap attempts between adjacent temperatures every 20 steps via standard Metropolis on ΔE × Δβ. Replica indices are tracked so the basin memory's per-temperature provenance can be inspected if needed (though the memory itself is shared across temperatures).

Tests: swap acceptance probability matches the analytical Metropolis value across many independent trials; replica ordering is preserved after swaps (no state mixing bugs).

**Outcome.** `ReplicaExchangeMCMMDriver` class added to `src/mcmm.py`, plus a private `_swap_walker_configs(a, b)` helper that exchanges `(coords, energy, current_basin_idx)` between two walkers while leaving `kt`, RNG, and counters with the slot. Architectural decision: swap configurations rather than temperatures, so per-temperature provenance is preserved (the walker at slot `(t, i)` always tracks the trajectory at `kts[t]`, even though individual configurations have hopped across temperatures). Per `step()`: batch propose across all walkers (one GPU call), then if `n_steps % swap_interval == 0` attempt swaps between every adjacent (t, t+1) temperature pair, paired by within-temp walker index. Constructor enforces uniform group size, uniform kt within each group, and strictly-increasing kt across groups so the ladder is well-defined. Swap acceptance: `p = min(1, exp((β_high − β_low)(E_high − E_low)))` with the `arg ≥ 0` (favorable) branch shortcircuited to 1. 18 tests in `tests/test_mcmm.py` covering `_swap_walker_configs` semantics (configs swap, counters and kt stay), constructor validation (empty, uneven group sizes, mixed kt within group, non-monotonic temperatures, invalid swap_interval), the swap probability formula in three regimes (conditional, favorable-always-1, equal-energy-1), swap mechanics (always-accept and always-reject paths), all-pairs scan order, `step()` flat-order results, swap-interval timing, and run-loop accept aggregation.

One test-fixture lesson: `_make_remd_walkers` initially placed within-temp walkers at offsets that fell within the basin-distinguishing threshold of each other, so multiple walkers latched onto the same initial basin and the Saunders bias kicked in faster than expected during the run-loop accept-counting test. Fixed by spacing offsets to gap = 10 (≈ 3.3 normalised-L1 units, well above the 0.5 threshold).

---

## Phase 3 — Integration

### Step 8: `get_mol_PE_mcmm` entry point — ✓ complete (orchestration); Step 8b pending (real proposer)

New function in `src/confsweeper.py`. Pipeline: enumerate backbone windows → initialize walkers from a seed conformer (e.g., one ETKDG conformer minimized with MMFF) → run replica-exchange MC for the configured step budget → MACE-rescore the basin set → call refactored `_minimize_score_filter_dedup` for the final filter/dedup/prune. Returns `(mol, conf_ids, energies)`.

Integration tests mirror `tests/test_pool_b.py`: mock GPU stages (MMFF, MACE, basin-memory if needed) and verify the contract — return shape, energy ordering, conformer-count consistency, zero-conformer safety, etc.

**Outcome.** `get_mol_PE_mcmm` lives in `src/confsweeper.py` alongside the other `get_mol_PE_*` family members. Pipeline: SMILES → `Chem.AddHs` → ETKDG embed (1 seed conformer) → MMFF minimise → MACE-score for initial energy → build shared `BasinMemory` and a temperature ladder of walkers (default 8 temps × 8 walkers) → build the batch proposer via `make_mcmm_proposer` → run `ReplicaExchangeMCMMDriver` for `n_steps` → extract every basin in memory as a conformer on the mol → pass through `_minimize_score_filter_dedup` for final MACE rescoring + 5 kT energy filter + `_energy_ranked_dedup` + non-centroid pruning. Defaults match issue #11: 64 walkers, 200 steps (12,800 minimisations), 300 K → 600 K geometric ladder, 20-step swap interval. Companion helper `_geometric_temperature_ladder(kt_low, kt_high, n)` computes the ladder.

**Step 8 is the orchestration; Step 8b is the real proposer.** `make_mcmm_proposer` in `src/mcmm.py` is currently a v0 stub that returns `success=False` for every walker — no MC moves are accepted, but the orchestration is fully wired and tested. Step 8b will replace the stub body with: per-walker DBT move proposal via `concerted_rotation.propose_move`, full-mol coordinate update with side-chain coupling, batched MMFF on a shared throwaway mol, batched MACE scoring, finite-difference Wu-Deem Jacobians. Splitting the work this way matched the pool_b PR pattern of mocked GPU stages in tests and let the orchestration shape land before sinking time into the geometry + GPU integration.

**One real architectural decision surfaced**: the lazy `from mcmm import …` inside `get_mol_PE_mcmm` made `confsweeper.make_mcmm_proposer` non-patchable (the name was never bound at module level), so test mocking failed. Moved the imports — `BasinMemory`, `MCMMWalker`, `ReplicaExchangeMCMMDriver`, `make_mcmm_proposer` — to `src/confsweeper.py`'s top-level imports. No circular dependency since `mcmm` doesn't import from `confsweeper`.

**Minor BasinMemory contract change**: `rmsd_threshold` validation went from `> 0` to `≥ 0`. The strict `<` comparison in `query_novelty` means threshold = 0 disables matching (every basin is unique), which matches the "disable geometric dedup" idiom used in pool_b and exhaustive tests.

11 entry-point tests in `tests/test_get_mol_PE_mcmm.py` covering the geometric ladder primitive, smoke test under the no-exploration stub, zero-conformers safety, proposer call count per step, basin-memory growth under an accepting mock proposer (4 init basins + 12 accepted = 16 final), unknown-mmff-backend rejection, and default-temperature-endpoint match (300 K / 600 K). Plus 3 stub tests for `make_mcmm_proposer` itself in `tests/test_mcmm.py` (callable contract, always-reject behaviour, length-matching across batch sizes).

### Step 8b: Real `make_mcmm_proposer` implementation — ✓ complete

Replace the v0 no-op stub in `src/mcmm.py` with the real DBT + MMFF + MACE proposer. The factory closure captures `mol` topology and GPU resources at build time; the returned `batch_propose_fn(coords_list)` per call:

1. **Per walker, generate a move**: pick a random backbone window (uniform over the 3K windows), a random drive index in {0..3}, and `drive_delta ~ N(0, drive_sigma_rad²)`. Call `concerted_rotation.propose_move` on the 7-atom backbone window positions. Record success/failure.
2. **Apply moves to the full mol**: for successful walkers, apply the resulting 4 dihedral deltas to the FULL coordinate array (rotate window backbone atoms r3..r6 plus side chains of r2..r6 around the appropriate bond axes — same chain-rebuild logic as `concerted_rotation.apply_dihedral_changes`, but operating on the full atom set). Side-chain coupling rule: each backbone atom's side-chain group transports rigidly with its parent (atoms reachable via non-backbone bonds, not crossing other backbone atoms).
3. **Stage as batched conformer set**: assemble the N candidate conformations as conformer IDs on a throwaway mol (one per walker). Run nvmolkit `MMFFOptimizeMoleculesConfs` in one batched call.
4. **Score in batches**: run MACE in chunks via `_mace_batch_energies` (or compute MMFF energies — pending the MMFF-vs-MACE inner-loop decision flagged in the plan's risk register).
5. **Compute per-walker Wu-Deem Jacobians** via finite differences (the `concerted_rotation.propose_move` helper already does this; we just need to plumb the result through).
6. **Return** `(coords_tensor, energy_float, det_j_float, success_bool)` per walker. Failed walkers (closure failure or MMFF blowup) get `success=False`.

Two sub-pieces to land before Step 8b's body:

- **`concerted_rotation.propose_move` returning the deltas**: currently returns `(new_positions, det_j, success)`. Step 8b needs the 4 dihedral deltas too so the full-mol application can replay them on the side chains. Refactor to a NamedTuple or 4-tuple return.
- **Side-chain group enumeration**: helper `_side_chain_group(mol, backbone_atom, backbone_atom_set)` doing BFS through non-backbone bonds. Used to precompute per-window downstream sets at factory-build time so the per-step move is fast.

Once 8b lands, Step 9 (sampler benchmark wiring) plugs `mcmm` into `scripts/sampler_benchmark.py`'s `SAMPLERS` dispatch and the end-to-end pipeline can run on real peptides.

**Outcome.** Three pieces shipped, in dependency order:

1. **`concerted_rotation.MoveProposal`** — `propose_move` now returns a NamedTuple with fields `(new_positions, det_jacobian, deltas, success)`. Tuple-unpackable as a 4-tuple so existing 3-tuple-style call sites need updating; the `deltas` field is the (4,) array of dihedral changes the closure solver applied. Updated 5 tests in `tests/test_concerted_rotation.py` to use attribute access.

2. **`concerted_rotation.apply_dihedral_changes_full_mol`** — generalises the 7-atom chain-rebuild primitive to operate on a full-molecule coordinate array, with explicit per-dihedral atom rotation sets. Pure numpy, no RDKit dependency. 4 tests cover the zero-deltas identity, the window-only equivalence with `apply_dihedral_changes`, side-chain bond-length preservation under rotation with parent, and shape-validation errors.

3. **Side-chain helpers in `src/mcmm.py`** — `_backbone_atom_set(mol)`, `_side_chain_group(mol, atom_idx, backbone_atom_set)`, `_compute_window_downstream_sets(mol, window, backbone_atom_set)`. The first returns the 3K backbone atom indices for a K-residue cyclic peptide; the second BFS-traverses non-backbone bonds without crossing the macrocycle; the third combines the two to produce the 4 per-dihedral rotation sets that `apply_dihedral_changes_full_mol` consumes. 9 tests cover residue-class-specific side chain sizes (Ala Cα → 5 atoms, amide N → 1 H, amide C → 1 O), disjointness across residues, downstream-set monotonicity, pivot-not-included, and pivot-side-chain-included invariants.

4. **Real `make_mcmm_proposer` body** — replaces the stub with the 5-stage pipeline:
   1. Per-walker DBT closure on the backbone window (CPU, sequential).
   2. Full-mol coordinate update with side-chain coupling (CPU, sequential).
   3. Batched MMFF on a fresh throwaway mol (one nvmolkit GPU call, or RDKit serial fallback).
   4. Batched MACE scoring chunked by `score_chunk_size` (one GPU call per chunk).
   5. Per-walker proposal assembly in walker order — failed walkers get pass-through with `success=False`.

   Tests cover: factory-build validation (rejects non-cyclic input, rejects unknown mmff_backend), proposal-count contract, the load-bearing "at least one of 8 walkers succeeds" closure check, finite energy + non-negative |det J| on success, failed-proposal pass-through with original coords, and the no-mutate invariant on input tensors.

**One real architectural decision recorded**: the proposer's lazy `from confsweeper import _mace_batch_energies` inside the closure body is intentional — it defers resolution past `confsweeper`'s module-load `from mcmm import make_mcmm_proposer`, breaking the circular dependency without splitting the MACE primitive into a third module.

**Test count delta**: +4 in `test_concerted_rotation.py` (24 total), +5 in `test_mcmm.py` (90 total), no changes in `test_get_mol_PE_mcmm.py` since it patches `confsweeper.make_mcmm_proposer` and is unaffected by the body swap. All 158 tests pass across the 5 directly-affected suites.

### Step 9: Sampler benchmark wiring — ✓ complete

Add `"mcmm"` row to `SAMPLERS` dispatch in `scripts/sampler_benchmark.py`. Adapter forwards default args; signature matches the existing `_run_*` adapters. End-to-end smoke run on cyclo(Ala)4 with `--samplers mcmm --n_seeds 100`. CLI default becomes `--samplers exhaustive_etkdg,pool_b,mcmm`.

**Outcome.** `_run_mcmm` adapter added to `scripts/sampler_benchmark.py` with the matched-budget mapping `n_steps = max(1, n_seeds // 64)` — at the issue-#11 default 8 × 8 = 64 walkers, this keeps total MMFF work proportional to `n_seeds` so MCMM and exhaustive ETKDG can be run at the same `--n_seeds` value for a fair comparison. Module docstring updated, CLI default `--samplers` is now `exhaustive_etkdg,pool_b,mcmm`. The `grids` argument is ignored by `_run_mcmm` (MCMM doesn't consume the Ramachandran prior). Verified: script imports cleanly, `SAMPLERS = ['exhaustive_etkdg', 'pool_b', 'mcmm']`, CLI help shows the updated default.

### Step 10: Documentation — ✓ complete

Update `src/README.md` and `scripts/README.md`: new module(s), function, sampler entry, plus a section explaining the move set, replica-exchange architecture, and basin-memory bookkeeping. Remove the shared-tail refactor flag.

**Outcome (2026-05-20).** Three READMEs updated. `src/README.md`: added full `## mcmm.py` section (architecture diagram, `BasinMemory` / `MCMMWalker` / `ReplicaExchangeMCMMDriver` / proposer-factory coverage, Kabsch + inertia helpers, sampler-benchmark adapter) and `## concerted_rotation.py` section (DBT + Coutsias closure, `MoveProposal`, `DEFAULT_CLOSURE_TOL`); added the `get_mol_PE_mcmm` narrative + parameter table to the pipeline-functions section; rewrote the old "Clustering" section into "Clustering and dedup" documenting the Step-11 Kabsch metric and Step-17 `dedup_mode='kabsch'|'crest'` AND-test; bumped the stale `rmsd_threshold` rows (0.1 L1 → 0.125 Å Kabsch) and added `dedup_mode` / `energy_threshold_eV` / `rotconst_anisotropy_threshold` rows; removed the obsolete "refactor flag" reference. `scripts/README.md`: added the `mcmm` row to the `sampler_benchmark.py` dispatch table (matched-budget `n_steps = max(1, n_seeds // 64)` + production tuning) and five new script sections (`analyze_basin_sdf.py`, `union_basin_count.py`, `cremp_collapse_test.py`, `sample_cremp_peptides.py`, `cremp_overlap_figure.py`). `src/validation/README.md`: cross-reference paragraphs from `cremp.py` and `make_validation_sets_cremp.py` to the parallel Step-19 scripts that share the pickle directory / `parse_topology`. A full `## References` section was also added to this plan (Saunders 1990, Chang-Guida-Still 1989, DBT 1993, Coutsias 2004, Wu-Deem 1999, GOAT) resolving the shorthand citations used in the docstrings.

---

## Phase 4 — Coverage and metric corrections (post-Step-9 findings)

The Step-9 benchmark surfaced two distinct issues that change the lever-pull priority ordering. Both are documented under "Findings 2026-05-05" below; the steps in this phase fix them.

### Step 11: Kabsch heavy-atom RMSD dedup — ✓ complete

Replaces the normalised-L1 distance metric (`Σ|Δr|/(3·n_atoms)`) used by `BasinMemory.query_novelty*` and `_energy_ranked_dedup` with Kabsch-aligned heavy-atom RMSD in Å. This is GOAT's primary dedup primitive (and CREST/CREMP's), and the Step-9 SDF analysis showed our current metric admits sub-Å duplicate basins on `cremp_sharp` (4 "basins" all within 0.21 Å of each other) and `pampa_small` (16 "basins" with median pairwise RMSD 0.25 Å). Every `n_basins` figure in the Step-9 results is partly noise until this lands.

**Implementation:**

- New helpers `_heavy_atom_kabsch_rmsd(a, b)` and `_heavy_atom_kabsch_rmsd_batch(query, refs)` in `src/mcmm.py`. Pure torch SVD; row-vec convention; standard determinant correction to reject reflections. Both helpers operate on already-sliced heavy-atom coords — slicing is the caller's responsibility.
- `BasinMemory.__init__` gains a `heavy_atom_indices: list[int] | None = None` kwarg. When provided, `query_novelty*` slice the candidate and stored coords to those indices before the Kabsch comparison. `add_basin` still stores full coords (the stored representative is what the proposer needs to apply moves to).
- `_energy_ranked_dedup` gains the same kwarg and same slicing behaviour.
- `_minimize_score_filter_dedup` derives heavy-atom indices from its `mol` argument and passes them through. `get_mol_PE_mcmm` does the same when constructing the shared `BasinMemory`.
- Default `rmsd_threshold` bumps from `0.1` (L1 units) to `0.125` (Å), matching CREMP / CREST / GOAT's conformer-uniqueness contract so `n_basins` is directly comparable to CREMP `uniqueconfs` for the issue-#10 benchmark. The threshold remains a kwarg — pass `rmsd_threshold=0.5` for coarser "chemical basin" clustering when sub-Å wobble shouldn't count as separate basins.

**Tests:**

- New synthetic fixtures whose perturbations survive Kabsch alignment (one-atom out-of-plane displacement; the existing rigid-translation `_line_conformer` collapses to RMSD = 0 under Kabsch and is not useful for boundary tests). Threshold-boundary tests numerically compute the expected Kabsch RMSD as the oracle.
- Heavy-atom slicing: tests pass `heavy_atom_indices=[0, 2]` on a 4-atom fixture and verify that the metric ignores atoms 1 and 3.
- All existing `BasinMemory` invariants (monotonic visit counter, threshold = 0 disables matching, batched-vs-individual equivalence) carry forward.

**Known semantic differences vs. the prior metric:**

- Translation invariance — rigid translations between proposals no longer register as new basins. This is mostly a no-op since MMFF doesn't translate, but removes a class of false positives.
- Heavy-atom only — H positions are noisier than the heavy framework on per-step MMFF runs, so removing them tightens the metric. Combined with the Kabsch alignment, this is closer to CREST's intent.
- No atom-permutation symmetry — methyl flips remain over-counted relative to CREST. Skipping per the deferred plan; the architectural cost (storing mols not just coords) isn't justified by the ~5% refinement.

**Outcome.** `_kabsch_rmsd_pairwise(queries, refs)` lives in `src/mcmm.py`; `BasinMemory` stores full coords but applies the metric over the configured heavy-atom subset; `_energy_ranked_dedup` accepts the same subset and shares the implementation. Defaults updated to 0.125 Å across `get_mol_PE_exhaustive`, `get_mol_PE_pool_b`, and `get_mol_PE_mcmm` (matches CREMP / CREST / GOAT for direct `n_basins` ↔ `uniqueconfs` comparability). Test fixtures replaced (rigid-translate fixtures collapse to RMSD = 0 under Kabsch, so the synthetic `_line_conformer` / `_line_coords` helpers in `tests/test_mcmm.py` and `tests/test_exhaustive_etkdg.py` were swapped for an atom-0-z-displacement pattern; an independent numpy Kabsch reference function serves as the test oracle). 4 new tests in `tests/test_mcmm.py` cover translation invariance, rotation invariance, heavy-atom slicing, and `heavy_atom_indices` validation. **All 173 tests pass** (was 169; +4 new).

**Retro-scoring of the multi-seed SDFs (2026-05-05).** Re-running the new dedup over the existing dumped basin centroids confirms the metric gap was the dominant noise source on the tightly-clustered peptides:

| peptide | old reported (L1, 0.1) | Kabsch 0.125 Å | Kabsch 0.5 Å | Kabsch 1.0 Å |
|---|---|---|---|---|
| cremp_typical | 2 | 2 | 2 | 2 |
| cremp_sharp | 4 | 2 | 1 | 1 |
| pampa_small | 16 | 6 | 2 | 2 |
| pampa_medium | 9 | 5 | 4 | 3 |
| pampa_large | 9 | 9 | 5 | 4 |

`cremp_typical` and `pampa_large` were genuinely diverse in basin space; `cremp_sharp` and `pampa_small` were almost entirely sub-Å wobble counted as distinct basins. At the new 0.5 Å default the basin counts collapse meaningfully on 3/5 peptides — those numbers are now interpretable, the prior ones were noise.

### Step 12: Cartesian-kick proposer alongside DBT — ✓ complete

Adds a second move type that complements DBT's dihedral parameterisation. GOAT's "topology-preserving uphill push" applies random Cartesian forces to atoms, holds bonds and ring sp² angles via constraints, then re-minimises. The v0 implementation here approximates the constraint preservation: an isotropic Gaussian kick (`sigma_kick_a`) applied independently to every atom-coordinate, then MMFF-relaxed. MMFF's bond-stretch and angle-bend terms have steep gradients, so for `sigma_kick_a ≤ 0.3 Å` the relaxation pulls covalent bonds and ring sp² angles back to equilibrium without needing explicit SHAKE-style constraints. Beyond ~0.3 Å, expect occasional MMFF non-convergence — that's the trigger to revisit explicit constraint projection.

Composition with DBT happens via `make_composite_proposer`: a routing layer that picks DBT vs. Cartesian-kick **per walker per step** by sampling against configurable weights. Each sub-proposer is invoked on its subset (preserving its internal batching), and results are reassembled in walker order. `make_cartesian_kick_proposer.stats` exposes `n_proposed`, `n_relax_failures`, `n_relax_successes` analogous to the DBT proposer's stats.

Wired into `get_mol_PE_mcmm` via two new kwargs: `sigma_kick_a` (default 0.1 Å) and `cartesian_weight` (default 0.0 = pure DBT). Set `cartesian_weight=0.5` for a 50/50 mix.

**Outcome.** `make_cartesian_kick_proposer` and `make_composite_proposer` in `src/mcmm.py`. Conditional construction in `get_mol_PE_mcmm`: pure DBT when `cartesian_weight=0.0` (legacy path); composite when > 0. 13 new proposer-level tests in `tests/test_mcmm.py` (factory validation, proposal-count, det_j=1 contract, stats, kick-actually-perturbs-coords, composite routing, weight extremes, walker-order preservation, sub-stats exposure) plus 3 integration tests in `tests/test_get_mol_PE_mcmm.py` (zero-weight skips factory, positive-weight constructs composite, negative-weight raises). **All 189 tests pass** (was 173; +16 new).

Open question (deferred): whether the v0 "kick + relax, no constraint" approach is sufficient for chemically-meaningful exploration on real cyclic peptides, or whether explicit SHAKE-style bond/angle constraints are needed. Trigger for the upgrade: Cartesian-weighted runs underperform pure-DBT on the issue-#10 benchmark, or MMFF non-convergence rate exceeds ~5% per step.

### Step 13: REMD vs. independent-T workers ablation — pending

GOAT runs independent temperature workers; we run replica exchange with swaps. For *minimum-finding* (vs. equilibrium sampling), independent workers may be more efficient — REMD's swap-rejection can prevent the cold replica from adopting newly-found hot geometries. Cheap experiment: same total walker-budget, configurable `enable_swaps: bool = True` on `ReplicaExchangeMCMMDriver` (or a new `IndependentTDriver` that omits swaps entirely). Run on the same 5-peptide benchmark and compare basin counts, time-to-first-novel-basin, and final coverage. If independent wins decisively, drop the REMD layer; if comparable, keep REMD for the marginal mixing benefit.

### Step 14: Rotational-constant anisotropy as a tertiary dedup gate — pending

GOAT uses three independent dedup criteria (RMSD AND energy difference AND rotational-constant anisotropy 1.00–2.50%). Rotational constants are the eigenvalues of the inertia tensor, cheap to compute, and catch "two structures with similar Kabsch RMSD but distinguishable overall shape" — useful when the heavy-atom framework is similar but mass distribution differs (relevant for peptides with bulky vs. compact side chains in different basins). Implementation: `_inertia_eigvals(coords, masses)` helper; basins are considered the same iff Kabsch RMSD < threshold AND any pairwise relative anisotropy difference < 1%. Low priority — only worth adding if Step-11 still over-counts after the metric switch.

### Step 15: Adaptive termination — pending

GOAT's "stop when no new global minimum found in two iterations" trick. For us: track the basin discovery curve in the `ReplicaExchangeMCMMDriver` run loop; if K consecutive sweeps add zero new basins to memory, terminate early. Saves compute on small peptides where the basin set saturates well before the matched-budget step count. ~30 lines plus a `min_progress: int = 0` kwarg that disables the check by default.

### Step 16: CREMP basin-collapse sanity check (diagnostic) — ✓ complete

The `n_basins ≪ uniqueconfs` gap (e.g. cremp_typical: 2 vs 471) has two incompatible interpretations and they need different actions:

1. **Our sampling is deficient** — MCMM never visits the basins CREMP found. Action: improve the proposer (already partly addressed by Step 12).
2. **CREMP overcounts** — CREST + GFN2-xTB classifies sub-Å wobble as distinct conformers; once relaxed via MMFF and deduped at 0.125 Å Kabsch, most "unique" CREMP confs collapse onto the same basin. **In this case `uniqueconfs` is not a meaningful comparison target, the small `n_basins` numbers are honest, and the action is *not* to change MCMM** but to post-process CREMP and retrain the downstream consumers that treat the inflated counts as ground truth.

The experiment that discriminates between them: feed CREMP's own conformers through our `MMFF + MACE + Kabsch-dedup` pipeline and count survivors at each stage.

**Implementation.** Standalone `scripts/cremp_collapse_test.py` (~100 lines). Per peptide:

1. Load the CREMP pickle's full conformer set via the existing `iter_validation_mols(subset_csv, pickle_dir)` in `src/validation/cremp.py` (`rd_mol` with all `uniqueconfs` confs attached, GFN2-xTB-relaxed, plus per-conformer xtb energies in `data["conformers"]`).
2. **Pre-MMFF Kabsch dedup at 0.125 Å** over the raw CREMP geometries via `_energy_ranked_dedup` with xtb energies for the energy-rank step and heavy-atom indices from the mol. Repeat at 0.5 Å.
3. **MMFF + MACE + 5 kT energy filter + Kabsch dedup at 0.125 Å** via `_minimize_score_filter_dedup`. Repeat the dedup at 0.5 Å.

Reports a row per peptide with: `uniqueconfs`, pre-MMFF survivors at 0.125 Å and 0.5 Å, post-MMFF survivors at 0.125 Å and 0.5 Å, e_min (MACE, eV). Test peptides: `cremp_typical (t.I.G.N, 471 uniqueconfs)` and `cremp_sharp (S.S.N.MeW.MeA.MeN, 190 uniqueconfs)`.

**Decision tree.**

| Pre-MMFF @ 0.125 | Post-MMFF @ 0.125 | Verdict & action |
|---|---|---|
| 471 → ~few | ~few | **CREMP overcounts.** No MCMM change. Post-process CREMP and retrain downstream — see "Downstream consequence" below. The benchmark target `n_basins` ↔ `uniqueconfs` is replaced with `n_basins` ↔ `n_basins_CREMP_pipeline`. |
| 471 → 471 | ~few | **MMFF disagrees with GFN2-xTB on basin structure.** CREMP's diverse geometries are real but collapse on MMFF relaxation. Either swap MMFF for MACE-relaxation in the inner loop (compute-expensive) or accept the disagreement and define our basins under MMFF (no MCMM change, but downstream retrain still required for consistency). |
| 471 → 471 | ~hundreds | **Sampling deficient.** Our pipeline preserves CREMP's diversity; our MCMM just never visits those geometries. Action: continue Steps 12+. |
| 471 → ~50 | ~30 | **Mixed.** Partial metric collapse, partial sampling gap. Both fixes apply. |

**Downstream consequence (CREMP-overcounts case).** If the experiment lands here, the work is localised to the validation harness and the downstream pretraining — *not* the MCMM sampler:

1. Re-run `_minimize_score_filter_dedup` on every CREMP pickle to produce a deduped conformer set per peptide. Save as a parallel pickle directory (e.g. `data/raw/cremp/pickle_kabsch_0125/`).
2. Update the validation subset CSV so `uniqueconfs` reflects the post-relaxation Kabsch-deduped count, and so `ground_truth_n_confs` in `sampler_benchmark.py`'s output is the corrected number.
3. Update the peptide-electrostatics pretraining pipeline at `/home/sabari/peptide_electrostatics` to load from the corrected directory. The autoencoder's training set, the Boltzmann-weight Janossy pooling, and the fine-tuning property heads all consume CREMP-style conformer sets — they'd train on inflated targets if the over-counting isn't fixed at the source.
4. Document the threshold choice (0.125 Å Kabsch heavy-atom RMSD) in the project READMEs so future consumers don't reintroduce the inflation.

The expected magnitude: medians may drop by 5–50× if the post-MMFF picture matches what we saw on our own benchmark SDFs (cremp_sharp: 4 → 1 at 0.5 Å; pampa_small: 16 → 2). That changes the data balance for the AE pretraining materially — peptides with ~10× the conformer count would otherwise dominate the training-set Janossy pooling.

This step is a diagnostic experiment, not an MCMM feature. Mark complete when the script has been run on both peptides and the verdict + action plan are documented in the Findings section.

**Outcome (2026-05-05).** `scripts/cremp_collapse_test.py` (~280 lines) loads each CREMP pickle directly, computes counts at five pipeline stages plus pre/post-MMFF MACE e_min. Headline numbers from `results/cremp_collapse_test.csv`:

| stage | cremp_typical (`t.I.G.N`) | cremp_sharp (`S.S.N.MeW.MeA.MeN`) |
|---|---|---|
| CREMP `uniqueconfs` | 471 | 190 |
| Pre-MMFF Kabsch @ 0.125 Å | **237** | **114** |
| Pre-MMFF Kabsch @ 0.5 Å | 84 | 45 |
| Post-MMFF, within 5 kT | 31 | 36 |
| Post-MMFF Kabsch @ 0.125 Å (after 5 kT filter) | **19** | **10** |
| Post-MMFF Kabsch @ 0.5 Å (after 5 kT filter) | 7 | 8 |
| ΔE_min(MACE) post − pre | +0.18 eV | +0.47 eV |
| Our MCMM `n_basins` (Kabsch-aware run) | 2 | 4 |

**Verdict.** Mixed across all three branches of the decision tree, with the dominant signal being **CREMP overcounts**:

1. **Pre-MMFF, same threshold: 471 → 237 (50%) and 190 → 114 (40%).** Half of CREMP's "unique" conformers are within 0.125 Å Kabsch RMSD of another conformer in the same set. They survive CREST's dedup because CREST applies *three concurrent* criteria — RMSD AND energy AND rotational-constant — where any one distinguishing the pair counts them as different. Our pure-Kabsch metric is more aggressive by definition, so a ~2× reduction at the same threshold is *expected*, not a bug. **Action: stop reporting CREMP raw `uniqueconfs` as the benchmark ground truth — define the ground truth as CREMP-conformers-through-our-pipeline.**
2. **MMFF basin-of-attraction collapse: 237 → 19 (12×) and 114 → 10 (11×).** Most CREMP-distinct geometries minimize to the same MMFF basin. This is GFN2-xTB-vs-MMFF disagreement at the relaxer level — chemically real (xtb's sub-basin structure on a peptide isn't a hallucination), but for our MCMM that uses MMFF as the inner-loop minimizer, those sub-basins aren't reachable. **Action: defer; switching to MACE-relaxation in-loop is compute-prohibitive at scale, and the ~10× MMFF collapse is structural to our pipeline.**
3. **Sampling deficiency, but bounded: 19 → 2 (cremp_typical) and 10 → 4 (cremp_sharp).** Our MCMM finds 11–40% of the basins it could find given perfect coverage of MMFF-relaxed CREMP geometries. Real but much smaller than the apparent 471 → 2 / 190 → 4 gap. **Action: continue with Step 12 (Cartesian kicks) and the other ★ levers; the headroom is meaningful but not catastrophic.**

**Pre-MMFF MACE e_min < post-MMFF MACE e_min (by 0.18–0.47 eV) on both peptides.** GFN2-xTB-relaxed geometries are *closer* to MACE's preferred minimum than MMFF-relaxed ones. Confirms the MMFF-vs-MACE relaxer mismatch flagged in the Risks section. Doesn't block this experiment but is worth carrying forward as a v2 consideration: the post-MCMM `_minimize_score_filter_dedup` re-runs MMFF on the basin centroids before MACE-scoring, which slightly degrades the e_min relative to a MACE-relax + MACE-score pipeline.

### Downstream action plan from Step 16 findings

The CREMP-overcounts signal triggers the downstream-retrain branch outlined above:

1. **Update `sampler_benchmark.py`'s ground-truth column.** Add `ground_truth_n_basins_pipeline` (the post-MMFF-+-Kabsch-0.125 Å number) alongside the existing `ground_truth_n_confs` (raw CREMP `uniqueconfs`). The new column is the apples-to-apples target for our `n_basins`. Compute it offline by running `cremp_collapse_test.py` over the validation subset and joining the results into `data/processed/cremp/validation_subset.csv`.
2. **Post-process the CREMP pickle directory** for any downstream consumer that's sensitive to per-peptide conformer counts. Concretely: re-run `_minimize_score_filter_dedup` on every CREMP pickle, save the deduped conformer set as a parallel pickle directory `data/raw/cremp/pickle_kabsch_0125/`. Prep work for the peptide-electrostatics retrain.
3. **Notify the peptide-electrostatics pretraining at `/home/sabari/peptide_electrostatics`.** The autoencoder's training set, the Boltzmann-weight Janossy pooling, and the fine-tune property heads consume CREMP-style sets — they currently train on inflated counts. Without the post-process, peptides with heavily-clustered conformer sets dominate the training-set Janossy pooling 2–10× more than they should. Update the project memory entry for the AE plan to flag the dependency.

These three are localised to the validation harness and the pretraining repo, not the core MCMM sampler.

### Step 17: CREST-style three-criteria dedup as opt-in `dedup_mode='crest'` — ✓ complete

Step 16 showed that CREST/CREMP's `uniqueconfs` is computed under three concurrent criteria — RMSD AND energy AND rotational-constant — where two conformers are merged only when *all three* agree, and any one distinguishing them keeps them separate. Pure Kabsch (our default) is more aggressive and produces ~50% fewer basins on the same geometries (471 → 237 on cremp_typical at the same 0.125 Å threshold).

**Why we want it as opt-in.** For day-to-day work, pure Kabsch remains the right default — the AE pretraining wants the deflated count, our internal definition of "basin" is chemically cleaner, and the in-run BasinMemory dynamics are tuned to it. But for the issue-#10 paper, we'll need to publish a **CREMP-comparable basin count** so reviewers don't read 2 vs 471 as a sampling failure. An opt-in flag keeps the default pipeline intact and gives us the comparable number on demand.

**Locked design decision: MACE energies, not MMFF, drive the energy criterion.** The energy used in the AND-test is the MACE-on-MMFF-relaxed-coord energy that the proposer already computes in stage 4 of `make_mcmm_proposer` and hands to the walker — it's sitting in `BasinMemory._energies` today, unused for the dedup decision. Same for the `energies` arg of `_energy_ranked_dedup` in the post-MCMM filter. **Cost: zero additional compute.** No new MACE calls, no new MMFF calls. MMFF energies are explicitly *not* used here — they'd be noisier (0.1–1 kcal/mol numerical jitter at MMFF convergence, vs MACE's float32 ~0.01–0.05 eV noise floor) and storing them would require running an extra MMFF energy evaluation we currently skip.

**Implementation surface (~80 lines net).**

1. **`_inertia_eigvals(coords, masses)`** in `src/mcmm.py`: build inertia tensor `I_ab = Σ_a m_a (||r_a||² δ_ab - r_a r_b)` from centred coords, return sorted `torch.linalg.eigvalsh(I)`. Three numbers per conformer, translation- and rotation-invariant. ~10 lines.

2. **`BasinMemory` constructor** gains `dedup_mode: Literal['kabsch', 'crest'] = 'kabsch'`, `energy_threshold_eV: float = 0.05`, `rotconst_anisotropy_threshold: float = 0.01`. When mode is `'crest'`, store per-basin rotational constants alongside coords/energies (`self._rotconsts: (K, 3)`); on `add_basin`, compute eigvals from atom masses (derived from atomic numbers via a small RDKit lookup at construction). ~15 lines.

3. **`query_novelty` / `query_novelty_batch`** branch on `dedup_mode`:

```python
rmsd_close = rmsd < self.rmsd_threshold
if self.dedup_mode == 'crest':
    de_close   = (stored_e - query_e).abs() < self.energy_threshold_eV
    rot_close  = max_relative_diff(query_rot, stored_rots) < self.rotconst_anisotropy_threshold
    same_basin = rmsd_close & de_close & rot_close
else:
    same_basin = rmsd_close
# closest among same-basin candidates
```

Pure Kabsch is recovered exactly by `dedup_mode='kabsch'` (the default) — backwards-compatible. ~25 lines change.

4. **`_energy_ranked_dedup` in `src/confsweeper.py`** gains the same three kwargs and the same AND-branch. ~15 lines.

5. **`get_mol_PE_*` entry points** thread `dedup_mode`, `energy_threshold_eV`, `rotconst_anisotropy_threshold` through to `_minimize_score_filter_dedup` and (for `get_mol_PE_mcmm`) `BasinMemory`. Defaults preserve current behaviour. ~10 lines × 3 sites.

6. **Tests:** translation/rotation invariance of `_inertia_eigvals`, AND-criterion branch on a synthetic 3-basin fixture (one pair Kabsch-close-but-energy-distinct, one pair energy-close-but-rotation-distinct, one fully-equivalent), `dedup_mode='kabsch'` matches the existing behaviour bit-for-bit. ~10 new tests.

**Caveats.**

- **MACE float32 noise floor ≈ 0.01–0.05 eV** (≈ 1e-6 relative × the 10⁴–10⁵ eV total energy). CREST's xtb-based 0.05 kcal/mol ≈ 0.002 eV threshold is 25× below this. Default `energy_threshold_eV=0.05` accordingly — still 2× below thermal kT_298 (0.026 eV) so it has discriminatory power, but it doesn't replicate CREST's tightness. Promoting MACE inference to float64 would close the gap (~2× memory, ~2× time) — explicitly *not* worth the cost; document the float32 caveat in the BasinMemory docstring.

- **Rotational-constant anisotropy default 0.01 (1%)** matches CREST's middle of the documented 1.0–2.5% range. Tunable via constructor.

- **In-run BasinMemory dynamics shift under CREST mode.** AND-criterion → looser merging → more basins → each basin gets fewer revisits → Saunders 1/√usage bias is weaker per basin. For paper-comparison runs this is acceptable (the goal is the basin count, not the sampling efficiency); for production runs we keep `dedup_mode='kabsch'`.

- **Need to align `BasinMemory` and `_energy_ranked_dedup` modes** within a single call to `get_mol_PE_mcmm` — running a CREST in-run dedup with a Kabsch post-filter (or vice-versa) would give incoherent counts. Validation in the entry point: if user passes `dedup_mode='crest'` to `get_mol_PE_mcmm`, propagate to both consumers.

**Reporting plan.** When the AE benchmark reports `n_basins`, include both numbers: the default Kabsch (chemical-basin scale) and the CREST-mode (CREMP-comparable). One extra column in `sampler_benchmark.py`'s output, no code reorg. Step 16's `ground_truth_n_basins_pipeline` likewise gets a `_kabsch` and `_crest` variant from `cremp_collapse_test.py` for direct apples-to-apples comparison.

**Trigger to start:** ready to land whenever — small change, well-scoped, low risk to defaults. Most natural after Step 12's Cartesian-kick benchmark settles, so we have one composite benchmark run that exercises both new features end-to-end before the paper draft.

**Outcome.** All five implementation pieces shipped, defaults preserved bit-for-bit:

1. `_inertia_eigvals(coords, masses)` and `_max_relative_eig_diff(query, stored)` in `src/mcmm.py`. Pure torch; translation- and rotation-invariant by construction.
2. `BasinMemory.__init__` gains `dedup_mode`, `energy_threshold_eV` (default 0.05 eV — chosen for the MACE float32 noise floor; documented in the docstring), `rotconst_anisotropy_threshold` (default 0.01), `atomic_numbers` (required only when `dedup_mode='crest'`; masses derived via `RDKit.Chem.GetPeriodicTable().GetAtomicWeight`). `_rotconsts` per-basin tensor populated only in crest mode.
3. `query_novelty` and `query_novelty_batch` now accept `energy=None` / `energies=None` (required in crest mode). AND-criterion under crest mode picks the closest-by-RMSD basin among same-basin matches; pure-kabsch path is unchanged.
4. `_energy_ranked_dedup` in `src/confsweeper.py` gains the same kwargs and the same AND-branch.
5. `_minimize_score_filter_dedup` and all three `get_mol_PE_*` entry points thread the dedup kwargs through. `get_mol_PE_mcmm` propagates the mode to both the in-run `BasinMemory` and the post-MCMM filter so they stay aligned within a single call.

`MCMMWalker.__init__` and `apply_proposal` now pass `energy` to `memory.query_novelty` (harmless in kabsch mode, required in crest). 10 new tests in `tests/test_mcmm.py` cover inertia-eigval translation/rotation invariance, crest-mode constructor validation, energy-required-in-crest-query, the three same/different-basin scenarios under the AND-criterion, kabsch-mode bit-equivalence with the pre-Step-17 path, batched-vs-individual crest equivalence, and per-basin `_rotconsts` storage. **All 199 tests pass** (was 189; +10 new).

Reporting-plan follow-up (dual `n_basins_kabsch` / `n_basins_crest` columns in `sampler_benchmark.py` and `cremp_collapse_test.py`) is a separate small task to take up alongside the next benchmark run.

### Step 18: Post-hoc union of basin sets across proposers — ✓ complete

The Cartesian-kick benchmark (Run B, 2026-05-05) showed that DBT-only and DBT+Cart explore qualitatively different regions of the landscape. On `pampa_large` specifically, DBT-only finds 11 basins on an upper plateau (mean RMSD 5 Å between them, distinctly different geometries) while DBT+Cart finds 1 deep well (3 conformers within 0.4 Å of each other). The post-MCMM 5 kT energy filter then prunes DBT's plateau because Cart's deeper e_min puts those basins 27–31 kT above the global minimum.

This isn't a sampling failure — both proposers are doing useful work. It's a **reporting** problem: the run-relative-energy-window metric is structurally biased against discovery of deeper minima (the deeper the floor, the fewer basins survive the relative window).

**Solution**: post-hoc union analysis. Take the dumped basin centroids from a DBT-only run and a DBT+Cart run, concatenate the conformers onto a single template mol, and report:

- **Discovery diversity** (`n_union_all`): all union conformers deduped at 0.125 Å Kabsch, no energy filter. The pure "how many distinct basins did either method find?" number, independent of where the global e_min lands.
- **Filtered union** (`n_union_filtered_5kT`): standard 5 kT filter relative to union's e_min, then dedup. Comparable to single-method `n_basins`.
- **Per-method split**: `n_dbt_only`, `n_cart_only`, `n_overlap` — answers "what did each proposer contribute that the other missed?"
- **Coverage % vs CREMP-rescored ceiling**: the union count divided by `post_mmff_kabsch_0125` (or `_crest_0125`) from `cremp_collapse_test.py`. The apples-to-apples comparison the paper needs.

**Implementation.** `scripts/union_basin_count.py` (~330 lines). Inputs are two SDF directories (DBT-only and DBT+Cart from prior `sampler_benchmark.py` runs) plus an optional CREMP-collapse CSV for coverage. Energies come from each SDF's `MACE_ENERGY` per-conformer property — no GPU needed. Heavy-atom indices inferred from the loaded mol. The dedup uses the in-tree `_energy_ranked_dedup`, so the metric is identical to what `get_mol_PE_mcmm`'s post-filter applies.

**Run output (existing SDFs, both runs in CREST mode):**

| peptide | n_dbt | n_cart | union_all (discovery) | union_filtered_5kT | CREMP ceiling | coverage_union vs ceiling |
|---|---|---|---|---|---|---|
| cremp_typical | 3 | 5 | **8** | 5 | 30 (kabsch) / 30 (crest) | 17% / 17% |
| cremp_sharp | 5 | 6 | **8** | 5 | 7 (kabsch) / 11 (crest) | 71% / 45% |
| pampa_small | 6 | 3 | **7** | 3 | n/a | n/a |
| pampa_medium | 11 | 17 | **21** | 9 | n/a | n/a |
| pampa_large | 11 | 3 | **14** | 3 | n/a | n/a |

**Interpretation.** Union-all (`n_union_all`) is the right "discovery" metric for the paper:

- `pampa_medium` jumps to 21 basins when we count what each method found independently, vs the single-method maxima of 11 (DBT) or 17 (Cart). DBT finds 4 basins Cart doesn't, plus 17 they share or Cart-only finds.
- `pampa_large` jumps to 14 (vs single-method 11 or 3). DBT's upper plateau (11 basins) and Cart's deep well (3 basins) are entirely disjoint in the discovery sense — they explore different chemical regions.
- `cremp_sharp` reaches **71% of the post-MMFF Kabsch CREMP ceiling** under union — meaningful coverage of what's reachable through our pipeline. Under CREST mode the ceiling is higher (11) so coverage drops to 45%.
- `cremp_typical` is still at 17% of either ceiling. The 8 union basins are all clustered in the same region; CREMP's 30 ceiling implies 22 more distinguishable basins our sampling never visits — clearest remaining headroom.

**Open follow-ups:**

- Per-method split is reported only against the *filtered* union (post-5 kT). Should also report against the discovery union (`union_all`) so we can see, e.g., that Cart contributes 3 unique deep-well basins on pampa_large *before* the filter erases them. Small script update.

**In-pipeline ensemble sampling — deferred follow-up.** The post-hoc union approximates running two MCMM tracks at once with shared state, but isn't the same. Worth thinking through carefully because the deferral decision depends on what the ensemble would actually buy us.

*Architecture sketch.* `EnsembleMCMMDriver(tracks)` where each track has its own `proposer`, `walkers_by_temp`, `kt` ladder, `swap_interval`, and `saunders_exponent`, but all tracks share one `BasinMemory`. Per ensemble step, every track's walkers propose and accept/reject independently; basin writes go to the shared memory, so the Saunders bias on basin K reflects total visits across all tracks. REMD swaps stay *within* tracks (different move spaces don't have a clean meaning for cross-track swap criteria). Total walker budget is the sum across tracks; for parity with a 64-walker single-track run we'd allocate e.g. 32 DBT + 32 Cart. ~250 lines of new driver + tests.

*What it would buy us beyond the current composite proposer:*

1. **Per-track hyperparameter tuning.** The composite proposer routes walkers to DBT or Cart per step but every walker shares the same `kt` ladder, `swap_interval`, and (post-Step-17) `saunders_exponent`. DBT and Cart have measurably different optimal regimes — DBT closure-fails 75% of the time at hot kt while Cart's 0% closure-failure means hot kt is "cheap" and bigger kicks are still useful. With per-track ladders we could push Cart's hot replica to 16× kT_298 while keeping DBT at 8× — currently a one-size-fits-all knob.
2. **Per-proposer Saunders.** Cart's deep-well attraction needed `saunders_exponent=1.0` to escape (Step 17 / 2026-05-06 finding). DBT's moves are smaller and might benefit from gentler suppression (0.5). Different exponents per track let each move type's bias decay at the rate that fits its move size.
3. **Per-track budget allocation.** If we want 2× more Cart walkers than DBT walkers (because Cart is the discovery driver and DBT is finishing the upper-plateau coverage), the current architecture can't express that.
4. **Cleaner diagnostics.** `closure_failure_rate` is a DBT concept; `n_relax_failures` is a Cart concept. The aggregated stats post-fix-from-Run-B work but blur the per-track signal.

*What it would NOT buy us (that I initially thought it would):*

- **Shared Saunders bias across move types** — *already there*. The current composite proposer's walkers all share one BasinMemory; the bias works across DBT and Cart visits already. Run B's regression wasn't from disjoint memories; it was from `1/√usage` decaying too slowly. Step-17 fix.
- **Diversity from cross-method repulsion** — empirically Cart and DBT have `overlap_all = 0` across all 5 peptides under both dedup modes (Run B and 2026-05-06 confirmed). They explore disjoint landscape regions independently. Shared memory has nothing to repel each other from.

*Trigger condition to actually implement:*

- Either: a peptide where the post-hoc union shows meaningful Cart-DBT overlap (so shared-memory sampling would have separated them in real time and increased diversity), OR
- A peptide where DBT and Cart benefit from clearly different `kt_high` / `saunders_exponent` and the single-knob composite proposer can't simultaneously satisfy both.

Currently neither condition is met. The 2026-05-06 results suggest the composite proposer at the recommended hyperparams is doing 95% of what an ensemble would do; the remaining 5% is the per-track tuning flexibility above. Defer until paper review or a peptide class where it matters surfaces.

### Step 19: CREMP overlap statistics at scale — ✓ complete

Step 16 confirmed on cremp_typical (471) and cremp_sharp (190) that CREMP's `uniqueconfs` collapses 2× at the same Kabsch threshold (CREST's three-criteria AND-test inflates) and another 10× under MMFF relaxation (xtb-vs-MMFF basin disagreement). The follow-up question is whether this is **a CREMP-wide phenomenon** or specific to those two peptides — important because the downstream peptide-electrostatics retrain (Step 16's "Downstream action plan") needs to know whether the inflation is uniform across topologies and amino-acid types.

**Sample design — stratified by topology AND per-residue features.**

Two structural axes drive cyclic-peptide conformational entropy and MMFF-vs-xtb disagreement:

1. **Backbone topology** — 4 classes already in CREMP's pipeline (`all-L`, `D-only`, `NMe-only`, `D+NMe`). D-amino acids and N-methylation flip backbone preference patterns; both relaxers' force-field parameters for these are derived from different training data than canonical L peptides, so MMFF-xtb disagreement plausibly varies across topology.
2. **Per-residue features** — specifically **proline** (locks φ to ~−60°, ring-pucker degree of freedom, cis/trans isomerization at the preceding amide) and **glycine** (no side chain, maximal backbone flexibility). Both are conformationally special: Pro restricts the φ-ψ space sharply while Gly opens it. We expect these to drive MMFF-vs-xtb disagreement disproportionately because (a) Pro's ring-pucker basins are sub-kT energy splits MMFF often misses, and (b) Gly's broad φ-ψ acceptance lets xtb find basins MMFF's harder constraints exclude.

**Stratification grid: 16 cells = 4 topology × 2 (has-Pro) × 2 (has-Gly).** Sample target: **100 peptides per cell where the natural pool supports it, capped at the natural max otherwise; minimum floor of 30 per cell** (drop cells smaller than 30 from the analysis). Total expected: **~1500 peptides**, ~9 GPU-hours at the 20 s/peptide we measured. Overnight run.

`num_monomers` (4/5/6) is treated as a *secondary* analysis axis — not stratified on (would push to 48 cells × 30 = 1440, comparable budget but thinner stats), but reported in the summary table so we can see length-dependence post-hoc.

**Sequence parser.** CREMP sequence tokens follow `[A-Za-z]` or `Me[A-Za-z]`: case denotes L (uppercase) vs D (lowercase), `Me` prefix denotes N-methylation. Per-peptide features:

- `has_proline`: any token whose residue letter (after stripping `Me`) is `P` or `p`
- `has_glycine`: same for `G`/`g`
- `topology`: derived as in `make_validation_sets_cremp.py` (`all-L` / `D-only` / `NMe-only` / `D+NMe`) — reused directly so the new sampler agrees with the existing validation_subset's labels

**Implementation: three scripts.**

**1. `scripts/sample_cremp_peptides.py`** (~80 lines, new):

- Reads `data/raw/cremp/summary.csv` (36k peptides).
- Parses each `sequence` to derive `topology`, `has_proline`, `has_glycine`, plus the existing `num_monomers`.
- Joins on `summary.csv` for `smiles`, `uniqueconfs`, etc.
- Stratifies: 16 cells, samples N per cell with deterministic seed, falls back to natural max for under-represented cells, drops cells below the min floor with a logged warning.
- Outputs a CSV with the sampled rows + the derived feature columns, ready to be consumed by `cremp_collapse_test.py`.

**2. `scripts/cremp_collapse_test.py`** (~80 lines net of changes):

- **Peptide-list source** — replace `--peptides` (multiple-flag) with `--peptide_list_csv PATH` accepting any CSV with a `sequence` column. The new sampler's output works as-is; existing manual peptide lists can be CSV-formatted with one column.
- **Resume logic** — at start of `main`, read existing `out_csv` and skip sequences already covered. Mirrors `sampler_benchmark.py`'s `_read_done_set` pattern.
- **Per-peptide error handling** — wrap `_run_one_peptide` in try/except. Some CREMP pickles fail atom-count consistency checks; the at-scale run shouldn't abort on any single peptide.
- **Per-row append + flush** — pairs with resume.
- **`--summarize` subcommand** — read an existing collapse-test CSV (joining the sampler-emitted feature columns) and emit:
  - Distribution histograms of `uniqueconfs / pre_mmff_kabsch_0125` and `pre_mmff_kabsch_0125 / post_mmff_kabsch_0125`.
  - Stratified medians by `(topology, has_proline, has_glycine)`, plus marginals per axis.
  - Stratified medians by `num_monomers` (post-hoc axis).
  - Fraction of peptides where `post_mmff_kabsch_0125 < uniqueconfs / 10`.
  - Plot-ready CSVs for the paper figure (one row per stratum with mean / median / quartiles of the collapse ratios).

**3. `scripts/cremp_overlap_figure.py`** (~150 lines, new):

- Reads the collapse-test CSV (which carries `topology`, `has_proline`, `has_glycine`, `num_monomers` from the sampler) plus the optional summary CSV.
- Renders the 3-panel paper figure described above, saved to PDF + PNG. Uses matplotlib (already a dependency).
- CLI: `--collapse_csv PATH --out_pdf PATH [--out_png PATH]`.
- Encapsulates one function per panel so the layout can be tweaked later without re-deriving the data.
- Computes panel-specific summary statistics and overlays them on the panels (median, IQR boxes, sample counts per cell).

**Paper figure (3-panel):**

- **Panel A**: Boxplot of pre-MMFF Kabsch collapse ratio (`uniqueconfs / pre_mmff_kabsch_0125`) across the 4 topology classes. Tests whether NMe-rich peptides have *more* CREST overcounting than canonical L (NMe creates extra energy/rotation distinctions, hence more AND-test inflation).
- **Panel B**: Boxplot of post-MMFF Kabsch collapse ratio across the 4 Pro/Gly cells (`neither` / `Pro-only` / `Gly-only` / `both`). The Pro-rich and Gly-rich cells should show the largest MMFF-xtb disagreement if our hypothesis is right — the relaxer-mismatch signal concentrates where Pro's ring-pucker or Gly's permissive φ-ψ matters most.
- **Panel C**: Heatmap of post-MMFF Kabsch median collapse ratio across the full 16-cell grid, with cell counts overlaid. Direct visual of whether the inflation is uniform across the (topology × Pro × Gly) grid or concentrated in specific cells.

**Downstream consequence.** If the at-scale collapse pattern generalises, the Step-16 "downstream action plan" applies CREMP-wide. If it concentrates in specific topology / Pro / Gly cells (the more likely outcome given conformational chemistry), the retrain plan can target the affected slices and leave the rest of CREMP alone — saving compute on the post-process pass over the pickle directory.

**Open extensions (deferred):**

- **Per-residue features beyond Pro / Gly** — aromatic (F/W/Y/H), charged (D/E/K/R/H), cysteine-disulfide. Useful if Panel C surfaces a chemistry-class signal we missed. Add as `has_aromatic` / `has_charged` flags in the sampler script (~5 lines each) and re-run if needed.
- **Sample size > 1500** — if any cell hits the 30-peptide floor uncomfortably (e.g., D+NMe with both Pro and Gly is rare), bump per-cell target to 200 (~3000 total, ~17 hours). Trigger: cell counts in the summarize output.

**Outcome (2026-05-20).** All three scripts shipped in `83efdc9` (`scripts/sample_cremp_peptides.py`, the `run`/`summarize` subcommands of `scripts/cremp_collapse_test.py`, `scripts/cremp_overlap_figure.py`) and the at-scale benchmark ran to completion: **1600 peptides, full 16-cell grid at N=100 each, zero failures.** The natural CREMP pool supported the 100-per-cell target everywhere, so no cell dropped below the 30-floor and the secondary-axis `num_monomers` strata are all well-populated. Headline: post-MMFF Kabsch collapse median **27.0×** sample-wide, **85.9 %** of peptides ≥10×, confirming the Step-16 CREMP-overcounts verdict generalises CREMP-wide. Two structural axes drive the inflation — **NMe topology** (NMe-only/D+NMe pre-MMFF median ~1.62 vs all-L/D-only ~1.45, confirming the Panel-A hypothesis) and **ring length** (4-mer 18× → 6-mer 43×). The Pro/Gly hypothesis was *refuted*: Pro/Gly cells collapse *less*, not more, consistently across all four topologies (see Findings 2026-05-20). Divergence from the plan: the 1180 post-partial peptides were processed in round-robin cell order (`overlap_benchmark_sample_roundrobin.csv`) so the topology contrast became interpretable at the midpoint; the run also spanned several detached restarts (host reaped the tracked background tasks — no code fault, see Stage-D notes in `~/.claude/plans/buzzing-chasing-gizmo.md`). Full numbers and the downstream-retrain consequence in the dated Findings entry below.

---

## Findings 2026-05-05

Two benchmark runs in `results/`:

- `sampler_benchmark_drive_sigma0.3_closure_tol0.05_kt_high_4.csv` — the tuned-drive single-seed run.
- `sampler_benchmark_tuned_with_multiseed.csv` — same params plus `n_init_confs=8`.

A third (`sampler_benchmark_kt_high_8.csv`) is in progress in the background to test whether the wider replica spread bridges genuinely-distinct basin pairs.

### Tuned drive vs. baseline

The tuned single-seed run resolved the basin-collapse pathology on `cremp_typical` (1 → 7 basins) and produced 2–7× more basins on every other peptide vs. the `drive_sigma=0.1, closure_tol=0.01, kt_high=2×kT_298K` baseline. `e_min_eV` dropped on 4/5 peptides, confirming the new basins are genuinely lower-energy minima rather than noise. Wall time grew ~30–40% because more proposals close successfully → more MMFF/MACE work per step.

### Multi-seed run: deeper minima but apparent basin-count regression

Multi-seed found lower minima on every peptide (Δe_min from −0.01 to −0.46 eV) but reported *fewer* basins on 4/5 peptides than single-seed. Initial hypothesis was a "deep-basin trap" (kt_high too cold to escape the new minimum). SDF analysis disproved this:

| peptide | reported n_basins | median heavy-atom Kabsch RMSD between confs | true distinct basins |
|---|---|---|---|
| cremp_typical | 2 | 1.42 Å | 2 (real) |
| cremp_sharp | 4 | **0.11 Å** | ~1 (all near-duplicates) |
| pampa_large | 9 | 5.65 Å | ~9 (real) |
| pampa_medium | 9 | 3.05 Å | mixed |
| pampa_small | 16 | **0.25 Å** | ~3–5 (mostly duplicates) |

Energy spreads on all 5 peptides were 3–5 kT — well within `kt_high=4×kT_298K` reach. There is no trap. The dedup metric is letting sub-Å wobble through as distinct basins, inflating `n_basins` on tightly-clustered runs and deflating the comparison on runs that genuinely diversified (because the post-MCMM filter collapses fewer when the spread is wide).

**Conclusion**: every `n_basins` number reported to date is contaminated by metric noise. Step 11 (Kabsch heavy-atom RMSD) is no longer optional — it's the prerequisite for any further benchmark interpretation.

### Cartesian-kick benchmark (Run B, 2026-05-05)

Five peptides, 10000-seed budget, MCMM with `cartesian_weight=0.5` (composite proposer routing 50/50 per walker per step between DBT and the new GOAT-style Cartesian-kick), both `dedup_mode='kabsch'` and `dedup_mode='crest'` reported. Compared against the matched-budget DBT-only run from earlier. Headline:

| peptide | DBT kabsch | DBT crest | DBT+Cart kabsch | DBT+Cart crest |
|---|---|---|---|---|
| cremp_typical | 2 | 3 | **4** | **5** |
| cremp_sharp | 4 | 5 | **8** | 6 |
| pampa_small | 7 | 6 | 4 | 3 |
| pampa_medium | 3 | 11 | 2 | **17** |
| pampa_large | 3 | 11 | **7** | 3 |

**Cartesian kicks unambiguously work as a discovery mechanism:**

- **Deeper MACE e_min on every peptide** (Δ −0.23 to −0.57 eV ≈ 9–22 kT_298): the new move type reaches geometries DBT doesn't.
- **In-run `basins_in_memory` 2–5× larger** under DBT+Cart vs DBT-only (cremp_sharp 72 → 364, pampa_small 70 → 285, pampa_medium 194 → 543, pampa_large 273 → 532). The composite proposer is doing real exploratory work; the post-MCMM filter is what decides what we report.
- **Wall time roughly doubled** (102→189s, ..., 277→512s). Composite makes two batched MMFF + MACE calls per step (one per sub-proposer) and Cart's 0% closure-failure rate means none of its walker-half short-circuits the pipeline.

**Post-filter `n_basins` is muddied by the energy window.** When Cart finds a minimum 0.4–0.6 eV deeper, the 5 kT (≈ 0.13 eV) window now sits at a much lower floor and prunes basins that DBT-without-the-deeper-min would have kept. We see this most clearly on:

- pampa_small kabsch (7 → 4) and pampa_medium kabsch (3 → 2): more in-run exploration, deeper minimum, fewer post-window survivors.
- pampa_large CREST (11 → 3): the largest peptide regresses sharply under crest after Cart finds its deepest well; the basins around the new minimum are similar in energy but distinct enough geometrically to be worth keeping (concentrate at max_bw=0.81).

**CREST mode partially compensates** because three-criteria dedup keeps more pre-filter basins distinct: `pampa_medium` DBT+Cart crest jumps to 17 (the headline number of the run), `cremp_typical` CREST goes 3 → 5.

**Comparison to Step-16 ceiling:**

| peptide | DBT+Cart CREST | CREMP `uniqueconfs` (raw) | Step 16 ceiling (CREMP→our pipeline @ crest) | coverage % |
|---|---|---|---|---|
| cremp_typical | 5 | 471 | 30 | 17% |
| cremp_sharp | 6 | 190 | 11 | 55% |

cremp_sharp closes to 55% of the apples-to-apples ceiling — meaningful headroom but the single-digit percentages we worried about earlier (5/471) reflect noise in the CREMP raw target, not sampling failure.

**Decision points exposed:**

1. **Energy window too tight.** `e_window_kT=5.0` was set when our minima were less deep. Now that Cart pushes minima 0.4 eV further down, the relative-window concept misses an entire stratum of basins above the deep well. Action: bump to 10.
2. **Saunders bias too weak under deep-well attraction.** `1/√usage` decays slowly enough that the deep wells Cart finds keep getting re-visited, dragging max_bw up (cremp_sharp 0.41 → 0.74; pampa_small 0.29 → 0.94). Action: try `1/usage` (lever C15) so re-discovered basins get suppressed faster, restoring exploration pressure away from the new traps.
3. **`pampa_large` CREST regression** (11 → 3) is the only meaningful outlier. Worth visualising the SDFs to see whether the 3 basins are structurally distinct or wobble around one deep well.

### Wider window + stronger Saunders (2026-05-06)

The Run-B post-mortem flagged two decisions: the 5 kT energy window was too tight once Cart pushed minima 0.4 eV deeper, and `1/√usage` Saunders bias decayed too slowly to escape Cart's deep wells. Re-ran the 5-peptide benchmark at `e_window_kT=10`, `saunders_exponent=1.0` (lever C15), `cartesian_weight=0.5`, `dedup_mode=both`. Results in `results/sampler_benchmark_wider_window_saunders1.csv`.

**Sampler-level: Cart contributions doubled-to-9× across 4/5 peptides:**

| peptide | n_cart Run B (5kT, √usage) | n_cart new (10kT, usage) | Δ |
|---|---|---|---|
| cremp_typical | 5 | **10** | 2× |
| cremp_sharp | 6 | **16** | 2.7× |
| pampa_small | 3 | 3 | 1× (no deep-well trap to escape) |
| pampa_medium | 17 | **30** | 1.8× |
| pampa_large | 3 | **26** | **8.7×** ← regression resolved |

**`pampa_large` CREST regression fully resolved.** Run B's 11 → 3 collapse (DBT-only → DBT+Cart) becomes 11 → 26 under the new settings. Cart now finds basins both above and below DBT's plateau region; the deeper-minimum-pulls-window-down problem is gone because (a) the 10 kT window is wide enough to keep DBT's plateau visible, and (b) the stronger 1/usage Saunders bias pushes walkers out of Cart's deep wells before they monopolise sampling.

**Union analysis (DBT-only baseline ∪ new Cart run, `union_basin_coverage_wider_window.csv`):**

| peptide | union @ kabsch | union @ **crest** | CREMP kabsch ceiling | CREMP crest ceiling | coverage_crest |
|---|---|---|---|---|---|
| cremp_typical | 11 | **13** | 28 | 30 | **43%** (was 27%) |
| cremp_sharp | 10 | **21** | 7 | 11 | **191%** (was 100%) |
| pampa_small | 7 | 9 | n/a | n/a | n/a |
| pampa_medium | 36 | **41** | n/a | n/a | n/a |
| pampa_large | 35 | **37** | n/a | n/a | n/a |

**Two paper-grade headlines:**

1. **`cremp_sharp` exceeds the CREMP ceiling under CREST** (21 vs 11). At the same metric we use to project CREMP through our pipeline, our union of DBT-only and DBT+Cart finds ≥10 basins CREMP's exhaustive GFN2-xTB-MD didn't visit. Defensible claim that MCMM-with-composite-proposer is *not* strictly bounded by the CREST baseline.
2. **`cremp_typical` coverage 27% → 43%** under the new tuning. Clearest evidence the deep-well-escape combo (wider window + stronger Saunders) generalises beyond the structural `pampa_large` case to the small-peptide CREMP comparison.

**`overlap_all = 0` preserved across all 5 peptides under both dedup modes.** DBT and Cart continue exploring disjoint regions of the landscape even with the more aggressive Saunders bias — the union's complementarity isn't a Run B artefact.

**Recommended production setting (2026-05-06):**

`drive_sigma_rad=0.3, closure_tol=0.05, kt_high=8×kT_298, n_init_confs=8, cartesian_weight=0.5, e_window_kT=10, saunders_exponent=1.0`.

The previous defaults (`e_window_kT=5, saunders_exponent=0.5`) remain accessible via the CLI for reproducing Run B. The new defaults are not auto-applied — the in-code defaults still match the original Saunders 1990 / 5 kT pipeline conventions; the benchmark `_run_mcmm` adapter is the right place to lock these values for production runs.

### CREMP overlap statistics — Step 19 partial run (2026-05-19)

First read on the at-scale CREMP collapse-test after the May-7 chain stopped mid-run. The Step-19 plumbing (sampler, collapse-test with resume, summarize, figure) was shipped in `83efdc9` but the at-scale chain only got 420 / 1600 peptides done before pausing. Smoke (16 peptides) + partial (420) both flow cleanly through `cremp_collapse_test.py summarize` and `cremp_overlap_figure.py` without code changes — the plumbing is verified end-to-end. Numbers below are read off `results/cremp_overlap_summary_partial.csv`.

**Coverage at the partial point.** Sampling is 100 / cell × 16 cells in `data/processed/cremp/overlap_benchmark_sample.csv`. The May-6 → May-7 chain processed peptides in alphabetical-by-cell order, so partial coverage is heavily skewed:

| topology | has_proline × has_glycine cells done | peptides done |
|---|---|---|
| D+NMe | all 4 cells × 100 | 400 / 400 |
| D-only | 1 cell × 20 | 20 / 400 |
| all-L | none | 0 / 400 |
| NMe-only | none | 0 / 400 |

Topology contrast (Panel A of the paper figure) is not yet renderable — two of four topology boxes are N=0.

**Sample-wide collapse medians (N = 420).**

| dedup mode | stage | median collapse ratio | IQR |
|---|---|---|---|
| kabsch (0.125 Å) | pre-MMFF | 1.61 | 1.42 – 1.85 |
| crest (0.125 Å + 0.05 eV + 1 % rot.) | pre-MMFF | 1.41 | 1.27 – 1.59 |
| kabsch (0.125 Å, 5 kT filter) | post-MMFF | **30.5** | 17.1 – 53.3 |
| crest (0.125 Å + 0.05 eV + 1 % rot., 5 kT filter) | post-MMFF | **27.2** | 15.2 – 45.8 |

The pre-MMFF Kabsch median of 1.61 matches Step 16's two-peptide pattern (471 → 237 = 1.99×; 190 → 114 = 1.67×) and is biased high in the partial because D+NMe alone is over-represented. **CREST's three-criteria AND-test inflates `uniqueconfs` by a median ~1.6× at the same Kabsch threshold across the (partial) sample** — Step 16's headline is replicating at scale, not a two-peptide artefact.

**88.8 % of peptides collapse ≥ 10×** under post-MMFF Kabsch dedup (415 / 420 above the 10× threshold). The CREMP-overcounts signal is the dominant interpretation of `n_basins ≪ uniqueconfs` for nearly all peptides in this partial sample, not just the two Step-16 picks.

**Length axis.** Median post-MMFF Kabsch collapse stratified by `num_monomers`:

| num_monomers | n_peptides | median collapse ratio |
|---|---|---|
| 4 | 165 | 21.8 |
| 5 | 177 | 36.1 |
| 6 | 78 | 42.8 |

Bigger ring → more MMFF-vs-xtb basin disagreement. The effect is monotone and roughly doubles between 4-mers and 6-mers; expected from peptide conformational entropy scaling, but the explicit confirmation at N = 420 is the new datum. Implication for the downstream peptide-electrostatics retrain (Step 16's action plan): the per-peptide Boltzmann-weight inflation factor is itself length-dependent, so the post-process pass over the CREMP pickle directory should not assume a uniform deflation factor.

**Provisional Pro × Gly pattern (D+NMe only).**

| has_proline | has_glycine | n | post-MMFF kabsch median |
|---|---|---|---|
| False | False | 100 | 39.1 |
| False | True | 100 | 27.9 |
| True | False | 100 | 31.1 |
| True | True | 100 | 22.0 |

**This is the opposite direction of the hypothesis in the Step-19 plan.** The plan predicted Pro / Gly cells would collapse *hardest* (xtb's ring-pucker and broad φ-ψ basins MMFF can't reach). Within D+NMe at least, Pro / Gly peptides collapse *less* than no-Pro / no-Gly. Plausible explanation: Pro restricts φ to ~−60° and Gly's broad permissive region is conformationally simpler — both leave CREST less room to find sub-Å wobble variants in the first place, so the "uniqueconfs vs Kabsch" inflation that drives the post-MMFF collapse ratio is smaller. The MMFF-vs-xtb disagreement may still concentrate in Pro / Gly cells in absolute terms; it's just that those peptides also have lower numerators (`uniqueconfs`), so the *ratio* is smaller. **This is provisional — the all-L cells will tell us whether the inversion is a D+NMe artefact or a universal pattern.** Mark for revisit in the at-scale Findings entry once Stage B completes.

**What remains.** Resume the chain on the 1180 remaining peptides (all-L 400, NMe-only 400, D-only 380). At the observed ~25–30 s / peptide rate from the May-6 log, ~8–10 wall hours on the free Quadro GV100. Resume is launched in Stage B of `~/.claude/plans/buzzing-chasing-gizmo.md`; at-scale Findings entry will follow on completion.

### CREMP overlap statistics — Step 19 at-scale, 1600 peptides (2026-05-20)

Full run complete: all 16 cells at N=100 (4 topology × 2 has-Pro × 2 has-Gly), 1600 peptides, zero failures. The remaining 1180 peptides after the 2026-05-19 partial were processed in **round-robin order across the 8 incomplete cells** (via `data/processed/cremp/overlap_benchmark_sample_roundrobin.csv` — identical peptides to `overlap_benchmark_sample.csv`, reordered so all-L and NMe-only filled in parallel rather than alphabetically; this only changed processing order, not the sample). Numbers below from `results/cremp_overlap_summary.csv`; figure in `results/cremp_overlap_figure.{svg,pdf,png}`.

**Sample-wide collapse medians (N=1600).**

| dedup mode | stage | median collapse ratio | IQR |
|---|---|---|---|
| kabsch (0.125 Å) | pre-MMFF | 1.54 | 1.33 – 1.79 |
| crest (0.125 Å + 0.05 eV + 1 % rot.) | pre-MMFF | 1.35 | 1.20 – 1.52 |
| kabsch (0.125 Å, 5 kT filter) | post-MMFF | **27.0** | 14.2 – 49.9 |
| crest (0.125 Å + …, 5 kT filter) | post-MMFF | **24.0** | 12.7 – 43.8 |

**85.9 % of peptides collapse ≥ 10×** under post-MMFF Kabsch. The headline numbers shifted slightly down from the 2026-05-19 partial (post-MMFF kabsch median 30.5 → 27.0; ≥10× fraction 88.8 % → 85.9 %) exactly as expected — the partial was D+NMe-heavy (the topology with the most inflation), and adding the all-L cells (the least) pulled the sample median down. **Step 16's CREMP-overcounts verdict generalises CREMP-wide:** raw `uniqueconfs` is not a usable benchmark ground truth, and the downstream peptide-electrostatics retrain (Step 16 action plan) applies across the whole dataset, not just the two diagnostic peptides.

**Panel A — topology effect (the question the partial couldn't answer).** Pre-MMFF Kabsch collapse by topology:

| topology | pre-MMFF kabsch median | post-MMFF kabsch median |
|---|---|---|
| all-L | 1.41 | 24.0 |
| D-only | 1.48 | 25.3 |
| NMe-only | 1.63 | 28.1 |
| D+NMe | 1.61 | 30.8 |

**The plan's Panel-A hypothesis is confirmed: N-methylated topologies show more CREST AND-test inflation than canonical L** (NMe-only/D+NMe ~1.61–1.63 vs all-L/D-only ~1.41–1.48 pre-MMFF; the ordering carries through post-MMFF). N-methylation adds energy/rotational distinctions that CREST's three-criteria test counts as separate conformers, so NMe-rich peptides have the most inflated raw `uniqueconfs`. The effect is modest (~15 % higher ratio) but monotone and consistent across both the pre- and post-MMFF metrics.

**Length axis (post-MMFF kabsch median):** 4-mer 18.0 (N=692), 5-mer 32.3 (N=656), 6-mer 42.9 (N=252). The monotone length-dependence from the partial holds and is strong — 6-mers collapse ~2.4× harder than 4-mers. The per-peptide inflation factor the downstream retrain must correct for is length-dependent; a uniform deflation factor across CREMP would be wrong.

**Panels B/C — the Pro/Gly inversion is confirmed universal.** Post-MMFF Kabsch median by Pro/Gly bucket (pooled): neither 32.4 > Pro-only 30.0 > Gly-only 23.5 > both 21.6. The provisional D+NMe-only inversion from the partial **holds in every topology row of the 16-cell heatmap** — "neither" is the hardest-collapsing cell and "both" the softest, within all four topologies (e.g. all-L 29.2 → 19.4; D+NMe 39.1 → 22.0). **This is the opposite of the Step-19 hypothesis** that Pro/Gly cells would collapse hardest. The structural reading: Pro restricts φ and Gly's permissive-but-simple φ-ψ both leave CREST less sub-Å wobble to over-discretise in the first place, so the `uniqueconfs / Kabsch` numerator is smaller for those peptides — the MMFF-vs-xtb disagreement may still concentrate there in absolute terms, but the *ratio* doesn't. The single hottest cell is **D+NMe / neither at 39.1** (NMe inflation × no-Pro/Gly wobble compounding).

**Downstream consequence.** The inflation is broadly distributed (not concentrated in a few cells), so the Step-16 "downstream action plan" — post-process the full CREMP pickle directory through `_minimize_score_filter_dedup` and retrain the peptide-electrostatics pipeline on corrected conformer counts — applies CREMP-wide. The two interpretable structure axes for the correction: **NMe topology** (more inflation) and **ring length** (more inflation). The retrain should not assume a uniform per-peptide deflation factor; it should at minimum stratify by length and NMe content. The peptide-electrostatics AE-plan memory already flags this dependency.

### Boltzmann-weighted coverage — metric upgrade + cross-method matching (2026-05-21)

Followup to the at-scale Step-19 work, prompted by PR review: the existing `union_basin_count.py` "coverage" was a pure count ratio (`n_union / cremp_ceiling`), which can exceed 1 and gives equal weight to a thermodynamically-irrelevant basin and the global-minimum basin. Replaced it with a **Boltzmann-weighted coverage** of the CREMP-rescored ceiling: weight each ceiling basin by its 298 K Boltzmann population (MACE energies) and sum the weights of the basins the sampler actually recovers via geometric matching. Three groups of new columns ship alongside the legacy count ratios (kept for continuity):

1. **Ceiling-only recovered mass (headline):** `coverage_bw_ceiling = Σ p_i · covered_C[i]` ∈ [0,1] — fraction of CREMP's thermally-relevant population the sampler found. Plus `coverage_count_matched` (a *true* matched fraction, replacing the misleading count ratio) and `max_missed_bw` (the single biggest thermodynamic gap).
2. **Joint reference (ceiling ∪ sampler):** Boltzmann distribution over the deduped joint set; reports `sampler_mass_joint`, `ceiling_mass_joint`, `missed_ceiling_mass_joint`.
3. **New-basins / discovery:** `n_new_basins`, `new_basin_mass_joint`, `e_min_new_eV`, `delta_emin_vs_ceiling`, `found_new_global_min` — captures basins the sampler found that CREMP missed *and* their thermodynamic weight.

**Implementation plumbing.** `cremp_collapse_test.py` gained `--dump_ceiling_sdf_dir` to persist the post-MMFF Kabsch-deduped ceiling basins (coords + `MACE_ENERGY`) as `<sequence>.sdf` — previously these were computed in stage 4 of `_run_one_peptide` and discarded. `union_basin_count.py` gained `--ceiling_sdf_dir`, `--coverage_kT` (default 298 K), `--match_rmsd` (default 0.5 Å — see below), and the matching/weighting block. `src/validation/cremp.py`'s `symmetric_rmsd` and `calc_coverage` gained a `strip` kwarg so matching is heavy-atom (consistent with the 0.125 Å dedup convention).

Two design lessons surfaced during implementation and are worth flagging for any future cross-method coverage work:

**Lesson 1 — `validation.cremp.calc_coverage`'s tensor pre-filter is rotation-naive.** The pre-filter centers translation but not rotation, so two conformers that are the same basin up to a rotation can have centered-no-rotation RMSD of several Å. With a tight `rmsd_cutoff` (e.g. 0.125 Å) and the default `filter_factor=2.0`, the pre-filter threshold (0.25 Å) rejects every genuinely-matching pair before spyrmsd's rotation-minimizing comparison ever runs — silently returning 0 coverage. For tight thresholds across rotational frames you MUST pass a very large `filter_factor` (the new `union_basin_count._boltzmann_coverage` uses `1.0e6`) so all pairs reach spyrmsd. Basin sets are tens of conformers, so the full pairwise spyrmsd is cheap. This is a latent issue in `calc_coverage` more broadly — at its default `rmsd_cutoff=1.0` × `filter_factor=2.0` = 2.0 Å threshold it's mostly OK on small peptides, but tighter cutoffs need a permissive `filter_factor` or the pre-filter must be replaced with a rotation-minimizing variant.

**Lesson 2 — basin identity and cross-method matching need different thresholds.** The 0.125 Å Kabsch convention defines *distinct basins within a single pipeline* (CREMP / CREST / GOAT and our own dedup). For *cross-method* matching — "did the sampler find the same basin CREMP found?" — that threshold is too strict because MMFF (sampler) and GFN2-xTB (CREMP) relax the same conformational basin to geometries that typically differ by a few tenths of an Å. Direct measurement: on cremp_typical the minimum heavy-atom symmetric RMSD between a DBT-only sampler basin and the closest ceiling basin is 0.36 Å — well above 0.125, well within 0.5. The codebase already uses two-threshold logic implicitly: `validation/cremp_coverage.py`'s benchmark uses `rmsd_cutoff=1.0` (later 0.5), not the 0.125 Å basin-dedup threshold. `union_basin_count._boltzmann_coverage` formalises this with two parameters: `basin_rmsd` (0.125 Å, fixed, defines the sampler basin set) and `match_rmsd` (default 0.5 Å, the cross-method tolerance). The sampler basin set is stable while τ varies — only the cross-method matching decision changes with τ.

**Numbers (cremp_typical, DBT-only ∪ DBT+Cart wider-window union, 11 sampler basins vs 22 ceiling basins):**

| match τ (Å) | cov_bw_ceiling | cov_count_matched | n_new_basins |
|---|---|---|---|
| 0.125 | 0.39 | 0.14 | 8 |
| **0.5** (default) | **0.83** | 0.36 | 6 |
| 1.0 | 1.00 | 1.00 | 1 |

The headline reading: **at the codebase's cross-method convention (τ=0.5 Å), the MCMM sampler's union covers 83 % of cremp_typical's Boltzmann population** — recovering only 36 % of the ceiling basins by count, but those are the high-weight ones the population is concentrated in. The count-ratio metric (44 % at the same denominator, `n_union_filtered / cremp_ceiling_kabsch = 11/22`) gives equal weight to a high-energy and a low-energy basin and so misses the strong signal at the population level.

**cremp_sharp is a stark, real finding.** At every tested τ (0.125 / 0.5 / 1.0), `coverage_bw_ceiling = 0.000`, `n_new_basins = 10` (all 10 sampler basins flagged new), and `max_missed_bw = 0.724` — the dominant CREMP basin alone holds 72 % of the Boltzmann population, and the sampler entirely misses it. Direct diagnostic confirms: minimum heavy-atom symmetric RMSD between any sampler basin and any ceiling basin is **3.1 Å** — the sampler and CREMP explore *geometrically disjoint* regions for this peptide. The count-ratio "coverage" (10/9 = 111 % under Kabsch, 21/11 = 191 % under CREST in the earlier Findings) was misleading on cremp_sharp: it counted lots of basins but at the wrong place on the landscape. This is exactly the kind of failure mode the Boltzmann-weighted metric was designed to surface. cremp_sharp's MCMM tuning is the next thing to look at.

### Reading on GOAT (ORCA's basin-hopping global optimiser)

GOAT (https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html) is a basin-hopping algorithm with: (1) a topology-preserving Cartesian uphill kick (freeze bonds and ring sp² angles), (2) parallel temperature workers without swaps (363/726/1452/2904 K), (3) strict three-way dedup (0.125 Å Kabsch RMSD AND 0.1 kcal/mol AND rotational-constant anisotropy 1–2.5%), (4) adaptive termination on "no new global min in 2 iterations".

Ours and GOAT's pipelines share the perturb-then-minimise loop and parallel temperatures, but diverge on three points worth importing: Kabsch dedup (Step 11, urgent), Cartesian topology-preserving kicks as a complementary move type to DBT (Step 12, medium priority), and the independent-T-worker design (Step 13, ablation). Steps 14–15 are GOAT details we can adopt incrementally.

---

## Risks to instrument from day one

- **DBT acceptance rate on macrocycles is unknown.** Literature reports 5–20 % on linear proteins; cyclic peptides may be lower. Instrument per-replica acceptance rate during runs and surface it in benchmark logs alongside `n_basins` and `max_bw`. If <1 % on `pampa_large`, the fallback is adaptive Δθ amplitude tuning (standard MC adaptation, ~10 lines of code).
- **Closure tolerance as a coverage lever.** `concerted_rotation.DEFAULT_CLOSURE_TOL` (currently 0.01 Å) controls the maximum r5 + r6 displacement norm tolerated as "ring-closed." Relaxing it monotonically improves geometry-acceptance and basin coverage — but only up to a sweet spot near 0.1 Å (the MMFF bond-stretch tolerance). Beyond that, MMFF drift can carry the structure into a different basin than the concerted-rotation move targeted, degrading toward "random perturbation + MMFF basin search" and erasing the algorithmic advantage. Instrument both the closure-pass rate and the post-MMFF RMSD-from-target during benchmark runs so we can detect when the lever is being used productively vs. defeating its own purpose. Couples to Δθ amplitude (relax both together for bigger directed moves).
- **MMFF/MACE basin tier mismatch.** Basin memory dedups at the MMFF level (where minimization happens); final scoring is MACE. Two distinct MMFF basins can collapse to one MACE basin (and vice versa). Instrument `n_basins_mmff` and `n_basins_mace` separately so we see whether tier mismatch is real before deciding whether to add MACE-as-minimizer in v1.
- **Polynomial root-finding numerical stability.** Coutsias 2004's reformulation is meaningfully better-conditioned than DBT 1993's original recipe. Validate against Coutsias's published test cases and watch for branch-selection ambiguity near the closure manifold's boundary.
- **Walker-budget shape vs. exhaustive ETKDG.** 64 × 200 = 12 800 minimizations is the headline matched budget, but two factors complicate the comparison: DBT-rejected moves still incur the MMFF cost, and MACE rescoring runs only on the deduped basin set (not every walker's accepted state). Report effective MACE-equivalent budget alongside raw step count.

---

## Deferred follow-ups

- **Atom-permutation symmetry for dedup.** CREST treats methyl-flip / equivalent-atom permutations as the same basin via permutation matching during RMSD. We don't, and Step 11 (Kabsch heavy-atom RMSD) preserves this gap. The architectural cost is meaningful — would require storing `mol` objects (or at least permutation groups) alongside coords in `BasinMemory`, not just coord tensors. Estimated 5% impact on basin counts based on CREST literature; defer unless benchmark data shows methyl-flip-driven over-counting.

- **Re-run baseline benchmark CSVs after Step 11 lands.** Every `n_basins`/`max_bw`/`eff_n` number in `results/sampler_benchmark*.csv` was collected with the normalised-L1 metric. Once the Kabsch swap is in, those baselines need re-running before any further interpretation.

---

## Lever menu for basin coverage

Brainstormed 2026-04-29 after cremp_typical's "1-basin pathology" diagnostic showed MMFF basin collapse despite 160 Metropolis-accepted moves. Items below are levers we can pull when MCMM under-counts basins relative to CREST. Each is tagged with rough implementation cost and a guess at impact for the small-peptide / basin-collapse case specifically.

Items marked **★** are flagged as initial priorities for implementation if the in-flight tuned-drive experiment (`drive_sigma_rad=0.3, closure_tol=0.05, kt_high=4×kT_298K`) doesn't fully resolve cremp_typical.

### A. Move generation (more diverse proposals)

1. **Larger `drive_sigma_rad`.** Already a tuning knob (default 0.1, currently 0.3 in `_run_mcmm`). Could push to 0.5–1.0 for very aggressive moves. Trade-off: closure failure rate climbs steeply with σ.
2. ★ **Looser `closure_tol`.** Currently 0.01 default, 0.05 in `_run_mcmm`. Push to 0.1 (the upper end of MMFF tolerance per `concerted_rotation.DEFAULT_CLOSURE_TOL`'s docstring). ~1 line.
3. ★ **Multi-window compound moves.** Each walker picks ONE window per step today. Apply DBT to 2–3 windows in sequence per step for bigger backbone perturbations. ~30 lines, low risk.
4. **Side-chain rotamer moves.** Currently backbone-only. Add chi-dihedral perturbations on side chains (Phe/Trp/Leu particularly). ~100 lines, medium impact for peptides with bulky side chains.
5. ★ **Heavy-tailed drive distribution.** Replace `N(0, σ²)` with Cauchy or a Gaussian mixture so occasional big jumps happen even when σ is small. ~5 lines, helps escape deep basins.

### B. Replica exchange (better mixing)

6. **Higher `kt_high`.** Already tunable (default 2×kT_298K, currently 4× in `_run_mcmm`). Push to 8×–16× for very hot replicas; caveat that very-high-T moves become essentially random Cartesian noise.
7. **Finer temperature ladder.** Currently 8 temps × 8 walkers. Try 16 × 4. Trades within-temp diversity for between-temp resolution.
8. ★ **Tighter swap interval.** Currently every 20 steps. Try 5–10 for faster mixing. ~1 line.

### C. Initialization (high-impact for the basin-collapse pathology)

9. ★ **Multi-seed initialization.** *Probably the largest single win for cremp_typical.* All 64 walkers currently start at the same ETKDG conformer, so they explore one basin's neighborhood. Embed K=8 distinct ETKDG conformers, distribute 8 walkers per seed. Each starting basin gets its own walker stack. ~30 lines.
10. ★ **Periodic ETKDG injection.** Every M steps (e.g. 50), replace the worst-performing walker's state with a fresh ETKDG conformer. Prevents permanent walker stagnation. ~50 lines.
11. ★ **Skip seed-MMFF.** Use the unminimized ETKDG geometry as the seed for some/all walkers. MMFF on the seed pulls all walkers into the same minimum basin; skipping lets them relax independently from a wider initial spread. ~5 lines.

### D. Walker structure / adaptive

12. **Heterogeneous walkers.** Different walkers use different `drive_sigma`. Some refine (small σ), some explore (big σ). An implicit replica-exchange in σ-space. ~30 lines.
13. **Walker reset on stagnation.** If a walker hasn't found a novel basin in K steps, reset its state to a random other walker's state (or a fresh ETKDG). Periodic kick. ~30 lines.
14. ★ **Adaptive `drive_sigma` per walker.** Standard MC trick: bump σ up when a walker's acceptance rate is too high; bump down when too low. Per-walker autotuning. ~50 lines.
15. ★ **Stronger Saunders bias.** Currently `1/√usage`. Try `1/usage` (decays faster) so re-discovered basins get suppressed harder, forcing exploration. ~3 lines.

### E. Memory / dedup

16. **Looser `rmsd_threshold`.** Currently 0.125 Å (Kabsch, CREMP-aligned). Raise toward 0.5–1.0 Å to merge sub-basins more aggressively when "chemically distinct minimum" matters more than direct CREMP comparison. ~1 line plus interpretation.
17. ★ **Heavy-atom Kabsch RMSD.** ✓ promoted to Step 11 of the main plan after the multi-seed run's SDF analysis showed sub-Å duplicate basins on `cremp_sharp` and `pampa_small`. No longer a coverage lever — a metric correction.

### F. Bigger lifts

18. **DBT analytical polynomial branches** (Option A from the plan). Multi-branch closure lets a single move jump to topologically distant ring conformations, not just the same homotopy class as the start. Could be transformative for ring-flip basins. **~400–500 lines.** Visualisation of MCMM basin sets vs. CREST basin sets via the multi-SDF dump (Step 10b of this plan) will show whether ring topology is changing in our runs — if not, this is the next major lever after the priorities above.
19. **Side-chain coupling refinement.** Currently we transport side chains as rigid bodies with their backbone parents. MMFF then relaxes them — should already handle internal side-chain dihedrals. Worth verifying via SDF visualisation that side-chain geometry isn't being corrupted by the rigid transport.

### Recommended order if the tuned-drive run doesn't resolve cremp_typical

The priorities marked ★ are roughly ordered: **9 → 4 → 10 → 2/3/5 → 14/15 → 8 → 11**. Multi-seed init (#9) is first because it directly addresses the structural problem that all walkers start in one basin; everything else amplifies what already works. Side-chain rotamer moves (#4) is the second priority specifically for *larger* peptides where backbone-only movement misses a chunk of the basin landscape — we don't know yet whether cremp_typical needs it.

---

## Decision points before coding

1. Land Step 1 (shared-tail refactor) as a standalone commit before any MCMM work, so its behavior-preservation can be verified in isolation against the existing `tests/test_exhaustive_etkdg.py` and `tests/test_pool_b.py` suites.
2. DBT geometry as a standalone `src/concerted_rotation.py` (potentially reusable for any macrocycle MC code) vs. inlined into `src/mcmm.py`. Standalone is preferred — clean separation, the geometry has no MCMM-specific state.
3. Implement DBT from scratch. No published reference exists in the pixi `mace` environment. v0 uses numerical closure (Option B above); the analytical polynomial (Option A) is deferred to a future PR if benchmark data shows multi-branch enumeration is necessary.

All three locked. Steps 1–9 complete (see Progress table at top). Step 10 (final docs in `src/README.md` and `scripts/README.md`) is the last item; after that the branch is ready for the issue-#10 benchmark run.

---

## References

The shorthand citations used throughout this plan and in the `src/mcmm.py` /
`src/concerted_rotation.py` docstrings:

- **Saunders 1990** — Saunders, M.; Houk, K. N.; Wu, Y.-D.; Still, W. C.; Lipton, M.; Chang, G.; Guida, W. C. "Conformations of Cycloheptadecane. A Comparison of Methods for Conformational Searching." *J. Am. Chem. Soc.* **1990**, *112* (4), 1419–1427. DOI: [10.1021/ja00160a020](https://pubs.acs.org/doi/10.1021/ja00160a020). The cycloheptadecane benchmark comparing internal- vs. external-coordinate, systematic vs. random conformational searches.
- **Chang-Guida-Still 1989** — Chang, G.; Guida, W. C.; Still, W. C. "An Internal-Coordinate Monte Carlo Method for Searching Conformational Space." *J. Am. Chem. Soc.* **1989**, *111* (12), 4379–4386. DOI: [10.1021/ja00194a035](https://pubs.acs.org/doi/10.1021/ja00194a035). The primary source for the usage-directed Monte Carlo Multiple Minimum (MCMM) algorithm itself — the `1/usage^p` re-discovery bias `BasinMemory` implements. Co-authored by Chang and Guida, who also co-author Saunders 1990; cite this as the MCMM method primary and Saunders 1990 as the benchmark comparison.
- **DBT 1993** — Dodd, L. R.; Boone, T. D.; Theodorou, D. N. "A Concerted Rotation Algorithm for Atomistic Monte Carlo Simulation of Polymer Melts and Glasses." *Mol. Phys.* **1993**, *78* (4), 961–996. The concerted-rotation backbone move (`concerted_rotation.propose_move`): a correlated change in seven backbone degrees of freedom that leaves the rest of the chain fixed.
- **Coutsias 2004** — Coutsias, E. A.; Seok, C.; Jacobson, M. P.; Dill, K. A. "A Kinematic View of Loop Closure." *J. Comput. Chem.* **2004**, *25* (4), 510–528. DOI: [10.1002/jcc.10416](https://onlinelibrary.wiley.com/doi/10.1002/jcc.10416), PMID 14735570. The better-conditioned reformulation of the six-torsion ring-closure problem as the real roots of a degree-16 polynomial; the geometry behind our numerical closure.
- **Wu-Deem 1999** — Wu, M. G.; Deem, M. W. "Analytical Rebridging Monte Carlo: Application to cis/trans Isomerization in Proline-Containing, Cyclic Peptides." *J. Chem. Phys.* **1999**, *111* (14), 6625–6632. DOI: [10.1063/1.480015](https://pubs.aip.org/aip/jcp/article-abstract/111/14/6625). The Jacobian correction that makes the concerted-rotation move satisfy detailed balance on cyclic peptides (`MoveProposal.det_j`). Companion: Wu, M. G.; Deem, M. W. "Efficient Monte Carlo methods for cyclic peptides." *Mol. Phys.* **1999**, *97* (4), 559–580.
- **GOAT** — Stahn, M.; Grimme, S. et al., ORCA's basin-hopping global optimiser. Manual: [faccts.de/docs/orca/6.0/manual — GOAT](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html). Source of the topology-preserving Cartesian kick (Step 12), the three-criteria dedup (Step 17), and the adaptive-termination idea (Step 15).

> **Verify before paper submission:** the `saunders_exponent=0.5` (`1/√usage`) functional form is attributed to Saunders 1990 in the code docstrings, but the search that produced these references did not confirm the exact `p=0.5` exponent appears verbatim there versus in Chang-Guida-Still 1989 or later formalizations. Check the primary PDFs before citing the specific exponent to a specific paper.
