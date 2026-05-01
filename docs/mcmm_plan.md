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
| 10 | Documentation | pending |

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

### Step 10: Documentation — pending (next)

Update `src/README.md` and `scripts/README.md`: new module(s), function, sampler entry, plus a section explaining the move set, replica-exchange architecture, and basin-memory bookkeeping. Remove the shared-tail refactor flag.

---

## Risks to instrument from day one

- **DBT acceptance rate on macrocycles is unknown.** Literature reports 5–20 % on linear proteins; cyclic peptides may be lower. Instrument per-replica acceptance rate during runs and surface it in benchmark logs alongside `n_basins` and `max_bw`. If <1 % on `pampa_large`, the fallback is adaptive Δθ amplitude tuning (standard MC adaptation, ~10 lines of code).
- **Closure tolerance as a coverage lever.** `concerted_rotation.DEFAULT_CLOSURE_TOL` (currently 0.01 Å) controls the maximum r5 + r6 displacement norm tolerated as "ring-closed." Relaxing it monotonically improves geometry-acceptance and basin coverage — but only up to a sweet spot near 0.1 Å (the MMFF bond-stretch tolerance). Beyond that, MMFF drift can carry the structure into a different basin than the concerted-rotation move targeted, degrading toward "random perturbation + MMFF basin search" and erasing the algorithmic advantage. Instrument both the closure-pass rate and the post-MMFF RMSD-from-target during benchmark runs so we can detect when the lever is being used productively vs. defeating its own purpose. Couples to Δθ amplitude (relax both together for bigger directed moves).
- **MMFF/MACE basin tier mismatch.** Basin memory dedups at the MMFF level (where minimization happens); final scoring is MACE. Two distinct MMFF basins can collapse to one MACE basin (and vice versa). Instrument `n_basins_mmff` and `n_basins_mace` separately so we see whether tier mismatch is real before deciding whether to add MACE-as-minimizer in v1.
- **Polynomial root-finding numerical stability.** Coutsias 2004's reformulation is meaningfully better-conditioned than DBT 1993's original recipe. Validate against Coutsias's published test cases and watch for branch-selection ambiguity near the closure manifold's boundary.
- **Walker-budget shape vs. exhaustive ETKDG.** 64 × 200 = 12 800 minimizations is the headline matched budget, but two factors complicate the comparison: DBT-rejected moves still incur the MMFF cost, and MACE rescoring runs only on the deduped basin set (not every walker's accepted state). Report effective MACE-equivalent budget alongside raw step count.

---

## Deferred follow-ups

- **Replace the dedup metric with heavy-atom Kabsch RMSD.** `_energy_ranked_dedup` and `BasinMemory` currently use a normalised L1 distance over all atoms (`Σ|Δr| / (3 × n_atoms)`, threshold 0.1) with no alignment and no symmetry handling. CREST and CREMP's `uniqueconfs` are defined by Kabsch-aligned heavy-atom RMSD with a 0.125 Å threshold (and CREST adds atom-permutation symmetry on top). Three differences vs. our metric:
  1. **No alignment in ours** — translations/rotations between walker proposals would falsely register as different basins. Rare in practice (MMFF doesn't translate) but possible.
  2. **All atoms incl. H in ours** — H atoms move much more than the heavy framework during dihedral changes, making our metric *more sensitive* than CREST's for the same move.
  3. **No symmetry correction in ours** — methyl flips / equivalent-atom permutations are different basins to us, same basin to CREST.

  Differences 2 and 3 bias toward over-counting basins relative to CREST. Difference 1 also biases toward over-counting. So the metric mismatch is unlikely to be the cause of *under-counting* (e.g. cremp_typical's 1-basin pathology), but it does mean our `n_basins` numbers can't be directly compared to CREMP's `uniqueconfs` for benchmark purposes.

  **Trigger condition for the actual fix**: if the tuned-drive experiment (`drive_sigma_rad=0.3, closure_tol=0.05, kt_high=4×kT_298K`) still produces 1 basin on cremp_typical, the metric likely matters and we should change it. If basins appear with the tuned drive, the metric is mostly cosmetic and we can defer.

  **Implementation plan when triggered**:
  1. Add `_heavy_atom_kabsch_rmsd(coords_a, coords_b, heavy_atom_indices)` in `confsweeper.py`. Pure torch (svd-based Kabsch). Skip atom-permutation symmetry (would require storing mols, not just coords; the 5% refinement isn't worth the architectural cost).
  2. Replace the L1 distance computation in `_energy_ranked_dedup`.
  3. Update `BasinMemory.query_novelty` / `query_novelty_batch` to use the same heavy-atom Kabsch RMSD.
  4. Bump default `rmsd_threshold` from 0.1 (normalised-L1 units) to 0.125 Å (Kabsch-RMSD units).
  5. Update the threshold-arithmetic tests in `test_exhaustive_etkdg.py`, `test_pool_b.py`, `test_mcmm.py` to use the new metric.
  6. **Re-run baseline benchmark CSVs** — existing `n_basins` numbers were collected under the old metric and aren't directly comparable to new ones.

  Affects all three samplers (`get_mol_PE_exhaustive`, `get_mol_PE_pool_b`, `get_mol_PE_mcmm`) for benchmark consistency.

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

16. **Tighter `rmsd_threshold`.** Currently 0.1. Try 0.05 for finer basin resolution; reveals sub-basins that 0.1 collapses. ~1 line plus interpretation.
17. **Heavy-atom Kabsch RMSD.** Already covered in Deferred follow-ups above. Trigger condition is the same (tuned-drive run still produces 1 basin).

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
