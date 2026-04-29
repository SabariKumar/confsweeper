# MCMM sampler with replica exchange — implementation plan

Branch: `10-benchmark-non-etkdg-samplers-against-exhaustive-etkdg-on-cyclic-peptides`. Implements issue #11 (sub-issue of #10).

This document is the working design for the third sampler in the issue-#10 benchmark: Multiple Minimum Monte Carlo (Saunders 1990) with replica exchange and Dodd-Boone-Theodorou concerted-rotation backbone moves. It captures the implementation order, the testable invariants at each layer, and the open decisions that block coding. Folded into `src/README.md` (or deleted) before the PR ships.

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

### Step 1: Refactor shared post-sampling tail

Extract the MMFF → batched MACE score → 5 kT energy filter → `_energy_ranked_dedup` → non-centroid prune block from `get_mol_PE_exhaustive` and `get_mol_PE_pool_b` into a private `_minimize_score_filter_dedup(mol, calc, hardware_opts, ...)` helper in `src/confsweeper.py`. Both existing functions become thin wrappers around their respective Phase 1 sampler plus the shared tail.

The refactor flag was deliberately deferred from the Pool B PR because two callers did not justify the risk of touching production code. Three callers do. MCMM cannot be cleanly written without it — otherwise the tail becomes a third copy and the next sampler (CREST-fast or REMD) becomes a fourth.

Behavior must be byte-equivalent. Verification: re-run `tests/test_exhaustive_etkdg.py` and `tests/test_pool_b.py` unmodified; all tests pass. Remove the refactor flag from `src/README.md` and from `get_mol_PE_pool_b`'s docstring.

### Step 2: DBT concerted rotation geometry

New self-contained `src/concerted_rotation.py`. No MC, no MACE, no torch — pure numpy. Standalone module so any other macrocycle MC code in this project (or downstream) can pick it up without depending on the MCMM driver.

Implements:

- 7-atom-window parameterization. Inputs: positions of 7 consecutive backbone atoms, choice of drive dihedral, drive perturbation Δθ. Outputs: new positions of the 5 inner atoms (outer two preserved exactly).
- Polynomial closure solver. Coutsias 2004 reformulation rather than DBT 1993 original — same algorithm, better-conditioned numerics. Up to 16 real solutions; we select the branch closest to the current geometry.
- Wu-Deem 1999 Jacobian. |det J| of the closure-constraint Jacobian with respect to the drive angle, used to weight the proposal probability for detailed balance.

Test invariants:

- Zero perturbation maps to identity (positions unchanged to machine precision).
- Outer atom positions preserved to 1e-9 after non-zero perturbation.
- Analytical Jacobian matches finite-difference numerical Jacobian to 1e-6 across a sweep of geometries.
- Polynomial degree check on hand-constructed cases (known number of real solutions).
- Coutsias 2004 published test cases reproduced.

This is the longest and trickiest piece of the implementation. Land it as its own commit so review focuses on the geometry. Pure-Python performance is fine for v0 — the bottleneck of the full pipeline is MMFF, not move proposal. Profile-driven port to torch or C only if profiling later shows DBT itself is hot.

---

## Phase 2 — MCMM algorithm core

All in a new `src/mcmm.py`. Builds on Step 2 (`concerted_rotation.py`) and Step 1 (refactored shared tail).

### Step 3: Backbone window enumeration

Given a cyclic peptide mol, return all valid 7-atom windows entirely inside the macrocycle. Reuses `torsional_sampling._BACKBONE_SMARTS` and `get_backbone_dihedrals` for the residue-level structure.

Tests on cyclo(Ala)4 and cyclo(Ala)6: verify expected window count and that every returned window's outer atoms are inside the ring.

### Step 4: Basin memory

`class BasinMemory` backed by torch tensors `[K, n_atoms, 3]` for stored basin coordinates and `[K]` for usage counts.

Operations:

- `add_basin(coords, energy)`: append to the tensor, initialize usage = 1.
- `query_novelty(coords) -> (idx_or_None, distance)`: batched normalized-L1 against all K basins (same metric as `_energy_ranked_dedup`); return the closest basin within the threshold or `None`.
- `record_visit(idx)`: increment usage[idx].
- `acceptance_bias(idx) -> float`: 1/√usage[idx] (Saunders form); 1.0 when idx is None (novel basin).

Tests: threshold behavior (boundary cases match `_energy_ranked_dedup`), usage counter monotonicity, batched novelty query against many proposals returns vector of indices in one call.

### Step 5: Single-walker MCMM driver

Sequential reference implementation. One step is: propose DBT move (random window, random drive dihedral, Gaussian Δθ) → MMFF minimize → query basin memory → Metropolis accept/reject with Saunders 1/√usage bias multiplied into the standard min(1, exp(−ΔE/kT)) factor → update memory and walker state.

Tests:

- T = 0 always rejects worse-energy proposals.
- T = ∞ always accepts (modulo the Wu-Deem Jacobian factor).
- Basin memory grows monotonically across steps.
- Saunders bias suppresses re-discovery: a walker repeatedly visiting the same basin sees acceptance probability decay as 1/√k where k is the visit count.

### Step 6: Parallel walkers (batched)

N walkers proposing concurrently. Each walker contributes one conformer to a shared mol; MMFF runs on the full set in one `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` call. Basin memory is shared across walkers; each walker's accept/reject decision is independent given the post-MMFF energy.

Verification: small-N batched results match the single-walker reference run sequentially with the same RNG seeds.

### Step 7: Replica exchange

8 temperatures geometric 300 K → 600 K. N walkers per temperature (default 8 → 64 walkers total). Swap attempts between adjacent temperatures every 20 steps via standard Metropolis on ΔE × Δβ. Replica indices are tracked so the basin memory's per-temperature provenance can be inspected if needed (though the memory itself is shared across temperatures).

Tests: swap acceptance probability matches the analytical Metropolis value across many independent trials; replica ordering is preserved after swaps (no state mixing bugs).

---

## Phase 3 — Integration

### Step 8: `get_mol_PE_mcmm` entry point

New function in `src/confsweeper.py`. Pipeline: enumerate backbone windows → initialize walkers from a seed conformer (e.g., one ETKDG conformer minimized with MMFF) → run replica-exchange MC for the configured step budget → MACE-rescore the basin set → call refactored `_minimize_score_filter_dedup` for the final filter/dedup/prune. Returns `(mol, conf_ids, energies)`.

Integration tests mirror `tests/test_pool_b.py`: mock GPU stages (MMFF, MACE, basin-memory if needed) and verify the contract — return shape, energy ordering, conformer-count consistency, zero-conformer safety, etc.

### Step 9: Sampler benchmark wiring

Add `"mcmm"` row to `SAMPLERS` dispatch in `scripts/sampler_benchmark.py`. Adapter forwards default args; signature matches the existing `_run_*` adapters. End-to-end smoke run on cyclo(Ala)4 with `--samplers mcmm --n_seeds 100`. CLI default becomes `--samplers exhaustive_etkdg,pool_b,mcmm`.

### Step 10: Documentation

Update `src/README.md` and `scripts/README.md`: new module(s), function, sampler entry, plus a section explaining the move set, replica-exchange architecture, and basin-memory bookkeeping. Remove the shared-tail refactor flag.

---

## Risks to instrument from day one

- **DBT acceptance rate on macrocycles is unknown.** Literature reports 5–20 % on linear proteins; cyclic peptides may be lower. Instrument per-replica acceptance rate during runs and surface it in benchmark logs alongside `n_basins` and `max_bw`. If <1 % on `pampa_large`, the fallback is adaptive Δθ amplitude tuning (standard MC adaptation, ~10 lines of code).
- **MMFF/MACE basin tier mismatch.** Basin memory dedups at the MMFF level (where minimization happens); final scoring is MACE. Two distinct MMFF basins can collapse to one MACE basin (and vice versa). Instrument `n_basins_mmff` and `n_basins_mace` separately so we see whether tier mismatch is real before deciding whether to add MACE-as-minimizer in v1.
- **Polynomial root-finding numerical stability.** Coutsias 2004's reformulation is meaningfully better-conditioned than DBT 1993's original recipe. Validate against Coutsias's published test cases and watch for branch-selection ambiguity near the closure manifold's boundary.
- **Walker-budget shape vs. exhaustive ETKDG.** 64 × 200 = 12 800 minimizations is the headline matched budget, but two factors complicate the comparison: DBT-rejected moves still incur the MMFF cost, and MACE rescoring runs only on the deduped basin set (not every walker's accepted state). Report effective MACE-equivalent budget alongside raw step count.

---

## Decision points before coding

1. Land Step 1 (shared-tail refactor) as a standalone commit before any MCMM work, so its behavior-preservation can be verified in isolation against the existing `tests/test_exhaustive_etkdg.py` and `tests/test_pool_b.py` suites.
2. DBT geometry as a standalone `src/concerted_rotation.py` (potentially reusable for any macrocycle MC code) vs. inlined into `src/mcmm.py`. Standalone is preferred — clean separation, the geometry has no MCMM-specific state.
3. Implement DBT from scratch following Coutsias 2004. No published reference exists in the pixi `mace` environment that wraps the algorithm cleanly, and writing it from scratch is the only path. Estimated scope: ~300–500 lines of geometry code plus tests.

Once those three are locked, work begins at Step 1.
