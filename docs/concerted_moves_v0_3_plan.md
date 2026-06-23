# Concerted-move proposers v0.3

Branch: `17-concerted-dihedral-rotation`. Implements issue #17. Follow-up to issue #15 / v0.2 (`docs/dihedral_kick_v0_2_plan.md`).

## 20260619

This document is the working design for v0.3 of the side-chain dihedral-kick proposer family — a **set of four new "concerted" move types** that attack the cremp_sharp residual flagged in the v0.2 Step 7 Findings (2026-06-17). The v0.2 sweep confirmed that **single-dihedral-per-step moves cannot reach the dominant cremp_sharp ceiling basin** at any combination of `aromatic_wells_deg` and `skip_mmff_relax`: `coverage_bw_ceiling = 0.000` and `max_missed_bw = 0.724023` (identical to 6 decimal places) across all four cells of the v0.2 2×2 ablation matrix at n_seeds=10000. The proposer IS actively useful on cremp_sharp (`new_mass = 8.8 × 10⁻³` at the v0.2 production cell — ~3000× the v0.1 baseline), it just isn't reaching *the* dominant basin. The geometric structure of that basin appears to require a concerted multi-bond move that the v0.2 proposer family cannot make.

**Scope of v0.3.** Per the design conversation locked 2026-06-19, v0.3 benchmarks **four** concerted move types against each other on cremp_sharp + cremp_typical, in this order:

1. **Move A** — concerted χ₁ + χ₂ side-chain rotation on aromatic residues (e.g. NMe-Trp).
2. **Move B** — cis-trans ω isomerization on N-methylated residues.
3. **Move C** — multi-window / larger-window DBT backbone concerted rotation.
4. **Move D** — backbone + side-chain coupled hybrid (DBT window + side-chain dihedral kick in one step).

Each move is implemented + validated in sequence; subsequent moves are *conditional* on the prior move not closing cremp_sharp. A final combined benchmark locks the v0.3 production mix.

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Lock Move A design forks | ✓ complete |
| 2 | Move A: new `make_concerted_dihedral_kick_proposer` factory + `_enumerate_concerted_dihedral_pairs` helper + tests | ✓ complete (5/5 targeted) |
| 3 | Move A: extend `make_default_mcmm_composite` to 4 sub-proposers + thread `concerted_dihedral_weight` through `get_mol_PE_mcmm` + CLI | ✓ complete (76/76 family; full suite pending) |
| 4 | Move A: Validation — cremp_sharp + cremp_typical at n_seeds=10000 | pending |
| 5 | **Decision point.** Does Move A close cremp_sharp (`cov_bw_ceil > 0.10`)? If yes → skip to Step 13. If no → proceed to Step 6. | pending |
| 6 | Lock Move B (cis-trans ω) design forks | ✓ complete (2026-06-22; forks locked, Step 4 ruled out Move A → escalated to B) |
| 7 | Move B: implementation + thread-through | conditional |
| 8 | Move B: Validation + decision point | conditional |
| 9 | Lock Move C (multi-window DBT) design forks | conditional |
| 10 | Move C: implementation + thread-through + Validation + decision point | conditional |
| 11 | Lock Move D (backbone + side-chain hybrid) design forks | conditional |
| 12 | Move D: implementation + thread-through + Validation | conditional |
| 13 | Final combined ablation matrix across all *shipped* moves + production-mix lock | pending |
| 14 | Documentation + PR + release notes | pending |

---

## Goals and constraints

**Primary target.** Close cremp_sharp's `coverage_bw_ceiling` residual — bring it above `0.10` at τ=0.5 (the issue-#12 success threshold from v0.1).

**Diagnostic target.** Each Move's Validation step records `max_missed_bw`, `cov_count`, `n_new`, `new_mass` for both cremp_sharp + cremp_typical. The combined Step-13 ablation shows which Move(s) actually drove which metric and locks the v0.3 production stack accordingly.

**Constraints inherited from v0.2 (issue #15):**

- Same `(mol, conf_ids, energies)` contract through `get_mol_PE_mcmm`'s shared post-sampling tail.
- All v0.2 production defaults preserved unchanged: `cartesian_weight=0.33, dihedral_weight=0.33, p_rotamer_jump=0.30, aromatic_wells_deg=(-90.0, 0.0, 90.0, 180.0), skip_mmff_relax=True`. Each v0.3 Move adds a new kwarg defaulting to off — v0.2 callers see no behavioural change.
- `det_j = 1.0` invariant for all new Moves where it's correct (any new closed-loop concerted move needs its own Jacobian — flagged inside each Move's design-fork table when relevant).
- Strict separation of move types — the new Moves are routed via `make_composite_proposer` alongside the existing DBT + Cartesian + dihedral kick.

**New constraints specific to v0.3:**

- Each Move ships its own factory + tests + CLI flag. The 8-site thread-through pattern (signature, body, `run_one` dispatcher, CLI flag, `main()` signature, log line, `run_one(...)` call site) used in issue-#12 / v0.2 is reused for each Move.
- Composite routing: when more than one v0.3 Move is enabled simultaneously, the composite proposer needs to handle 5+ sub-proposers. `make_default_mcmm_composite` (from v0.1's [src/proposers.py:990](../src/proposers.py#L990)) currently handles 3 (DBT, Cart, dihedral); extending it to 5+ is part of the Step-13 work.

---

## Move A — Concerted side-chain χ₁ + χ₂ rotation

### Background

cremp_sharp has one aromatic residue (NMe-Trp at position 4). Its χ₁ (Cα-Cβ) and χ₂ (Cβ-Cγ_indole) are coupled — the indole orientation in the dominant ceiling basin sits at a particular `(χ₁, χ₂)` joint state that a single-bond move cannot reach because rotating *only* χ₁ leaves χ₂ at the wrong orientation, MMFF/MACE rejects, and the proposer goes back. v0.2's aromatic-aware wells made χ₂ jumps land at the right wells but the v0.2 proposer can only jump *one* bond per step.

### Design choices to lock (Step 1)

These four forks are intentionally left open in this plan and the GitHub issue summary. Step 1 of the work IS locking them, with the conversation captured under a dated Findings entry below.

#### A.1 — Move shape

| option | mechanics | branching factor | notes |
|---|---|---|---|
| **(a1) Independent Gaussian Δχ₁ + Δχ₂** | sample two independent Δχ values from `N(0, σ_chi_rad)` and apply both | inherits the v0.2 `p_rotamer_jump` branching (~3 per bond) | simplest; reuses v0.2 Gaussian/rotamer-jump infrastructure |
| **(a2) Independent rotamer-jump (χ₁ × χ₂) cross-product** | sample `(χ₁, χ₂)` jointly from the cross-product of `rotamer_wells_deg × aromatic_wells_deg` (3 × 4 = 12 joint states) | 12 per residue | inflates rotamer-jump branching; more expressive |
| **(a3) Hybrid — joint Gaussian by default, joint rotamer-jump with probability `p_concerted_jump`** | mirror v0.2's hybrid pattern at the joint level | controllable | most flexible; default for v0.3 Move A |

Running recommendation: **(a3) hybrid**. Same pattern as v0.2 single-bond proposer.

#### A.2 — Eligibility detection

| option | mechanics | edge cases |
|---|---|---|
| **(b1) Atom-c aromaticity flag — match χ₁ if its `c` neighbours an aromatic C** | for each χ₁ candidate, check if its atom `c` (Cβ) has an aromatic neighbour; if so, look up the χ₂ that shares this Cβ | Pro / NMe edge cases — verify on cremp_sharp ground-truth |
| (b2) Per-residue lookup (Trp/Phe/Tyr/His named) | match named residue patterns; less general | requires per-residue handling for modified residues (NMe-Trp etc.) |

Running recommendation: **(b1) atom-c aromaticity flag** — reuses the v0.2 `_classify_rotamer_wells` detection logic.

#### A.3 — Composition with existing proposers

| option | mechanics | trade-off |
|---|---|---|
| **(c1) Per-step probability `p_concerted: float = 0.0`** — when an aromatic bond is picked by the v0.2 proposer, with probability `p_concerted` swap the single-bond move for the concerted (χ₁, χ₂) move on that residue | tied to the existing v0.2 dihedral-kick proposer; one composite | simplest integration; v0.3 default off |
| (c2) New top-level proposer routed by `make_composite_proposer` | concerted move is a separate factory; composite routes per-walker per-step between DBT, Cartesian, single-bond dihedral, concerted dihedral | requires `make_default_mcmm_composite` extension to 4+ proposers | cleaner separation; composite extension lands in Step 13 anyway |

Running recommendation: **(c1) per-step probability** for v0.3 Move A — fastest path to validation. If Move A wins and Moves B/C/D follow, Step 13 promotes it to (c2) as part of the composite-routing refactor.

#### A.4 — Jacobian

A coupled rotation of two open-tree (non-ring) side-chain dihedrals is volume-preserving in the joint `(χ₁, χ₂)` space: `det J = 1.0`. Same property as v0.1 single-bond rotation. No Wu-Deem-style correction needed.

Running recommendation: **`det_j = 1.0`** for the concerted side-chain move; document in the factory docstring with a one-line note. No code change.

### Tests for Move A (Step 2)

Mirror the v0.2 Step-2 / Step-5 test families:

- `test_concerted_dihedral_eligibility_finds_trp_chi1_chi2_pair_on_cremp_sharp` — locate the (Cα-Cβ, Cβ-Cγ) pair via SMARTS, assert the eligibility helper returns it.
- `test_concerted_dihedral_eligibility_returns_empty_on_pure_sp3_peptide` — cycloAla4 / cremp_typical have no aromatic residues; eligibility returns `[]`.
- `test_make_dihedral_kick_proposer_p_concerted_default_zero_preserves_v0_2_behaviour` — default `p_concerted=0.0` matches v0.2 byte-for-byte.
- `test_make_dihedral_kick_proposer_p_concerted_one_always_takes_concerted_move_on_aromatic_bonds` — `p_concerted=1.0` + aromatic bond picked → concerted move executed; assert via stats counter `n_concerted_moves`.
- `test_concerted_dihedral_handles_non_finite_mace` — non-finite MACE energy still produces `success=False` exactly as the single-bond path does.

Targeted run expected: 5 new tests in ~2-3 s. Full suite expected: **446 passed, 8 skipped** (+5 over the v0.2 baseline of 441).

### Validation A (Step 4)

Driver script `scripts/sweep_v0_3_move_a.sh` runs the v0.2 production mix PLUS `p_concerted ∈ {0.0, 0.5, 1.0}` × 2 peptides at n_seeds=10000. Three cells:

- A.1: `p_concerted=0.0` (regression check; matches v0.2 B.4).
- A.2: `p_concerted=0.5` (the headline candidate).
- A.3: `p_concerted=1.0` (pure-concerted ablation).

**Success criteria:**

- cremp_sharp: `coverage_bw_ceiling > 0` at any A.2 or A.3 cell — v0.3 win.
- cremp_typical: no regression — `coverage_bw_ceiling ≥ 0.99` at the A.2 cell.
- Diagnostic: `max_missed_bw` drops below 0.724023 on cremp_sharp at any cell — even partial movement is meaningful evidence.

**Decision point at end of Step 4:**

- If A.2 closes cremp_sharp → lock as v0.3 production; skip to Step 13.
- If A.2 leaves cremp_sharp at zero → proceed to Move B (Step 6+).

---

## Move B — Cis-trans ω isomerization on NMe residues (conditional)

### Background sketch

cremp_sharp has 3 NMe residues (positions 4, 5, 6). NMe peptide bonds strongly prefer cis ω in solution (the lone Me group on N reduces the steric penalty for cis vs trans by ~3-5 kcal/mol relative to the unsubstituted amide). DBT's concerted-rotation closure equation assumes trans ω (ω = 180°); flipping any ω to cis (ω = 0°) is a topology change DBT cannot make.

**Hypothesis.** The dominant cremp_sharp ceiling basin may be a cis-ω state at one (or more) of the three NMe positions that no v0.2 proposer can reach.

### Design forks — LOCKED (2026-06-22, Step 6)

Step 4 ruled out Move A (cremp_sharp `cov_bw_ceil = 0.000` across concerted weights 0.00/0.17/0.34, both dedup modes; dominant ceiling basin `max_missed_bw = 0.724` unreached), so per Step 5 we escalate to Move B. Forks locked:

| fork | locked choice | rationale |
|---|---|---|
| **B.1 Closure scheme** | Numerical closure with ω driven to the cis target — reuse the existing `concerted_rotation` `scipy.optimize.least_squares` solver, fixing the flipped ω at 0° and solving the remaining backbone dihedrals to re-close the macrocycle. | No new analytical derivation needed — this is exactly why v0 chose numerical over the trans-only Coutsias polynomial. Sidesteps the "cis-ω closure requires non-trivial geometry work" risk entirely. Effort: low. Analytical cis-ω closure (B.1c) deferred unless numerical proves insufficient. |
| **B.2 Eligibility** | NMe ω bonds only. Detect via `torsional_sampling.classify_backbone_residues` "NMe" class; the ω is the C(=O)–N amide bond at each NMe backbone nitrogen. | Matches the locked v0.3 scope and the hypothesis (cremp_sharp's 3 NMe residues, pos 4–6). Non-NMe and proline cis ω are rarer and combinatorially heavier — deferred follow-ups. |
| **B.3 Proposal probability** | Uniform single-ω cis↔trans toggle: pick one eligible NMe ω uniformly, set its target to the opposite state (trans→cis or cis→trans). One topology change per step. | Symmetric proposal → clean detailed balance; one move per step matches the conservative pattern of the other proposers; Metropolis + MACE energy naturally selects the favored state, so no proposal bias (B.3b) is needed. Multi-ω (B.3c) deferred. |
| **B.4 Jacobian / reversibility** | Wu–Deem 1999 finite-difference Jacobian, reusing the `MoveProposal.det_j` machinery. | Cascades from B.1a: the cis-ω re-closure adjusts dependent backbone dihedrals through the same non-volume-preserving constrained map as DBT, so the same correction applies. Wu–Deem 1999 is literally the analytical-rebridging method for cis/trans isomerization in cyclic peptides — it covers reversibility of the trans↔cis path. |
| **B.5 Composition with DBT** | 5th independent sub-proposer with its own routing weight (`omega_flip_weight`), dispatched per-walker-per-step by `make_composite_proposer`. | Same pattern as Cartesian-kick / dihedral-kick / concerted-dihedral → clean isolated ablation. The composite refactor to 4–6 sub-proposers is already planned for Step 13. |

### Findings 2026-06-22 — Step 7 geometry: closure cannot tightly close large ω flips

`concerted_rotation.propose_omega_flip` landed (continuation closure reusing
`closure_residual` + Wu–Deem `det_j`) with 7 passing unit tests. The primitive
is correct: ω lands exactly on target, identity/failure/validation paths hold.
**But a geometry probe shows the v0 closure cannot tightly re-close a large ω
flip.** The closure holds r5+r6 fixed (6 constraints) against only 3 free
dihedrals — over-determined — so the best-fit residual grows steeply with the
drive (synthetic 7-atom chain):

| ω drive (rad) | closure residual (Å) |
|---|---|
| 0.05 | 0.015 |
| 0.25 | 0.103 |
| 0.50 | 0.34 |
| 1.0 | ~1.0 |
| π (full cis) | 3.4 |

Continuation tracks the *same* best-fit (no improvement) — warm-starting can't
beat an over-determined minimum. A real trans→cis flip (~π) leaves an ~Å ring
distortion, far above the DBT tolerance (0.01 Å). This is the flagged Move-B
risk materializing at the geometry level: the locked **B.1a** ("reuse the
existing numerical closure") is geometrically insufficient for large flips
because the existing solver exposes only 3 free dihedrals.

**Decision (2026-06-22, resolved):** probe real geometry first; if it doesn't
close, widen the closure (more free dihedrals); do not depend on MMFF.

Real-geometry probe on cremp_sharp conformer 0 (3 NMe ω sites) confirmed the
synthetic finding — the W=7 closure leaves 1.4–2.0 Å residuals on full cis
flips (N@4: 1.43 Å, N@16: 2.02 Å; N@1 was already cis). Real backbones close
large flips no better than the synthetic chain. So per the locked fallback we
**widen the window**. A second probe with a generalised W-atom closure
(fix the last 2 atoms; drive ω centred; W−4 free dihedrals; `trf` solver to
handle the under-determined large-W case) found:

| NMe site | W=7 | W=10 | W=12 | W=14 |
|---|---|---|---|---|
| N@1 | 0.002 | 0.000 | 0.000 | 0.000 |
| N@4 | 1.431 | 0.000 | 0.000 | 0.000 |
| N@16 | 2.017 | 0.000 | 0.000 | 0.000 |

**B.1 re-locked → B.1a-widened: numerical closure on a 10-atom window.** W=10
gives 6 free dihedrals after the ω drive — exactly-determined for the 6 r₈/r₉
position constraints — and closes every real NMe full cis flip to 0.000 Å with
no MMFF. W=10 is the minimal clean close (W=12/14 also work but perturb more of
the ring per move and cost more; W=7 is insufficient). Implementation
consequence: `concerted_rotation` is generalised from a hardcoded 7-atom /
4-dihedral chain to variable window size (DBT keeps using W=7; the ω-flip path
uses W=10), and the ω-flip solver switches to `trf` (handles the
exactly/under-determined shapes `lm` rejects).

### Validation B sketch

`scripts/sweep_v0_3_move_b.sh` similar 3-cell structure × 2 peptides at n_seeds=10000.

---

## Move C — Multi-window / larger-window DBT (conditional)

### Background sketch

The existing DBT proposer enumerates 4-residue backbone windows. A "multi-window" extension would either (i) rotate two disjoint windows simultaneously (joint closure), or (ii) extend the window to 5–6 residues (more degrees of freedom per move). Both are concerted backbone moves beyond what single-window DBT can produce.

**Hypothesis.** The dominant cremp_sharp ceiling basin may be a backbone-topology state that requires a multi-residue concerted backbone perturbation single-window DBT misses.

### Design forks (TBD when Step 9 lands)

Detailed design-fork tables for C.1 (window count / size), C.2 (closure equations — Coutsias 2004 generalised), C.3 (Jacobian — joint Wu-Deem), C.4 (composition with single-window DBT) — left open until Step 9.

---

## Move D — Backbone + side-chain coupled hybrid (conditional)

### Background sketch

A single proposal that rotates a DBT backbone window AND a coupled side-chain dihedral on the residue inside that window. Captures the case where the dominant basin requires simultaneous backbone-and-side-chain motion that the v0.2 + Move A/B/C proposers cannot make as a single step.

**Hypothesis.** If A, B, C all fail individually, the cremp_sharp dominant basin may require a coupled backbone+side-chain move type that no v0.x proposer can produce as a single step.

### Design forks (TBD when Step 11 lands)

Detailed design-fork tables for D.1 (which side-chain dihedral to couple), D.2 (rotation order / joint detailed balance), D.3 (Jacobian — joint coupled), D.4 (composition with other proposers) — left open until Step 11.

---

## Phase 6 — Final combined ablation matrix (Step 13)

Once one or more Moves close cremp_sharp (or all four ship without closing it), the final benchmark sweep runs the combined matrix of all *shipped* Moves on cremp_sharp + cremp_typical at n_seeds=10000. The matrix size depends on which Moves are needed; minimum is `2^k` where `k` = shipped Move count. The v0.3 production mix is whichever cell maximises cremp_sharp `cov_bw_ceil` without regressing cremp_typical below 0.99.

If no Move closes cremp_sharp individually, the matrix tests combinations (Move A + Move B, Move A + Move C, etc.) to find any pair (or larger set) that does.

---

## Phase 7 — Documentation (Step 14)

- Update `src/README.md` `## mcmm.py` → `make_dihedral_kick_proposer` paragraph + the production-mix block: add v0.3 Move(s), the new CLI flags, the new production mix (whatever Step 13 locks).
- Update `scripts/README.md` `### sampler_benchmark.py`: document new CLI flags + production tuning block.
- Append a dated Findings entry to this plan covering the Step-4 / Step-7 / Step-10 / Step-12 / Step-13 sweep results.
- Append a dated Findings entry to `docs/mcmm_plan.md` extending the 2026-06-17 v0.2 cross-reference with the v0.3 numbers.
- Append a dated Findings entry to `docs/dihedral_kick_v0_2_plan.md` (the v0.2 plan) noting that the cremp_sharp residual was attacked and closed/deferred in v0.3.

---

## Risks to instrument from day one

- **Move A may not be the right culprit.** The v0.2 sweep didn't decisively point at any single move type — the cremp_sharp residual is structural at *some* level, and any of A/B/C/D could be the right answer. The validation order is per the user's locked preference; instrument Step-4 (Validation A) diagnostics carefully so we can rule out A definitively if it doesn't close cremp_sharp.
- **Multi-Move composite routing may need refactoring.** `make_default_mcmm_composite` currently handles 3 sub-proposers. Adding Moves A/B/C/D would push the composite to 4-6 sub-proposers depending on which ship. Refactor lands as part of Step 13.
- **Cis-trans ω closure (Move B) requires non-trivial geometry work.** The Coutsias 2004 closure equations assume trans ω throughout the chain. A cis-ω closure path needs new closure-equation derivation. Risk: Move B may be longer than the other Moves.
- **n_seeds=10000 is the validation budget.** Each Move's Validation runs at this budget on 2 peptides — ~30-40 min wall-clock per Move. With all 4 Moves shipping, validation alone is ~2-3 hours; combined ablation matrix in Step 13 adds another ~hour. Plan for ~half a day's worth of GPU time across the v0.3 work.

## Deferred follow-ups

- **2D joint rotamer-well grid for Move A** (option (a2) above). Trigger: if (a3) doesn't close cremp_sharp at any A.2/A.3 cell, the joint-grid variant becomes the next candidate before Move B.
- **Non-NMe cis-ω support for Move B.** v0.3 Move B targets NMe positions only; non-NMe cis ω is much rarer and adds combinatorial complexity. Deferred unless Move B succeeds on NMe and the diagnostics suggest extending.
- **Move C window size > 6.** v0.3 Move C caps at 5–6 residues; longer windows are deferred unless C succeeds at the smaller size and there's residual room to grow.
- **Widen the DBT window from W=7 to W=10 (broader-exploration lever).** The Step-7 ω-flip work generalised `concerted_rotation` to any window size, so DBT could enumerate W=10 windows (6 free dihedrals, exactly-determined closure) instead of W=7 (3 free, over-determined) at essentially zero implementation cost — just emit 10-atom windows and rebuild the per-window downstream sets. Trade-off (user-accepted direction, 2026-06-22): a W=10 DBT move rearranges more of the ring per step, so it explores more broadly at the cost of a lower closure-success / acceptance rate, and the Wu–Deem Jacobian column grows from 3-d to 6-d. Not pulled now because DBT's `drive_sigma_rad` / `closure_tol` and the v0.2 / Step-4 benchmarks are all tuned for W=7; pulling it means re-tuning and re-benchmarking DBT. **Trigger:** if backbone exploration saturates (DBT basin discovery plateaus, or cremp-style coverage stalls with the dominant gap looking like a large-amplitude backbone rearrangement) and we want more aggressive, less-local backbone moves. Could ship as a configurable `window_size` on the DBT proposer (default 7) rather than a hard switch, so both regimes stay available.

---

## Findings

(append-only, dated)

### Move A design choices locked (2026-06-19)

All four open forks resolved. Three at the running recommendations from the tables above; one (A.3, composition with existing proposer) upgraded from the recommended `p_concerted` per-step probability to a separate top-level proposer routed by `make_composite_proposer` — the cleaner architectural choice that fold the composite-routing extension forward into Step 3 rather than delaying it to Step 13.

| fork | locked choice | rationale |
|---|---|---|
| **A.1 Move shape** | Hybrid — joint Gaussian + joint rotamer-jump at probability `p_concerted_jump` | Mirrors v0.2's single-bond hybrid pattern at the joint level. Lets us turn the rotamer-jump dial like in v0.2. |
| **A.2 Eligibility** | Atom-c aromaticity flag — reuse v0.2 `_classify_rotamer_wells` detection | Reuses the v0.2 helper logic; generalises to all aromatic side chains (Trp, Phe, Tyr, His) without per-residue hard-coding. |
| **A.3 Composition** | New top-level proposer routed by `make_composite_proposer` | Cleanest separation: single-bond and concerted moves stay distinct. Requires extending `make_default_mcmm_composite` to handle 4 sub-proposers (DBT, Cart, single-bond dihedral, concerted dihedral) — work that lands in Step 3 (was previously deferred to Step 13). |
| **A.4 Jacobian** | `det_j = 1.0` — joint open-tree rotation in 2D dihedral space is volume-preserving | Same property as v0.1 single-bond rotation; no Wu-Deem-style correction. Document in factory docstring. |

**v0.3 Move A default knobs that fall out:**

- `sigma_concerted_chi_rad: float = 0.5` — joint Gaussian σ in radians for the refinement step (≈28°). Same magnitude as v0.2 single-bond `sigma_chi_rad`.
- `p_concerted_jump: float = 0.3` — probability per walker per step of taking a joint rotamer jump instead of a joint Gaussian step. Same magnitude as v0.2 single-bond `p_rotamer_jump`.
- Joint rotamer wells: the cross-product of `rotamer_wells_deg × aromatic_wells_deg = (-60, 60, 180) × (-90, 0, 90, 180) = 12 joint states`. Computed once per eligible bond pair at factory build time.
- `concerted_dihedral_weight: float = 0.0` (in-code default on `get_mol_PE_mcmm`) — opt-in for v0.3 Move A; defaults preserve issue-#15 / v0.2 behaviour.

**Architectural implications.** A.3's lock changes Step 2 + Step 3 scope:

- **Step 2** = new factory `make_concerted_dihedral_kick_proposer` in `src/proposers.py` (NOT a kwarg on the existing `make_dihedral_kick_proposer`). New eligibility helper `_enumerate_concerted_dihedral_pairs(mol)`. ~5 new tests.
- **Step 3** = extend `make_default_mcmm_composite` from 3 sub-proposers → 4 sub-proposers (the `weight > 0 ↔ proposer not None` contract generalises naturally; sum constraint becomes `cartesian_weight + dihedral_weight + concerted_dihedral_weight ≤ 1`). Thread `concerted_dihedral_weight` through `get_mol_PE_mcmm` + CLI flag on `sampler_benchmark.py`. ~3-5 new tests for the composite extension + integration spies.

Step 1 closes; Step 2 (new factory + eligibility helper + tests) unblocked.

### Step 2 Outcome — make_concerted_dihedral_kick_proposer factory + helper + tests (2026-06-19)

**Eligibility helper** `_enumerate_concerted_dihedral_pairs(mol)` shipped at [src/proposers.py:323-373](../src/proposers.py#L323). Algorithm: from `_enumerate_side_chain_dihedrals(mol)`, filter to χ₂ candidates (atom-c aromaticity flag, reusing v0.2's `_classify_rotamer_wells` logic); for each χ₂ candidate `(_, b₂, _, _)`, find the unique upstream χ₁ whose downstream atom `c₁` equals `b₂` (the shared Cβ). Returns `list[tuple[chi1_quadruple, chi2_quadruple]]`. Smoke-test on cremp_sharp: 8 side-chain dihedrals total, exactly 1 concerted pair (the NMe-Trp χ₁ = (Cα, Cβ), χ₂ = (Cβ, Cγ_indole) coupling) — matches expectation.

**New factory** `make_concerted_dihedral_kick_proposer` shipped at [src/proposers.py:1078-1318](../src/proposers.py#L1078). Mirrors `make_dihedral_kick_proposer`'s per-call pipeline (stage walker coords → joint-rotation → batched MMFF → batched MACE → assemble proposals) but:

- Picks a (χ₁, χ₂) pair from `_enumerate_concerted_dihedral_pairs` per walker per step.
- **Hybrid joint move shape** per A.1 lock: with prob `p_concerted_jump` (default 0.3) take a joint rotamer jump (χ₁_target sampled from `sp3_wells_deg`, χ₂_target sampled from `aromatic_wells_deg`); otherwise take a joint Gaussian step (independent Δχ₁, Δχ₂ each from `N(0, sigma_concerted_chi_rad)`).
- Rotations applied via two successive `rdMolTransforms.SetDihedralDeg` calls — χ₁ first (its rotation moves χ₂'s b and c atoms, so the Gaussian branch reads χ₂'s current angle AFTER applying χ₁).
- 6 stats counters: `n_proposed`, `n_concerted_gaussian_steps`, `n_concerted_rotamer_jumps`, `n_relax_failures`, `n_relax_successes`, `n_mmff_skipped`. Threads v0.2's `skip_mmff_relax` for the same MMFF-snap-back ablation path.
- `det_j = 1.0` per A.4 lock (joint open-tree 2D rotation is volume-preserving).
- 5 validation paths at factory build time (mmff_backend, p_concerted_jump range, sigma sign, empty sp3 wells, empty aromatic wells, plus the no-aromatic-pairs ValueError).

**5 new tests in `tests/test_mcmm.py`** (Step-2 / Move-A section):

- `test_enumerate_concerted_dihedral_pairs_finds_trp_chi1_chi2_on_cremp_sharp` — locks the shared-Cβ invariant (c₁ == b₂) and the χ₂-c-aromaticity contract on cremp_sharp's NMe-Trp pair.
- `test_enumerate_concerted_dihedral_pairs_empty_on_pure_sp3_peptide` — cyclic-Ala-Ser returns `[]`; same contract as `make_concerted_dihedral_kick_proposer` raising at factory build time.
- `test_make_concerted_dihedral_kick_proposer_raises_on_pure_sp3_peptide` — factory raises `ValueError("no enumerable aromatic side-chain")` instead of silently building a no-op proposer.
- `test_make_concerted_dihedral_kick_proposer_validation_kwargs` — exercises the 5 factory-build validation paths.
- `test_concerted_dihedral_kick_pure_rotamer_jump_lands_in_joint_wells` — locks the rotamer-jump branch invariant: at `p_concerted_jump=1.0` with MMFF mocked as no-op, every proposal's post-rotation (χ₁, χ₂) reads back to one of the joint wells exactly (χ₁ in `sp3_wells_deg`, χ₂ in `aromatic_wells_deg`).

Targeted run: **5/5 in 2.74 s.**

### Step 3 Outcome — 4-way composite extension + CLI thread-through + integration tests (2026-06-19)

**`make_default_mcmm_composite` extended to 4 sub-proposers** at [src/proposers.py:1432-1572](../src/proposers.py#L1432). New `concerted_dihedral_proposer` positional + `concerted_dihedral_weight: float = 0.0` keyword. The (weight > 0 ↔ proposer not None) contract generalises naturally; sum constraint becomes `cartesian_weight + dihedral_weight + concerted_dihedral_weight ≤ 1`. The short-circuit-to-single-active-proposer path still fires when only one weight is positive (zero composite-routing overhead preserved). All 16 existing 3-way tests pass unchanged → 4-way extension is fully back-compat.

**`get_mol_PE_mcmm` threaded** at [src/confsweeper.py](../src/confsweeper.py): 3 new kwargs (`concerted_dihedral_weight: float = 0.0`, `sigma_concerted_chi_rad: float = 0.5`, `p_concerted_jump: float = 0.3`) — defaults preserve v0.2 byte-for-byte. New `concerted_dihedral_weight < 0` validation. New routing branch builds the concerted proposer only when `concerted_dihedral_weight > 0` (seed offset `+4_444_444`). `aromatic_wells_deg` is plumbed: when the caller's value is None (preserving v0.1 single-bond behaviour), the concerted factory's call site falls back to the v0.2 locked `(-90.0, 0.0, 90.0, 180.0)` — the concerted move always operates with aromatic wells since it's eligibility-bound to aromatic side chains by construction. `skip_mmff_relax` threads through to both single-bond and concerted factories simultaneously, matching the docstring contract.

**`scripts/sampler_benchmark.py` CLI extended** at [scripts/sampler_benchmark.py](../scripts/sampler_benchmark.py): two new flags `--concerted_dihedral_weight FLOAT` and `--p_concerted_jump FLOAT`. Threaded through 8 sites following the Step-3 / Step-6 pattern: `_run_mcmm` signature + body pass to `get_mol_PE_mcmm`, `run_one` signature + dispatcher, click options, `main()` signature, start-of-run log line (`concerted_dihedral_weight=%.2f p_concerted_jump=%.2f`), and `run_one(...)` call. Verified live via `pixi run python scripts/sampler_benchmark.py --help`.

**7 new composite tests in `tests/test_mcmm.py`** (4-way extension section):

- `test_default_mcmm_composite_validation_negative_concerted_weight` — non-negative-weight guard for the new kwarg.
- `test_default_mcmm_composite_validation_4way_sum_exceeds_one` — sum constraint generalises to 4-way.
- `test_default_mcmm_composite_validation_concerted_weight_without_proposer` / `_concerted_proposer_without_weight` — the contract from the 3-way version applied to the 4th sub-proposer.
- `test_default_mcmm_composite_pure_concerted_short_circuits` — `concerted_dihedral_weight=1.0` returns the proposer directly (no composite overhead).
- `test_default_mcmm_composite_four_way_distribution` — DBT=0.4, cart=0.2, dihedral=0.2, concerted=0.2 across 4000 walkers; each route within ±10 % of expected share.
- `test_default_mcmm_composite_four_way_substats_reachable` — `composite.stats == [dbt.stats, cart.stats, dih.stats, cdih.stats]` in route order.

**4 new integration tests in `tests/test_get_mol_PE_mcmm.py`** (Move-A thread-through section):

- `test_mcmm_concerted_dihedral_weight_zero_skips_concerted_factory` — default zero ⇒ factory not constructed (no MMFF + MACE setup paid).
- `test_mcmm_concerted_dihedral_weight_positive_builds_4way_composite` — `concerted_dihedral_weight=0.5` alone builds DBT + concerted only (other factories not called).
- `test_mcmm_concerted_dihedral_weight_negative_raises` — entry-point validation.
- `test_mcmm_4way_routing_builds_all_four_factories` — `cart=0.2 dih=0.2 concerted=0.2` exercises full 4-way; all four factories called exactly once.

Targeted runs: 7/7 composite + 4/4 integration in 2.97 s. **Combined regression across dihedral + composite + 4-way-routing families: 76/76 in 5.12 s** — no regression on any v0.1 / v0.2 test. Full suite kicked off in parallel; predicted **457 passed, 8 skipped** (+16 over v0.2's 441/8 = 5 Step-2 + 7 Step-3-composite + 4 Step-3-integration). Step 3 closes; Step 4 (Validation A sweep) unblocked.
