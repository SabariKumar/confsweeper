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
| 2 | Move A: new `make_concerted_dihedral_kick_proposer` factory + `_enumerate_concerted_dihedral_pairs` helper + tests | pending |
| 3 | Move A: extend `make_default_mcmm_composite` to 4 sub-proposers + thread `concerted_dihedral_weight` through `get_mol_PE_mcmm` + CLI | pending |
| 4 | Move A: Validation — cremp_sharp + cremp_typical at n_seeds=10000 | pending |
| 5 | **Decision point.** Does Move A close cremp_sharp (`cov_bw_ceil > 0.10`)? If yes → skip to Step 13. If no → proceed to Step 6. | pending |
| 6 | Lock Move B (cis-trans ω) design forks | conditional |
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

### Design forks (TBD when Step 6 lands)

Detailed design-fork tables for B.1 (closure scheme for cis-ω chains), B.2 (per-NMe-residue eligibility), B.3 (proposal probability), B.4 (Jacobian / reversibility), B.5 (composition with DBT) — left open until Step 6.

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
