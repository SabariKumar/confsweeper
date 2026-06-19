# Monte Carlo Sampling in confsweeper — Evolution Summary

## The original confsweeper pipeline

Before any Monte Carlo machinery, confsweeper was a *randomized, single-shot* conformer generator built for throughput on a GPU. Given a molecule (SMILES or RDKit mol), it ran a fixed three-stage pipeline with no feedback loop between stages:

**Stage 1 — GPU ETKDGv3 embedding (the conformer pool).**
The entry point `get_mol_PE_batched` generates a large pool of conformers (e.g. `n_confs=1000`) in one batched call using nvmolkit's GPU implementation of RDKit's ETKDGv3. ETKDGv3 is *experimental-torsion knowledge distance geometry*: it samples random distance matrices consistent with the molecular graph, embeds them into 3D, and biases torsions toward values observed in the Cambridge Structural Database. Every conformer is an independent random draw — there is no walk, no memory, no acceptance test. For macrocyclic peptides, `get_embed_params_macrocycle()` swaps in ring-closure-aware ETKDGv3 flags; all other molecules use `get_embed_params()`. This stage is pure stochastic coverage: throw many seeds and hope the relevant basins get hit.

**Stage 2 — GPU Butina clustering (deduplication).**
The raw pool is heavily redundant (many seeds collapse to the same geometry). A single GPU Butina pass (nvmolkit) deduplicates it: it computes pairwise RMSD, then greedily forms clusters around the most-connected centroids within an RMSD radius, and keeps one representative per cluster. This happens *before* any neural-network inference, so the expensive scoring step only ever sees unique geometries. Butina is unsupervised and threshold-driven — it knows nothing about energy, only geometric similarity.

**Stage 3 — MACE-OFF energy scoring.**
Each surviving Butina representative gets one MACE-OFF forward pass to assign a potential energy (returned in eV). MACE-OFF is a pretrained foundation-model interatomic potential; here it is used purely as a fast, accurate single-point energy estimator, not for dynamics or optimization. The output is an SDF carrying the representative coordinates plus a per-conformer energy annotation.

**The macrocycle-specific two-pool extension (`etkdg+torsional`).**
For cyclic peptides, ETKDGv3 systematically under-samples certain backbone (φ, ψ) regions that GFN2-xTB is known to visit. The optional second pool (Pool B), in `src/torsional_sampling.py`, fills those gaps: it draws target (φ, ψ) angles from a **CREMP-derived Ramachandran prior** — specifically the non-zero but low-probability cells — and generates conformers by backbone-dihedral-*constrained* distance geometry. Pool B is merged into the same mol as Pool A and the two are deduplicated together in a single Butina pass, so the added coverage costs no extra scoring. This is still a randomized sampler — it just biases *where* the random draws start.

**The key limitation that motivated everything after.**
This whole pipeline is *embarrassingly parallel but memoryless*. Coverage scales only with how many independent seeds you spend, and on hard cyclic peptides it **saturates**: certain low-energy basins are separated from the ETKDG-reachable region by barriers that no amount of independent random embedding will cross. The conformer distribution collapses toward one-hot against the reference. This is a **connectivity problem**, not a density problem — and it is what the Monte Carlo work was built to solve.

---

The remainder of this document covers the *adaptive walk* samplers that followed. A single benchmark — Boltzmann coverage against the CREMP ceiling distribution — drives every architectural step. Each version adds or refines exactly one proposer type aimed at the specific basin the previous version provably could not reach, with ablation matrices to confirm which move mechanism is responsible.

## Phase 1 — MCMM baseline (issue #11, `docs/mcmm_plan.md`)

Introduced a full Metropolis Monte Carlo Multiple-Minimum sampler in `src/mcmm.py` with move geometry in `src/concerted_rotation.py`.

- **Move type:** Dodd–Boone–Theodorou (DBT) / Coutsias-2004 **concerted backbone rotation** on 7-atom ring windows — drive one backbone dihedral, then *numerically* solve ring-closure for the other three (scipy `least_squares`, rather than the algebraic degree-16 polynomial) so the macrocycle stays closed.
- **Detailed balance:** Wu–Deem 1999 Jacobian (`det J`, computed by finite differences) corrects the forward/reverse asymmetry of the closure move.
- **Acceptance:** Metropolis `exp(−ΔE/kT)` × Saunders `1/√usage` re-discovery penalty × `|det J|`, with energies from MMFF-minimize → MACE.
- **Basin memory** (`BasinMemory`): shared across walkers, dedup by **Kabsch heavy-atom RMSD** at 0.125 Å (matching CREMP's `uniqueconfs` contract); also a CREST-style 3-criteria mode (RMSD ∧ energy ∧ rotational-constant).
- **Three stacked drivers:** single `MCMMWalker` → `ParallelMCMMDriver` (N walkers sharing memory, one batched GPU MMFF/MACE call per step) → `ReplicaExchangeMCMMDriver` (temperature ladder with adjacent-T swaps for barrier crossing).
- **Key benchmark finding:** the original normalized-L1 distance metric admitted sub-Å duplicates; switching to Kabsch RMSD fixed basin counting and validated the "CREMP overcounts" hypothesis.

## Phase 2 — side-chain dihedral kicks (issue #12, `docs/dihedral_kick_plan.md`)

DBT moves the backbone but cannot cross side-chain rotameric barriers (e.g. NMe-Trp χ₁/χ₂, 10–15 kcal/mol indole barriers). This phase refactored proposer factories out of `mcmm.py` into `src/proposers.py` and added two more move types plus a router.

- **Dihedral-kick proposer:** rotate one side-chain dihedral per step, **hybrid** — Gaussian Δχ most of the time, discrete jump to a rotamer well with `p_rotamer_jump=0.3`. `det J = 1` (open-tree rotation is volume-preserving).
- **Cartesian-kick proposer:** GOAT-style isotropic Gaussian position kick, topology preserved by MMFF relax.
- **Composite proposer / routing:** `make_composite_proposer` dispatches each walker to a sub-proposer per step by weight; `make_default_mcmm_composite` wires the DBT + Cartesian + dihedral 3-way mix.
- **Result:** cremp_typical reached **0.991** Boltzmann coverage (+0.16 over baseline). **cremp_sharp stayed at 0.000** — one ceiling basin holding 72% of the population was missed by every proposer, flagged as a structural wall and deferred.

## Phase 3 — aromatic-aware wells + MMFF ablation (issue #15, `docs/dihedral_kick_v0_2_plan.md`)

Two hypotheses for the cremp_sharp wall, tested as a 2×2 ablation.

- **Aromatic-aware rotamer wells:** when the downstream pivot atom is aromatic, use `(−90, 0, 90, 180)°` wells instead of the sp3 `(−60, 60, 180)°` default (correct for indole χ₂).
- **`skip_mmff_relax` flag:** bypass the Stage-2 MMFF relax that was dragging rotamer jumps back over barriers before MACE saw them.
- **Result:** neither fix alone helped (each slightly *regressed* cremp_typical), but **both together synergize to 0.997** on cremp_typical — the branch record. cremp_sharp still 0.000, but `new_mass` rose ~3000×, proving the proposer now finds thermodynamically real basins there, just not the dominant one.

## Phase 4 — concerted multi-dihedral moves (issue #17, current branch, `docs/concerted_moves_v0_3_plan.md`)

Conclusion from Phase 3: cremp_sharp's dominant basin is geometrically inaccessible to *any* single-dihedral-per-step move. The v0.3 plan proposes four conditional move types, validated in sequence.

- **Move A — concerted χ₁+χ₂** joint side-chain rotation (12-state cross-product of the two well sets) — the immediate work in progress.
- **Move B — cis/trans ω isomerization** on NMe peptide bonds (a topology change DBT's trans-assuming closure cannot make).
- **Move C — multi-window / larger-window DBT.**
- **Move D — coupled backbone + side-chain hybrid move.**

Moves B/C/D are each gated on Move A failing to close cremp_sharp.

## Phase summary table

| Phase | Issue | Core contribution | cremp_typical | cremp_sharp | Status |
|-------|-------|-------------------|---------------|-------------|--------|
| 0. Original pipeline | — | Randomized GPU ETKDGv3 → Butina → MACE (memoryless) | — | saturates / one-hot | Baseline |
| 1. MCMM baseline | #11 | Adaptive walk, basin memory, replica exchange, DBT concerted rotation | metric corrected via Kabsch (0.125 Å) | — | Complete |
| 2. Single-dihedral kick | #12 | Side-chain rotatable bonds, hybrid Gaussian + rotamer jump, composite routing | 0.991 | 0.000 | Complete |
| 3. Aromatic-aware + no-MMFF | #15 | Per-bond aromatic wells, MMFF ablation, synergistic interaction | 0.997 | 0.000 (new_mass +3000×) | Complete |
| 4. Concerted multi-bond | #17 | Concerted χ₁+χ₂, cis-ω, multi-window DBT, backbone+side-chain hybrid | — | — | In progress |

## Related thread

`src/torsional_sampling.py` provides Ramachandran-prior constrained distance-geometry embedding (the original Pool B). It complements the MC work on the pool-generation side (biasing where conformers start) rather than the walk side (how the sampler moves between basins).
