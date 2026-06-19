"""
MCMM proposer factories: DBT concerted rotation, GOAT-style Cartesian kick,
and a composite that routes per-walker per-step across them. Extracted from
`src/mcmm.py` so the proposer factories live in their own module — needed
because the next addition is a side-chain dihedral-kick proposer
(`make_dihedral_kick_proposer`) lands alongside the existing two. See
`docs/dihedral_kick_plan.md` for the design.

Each factory returns a `batch_propose_fn(coords_list) -> list[tuple]` matching
the contract `ParallelMCMMDriver` / `ReplicaExchangeMCMMDriver` consume:
`(new_coords, new_energy, det_j, success)` per walker in walker order.

`mcmm.py` re-exports `make_mcmm_proposer`, `make_cartesian_kick_proposer`, and
`make_composite_proposer` from this module for back-compat — pre-refactor
imports like `from mcmm import make_mcmm_proposer` continue to work.
"""

import math

import ase
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Helpers consumed by the proposers but kept in mcmm.py because BasinMemory,
# walkers, and drivers depend on them too. Imported at module load: mcmm.py's
# top-of-file definitions (lines ~50 / ~94) are in place before any code in
# mcmm.py triggers a `from proposers import …` re-export at its bottom, so
# the cycle resolves without partial-import issues.
from mcmm import _ordered_macrocycle_atoms, enumerate_backbone_windows

# ---------------------------------------------------------------------------
# Side-chain enumeration — used by the DBT proposer for full-mol moves
# ---------------------------------------------------------------------------


def _backbone_atom_set(mol: Chem.Mol) -> set:
    """
    Return the set of atom indices that lie on the macrocycle backbone.

    Backbone atoms are the atoms on the largest ring as identified by
    RDKit ring perception (see `_ordered_macrocycle_atoms`). Used as
    the "stop set" for the side-chain BFS in `_side_chain_group` —
    atoms outside this set are side-chain candidates; atoms inside
    are part of the ring and must not be crossed during the BFS.

    Params:
        mol: Chem.Mol : a cyclic molecule with a macrocycle ring
    Returns:
        set[int] : ring atom indices; empty if no qualifying ring exists
    """
    return set(_ordered_macrocycle_atoms(mol))


def _side_chain_group(mol: Chem.Mol, atom_idx: int, backbone_atom_set: set) -> set:
    """
    BFS from `atom_idx` through non-backbone bonds.

    Returns the set of atom indices reachable from `atom_idx` without
    crossing any backbone atom (the starting atom itself is excluded
    from the result; only its non-backbone neighbours and beyond are
    included). For a backbone atom in a cyclic peptide, this returns
    the side chain attached at that residue — Hα, Cβ, side-chain
    branches, etc. — without leaking into adjacent residues' side
    chains via the macrocycle.

    For a non-backbone starting atom, the result is the connected
    non-backbone component containing it (minus the starting atom),
    which is rarely what callers want; pass a backbone atom.

    Params:
        mol: Chem.Mol : input molecule
        atom_idx: int : starting atom (typically a backbone atom)
        backbone_atom_set: set[int] : atoms forming the macrocycle ring;
            traversal does not cross these
    Returns:
        set[int] : reachable atom indices, excluding `atom_idx` itself
    """
    side_chain: set = set()
    queue: list = []
    for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nb_idx = nb.GetIdx()
        if nb_idx not in backbone_atom_set:
            queue.append(nb_idx)
    while queue:
        idx = queue.pop()
        if idx in side_chain:
            continue
        side_chain.add(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in backbone_atom_set or nb_idx in side_chain:
                continue
            queue.append(nb_idx)
    return side_chain


def _compute_window_downstream_sets(
    mol: Chem.Mol, window: tuple, backbone_atom_set: set
) -> list:
    """
    Compute the full-mol atom indices that should rotate per dihedral
    when a DBT move acts on `window`.

    For dihedral k around bond (window[k+1], window[k+2]):
      - Window backbone atoms strictly downstream: window[k+3..6].
      - Side chains of window[k+2..6] (the pivot atom k+2's side chain
        rotates with the local frame at the pivot; downstream backbone
        atoms' side chains rigidly follow their parents).

    The pivot atom window[k+2] itself stays on the rotation axis and
    does not move. Side chains of window[0..k+1] are upstream of the
    bond and do not rotate.

    Params:
        mol: Chem.Mol : input molecule
        window: tuple[int, ...] : 7 atom indices, in chain order
        backbone_atom_set: set[int] : from `_backbone_atom_set(mol)`,
            passed in to avoid recomputation across windows
    Returns:
        list of 4 frozenset[int] : per-dihedral rotation set, suitable
            for `concerted_rotation.apply_dihedral_changes_full_mol`
    """
    side_chains = {
        atom_idx: _side_chain_group(mol, atom_idx, backbone_atom_set)
        for atom_idx in window
    }
    downstream_sets: list = []
    for k in range(4):
        rotated: set = set()
        rotated.update(side_chains[window[k + 2]])
        for j in range(k + 3, 7):
            rotated.add(window[j])
            rotated.update(side_chains[window[j]])
        downstream_sets.append(frozenset(rotated))
    return downstream_sets


# ---------------------------------------------------------------------------
# Side-chain rotatable-bond enumeration — used by the dihedral-kick proposer
# ---------------------------------------------------------------------------


# RDKit's canonical rotatable-bond SMARTS: single bond between two
# non-triple-bonded heavy atoms with degree > 1 and not in a ring. Matches
# pairs (b, c) for each rotatable bond. Cached at module load.
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def _heavy_degree(mol: Chem.Mol, atom_idx: int) -> int:
    """
    Return the number of non-hydrogen neighbours of `atom_idx` in `mol`.

    Heavy-atom degree of 1 indicates a terminal heavy atom (e.g., the
    methyl carbon of a CH3, the Oγ of a Ser hydroxyl, the Nδ of an Asn
    amide head). Rotating around the bond to its parent is 3-fold
    symmetric or H-dominated — conformationally trivial — so the
    side-chain dihedral enumeration filters these out.

    Params:
        mol: Chem.Mol : input molecule
        atom_idx: int : atom index
    Returns:
        int : count of non-H neighbours
    """
    return sum(
        1
        for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nb.GetAtomicNum() != 1
    )


def _pick_flanking_atom(mol: Chem.Mol, atom_idx: int, exclude: int) -> int | None:
    """
    Pick a flanking atom for a dihedral defined around a bond ending at
    `atom_idx`. Used to assemble the `(a, b, c, d)` four-atom tuple given
    the bond `(b, c)`: pick `a` ∈ neighbours(b) − {c}; pick `d` ∈
    neighbours(c) − {b}.

    Selection rule: lowest-index non-H neighbour preferred so the
    enumeration is deterministic across runs. Falls back to any neighbour
    (H included) if no non-H candidate is available — this branch is
    defensive (it shouldn't fire for the rotatable bonds the caller
    enumerates) but guards the result. Returns `None` when `atom_idx`
    has no neighbour besides `exclude`.

    Params:
        mol: Chem.Mol : input molecule
        atom_idx: int : atom whose neighbour to pick
        exclude: int : atom index to exclude from candidates
    Returns:
        int | None : neighbour atom index, or None if none available
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    non_h = sorted(
        nb.GetIdx()
        for nb in atom.GetNeighbors()
        if nb.GetIdx() != exclude and nb.GetAtomicNum() != 1
    )
    if non_h:
        return non_h[0]
    any_nb = sorted(nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() != exclude)
    return any_nb[0] if any_nb else None


def _enumerate_side_chain_dihedrals(
    mol: Chem.Mol,
) -> list[tuple[int, int, int, int]]:
    """
    Enumerate side-chain rotatable dihedrals of a cyclic peptide as
    four-atom tuples `(a, b, c, d)`, where `(b, c)` is the rotation-axis
    bond. The dihedral-kick proposer samples one of these tuples per
    walker per step (see `docs/dihedral_kick_plan.md`, Step 3).

    A bond `(b, c)` is included iff:

      - It matches RDKit's canonical rotatable-bond SMARTS
        `[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]` (single, non-ring, no
        triple-bond endpoints, no terminal-degree endpoints).
      - It is NOT a backbone bond — defined as both endpoints lying on
        the macrocycle ring (`_backbone_atom_set`). DBT already owns
        backbone dihedrals; strict separation keeps the move sets
        disjoint.
      - Neither endpoint has heavy-atom degree 1 (filters Cα-CH3 of Ala,
        N-CH3 of N-methylated residues, side-chain methyls and -OH/-NH2
        terminal heavies — all conformationally trivial under heavy-atom
        rotation, since the moving group is symmetric or H-dominated).

    Flanking atoms `a` (on b's side) and `d` (on c's side) are chosen by
    lowest-index non-H neighbour, falling back to any neighbour. Order
    over the rotatable-bond matches follows RDKit's deterministic
    iteration order so the enumeration is reproducible across runs.

    Params:
        mol: Chem.Mol : a cyclic peptide with explicit Hs. The
            macrocycle must be ring-perceived (implicit on mol
            construction); side chains and Hs are otherwise handled
            transparently.
    Returns:
        list[tuple[int, int, int, int]] : per-dihedral `(a, b, c, d)`
            atom indices, ready for `Chem.rdMolTransforms.SetDihedralDeg`
            or `GetDihedralDeg`. Empty list if no side-chain rotatable
            bonds qualify (e.g., cyclic homo-alanine: every side chain
            is a methyl, every rotatable-bond match is filtered).
    """
    backbone_atoms = _backbone_atom_set(mol)
    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)
    result: list = []
    for b, c in matches:
        # Strict separation from DBT: backbone bonds belong to DBT's territory.
        if b in backbone_atoms and c in backbone_atoms:
            continue
        # Filter methyl-type rotations (3-fold symmetric, no useful diversity)
        # and -OH / -NH2 terminal-heavy rotations (H-dominated).
        if _heavy_degree(mol, b) == 1 or _heavy_degree(mol, c) == 1:
            continue
        a = _pick_flanking_atom(mol, b, exclude=c)
        d = _pick_flanking_atom(mol, c, exclude=b)
        if a is None or d is None:
            continue
        result.append((a, b, c, d))
    return result


def _classify_rotamer_wells(
    mol: Chem.Mol,
    dihedrals,
    aromatic_wells_deg,
    sp3_wells_deg,
):
    """
    Build the per-bond rotamer-well lookup consumed by the dihedral-kick
    proposer's rotamer-jump branch. For each rotatable bond
    `(a, b, c, d)` produced by `_enumerate_side_chain_dihedrals`, check
    whether the downstream endpoint `c` is aromatic. Aromatic-anchored
    bonds get the `aromatic_wells_deg` well set (e.g. NMe-Trp χ₂ —
    indole-edge rotamers near ±90°); all others get `sp3_wells_deg`
    (standard sp3-χ₁ wells near ±60°, 180°).

    When `aromatic_wells_deg is None`, every bond gets `sp3_wells_deg`
    — the v0.1 (issue #12) behaviour, preserved as the default of
    `make_dihedral_kick_proposer` so existing callers see no change.

    Why atom c (not b)? Atom c is the "downstream" endpoint of the
    rotated bond — the atom that anchors the side-chain subtree being
    swung around. For NMe-Trp χ₂ this is the indole Cγ; for sp3-χ₁ on
    Ser / Thr / Asn / Asp / etc. this is the side-chain sp3 carbon
    bound to the rotated β atom. The aromaticity of `c` is the cleanest
    structural signal distinguishing the two regimes (locked Step-1
    design choice A, see docs/dihedral_kick_v0_2_plan.md Findings
    2026-06-15).

    Params:
        mol: Chem.Mol : the cyclic peptide with explicit Hs.
        dihedrals: list[tuple[int, int, int, int]] : output of
            `_enumerate_side_chain_dihedrals(mol)`.
        aromatic_wells_deg: tuple[float, ...] | None : well centres in
            degrees for aromatic-anchored bonds. None disables per-bond
            classification entirely (every bond gets `sp3_wells_deg`).
        sp3_wells_deg: tuple[float, ...] : well centres in degrees for
            non-aromatic (sp3) bonds. Default sp3-χ₁ wells `(-60, 60,
            180)` come from the make_dihedral_kick_proposer caller.
    Returns:
        list[np.ndarray] : one np.float64 array per dihedral, in the
            same order as `dihedrals`. Each entry is the well-set
            array the rotamer-jump branch should sample uniformly from
            for the corresponding bond.
    """
    sp3_arr = np.asarray(sp3_wells_deg, dtype=np.float64)
    if aromatic_wells_deg is None:
        return [sp3_arr] * len(dihedrals)
    aromatic_arr = np.asarray(aromatic_wells_deg, dtype=np.float64)
    wells_per_bond = []
    for (_, _, c, _) in dihedrals:
        if mol.GetAtomWithIdx(int(c)).GetIsAromatic():
            wells_per_bond.append(aromatic_arr)
        else:
            wells_per_bond.append(sp3_arr)
    return wells_per_bond


def _enumerate_concerted_dihedral_pairs(mol: Chem.Mol):
    """
    Enumerate `(χ₁, χ₂)` rotatable-bond pairs on aromatic side chains.
    Each pair is the input to `make_concerted_dihedral_kick_proposer`
    (v0.3 Move A); rotating both dihedrals together is the concerted
    move that v0.2's single-bond proposer cannot make.

    Eligibility: for each rotatable side-chain dihedral
    `(a₂, b₂, c₂, d₂)` from `_enumerate_side_chain_dihedrals` whose
    downstream endpoint `c₂` is aromatic (the v0.2 atom-c aromaticity
    flag), find the *upstream* dihedral `(a₁, b₁, c₁, d₁)` whose `c₁`
    equals `b₂` — i.e., χ₁'s downstream Cβ is χ₂'s upstream Cβ. For
    standard amino-acid topology this uniquely identifies the χ₁ that
    rotates the side chain holding the χ₂ aromatic ring. NMe-Trp is the
    canonical case on cremp_sharp: χ₁ = (Cα, Cβ), χ₂ = (Cβ, Cγ_indole).

    Reuses the v0.2 atom-c aromaticity detection from
    `_classify_rotamer_wells` (the locked Step-1 design choice A.2 of
    v0.3 — see docs/concerted_moves_v0_3_plan.md Findings 2026-06-19).

    Params:
        mol: Chem.Mol : cyclic peptide with explicit Hs.
    Returns:
        list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]
            : pairs of (χ₁_quadruple, χ₂_quadruple). χ₂'s c atom is
            guaranteed aromatic. Empty list if no aromatic side-chain
            rotatable bonds exist on `mol` (e.g., cyclo-Ala, cremp_typical
            — both pure-sp3 peptides), in which case
            `make_concerted_dihedral_kick_proposer` will raise at factory
            build time.
    """
    dihedrals = _enumerate_side_chain_dihedrals(mol)
    # χ₂ candidates: bonds whose downstream atom c is aromatic.
    chi2_bonds = [d for d in dihedrals if mol.GetAtomWithIdx(int(d[2])).GetIsAromatic()]
    pairs = []
    for chi2 in chi2_bonds:
        _, b2, _, _ = chi2  # b2 is Cβ — the atom χ₁ rotates around with Cα.
        for chi1 in dihedrals:
            if chi1 == chi2:
                continue
            _, _, c1, _ = chi1
            if c1 == b2:
                pairs.append((chi1, chi2))
                break  # one χ₁ per χ₂ — the unique upstream Cα-Cβ pair.
    return pairs


# ---------------------------------------------------------------------------
# Real-mol proposer factory — DBT concerted rotation
# ---------------------------------------------------------------------------


def make_mcmm_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    drive_sigma_rad: float = 0.1,
    closure_tol: float = 0.01,
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` for `ReplicaExchangeMCMMDriver` that
    proposes DBT moves on the backbone windows of `mol`, batches MMFF +
    MACE across walkers per call, and returns per-walker
    `(new_coords, new_energy, det_j, success)` tuples.

    Per-call pipeline:

      1. **Per-walker move generation** (CPU, sequential): pick a random
         backbone window, drive dihedral, and `drive_delta ~ N(0,
         drive_sigma_rad²)`. Run `concerted_rotation.propose_move` on
         the 7-atom backbone window positions to solve for the closure
         deltas. Walkers whose closure fails are flagged.
      2. **Full-mol coordinate update** (CPU, sequential): for
         successful walkers, replay the per-dihedral rotations on the
         full atom array via
         `concerted_rotation.apply_dihedral_changes_full_mol`,
         transporting side-chain atoms rigidly with their backbone
         parents according to the precomputed
         `_compute_window_downstream_sets` for the chosen window.
      3. **Batched MMFF** (GPU, one call): stage every successful
         candidate as a conformer on a shared throwaway mol and run
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`
         (`mmff_backend='gpu'`) or RDKit's serial MMFF
         (`mmff_backend='cpu'`) for in-place minimisation.
      4. **Batched MACE** (GPU, chunked): score every minimised
         candidate via `_mace_batch_energies` in chunks of
         `score_chunk_size`.
      5. **Return** `(coords_tensor, energy_float, det_j_float,
         success_bool)` per walker, in walker order. Failed walkers
         pass through with their pre-move coords and `success=False`
         so the driver's `apply_proposal` rejects without further work.

    Topology — backbone windows, side-chain groups, the throwaway-mol
    template — is captured at factory build time and reused per call,
    keeping the per-step CPU overhead bounded by the move-generation
    loop and the conformer-staging step.

    Lazy import of `_mace_batch_energies` from `confsweeper` avoids the
    confsweeper → mcmm → proposers circular dependency at module load time.

    Params:
        mol: Chem.Mol : a head-to-tail cyclic peptide with explicit Hs.
            Topology is captured at factory-build time; the mol must
            not be mutated structurally afterwards (conformer additions
            and edits are fine).
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only consulted when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        drive_sigma_rad: float : Gaussian standard deviation for the
            drive-angle perturbation in radians (default 0.1 ≈ 5.7°).
            Larger values give bigger moves at lower closure-success
            rate; couples to closure_tol per docs/mcmm_plan.md.
        closure_tol: float : passed through to `propose_move` as the
            maximum r5+r6 displacement-norm tolerated as ring-closed.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500, matches `_minimize_score_filter_dedup`).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA, default) or
            'cpu' (RDKit serial). MMFF runs on the throwaway mol.
        seed: int : base seed for the move-RNG; deterministic across
            replicate runs.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]` matching
            the contract expected by `ParallelMCMMDriver` and
            `ReplicaExchangeMCMMDriver`.
    Raises:
        ValueError: if `mol` has no enumerable backbone windows (input
            is not a cyclic peptide of ≥ 3 residues).
    """
    from concerted_rotation import (
        N_DIHEDRALS,
        apply_dihedral_changes_full_mol,
        propose_move,
    )

    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )

    windows = enumerate_backbone_windows(mol)
    if not windows:
        raise ValueError(
            "mol has no enumerable backbone windows; "
            "check that it is a head-to-tail cyclic peptide of at least 3 residues"
        )
    backbone_atoms = _backbone_atom_set(mol)
    window_downstream_sets = [
        _compute_window_downstream_sets(mol, w, backbone_atoms) for w in windows
    ]
    n_windows = len(windows)
    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    # Throwaway-mol template: structure-only, no conformers. Cloning per
    # call is required because nvmolkit MMFF mutates conformers in place
    # and we don't want to corrupt walker state across step() calls.
    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    rng = np.random.default_rng(seed)

    # Per-call cumulative diagnostic counters. Attached to the returned
    # proposer function as `.stats` so callers (`get_mol_PE_mcmm`, tests)
    # can read them after the run to diagnose acceptance regressions —
    # especially the "1 basin" pathology on small peptides where DBT
    # closure fails on most moves.
    stats = {
        "n_proposed": 0,
        "n_closure_failures": 0,
        "n_closure_successes": 0,
    }

    def batch_propose_fn(coords_list):
        # Lazy import: confsweeper imports from mcmm at module load time;
        # importing _mace_batch_energies here defers resolution until the
        # closure is actually called, breaking the circular dependency.
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        stats["n_proposed"] += n_walkers

        # Stage 1: per-walker DBT closure on the backbone window.
        # Successful walkers contribute a (new_full_coords, det_j) entry.
        successful_meta: list = []
        success_walker_indices: list = []
        for w_idx, coords in enumerate(coords_list):
            window_idx = int(rng.integers(n_windows))
            window = windows[window_idx]
            drive_idx = int(rng.integers(N_DIHEDRALS))
            drive_delta = float(rng.normal(0.0, drive_sigma_rad))

            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            window_pos = coords_np[list(window)]
            result = propose_move(
                window_pos, drive_idx, drive_delta, closure_tol=closure_tol
            )
            if not result.success:
                continue
            new_full = apply_dihedral_changes_full_mol(
                coords_np,
                list(window),
                result.deltas,
                window_downstream_sets[window_idx],
            )
            successful_meta.append(
                {"new_full": new_full, "det_j": float(result.det_jacobian)}
            )
            success_walker_indices.append(w_idx)

        stats["n_closure_successes"] += len(successful_meta)
        stats["n_closure_failures"] += n_walkers - len(successful_meta)

        # Stage 2: short-circuit if every walker failed.
        if not successful_meta:
            return [(coords_list[i], 0.0, 0.0, False) for i in range(n_walkers)]

        # Stage 3: stage successful candidates as conformers on a fresh
        # throwaway mol, run batched MMFF.
        throwaway = Chem.Mol(template_mol)
        for meta in successful_meta:
            conf = Chem.Conformer(n_atoms)
            new_full = meta["new_full"]
            for a_idx in range(n_atoms):
                x, y, z = new_full[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)

        if mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in [c.GetId() for c in throwaway.GetConformers()]:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 4: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 5: assemble per-walker proposals in walker order.
        proposals: list = [None] * n_walkers
        for slot, w_idx in enumerate(success_walker_indices):
            cid = post_mmff_conf_ids[slot]
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            proposals[w_idx] = (
                new_coords,
                float(energies[slot]),
                successful_meta[slot]["det_j"],
                True,
            )
        # Failed walkers get a no-op proposal in their original slot.
        for w_idx in range(n_walkers):
            if proposals[w_idx] is None:
                proposals[w_idx] = (coords_list[w_idx], 0.0, 0.0, False)

        return proposals

    # Expose cumulative stats so the orchestrator (get_mol_PE_mcmm) can
    # read closure-failure rates and diagnose "1 basin" regressions.
    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Cartesian-kick proposer — GOAT-style topology-preserving move
# ---------------------------------------------------------------------------


def make_cartesian_kick_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    sigma_kick_a: float = 0.1,
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that applies an isotropic Gaussian kick
    to all atom positions, MMFF-relaxes, and MACE-scores. The
    GOAT-flavour move type: complements DBT's dihedral-space
    parameterisation by reaching geometries that small dihedral
    perturbations can't (side-chain rotamer flips, depsipeptide ester
    rearrangements, etc.).

    Per-call pipeline:

      1. **Per-walker kick** (CPU, vectorised): add `N(0, sigma_kick_a²)`
         noise independently to every atom-coordinate of every walker.
         No closure step or backbone parameterisation; rely on MMFF to
         pull bond lengths and ring sp² angles back to equilibrium.
      2. **Batched MMFF** (GPU, one call): stage every kicked candidate
         as a conformer on a shared throwaway mol and run
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`.
      3. **Batched MACE** (GPU, chunked) via `_mace_batch_energies`.
      4. **Return** `(coords_tensor, energy_float, det_j=1.0,
         success=True)` per walker. The Gaussian kick is symmetric in
         coordinate space so detailed balance needs no Jacobian
         correction; the post-MMFF state is treated as the proposal
         outcome regardless of the relaxation trajectory (the same
         convention `make_mcmm_proposer` uses).

    Topology preservation is approximate, not strict: GOAT freezes
    bonds and ring sp² angles via constraints during the uphill push,
    while we let MMFF relax them. For peptide cyclisations and N-Me
    bonds this is fine — MMFF's bond-stretch term has a steep gradient
    that pulls covalent bonds back to equilibrium even from large
    kicks. For sigma_kick_a beyond ~0.3 Å, expect occasional MMFF
    non-convergence or bond-breaking; tune accordingly.

    Params:
        mol: Chem.Mol : reference molecule with explicit Hs. Topology
            captured at factory build time; do not mutate structurally.
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only used when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        sigma_kick_a: float : Gaussian standard deviation in Å applied
            independently to every atom-coordinate (default 0.1 Å).
            Treat as the move-magnitude analogue of DBT's
            `drive_sigma_rad`.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA) or 'cpu'
            (RDKit serial).
        seed: int : RNG seed for the per-step kicks.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`
            matching the `ParallelMCMMDriver` /
            `ReplicaExchangeMCMMDriver` proposer contract.
    Raises:
        ValueError: on unknown `mmff_backend`.
    """
    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )

    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    rng = np.random.default_rng(seed)
    stats = {"n_proposed": 0, "n_relax_failures": 0, "n_relax_successes": 0}

    def batch_propose_fn(coords_list):
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        if n_walkers == 0:
            return []
        stats["n_proposed"] += n_walkers

        # Stage 1: per-walker isotropic Gaussian kick.
        kicked_coords: list = []
        for coords in coords_list:
            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            kick = rng.normal(0.0, sigma_kick_a, size=coords_np.shape)
            kicked_coords.append(coords_np + kick)

        # Stage 2: batched MMFF on a fresh throwaway mol.
        throwaway = Chem.Mol(template_mol)
        for kc in kicked_coords:
            conf = Chem.Conformer(n_atoms)
            for a_idx in range(n_atoms):
                x, y, z = kc[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)

        if mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in [c.GetId() for c in throwaway.GetConformers()]:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 3: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 4: assemble per-walker proposals. det_j=1.0 for the
        # symmetric Gaussian kick. Walkers whose post-MMFF energy is
        # non-finite (MMFF blow-up — rare but possible at large
        # sigma_kick_a) get success=False so the driver rejects.
        proposals: list = []
        for slot, cid in enumerate(post_mmff_conf_ids):
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            e = float(energies[slot])
            if not np.isfinite(e):
                stats["n_relax_failures"] += 1
                proposals.append((coords_list[slot], 0.0, 0.0, False))
            else:
                stats["n_relax_successes"] += 1
                proposals.append((new_coords, e, 1.0, True))
        return proposals

    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Side-chain dihedral-kick proposer — hybrid Gaussian + rotamer jump
# ---------------------------------------------------------------------------


def make_dihedral_kick_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    sigma_chi_rad: float = 0.5,
    p_rotamer_jump: float = 0.3,
    rotamer_wells_deg: tuple = (-60.0, 60.0, 180.0),
    aromatic_wells_deg: tuple | None = None,
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    skip_mmff_relax: bool = False,
    dihedral_weight_by_atom_count: bool = False,
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that perturbs one side-chain dihedral per
    walker per step (hybrid Gaussian + discrete rotamer jump),
    MMFF-relaxes, and MACE-scores. Complements DBT (backbone-only) and
    the Cartesian kick (atom-position perturbation) by reaching
    side-chain rotamer states neither of the other proposers can flip.

    Motivated by the 2026-05-21 Boltzmann-coverage Finding in
    `docs/mcmm_plan.md`: cremp_sharp's MCMM basin set sits geometrically
    3+ Å from every CREMP ceiling basin and a single CREMP basin holds
    72 % of the 298 K Boltzmann population. The failure points at
    NMe-Trp χ₁ / χ₂ rotamers DBT (backbone-only) cannot reach and
    σ ≈ 0.1 Å Cartesian kicks cannot push through (indole χ barriers
    are ~10–15 kcal/mol — MMFF snaps small Δχ back to the starting
    well). See `docs/dihedral_kick_plan.md` for the full design and
    locked design choices (Step 1).

    Per-call pipeline:

      1. **Per-walker rotation** (CPU, sequential): pick one side-chain
         dihedral uniformly at random from
         `_enumerate_side_chain_dihedrals(mol)` (captured at factory
         build time); sample Δχ via the hybrid rule (Gaussian
         `N(0, σ_chi_rad)` by default; with probability
         `p_rotamer_jump`, set χ to a discrete rotameric well from
         `rotamer_wells_deg`). Apply via
         `Chem.rdMolTransforms.SetDihedralDeg`, which rotates only the
         atomic subtree downstream of the bond — backbone, ring, and
         other side chains stay put.
      2. **Batched MMFF** (GPU, one call): every rotated candidate
         lives as a conformer on a shared throwaway mol; run
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` for
         in-place minimisation. When `skip_mmff_relax=True` (issue
         #15 / v0.2 no-MMFF ablation) this stage is bypassed entirely
         and Stage 3 (MACE) scores the raw rotated geometry directly.
      3. **Batched MACE** (GPU, chunked) via `_mace_batch_energies`.
      4. **Return** `(coords_tensor, energy_float, det_j=1.0,
         success=True)` per walker. A single open-tree dihedral
         rotation is volume-preserving in dihedral space, so detailed
         balance needs no Jacobian correction (unlike DBT's closed-loop
         concerted rotation with its Wu-Deem term). Walkers whose
         post-MMFF energy is non-finite get `success=False` so the
         driver rejects.

    **Known risk — MMFF rotamer snap-back / artificial collapse.**
    Stage 2's batched MMFF relax operates on the MMFF94 potential
    energy surface, NOT MACE. For small Δχ on the Gaussian path this is
    benign (MMFF faithfully relaxes within the starting rotameric
    well). For rotamer jumps that land near a well boundary, MMFF can
    sometimes pull the geometry back across the barrier into the
    starting well — collapsing the intended diversity before MACE ever
    sees the proposal. If Step-7 validation shows cremp_sharp coverage
    stays poor despite the new proposer landing, the diagnostic is to
    disable Stage 2 (skip MMFF, score MACE on the rotated geometry
    directly) and re-check whether the union basin set diversifies.
    Tracked under "Risks to instrument" and "Deferred follow-ups" in
    `docs/dihedral_kick_plan.md`.

    Lazy import of `_mace_batch_energies` from `confsweeper` avoids the
    confsweeper → mcmm → proposers circular dependency at module load
    time (same convention as `make_mcmm_proposer` and
    `make_cartesian_kick_proposer`).

    Params:
        mol: Chem.Mol : reference cyclic peptide with explicit Hs.
            Topology + side-chain dihedral enumeration are captured at
            factory build time; do not mutate the mol structurally
            afterwards.
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only used when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        sigma_chi_rad: float : Gaussian standard deviation in radians
            for the refinement-mode step (default 0.5 ≈ 28.6°). Tuned
            to leave the starting micro-well within a few steps but
            stay inside a rotameric well so MMFF doesn't immediately
            snap back. Locked by `docs/dihedral_kick_plan.md` Findings
            2026-05-22.
        p_rotamer_jump: float : probability (per walker per step) of a
            discrete rotamer jump instead of a Gaussian refinement step
            (default 0.3). Set to 0 for pure Gaussian (ablation); set
            to 1 for pure rotameric jumps. The jump path provides
            barrier-crossing — without it MMFF tends to undo small Δχ
            and the proposer can't escape its starting rotamer.
        rotamer_wells_deg: tuple[float, ...] : sp3 rotameric well
            centres in degrees (default `(-60.0, 60.0, 180.0)` —
            standard sp3 χ₁ wells). Used for any side-chain rotatable
            bond whose downstream endpoint `c` is non-aromatic, AND for
            all bonds when `aromatic_wells_deg is None` (the v0.1
            issue-#12 behaviour).
        aromatic_wells_deg: tuple[float, ...] | None : rotameric well
            centres in degrees applied to side-chain rotatable bonds
            whose downstream endpoint `c` is aromatic (e.g. NMe-Trp χ₂
            — indole-edge / face-on states near ±90°). Default `None`
            preserves v0.1 behaviour (every bond uses
            `rotamer_wells_deg`). The v0.2 locked aromatic well set
            (issue #15) is `(-90.0, 0.0, 90.0, 180.0)` — see
            `docs/dihedral_kick_v0_2_plan.md` Findings 2026-06-15 for
            the rationale.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA, default) or
            'cpu' (RDKit serial).
        skip_mmff_relax: bool : v0.2 ablation toggle (issue #15). When
            True, the Stage-2 MMFF94 batched relax is bypassed entirely
            — the rotated coordinates pass directly to the Stage-3
            MACE batched scorer. Diagnostic-grade ablation for the
            MMFF-snap-back hypothesis flagged in this docstring's
            "Known risk" section: MMFF can drag rotamer jumps back
            across their barriers before MACE sees them, collapsing
            the dihedral-kick's intended diversity onto the MMFF94 PES
            instead of MACE's. Default False preserves v0.1 / Step-2
            behaviour. Note: skipping MMFF may leave neighbouring
            atoms in mildly strained Cartesian positions on rotamer
            jumps; if MACE-acceptance with `skip_mmff_relax=True`
            drops below ~5 % the ablation is uninterpretable and a
            v0.3 partial-MMFF candidate becomes next-up. See
            `docs/dihedral_kick_v0_2_plan.md` Findings 2026-06-15 for
            the rationale.
        dihedral_weight_by_atom_count: bool : v0 stub — when False
            (default, locked Step 1 choice), pick a side-chain dihedral
            uniformly at random. The True branch (bias toward bulky
            side chains by their heavy-atom count) is plumbed for a
            future follow-up; raises `NotImplementedError` if set True
            so we don't ship a silent stub.
        seed: int : RNG seed for the per-step dihedral pick + Δχ sample.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`
            matching the `ParallelMCMMDriver` /
            `ReplicaExchangeMCMMDriver` proposer contract.
    Raises:
        ValueError: on unknown `mmff_backend`, `p_rotamer_jump` not in
            [0, 1], non-positive `sigma_chi_rad`, empty
            `rotamer_wells_deg` when `p_rotamer_jump > 0`, empty
            `aromatic_wells_deg` (when not None) when
            `p_rotamer_jump > 0`, or when `mol` has no enumerable
            side-chain rotatable dihedrals (e.g., cyclic homo-alanine
            — every side chain is a methyl, every rotatable-bond match
            is filtered).
        NotImplementedError: if `dihedral_weight_by_atom_count=True`
            (v0 only ships uniform selection).
    """
    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )
    if not 0.0 <= p_rotamer_jump <= 1.0:
        raise ValueError(f"p_rotamer_jump must be in [0, 1], got {p_rotamer_jump}")
    if sigma_chi_rad <= 0.0:
        raise ValueError(f"sigma_chi_rad must be positive, got {sigma_chi_rad}")
    if p_rotamer_jump > 0.0 and len(rotamer_wells_deg) == 0:
        raise ValueError("rotamer_wells_deg must be non-empty when p_rotamer_jump > 0")
    if (
        p_rotamer_jump > 0.0
        and aromatic_wells_deg is not None
        and len(aromatic_wells_deg) == 0
    ):
        raise ValueError(
            "aromatic_wells_deg must be non-empty when p_rotamer_jump > 0 "
            "and aromatic_wells_deg is provided (pass None to disable per-bond classification)"
        )
    if dihedral_weight_by_atom_count:
        raise NotImplementedError(
            "dihedral_weight_by_atom_count=True is a v0 deferred branch; "
            "see docs/dihedral_kick_plan.md Deferred follow-ups"
        )

    dihedrals = _enumerate_side_chain_dihedrals(mol)
    if not dihedrals:
        raise ValueError(
            "mol has no enumerable side-chain rotatable dihedrals; "
            "check that the input is a cyclic peptide with non-methyl side chains"
        )
    n_dihedrals = len(dihedrals)
    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    # Throwaway-mol template: structure-only, no conformers. Cloning per
    # call is required because nvmolkit MMFF mutates conformers in place
    # and we don't want to corrupt walker state across step() calls.
    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    rotamer_wells_per_bond = _classify_rotamer_wells(
        mol, dihedrals, aromatic_wells_deg, rotamer_wells_deg
    )
    sigma_chi_deg = math.degrees(sigma_chi_rad)

    rng = np.random.default_rng(seed)
    stats = {
        "n_proposed": 0,
        "n_gaussian_steps": 0,
        "n_rotamer_jumps": 0,
        "n_relax_failures": 0,
        "n_relax_successes": 0,
        "n_mmff_skipped": 0,
    }

    def batch_propose_fn(coords_list):
        # Lazy import: confsweeper imports from mcmm at module load time;
        # importing _mace_batch_energies here defers resolution until the
        # closure is actually called, breaking the circular dependency.
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        if n_walkers == 0:
            return []
        stats["n_proposed"] += n_walkers

        # Stage 1: stage walker coords on a fresh throwaway mol, then apply
        # the per-walker rotation in place. SetDihedralDeg rotates only the
        # subtree downstream of the bond, so backbone + other side chains
        # are preserved by construction.
        throwaway = Chem.Mol(template_mol)
        for coords in coords_list:
            conf = Chem.Conformer(n_atoms)
            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            for a_idx in range(n_atoms):
                x, y, z = coords_np[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)
        pre_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        for cid in pre_mmff_conf_ids:
            conf = throwaway.GetConformer(cid)
            dihedral_idx = int(rng.integers(n_dihedrals))
            a, b, c, d = dihedrals[dihedral_idx]
            if rng.uniform() < p_rotamer_jump:
                stats["n_rotamer_jumps"] += 1
                wells = rotamer_wells_per_bond[dihedral_idx]
                new_chi_deg = float(wells[rng.integers(len(wells))])
            else:
                stats["n_gaussian_steps"] += 1
                current = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
                delta_deg = float(rng.normal(0.0, sigma_chi_deg))
                new_chi_deg = current + delta_deg
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, new_chi_deg)

        # Stage 2: batched MMFF on the throwaway mol. When
        # `skip_mmff_relax=True` (issue #15 / v0.2 ablation), bypass the
        # MMFF94 step entirely and let Stage 3 (MACE) score the raw
        # rotated geometry directly. This is the no-MMFF ablation called
        # for by the issue-#12 Step-7 phase 2 Findings: MMFF can drag
        # rotamer jumps back across their barriers before MACE sees
        # them, collapsing the dihedral-kick's intended diversity onto
        # the MMFF94 PES instead of MACE's. Skipping MMFF is
        # diagnostic-grade — raw-rotation geometries may carry minor
        # bond-stretch strain MMFF would have fixed; if MACE-acceptance
        # drops below ~5 % with the flag on, the ablation is
        # uninterpretable for the wrong reason and the v0.3
        # partial-MMFF candidate becomes next-up.
        if skip_mmff_relax:
            stats["n_mmff_skipped"] += len(pre_mmff_conf_ids)
        elif mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in pre_mmff_conf_ids:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 3: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 4: assemble per-walker proposals. det_j=1.0 (open-tree
        # rotation is volume-preserving). Non-finite energies → reject.
        proposals: list = []
        for slot, cid in enumerate(post_mmff_conf_ids):
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            e = float(energies[slot])
            if not np.isfinite(e):
                stats["n_relax_failures"] += 1
                proposals.append((coords_list[slot], 0.0, 0.0, False))
            else:
                stats["n_relax_successes"] += 1
                proposals.append((new_coords, e, 1.0, True))
        return proposals

    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Concerted dihedral kick — v0.3 Move A (joint χ₁ + χ₂ on aromatic side chains)
# ---------------------------------------------------------------------------


def make_concerted_dihedral_kick_proposer(
    mol: Chem.Mol,
    hardware_opts,
    calc,
    sigma_concerted_chi_rad: float = 0.5,
    p_concerted_jump: float = 0.3,
    sp3_wells_deg: tuple = (-60.0, 60.0, 180.0),
    aromatic_wells_deg: tuple = (-90.0, 0.0, 90.0, 180.0),
    score_chunk_size: int = 500,
    mmff_backend: str = "gpu",
    skip_mmff_relax: bool = False,
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that rotates a `(χ₁, χ₂)` side-chain
    dihedral pair *together* per walker per step (joint hybrid Gaussian
    + joint rotamer-jump), MMFF-relaxes, and MACE-scores. v0.3 Move A
    (issue #17). Attacks the cremp_sharp residual flagged in the v0.2
    Step-7 Findings: single-bond moves cannot reach the dominant
    ceiling basin because rotating χ₁ alone leaves χ₂ at the wrong
    orientation (MMFF/MACE rejects), and rotating χ₂ alone leaves the
    side-chain anchor (Cβ position) at the wrong location. Both bonds
    must rotate together to reach the joint state. See
    `docs/concerted_moves_v0_3_plan.md` Move A design lock (2026-06-19)
    for the locked design choices and rationale.

    Per-call pipeline (same shape as `make_dihedral_kick_proposer`):

      1. **Per-walker joint rotation** (CPU, sequential): pick a
         `(χ₁, χ₂)` pair uniformly at random from
         `_enumerate_concerted_dihedral_pairs(mol)`; sample either
         (a) joint Gaussian Δχ₁, Δχ₂ each from `N(0, σ_concerted_chi_rad)`
         with probability `1 - p_concerted_jump`, or (b) joint rotamer
         jump — `χ₁_target` sampled uniformly from `sp3_wells_deg`,
         `χ₂_target` sampled uniformly from `aromatic_wells_deg` —
         with probability `p_concerted_jump`. Apply via two successive
         `Chem.rdMolTransforms.SetDihedralDeg` calls (χ₁ first because
         the χ₂ rotation axis is downstream of χ₁ — rotating χ₁ moves
         χ₂'s `b` and `c` atoms; the χ₂ rotation then operates on the
         post-χ₁ geometry).
      2. **Batched MMFF** (GPU, one call): every rotated candidate
         lives as a conformer on a shared throwaway mol;
         `nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`
         minimises in place. When `skip_mmff_relax=True` (the v0.2
         ablation toggle, plumbed through to this factory) Stage 2 is
         bypassed entirely; Stage 3 scores the raw rotated geometry.
      3. **Batched MACE** (GPU, chunked) via `_mace_batch_energies`.
      4. **Return** `(coords_tensor, energy_float, det_j=1.0,
         success=True)` per walker. A joint open-tree rotation in 2D
         dihedral space is volume-preserving (the Jacobian of two
         successive open-tree rotations is the product of two unit
         Jacobians), so detailed balance needs no correction. Walkers
         whose post-MMFF / post-MACE energy is non-finite get
         `success=False` exactly as the single-bond path does.

    Strict separation from DBT and from v0.2's single-bond
    `make_dihedral_kick_proposer`: the eligibility helper
    `_enumerate_concerted_dihedral_pairs` returns ONLY pairs whose χ₂'s
    `c` atom is aromatic, so this factory will never touch a backbone
    dihedral and never touch a sp3-only side chain. Calling this
    factory on a peptide with no aromatic side chains (e.g.,
    cremp_typical) raises `ValueError` at factory build time so the
    caller learns immediately instead of silently producing a no-op
    proposer at runtime.

    Composition with the rest of the proposer family: this factory is
    a new top-level proposer routed by `make_composite_proposer`
    alongside DBT, Cartesian, and v0.2's single-bond dihedral kick. The
    Step-3 v0.3 work extends `make_default_mcmm_composite` from 3
    sub-proposers to 4 to include this one — the
    `weight > 0 ↔ proposer not None` contract generalises naturally
    (see locked Step-1 design choice A.3 in
    `docs/concerted_moves_v0_3_plan.md`).

    Lazy import of `_mace_batch_energies` from `confsweeper` avoids the
    confsweeper → mcmm → proposers circular dependency at module load
    time (same convention as the other proposer factories in this
    module).

    Params:
        mol: Chem.Mol : reference cyclic peptide with explicit Hs.
            Topology + `(χ₁, χ₂)` pair enumeration are captured at
            factory build time; do not mutate the mol structurally
            afterwards.
        hardware_opts : nvmolkit hardware options for batched MMFF
            (only used when `mmff_backend='gpu'`).
        calc : MACECalculator from `get_mace_calc()`.
        sigma_concerted_chi_rad: float : Gaussian standard deviation in
            radians for each component of the joint Gaussian refinement
            step (default 0.5 ≈ 28.6° — same magnitude as v0.2's
            single-bond `sigma_chi_rad`).
        p_concerted_jump: float : probability (per walker per step) of
            a joint rotamer jump instead of a joint Gaussian
            refinement step (default 0.3). Set to 0 for pure Gaussian;
            set to 1 for pure joint rotamer jumps. Same locked default
            as v0.2's single-bond `p_rotamer_jump`.
        sp3_wells_deg: tuple[float, ...] : χ₁-component rotameric well
            centres in degrees (default `(-60, 60, 180)` — standard
            sp3 χ₁). Used for the χ₁ side of every joint rotamer jump.
        aromatic_wells_deg: tuple[float, ...] : χ₂-component rotameric
            well centres in degrees (default `(-90, 0, 90, 180)` —
            v0.2's locked aromatic well set). Used for the χ₂ side of
            every joint rotamer jump.
        score_chunk_size: int : MACE per-batch forward pass cap
            (default 500).
        mmff_backend: str : 'gpu' (nvmolkit batched CUDA, default) or
            'cpu' (RDKit serial).
        skip_mmff_relax: bool : v0.2 ablation toggle (issue #15). When
            True, Stage-2 MMFF94 is bypassed and Stage-3 MACE scores
            the raw rotated geometry. Default False preserves the
            v0.2-with-MMFF behaviour.
        seed: int : RNG seed for the per-step pair pick + joint
            Δχ-or-jump sample.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`
            matching the `ParallelMCMMDriver` /
            `ReplicaExchangeMCMMDriver` proposer contract.
    Raises:
        ValueError: on unknown `mmff_backend`, `p_concerted_jump` not
            in [0, 1], non-positive `sigma_concerted_chi_rad`, empty
            `sp3_wells_deg` or `aromatic_wells_deg` when
            `p_concerted_jump > 0`, or when `mol` has no enumerable
            aromatic side-chain `(χ₁, χ₂)` pairs (e.g., cyclo-Ala,
            cremp_typical — call the v0.2 single-bond factory instead
            on those).
    """
    if mmff_backend not in ("gpu", "cpu"):
        raise ValueError(
            f"unknown mmff_backend {mmff_backend!r}; expected 'gpu' or 'cpu'"
        )
    if not 0.0 <= p_concerted_jump <= 1.0:
        raise ValueError(f"p_concerted_jump must be in [0, 1], got {p_concerted_jump}")
    if sigma_concerted_chi_rad <= 0.0:
        raise ValueError(
            f"sigma_concerted_chi_rad must be positive, got {sigma_concerted_chi_rad}"
        )
    if p_concerted_jump > 0.0 and len(sp3_wells_deg) == 0:
        raise ValueError("sp3_wells_deg must be non-empty when p_concerted_jump > 0")
    if p_concerted_jump > 0.0 and len(aromatic_wells_deg) == 0:
        raise ValueError(
            "aromatic_wells_deg must be non-empty when p_concerted_jump > 0"
        )

    pairs = _enumerate_concerted_dihedral_pairs(mol)
    if not pairs:
        raise ValueError(
            "mol has no enumerable aromatic side-chain (χ₁, χ₂) pairs; "
            "check that the input is a cyclic peptide with at least one "
            "aromatic side chain (Trp / Phe / Tyr / His). For pure-sp3 "
            "peptides use `make_dihedral_kick_proposer` (v0.2 single-bond) "
            "instead."
        )
    n_pairs = len(pairs)
    n_atoms = mol.GetNumAtoms()
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    template_mol = Chem.Mol(mol)
    template_mol.RemoveAllConformers()

    sp3_wells_arr = np.asarray(sp3_wells_deg, dtype=np.float64)
    aromatic_wells_arr = np.asarray(aromatic_wells_deg, dtype=np.float64)
    n_sp3_wells = len(sp3_wells_arr)
    n_aromatic_wells = len(aromatic_wells_arr)
    sigma_concerted_chi_deg = math.degrees(sigma_concerted_chi_rad)

    rng = np.random.default_rng(seed)
    stats = {
        "n_proposed": 0,
        "n_concerted_gaussian_steps": 0,
        "n_concerted_rotamer_jumps": 0,
        "n_relax_failures": 0,
        "n_relax_successes": 0,
        "n_mmff_skipped": 0,
    }

    def batch_propose_fn(coords_list):
        from confsweeper import _mace_batch_energies

        n_walkers = len(coords_list)
        if n_walkers == 0:
            return []
        stats["n_proposed"] += n_walkers

        # Stage 1: stage walker coords on a fresh throwaway mol, then
        # apply the per-walker joint rotation in place.
        throwaway = Chem.Mol(template_mol)
        for coords in coords_list:
            conf = Chem.Conformer(n_atoms)
            coords_np = coords.detach().cpu().numpy().astype(np.float64)
            for a_idx in range(n_atoms):
                x, y, z = coords_np[a_idx]
                conf.SetAtomPosition(a_idx, (float(x), float(y), float(z)))
            throwaway.AddConformer(conf, assignId=True)
        pre_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        for cid in pre_mmff_conf_ids:
            conf = throwaway.GetConformer(cid)
            pair_idx = int(rng.integers(n_pairs))
            chi1, chi2 = pairs[pair_idx]
            a1, b1, c1, d1 = chi1
            a2, b2, c2, d2 = chi2
            if rng.uniform() < p_concerted_jump:
                stats["n_concerted_rotamer_jumps"] += 1
                new_chi1_deg = float(sp3_wells_arr[rng.integers(n_sp3_wells)])
                new_chi2_deg = float(aromatic_wells_arr[rng.integers(n_aromatic_wells)])
                rdMolTransforms.SetDihedralDeg(conf, a1, b1, c1, d1, new_chi1_deg)
                rdMolTransforms.SetDihedralDeg(conf, a2, b2, c2, d2, new_chi2_deg)
            else:
                stats["n_concerted_gaussian_steps"] += 1
                # Apply χ₁ first; its rotation moves χ₂'s b and c atoms.
                # Read χ₂'s current angle AFTER applying χ₁ so the Δχ₂ is
                # relative to the post-χ₁ geometry (the natural way joint
                # dihedral perturbations compose).
                current_chi1 = rdMolTransforms.GetDihedralDeg(conf, a1, b1, c1, d1)
                new_chi1_deg = current_chi1 + float(
                    rng.normal(0.0, sigma_concerted_chi_deg)
                )
                rdMolTransforms.SetDihedralDeg(conf, a1, b1, c1, d1, new_chi1_deg)
                current_chi2 = rdMolTransforms.GetDihedralDeg(conf, a2, b2, c2, d2)
                new_chi2_deg = current_chi2 + float(
                    rng.normal(0.0, sigma_concerted_chi_deg)
                )
                rdMolTransforms.SetDihedralDeg(conf, a2, b2, c2, d2, new_chi2_deg)

        # Stage 2: batched MMFF on the throwaway mol; skip when
        # skip_mmff_relax=True (v0.2 ablation path, plumbed through).
        if skip_mmff_relax:
            stats["n_mmff_skipped"] += len(pre_mmff_conf_ids)
        elif mmff_backend == "gpu":
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            MMFFOptimizeMoleculesConfs([throwaway], hardwareOptions=hardware_opts)
        else:
            from rdkit.Chem import AllChem as _AllChem

            for cid in pre_mmff_conf_ids:
                _AllChem.MMFFOptimizeMolecule(throwaway, confId=cid)

        # Stage 3: batched MACE scoring, chunked.
        post_mmff_conf_ids = [c.GetId() for c in throwaway.GetConformers()]
        energies: list = []
        for start in range(0, len(post_mmff_conf_ids), score_chunk_size):
            chunk_ids = post_mmff_conf_ids[start : start + score_chunk_size]
            ase_mols = [
                ase.Atoms(
                    positions=throwaway.GetConformer(cid).GetPositions(),
                    numbers=atomic_nums,
                )
                for cid in chunk_ids
            ]
            energies.extend(_mace_batch_energies(calc, ase_mols))

        # Stage 4: assemble per-walker proposals.
        proposals: list = []
        for slot, cid in enumerate(post_mmff_conf_ids):
            new_coords = torch.tensor(
                throwaway.GetConformer(cid).GetPositions(), dtype=torch.float64
            )
            e = float(energies[slot])
            if not np.isfinite(e):
                stats["n_relax_failures"] += 1
                proposals.append((coords_list[slot], 0.0, 0.0, False))
            else:
                stats["n_relax_successes"] += 1
                proposals.append((new_coords, e, 1.0, True))
        return proposals

    batch_propose_fn.stats = stats
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Composite proposer — randomly route walkers to one of several sub-proposers
# ---------------------------------------------------------------------------


def make_composite_proposer(
    proposers,
    weights=None,
    seed: int = 0,
):
    """
    Build a `batch_propose_fn` that routes each walker to one of N
    sub-proposers per step, sampled by `weights`. Lets DBT and
    Cartesian-kick (or future move types) coexist in one MCMM run.

    Routing happens at the walker level, not the step level — so a
    single REMD step can have some walkers proposing DBT moves and
    others proposing Cartesian kicks. Walkers are partitioned by
    chosen proposer, each sub-proposer is invoked on its subset
    (preserving its internal batching), and results are reassembled
    in walker order.

    Each sub-proposer's `.stats` dict is preserved on the composite
    return value as `.stats[i]` so callers can inspect per-proposer
    diagnostics. The composite itself does not aggregate counters.

    Params:
        proposers: list[callable] : sub-proposers, each matching the
            `batch_propose_fn(coords_list) -> list[tuple]` contract.
        weights: list[float] | None : sampling weight per proposer.
            Default None means uniform. Normalised internally.
        seed: int : routing RNG seed; deterministic across replicate
            runs.
    Returns:
        callable : `batch_propose_fn(coords_list) -> list[tuple]`.
            Carries `.stats = [p.stats for p in proposers]` (list,
            indexed by proposer position).
    Raises:
        ValueError: empty `proposers`, weight/proposer count mismatch,
            or any non-positive weight.
    """
    if not proposers:
        raise ValueError("proposers must be non-empty")
    if weights is None:
        weights_arr = np.full(len(proposers), 1.0 / len(proposers))
    else:
        if len(weights) != len(proposers):
            raise ValueError(
                f"weights ({len(weights)}) must match proposers ({len(proposers)})"
            )
        if any(w <= 0 for w in weights):
            raise ValueError(f"weights must be positive, got {weights}")
        w_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = w_arr / w_arr.sum()

    rng = np.random.default_rng(seed)

    def batch_propose_fn(coords_list):
        n = len(coords_list)
        if n == 0:
            return []
        choices = rng.choice(len(proposers), size=n, p=weights_arr)

        results: list = [None] * n
        for p_idx, propose_fn in enumerate(proposers):
            walker_idx = [w for w in range(n) if choices[w] == p_idx]
            if not walker_idx:
                continue
            sub_coords = [coords_list[w] for w in walker_idx]
            sub_results = propose_fn(sub_coords)
            if len(sub_results) != len(sub_coords):
                raise RuntimeError(
                    f"proposer {p_idx} returned {len(sub_results)} results for "
                    f"{len(sub_coords)} walkers"
                )
            for w, r in zip(walker_idx, sub_results):
                results[w] = r
        return results

    batch_propose_fn.stats = [p.stats for p in proposers if hasattr(p, "stats")]
    return batch_propose_fn


# ---------------------------------------------------------------------------
# Default 3-way MCMM composite — DBT + optional Cartesian + optional dihedral
# ---------------------------------------------------------------------------


def make_default_mcmm_composite(
    dbt_proposer,
    cart_proposer=None,
    dihedral_proposer=None,
    *,
    cartesian_weight: float = 0.0,
    dihedral_weight: float = 0.0,
    seed: int = 0,
):
    """
    Assemble the canonical (DBT, Cartesian-kick, dihedral-kick) composite
    used by `get_mol_PE_mcmm`. DBT residual weight is
    `1 - cartesian_weight - dihedral_weight`; the helper validates the
    sum and the (weight > 0 ↔ proposer not None) contract, then
    short-circuits to the lone active proposer when only one weight is
    positive (zero composite-routing overhead — matches the existing
    "if cartesian_weight == 0: batch_propose_fn = dbt_proposer" path
    the issue-#10 routing relied on).

    The caller is responsible for NOT building a sub-proposer whose
    weight is zero; the helper enforces this so a `cart_proposer` is
    never silently constructed and then ignored, which would waste the
    MMFF + MACE setup cost.

    Params:
        dbt_proposer: callable : the DBT (`make_mcmm_proposer`) batch
            propose function — always present (no API to disable DBT).
        cart_proposer: callable | None : Cartesian-kick
            (`make_cartesian_kick_proposer`) batch propose function.
            Must be None iff `cartesian_weight == 0.0`.
        dihedral_proposer: callable | None : dihedral-kick
            (`make_dihedral_kick_proposer`) batch propose function.
            Must be None iff `dihedral_weight == 0.0`.
        cartesian_weight: float : routing weight for the Cartesian kick,
            in [0, 1]. Default 0.0.
        dihedral_weight: float : routing weight for the dihedral kick,
            in [0, 1]. Default 0.0.
        seed: int : routing RNG seed; threaded into `make_composite_proposer`
            when a composite is needed.
    Returns:
        callable : a `batch_propose_fn(coords_list) -> list[tuple]`. When
            only one weight is positive, the corresponding sub-proposer
            is returned directly (its own `.stats` shape is preserved —
            a dict, not a list — and the existing dict/list aggregation
            in `get_mol_PE_mcmm` handles both shapes).
    Raises:
        ValueError: any weight is negative; `cartesian_weight + dihedral_weight > 1`;
            or the (weight > 0 ↔ proposer not None) contract is violated
            for either sub-proposer.
    """
    if cartesian_weight < 0.0 or dihedral_weight < 0.0:
        raise ValueError(
            f"weights must be non-negative, got cartesian_weight={cartesian_weight}, "
            f"dihedral_weight={dihedral_weight}"
        )
    total_non_dbt = cartesian_weight + dihedral_weight
    if total_non_dbt > 1.0 + 1e-12:
        raise ValueError(
            f"cartesian_weight + dihedral_weight = {total_non_dbt} > 1.0; "
            "DBT residual weight would be negative"
        )
    if (cartesian_weight > 0.0) != (cart_proposer is not None):
        raise ValueError(
            f"cartesian_weight={cartesian_weight} but cart_proposer is "
            f"{'None' if cart_proposer is None else 'not None'}; "
            "weight > 0 iff proposer not None"
        )
    if (dihedral_weight > 0.0) != (dihedral_proposer is not None):
        raise ValueError(
            f"dihedral_weight={dihedral_weight} but dihedral_proposer is "
            f"{'None' if dihedral_proposer is None else 'not None'}; "
            "weight > 0 iff proposer not None"
        )

    dbt_weight = 1.0 - total_non_dbt

    active: list = []
    if dbt_weight > 0.0:
        active.append((dbt_weight, dbt_proposer))
    if cart_proposer is not None:
        active.append((cartesian_weight, cart_proposer))
    if dihedral_proposer is not None:
        active.append((dihedral_weight, dihedral_proposer))

    if len(active) == 1:
        return active[0][1]

    return make_composite_proposer(
        [p for _, p in active],
        weights=[w for w, _ in active],
        seed=seed,
    )
