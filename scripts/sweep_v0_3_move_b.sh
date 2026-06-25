#!/usr/bin/env bash
# v0.3 Step 8 — Validation B: ω-flip (Move B) layered on the v0.2 production
# mix, cremp_typical + cremp_sharp at n_seeds=10000.
#
# Quantifies the impact of the new cis/trans ω-isomerization move (Move B)
# against the v0.2 methodology. Step 4 ruled out Move A (concerted χ₁+χ₂) on
# cremp_sharp (cov_bw_ceil stayed 0.000), so the comparison baseline here is
# the v0.2 production stack WITHOUT any v0.3 move, and we layer ω-flip on top.
#
# Three cells, all on v0.2's locked production stack underneath
# (cart=0.33, dih=0.33, p_rotamer_jump=0.30, --aromatic_wells, --skip_mmff_relax):
#   - B1_omega_0_00 : omega_flip_weight=0.00, DBT residual=0.34
#                     → exact reproduction of the v0.2 Step-7 B.4 production
#                       cell; regression check + the comparison baseline.
#   - B2_omega_0_17 : omega_flip_weight=0.17, DBT residual=0.17
#                     → half-strength ω-flip layered on the v0.2 mix.
#   - B3_omega_0_34 : omega_flip_weight=0.34, DBT residual=0.00
#                     → full-strength ω-flip (matches cart/dih shares);
#                       tests whether the cis-ω move closes cremp_sharp.
#
# Both peptides run in every cell so we can read cremp_sharp improvement AND
# cremp_typical non-regression (the Step-4 sweep only swept cremp_sharp in the
# A2/A3 cells, leaving the typical-regression question open — fixed here).
#
# dedup_mode = kabsch ONLY. The go/no-go decision needs only the kabsch
# cov_bw_ceil; crest dedup (CREMP-comparable basin count for the paper) is a
# separate full walk per run (it changes in-run BasinMemory dynamics) and is
# deferred to a follow-up pass on the winning mix if Move B clears 0.10. This
# halves the sweep (~1.5 hr vs ~3 hr for --dedup_mode both).
#
# Decision point at end of sweep (locked Step 8 of v0.3 plan, mirrors Step 5):
#   - If B.2 or B.3 lifts cremp_sharp's cov_bw_ceil above 0.10 → v0.3 Move B
#     is the production fix; skip to Step 13 / Step 14.
#   - If both leave cremp_sharp at zero → proceed to Move C (multi-window DBT).

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_3_move_b_logs
mkdir -p "$LOGDIR"

# (cell_name, omega_flip_weight)
CELLS=(
  "B1_omega_0_00:0.00"
  "B2_omega_0_17:0.17"
  "B3_omega_0_34:0.34"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL OMEGA_W <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_3_move_b_${CELL}"
  SAMP_CSV="results/sweep_v0_3_move_b_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_3_move_b_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL  omega_flip_weight=$OMEGA_W  n_seeds=$N_SEEDS"
  echo "========================================================"

  pixi run python scripts/sampler_benchmark.py \
    --peptide_list_csv "$PEPTIDE_LIST" \
    --out_csv "$SAMP_CSV" \
    --samplers mcmm \
    --n_seeds "$N_SEEDS" \
    --dedup_mode kabsch \
    --dump_sdf_dir "$SDF_DIR" \
    --cartesian_weight 0.33 \
    --dihedral_weight 0.33 \
    --p_rotamer_jump 0.30 \
    --aromatic_wells \
    --skip_mmff_relax \
    --omega_flip_weight "$OMEGA_W" \
    2>&1 | tee "$LOGDIR/${CELL}_sampler.log"

  pixi run python scripts/union_basin_count.py \
    --dbt_sdf_dir "$SDF_DIR" \
    --cart_sdf_dir "$SDF_DIR" \
    --ceiling_sdf_dir "$CEILING_DIR" \
    --cremp_collapse_csv "$CREMP_COLLAPSE_CSV" \
    --out_csv "$COV_CSV" \
    --dedup_mode kabsch \
    2>&1 | tee "$LOGDIR/${CELL}_coverage.log"

  echo "[$(date '+%H:%M:%S')] DONE $CELL"
done

echo "========================================================"
echo "[$(date '+%H:%M:%S')] MOVE B VALIDATION COMPLETE"
echo "========================================================"
