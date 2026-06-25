#!/usr/bin/env bash
# v0.3 Step 10 — Validation C: large-window DBT (Move C) on cremp_sharp,
# n_seeds=10000, kabsch only. Sweeps the backbone window size W ∈ {10,13,16}.
#
# Move A (Step 4), Move B (Step 8), and the A+B combination (combo probe) all
# left cremp_sharp at cov_bw_ceil=0.000 with max_missed_bw pinned at 0.724 —
# the dominant ceiling basin needs a larger concerted BACKBONE rearrangement
# than single-window (W=7) DBT produces. Move C drives one backbone dihedral
# over a larger window and re-closes more free dihedrals.
#
# Design isolates the backbone-window-size effect. All cells keep the v0.2
# side-chain base (cart=0.33, dih=0.33, aromatic_wells, skip_mmff_relax) and
# put the remaining 0.34 on the BACKBONE concerted move:
#   - baseline (NOT re-run here): W=7 DBT at 0.34 = the move_b B1 cell, already
#     measured at cov_bw_ceil=0.000, 2 basins. That is the comparison control.
#   - C_w10 : large_window_dbt_weight=0.34, large_window_size=10 (W=7 DBT → 0)
#   - C_w13 : large_window_dbt_weight=0.34, large_window_size=13
#   - C_w16 : large_window_dbt_weight=0.34, large_window_size=16
# So B1 → C_w* isolates "W=7 backbone move" → "W={10,13,16} backbone move".
#
# cremp_sharp ONLY (18-atom ring; hosts up to W=16). cremp_typical is a
# 12-atom ring < 16 and would degrade to plain DBT, so it is uninformative for
# the size sweep (and W=16 is undefined there).
#
# Decision (Step 10): if any window size lifts cremp_sharp cov_bw_ceil > 0.10 →
# Move C is the fix (lock the winning W, proceed to Step 13). If all stay
# 0.000 → escalate to Move D (coupled backbone + side-chain).

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/cremp_sharp_only.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_3_move_c_logs
mkdir -p "$LOGDIR"

# (cell_name, large_window_size)
CELLS=(
  "C_w10:10"
  "C_w13:13"
  "C_w16:16"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL WSIZE <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_3_move_c_${CELL}"
  SAMP_CSV="results/sweep_v0_3_move_c_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_3_move_c_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL large_window_dbt_weight=0.34 large_window_size=$WSIZE"
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
    --large_window_dbt_weight 0.34 \
    --large_window_size "$WSIZE" \
    --p_rotamer_jump 0.30 \
    --aromatic_wells \
    --skip_mmff_relax \
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
echo "[$(date '+%H:%M:%S')] MOVE C VALIDATION COMPLETE"
echo "========================================================"
