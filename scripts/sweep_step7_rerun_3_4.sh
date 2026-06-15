#!/usr/bin/env bash
# Re-run cells 3 and 4 after the rdMolTransforms import-bug fix.
set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=5000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_step7_logs

CELLS=(
  "cell3_dihedral_only:0.00:0.33"
  "cell4_three_way:0.33:0.33"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL CART_W DIH_W <<< "$cell_spec"
  SDF_DIR="results/sweep_step7_${CELL}"
  SAMP_CSV="results/sweep_step7_sampler_${CELL}.csv"
  COV_CSV="results/sweep_step7_coverage_${CELL}.csv"

  # Clear stale outputs from the failed run.
  rm -f "$SAMP_CSV" "$COV_CSV"
  rm -rf "$SDF_DIR"
  mkdir -p "$SDF_DIR"

  echo "[$(date '+%H:%M:%S')] CELL=$CELL  cart_w=$CART_W  dih_w=$DIH_W"

  pixi run python scripts/sampler_benchmark.py \
    --peptide_list_csv "$PEPTIDE_LIST" \
    --out_csv "$SAMP_CSV" \
    --samplers mcmm \
    --n_seeds "$N_SEEDS" \
    --dedup_mode both \
    --dump_sdf_dir "$SDF_DIR" \
    --cartesian_weight "$CART_W" \
    --dihedral_weight "$DIH_W" \
    2>&1 | tee "$LOGDIR/${CELL}_sampler.log"

  pixi run python scripts/union_basin_count.py \
    --dbt_sdf_dir "$SDF_DIR" \
    --cart_sdf_dir "$SDF_DIR" \
    --ceiling_sdf_dir "$CEILING_DIR" \
    --cremp_collapse_csv "$CREMP_COLLAPSE_CSV" \
    --out_csv "$COV_CSV" \
    --dedup_mode both \
    2>&1 | tee "$LOGDIR/${CELL}_coverage.log"

  echo "[$(date '+%H:%M:%S')] DONE $CELL"
done
echo "[$(date '+%H:%M:%S')] RERUN COMPLETE"
