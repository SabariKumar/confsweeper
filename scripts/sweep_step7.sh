#!/usr/bin/env bash
# Step 7 sweep: 4 mix cells × 2 CREMP peptides at n_seeds=5000.
# Each cell: sampler_benchmark.py (writes basin SDFs) → union_basin_count.py
# (computes Boltzmann coverage vs CREMP ceiling SDFs by passing the cell
# dir as BOTH --dbt_sdf_dir and --cart_sdf_dir so union(A,A)=A).

set -euo pipefail

cd /home/sabari/confsweeper

N_SEEDS=5000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_step7_logs
mkdir -p "$LOGDIR"

# (cell_name, cart_w, dih_w)
CELLS=(
  "cell1_dbt_only:0.00:0.00"
  "cell2_cart_only:0.33:0.00"
  "cell3_dihedral_only:0.00:0.33"
  "cell4_three_way:0.33:0.33"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL CART_W DIH_W <<< "$cell_spec"
  SDF_DIR="results/sweep_step7_${CELL}"
  SAMP_CSV="results/sweep_step7_sampler_${CELL}.csv"
  COV_CSV="results/sweep_step7_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL  cart_w=$CART_W  dih_w=$DIH_W"
  echo "========================================================"

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

echo "========================================================"
echo "[$(date '+%H:%M:%S')] SWEEP COMPLETE"
echo "========================================================"
