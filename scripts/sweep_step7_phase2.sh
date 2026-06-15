#!/usr/bin/env bash
# Step 7 phase 2: headline (n_seeds=10000, p_rotamer_jump=0.30) + diagnostic
# (n_seeds=10000, p_rotamer_jump=0.70) on both CREMP peptides at the 3-way
# (cart=0.33, dih=0.33) mix locked by phase 1's sweep.
#
#   - Headline confirms cremp_typical 0.989 holds at the production budget
#     (phase 1 ran at half-budget n_seeds=5000).
#   - Diagnostic tests the snap-back hypothesis on cremp_sharp per the locked
#     Step-1 follow-up trigger ("if snap-back rate > 50 %, raise p_rotamer_jump
#     toward 0.5 or higher"). cell-4 phase-1 result was 0.000 at p=0.30.

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_step7_logs
mkdir -p "$LOGDIR"

# (run_name, cart_w, dih_w, p_rotamer_jump)
RUNS=(
  "headline_n10k_pjump30:0.33:0.33:0.30"
  "diagnostic_n10k_pjump70:0.33:0.33:0.70"
)

for run_spec in "${RUNS[@]}"; do
  IFS=":" read -r RUN CART_W DIH_W PJUMP <<< "$run_spec"
  SDF_DIR="results/sweep_step7_${RUN}"
  SAMP_CSV="results/sweep_step7_sampler_${RUN}.csv"
  COV_CSV="results/sweep_step7_coverage_${RUN}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] RUN=$RUN  cart_w=$CART_W  dih_w=$DIH_W  p_rotamer_jump=$PJUMP  n_seeds=$N_SEEDS"
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
    --p_rotamer_jump "$PJUMP" \
    2>&1 | tee "$LOGDIR/${RUN}_sampler.log"

  pixi run python scripts/union_basin_count.py \
    --dbt_sdf_dir "$SDF_DIR" \
    --cart_sdf_dir "$SDF_DIR" \
    --ceiling_sdf_dir "$CEILING_DIR" \
    --cremp_collapse_csv "$CREMP_COLLAPSE_CSV" \
    --out_csv "$COV_CSV" \
    --dedup_mode both \
    2>&1 | tee "$LOGDIR/${RUN}_coverage.log"

  echo "[$(date '+%H:%M:%S')] DONE $RUN"
done

echo "========================================================"
echo "[$(date '+%H:%M:%S')] PHASE 2 COMPLETE"
echo "========================================================"
