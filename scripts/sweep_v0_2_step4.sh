#!/usr/bin/env bash
# v0.2 Step 4 — Validation A: aromatic-wells-alone sweep on cremp_typical +
# cremp_sharp at n_seeds=10000.
#
# Two cells, same 3-way mix (cart=0.33, dih=0.33, p_rotamer_jump=0.30):
#   - aromatic_off : --no-aromatic_wells  (replicates issue-#12 phase-2 HEADLINE
#                                          as the apples-to-apples baseline at
#                                          this branch — also a regression check
#                                          that Steps 2-3 didn't perturb v0.1)
#   - aromatic_on  : --aromatic_wells     (engages the v0.2 four-well aromatic set
#                                          for any bond whose downstream endpoint
#                                          is aromatic, i.e. NMe-Trp χ₂ on
#                                          cremp_sharp; sp3 bonds keep the
#                                          existing (-60, 60, 180) wells)
#
# Per-cell pipeline (same as Step 7 phase 2): sampler_benchmark.py dumps basin
# SDFs → union_basin_count.py (passing the same dir as both --dbt_sdf_dir and
# --cart_sdf_dir so union(A, A) = A gives per-cell Boltzmann coverage) → coverage
# CSV with the headline `coverage_bw_ceiling` column.

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_2_step4_logs
mkdir -p "$LOGDIR"

# (cell_name, --aromatic_wells flag)
CELLS=(
  "aromatic_off:--no-aromatic_wells"
  "aromatic_on:--aromatic_wells"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL FLAG <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_2_step4_${CELL}"
  SAMP_CSV="results/sweep_v0_2_step4_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_2_step4_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL  flag=$FLAG  n_seeds=$N_SEEDS"
  echo "========================================================"

  pixi run python scripts/sampler_benchmark.py \
    --peptide_list_csv "$PEPTIDE_LIST" \
    --out_csv "$SAMP_CSV" \
    --samplers mcmm \
    --n_seeds "$N_SEEDS" \
    --dedup_mode both \
    --dump_sdf_dir "$SDF_DIR" \
    --cartesian_weight 0.33 \
    --dihedral_weight 0.33 \
    --p_rotamer_jump 0.30 \
    "$FLAG" \
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
echo "[$(date '+%H:%M:%S')] STEP 4 COMPLETE"
echo "========================================================"
