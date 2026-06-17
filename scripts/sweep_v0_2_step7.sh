#!/usr/bin/env bash
# v0.2 Step 7 — Validation B: 2×2 ablation matrix to decompose the
# cremp_sharp failure mode that Step 4 left at coverage_bw_ceiling=0.000
# despite the aromatic-wells fix. Reuses cells B.1 (aromatic_off,
# skip_mmff_off = Step-4 aromatic_off) and B.2 (aromatic_on,
# skip_mmff_off = Step-4 aromatic_on); only runs the two new cells:
#
#   B.3 = aromatic_off + skip_mmff_on  → isolates MMFF snap-back hypothesis
#   B.4 = aromatic_on  + skip_mmff_on  → both fixes together
#
# Decomposition: if B.3 lifts cremp_sharp off zero, MMFF was the dominant
# issue. If only B.4 lifts it (neither B.2 nor B.3 alone), the failure
# is multiplicative and both fixes are needed together. If all 4 stay at
# zero, the cremp_sharp failure is structural beyond what v0.2 can fix
# (deferred to v0.3 — e.g. concerted χ₁ + χ₂ rotation).

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_2_step7_logs
mkdir -p "$LOGDIR"

# (cell_name, --aromatic_wells flag)
# skip_mmff_relax is always on for these two cells (the new axis being tested).
CELLS=(
  "B3_aromatic_off_skip_mmff_on:--no-aromatic_wells"
  "B4_aromatic_on_skip_mmff_on:--aromatic_wells"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL FLAG <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_2_step7_${CELL}"
  SAMP_CSV="results/sweep_v0_2_step7_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_2_step7_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL  $FLAG --skip_mmff_relax  n_seeds=$N_SEEDS"
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
    --skip_mmff_relax \
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
echo "[$(date '+%H:%M:%S')] STEP 7 COMPLETE"
echo "========================================================"
