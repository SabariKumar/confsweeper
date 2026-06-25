#!/usr/bin/env bash
# v0.3 Step 4 — Validation A: concerted-dihedral layered on the v0.2 production
# mix, cremp_typical + cremp_sharp at n_seeds=10000.
#
# Three cells, all using v0.2's locked production stack underneath
# (cart=0.33, dih=0.33, p_rotamer_jump=0.30, --aromatic_wells, --skip_mmff_relax):
#   - A1_concerted_0_00 : concerted_dihedral_weight=0.00, DBT residual=0.34
#                         → exact reproduction of v0.2 Step-7 B.4 production cell
#                         → regression check that v0.3 Steps 2-3 didn't perturb v0.2
#   - A2_concerted_0_17 : concerted_dihedral_weight=0.17, DBT residual=0.17
#                         → half-strength concerted layered on v0.2 mix
#   - A3_concerted_0_34 : concerted_dihedral_weight=0.34, DBT residual=0.00
#                         → full-strength concerted (matches cart/dihedral shares);
#                            tests whether the v0.3 move closes cremp_sharp
#
# Decision point at end of sweep (locked Step 5 of v0.3 plan):
#   - If A.2 or A.3 lifts cremp_sharp's cov_bw_ceil above 0.10 → v0.3 Move A is
#     the production fix; skip directly to Step 13 / Step 14.
#   - If both leave cremp_sharp at zero → proceed to Move B (cis-trans ω).

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_3_step4_logs
mkdir -p "$LOGDIR"

# (cell_name, concerted_dihedral_weight)
CELLS=(
  "A1_concerted_0_00:0.00"
  "A2_concerted_0_17:0.17"
  "A3_concerted_0_34:0.34"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL CONCERTED_W <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_3_step4_${CELL}"
  SAMP_CSV="results/sweep_v0_3_step4_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_3_step4_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL  concerted_dihedral_weight=$CONCERTED_W  n_seeds=$N_SEEDS"
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
    --aromatic_wells \
    --skip_mmff_relax \
    --concerted_dihedral_weight "$CONCERTED_W" \
    --p_concerted_jump 0.30 \
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
