#!/usr/bin/env bash
# v0.3 combination probe — Move A (concerted χ₁+χ₂) + Move B (cis/trans ω)
# TOGETHER on cremp_sharp, n_seeds=10000, kabsch only.
#
# Motivation: Move A (Step 4) and Move B (Step 8) each ENRICH cremp_sharp's
# basin set (~9 basins vs 2 baseline) but neither reaches the dominant ceiling
# basin alone (cov_bw_ceil=0.000, max_missed_bw=0.724 in both). The consistent
# "enriches but misses the dominant basin" pattern suggests the dominant basin
# may need a COMBINATION of moves rather than any single new move type. This
# probe tests whether A+B together clear cov_bw_ceil > 0.10 before committing
# to implementing Move C (multi-window DBT).
#
# cremp_sharp ONLY — it is the only test peptide with BOTH an aromatic side
# chain (Trp, required by Move A) and an N-methylated amide (required by Move
# B). cremp_typical has neither, so the combination is undefined there (and
# Move A still crashes on non-aromatic peptides — graceful degradation for
# concerted is a deferred follow-up).
#
# All cells on the v0.2 production base (aromatic_wells + skip_mmff_relax,
# p_rotamer_jump=0.30). Three A/B splits to cover the variance (the single-move
# sweeps showed bursty basin counts 2↔9 across weights):
#   - AB1: cart=0.33 dih=0.33 concerted=0.17 omega=0.17  (DBT=0; v0.2 base + A/B)
#   - AB2: cart=0.25 dih=0.25 concerted=0.25 omega=0.25  (DBT=0; balanced)
#   - AB3: cart=0.10 dih=0.10 concerted=0.35 omega=0.35  (DBT=0.10; A/B-heavy)
#
# Decision: if any cell lifts cremp_sharp cov_bw_ceil > 0.10 → the answer is a
# move COMBINATION (pursue the joint/coupled route, e.g. Move D), skip Move C.
# If all stay 0.000 → side-chain + ω moves are jointly insufficient; the
# dominant basin needs a backbone rearrangement → implement Move C.

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/cremp_sharp_only.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_3_combo_probe_logs
mkdir -p "$LOGDIR"

# (cell_name, cartesian_weight, dihedral_weight, concerted_weight, omega_weight)
CELLS=(
  "AB1:0.33:0.33:0.17:0.17"
  "AB2:0.25:0.25:0.25:0.25"
  "AB3:0.10:0.10:0.35:0.35"
)

for cell_spec in "${CELLS[@]}"; do
  IFS=":" read -r CELL CART DIH CONC OMEGA <<< "$cell_spec"
  SDF_DIR="results/sweep_v0_3_combo_probe_${CELL}"
  SAMP_CSV="results/sweep_v0_3_combo_probe_sampler_${CELL}.csv"
  COV_CSV="results/sweep_v0_3_combo_probe_coverage_${CELL}.csv"
  mkdir -p "$SDF_DIR"

  echo "========================================================"
  echo "[$(date '+%H:%M:%S')] CELL=$CELL cart=$CART dih=$DIH concerted=$CONC omega=$OMEGA"
  echo "========================================================"

  pixi run python scripts/sampler_benchmark.py \
    --peptide_list_csv "$PEPTIDE_LIST" \
    --out_csv "$SAMP_CSV" \
    --samplers mcmm \
    --n_seeds "$N_SEEDS" \
    --dedup_mode kabsch \
    --dump_sdf_dir "$SDF_DIR" \
    --cartesian_weight "$CART" \
    --dihedral_weight "$DIH" \
    --concerted_dihedral_weight "$CONC" \
    --omega_flip_weight "$OMEGA" \
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
echo "[$(date '+%H:%M:%S')] COMBO PROBE COMPLETE"
echo "========================================================"
