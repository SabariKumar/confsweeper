#!/usr/bin/env bash
# v0.2 Step 4 recovery: the cremp_typical cell-1 (aromatic_off) data is
# fine (11 + 15 basins via kabsch + crest at n_seeds=10000). The cremp_sharp
# cell-1 rows crashed at 14:16:30 with CUDA OOM (Step-5 test suite was
# competing for GPU memory in parallel — my mistake). Surgical fix:
#   - strip the two crashed cremp_sharp rows from cell-1 sampler CSV
#   - delete the empty cell-1 cremp_sharp SDF
#   - re-run sampler_benchmark: sampler_benchmark.py resume-aware logic
#     skips cremp_typical (already done), re-runs cremp_sharp
#   - run union_basin_count.py on cell 1
#   - run cell 2 (aromatic_on) from scratch on both peptides
# Test suite is NOT running in parallel this time.

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPTIDE_LIST=data/processed/cremp/sweep_step7_peptides.csv
CEILING_DIR=results/cremp_ceiling_sdfs
CREMP_COLLAPSE_CSV=results/cremp_collapse_test_dual.csv
LOGDIR=results/sweep_v0_2_step4_logs

# Cell 1 cleanup: strip the 2 crashed cremp_sharp rows; delete empty SDF.
SAMP_CSV_OFF=results/sweep_v0_2_step4_sampler_aromatic_off.csv
SDF_DIR_OFF=results/sweep_v0_2_step4_aromatic_off
TMP_CSV=$(mktemp)
grep -v "^cremp:S.S.N.MeW.MeA.MeN," "$SAMP_CSV_OFF" > "$TMP_CSV"
mv "$TMP_CSV" "$SAMP_CSV_OFF"
rm -f "$SDF_DIR_OFF/cremp_S.S.N.MeW.MeA.MeN_mcmm.sdf"
echo "[$(date '+%H:%M:%S')] cleaned up cell-1 cremp_sharp; remaining rows:"
wc -l "$SAMP_CSV_OFF"

# Cell 1 re-run (cremp_sharp only — resume skips cremp_typical).
echo "========================================================"
echo "[$(date '+%H:%M:%S')] CELL=aromatic_off  (cremp_sharp rerun only)"
echo "========================================================"
pixi run python scripts/sampler_benchmark.py \
  --peptide_list_csv "$PEPTIDE_LIST" \
  --out_csv "$SAMP_CSV_OFF" \
  --samplers mcmm \
  --n_seeds "$N_SEEDS" \
  --dedup_mode both \
  --dump_sdf_dir "$SDF_DIR_OFF" \
  --cartesian_weight 0.33 \
  --dihedral_weight 0.33 \
  --p_rotamer_jump 0.30 \
  --no-aromatic_wells \
  2>&1 | tee -a "$LOGDIR/aromatic_off_sampler.log"

# Cell 1 union_basin_count.
COV_CSV_OFF=results/sweep_v0_2_step4_coverage_aromatic_off.csv
pixi run python scripts/union_basin_count.py \
  --dbt_sdf_dir "$SDF_DIR_OFF" \
  --cart_sdf_dir "$SDF_DIR_OFF" \
  --ceiling_sdf_dir "$CEILING_DIR" \
  --cremp_collapse_csv "$CREMP_COLLAPSE_CSV" \
  --out_csv "$COV_CSV_OFF" \
  --dedup_mode both \
  2>&1 | tee "$LOGDIR/aromatic_off_coverage.log"
echo "[$(date '+%H:%M:%S')] DONE aromatic_off"

# Cell 2 (aromatic_on) from scratch.
SAMP_CSV_ON=results/sweep_v0_2_step4_sampler_aromatic_on.csv
SDF_DIR_ON=results/sweep_v0_2_step4_aromatic_on
COV_CSV_ON=results/sweep_v0_2_step4_coverage_aromatic_on.csv
mkdir -p "$SDF_DIR_ON"

echo "========================================================"
echo "[$(date '+%H:%M:%S')] CELL=aromatic_on  (full rerun)"
echo "========================================================"
pixi run python scripts/sampler_benchmark.py \
  --peptide_list_csv "$PEPTIDE_LIST" \
  --out_csv "$SAMP_CSV_ON" \
  --samplers mcmm \
  --n_seeds "$N_SEEDS" \
  --dedup_mode both \
  --dump_sdf_dir "$SDF_DIR_ON" \
  --cartesian_weight 0.33 \
  --dihedral_weight 0.33 \
  --p_rotamer_jump 0.30 \
  --aromatic_wells \
  2>&1 | tee "$LOGDIR/aromatic_on_sampler.log"

pixi run python scripts/union_basin_count.py \
  --dbt_sdf_dir "$SDF_DIR_ON" \
  --cart_sdf_dir "$SDF_DIR_ON" \
  --ceiling_sdf_dir "$CEILING_DIR" \
  --cremp_collapse_csv "$CREMP_COLLAPSE_CSV" \
  --out_csv "$COV_CSV_ON" \
  --dedup_mode both \
  2>&1 | tee "$LOGDIR/aromatic_on_coverage.log"
echo "[$(date '+%H:%M:%S')] DONE aromatic_on"

echo "========================================================"
echo "[$(date '+%H:%M:%S')] STEP 4 RECOVERY COMPLETE"
echo "========================================================"
