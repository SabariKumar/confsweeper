#!/usr/bin/env bash
# Broaden the ETKDG-vs-MC ceiling-coverage comparison (2026-06-19 finding follow-up).
#
# The 2-peptide finding (exhaustive ETKDG ties MC B.4 at matched 10k budget) is
# too narrow to say where MC actually helps. This sweep scores both samplers
# through the identical coverage harness across a stratified 8-peptide CREMP
# subset (4 topologies x {aromatic, non-aromatic}, all small for tractable cost),
# generating the ground-truth ceilings first.
#
# Stages (sequential; sampler_benchmark + cremp_collapse_test are checkpointed,
# so a killed run resumes):
#   1. ceilings   — MACE-rescore CREMP reference confs -> per-peptide ceiling SDFs
#   2. ETKDG      — exhaustive_etkdg sampler, n_seeds=10000
#   3. MC (B.4)   — mcmm with aromatic wells + skip-MMFF, 3-way mix
#   4/5. score    — union_basin_count for each sampler vs the generated ceilings

set -euo pipefail
cd /home/sabari/confsweeper

N_SEEDS=10000
PEPS=data/processed/cremp/broaden_peptides.csv
CEIL=results/broaden_ceiling_sdfs
COLLAPSE=results/broaden_collapse.csv

echo "==== [$(date '+%H:%M:%S')] Stage 1/5: generate ceilings ===="
pixi run python scripts/cremp_collapse_test.py run \
  --pickle_dir data/raw/cremp/pickle \
  --peptide_list_csv "$PEPS" \
  --dump_ceiling_sdf_dir "$CEIL" \
  --out_csv "$COLLAPSE"

echo "==== [$(date '+%H:%M:%S')] Stage 2/5: exhaustive ETKDG sampler ===="
pixi run python scripts/sampler_benchmark.py \
  --peptide_list_csv "$PEPS" \
  --out_csv results/broaden_etkdg_sampler.csv \
  --samplers exhaustive_etkdg \
  --n_seeds "$N_SEEDS" \
  --dedup_mode both \
  --dump_sdf_dir results/broaden_etkdg_sdfs

echo "==== [$(date '+%H:%M:%S')] Stage 3/5: MC sampler (B.4 config) ===="
pixi run python scripts/sampler_benchmark.py \
  --peptide_list_csv "$PEPS" \
  --out_csv results/broaden_mc_sampler.csv \
  --samplers mcmm \
  --n_seeds "$N_SEEDS" \
  --dedup_mode both \
  --cartesian_weight 0.33 \
  --dihedral_weight 0.33 \
  --p_rotamer_jump 0.30 \
  --aromatic_wells \
  --skip_mmff_relax \
  --dump_sdf_dir results/broaden_mc_sdfs

echo "==== [$(date '+%H:%M:%S')] Stage 4/5: rename ETKDG SDFs + score coverage ===="
mkdir -p results/broaden_etkdg_sdfs_mcmm
for f in results/broaden_etkdg_sdfs/*_exhaustive_etkdg.sdf; do
  base=$(basename "$f" _exhaustive_etkdg.sdf)
  cp "$f" "results/broaden_etkdg_sdfs_mcmm/${base}_mcmm.sdf"
done
pixi run python scripts/union_basin_count.py \
  --dbt_sdf_dir results/broaden_etkdg_sdfs_mcmm \
  --cart_sdf_dir results/broaden_etkdg_sdfs_mcmm \
  --ceiling_sdf_dir "$CEIL" \
  --cremp_collapse_csv "$COLLAPSE" \
  --out_csv results/broaden_etkdg_coverage.csv \
  --dedup_mode both

echo "==== [$(date '+%H:%M:%S')] Stage 5/5: score MC coverage ===="
pixi run python scripts/union_basin_count.py \
  --dbt_sdf_dir results/broaden_mc_sdfs \
  --cart_sdf_dir results/broaden_mc_sdfs \
  --ceiling_sdf_dir "$CEIL" \
  --cremp_collapse_csv "$COLLAPSE" \
  --out_csv results/broaden_mc_coverage.csv \
  --dedup_mode both

echo "==== [$(date '+%H:%M:%S')] DONE — results/broaden_{etkdg,mc}_coverage.csv ===="
