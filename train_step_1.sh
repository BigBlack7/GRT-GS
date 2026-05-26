#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 1: PCC keep-ratio sweep.
#
# Purpose:
#   Find a stable global PCC soft-reset strength before testing R2IF/NCIF.
#   This script isolates PCC: R2IF, NCIF, CGI and envmap regularization are off.
#
# Default:
#   - 50k diagnostic schedule, matching the rollback baseline.
#   - Sweep pcc_keep_ratio = 0.50 / 0.65 / 0.75 on the key rollback set.
#
# Example:
#   bash train_step_1.sh
#   STEP_ITERS=30000 OUT_ROOT=/data2/zmh/output_physnorm_steps_30k bash train_step_1.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-50000}
STEP_NAME=${STEP_NAME:-step1_pcc_sweep_${STEP_ITERS}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

if [ "$STEP_ITERS" -ge 50000 ]; then
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 41000 42000 44000 45000 46000 47000 48000 50000 --save_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 41000 42000 44000 45000 46000 47000 48000 50000"
else
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 28000 30000 --save_iterations 18000 20000 22000 25000 28000 30000"
fi

SYNTH_EVAL="--eval --white_background"
PCC_ONLY="--use_pcc --no_use_ncif --no_use_r2if --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"
RATIOS=${RATIOS:-"0.50 0.65 0.75"}

run_synth_ratio() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local ratio="$4"
  local extra_args="$5"
  local tag="${ratio/./}"
  local model_path="$OUT_DIR/$dataset/${scene}_keep${tag}"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $PCC_ONLY --pcc_keep_ratio "$ratio"
  eval_synth "$model_path"
}

for ratio in $RATIOS; do
  # Severe final rollback in the 50k baseline.
  run_synth_ratio /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$ratio" ""
  run_synth_ratio /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$ratio" ""

  # Transient stage shock controls.
  run_synth_ratio /data/zmh/Projects/data/ShinyBlender/ball ShinyBlender ball "$ratio" "--lambda_normal_smooth 1.0"
  run_synth_ratio /data/zmh/Projects/data/blender/tbell_blender GlossySynthetic tbell "$ratio" "--lambda_normal_smooth 1.0"

  # Weak/non-reflective delayed rollback probes.
  run_synth_ratio /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$ratio" ""
  run_synth_ratio /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic "$ratio" ""
done

python data_collect.py "$OUT_DIR"
