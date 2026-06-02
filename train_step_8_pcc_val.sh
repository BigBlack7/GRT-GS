#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 8: fixed-PCC internal baseline with validation-based checkpoint selection.
#
# Purpose:
#   Rebuild the strong internal PCC baseline under a paper-safe checkpoint
#   protocol. Training holds out a deterministic subset of training cameras as
#   validation cameras, selects checkpoints from val/val_fast curves, and only
#   uses test cameras for final offline metrics.
#
# Example:
#   bash train_step_8_pcc_val.sh
#   PCC_KEEP=0.50 VAL_HOLD=10 VAL_MAX=6 bash train_step_8_pcc_val.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
VAL_HOLD=${VAL_HOLD:-8}
VAL_OFFSET=${VAL_OFFSET:-0}
VAL_MAX=${VAL_MAX:-8}
STEP_NAME=${STEP_NAME:-step8_pcc_val_${STEP_ITERS}_pcc${PCC_TAG}_hold${VAL_HOLD}_max${VAL_MAX}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

echo "[Step8 PCC validation] OUT_DIR=$OUT_DIR"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

if [ "$STEP_ITERS" -ge 50000 ]; then
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 --save_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000"
else
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000 --save_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
fi

SYNTH_EVAL="--eval --white_background"
CHECK_REAL="--test_iterations 10000 12000 14000 15000 18000 20000 --save_iterations 10000 12000 14000 15000 18000 20000"
VAL_ARGS="--val_hold $VAL_HOLD --val_offset $VAL_OFFSET --val_max $VAL_MAX"
PCC_ONLY="--use_pcc --pcc_keep_ratio $PCC_KEEP --no_use_ncif --no_use_r2if --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $VAL_ARGS $CHECK_SYNTH $extra_args $PCC_ONLY
  eval_synth "$model_path"
}

run_real() {
  local source_path="$1"
  local scene="$2"
  local resolution="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/RefReal/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    --eval $VAL_ARGS --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $PCC_ONLY
  eval_real "$model_path"
}

# Reflective controls.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell ""
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster ""

# Diffuse/weak-reflective controls.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair ""
run_synth /data/zmh/Projects/data/nerf_synthetic/hotdog NerfSynthetic hotdog "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "--lambda_normal_render_depth 0.05"

# Real scenes.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""

if [ ! -d "$OUT_DIR" ]; then
  echo "[停止] 未找到输出目录: $OUT_DIR"
  exit 1
fi

python data_collect.py "$OUT_DIR"
python eval_best_from_curve.py "$OUT_DIR" --select-splits val,val_fast --selection-mode curve --summary-suffix val --max-iteration "$STEP_ITERS" --eval-top-k 1 --save-images
