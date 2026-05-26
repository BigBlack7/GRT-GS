#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 2: R2SF/R2IF + environment-map regularization.
#
# Purpose:
#   Test whether reliable far-field specular gating and envmap TV/energy
#   regularization reduce noisy environment maps in weakly reflective and
#   real scenes. This is the far-field branch of RAP-GI.
#
# This script uses one global PCC keep ratio for all scenes. After Step 1,
# override PCC_KEEP if the sweep shows a better global value.
#
# Example:
#   bash train_step_2.sh
#   PCC_KEEP=0.50 bash train_step_2.sh
#   STEP_ITERS=50000 PCC_KEEP=0.65 bash train_step_2.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
STEP_NAME=${STEP_NAME:-step2_r2if_env_${STEP_ITERS}_pcc${PCC_TAG}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

if [ "$STEP_ITERS" -ge 50000 ]; then
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 --save_iterations 18000 20000 22000 25000 26000 27000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000"
else
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 28000 30000 --save_iterations 18000 20000 22000 25000 28000 30000"
fi

SYNTH_EVAL="--eval --white_background"
CHECK_REAL="--test_iterations 10000 15000 18000 20000 --save_iterations 10000 15000 18000 20000"
R2IF_ENV="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate 0.02"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $R2IF_ENV
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
    --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $R2IF_ENV
  eval_real "$model_path"
}

# Reflective rollback probes.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster ""

# Weak/non-reflective envmap pollution probes.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "--lambda_normal_render_depth 0.05"

# Real scenes: test whether envmap regularization helps large-scale weak-reflection scenes.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""

python data_collect.py "$OUT_DIR"
