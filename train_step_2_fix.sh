#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 2 fix check: OAH-GS R2SF env-light-only gradient gate.
#
# Purpose:
#   Re-test the far-field specular branch after moving the R2SF gate from
#   the whole specular branch to the envmap light query only.
#
# Run this before any full rerun:
#   bash train_step_2_fix.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
STEP_NAME=${STEP_NAME:-step2_r2sf_envgate_${STEP_ITERS}_pcc${PCC_TAG}_mingate${MIN_TAG}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000 --save_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
CHECK_REAL="--test_iterations 10000 12000 14000 15000 18000 20000 --save_iterations 10000 12000 14000 15000 18000 20000"
R2SF_ENV="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $R2SF_ENV
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $R2SF_ENV
  eval_real "$model_path"
}

# The two scenes that exposed forward-dimming failures.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster ""

# Diffuse / weak-reflective controls.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials ""

# Real scene smoke test.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"

python data_collect.py "$OUT_DIR"
