#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 4 fix check: GIP-0 / NCIF with fixed R2SF gradient-only gate.
#
# Purpose:
#   Re-test the current per-Gaussian irradiance proxy after the far-field
#   specular branch no longer dims forward specular color.
#
# Run after train_step_2_fix.sh:
#   bash train_step_4_fix.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
STEP_NAME=${STEP_NAME:-step4_gip0_gradfix_${STEP_ITERS}_pcc${PCC_TAG}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 25000 26000 27000 28000 30000 --save_iterations 18000 20000 22000 25000 26000 27000 28000 30000"
CHECK_REAL="--test_iterations 10000 12000 15000 18000 20000 --save_iterations 10000 12000 15000 18000 20000"
GIP0_COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate 0.02"
GIP0_STRONG="--ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001"
GIP0_MEDIUM="--ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.003 --lambda_ncif_magnitude 0.001"
GIP0_WEAK="--ncif_from_iter 8000 --ncif_tau 0.06 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001"
GIP0_REAL="--ncif_from_iter 5000 --ncif_tau 0.10 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local gip_args="$4"
  local extra_args="$5"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $GIP0_COMMON $gip_args
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $GIP0_COMMON $GIP0_REAL
  eval_real "$model_path"
}

# Reflective controls: these should recover if the Step 2 failure was forward dimming.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$GIP0_WEAK" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$GIP0_WEAK" ""

# Diffuse / weak-reflective targets.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$GIP0_STRONG" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic "$GIP0_MEDIUM" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials "$GIP0_MEDIUM" ""

# Real scene smoke test.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"

python data_collect.py "$OUT_DIR"
