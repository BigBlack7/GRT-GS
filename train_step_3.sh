#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 3: optional CGI diagnostic.
#
# Current status:
#   The first CGI run improved some scenes but hurt toaster badly, so CGI is
#   not part of the next main line. Keep this script for targeted debugging
#   after Step 2 shows whether R2IF/env alone is stable.
#
# Example:
#   bash train_step_3.sh
#   STEP_ITERS=50000 PCC_KEEP=0.65 bash train_step_3.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-50000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
STEP_NAME=${STEP_NAME:-step3_cgi_debug_${STEP_ITERS}_pcc${PCC_TAG}}
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
CGI_DEFAULT="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_cgi --no_use_ncif --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --cgi_ramp_iters 5000 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate 0.02"
CGI_STRICT="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_cgi --no_use_ncif --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --cgi_ramp_iters 15000 --r2if_specular_alpha 2.0 --r2if_specular_beta 2.0 --r2if_min_specular_gate 0.00"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local suffix="$4"
  local mode_args="$5"
  local extra_args="$6"
  local model_path="$OUT_DIR/$dataset/${scene}_${suffix}"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $mode_args
  eval_synth "$model_path"
}

# Run default/strict only on the two scenes that expose CGI instability most clearly.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell default "$CGI_DEFAULT" ""
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell strict "$CGI_STRICT" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster default "$CGI_DEFAULT" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster strict "$CGI_STRICT" ""

# Controls: one default pass is enough unless Step 2 suggests CGI should be revisited.
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic default "$CGI_DEFAULT" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair default "$CGI_DEFAULT" ""
run_synth /data/zmh/Projects/data/blender/tbell_blender GlossySynthetic tbell default "$CGI_DEFAULT" "--lambda_normal_smooth 1.0"
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot default "$CGI_DEFAULT" ""

python data_collect.py "$OUT_DIR"
