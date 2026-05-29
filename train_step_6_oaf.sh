#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 6: PCC + OAF (Observability-Aware Appearance Fallback).
#
# Purpose:
#   Use PCC as the internal baseline and test whether a low-observability
#   appearance fallback can improve diffuse/real scenes without hurting
#   reflective controls. This disables NCIF/GIP-0 so the gain can be attributed
#   to OAF rather than the previous per-Gaussian irradiance proxy.
#
# Example:
#   bash train_step_6_oaf.sh
#   PCC_KEEP=0.50 OAF_TAU=0.35 bash train_step_6_oaf.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
OAF_TAU=${OAF_TAU:-0.30}
OAF_TAG=${OAF_TAU/./}
STEP_NAME=${STEP_NAME:-step6_oah_oaf_${STEP_ITERS}_pcc${PCC_TAG}_tau${OAF_TAG}_mingate${MIN_TAG}}
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
  CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000 --save_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
fi

SYNTH_EVAL="--eval --white_background"
CHECK_REAL="--test_iterations 10000 12000 14000 15000 18000 20000 --save_iterations 10000 12000 14000 15000 18000 20000"

OAF_COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_oaf --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE"
OAF_DIFFUSE="--oaf_from_iter 18000 --oaf_tau $OAF_TAU --oaf_ramp_iters 4000 --oaf_power 1.0 --oaf_max_blend 0.35"
OAF_REAL="--oaf_from_iter 8000 --oaf_tau 0.35 --oaf_ramp_iters 4000 --oaf_power 1.0 --oaf_max_blend 0.45"
OAF_REFLECTIVE="--oaf_from_iter 20000 --oaf_tau 0.10 --oaf_ramp_iters 4000 --oaf_power 1.5 --oaf_max_blend 0.15"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local oaf_args="$4"
  local extra_args="$5"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $OAF_COMMON $oaf_args
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $OAF_COMMON $OAF_REAL
  eval_real "$model_path"
}

# Diffuse/weak-reflective targets.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$OAF_DIFFUSE" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials "$OAF_DIFFUSE" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "$OAF_DIFFUSE" "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic "$OAF_DIFFUSE" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/hotdog NerfSynthetic hotdog "$OAF_DIFFUSE" "--lambda_normal_render_depth 0.05"

# Real mixed/weak-reflective targets.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""

# Reflective controls: OAF should remain weak where specular evidence is reliable.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$OAF_REFLECTIVE" ""
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot "$OAF_REFLECTIVE" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$OAF_REFLECTIVE" ""

python data_collect.py "$OUT_DIR"
python eval_best_from_curve.py "$OUT_DIR" --max-iteration "$STEP_ITERS" --eval-top-k 5 --save-images
