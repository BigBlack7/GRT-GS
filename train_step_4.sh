#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 4: OAH-GS diffuse low-frequency branch proxy.
#
# Purpose:
#   Test the current per-Gaussian NCIF residual as GIP-0, a minimal proxy for
#   probe-based diffuse irradiance. This is still not the final grid-probe
#   implementation, but it tells us whether low-frequency diffuse correction
#   helps before we implement true spatial probes.
#
# Run this after Step 1 and Step 2. Override PCC_KEEP according to the Step 1
# sweep if needed.
#
# Example:
#   bash train_step_4.sh
#   PCC_KEEP=0.50 bash train_step_4.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
STEP_NAME=${STEP_NAME:-step4_oah_gip0_envgate_${STEP_ITERS}_pcc${PCC_TAG}_mingate${MIN_TAG}}
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
NCIF_COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE"
NCIF_STRONG="--ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001"
NCIF_MEDIUM="--ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.003 --lambda_ncif_magnitude 0.001"
NCIF_WEAK="--ncif_from_iter 8000 --ncif_tau 0.06 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001"
NCIF_REAL="--ncif_from_iter 5000 --ncif_tau 0.10 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local ncif_args="$4"
  local extra_args="$5"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $NCIF_COMMON $ncif_args
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $NCIF_COMMON $NCIF_REAL
  eval_real "$model_path"
}

# Diffuse/weak-reflective scenes: primary GIP-0 target.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$NCIF_STRONG" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ficus NerfSynthetic ficus "$NCIF_STRONG" "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/hotdog NerfSynthetic hotdog "$NCIF_STRONG" "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/lego NerfSynthetic lego "--ncif_from_iter 2000 --ncif_tau 0.20 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.004 --lambda_ncif_magnitude 0.001" "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials "$NCIF_MEDIUM" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic "$NCIF_MEDIUM" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "$NCIF_MEDIUM" "--lambda_normal_render_depth 0.05"

# Real scenes: large-scale weak/reflection-mixed target.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""

# Reflective controls: GIP-0 should not steal specular explanation.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$NCIF_WEAK" ""
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot "$NCIF_WEAK" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$NCIF_WEAK" ""

python data_collect.py "$OUT_DIR"
