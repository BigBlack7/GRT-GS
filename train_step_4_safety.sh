#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 4 safety sweep: conservative OAH-GS GIP-0 settings.
#
# Purpose:
#   Step 4 fix proves that GIP-0 can rescue reflective rollback scenes, but
#   the previous strong diffuse setting hurts chair/materials. This script
#   keeps the successful weak reflective setting and tests safer low-frequency
#   diffuse residual settings before any full Step 4 rerun.
#
# Run:
#   bash train_step_4_safety.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
STEP_NAME=${STEP_NAME:-step4_oah_gip0_safety_${STEP_ITERS}_pcc${PCC_TAG}_mingate${MIN_TAG}}
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
OAH_COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE"

# Successful setting from Step 4 fix for reflective controls.
GIP0_REFLECTIVE_WEAK="--ncif_from_iter 8000 --ncif_tau 0.06 --ncif_ramp_iters 5000 --ncif_diffuse_mu 1.0 --ncif_diffuse_nu 1.0 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001"

# Safer settings for diffuse/weak-reflective scenes.
GIP0_SAFE005="--ncif_from_iter 10000 --ncif_tau 0.05 --ncif_ramp_iters 8000 --ncif_diffuse_mu 1.0 --ncif_diffuse_nu 2.0 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.002"
GIP0_SAFE008="--ncif_from_iter 10000 --ncif_tau 0.08 --ncif_ramp_iters 8000 --ncif_diffuse_mu 1.0 --ncif_diffuse_nu 2.0 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.002"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local suffix="$4"
  local gip_args="$5"
  local extra_args="$6"
  local model_path="$OUT_DIR/$dataset/${scene}_${suffix}"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $OAH_COMMON $gip_args
  eval_synth "$model_path"
}

run_real() {
  local source_path="$1"
  local scene="$2"
  local suffix="$3"
  local resolution="$4"
  local extra_args="$5"
  local gip_args="$6"
  local model_path="$OUT_DIR/RefReal/${scene}_${suffix}"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $OAH_COMMON $gip_args
  eval_real "$model_path"
}

# Reflective controls: keep the setting that made Step 4 fix stable.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell weak "$GIP0_REFLECTIVE_WEAK" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster weak "$GIP0_REFLECTIVE_WEAK" ""

# Diffuse/weak-reflective scenes: test conservative settings before full Step 4.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair safe005 "$GIP0_SAFE005" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair safe008 "$GIP0_SAFE008" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials safe005 "$GIP0_SAFE005" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials safe008 "$GIP0_SAFE008" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic safe005 "$GIP0_SAFE005" ""

# Real smoke test: keep a conservative late low-frequency residual.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres safe005 4 "--lambda_normal_smooth 0.45" "$GIP0_SAFE005"

python data_collect.py "$OUT_DIR"
