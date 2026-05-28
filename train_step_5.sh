#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 5: OAH-GS GIP-1 learnable grid irradiance probes.
#
# Purpose:
#   Test the true low-frequency diffuse probe branch after the code-level
#   GIP-1 module is implemented. This script intentionally checks for
#   --use_probe_gi before training so it will not silently run an old
#   NCIF-only code path.
#
# Required future flags:
#   --use_probe_gi
#   --probe_lr
#   --probe_from_iter
#   --probe_ramp_iters
#   --probe_grid_res
#   --probe_sh_degree
#   --probe_tau
#   --probe_diffuse_mu
#   --probe_diffuse_nu
#   --lambda_probe_smooth
#   --lambda_probe_energy
#   --lambda_probe_magnitude
#
# Example after implementation:
#   bash train_step_5.sh
#   PCC_KEEP=0.50 PROBE_GRID_RES=8 bash train_step_5.sh

if ! python train.py --help 2>&1 | grep -q -- "--use_probe_gi"; then
  echo "[停止] 当前代码还没有注册 --use_probe_gi。请先实现 GIP-1 probe field，再运行 train_step_5.sh。"
  exit 1
fi

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
PROBE_GRID_RES=${PROBE_GRID_RES:-8}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
STEP_NAME=${STEP_NAME:-step5_oah_gip1_probe_${STEP_ITERS}_pcc${PCC_TAG}_grid${PROBE_GRID_RES}_mingate${MIN_TAG}}
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

PROBE_COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_probe_gi --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE --probe_grid_res $PROBE_GRID_RES --probe_sh_degree 2 --probe_lr 0.002"
PROBE_STRONG="--probe_from_iter 2000 --probe_tau 0.25 --probe_ramp_iters 4000 --probe_diffuse_mu 1.0 --probe_diffuse_nu 1.0 --lambda_probe_smooth 0.005 --lambda_probe_energy 0.0001 --lambda_probe_magnitude 0.001"
PROBE_MEDIUM="--probe_from_iter 3000 --probe_tau 0.15 --probe_ramp_iters 5000 --probe_diffuse_mu 1.0 --probe_diffuse_nu 1.0 --lambda_probe_smooth 0.003 --lambda_probe_energy 0.0001 --lambda_probe_magnitude 0.001"
PROBE_REAL="--probe_from_iter 5000 --probe_tau 0.12 --probe_ramp_iters 5000 --probe_diffuse_mu 1.0 --probe_diffuse_nu 1.0 --lambda_probe_smooth 0.004 --lambda_probe_energy 0.0002 --lambda_probe_magnitude 0.001"
PROBE_WEAK="--probe_from_iter 8000 --probe_tau 0.06 --probe_ramp_iters 5000 --probe_diffuse_mu 1.0 --probe_diffuse_nu 1.0 --lambda_probe_smooth 0.001 --lambda_probe_energy 0.0001 --lambda_probe_magnitude 0.001"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local probe_args="$4"
  local extra_args="$5"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    $SYNTH_EVAL $CHECK_SYNTH $extra_args $PROBE_COMMON $probe_args
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $PROBE_COMMON $PROBE_REAL
  eval_real "$model_path"
}

# Diffuse/weak-reflective probe targets.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$PROBE_STRONG" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials "$PROBE_MEDIUM" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "$PROBE_MEDIUM" "--lambda_normal_render_depth 0.05"
run_synth /data/zmh/Projects/data/nerf_synthetic/mic NerfSynthetic mic "$PROBE_MEDIUM" ""

# Real scenes.
run_real /data/zmh/Projects/data/ref_real/gardenspheres gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""

# Reflective controls.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$PROBE_WEAK" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$PROBE_WEAK" ""
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot "$PROBE_WEAK" ""

python data_collect.py "$OUT_DIR"
