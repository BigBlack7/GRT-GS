#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 9: PRT-GS / Gaussian Radiance Transfer.
#
# Purpose:
#   Test a lightweight PRT-inspired diffuse transfer branch on top of the
#   fixed PCC=0.65 baseline. This replaces dense grid probes with shared
#   low-frequency lighting SH and per-Gaussian transfer occlusion.

if ! python train.py --help 2>&1 | grep -q -- "--use_prt_gs"; then
  echo "[停止] 当前代码还没有注册 --use_prt_gs。请先同步 PRT-GS 代码后再运行。"
  exit 1
fi

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
STEP_ITERS=${STEP_ITERS:-30000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
R2SF_MIN_GATE=${R2SF_MIN_GATE:-0.15}
MIN_TAG=${R2SF_MIN_GATE/./}
VAL_HOLD=${VAL_HOLD:-8}
VAL_OFFSET=${VAL_OFFSET:-0}
VAL_MAX=${VAL_MAX:-8}
PRT_TAU=${PRT_TAU:-0.12}
TAU_TAG=${PRT_TAU/./}
STEP_NAME=${STEP_NAME:-step9_prt_gs_${STEP_ITERS}_pcc${PCC_TAG}_tau${TAU_TAG}_mingate${MIN_TAG}_valhold${VAL_HOLD}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

eval_synth() {
  python eval.py -m "$1" --iteration "$STEP_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

CHECK_SYNTH="--iterations $STEP_ITERS --test_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000 --save_iterations 18000 20000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
CHECK_REAL="--test_iterations 10000 12000 14000 15000 18000 20000 --save_iterations 10000 12000 14000 15000 18000 20000"
VAL_ARGS="--val_hold $VAL_HOLD --val_offset $VAL_OFFSET --val_max $VAL_MAX"

COMMON="--use_pcc --pcc_keep_ratio $PCC_KEEP --use_r2if --use_prt_gs --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate $R2SF_MIN_GATE"
PRT_COMMON="--prt_sh_degree 2 --prt_lr 0.001 --prt_occlusion_lr 0.001 --prt_tau $PRT_TAU --prt_ramp_iters 5000 --prt_diffuse_mu 1.0 --prt_diffuse_nu 1.0 --lambda_prt_smooth 0.001 --lambda_prt_energy 0.0001 --lambda_prt_magnitude 0.001 --lambda_prt_occlusion 0.0001"
PRT_DIFFUSE="--prt_from_iter 3000"
PRT_REAL="--prt_from_iter 5000"
PRT_REFLECT="--prt_from_iter 8000 --prt_tau 0.06 --lambda_prt_smooth 0.0005 --lambda_prt_magnitude 0.0005"

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local prt_args="$4"
  local extra_args="$5"
  local model_path="$OUT_DIR/$dataset/$scene"

  python train.py \
    -s "$source_path" \
    -m "$model_path" \
    --eval --white_background $VAL_ARGS $CHECK_SYNTH $extra_args $COMMON $PRT_COMMON $prt_args
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
    --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $COMMON $PRT_COMMON $PRT_REAL
  eval_real "$model_path"
}

# Diffuse/weak-reflective targets.
run_synth /data/zmh/Projects/data/nerf_synthetic/chair NerfSynthetic chair "$PRT_DIFFUSE" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/materials NerfSynthetic materials "$PRT_DIFFUSE" ""
run_synth /data/zmh/Projects/data/nerf_synthetic/ship NerfSynthetic ship "$PRT_DIFFUSE" "--lambda_normal_render_depth 0.05"

# Real scenes.
run_real /data/zmh/Projects/data/ref_real/sedan sedan 8 ""
run_real /data/zmh/Projects/data/ref_real/toycar toycar 4 ""

# Reflective controls.
run_synth /data/zmh/Projects/data/blender/bell_blender GlossySynthetic bell "$PRT_REFLECT" ""
run_synth /data/zmh/Projects/data/ShinyBlender/toaster ShinyBlender toaster "$PRT_REFLECT" ""
run_synth /data/zmh/Projects/data/blender/teapot_blender GlossySynthetic teapot "$PRT_REFLECT" ""

python data_collect.py "$OUT_DIR"
python eval_best_from_curve.py "$OUT_DIR" --select-splits val,val_fast --selection-mode curve --summary-suffix val --max-iteration "$STEP_ITERS" --eval-top-k 1 --save-images
