#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 12: full GRT-GS key-scene parameter test.
#
# Fixed baseline:
#   PCC_KEEP=0.65, no NCIF/R2IF/CGI/OAF, no old probe/prt diffuse residual.
#
# GRT parameters to sweep:
#   GRT_MODE=dot|directional|hybrid
#   GRT_TAU=0.25|0.5|1.0
#   PROBE_RES=4|8
#   GRT_VIS_RAYS=32|64|96
#   GRT_TRANSFER_REFRESH=0|2000
#
# Examples:
#   bash train_step_12_grt.sh
#   GRT_MODE=hybrid GRT_TAU=0.5 bash train_step_12_grt.sh
#   PROBE_RES=4 GRT_TAU=0.25 STEP_NAME=step12_grt_res4_tau025 bash train_step_12_grt.sh
#   GRT_VIS_RAYS=96 GRT_TRANSFER_REFRESH=2000 STEP_NAME=step12_grtfull_refresh bash train_step_12_grt.sh
#   RUN_TRAIN=0 bash train_step_12_grt.sh

DATA_ROOT=${DATA_ROOT:-/data/zmh/Projects/data}
OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
SYNTH_ITERS=${SYNTH_ITERS:-30000}
REAL_ITERS=${REAL_ITERS:-20000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
GRT_MODE=${GRT_MODE:-dot}
GRT_TAU=${GRT_TAU:-0.5}
GRT_TAU_TAG=${GRT_TAU/./}
PROBE_RES=${PROBE_RES:-8}
GRT_SH=${GRT_SH:-2}
GRT_FROM_SYNTH=${GRT_FROM_SYNTH:-20000}
GRT_FROM_REAL=${GRT_FROM_REAL:-10000}
GRT_RAMP=${GRT_RAMP:-4000}
PROBE_LR=${PROBE_LR:-0.001}
GRT_TRANSFER_LR=${GRT_TRANSFER_LR:-0.001}
GRT_VIS_RAYS=${GRT_VIS_RAYS:-64}
GRT_VIS_CHUNK=${GRT_VIS_CHUNK:-4096}
GRT_VIS_EPS=${GRT_VIS_EPS:-0.02}
GRT_VIS_MAX_DISTANCE=${GRT_VIS_MAX_DISTANCE:-0}
GRT_TRANSFER_REFRESH=${GRT_TRANSFER_REFRESH:-0}
GRT_TRANSFER_BLEND=${GRT_TRANSFER_BLEND:-1.0}
GRT_TRANSFER_CLAMP=${GRT_TRANSFER_CLAMP:-4.0}
LAMBDA_GRT_SMOOTH=${LAMBDA_GRT_SMOOTH:-0.0005}
LAMBDA_GRT_ENERGY=${LAMBDA_GRT_ENERGY:-0.0001}
LAMBDA_GRT_TRANSFER=${LAMBDA_GRT_TRANSFER:-0.0001}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_COLLECT=${RUN_COLLECT:-1}
RUN_BEST=${RUN_BEST:-1}
EVAL_TOP_K=${EVAL_TOP_K:-3}
STEP_NAME=${STEP_NAME:-step12_grtfull_${SYNTH_ITERS}_pcc${PCC_TAG}_${GRT_MODE}_tau${GRT_TAU_TAG}_res${PROBE_RES}_rays${GRT_VIS_RAYS}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

echo "[Step12 GRT-GS]"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  OUT_DIR=$OUT_DIR"
echo "  GRT_MODE=$GRT_MODE"
echo "  GRT_TAU=$GRT_TAU"
echo "  PROBE_RES=$PROBE_RES"
echo "  GRT_VIS_RAYS=$GRT_VIS_RAYS"
echo "  GRT_TRANSFER_REFRESH=$GRT_TRANSFER_REFRESH"

make_iters() {
  local start="$1"
  local end="$2"
  local step="$3"
  local out=""
  local i
  for ((i=start; i<=end; i+=step)); do
    out="$out $i"
  done
  echo "$out"
}

SYNTH_CHECK="$(make_iters 18000 "$SYNTH_ITERS" 1000)"
REAL_CHECK="$(make_iters 8000 "$REAL_ITERS" 1000)"
CHECK_SYNTH="--iterations $SYNTH_ITERS --test_iterations $SYNTH_CHECK --save_iterations $SYNTH_CHECK"
CHECK_REAL="--iterations $REAL_ITERS --test_iterations $REAL_CHECK --save_iterations $REAL_CHECK"
PCC_ONLY="--use_pcc --pcc_keep_ratio $PCC_KEEP"
BASE_OFF="--no_use_ncif --no_use_r2if --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"

grt_args() {
  local grt_from="$1"
  echo "--use_grt --use_grt_visibility_init --grt_mode $GRT_MODE --grt_tau $GRT_TAU --grt_from_iter $grt_from --grt_ramp_iters $GRT_RAMP --grt_sh_degree $GRT_SH --probe_grid_res $PROBE_RES --probe_lr $PROBE_LR --grt_transfer_lr $GRT_TRANSFER_LR --grt_visibility_rays $GRT_VIS_RAYS --grt_visibility_chunk $GRT_VIS_CHUNK --grt_visibility_eps $GRT_VIS_EPS --grt_visibility_max_distance $GRT_VIS_MAX_DISTANCE --grt_transfer_refresh_interval $GRT_TRANSFER_REFRESH --grt_transfer_blend $GRT_TRANSFER_BLEND --grt_transfer_clamp $GRT_TRANSFER_CLAMP --lambda_grt_smooth $LAMBDA_GRT_SMOOTH --lambda_grt_energy $LAMBDA_GRT_ENERGY --lambda_grt_transfer $LAMBDA_GRT_TRANSFER"
}

eval_synth() {
  python eval.py -m "$1" --iteration "$SYNTH_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration "$REAL_ITERS" --save_images
}

run_synth() {
  local source_path="$1"
  local dataset="$2"
  local scene="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/$dataset/$scene"

  if [ "$RUN_TRAIN" = "1" ]; then
    python train.py \
      -s "$source_path" \
      -m "$model_path" \
      --eval --white_background $CHECK_SYNTH $extra_args $PCC_ONLY $BASE_OFF $(grt_args "$GRT_FROM_SYNTH")
    eval_synth "$model_path"
  else
    echo "[skip train] $dataset/$scene"
  fi
}

run_real() {
  local source_path="$1"
  local scene="$2"
  local resolution="$3"
  local extra_args="$4"
  local model_path="$OUT_DIR/RefReal/$scene"

  if [ "$RUN_TRAIN" = "1" ]; then
    python train.py \
      -s "$source_path" \
      -m "$model_path" \
      --eval $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
      --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $PCC_ONLY $BASE_OFF $(grt_args "$GRT_FROM_REAL")
    eval_real "$model_path"
  else
    echo "[skip train] RefReal/$scene"
  fi
}

# Reflective controls.
run_synth "$DATA_ROOT/blender/bell_blender" GlossySynthetic bell ""
run_synth "$DATA_ROOT/blender/teapot_blender" GlossySynthetic teapot ""
run_synth "$DATA_ROOT/ShinyBlender/toaster" ShinyBlender toaster ""

# Diffuse / weak-reflective controls.
run_synth "$DATA_ROOT/nerf_synthetic/chair" NerfSynthetic chair ""
run_synth "$DATA_ROOT/nerf_synthetic/materials" NerfSynthetic materials ""
run_synth "$DATA_ROOT/nerf_synthetic/ship" NerfSynthetic ship ""

# Real scenes.
run_real "$DATA_ROOT/ref_real/gardenspheres" gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real "$DATA_ROOT/ref_real/sedan" sedan 8 ""
run_real "$DATA_ROOT/ref_real/toycar" toycar 4 ""

if [ ! -d "$OUT_DIR" ]; then
  echo "[stop] output directory not found: $OUT_DIR"
  exit 1
fi

if [ "$RUN_COLLECT" = "1" ]; then
  python data_collect.py "$OUT_DIR"
fi

if [ "$RUN_BEST" = "1" ]; then
  python eval_best_from_curve.py "$OUT_DIR" \
    --select-splits test_fast,test \
    --selection-mode offline \
    --summary-suffix grt_oracle \
    --eval-top-k "$EVAL_TOP_K" \
    --save-images
fi
