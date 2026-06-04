#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e

# Step 11: fixed-PCC full-scene baseline.
#
# Goal:
#   Build the strongest fixed-PCC internal baseline before adding any new
#   rendering/module innovation. This script disables every non-PCC module and
#   evaluates dense saved checkpoints so each scene can get its own best
#   iteration point.
#
# Default protocol:
#   - Synthetic/Shiny/Glossy scenes: train to 30k, dense checkpoints from 18k.
#   - Ref-Real scenes: official 20k schedule.
#   - No validation holdout by default, so training uses the official train set.
#   - Best-point summary uses test/test_fast as an internal oracle baseline.
#
# Useful reruns:
#   RUN_TRAIN=0 bash train_step_11_pcc_full.sh
#   SYNTH_ITERS=50000 bash train_step_11_pcc_full.sh
#   SYNTH_SAVE_STEP=2000 bash train_step_11_pcc_full.sh
#   PCC_KEEP=0.50 STEP_NAME=my_pcc050 bash train_step_11_pcc_full.sh

DATA_ROOT=${DATA_ROOT:-/data/zmh/Projects/data}
OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}
SYNTH_ITERS=${SYNTH_ITERS:-30000}
REAL_ITERS=${REAL_ITERS:-20000}
SYNTH_SAVE_STEP=${SYNTH_SAVE_STEP:-1000}
REAL_SAVE_STEP=${REAL_SAVE_STEP:-1000}
PCC_KEEP=${PCC_KEEP:-0.65}
PCC_TAG=${PCC_KEEP/./}
VAL_HOLD=${VAL_HOLD:-0}
VAL_OFFSET=${VAL_OFFSET:-0}
VAL_MAX=${VAL_MAX:-8}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_COLLECT=${RUN_COLLECT:-1}
RUN_BEST=${RUN_BEST:-1}
EVAL_TOP_K=${EVAL_TOP_K:-3}
STEP_NAME=${STEP_NAME:-step11_pcc_full_${SYNTH_ITERS}_pcc${PCC_TAG}}
OUT_DIR="$OUT_ROOT/$STEP_NAME"

echo "[Step11 fixed PCC full baseline]"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  OUT_DIR=$OUT_DIR"
echo "  SYNTH_ITERS=$SYNTH_ITERS"
echo "  REAL_ITERS=$REAL_ITERS"
echo "  SYNTH_SAVE_STEP=$SYNTH_SAVE_STEP"
echo "  REAL_SAVE_STEP=$REAL_SAVE_STEP"
echo "  PCC_KEEP=$PCC_KEEP"
echo "  VAL_HOLD=$VAL_HOLD"

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

SYNTH_CHECK="$(make_iters 18000 "$SYNTH_ITERS" "$SYNTH_SAVE_STEP")"
REAL_CHECK="$(make_iters 8000 "$REAL_ITERS" "$REAL_SAVE_STEP")"

CHECK_SYNTH="--iterations $SYNTH_ITERS --test_iterations $SYNTH_CHECK --save_iterations $SYNTH_CHECK"
CHECK_REAL="--iterations $REAL_ITERS --test_iterations $REAL_CHECK --save_iterations $REAL_CHECK"
SYNTH_EVAL="--eval --white_background"
PCC_ONLY="--use_pcc --pcc_keep_ratio $PCC_KEEP --no_use_ncif --no_use_r2if --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"
VAL_ARGS=""
if [ "$VAL_HOLD" -gt 0 ]; then
  VAL_ARGS="--val_hold $VAL_HOLD --val_offset $VAL_OFFSET --val_max $VAL_MAX"
fi

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
      $SYNTH_EVAL $VAL_ARGS $CHECK_SYNTH $extra_args $PCC_ONLY
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
      --eval $VAL_ARGS $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
      --initial 1 --init_until_iter 3000 -r "$resolution" $extra_args $PCC_ONLY
    eval_real "$model_path"
  else
    echo "[skip train] RefReal/$scene"
  fi
}

# ==============================
# Shiny Blender / Ref-NeRF style
# ==============================

run_synth "$DATA_ROOT/ShinyBlender/ball" ShinyBlender ball "--lambda_normal_smooth 1.0"
run_synth "$DATA_ROOT/ShinyBlender/car" ShinyBlender car ""
run_synth "$DATA_ROOT/ShinyBlender/coffee" ShinyBlender coffee ""
run_synth "$DATA_ROOT/ShinyBlender/helmet" ShinyBlender helmet "--lambda_normal_smooth 1.0"
run_synth "$DATA_ROOT/ShinyBlender/teapot" ShinyBlender teapot ""
run_synth "$DATA_ROOT/ShinyBlender/toaster" ShinyBlender toaster ""

# ==============================
# Glossy Synthetic
# ==============================

run_synth "$DATA_ROOT/blender/angel_blender" GlossySynthetic angel ""
run_synth "$DATA_ROOT/blender/bell_blender" GlossySynthetic bell ""
run_synth "$DATA_ROOT/blender/cat_blender" GlossySynthetic cat ""
run_synth "$DATA_ROOT/blender/horse_blender" GlossySynthetic horse ""
run_synth "$DATA_ROOT/blender/luyu_blender" GlossySynthetic luyu ""
run_synth "$DATA_ROOT/blender/potion_blender" GlossySynthetic potion ""
run_synth "$DATA_ROOT/blender/tbell_blender" GlossySynthetic tbell "--lambda_normal_smooth 1.0"
run_synth "$DATA_ROOT/blender/teapot_blender" GlossySynthetic teapot ""

# ==============================
# NeRF Synthetic
# ==============================

run_synth "$DATA_ROOT/nerf_synthetic/chair" NerfSynthetic chair ""
run_synth "$DATA_ROOT/nerf_synthetic/drums" NerfSynthetic drums ""
run_synth "$DATA_ROOT/nerf_synthetic/ficus" NerfSynthetic ficus ""
run_synth "$DATA_ROOT/nerf_synthetic/hotdog" NerfSynthetic hotdog ""
run_synth "$DATA_ROOT/nerf_synthetic/lego" NerfSynthetic lego ""
run_synth "$DATA_ROOT/nerf_synthetic/materials" NerfSynthetic materials ""
run_synth "$DATA_ROOT/nerf_synthetic/mic" NerfSynthetic mic ""
run_synth "$DATA_ROOT/nerf_synthetic/ship" NerfSynthetic ship ""

# ==============================
# Ref-Real
# ==============================

run_real "$DATA_ROOT/ref_real/gardenspheres" gardenspheres 4 "--lambda_normal_smooth 0.45"
run_real "$DATA_ROOT/ref_real/toycar" toycar 4 ""
run_real "$DATA_ROOT/ref_real/sedan" sedan 8 ""

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
    --summary-suffix pcc_oracle \
    --eval-top-k "$EVAL_TOP_K" \
    --save-images

  if [ "$VAL_HOLD" -gt 0 ]; then
    python eval_best_from_curve.py "$OUT_DIR" \
      --select-splits val_fast,val \
      --selection-mode curve \
      --summary-suffix pcc_val \
      --eval-top-k 1 \
      --save-images
  fi
fi
