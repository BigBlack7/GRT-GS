#!/usr/bin/env bash
set -e

# Step 0 full-scene diagnostic baseline.
# New PhysNorm-GS modules are disabled. Non-real scenes use the official
# Ref-Gaussian 50k schedule by default so that late regressions can be found.
# Ref-Real follows the official 20k schedule.
#
# For a 30k-only baseline pass:
#   BASELINE_ITERS=30000 OUT_ROOT=/data2/zmh/output_physnorm_baseline_30k bash train_step_0.sh

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_baseline_diag}
BASELINE_ITERS=${BASELINE_ITERS:-50000}

eval_synth() {
  python eval.py -m "$1" --iteration "$BASELINE_ITERS" --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_SYNTH="--iterations $BASELINE_ITERS --test_iterations 18000 20000 22000 25000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 --save_iterations 18000 20000 22000 25000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000"
CHECK_REAL="--test_iterations 10000 15000 18000 20000 --save_iterations 10000 15000 18000 20000"
BASE_OFF="--no_use_ncif --no_use_r2if --no_use_pcc --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"


# ==============================
# Shiny Blender / Ref-NeRF style
# ==============================

python train.py -s /data/zmh/Projects/data/ShinyBlender/ball -m "$OUT_ROOT/step0_baseline/ShinyBlender/ball" $SYNTH_EVAL $CHECK_SYNTH --lambda_normal_smooth 1.0 $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/ball"

python train.py -s /data/zmh/Projects/data/ShinyBlender/car -m "$OUT_ROOT/step0_baseline/ShinyBlender/car" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/car"

python train.py -s /data/zmh/Projects/data/ShinyBlender/coffee -m "$OUT_ROOT/step0_baseline/ShinyBlender/coffee" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/coffee"

python train.py -s /data/zmh/Projects/data/ShinyBlender/helmet -m "$OUT_ROOT/step0_baseline/ShinyBlender/helmet" $SYNTH_EVAL $CHECK_SYNTH --lambda_normal_smooth 1.0 $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/helmet"

python train.py -s /data/zmh/Projects/data/ShinyBlender/teapot -m "$OUT_ROOT/step0_baseline/ShinyBlender/teapot" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/teapot"

python train.py -s /data/zmh/Projects/data/ShinyBlender/toaster -m "$OUT_ROOT/step0_baseline/ShinyBlender/toaster" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/ShinyBlender/toaster"


# ==============================
# Glossy Synthetic
# ==============================

python train.py -s /data/zmh/Projects/data/blender/angel_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/angel" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/angel"

python train.py -s /data/zmh/Projects/data/blender/bell_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/bell" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/bell"

python train.py -s /data/zmh/Projects/data/blender/cat_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/cat" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/cat"

python train.py -s /data/zmh/Projects/data/blender/horse_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/horse" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/horse"

python train.py -s /data/zmh/Projects/data/blender/luyu_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/luyu" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/luyu"

python train.py -s /data/zmh/Projects/data/blender/potion_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/potion" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/potion"

python train.py -s /data/zmh/Projects/data/blender/tbell_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/tbell" $SYNTH_EVAL $CHECK_SYNTH --lambda_normal_smooth 1.0 $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/tbell"

python train.py -s /data/zmh/Projects/data/blender/teapot_blender -m "$OUT_ROOT/step0_baseline/GlossySynthetic/teapot" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/GlossySynthetic/teapot"


# ==============================
# NeRF Synthetic
# ==============================

python train.py -s /data/zmh/Projects/data/nerf_synthetic/chair -m "$OUT_ROOT/step0_baseline/NerfSynthetic/chair" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/chair"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/drums -m "$OUT_ROOT/step0_baseline/NerfSynthetic/drums" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/drums"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/ficus -m "$OUT_ROOT/step0_baseline/NerfSynthetic/ficus" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/ficus"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/hotdog -m "$OUT_ROOT/step0_baseline/NerfSynthetic/hotdog" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/hotdog"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/lego -m "$OUT_ROOT/step0_baseline/NerfSynthetic/lego" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/lego"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/materials -m "$OUT_ROOT/step0_baseline/NerfSynthetic/materials" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/materials"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/mic -m "$OUT_ROOT/step0_baseline/NerfSynthetic/mic" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/mic"

python train.py -s /data/zmh/Projects/data/nerf_synthetic/ship -m "$OUT_ROOT/step0_baseline/NerfSynthetic/ship" $SYNTH_EVAL $CHECK_SYNTH $BASE_OFF
eval_synth "$OUT_ROOT/step0_baseline/NerfSynthetic/ship"


# ==============================
# Ref-Real
# ==============================

python train.py -s /data/zmh/Projects/data/ref_real/gardenspheres -m "$OUT_ROOT/step0_baseline/RefReal/gardenspheres" --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 $BASE_OFF
eval_real "$OUT_ROOT/step0_baseline/RefReal/gardenspheres"

python train.py -s /data/zmh/Projects/data/ref_real/toycar -m "$OUT_ROOT/step0_baseline/RefReal/toycar" --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 -r 4 $BASE_OFF
eval_real "$OUT_ROOT/step0_baseline/RefReal/toycar"

python train.py -s /data/zmh/Projects/data/ref_real/sedan -m "$OUT_ROOT/step0_baseline/RefReal/sedan" --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 -r 8 $BASE_OFF
eval_real "$OUT_ROOT/step0_baseline/RefReal/sedan"

python data_collect.py "$OUT_ROOT/step0_baseline"
