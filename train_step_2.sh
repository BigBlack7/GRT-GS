#!/usr/bin/env bash
set -e

# Step 2: R2IF + envmap regularization.
# NCIF and CGI are disabled. This tests whether reliable specular gating and
# envmap regularization reduce noisy environment maps in diffuse/real scenes.

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}

eval_synth() {
  python eval.py -m "$1" --iteration 30000 --white_background --save_images
}

eval_real() {
  python eval.py -m "$1" --iteration 20000 --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_30K="--iterations 30000 --test_iterations 18000 20000 22000 30000 --save_iterations 18000 20000 22000 30000"
CHECK_REAL="--test_iterations 10000 15000 18000 20000 --save_iterations 10000 15000 18000 20000"
R2IF_ONLY="--use_pcc --pcc_keep_ratio 0.75 --use_r2if --no_use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --r2if_specular_alpha 1.0 --r2if_specular_beta 1.0 --r2if_min_specular_gate 0.02"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/chair \
  -m "$OUT_ROOT/step2_r2if_env/NerfSynthetic/chair" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $R2IF_ONLY
eval_synth "$OUT_ROOT/step2_r2if_env/NerfSynthetic/chair"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/ship \
  -m "$OUT_ROOT/step2_r2if_env/NerfSynthetic/ship" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $R2IF_ONLY
eval_synth "$OUT_ROOT/step2_r2if_env/NerfSynthetic/ship"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/lego \
  -m "$OUT_ROOT/step2_r2if_env/NerfSynthetic/lego" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $R2IF_ONLY
eval_synth "$OUT_ROOT/step2_r2if_env/NerfSynthetic/lego"

python train.py \
  -s /data/zmh/Projects/data/ref_real/gardenspheres \
  -m "$OUT_ROOT/step2_r2if_env/RefReal/gardenspheres" \
  --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
  --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 $R2IF_ONLY
eval_real "$OUT_ROOT/step2_r2if_env/RefReal/gardenspheres"
