#!/usr/bin/env bash
set -e

# Step 1: PCC soft-reset only.
# R2IF, NCIF, CGI and envmap regularization are disabled. This isolates whether
# soft material continuation alleviates the 18k-22k / delayed-rendering drop.

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}

eval_synth() {
  python eval.py -m "$1" --iteration 30000 --white_background --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_30K="--iterations 30000 --test_iterations 18000 20000 22000 30000 --save_iterations 18000 20000 22000 30000"
PCC_ONLY="--use_pcc --no_use_ncif --no_use_r2if --no_use_cgi --lambda_env_tv 0 --lambda_env_energy 0"

# Bell is the ratio sweep scene. If only one run is needed, keep the 0.75 run.
python train.py \
  -s /data/zmh/Projects/data/blender/bell_blender \
  -m "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep050" \
  $SYNTH_EVAL $CHECK_30K $PCC_ONLY --pcc_keep_ratio 0.50
eval_synth "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep050"

python train.py \
  -s /data/zmh/Projects/data/blender/bell_blender \
  -m "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep075" \
  $SYNTH_EVAL $CHECK_30K $PCC_ONLY --pcc_keep_ratio 0.75
eval_synth "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep075"

python train.py \
  -s /data/zmh/Projects/data/blender/bell_blender \
  -m "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep090" \
  $SYNTH_EVAL $CHECK_30K $PCC_ONLY --pcc_keep_ratio 0.90
eval_synth "$OUT_ROOT/step1_pcc/GlossySynthetic/bell_keep090"

python train.py \
  -s /data/zmh/Projects/data/ShinyBlender/toaster \
  -m "$OUT_ROOT/step1_pcc/ShinyBlender/toaster_keep075" \
  $SYNTH_EVAL $CHECK_30K $PCC_ONLY --pcc_keep_ratio 0.75
eval_synth "$OUT_ROOT/step1_pcc/ShinyBlender/toaster_keep075"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/lego \
  -m "$OUT_ROOT/step1_pcc/NerfSynthetic/lego_keep075" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $PCC_ONLY --pcc_keep_ratio 0.75
eval_synth "$OUT_ROOT/step1_pcc/NerfSynthetic/lego_keep075"
