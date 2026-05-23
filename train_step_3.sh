#!/usr/bin/env bash
set -e

# Step 3: CGI interreflection gate.
# NCIF is disabled. This isolates whether confidence-gated indirect lighting
# improves reflective scenes without polluting weakly reflective regions.

OUT_ROOT=${OUT_ROOT:-/data2/zmh/output_physnorm_steps}

eval_synth() {
  python eval.py -m "$1" --iteration 30000 --white_background --save_images
}

SYNTH_EVAL="--eval --white_background"
CHECK_30K="--iterations 30000 --test_iterations 18000 20000 22000 30000 --save_iterations 18000 20000 22000 30000"
CGI_ONLY="--use_pcc --pcc_keep_ratio 0.75 --use_r2if --use_cgi --no_use_ncif --lambda_env_tv 0.0001 --lambda_env_energy 0.0001 --cgi_ramp_iters 5000"

python train.py \
  -s /data/zmh/Projects/data/blender/bell_blender \
  -m "$OUT_ROOT/step3_cgi/GlossySynthetic/bell" \
  $SYNTH_EVAL $CHECK_30K $CGI_ONLY
eval_synth "$OUT_ROOT/step3_cgi/GlossySynthetic/bell"

python train.py \
  -s /data/zmh/Projects/data/blender/tbell_blender \
  -m "$OUT_ROOT/step3_cgi/GlossySynthetic/tbell" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_smooth 1.0 $CGI_ONLY
eval_synth "$OUT_ROOT/step3_cgi/GlossySynthetic/tbell"

python train.py \
  -s /data/zmh/Projects/data/blender/teapot_blender \
  -m "$OUT_ROOT/step3_cgi/GlossySynthetic/teapot" \
  $SYNTH_EVAL $CHECK_30K $CGI_ONLY
eval_synth "$OUT_ROOT/step3_cgi/GlossySynthetic/teapot"

python train.py \
  -s /data/zmh/Projects/data/ShinyBlender/toaster \
  -m "$OUT_ROOT/step3_cgi/ShinyBlender/toaster" \
  $SYNTH_EVAL $CHECK_30K $CGI_ONLY
eval_synth "$OUT_ROOT/step3_cgi/ShinyBlender/toaster"
