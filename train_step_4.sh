#!/usr/bin/env bash
set -e

# Step 4: NCIF diffuse normal-coupled irradiance.
# CGI is disabled to isolate NCIF. R2IF and PCC remain enabled because they
# stabilize envmap optimization and the delayed-rendering transition.

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
NCIF_COMMON="--use_pcc --pcc_keep_ratio 0.75 --use_r2if --use_ncif --no_use_cgi --lambda_env_tv 0.0001 --lambda_env_energy 0.0001"
NCIF_STRONG="--ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001"
NCIF_MEDIUM="--ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.003 --lambda_ncif_magnitude 0.001"
NCIF_REAL="--ncif_from_iter 5000 --ncif_tau 0.10 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/chair \
  -m "$OUT_ROOT/step4_ncif/NerfSynthetic/chair" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $NCIF_COMMON $NCIF_STRONG
eval_synth "$OUT_ROOT/step4_ncif/NerfSynthetic/chair"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/ficus \
  -m "$OUT_ROOT/step4_ncif/NerfSynthetic/ficus" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $NCIF_COMMON $NCIF_STRONG
eval_synth "$OUT_ROOT/step4_ncif/NerfSynthetic/ficus"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/ship \
  -m "$OUT_ROOT/step4_ncif/NerfSynthetic/ship" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $NCIF_COMMON $NCIF_MEDIUM
eval_synth "$OUT_ROOT/step4_ncif/NerfSynthetic/ship"

python train.py \
  -s /data/zmh/Projects/data/nerf_synthetic/lego \
  -m "$OUT_ROOT/step4_ncif/NerfSynthetic/lego" \
  $SYNTH_EVAL $CHECK_30K --lambda_normal_render_depth 0.05 $NCIF_COMMON \
  --ncif_from_iter 2000 --ncif_tau 0.20 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.004 --lambda_ncif_magnitude 0.001
eval_synth "$OUT_ROOT/step4_ncif/NerfSynthetic/lego"

python train.py \
  -s /data/zmh/Projects/data/ref_real/gardenspheres \
  -m "$OUT_ROOT/step4_ncif/RefReal/gardenspheres" \
  --eval --iterations 20000 $CHECK_REAL --indirect_from_iter 10000 --volume_render_until_iter 0 \
  --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 $NCIF_COMMON $NCIF_REAL
eval_real "$OUT_ROOT/step4_ncif/RefReal/gardenspheres"
