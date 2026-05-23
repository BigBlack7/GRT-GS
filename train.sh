# PhysNorm-GS training commands.
#
# 调参原则：
# 1. synthetic / blender 场景保留 --eval --white_background。
# 2. 官方自带 --lambda_normal_smooth 1.0 的 ball / helmet / tbell 保留该设置。
# 3. 方法对比主表统一使用 30k；Ref-Real 保持官方 20k。
#    若 baseline 诊断发现某场景明显开倒车，再单独确定 early-stop 轮次。
# 4. 反射强场景使用较弱 NCIF；漫反射/混合场景使用较强 NCIF。
# 5. R2IF/PCC/CGI 默认开启；需要消融时使用 --no_use_r2if / --no_use_pcc / --no_use_cgi。


# ==============================
# Shiny Blender / Ref-NeRF style
# 单个强反射物体；NCIF 只作为弱的漫反射法向辅助。
# ==============================

python train.py -s /data/zmh/Projects/data/ShinyBlender/ball -m /data2/zmh/output/ShinyBlender/ball --eval --white_background --iterations 30000 --lambda_normal_smooth 1.0 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/ShinyBlender/car -m /data2/zmh/output/ShinyBlender/car --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/ShinyBlender/coffee -m /data2/zmh/output/ShinyBlender/coffee --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/ShinyBlender/helmet -m /data2/zmh/output/ShinyBlender/helmet --eval --white_background --iterations 30000 --lambda_normal_smooth 1.0 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/ShinyBlender/teapot -m /data2/zmh/output/ShinyBlender/teapot --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/ShinyBlender/toaster -m /data2/zmh/output/ShinyBlender/toaster --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001


# ==============================
# Glossy Synthetic
# 光泽/反射物体；bell 使用官方阶段设置下的 30k 短训。
# ==============================

python train.py -s /data/zmh/Projects/data/blender/angel_blender -m /data2/zmh/output/GlossySynthetic/angel --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/bell_blender -m /data2/zmh/output/GlossySynthetic/bell --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/cat_blender -m /data2/zmh/output/GlossySynthetic/cat --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/horse_blender -m /data2/zmh/output/GlossySynthetic/horse --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/luyu_blender -m /data2/zmh/output/GlossySynthetic/luyu --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/potion_blender -m /data2/zmh/output/GlossySynthetic/potion --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/tbell_blender -m /data2/zmh/output/GlossySynthetic/tbell --eval --white_background --iterations 30000 --lambda_normal_smooth 1.0 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/blender/teapot_blender -m /data2/zmh/output/GlossySynthetic/teapot --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.08 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001


# ==============================
# NeRF Synthetic
# 漫反射/混合物体；R2IF 会抑制低反射区域对高频 envmap 的污染。
# ==============================

python train.py -s /data/zmh/Projects/data/nerf_synthetic/chair -m /data2/zmh/output/NerfSynthetic/chair --eval --white_background --iterations 30000 --ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05

python train.py -s /data/zmh/Projects/data/nerf_synthetic/drums -m /data2/zmh/output/NerfSynthetic/drums --eval --white_background --iterations 30000 --ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.003 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05

python train.py -s /data/zmh/Projects/data/nerf_synthetic/ficus -m /data2/zmh/output/NerfSynthetic/ficus --eval --white_background --iterations 30000 --ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05

python train.py -s /data/zmh/Projects/data/nerf_synthetic/hotdog -m /data2/zmh/output/NerfSynthetic/hotdog --eval --white_background --iterations 30000 --ncif_from_iter 2000 --ncif_tau 0.25 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05

python train.py -s /data/zmh/Projects/data/nerf_synthetic/lego -m /data2/zmh/output/NerfSynthetic/lego --eval --white_background --iterations 30000 --ncif_from_iter 2000 --ncif_tau 0.20 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.004 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05

python train.py -s /data/zmh/Projects/data/nerf_synthetic/materials -m /data2/zmh/output/NerfSynthetic/materials --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.05 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.001 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/nerf_synthetic/mic -m /data2/zmh/output/NerfSynthetic/mic --eval --white_background --iterations 30000 --ncif_from_iter 8000 --ncif_tau 0.10 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001

python train.py -s /data/zmh/Projects/data/nerf_synthetic/ship -m /data2/zmh/output/NerfSynthetic/ship --eval --white_background --iterations 30000 --ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.003 --lambda_ncif_magnitude 0.001 --lambda_normal_render_depth 0.05


# ==============================
# Ref-Real
# 真实世界场景沿用官方短训/初始化协议；不强制 white_background。
# ==============================

python train.py -s /data/zmh/Projects/data/ref_real/gardenspheres -m /data2/zmh/output/RefReal/gardenspheres --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 --ncif_from_iter 5000 --ncif_tau 0.10 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001
python train.py -s /data/zmh/Projects/data/ref_real/toycar -m /data2/zmh/output/RefReal/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 -r 4 --ncif_from_iter 5000 --ncif_tau 0.08 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001
python train.py -s /data/zmh/Projects/data/ref_real/sedan -m /data2/zmh/output/RefReal/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 -r 8 --ncif_from_iter 5000 --ncif_tau 0.08 --ncif_ramp_iters 4000 --lambda_ncif_smooth 0.002 --lambda_ncif_magnitude 0.001
