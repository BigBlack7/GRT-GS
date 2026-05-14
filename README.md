# PhysNorm-GS

PhysNorm-GS builds on Ref-Gaussian and adds a deferred, material-aware IDIV diffuse branch inspired by Normal-GS.

## Unified Training

All datasets now use the same entrypoint:

```bash
python train.py -s <DATASET_PATH> -m <OUTPUT_PATH> --eval --gpu 0
```

Ref-Gaussian style synthetic or real reflective scenes:

```bash
python train.py -s data/ref_nerf/coffee -m output/coffee/physnorm --eval --white_background --gpu 0
python eval.py -m output/coffee/physnorm --white_background --save_images --gpu 0
```

Mip-NeRF 360 / Tanks and Temples / Deep Blending style COLMAP scenes:

```bash
python train.py -s data/mipnerf360/bonsai -m output/bonsai/physnorm --eval --lod 0 --llffhold 8 --gpu 0
python eval.py -m output/bonsai/physnorm --save_images --gpu 0
```

Blender/NeRF synthetic style scenes:

```bash
python train.py -s data/nerf_synthetic/lego -m output/lego/physnorm --eval --white_background --gpu 0
python eval.py -m output/lego/physnorm --white_background --save_images --gpu 0
```

## Important Parameters

Core method switches:

- `--use_idiv` / `--no_use_idiv`: enable or disable the deferred IDIV branch.
- `--idiv_from_iter`: first iteration where IDIV contributes to rendering and smoothing loss. Default: `3000`.
- `--lambda_idiv`: edge-aware IDIV smoothness weight. Suggested range: `0.001` to `0.01`.
- `--init_metalness_value`: initial metallic value for Gaussian material gating. Default: `0.05`.
- `--metal_msk_thr`: metalness threshold used by material-aware normal propagation. Default: `0.5`.

Geometry and normal:

- `--lambda_normal_render_depth`: depth-normal consistency. Start with `0.05`.
- `--lambda_normal_smooth`: image-edge-aware normal smoothness. Start with `0.2` to `1.0` on noisy geometry.
- `--normal_prop_until_iter`: normal propagation/densification phase length.

Training schedule:

- `--iterations`: total iterations.
- `--init_until_iter`: optional initial RGB/geometry stage.
- `--volume_render_until_iter`: volume/deferred transition point.
- `--indirect_from_iter`: enables mesh-based inter-reflection after enough geometry is formed.

Dataset split:

- `--eval`: keeps a test split.
- `--lod`: Normal-GS compatible holdout behavior; `0` means LLFF-style holdout.
- `--llffhold`: test every N-th COLMAP image when `--lod 0`. Default: `8`.

## Gradual Tuning Guide

1. Baseline check:

```bash
python train.py -s <DATASET> -m <OUT>/baseline --eval --no_use_idiv --gpu 0
```

This verifies the Ref-Gaussian path on the same codebase.

2. Conservative PhysNorm-GS:

```bash
python train.py -s <DATASET> -m <OUT>/idiv_calm --eval --idiv_from_iter 5000 --lambda_idiv 0.002 --init_metalness_value 0.05 --gpu 0
```

Use this when early geometry is unstable.

3. Stronger diffuse-normal coupling:

```bash
python train.py -s <DATASET> -m <OUT>/idiv_strong --eval --idiv_from_iter 2000 --lambda_idiv 0.005 --lambda_normal_render_depth 0.05 --gpu 0
```

Use this for mostly diffuse or mixed-material scenes.

4. Reflective scenes:

```bash
python train.py -s <DATASET> -m <OUT>/reflective --eval --idiv_from_iter 5000 --lambda_idiv 0.001 --indirect_from_iter 20000 --gpu 0
```

Keep IDIV smoothness small so high-metal regions remain dominated by the PBR specular path.

Evaluation writes `metric.txt` and `results.json` with PSNR, SSIM, LPIPS, FPS, and Normal MAE when GT normals are available under `normal/*_normal.png` or `normals/*.png`.

==================================================反射=====================================================

python train.py -s /data/zmh/Projects/data/blender/angel_blender -m /data2/zmh/output/GlossySynthetic/angel_idiv --eval --idiv_from_iter 8000 --lambda_idiv 0.0005 --indirect_from_iter 20000 --white_background

python eval.py --white_background --save_images --model_path /data2/zmh/output/GlossySynthetic/angel_idiv --eval_indirect

==================================================混合=====================================================
python train.py -s <DATASET> -m <OUT>/idiv_strong --eval --idiv_from_iter 2000 --lambda_idiv 0.005 --lambda_normal_render_depth 0.05 --white_background