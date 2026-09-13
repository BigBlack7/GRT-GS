# GRT-GS: Spatial Probe-Guided Radiance Transfer with Gaussian Splatting

本仓库为 GRT-GS 的论文配套研究代码，用于反射场景的新视角合成与逆渲染。

## 方法概述

GRT-GS 在表面对齐的二维 Gaussian 表示和多阶段延迟渲染框架上，通过空间辐射探针与 Gaussian 方向传输共同建模间接入射光：

- **空间辐射探针**：使用规则网格上的球谐系数表示空间连续的入射辐射。
- **Gaussian Radiance Transfer（GRT）**：结合 Gaussian 级球谐传输系数与探针辐射，支持基于 TSDF/BVH 的可见性初始化。
- **Phase-Consistent Continuation（PCC）**：在训练阶段切换时对材质属性进行软连续更新，缓解硬重置带来的优化突变。

## 训练与评估

所有命令均在仓库根目录执行。数据读取支持 Blender 风格数据和 COLMAP 场景；数据路径与输出路径由 `-s`、`-m` 指定。

以下为启用 GRT 与 PCC 的合成场景训练示例，具体配置需按数据集调整，并以最终发布的实验协议为准：

```bash
python train.py -s <DATASET_SCENE> -m <OUTPUT_DIR> \
  --eval --white_background --iterations 30000 \
  --use_pcc --pcc_keep_ratio 0.65 \
  --use_grt --use_grt_visibility_init \
  --grt_mode dot --grt_tau 0.5 --grt_from_iter 20000 \
  --grt_ramp_iters 4000 --grt_sh_degree 2 --probe_grid_res 8
```

评估保存的迭代并导出图像：

```bash
python eval.py -m <OUTPUT_DIR> --iteration 30000 --white_background --save_images
```

`--white_background` 适用于使用白背景的合成数据，真实场景应按数据集协议设置。完整参数定义见 `arguments/__init__.py`，也可在安装依赖后运行 `python train.py --help` 和 `python eval.py --help`。
