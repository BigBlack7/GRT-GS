# PhysNorm-GS

PhysNorm-GS 是一个面向混合反射场景的 Gaussian inverse rendering 工程。当前实现以干净的 deferred PBR / 2D Gaussian 表面渲染链路为主体，并加入四个稳定性与泛化模块：

- **R2IF**：可靠反射引导的照明因子分解。低反射或高粗糙区域会减弱高频环境贴图监督，避免 envmap 退化成彩色噪声。
- **NCIF**：法向耦合辐照场。每个 Gaussian 维护一个零初始化的局部辐照方向残差，以有界调制方式补充漫反射区域的法向梯度。
- **PCC**：阶段一致连续优化。默认使用软重置缓解 delayed rendering 阶段切换时的硬重置突降。
- **CGI**：置信度门控互反射。mesh-based indirect 不再作为无差别强反馈，而是随反射/粗糙度可靠性渐进启用。

## 安装

```bash
conda create -n physnorm-gs python=3.8
conda activate physnorm-gs

pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
pip install submodules/cubemapencoder
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
pip install submodules/raytracing
pip install -r requirements.txt
```

## 基础训练

```bash
python train.py -s <DATASET> -m <OUT> --eval --white_background
```

关闭本文新增模块，验证主体渲染链路：

```bash
python train.py -s <DATASET> -m <OUT>/baseline --eval --white_background --no_use_ncif --no_use_r2if --no_use_pcc --no_use_cgi
```

启用完整 PhysNorm-GS：

```bash
python train.py -s <DATASET> -m <OUT>/physnorm --eval --white_background --ncif_from_iter 3000 --ncif_tau 0.15 --ncif_ramp_iters 5000 --lambda_ncif_smooth 0.005 --lambda_ncif_magnitude 0.001
```

## 关键参数

- `--use_ncif` / `--no_use_ncif`：是否启用 NCIF。
- `--ncif_from_iter`：NCIF 开始参与渲染和正则的迭代数。
- `--ncif_tau`：NCIF 对 diffuse 的最大有界调制幅度。
- `--ncif_ramp_iters`：NCIF 从 0 ramp 到 `ncif_tau` 的迭代长度。
- `--ncif_lr`：per-Gaussian NCIF 向量学习率。
- `--lambda_ncif_smooth`：NCIF map 的边缘感知平滑权重。
- `--lambda_ncif_magnitude`：NCIF 幅值正则。
- `--use_r2if` / `--no_use_r2if`：是否启用反射可靠性 specular gate。
- `--r2if_specular_alpha`、`--r2if_specular_beta`：反射强度和粗糙度在 R2IF gate 中的指数。
- `--r2if_min_specular_gate`：高频环境贴图 specular gate 的下界。
- `--use_pcc` / `--no_use_pcc`：是否使用阶段切换软重置。
- `--pcc_keep_ratio`：阶段切换时保留旧材质的比例。
- `--use_cgi` / `--no_use_cgi`：是否启用置信度门控互反射。
- `--cgi_ramp_iters`：indirect gate 的渐进启用长度。
- `--lambda_env_tv`：环境贴图 TV 正则。
- `--lambda_env_energy`：两个训练阶段环境贴图之间的能量一致性正则。

## 阶段参数

- `--volume_render_until_iter`：volume rendering 阶段结束迭代，默认 `18000`。
- `--normal_prop_until_iter`：材质感知法线传播截止迭代，默认 `25000`。
- `--indirect_from_iter`：mesh-based inter-reflection 启用迭代。几何不稳时建议延后到 `30000` 或更晚。
- `--iterations`：总训练迭代数，默认 `50000`。

## 数据集参数

- Blender / NeRF Synthetic / Glossy Synthetic：通常使用 `--eval --white_background`，数据划分来自 `transforms_train.json` 和 `transforms_test.json`。
- Ref-Real：通常使用官方短训初始化协议，不强制 `--white_background`。
- COLMAP 数据：仍保留 `--lod` 与 `--llffhold` 作为数据划分参数；它们不是 NCIF/R2IF 方法参数。

## 评估

```bash
python eval.py -m <OUT>/physnorm --white_background --save_images
```

强制使用已保存 mesh indirect：

```bash
python eval.py -m <OUT>/physnorm --white_background --save_images --force_indirect
```

关闭 indirect 评估：

```bash
python eval.py -m <OUT>/physnorm --white_background --save_images --no_indirect
```

## 批量训练

当前保留的训练入口如下：

- `train.sh`：原始批量训练命令集合，主要用于和早期实验对照。
- `train_step_0.sh`：Ref-Gaussian baseline 全场景诊断。
- `train_step_8_pcc_val.sh`：固定 `PCC=0.65` 强基准，使用 validation top-1 选点。
- `train_step_5.sh`：GIP-1 grid probe 消融入口。当前实验结论是不进入默认主线，仅用于复现和对照。
- `train_step_9_prt.sh`：PRT-GS / Gaussian Radiance Transfer 首轮关键场景实验。

运行示例：

```bash
sh train.sh
bash train_step_8_pcc_val.sh
bash train_step_9_prt.sh
```

路径需要根据本机数据集位置调整。旧的 sweep/fix/debug step 脚本已经清理，相关实验结论记录在 `Experiment.md`。

## 代码清理状态

- 已移除 anchor-based MLP 主路径。
- 已移除 IIV 分支。
- 旧 diffuse residual 训练参数已统一替换为 `ncif_*`。
- 旧 indirect 评估开关已替换为更清晰的 `force_indirect/no_indirect`。
- `lod/llffhold` 仅保留为 COLMAP 数据划分参数，不用于 synthetic / Ref-Real 训练命令。
