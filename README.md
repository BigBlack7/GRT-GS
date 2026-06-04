# GRT-GS

当前工程主线已经收敛为 **GRT-GS: Gaussian Radiance Transfer for Reflective Gaussian Splatting**。方法由三个部分组成：

- **DDGI Probe Radiance**：用规则探针网格学习空间连续的低频入射辐射 SH 系数。
- **Gaussian PRT Transfer**：每个 Gaussian 维护低阶 SH 传输系数，描述该点如何接收 probe radiance。
- **PCC 稳定训练**：在 delayed rendering 阶段切换时使用固定比例软重置，缓解硬重置导致的指标开倒车。

旧的 NCIF、R2IF、CGI、OAF、旧 probe diffuse residual、旧 PRT diffuse residual 已退出当前主线。代码中仅保留少量历史 checkpoint/PLY 兼容入口，默认不启用，不参与新实验训练。

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

## 训练入口

固定 PCC 全场景基准：

```bash
bash train_step_11_pcc_full.sh
```

GRT-GS 关键场景参数搜索：

```bash
bash train_step_12_grt.sh
GRT_MODE=hybrid GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_MODE=directional GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_MODE=dot GRT_TAU=0.25 bash train_step_12_grt.sh
PROBE_RES=4 GRT_MODE=dot GRT_TAU=0.5 bash train_step_12_grt.sh
```

服务器中断后只重新收集和续跑 best-iteration 评估：

```bash
RUN_TRAIN=0 bash train_step_12_grt.sh
```

## 关键参数

- `--use_pcc`：启用固定 PCC 软重置。
- `--pcc_keep_ratio`：阶段切换时旧材质属性保留比例，当前强基准默认 `0.65`。
- `--use_grt`：启用 GRT 间接光分支。
- `--grt_from_iter`：GRT 开始混入原生 learned indirect 的迭代。
- `--grt_tau`：GRT 与原生 learned indirect 的最大混合权重。
- `--grt_ramp_iters`：GRT 从 0 ramp 到 `grt_tau` 的迭代长度。
- `--grt_mode`：`dot`、`directional` 或 `hybrid`。
- `--grt_sh_degree`：GRT probe 与 transfer 使用的 SH 阶数，当前默认 `2`。
- `--probe_grid_res`：DDGI-style probe grid 分辨率，当前建议先测 `8` 与 `4`。
- `--probe_lr`：probe radiance SH 系数学习率。
- `--grt_transfer_lr`：Gaussian PRT transfer 系数学习率。
- `--use_grt_visibility_init`：启用 TSDF/BVH 半球可见性拟合，用显式几何初始化 Gaussian transfer，完整 GRT 主实验默认开启。
- `--grt_visibility_rays`：每个 Gaussian 用于拟合 transfer 的半球采样射线数，当前默认 `64`。
- `--grt_visibility_chunk`：BVH 初始化时每批处理的 Gaussian 数量，显存不足时可降低。
- `--grt_transfer_refresh_interval`：transfer 初始化刷新间隔；`0` 表示只在首次 mesh/BVH 可用时初始化一次，`2000` 表示随 Ref-Gaussian mesh extraction 周期刷新。
- `--grt_transfer_blend`：BVH 拟合结果写回当前 transfer 的混合比例。
- `--lambda_grt_smooth`：屏幕空间 GRT 间接光平滑正则。
- `--lambda_grt_energy`：probe radiance 能量与空间平滑正则。
- `--lambda_grt_transfer`：Gaussian transfer 高频系数正则。

## 评估和可视化

普通评估：

```bash
python eval.py -m <OUT> --iteration <ITER> --white_background --save_images
```

评估会输出：

- `test/renders/rgb`：最终渲染结果。
- `test/renders/normal`：2DGS 表面法线。
- `test/renders/grt`：GRT 贡献的间接辐射。
- `test/renders/grt_probe`：反射方向 probe radiance 查询结果。
- `test/renders/grt_visibility`：Gaussian transfer 方向可见性/响应图。
- `test/renders/specular_reliability`：材质相关的镜面可靠性诊断图，仅用于观察，不再门控前向渲染。

指标汇总：

```bash
python data_collect.py <OUT_DIR>
python eval_best_from_curve.py <OUT_DIR> --select-splits test_fast,test --selection-mode offline --summary-suffix grt_oracle --eval-top-k 3 --save-images
```

## 数据集参数

- Blender / NeRF Synthetic / Glossy Synthetic：通常使用 `--eval --white_background`。
- Ref-Real：通常使用官方短训初始化协议，不强制 `--white_background`。
- `lod/llffhold` 只保留为 COLMAP/LLFF 数据划分参数，不是 GRT-GS 方法参数。

## 当前脚本

- `train.sh`：早期原始命令集合，用于对照。
- `train_step_0.sh`：Ref-Gaussian baseline 诊断。
- `train_step_11_pcc_full.sh`：固定 PCC 全场景强基准。
- `train_step_12_grt.sh`：完整 GRT-GS 关键场景参数搜索。
- `train_step_13_grt_full.sh`：完整 GRT-GS 全场景评估。

历史 sweep/fix/debug 脚本已经清理或停用，相关结论保存在 `Experiment.md`。
