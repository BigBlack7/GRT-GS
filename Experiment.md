# OAH-GS 实验记录与阶段计划

本文档只记录工程落地、训练脚本、阶段实验计划和阶段结论。理论方法、公式推导和论文叙事统一放在 `Proposal.md`。

---

## 1. 当前总路线

当前项目从“Ref-Gaussian 主链路 + NCIF/R2IF/PCC/CGI 若干模块”升级为 **OAH-GS：Observability-Aware Hybrid Gaussian Splatting** 主线。新的核心判断是：

1. 高频环境贴图适合表达远场镜面反射，不适合被漫反射区域强监督。
2. 漫反射、弱反射和真实复杂区域应优先保留外观基底，再用低频 irradiance probes 做受控修正。
3. 近场物体反射和互反射应由局部 ray tracing / local transport 解释。
4. 当物理因子不可观测时，训练应安全回退到外观解释，而不是强行污染 envmap 或 local transport。
5. PCC 负责解决 delayed rendering 和材质重置带来的训练开倒车。

因此后续工程不再把 NCIF 视为最终主创新，而把它视为 **GIP-0：per-Gaussian low-frequency irradiance proxy**。若 GIP-0 有效，继续实现真正的 **GIP-1：learnable grid irradiance probes**；若 GIP-0 对真实场景无效，则优先补充 appearance/exposure residual 和更稳的 anti-aliasing/geometry，再决定是否推进 DDGI-lite visibility。

当前实验目标也随之调整：不再要求所有场景都被同一个物理模块显著提升，而是验证 **反射场景有收益、非反射场景不伤害、真实场景更稳定、envmap 更干净**。

从当前阶段开始，实验推进采用两级基准：

1. **外部最低基准。** Ref-Gaussian 30k 作为必须超过或至少不低于的下限，用于证明工程主线没有退化，具体实验数据放在ref_gaussian_baseline.txt中作为参考。
2. **内部强基准。** PCC + best checkpoint 作为后续创新模块的默认比较对象。R2SF、GIP/OAF、probe 等模块只有在叠加到 PCC 后继续带来收益，才视为对当前工程有增量价值。

---

## 2. 已完成实验结论

### 2.1 Step 0：Ref-Gaussian baseline 复现

脚本：

```bash
bash train_step_0.sh
```

结论：

- 当前工程在关闭新模块后，可以复现 Ref-Gaussian 主体结果。
- 50k 诊断发现部分场景存在明显后期退化。
- 严重最终开倒车场景包括 `GlossySynthetic/bell`、`ShinyBlender/toaster`、`NerfSynthetic/mic`、`NerfSynthetic/chair`。
- `ShinyBlender/ball`、`GlossySynthetic/tbell` 存在阶段切换附近的瞬时 shock，但最终能部分恢复。

### 2.2 Step 1 首轮：PCC 软重置

首轮结果来自旧版 Step 1 脚本。当时只对 `bell` 扫了 `0.50/0.75/0.90`，其他关键场景主要使用 `0.75`。当前 `train_step_1.sh` 已升级为全关键场景 `0.50/0.65/0.75` 小扫参。

首轮结果摘要：

| 场景 | Ref-Gaussian 50k PSNR | PCC PSNR | 结论 |
| --- | ---: | ---: | --- |
| `bell_keep050` | 25.08 | 32.85 | 明显提升，且几乎消除最终回退 |
| `bell_keep075` | 25.08 | 29.13 | 有提升，但仍存在回退 |
| `bell_keep090` | 25.08 | 28.46 | 有提升，但回退更明显 |
| `toaster_keep075` | 25.89 | 28.40 | 明显提升 |
| `ball_keep075` | 36.09 | 36.85 | 小幅提升，稳定 |
| `chair_keep075` | 34.11 | 34.06 | 基本持平 |
| `mic_keep075` | 32.77 | 33.08 | 小幅提升 |
| `tbell_keep075` | 29.62 | 27.57 | 负收益，需要继续分析 |

阶段结论：

- PCC 是当前最确定有效的稳定模块。
- `pcc_keep_ratio` 不能最终场景手调，需要继续扫参并发展为自适应 keep。

### 2.3 Step 3 首轮：CGI + R2IF 组合

首轮结果来自旧版 Step 3 脚本。当前 `train_step_3.sh` 已改为可选 CGI debug 脚本，不作为下一轮主线。

阶段结论：

- CGI/R2IF 组合对 `mic` 有收益，对 `bell` 有部分收益，但对 `toaster` 明显负收益。
- CGI 暂时不进入主线。后续若重新启用，应作为 local ray-traced transport 的可见性/能量门控，而不是粗暴抑制镜面结果。

### 2.4 Step 1/2/4 最新实验回传结论

用户已完成新版 `train_step_1.sh`、`train_step_2.sh`、`train_step_4.sh` 的实验。

Step 1：PCC keep-ratio 小扫参。

- `pcc_keep_ratio=0.65` 是目前最好的全局折中，整体平均 PSNR 最高。
- `bell` 在 `0.50/0.65` 下稳定且大幅优于 baseline，`0.75` 会继续回退。
- `tbell` 明显偏向 `0.75`，说明最终不能依赖固定场景手调，需要发展 adaptive PCC。
- `mic` 仍存在后期回退，`0.65` 相比 `0.50/0.75` 更稳。

Step 2：R2IF/env。

- `bell=26.83`、`toaster=21.42`、`chair=31.40`，明显低于期望。
- 26k 附近 `bell/toaster` 出现明显回退。
- 代码检查发现当时 R2IF 直接执行 `specular = specular * specular_reliability`，这会压暗真实镜面颜色。该结果不能否定 R2SF 理论，只说明实现方式错误。

Step 4：GIP-0/NCIF。

- GIP-0 比 Step 2 稳定，`chair=33.89`、`mic=34.67`、`toaster=23.56`，说明低频漫反射辐照代理有一定信号。
- Ref-Real 基本没有显著提升，说明 per-Gaussian NCIF 不是最终解，仍需要真正的 spatial irradiance probes。
- `bell` 仍在 26k 后回退，和 R2IF 前向压暗问题相关，需要先修 R2SF 再复测。

工程处理：

- 已将 R2SF 改为 **forward-preserving gradient gate**：前向镜面颜色不变，反向梯度按 `q_spec` 缩放。
- 新增 `r2if_gate_render` 参数，默认 `False`。只有显式开启时才恢复旧的“直接压暗前向镜面”行为。
- 新增 `train_step_2_fix.sh` 和 `train_step_4_fix.sh`，用于小规模复测修复是否有效。

### 2.5 Step 2/4 fix 实验回传结论

用户已完成第一次 `train_step_2_fix.sh` 和 `train_step_4_fix.sh`。该版本把门控从“前向压暗”改成了“整个镜面分支的 gradient-only gate”。

Step 2 fix：`step2_r2sf_gradfix_30000_pcc065`。

- `bell=24.65`、`toaster=20.65`，仍明显低于 baseline 和 Step 1。
- `chair=31.14`，最终相对峰值回退 `1.78 dB`。
- 结论：只保证前向不变还不够。如果门控施加到整个 `specular`，低反射初始化会同时削弱材质、粗糙度、反射强度、法向和 Fresnel 权重学习，模型会被锁在低镜面解释。

Step 4 fix：`step4_gip0_gradfix_30000_pcc065`。

- `bell=24.69`、`toaster=20.72`，没有恢复。
- `chair=33.78`，比 Step 2 fix 稳定，但仍不能证明 R2SF 修复有效。
- 结论：GIP-0/NCIF 能缓解部分 diffuse 场景，但不能修复错误的远场镜面梯度分配。

第二次工程修正：

- R2SF 不再门控整个镜面分支，只门控环境贴图查询得到的远场光照 `L_env`。
- BRDF 材质因子、粗糙度、反射强度和法向继续完整接收重建损失梯度。
- 复测脚本默认提高 `r2if_min_specular_gate` 到 `0.15`，避免训练初期因为反射强度初始化偏低而完全切断 envmap 学习。
- 新输出目录改名为 `envgate`，与第一次失败的 `gradfix` 结果区分。

### 2.6 Step 2/4 envgate fix 实验回传结论

用户已完成第二次 `train_step_2_fix.sh` 与 `train_step_4_fix.sh`，即只门控远场环境贴图光照梯度的版本。以下以 Ref-Gaussian 30k 为主要基准。

Step 2 fix：`step2_r2sf_envgate_30000_pcc065_mingate015`。

| 场景 | Ref-Gaussian 30k | Step 2 fix | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `bell` | 26.68 | 26.76 | +0.08 | 最终略高，但训练峰值 32.37 后掉 4.64dB，仍不稳定 |
| `chair` | 34.21 | 34.30 | +0.09 | 小幅正收益 |
| `materials` | 30.53 | 30.07 | -0.46 | 负收益 |
| `mic` | 34.81 | 34.87 | +0.05 | 基本持平 |
| `gardenspheres` | 23.13 | 23.11 | -0.02 | 基本持平 |
| `toaster` | 27.76 | 27.42 | -0.35 | 低于 30k，但高于 50k 回退结果 |

Step 2 结论：R2SF env-light-only gate 修复了最严重的实现错误，但单独远场环境分支仍不能作为全量主线；`bell` 仍存在严重后期回退，`materials/toaster` 低于 30k 基准。

Step 4 fix：`step4_gip0_envgate_30000_pcc065_mingate015`。

| 场景 | Ref-Gaussian 30k | Step 4 fix | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `bell` | 26.68 | 31.39 | +4.71 | 明显解决回退，是当前最强正收益 |
| `chair` | 34.21 | 33.29 | -0.93 | 明显负收益，GIP-0 对 diffuse 设置过强 |
| `materials` | 30.53 | 30.18 | -0.36 | 负收益 |
| `mic` | 34.81 | 34.83 | +0.02 | 基本持平 |
| `gardenspheres` | 23.13 | 23.12 | -0.01 | 基本持平 |
| `toaster` | 27.76 | 27.85 | +0.09 | 小幅正收益，且无明显回退 |

Step 4 结论：GIP-0/NCIF 不是普适增益模块。它对 `bell` 这种后期回退反射场景很有效，对 `toaster/mic/gardenspheres` 基本安全，但当前强 diffuse 设置伤害 `chair/materials`。因此平均 PSNR 的提升主要来自 `bell`，不能据此直接跑全量 Step 4。

下一步工程处理：

- 暂不跑全量 Step 2/4。
- 新增 `train_step_4_safety.sh`，只做 GIP-0 安全性小扫参。
- 对反射控制场景保留已成功的 weak GIP-0 设置。
- 对 diffuse/weak-reflective 场景改用更晚、更弱、更受 roughness 约束的 conservative GIP-0 设置。

### 2.7 Step 4 safety sweep 实验回传结论

用户已完成 `train_step_4_safety.sh`。本轮目标是验证更保守的 GIP-0 是否能止住 `chair/materials` 的负收益，同时保留 `bell/toaster` 的反射场景收益。

结果：

| 场景 | Ref-Gaussian 30k | Safety | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `bell_weak` | 26.68 | 25.59 | -1.09 | 失败；训练曲线 25k 峰值 32.42，但 30k 离线 eval 崩到 25.59 |
| `toaster_weak` | 27.76 | 27.79 | +0.03 | 基本安全 |
| `chair_safe005` | 34.21 | 34.12 | -0.09 | 基本止损 |
| `chair_safe008` | 34.21 | 34.08 | -0.14 | 基本止损，略弱于 safe005 |
| `materials_safe005` | 30.53 | 30.28 | -0.25 | 仍偏低 |
| `materials_safe008` | 30.53 | 30.49 | -0.04 | 基本止损，是当前 materials 较好设置 |
| `mic_safe005` | 34.81 | 34.84 | +0.03 | 基本安全 |
| `gardenspheres_safe005` | 23.13 | 23.10 | -0.03 | 基本安全 |

结论：

- 更保守的 GIP-0 设置有效解决了 diffuse 场景被强低频残差伤害的问题。
- `bell` 的失败不是 diffuse 参数问题，而是 25k 后出现严重后期回退；这与 Ref-Gaussian 官方实现中部分场景的 delayed rendering 后期崩溃一致。
- 当前不应继续全量 Step 4，也不应立刻实现 GIP-1。必须先确认 best checkpoint/早停策略是否能稳定拿到 `bell` 的 25k 峰值。

工程处理：

- 新增 `eval_best_from_curve.py`，用于读取 `eval_curve.txt` 的峰值 iteration，并对对应保存点做离线 eval。
- 下一步先对 safety 输出运行 best checkpoint 离线评估，确认 `bell_weak@25000` 是否真的能得到 32dB 级别结果。

### 2.8 Best checkpoint 离线验证结论

用户已运行：

```bash
python eval_best_from_curve.py /data2/zmh/output_physnorm_steps/step4_oah_gip0_safety_30000_pcc065_mingate015 --only-flagged --save-images
```

结果：

| 场景 | 选择迭代 | 曲线峰值 | 曲线最终 | 离线 PSNR | 离线 SSIM | 离线 LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bell_weak` | 25000 | 32.4203 | 26.7620 | 31.8244 | 0.9668 | 0.0448 |

结论：

- `bell_weak@25000` 离线 eval 真实有效，说明 OAH-GS/GIP-0 在该场景确实能显著超过 Ref-Gaussian 30k。
- 失败原因不是方法无效，而是继续训练到 30k 后物理/材质分支崩溃。
- 后续实验必须把 **best checkpoint selection / early stopping** 纳入标准流程。

注意：

- 当前 `eval_curve.txt` 使用的是训练过程中的 test cameras，因此用于工程诊断是合理的。
- 若进入论文正式结果，不能直接用 test-set best selection 作为最终数字；需要改为 validation-based selection，或报告固定 iteration 并把 best checkpoint 作为稳定性分析。

### 2.9 Safety sweep 全场景 best 结果分析

用户已对 safety sweep 运行新版全场景 best checkpoint 离线评估。以下以 Ref-Gaussian 30k 为主要基准。本轮数据来自 `step4_oah_gip0_safety_30000_pcc065_mingate015`，因为包含 `chair/materials` 的两个参数变体，平均值只用于诊断，不作为最终论文平均指标。

| 场景 | Ref-Gaussian 30k | Safety best | 选择迭代 | 差值 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bell_weak` | 26.68 | 31.82 | 25k | +5.14 | 强正收益，但必须 best checkpoint |
| `toaster_weak` | 27.76 | 27.79 | 30k | +0.03 | 基本安全 |
| `chair_safe005` | 34.21 | 34.12 | 30k | -0.09 | 基本止损，是 chair 推荐设置 |
| `chair_safe008` | 34.21 | 33.42 | 28k | -0.80 | 曲线与离线不一致，不采用 |
| `materials_safe005` | 30.53 | 30.28 | 30k | -0.25 | 仍偏低 |
| `materials_safe008` | 30.53 | 30.49 | 30k | -0.04 | 基本止损，是 materials 推荐设置 |
| `mic_safe005` | 34.81 | 34.84 | 30k | +0.03 | 基本安全 |
| `gardenspheres_safe005` | 23.13 | 23.20 | 12k | +0.07 | 小幅正收益，但 LPIPS 偏高，需要后续看图 |

结论：

- Conservative GIP-0 + best checkpoint 已经形成可跑主线：强收益集中在后期回退反射场景，非反射/真实场景基本不伤害。
- `chair_safe008` 暴露出曲线 PSNR 与离线 eval 可不一致；后续不能只评估曲线最高的单个保存点，应对曲线 top-k 保存点逐个离线 eval。
- 全量 Step 4 可以继续，但必须采用 conservative 参数，并在训练后执行 top-k offline best selection。

工程处理：

- 已更新 `eval_best_from_curve.py`：默认从已保存的 point cloud iteration 中选择曲线 PSNR 最高的可离线评估点，不再因为峰值未保存而跳过场景。
- 已加密 Step 2/4/5 脚本在 22k-30k 的保存点，并加入 Ref-Real 的 14k 保存点，避免后续再错过峰值。
- 已将全量 `train_step_4.sh` 从旧的 strong/medium GIP-0 改为 safety 验证过的 conservative GIP-0 设置，默认输出改为 `step4_oah_gip0_safe_*`。
- 已增强 `eval_best_from_curve.py`：支持 `--eval-top-k K`，可对曲线排序前 K 个已保存点逐个离线 eval，并按离线 PSNR 选择真正 best checkpoint。

下一步运行全量 Step 4 safe：

```bash
bash train_step_4.sh
python eval_best_from_curve.py /data2/zmh/output_physnorm_steps/step4_oah_gip0_safe_30000_pcc065_mingate015 --eval-top-k 3 --save-images
```

如果全量 Step 4 safe 稳定，再决定是否实现 validation-based checkpoint selection 或进入 GIP-1。

### 2.10 全量 Step 4 safe 结果分析

用户已完成全量 `train_step_4.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step4_oah_gip0_safe_30000_pcc065_mingate015
```

固定最终轮次 `summary.txt` 显示：

- `bell` 最终 PSNR 只有 `25.5065`，训练曲线峰值在 `22k`，最终掉落 `5.5308 dB`。
- `chair` 最终 PSNR 只有 `32.6276`，但曲线峰值附近离线 eval 能恢复到 `34.1787`，说明曲线 drop 阈值不足以发现所有离线退化。
- 除 `bell` 外，`training_regression_summary.txt` 没有发现超过 `1 dB` 的曲线级回退。

`--eval-top-k 3` best checkpoint 离线评估结果如下，仍以 Ref-Gaussian 30k 为主要诊断基准：

| 场景 | Ref-Gaussian 30k | Step 4 final | Step 4 best | best 迭代 | best 差值 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bell` | 26.68 | 25.51 | 32.00 | 22k | +5.32 | best 很强，但最终严重崩溃 |
| `teapot` | 25.40 | 26.55 | 26.55 | 30k | +1.15 | 明显正收益 |
| `chair` | 34.21 | 32.63 | 34.18 | 28k | -0.04 | final 崩，best 基本持平 |
| `ficus` | 35.58 | 35.43 | 35.43 | 30k | -0.15 | 小幅负收益 |
| `hotdog` | 37.08 | 37.23 | 37.23 | 30k | +0.15 | 小幅正收益 |
| `lego` | 32.94 | 32.95 | 32.95 | 28k | +0.01 | 持平 |
| `materials` | 30.53 | 30.34 | 30.34 | 30k | -0.19 | 小幅负收益 |
| `mic` | 34.81 | 34.81 | 34.81 | 29k | 0.00 | 持平 |
| `ship` | 30.12 | 30.00 | 30.08 | 24k | -0.04 | 基本持平，但 LPIPS 偏高 |
| `gardenspheres` | 23.13 | 23.11 | 23.29 | 14k | +0.16 | PSNR 提升，但 LPIPS 变差，需要看图 |
| `sedan` | 26.23 | 26.08 | 26.08 | 20k | -0.15 | 小幅负收益 |
| `toycar` | 24.73 | 24.78 | 24.86 | 12k | +0.13 | PSNR/SSIM 提升，LPIPS 需看图 |
| `toaster` | 27.76 | 27.72 | 27.72 | 30k | -0.04 | 基本持平 |

阶段结论：

- 全量 Step 4 safe 已经证明 conservative GIP-0 不会大面积破坏 baseline，且在 `teapot/hotdog/gardenspheres/toycar` 有小幅正收益。
- 最大的表观收益来自 `bell` 的 best checkpoint。后续不再围绕 Ref-Gaussian 早期峰值反复审计，而是把 PCC 作为内部强基准，判断新模块是否仍有增量收益。
- `chair` 暴露了更严重的问题：固定最终 eval 与曲线 best 差距很大，但曲线级 drop 不超过 `1 dB`。因此后续必须用 top-k 离线评估，而不能只看 `training_regression_summary.txt`。
- 当前结果还不足以进入 GIP-1 或论文主结果。下一步以 PCC + best checkpoint 为内部强基准，继续测试能否通过 OAF 等模块提升弱反射和真实场景。
- 根据最新决策，Ref-Gaussian 30k 不再反复作为工程推进中心；后续以 PCC + best checkpoint 为内部强基准，Step 4/GIP-0 只在证明超过 PCC 时才保留为主线模块。

工程处理：

- 已给 `eval_best_from_curve.py` 增加 `--min-iteration/--max-iteration`，用于在同一训练预算内做公平 best checkpoint 比较。
- 下一轮无需重训，先对已有 output 做公平离线审计。

### 2.11 当前决策：以 PCC 为内部基准继续推进

最新实验策略调整如下：

- Ref-Gaussian 30k 作为外部最低下限，不再反复围绕其早期 best 做大量审计。
- PCC 已经确认为有效稳定模块，后续把 `PCC + best checkpoint` 视为内部强基准。
- 每个新模块都必须回答：叠加到 PCC 后，是否在关键场景上超过 PCC；若只超过 Ref-Gaussian 但不超过 PCC，则不能作为主线创新。
- 因此下一项实验不再继续回头测 Ref-Gaussian，而是测试 **OAF：低可观测区域外观回退**。

OAF 目标：

1. 对 `chair/materials/ship` 等弱反射或漫反射场景，避免物理分支后期把指标拉低。
2. 对 `gardenspheres/toycar/sedan` 等真实场景，缓解曝光、模糊、细结构和不可观测物理残差对 PBR 分支的污染。
3. 对 `bell/teapot/toaster` 等反射控制场景，OAF 必须保持弱作用，不能偷走可靠镜面解释。

新增脚本：

```bash
bash train_step_6_oaf.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step6_oah_oaf_30000_pcc065_tau030_mingate015
```

脚本会在训练后自动运行：

```bash
python data_collect.py "$OUT_DIR"
python eval_best_from_curve.py "$OUT_DIR" --max-iteration 30000 --eval-top-k 5 --save-images
```

回传文件：

```bash
summary.txt
training_regression_summary.txt
training_regression_drop.svg
best_iteration_eval_summary.txt
best_iteration_eval_summary.json
```

### 2.12 Step 6 OAF 实验回传结论

用户已完成 `train_step_6_oaf.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step6_oah_oaf_30000_pcc065_tau030_mingate015
```

本轮使用 Ref-Gaussian baseline 文件 `ref_gaussian_baseline.txt` 作为外部下限。需要注意，该文件中的 `bell=25.08`、`mic=32.77`、`toaster=25.89` 更接近 Ref-Gaussian 50k/回退后的结果；因此它适合作为外部下限，但不能代替 PCC 内部强基准。

OAF best checkpoint 结果：

| 场景 | Ref-Gaussian baseline | OAF best | 差值 | 与 Step 4 safe 对比 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bell` | 25.08 | 32.25 | +7.17 | +0.25 | 强于 Ref 下限，但仍低于已知 PCC `bell_keep050=32.85` |
| `teapot` | 25.40 | 26.53 | +1.13 | -0.01 | 与 Step 4 基本一致 |
| `chair` | 34.11 | 34.13 | +0.02 | -0.05 | 基本持平，不构成明确增量 |
| `hotdog` | 36.99 | 37.19 | +0.20 | -0.04 | 小幅高于 Ref，下游价值有限 |
| `materials` | 30.53 | 30.11 | -0.42 | -0.23 | 负收益 |
| `mic` | 32.77 | 34.83 | +2.06 | +0.02 | 高于 Ref 回退值，但与 Step 4 基本一致 |
| `ship` | 30.12 | 30.03 | -0.09 | -0.05 | 小幅负收益 |
| `gardenspheres` | 23.13 | 23.29 | +0.16 | -0.01 | PSNR 小升，但 LPIPS 从约 0.266 变差到 0.289 |
| `sedan` | 26.23 | 26.00 | -0.23 | -0.08 | 负收益 |
| `toycar` | 24.73 | 24.86 | +0.13 | 0.00 | PSNR 小升，但 LPIPS 比 Ref 下限更差 |
| `toaster` | 25.89 | 27.70 | +1.81 | -0.02 | 高于 Ref 下限，但低于已知 PCC `toaster_keep075=28.40` |

训练稳定性：

- 只有 `bell` 被 `training_regression_summary.txt` 标记为明显回退：峰值 `32.4361`，最终曲线 `27.8404`，drop `4.5957 dB`。
- OAF 能把 `bell` 最终值从 Step 4 final 的 `25.51` 提到 `27.32`，说明外观回退有一定抗崩作用，但仍不能替代 checkpoint selection。

阶段结论：

- OAF 相对 Ref-Gaussian 回退下限有收益，但相对 PCC/Step 4 内部强基准没有形成稳定增量。
- OAF 对 `materials/sedan` 有明确负收益，对真实场景的 LPIPS 也偏差，说明当前“渲染结果级混合”过于粗糙。
- OAF 暂不作为主线创新，只保留为 ablation 或真实低可观测区域的候选机制。
- 下一步不继续调 OAF，而是强化已经被验证有效的 PCC，进入 **APCC：Adaptive Phase-Consistent Continuation**。

工程处理：

- 已实现 `--use_adaptive_pcc`，将固定 `pcc_keep_ratio` 扩展为 Gaussian 级自适应 keep。
- 已新增 `train_step_7_apcc.sh`，用于隔离测试 APCC，本轮关闭 R2SF/GIP/OAF/CGI，避免归因混乱。

### 2.13 Step 7 APCC 实验回传结论

用户已完成 `train_step_7_apcc.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step7_apcc_30000_keep045-085
```

本轮评估脚本结果正确：`best_iteration_eval_summary.txt` 中已经显示 `Resume cache: on (metric_best_ITER.txt)`，JSON 中每个候选点均带有 `cached: true` 和 `used_cache: true`。这说明断网后重跑脚本时确实从已生成的 `metric_best_ITER.txt` 继续汇总，没有从头重复离线评估。

APCC best checkpoint 结果如下：

| 场景 | Ref-Gaussian baseline | APCC best | 差值 | 与现有内部基线对比 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bell` | 25.08 | 31.99 | +6.91 | 低于已知 PCC `bell_keep050=32.85`，接近 Step4/OAF best | APCC 缓解 final 崩溃，但 best 不够高 |
| `teapot` | 25.40 | 26.62 | +1.22 | 略高于 Step4/OAF | 正收益 |
| `chair` | 34.11 | 33.96 | -0.15 | 低于 Step4/OAF/PCC 附近结果 | 负收益 |
| `hotdog` | 36.99 | 37.12 | +0.13 | 低于 Step4/OAF | 小幅正收益但不强 |
| `materials` | 30.53 | 30.66 | +0.13 | 明显高于 Step4/OAF | 当前 APCC 最有价值的场景 |
| `mic` | 32.77 | 34.78 | +2.01 | 与 Step4/OAF 接近，明显高于 Ref 回退值 | 有效但非新增强收益 |
| `ship` | 30.12 | 30.00 | -0.12 | 略低于 Ref/Step4/OAF | 小幅负收益 |
| `toaster` | 25.89 | 27.48 | +1.59 | 低于已知 PCC `toaster_keep075=28.40`，也低于 Step4/OAF | 反射控制不合格 |

训练稳定性：

- `bell` 仍被标记为回退，但 drop 从 Step4 的 `5.5308 dB`、OAF 的 `4.5957 dB` 降到 `1.3138 dB`，说明 APCC 对阶段退化确实有稳定作用。
- `toaster` 没有超过 `1 dB` 曲线回退，但 best 和 final 都低于固定 PCC 经验值，说明问题不是 checkpoint，而是 APCC 当前 keep 策略本身对反射物体不合适。

阶段结论：

- APCC 作为理论方向是有信号的：它提升 `materials`，显著降低 `bell` 后期崩溃幅度。
- 当前 APCC 参数没有超过固定 PCC 内部强基准，主要短板是 `toaster/chair`。
- 当前自适应策略对反射物体保留不足，且对所有属性使用同一个 keep tensor 过于粗糙。
- 不应直接把 APCC 设为默认基准；也不应立刻放弃。下一步只做一轮 high-keep APCC 修正实验，如果仍不超过固定 PCC，则回退固定 PCC 并进入 GIP-1。

下一步新增脚本：

```bash
bash train_step_7_apcc_highkeep.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step7_apcc_highkeep_30000_keep055-095_spec06_gamma075
```

该脚本把 keep 范围从 `0.45-0.85` 提高到 `0.55-0.95`，并提高 specular confidence 权重，目标是修复 `toaster` 和反射控制场景的负收益。

### 2.14 Step 7 APCC high-keep 实验回传结论

用户已完成 `train_step_7_apcc_highkeep.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step7_apcc_highkeep_30000_keep055-095_spec06_gamma075
```

固定最终轮次 `summary.txt` 显示全部场景最终曲线已经较稳定，`training_regression_summary.txt` 未发现超过 `1 dB` 的开倒车场景。`--eval-top-k 5` best checkpoint 离线评估结果如下：

| 场景 | Ref-Gaussian baseline | 首版 APCC best | APCC high-keep best | 变化 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bell` | 25.08 | 31.99 | 32.70 | +0.71 | 明显改善，且最终不再回退；但仍略低于已知固定 PCC `bell_keep050=32.85` |
| `teapot` | 25.40 | 26.62 | 26.39 | -0.23 | 仍高于 Ref 下限，但 high-keep 使该场景退化 |
| `chair` | 34.11 | 33.96 | 34.11 | +0.15 | 回到 Ref 下限附近，但未形成明确正收益 |
| `hotdog` | 36.99 | 37.12 | 37.11 | -0.01 | 基本持平，小幅高于 Ref 下限 |
| `materials` | 30.53 | 30.66 | 30.61 | -0.06 | 保留正收益，但略低于首版 APCC |
| `mic` | 32.77 | 34.78 | 34.73 | -0.05 | 仍显著高于 Ref 回退值，但不是新增强收益 |
| `ship` | 30.12 | 30.00 | 30.01 | +0.01 | 仍略低于 Ref 下限，说明 APCC 对该场景无效 |
| `toaster` | 25.89 | 27.48 | 28.02 | +0.54 | 比首版 APCC 明显改善，但仍低于已知固定 PCC `toaster_keep075=28.40` |

阶段结论：

- high-keep APCC 确认了一个重要现象：提高保留率和镜面置信度权重能显著降低阶段切换后的退化，`bell` 从首版 APCC 的 `1.3138 dB` 回退降低到 `0`，全局没有超过 `1 dB` 的开倒车场景。
- high-keep APCC 不是新的默认主线。它在 `bell/toaster/chair` 上相对首版 APCC 有改善，但仍没有稳定超过固定 PCC 内部强基准，尤其 `toaster` 还差约 `0.38 dB`。
- APCC 的当前形式适合作为 **stability ablation** 或后续的属性分组软重置基础，但不应继续做大量手工扫参。继续扫 `keep/min/max/gamma` 的收益会变得经验化，不利于论文创新主线。
- 下一阶段回到固定 `pcc_keep_ratio=0.65` 作为内部强基准，优先推进真正服务于混合反射/漫反射光照分解的 GIP-1/MGIP；APCC high-keep 作为可选稳定模块保留，不进入默认主方法。

### 2.15 Step 8 固定 PCC validation 强基准回传结论

用户已完成 `train_step_8_pcc_val.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step8_pcc_val_30000_pcc065_hold8_max8
```

固定最终轮次 `summary.txt` 显示所有场景均无超过 `1 dB` 的开倒车，说明固定 PCC 在 validation split 下仍是稳定强基准。最终轮次相对 Ref-Gaussian baseline 的关键变化如下：

| 场景 | Ref-Gaussian baseline | PCC-val final | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `bell` | 25.08 | 32.63 | +7.55 | 强正收益，阶段回退基本解决 |
| `teapot` | 25.40 | 25.57 | +0.17 | 小幅正收益 |
| `toaster` | 25.89 | 27.89 | +2.00 | 正收益，但低于早期固定 PCC 诊断记录 `28.40` |
| `chair` | 34.11 | 33.52 | -0.59 | 明显低于 Ref 30k，需要后续模块修复 |
| `hotdog` | 36.99 | 36.50 | -0.49 | 低于 Ref 30k，说明固定 PCC 不是 diffuse 场景最优 |
| `materials` | 30.53 | 30.68 | +0.15 | 小幅正收益 |
| `mic` | 32.77 | 34.77 | +2.00 | 明显高于 Ref 回退下限 |
| `ship` | 30.12 | 29.99 | -0.13 | 小幅负收益 |
| `gardenspheres` | 23.13 | 23.11 | -0.02 | 基本持平 |
| `sedan` | 26.23 | 26.17 | -0.06 | 基本持平 |
| `toycar` | 24.73 | 24.66 | -0.07 | 基本持平 |

需要特别说明：本轮首个 `best_iteration_eval_summary.txt` 是用 `--eval-top-k 5` 生成的诊断摘要。它先按 `val/val_fast` 排出前 5 个候选，再按离线 test PSNR 选择其中最优点，因此适合分析 validation-test 相关性，但不能作为正式论文数值。该诊断揭示了一个重要问题：

| 场景 | 纯 val top-1 PSNR | top-k oracle PSNR | 差值 | 现象 |
| --- | ---: | ---: | ---: | --- |
| `bell` | 32.01 | 32.64 | +0.63 | val top-1 与 test 最优不完全一致 |
| `hotdog` | 35.75 | 36.50 | +0.76 | val top-1 选早了 |
| `sedan` | 24.90 | 26.18 | +1.28 | validation split 与 test 明显失配 |
| `toaster` | 27.19 | 27.89 | +0.71 | val top-1 选早了 |

用户随后完成了严格口径 `eval_step_8_pcc_val_strict.sh`，即 `--selection-mode curve --eval-top-k 1 --summary-suffix val`。该口径完全按照 validation 曲线 top-1 选点，再只评估一次 test cameras：

| 场景 | strict val 选点 | strict test PSNR | SSIM | LPIPS | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bell` | 29000 | 32.0101 | 0.9694 | 0.0404 | 仍显著高于 Ref-Gaussian，PCC 稳定性成立 |
| `teapot` | 29000 | 25.3106 | 0.9385 | 0.0633 | 基本接近 Ref 下限 |
| `chair` | 24000 | 33.7964 | 0.9775 | 0.0241 | 低于 Ref 30k，diffuse 类需要新模块 |
| `hotdog` | 24000 | 35.7486 | 0.9790 | 0.0333 | 低于 Ref 30k，validation 选点偏保守 |
| `materials` | 30000 | 30.6758 | 0.9662 | 0.0364 | 小幅正收益 |
| `mic` | 28000 | 34.7237 | 0.9902 | 0.0081 | 明显高于 Ref 回退下限 |
| `ship` | 24000 | 29.9910 | 0.8900 | 0.1398 | 仍弱于 Ref 30k |
| `gardenspheres` | 14000 | 23.0816 | 0.6191 | 0.2918 | Ref-Real 没有收益 |
| `sedan` | 10000 | 24.8962 | 0.7237 | 0.2826 | validation split 与 test 失配明显 |
| `toycar` | 10000 | 24.6158 | 0.6795 | 0.2751 | 基本持平偏低 |
| `toaster` | 29000 | 27.1872 | 0.9495 | 0.0694 | 高于 Ref 50k，低于 oracle 诊断 |
| **平均** | - | **29.2761** | **0.8802** | **0.1149** | 固定 PCC 可作为强基准，但不是最终创新 |

阶段结论：

- 固定 PCC=0.65 可以作为后续内部强基准，但正式报告必须使用纯 validation top-1 selection 或固定最终迭代，不能使用 top-k oracle。
- 当前 validation split 在 `sedan/hotdog/toaster` 上与 test 最优点相关性不足。后续若进入正式论文结果，需要统一使用 validation top-1、固定最终迭代，或为 Ref-Real 单独论证固定 20k 口径；不能混用 oracle 选点。
- 从方法推进角度，固定 PCC 已经解决了反射场景稳定性，但对 `chair/hotdog/ship/RefReal` 没有带来系统提升。因此下一项真正有意义的创新仍然是 GIP-1/MGIP，而不是继续调 PCC。

工程处理：

- `eval_best_from_curve.py` 新增 `--selection-mode {offline,curve}` 与 `--summary-suffix`。`offline` 保留旧的 top-k oracle 诊断；`curve` 使用曲线第一名作为正式选点。
- `train_step_8_pcc_val.sh` 已改为默认使用 `--selection-mode curve --eval-top-k 1 --summary-suffix val`。
- `eval_step_8_pcc_val_strict.sh` 已完成验证，可在不重训的情况下对已有 step8 output 生成严格 validation top-1 摘要。
- 固定 PCC 已冻结为后续内部强基准。下一步不再扫 PCC/APCC，而是实现并测试 GIP-1/MGIP。

### 2.16 Step 5 GIP-1 grid probe 实验回传结论

用户已完成 `train_step_5.sh`，输出目录为：

```bash
/data2/zmh/output_physnorm_steps/step5_oah_gip1_probe_30000_pcc065_grid8_mingate015_valhold8
```

严格 validation top-1 结果如下：

| 场景 | PCC strict | GIP-1 strict | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `bell` | 32.0101 | 31.9364 | -0.0737 | 轻微负收益，反射控制基本未崩 |
| `teapot` | 25.3106 | 26.2949 | +0.9843 | 明显正收益，是当前 GIP-1 最清晰信号 |
| `chair` | 33.7964 | 33.8270 | +0.0306 | 基本持平 |
| `materials` | 30.6758 | 29.9529 | -0.7229 | 明显负收益，说明 probe 抢了错误解释权 |
| `mic` | 34.7237 | 34.7219 | -0.0018 | 持平 |
| `ship` | 29.9910 | 30.1840 | +0.1930 | 小幅正收益 |
| `gardenspheres` | 23.0816 | 23.1034 | +0.0218 | 持平 |
| `sedan` | 24.8962 | 24.6580 | -0.2382 | strict 选点负收益；final 有较高 PSNR，但不能作为正式口径 |
| `toycar` | 24.6158 | 24.6169 | +0.0011 | 持平 |
| `toaster` | 27.1872 | 25.3095 | -1.8777 | 严重负收益，反射控制失败 |

共同场景平均约为 `28.46 dB`，低于固定 PCC strict 共同场景平均约 `28.63 dB`。若直接看全输出 final，Step 5 平均 `28.6474 dB`、`LPIPS=0.1169`、`FPS=52.51`；但 strict best 平均 `28.4605 dB`、`LPIPS=0.1245`、`FPS=38.60`。因此，GIP-1 grid probe 目前不能作为默认主线。

训练效率也出现明显问题：真实场景训练时间从约 `40 min` 增至约 `2 h`。主要原因不是 probe 参数量本身，而是每次渲染都需要对大量 Gaussians 做空间 grid 查询、SH irradiance 计算并增加 rasterized feature 通道；真实场景 Gaussian 数量较大，因此开销被放大。

阶段结论：

- GIP-1 grid probe 有局部信号，例如 `teapot/ship`，说明“低频漫反射光照分支”这个方向不是完全错误。
- 但当前规则空间 probe 太像可学习空间颜色场，容易和原始 Gaussian 颜色、envmap、roughness/refl 分支抢解释权。
- `toaster` 大幅下降说明 probe 即使有弱门控，也会破坏反射控制场景的高频镜面路径。
- 对真实场景，GIP-1 strict 口径没有系统收益，且训练时间不可接受。
- 因此，**GIP-1 grid probe 暂停作为默认主线**；保留为 ablation，不继续做大规模 grid/visibility/DDGI-lite 扩展。

下一步改为 **PRT 启发的 Gaussian Radiance Transfer**：不再学习密集空间 probe grid，而是把低频光照表示为共享 SH lighting，把每个 Gaussian 的局部传输表示为由法线、材质和少量可学习遮蔽/低秩系数控制的 transfer function。目标是保留 MGIP 的物理解释，同时降低训练成本并减少与颜色场抢解释权。

---

## 3. 工程实现路线图

### 3.1 已实现并可继续实验

- [x] Ref-Gaussian 主渲染链路复现。
- [x] PCC soft reset 首版。
- [x] APCC Gaussian 级自适应软重置首版。
- [x] R2SF/R2IF env-light-only gradient gate 与 env regularization 首版。
- [x] Per-Gaussian NCIF residual，即 GIP-0 低频辐照代理。
- [x] GIP-1 learnable grid irradiance probes 首版：规则空间 probe grid、SH2 辐照、trilinear interpolation、diffuse responsibility gate、保存/加载与训练可视化。当前实验显示不进入默认主线，仅保留为 ablation。
- [x] OAF 低可观测外观回退首版，但当前结果不足以进入主线。
- [x] CGI 首版，但当前结果不稳定，暂缓主线使用。
- [x] 训练曲线记录、开倒车检测和 `data_collect.py` 汇总。

### 3.2 当前优先验证：OAH 安全性边界

目标：确认可观测性门控不会伤害 Ref-Gaussian/PCC 已经能做好的场景。只有 `bell/toaster` 等反射控制场景恢复、`chair/materials/mic` 等弱反射场景不出现明显负收益，才继续跑全量 Step 2/4。

验证内容：

1. 已完成 `train_step_2_fix.sh`：R2SF env-light-only gate 基本修复实现错误，但 Step 2 单独不够稳定。
2. 已完成 `train_step_4_fix.sh`：GIP-0 能救 `bell`，但伤害 `chair/materials`。
3. 已完成 `train_step_4_safety.sh`：更保守 GIP-0 基本保住 `chair/materials`，但 `bell` 再次后期崩溃。
4. 已完成 `eval_best_from_curve.py` 验证：`bell_weak@25000` 离线 PSNR 为 `31.8244`，峰值 checkpoint 真实有效。
5. 已发现旧版 best 工具会跳过峰值未保存的场景；已修复为从已保存 iteration 中选择 best，并加密后续训练脚本保存点。
6. 已完成新版 safety sweep 全场景 best 汇总；下一轮跑全量 `train_step_4.sh` 的 safe 版本，并用 `--eval-top-k 3` 选择离线 best。

### 3.3 下一步候选实现：GIP-1 最小探针场

目标：实现真正的可学习 spatial irradiance probe field，验证“漫反射区域监督低频探针，而不是污染 envmap”这一核心主张。

代码任务：

1. [x] 在 `GaussianModel` 中维护规则 grid probes。
2. [x] 每个 probe 存 SH2 RGB 系数，默认 `probe_grid_res=8`，RGB 共 `9 * 3` 维。
3. [x] 根据 scene bbox 初始化 probe grid，支持 `probe_grid_res`。
4. [x] 渲染时对 Gaussian 位置做 trilinear interpolation，计算 `I_probe(x, n)`，再随高斯属性 rasterize 到像素。
5. [x] 使用 `q_diff=(1-reflectance)^mu * roughness^nu` 控制 probe 对 diffuse 的贡献。
6. [x] 加入 probe image smoothness、grid energy、rendered magnitude 正则。
7. [x] 支持保存/加载 `.probe` 参数，训练可视化中输出 `probe_map` 与 `probe_responsibility`。
8. [ ] Probe grid 导出为独立三维可视化体或点云；首轮实验先用训练可视化和指标判断。

新增参数建议：

```text
--use_probe_gi
--probe_lr
--probe_from_iter
--probe_ramp_iters
--probe_grid_res
--probe_sh_degree
--probe_tau
--probe_diffuse_mu
--probe_diffuse_nu
--lambda_probe_smooth
--lambda_probe_energy
--lambda_probe_magnitude
```

### 3.4 下一步候选：PRT-GS / Gaussian Radiance Transfer

目标：用 PRT 风格的低频 radiance transfer 替代密集 spatial probe grid。PRT 的核心不是 probe 数量，而是把动态低频光照 $L$ 和局部传输 $T$ 分解到 SH 基函数中，渲染时计算 $\langle L,T\rangle$。这更适合 3DGS：高斯本身已有 SH 工具链，但原生 SH 表示的是 view-dependent radiance，不是物理光照和传输。下一步要做的是把它改造成“光照 SH + Gaussian transfer SH”的受控分解。

代码任务：

1. 新增全局或少量分区的 diffuse lighting SH，默认 SH2，RGB 共 `9 * 3` 维。
2. 每个 Gaussian 使用解析 Lambertian transfer basis，由法线给出低频 clamped-cosine transfer。
3. 每个 Gaussian 只额外学习低维 transfer residual，例如 `occlusion` 标量或 `rank=2/4` 的低秩系数，而不是学习完整空间 probe grid。
4. 使用 `q_diff=(1-refl)^mu * roughness^nu` 控制该分支只影响漫反射/粗糙区域。
5. 对 transfer residual 加稀疏、幅值和空间平滑正则，避免变成任意颜色残差。
6. 渲染时将 $\langle L,T_i\rangle$ 作为 per-Gaussian diffuse irradiance feature rasterize，避免 grid trilinear 查询。
7. 首轮只测 `chair/materials/ship/sedan/toycar/bell/toaster/teapot`，不跑全量。

预期优势：

- 比 GIP-1 grid probe 快：没有空间 grid 查询，也没有大量 probe 参数。
- 比原始 Gaussian SH 更可解释：原生 3DGS 的 SH 是视角相关颜色，PRT-GS 的 SH 是低频 lighting/transfer 内积。
- 比 per-Gaussian NCIF 更不容易过拟合：transfer 由法线解析项主导，只允许小 residual。
- 更适合论文叙事：核心创新从“加一个 probe”转为“可观测性驱动的 Gaussian radiance transfer factorization”。

### 3.5 延后候选：DDGI-lite 可见性

目标：若 PRT-GS 在真实场景有收益但出现遮挡/漏光，再利用已有 ray tracing 能力补充 visibility moments。

代码任务：

1. 每隔若干步，从 probe 向固定方向采样 rays。
2. 与已有 mesh / Gaussian surface 交互，记录 hit depth moments。
3. 查询 probe 时使用 visibility weight 抑制穿墙或遮挡错误。
4. 将 ray-traced local transport 和 probe irradiance 区分：前者负责近场镜面，后者负责低频漫反射。

### 3.6 最后阶段：完整 OAH-GS

目标：形成最终方法。

组成：

- 外观基底和低可观测区域安全回退。
- PCC 自适应软重置。
- R2SF 远场 envmap 可靠监督。
- GIP-1/2 probe irradiance。
- DDGI-lite visibility。
- Ray-traced local specular/interreflection。
- 统一数据汇总、曲线、envmap/probe 可视化。

---

## 4. 训练脚本状态

当前工作区已经清理旧 sweep/fix/debug 脚本。保留的入口只有：

| 脚本 | 状态 | 用途 |
| --- | --- | --- |
| `train.sh` | 保留 | 原始/批量训练命令集合，便于和早期实验对照 |
| `train_step_0.sh` | 保留 | Ref-Gaussian baseline 全场景诊断 |
| `train_step_8_pcc_val.sh` | 保留 | 固定 `PCC=0.65` 强基准，validation top-1 选点 |
| `train_step_5.sh` | 保留为 ablation | GIP-1 grid probe 复现实验；当前结论是不进入默认主线 |
| `train_step_9_prt.sh` | 新增 | PRT-GS / Gaussian Radiance Transfer 首轮关键场景实验 |

已删除的脚本包括 `train_step_1/2/2_fix/3/4/4_fix/4_safety/6/7/7_highkeep.sh` 与 `eval_step_8_pcc_val_strict.sh`。这些脚本对应的实验结论已经记录在第 2 节；后续不再从文件入口运行它们。

下面 4.1-4.10 保留为历史说明，用于追踪旧实验设计，不代表当前需要运行。

### 4.1 `train_step_0.sh`

用途：全场景 Ref-Gaussian baseline 诊断。

状态：已完成，不需要立刻重跑。

### 4.2 `train_step_1.sh`

用途：PCC keep-ratio 小扫参。

当前测试内容：

- 场景：`bell / toaster / ball / tbell / chair / mic`
- 参数：`pcc_keep_ratio = 0.50 / 0.65 / 0.75`
- 关闭：R2IF、GIP/NCIF、CGI、envmap 正则
- 默认轮次：50k

运行：

```bash
bash train_step_1.sh
```

输出：

```bash
/data2/zmh/output_physnorm_steps/step1_pcc_sweep_50000
```

### 4.3 `train_step_2.sh`

用途：OAH-GS 远场镜面分支，即 R2SF env-light-only gate + 环境贴图 TV/energy 正则。

当前测试内容：

- 合成场景：`bell / toaster / chair / mic / materials / ship`
- 真实场景：`gardenspheres / toycar / sedan`
- 开启：PCC、R2IF/env 正则
- 关闭：GIP/NCIF、CGI
- 默认轮次：合成 30k，Ref-Real 20k
- 默认 `PCC_KEEP=0.65`
- 默认 `R2SF_MIN_GATE=0.15`

运行：

```bash
bash train_step_2.sh
```

若 Step 1 明确支持其他统一 PCC 值：

```bash
PCC_KEEP=0.50 bash train_step_2.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step2_oah_r2sf_envgate_30000_pcc065_mingate015
```

### 4.3.1 `train_step_2_fix.sh`

用途：R2SF env-light-only gradient gate 修复后的远场镜面分支小规模复测。

测试场景：

- `bell / toaster / chair / mic / materials / gardenspheres`

运行：

```bash
bash train_step_2_fix.sh
```

输出：

```bash
/data2/zmh/output_physnorm_steps/step2_r2sf_envgate_30000_pcc065_mingate015
```

### 4.4 `train_step_3.sh`

用途：CGI / local transport debug。

状态：暂缓主线。只有当 Step 2 和 GIP 实验完成后，再用它判断 ray-traced local transport 应该如何接回最终方法。

### 4.5 `train_step_4.sh`

用途：OAH-GS 低频漫反射代理分支，即当前 per-Gaussian NCIF irradiance proxy。

当前测试内容：

- NeRF Synthetic：`chair / ficus / hotdog / lego / materials / mic / ship`
- Ref-Real：`gardenspheres / toycar / sedan`
- 反射控制：`bell / teapot / toaster`
- 开启：PCC、R2IF/env、GIP-0/NCIF
- 关闭：CGI
- 默认轮次：合成 30k，Ref-Real 20k
- 默认 `R2SF_MIN_GATE=0.15`

运行：

```bash
bash train_step_4.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step4_oah_gip0_safe_30000_pcc065_mingate015
```

### 4.5.1 `train_step_4_fix.sh`

用途：R2SF env-light-only gradient gate 修复后的 GIP-0/NCIF 小规模复测。

测试场景：

- `bell / toaster / chair / mic / materials / gardenspheres`

运行：

```bash
bash train_step_4_fix.sh
```

输出：

```bash
/data2/zmh/output_physnorm_steps/step4_gip0_envgate_30000_pcc065_mingate015
```

### 4.6 `train_step_5.sh`

用途：OAH-GS GIP-1 learnable grid irradiance probes。

状态：GIP-1 首版已实现，可以直接运行。脚本仍会检测 `--use_probe_gi` 参数，避免服务器同步到旧代码时误跑旧逻辑。默认输出名为：

```bash
/data2/zmh/output_physnorm_steps/step5_oah_gip1_probe_30000_pcc065_grid8_mingate015_valhold8
```

运行：

```bash
bash train_step_5.sh
```

本轮目的：以固定 PCC=0.65 为内部强基准，验证 spatial probe 是否能补强 `chair/materials/ship/RefReal` 这类低反射或大场景，并确认 `bell/teapot/toaster` 不被 probe 吞掉高频镜面。

### 4.7 `train_step_6_oaf.sh`

用途：PCC + OAF 低可观测区域外观回退。

当前测试内容：

- Diffuse/weak-reflective：`chair / materials / ship / mic / hotdog`
- Ref-Real：`gardenspheres / toycar / sedan`
- Reflective controls：`bell / teapot / toaster`
- 开启：PCC、R2SF、OAF
- 关闭：NCIF/GIP-0、CGI
- 默认轮次：合成 30k，Ref-Real 20k

运行：

```bash
bash train_step_6_oaf.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step6_oah_oaf_30000_pcc065_tau030_mingate015
```

当前结论：OAF 不进入下一轮主线，只作为 ablation 保留。

### 4.8 `train_step_7_apcc.sh`

用途：APCC 自适应阶段一致连续优化。

当前测试内容：

- Synthetic controls：`bell / teapot / toaster`
- Diffuse/weak-reflective：`chair / materials / mic / ship / hotdog`
- 开启：PCC、Adaptive PCC
- 关闭：R2SF、NCIF/GIP-0、OAF、CGI
- 默认轮次：30k

运行：

```bash
bash train_step_7_apcc.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step7_apcc_30000_keep045-085
```

当前结论：首版 APCC 不进入默认主线，但保留为有信号的候选。下一步运行 high-keep 变体。

### 4.9 `train_step_7_apcc_highkeep.sh`

用途：APCC 高保留率修正实验。

差异：

- `pcc_min_keep=0.55`
- `pcc_max_keep=0.95`
- `pcc_opacity_weight=0.4`
- `pcc_specular_weight=0.6`
- `pcc_confidence_gamma=0.75`

运行：

```bash
bash train_step_7_apcc_highkeep.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step7_apcc_highkeep_30000_keep055-095_spec06_gamma075
```

当前结论：high-keep APCC 是稳定性有效、峰值不够强的候选模块。它不进入默认主线，但作为 PCC 相关消融保留。

### 4.10 `train_step_8_pcc_val.sh`

用途：固定 PCC 强基准的 validation-based checkpoint selection 版本。

当前测试内容：

- Reflective controls：`bell / teapot / toaster`
- Diffuse/weak-reflective：`chair / hotdog / materials / mic / ship`
- Ref-Real：`gardenspheres / toycar / sedan`
- 开启：固定 `PCC_KEEP=0.65`
- 关闭：R2SF、NCIF/GIP-0、OAF、CGI
- 从训练视角中按 `VAL_HOLD=8, VAL_MAX=8` 固定抽取 validation cameras
- 用 `val/val_fast` 曲线选择 checkpoint，再对 test cameras 做离线评估
- 默认生成 `best_iteration_eval_summary_val.txt/json`，这是严格 validation top-1 口径

运行：

```bash
bash train_step_8_pcc_val.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step8_pcc_val_30000_pcc065_hold8_max8
```

---

## 5. 接下来执行顺序

当前不要继续重跑 GIP-1 grid probe，也不要继续扫 PCC/APCC/OAF/NCIF。下一轮主线改为：

1. **实现 PRT-GS / Gaussian Radiance Transfer 首版。** 使用共享低频 lighting SH 和 Gaussian 解析/低秩 transfer，替代密集 grid probe。
2. **保持固定 PCC=0.65 强基准。** 所有新模块都必须和 `train_step_8_pcc_val.sh` 的 strict 口径比较。
3. **第一轮只跑小规模关键场景。** 建议 `chair / materials / ship / sedan / toycar / bell / toaster / teapot`，先判断方向，不跑全量。
4. **效率必须作为硬指标。** 若训练时间接近 GIP-1 grid probe 的 `2 h` 级别，即使指标小涨也不进入主线。
5. **若 PRT-GS 有信号，再做 visibility/分区扩展。** 若没有信号，回退到固定 PCC + appearance/exposure/anti-aliasing/geometry 稳定化路线。

PRT-GS 首版已经落地后，运行：

```bash
bash train_step_9_prt.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step9_prt_gs_30000_pcc065_tau012_mingate015_valhold8
```

回传 `summary.txt`、`training_regression_summary.txt`、`training_regression_drop.svg`、`best_iteration_eval_summary_val.txt/json`。若某个场景异常，再补 `test/renders/prt/` 与 `test/renders/prt_responsibility/`。

以下历史执行顺序仅用于解释此前实验脉络。

### 第一轮：APCC high-keep 修正实验已完成

结论：high-keep APCC 明显提升稳定性，但没有稳定超过固定 PCC 内部强基准。因此 APCC 暂不作为默认主方法，只作为 stability ablation 和后续属性分组软重置的候选。

当前主线回到：

```bash
PCC_KEEP=0.65
```

后续新模块必须叠加在固定 PCC 内部强基准上比较。

### 第二轮：validation-based checkpoint selection

当前已实现 validation-based checkpoint selection。`eval_curve.txt` 会同时记录 `val/val_fast` 与 `test/train`；`eval_best_from_curve.py` 在新脚本中使用 `--select-splits val,val_fast --selection-mode curve` 选择 checkpoint。当前 `test/test_fast` 只保留为工程诊断和可视化参考。

目标：

1. 从训练视角中固定抽取 validation cameras。
2. 用 validation 曲线选择 checkpoint。
3. 最终只在 test cameras 上离线评估一次。
4. baseline、PCC、Step 4 使用完全相同的 selection protocol。

Step 8 严格 validation top-1 摘要已经完成。固定 PCC=0.65 在反射控制场景上足够稳定，但对 `chair/hotdog/ship/RefReal` 没有系统收益。因此它现在冻结为后续内部强基准，不再继续扫 PCC/APCC。

### 第三轮：GIP-1 / MGIP 最小实现

完成 validation selection 后，进入 GIP-1 grid irradiance probes。理由是 APCC/OAF/GIP-0 都更多是在稳定训练或做 per-Gaussian 低频代理；真正能回应“漫反射区域不应污染高频 envmap、真实大场景需要空间低频照明”的核心创新，仍然是 MGIP 的空间 probe 场。

GIP-1 首版已经实现。下一步运行：

```bash
bash train_step_5.sh
```

重点场景：

- Diffuse/weak-reflective：`chair / materials / ship / mic`
- Real：`gardenspheres / toycar / sedan`
- Reflective controls：`bell / toaster / teapot`

判断标准：

- NeRF Synthetic 和 Ref-Real 的 PSNR/LPIPS 是否提升。
- envmap 是否比 baseline 更干净。
- `probe_map` 是否平滑、低频、空间合理，`probe_responsibility` 是否主要落在低反射/粗糙区域。
- 反射控制场景是否没有被 probe 过度吸收高光。

### 第四轮：决定是否实现 DDGI-lite

若 GIP-1 提升真实场景但出现漏光或遮挡错误，则实现 GIP-3 visibility。若 GIP-1 已经稳定提升，则 DDGI-lite 可作为论文增强模块，而不是第一版必须模块。

---

## 6. 数据回传需求

每个 step 跑完后，优先回传对应 output 根目录下：

1. `summary.txt`
2. `training_regression_summary.txt`
3. `training_regression_drop.svg`

如果某个场景异常，再补该场景文件夹中的：

1. `eval_curve.svg`
2. `metric.txt`
3. `env1.png`
4. `env2.png`

GIP-1 实现后还需要新增回传：

1. 训练输出 `visualize/*.png` 拼图，即其中新增的 `probe_map` 面板。
2. `test/renders/probe/` 中的 probe irradiance 图。
3. `test/renders/probe_responsibility/` 漫反射责任图。
4. `test/renders/specular_reliability/` 镜面可靠性图。
5. 若某场景异常，再补该场景的 `env1.png / env2.png`，检查 envmap 是否仍被漫反射污染。

---

## 7. 下一轮决策规则

1. **PCC 若稳定提升**：实现自适应 PCC，避免固定场景手调。
2. **R2SF/env 若改善 envmap 但不涨 PSNR**：说明远场镜面分解方向正确，但指标瓶颈可能在 diffuse、geometry 或真实外观残差。
3. **GIP-0 若有效且不伤害反射控制**：实现 GIP-1，并把 MGIP 作为低频漫反射创新。
4. **GIP-0 若只少量有效或不稳定**：暂缓 GIP-1，优先强化 `q_diff/q_spec` 责任门控和 appearance/exposure residual。
5. **真实场景若仍低**：不要继续堆 GI，优先检查曝光/白平衡、模糊、遮挡、细结构几何、抗锯齿和评价分辨率。
6. **GIP-1 若提升 Ref-Real**：继续实现 adaptive probes 或 DDGI-lite。
7. **CGI 若继续不稳定**：从主方法中移除，只保留 ray-traced local transport 的可靠版本。

---

## 8. 最终论文消融规划

正式论文实验应整理为如下消融：

| 设置 | 目的 | 预期观察 |
| --- | --- | --- |
| `w/o PCC` | 验证阶段切换稳定性 | 18k-20k 或后期更容易回退 |
| `w/o R2SF` | 验证远场镜面可靠监督 | 非反射/真实场景 envmap 噪声增强 |
| `w/o MGIP` | 验证 probe diffuse GI | NeRF Synthetic 与 Ref-Real 指标下降 |
| `w/o appearance fallback` | 验证低可观测区域安全回退 | 真实低反射场景 PSNR/LPIPS 下降 |
| `w/o probe smoothness` | 验证 probe 低频约束 | probe 出现局部彩色噪声 |
| `w/o local transport` | 验证近场反射 | 高反射局部区域反射不准 |
| `w/o DDGI visibility` | 验证 probe 可见性 | 大场景或遮挡区域漏光 |
| `global envmap only` | 验证光照三分解必要性 | envmap 同时吸收背景、物体反射和漫反射残差 |

关键可视化：

1. 高频环境贴图 `env1.png / env2.png`。
2. Probe irradiance volume / probe energy。
3. `q_diff`、`q_spec`、local transport gate。
4. 法向、粗糙度、反射强度。
5. 训练曲线与开倒车统计。
