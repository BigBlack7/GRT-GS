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

---

## 3. 工程实现路线图

### 3.1 已实现并可继续实验

- [x] Ref-Gaussian 主渲染链路复现。
- [x] PCC soft reset 首版。
- [x] R2SF/R2IF env-light-only gradient gate 与 env regularization 首版。
- [x] Per-Gaussian NCIF residual，即 GIP-0 低频辐照代理。
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

1. 新增 `scene/probe_field.py` 或等价模块，维护规则 grid probes。
2. 每个 probe 存低阶 SH 辐照系数，首版建议 `SH order = 2`，RGB 共 `9 * 3` 维。
3. 根据 scene bbox 初始化 probe grid，支持 `probe_grid_res` 或 `probe_spacing`。
4. 渲染时对像素/Gaussian 位置做 trilinear interpolation，计算 `I_probe(x, n)`。
5. 使用 `q_diff=(1-reflectance)^mu * roughness^nu` 控制 probe 对 diffuse 的贡献。
6. 加入 probe smoothness、energy、magnitude/non-negative 正则。
7. 支持保存/加载 probe 参数和导出 probe irradiance 可视化。

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

### 3.4 第二阶段候选：GIP-2 自适应探针

目标：提升大场景效率和局部细节表达。

代码任务：

1. 根据 Gaussian 密度或 visibility 生成 adaptive probe anchors。
2. 对稀疏区域降低 probe 密度，对物体密集区域提高 probe 密度。
3. 支持 kNN probe interpolation 替代规则 grid interpolation。
4. 记录每个 probe 的可见图像数量和局部 Gaussian 覆盖率。

### 3.5 第三阶段候选：GIP-3 / DDGI-lite 可见性

目标：利用已有 ray tracing 能力解决 probe 漏光和遮挡错误。

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

## 4. 当前训练脚本状态

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

状态：训练模板已准备，但必须先完成 GIP-1 代码实现和参数注册后才能运行。脚本会在检测不到 `--use_probe_gi` 参数时主动退出，避免误跑旧逻辑。默认输出名已改为 `step5_oah_gip1_probe_*`。

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

---

## 5. 接下来执行顺序

### 第一轮：PCC 基准上的 OAF 增量实验

当前不再优先回头审计 Ref-Gaussian，而是直接检验新模块是否能超过 PCC 内部强基准。先运行：

```bash
bash train_step_6_oaf.sh
```

重点比较对象：

- PCC-only 的对应场景 best checkpoint。
- Step 4 safe 的对应场景 best checkpoint。
- Ref-Gaussian 30k 只作为下限参考。

回传：

```bash
summary.txt
training_regression_summary.txt
training_regression_drop.svg
best_iteration_eval_summary.txt
best_iteration_eval_summary.json
```

合格线：

- OAF 应在 `chair/materials/ship/real` 中至少部分超过 PCC 或 Step 4 safe。
- OAF 不能显著伤害 `bell/teapot/toaster`。
- 如果 OAF 对真实场景 PSNR 提升但 LPIPS 变差，需要看图决定是否是过平滑或曝光偏移。

### 第二轮：validation-based checkpoint selection

如果 OAF 或 Step 4 中任一模块确认超过 PCC，必须实现 validation-based checkpoint selection。当前 `eval_curve.txt` 使用 test/test_fast 做选择，只能用于工程诊断，不能作为正式论文最终数字。

目标：

1. 从训练视角中固定抽取 validation cameras。
2. 用 validation 曲线选择 checkpoint。
3. 最终只在 test cameras 上离线评估一次。
4. baseline、PCC、Step 4 使用完全相同的 selection protocol。

### 第三轮：决定下一项实现

根据 OAF 相对 PCC 的结果分支决策：

1. **OAF 超过 PCC 且不伤害反射控制。** 将 OAF 纳入 OAHD 主线，并继续实现 validation-based selection。
2. **OAF 只改善真实场景但伤害合成场景。** 将 OAF 限定为 Ref-Real/低可观测区域模块，增强门控或降低 `oaf_tau`。
3. **OAF 与 PCC 接近。** 说明当前瓶颈不是低可观测外观回退，转向 GIP-1 grid irradiance probes。
4. **OAF 明显低于 PCC。** 放弃 OAF 主线，只保留为 ablation，优先做 adaptive PCC / early stopping。
5. **真实场景 LPIPS 变差。** 优先看图判断是否过平滑、曝光偏移或背景污染，不急着堆 DDGI-lite。

实现 GIP-1 后运行：

```bash
bash train_step_5.sh
```

重点场景：

- Diffuse/weak-reflective：`chair / materials / ship`
- Real：`gardenspheres / toycar / sedan`
- Reflective controls：`bell / toaster / teapot`

判断标准：

- NeRF Synthetic 和 Ref-Real 的 PSNR/LPIPS 是否提升。
- envmap 是否比 baseline 更干净。
- probe irradiance 是否平滑、低频、空间合理。
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

1. probe irradiance 可视化。
2. `q_diff` 漫反射责任图。
3. `q_spec` 镜面责任图。
4. probe energy 统计。

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
