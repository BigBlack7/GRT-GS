# OAH-GS 实验记录与阶段计划

本文档只记录工程落地、训练脚本、阶段实验计划和阶段结论。理论方法、公式推导和论文叙事统一放在 `PhysNorm-GS_Proposal.md`。

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

1. 运行 `train_step_2_fix.sh`，验证 R2SF env-light-only gate。
2. 若 Step 2 fix 合格，运行 `train_step_4_fix.sh`，验证 GIP-0 是否在 diffuse 场景提供低频收益。
3. 如果 Step 2 fix 仍压低 `bell/toaster`，暂停所有 GI/probe 扩展，优先调 R2SF 门控下界、env 正则和 PCC。
4. 如果 Step 4 fix 只提升 diffuse 但伤害 reflective control，则 GIP-0/1 必须加入更强的 `q_diff` 责任约束。

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
/data2/zmh/output_physnorm_steps/step4_oah_gip0_envgate_30000_pcc065_mingate015
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

---

## 5. 接下来执行顺序

### 第一轮：OAH 安全性小规模复测

先不要再跑全量 Step 2/4。第一次 `gradfix` 已证明“门控整个镜面分支”会压制材质学习。现在优先验证第二次修正：只门控远场环境贴图光照梯度，保持 BRDF/material 分支正常学习。

```bash
bash train_step_2_fix.sh
```

本次输出目录应为：

```bash
/data2/zmh/output_physnorm_steps/step2_r2sf_envgate_30000_pcc065_mingate015
```

合格线：

- `bell/toaster` 不能继续低于第一次 `gradfix`，最好恢复到 Step 1/PCC 附近。
- `chair/materials/mic` 不能出现新的明显最终回退。
- envmap 不能比 baseline 更噪。

如果以上条件满足，再跑：

```bash
bash train_step_4_fix.sh
```

输出目录应为：

```bash
/data2/zmh/output_physnorm_steps/step4_gip0_envgate_30000_pcc065_mingate015
```

### 第二轮：全量 OAH-GS Step 2/4

若两个小规模复测稳定，再重跑完整 Step 2/4。脚本已经更新为 `envgate + min_gate=0.15`，不会回到旧参数：

```bash
bash train_step_2.sh
bash train_step_4.sh
```

默认输出：

```bash
/data2/zmh/output_physnorm_steps/step2_oah_r2sf_envgate_30000_pcc065_mingate015
/data2/zmh/output_physnorm_steps/step4_oah_gip0_envgate_30000_pcc065_mingate015
```

目的：

- 判断“PCC + 远场 envmap 可靠性 + per-Gaussian 低频辐照代理”是否已经形成稳定趋势。
- 按场景分组看结果：高反射单物体、低反射单物体、真实反射、真实低反射分别统计。
- 如果 GIP-0 只对少数 diffuse 场景有效但不稳定，先不急着实现完整 DDGI，而是补强责任门控或外观残差。

### 第三轮：决定下一项实现

根据全量 Step 2/4 结果分支决策：

1. **Step 2 合格，Step 4 明显提升 diffuse/real。** 进入 GIP-1，实现真正的 learnable grid irradiance probes。
2. **Step 2 合格，Step 4 无明显收益。** 暂缓 GIP-1，优先实现 appearance/exposure residual 或 adaptive PCC，因为真实场景误差可能主要不是 GI。
3. **Step 2 仍伤害反射场景。** 暂停 probe/GI，回到 R2SF 门控、env 正则和远场/近场拆分。
4. **Step 4 提升 diffuse 但伤害反射控制。** 增强 `q_diff`，让低频分支更严格只作用于漫反射/粗糙区域。

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
