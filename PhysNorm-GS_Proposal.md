# PhysNorm-GS：面向反射与非反射场景平衡建模的物理感知法向量高斯溅射

> **研究定位**：以 Ref-Gaussian 的 deferred physically based rendering 为基础，引入 Normal-GS 的 normal-involved illumination 思想，缓解 3D Gaussian Splatting 在反射区域与非反射区域之间的法向量约束不平衡问题。

---

## 一、问题分析与核心动机

3D Gaussian Splatting 在新视角合成中具有高效渲染优势，但在具有复杂材质的场景中仍然面临一个关键问题：**不同材质区域获得的法向量约束强度并不均衡**。

对于高金属度、低粗糙度的反射区域，外观对法向量非常敏感。Ref-Gaussian 通过 deferred physically based rendering、split-sum 近似、环境光照建模以及 Gaussian-grounded inter-reflection，使镜面区域能够通过反射方向和可见性获得较强的 photometric normal gradient。

然而，对于非金属或漫反射主导区域，颜色对法向量的显式依赖较弱，法向量优化更多依赖深度-法向一致性、平滑正则和几何传播。换言之，Ref-Gaussian 在反射建模上很强，但其法向量梯度结构天然更偏向镜面区域，漫反射区域缺少类似 Normal-GS 那样直接的 normal-color coupling。

Normal-GS 的核心贡献是将漫反射颜色重参数化为 albedo 与 normal-illumination dot product，使 RGB 重建损失能够直接约束法向量。但 Normal-GS 主要在逐高斯或前向着色层面引入 IDIV，缺少 Ref-Gaussian deferred rendering 所带来的像素级材质聚合、PBR 分解和反射互反射建模能力。

因此，本文的核心问题是：

> 如何在 Ref-Gaussian 的 deferred PBR 框架中引入 Normal-GS 的 IDIV 机制，使非反射/漫反射区域也获得显式法向量梯度，同时保留 Ref-Gaussian 对反射区域的建模能力？

本文的核心思想不是简单拼接 Normal-GS 与 Ref-Gaussian，而是将 IDIV 改造成 deferred rendering 中的材质感知 G-buffer 属性，使漫反射区域与镜面区域分别通过适合自身材质的物理路径约束法向量。

---

## 二、相关方法的互补性

### 2.1 Normal-GS 的优势与局限

Normal-GS 提出 normal-involved rendering，将颜色表示为法向量与 Integrated Directional Illumination Vector, IDIV 的点积形式：

$$
C_D = \Lambda \cdot \max(0, \mathbf{n} \cdot \mathbf{l}_D)
$$

其中 $\Lambda$ 表示漫反射率，$\mathbf{n}$ 表示法向量，$\mathbf{l}_D$ 表示局部综合入射光照方向向量。该表示使 RGB 重建损失可以直接向法向量传播梯度，从而提升表面法向量质量。

但 Normal-GS 的局限在于，它主要解决漫反射或一般外观下的 normal-color coupling，并没有完整处理高反射材质中的环境反射、粗糙度、金属度和互反射问题。对于复杂反射场景，仅依靠 IDIV 难以表达高频镜面外观。

### 2.2 Ref-Gaussian 的优势与局限

Ref-Gaussian 以 2D Gaussian Splatting 为基础，通过 deferred rendering 先 alpha-blending 得到像素级 G-buffer，再执行 physically based shading。其核心优势包括：

1. 像素级 deferred PBR，减少逐高斯着色造成的噪声。
2. 使用 metallic、roughness、albedo、normal 等材质属性进行物理分解。
3. 通过 split-sum 近似建模镜面反射。
4. 通过 ray tracing 与 TSDF/BVH 支持 Gaussian-grounded inter-reflection。
5. 使用 material-aware normal propagation 增强反射区域的几何一致性。

但 Ref-Gaussian 的法向量主要通过几何正则、平滑项、反射方向和材质传播间接优化。对于非金属漫反射区域，RGB 损失对法向量的显式约束较弱。因此，它在反射区域和非反射区域之间可能存在法向量优化强度不均衡的问题。

### 2.3 本文的互补性切入点

Normal-GS 擅长为漫反射区域提供显式法向量梯度；Ref-Gaussian 擅长为反射区域提供物理镜面建模和互反射建模。本文以 Ref-Gaussian 为基础，将 Normal-GS 的 IDIV 作为 deferred G-buffer 属性引入，从而形成一个材质感知的统一法向量优化框架。

---

## 三、核心方法：Deferred Material-Aware IDIV

### 3.1 高斯属性定义

对每个 2D Gaussian $G_i$，维护或解码以下属性：

$$
\{ \Lambda_i, m_i, r_i, \mathbf{n}_i, \mathbf{l}_{D,i} \}
$$

其中：

| 符号 | 含义 |
|---|---|
| $\Lambda_i$ | diffuse albedo |
| $m_i$ | metallic |
| $r_i$ | roughness |
| $\mathbf{n}_i$ | Gaussian normal |
| $\mathbf{l}_{D,i}$ | diffuse IDIV |

其中 $\mathbf{l}_{D,i}$ 表示局部入射光照的一阶方向性统计：

$$
\mathbf{l}_{D,i}
\approx
\int_{\Omega^+} L_i(\omega_i)\,\omega_i\,d\omega_i
$$

该项继承 Normal-GS 的思想，用于将漫反射颜色与法向量显式耦合。

### 3.2 IDIV 作为 deferred G-buffer 属性

Ref-Gaussian 的 deferred rendering 首先通过 alpha-blending 得到像素级属性。本文将 IDIV 也作为 G-buffer 属性进行 alpha-blending：

$$
\bar{\mathbf{l}}_D
=
\sum_i T_i \alpha_i \mathbf{l}_{D,i}
$$

$$
\bar{\mathbf{N}}
=
\operatorname{normalize}
\left(
\sum_i T_i \alpha_i \mathbf{n}_i
\right)
$$

其中：

$$
T_i
=
\prod_{j<i}(1-\alpha_j)
$$

类似地，可以得到像素级材质属性：

$$
\bar{\Lambda},\quad \bar{M},\quad \bar{R}
$$

本文不在每个 Gaussian 上直接执行 IDIV 着色，而是在 alpha-blending 后的像素级 deferred shading 阶段计算 diffuse-IDIV 贡献：

$$
L_D^{\mathrm{IDIV}}
=
(1-\bar{M}) \cdot \bar{\Lambda} \cdot
\max
\left(
0,
\bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D
\right)
$$

该设计的作用是：让 IDIV 继承 Ref-Gaussian deferred rendering 的平滑优势，避免逐高斯着色中由深度排序、透明混合和局部几何噪声导致的不稳定梯度。

需要注意的是，$\bar{\mathbf{N}}\cdot\bar{\mathbf{l}}_D$ 是 deferred rendering 下的局部一致表面近似。它并不严格等价于先计算每个 Gaussian 的 $\mathbf{n}_i\cdot\mathbf{l}_{D,i}$ 再进行 alpha-blending。本文采用该近似，是因为 Ref-Gaussian 本身也将材质属性先聚合到像素级，再执行 PBR shading。该近似在局部表面一致、前景 Gaussian 占主导的情况下是合理的，并能带来更平滑的优化信号。

### 3.3 材质感知 diffuse gating

本文使用 metallic 对 diffuse-IDIV 项进行调制：

$$
L_D^{\mathrm{IDIV}}
=
(1-\bar{M}) \cdot \bar{\Lambda} \cdot
\max
\left(
0,
\bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D
\right)
$$

这里的 $(1-\bar{M})$ 不是硬切换，而是材质感知的 diffuse gating：

- 当 $\bar{M}\to 0$ 时，区域以非金属漫反射为主，IDIV 提供强 normal-color coupling。
- 当 $\bar{M}\to 1$ 时，diffuse-IDIV 贡献自然减弱，外观主要由 Ref-Gaussian 的 specular PBR 项解释。
- 当材质处于中间状态时，diffuse-IDIV 与 specular-BRDF 共同参与优化。

本文不强制高金属区域的 $\mathbf{l}_D$ 本身趋近于零，因为最终颜色贡献已经由 $(1-\bar{M})$ 控制。强行约束 IDIV 为零可能导致梯度饥饿，并削弱局部光照表达能力。

---

## 四、镜面与互反射建模

本文保留 Ref-Gaussian 的镜面建模方式，不额外引入 Ref-NeRF IDE 作为主分支。

镜面项写为：

$$
L_S^{\mathrm{RefG}}
=
L_S^{\mathrm{split\text{-}sum}}
(
\bar{\mathbf{N}},
\bar{R},
\bar{M},
\omega_o,
E
)
$$

其中反射方向为：

$$
\omega_r
=
2(\omega_o \cdot \bar{\mathbf{N}})\bar{\mathbf{N}}
-
\omega_o
$$

该项通过 $\bar{\mathbf{N}}$ 显式依赖法向量，因此高反射区域仍然能够从镜面重建误差中获得强法向量梯度。

互反射项沿用 Ref-Gaussian 的 Gaussian-grounded inter-reflection：

$$
L_{\mathrm{ind}}^{\mathrm{RefG}}
$$

该项通过周期性 TSDF 网格提取、BVH 加速和 ray tracing visibility 建模间接反射。本文不重新设计 indirect illumination vector，因为 Ref-Gaussian 已经提供了与其 PBR 管线一致的互反射建模方式。额外引入新的 IIV 分支可能与已有 indirect light 表达竞争，削弱材质与光照分解的可解释性。

---

## 五、完整渲染公式

最终像素颜色为：

$$
L_{\mathrm{out}}
=
L_D^{\mathrm{IDIV}}
+
L_S^{\mathrm{RefG}}
+
L_{\mathrm{ind}}^{\mathrm{RefG}}
$$

其中：

$$
L_D^{\mathrm{IDIV}}
=
(1-\bar{M}) \cdot \bar{\Lambda} \cdot
\max
\left(
0,
\bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D
\right)
$$

$$
L_S^{\mathrm{RefG}}
=
L_S^{\mathrm{split\text{-}sum}}
(
\bar{\mathbf{N}},
\bar{R},
\bar{M},
\omega_o,
E
)
$$

$$
L_{\mathrm{ind}}^{\mathrm{RefG}}
=
\text{Ref-Gaussian inter-reflection term}
$$

该公式的核心含义是：非金属漫反射区域主要通过 IDIV 获得显式法向量约束，金属镜面区域主要通过 PBR specular 和 inter-reflection 获得法向量约束。两类区域通过 metallic gating 连续过渡，从而缓解反射与非反射物体之间的优化不平衡。

---

## 六、梯度分析

对于 diffuse-IDIV 项：

$$
L_D^{\mathrm{IDIV}}
=
(1-\bar{M})\bar{\Lambda}
\max
\left(
0,
\bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D
\right)
$$

当 $\bar{\mathbf{N}}\cdot\bar{\mathbf{l}}_D > 0$ 时：

$$
\frac{\partial L_D^{\mathrm{IDIV}}}{\partial \bar{\mathbf{N}}}
=
(1-\bar{M})\bar{\Lambda}\bar{\mathbf{l}}_D
$$

因此 RGB reconstruction loss 可以通过 diffuse-IDIV 项直接向像素级法向量传播梯度。

再由 alpha-blending 链式传播到单个 Gaussian：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{n}_i}
=
\frac{\partial \mathcal{L}}{\partial \bar{\mathbf{N}}}
\frac{\partial \bar{\mathbf{N}}}{\partial \mathbf{n}_i}
$$

这说明本文方法相比 Ref-Gaussian，在漫反射区域额外提供了一条显式 photometric normal gradient 路径；相比 Normal-GS，该路径又经过 deferred alpha-blending 聚合，具有更平滑的像素级优化性质。

对于镜面区域，法向量仍然通过：

$$
\omega_r
=
2(\omega_o \cdot \bar{\mathbf{N}})\bar{\mathbf{N}}
-
\omega_o
$$

影响 split-sum specular term。因此，本文的法向量梯度由三部分组成：

$$
\frac{\partial \mathcal{L}}{\partial \bar{\mathbf{N}}}
=
\frac{\partial \mathcal{L}}{\partial L_D^{\mathrm{IDIV}}}
\frac{\partial L_D^{\mathrm{IDIV}}}{\partial \bar{\mathbf{N}}}
+
\frac{\partial \mathcal{L}}{\partial L_S^{\mathrm{RefG}}}
\frac{\partial L_S^{\mathrm{RefG}}}{\partial \bar{\mathbf{N}}}
+
\frac{\partial \mathcal{L}}{\partial L_{\mathrm{ind}}^{\mathrm{RefG}}}
\frac{\partial L_{\mathrm{ind}}^{\mathrm{RefG}}}{\partial \bar{\mathbf{N}}}
$$

这构成本文的核心机制：漫反射区域由 IDIV 主导法向量约束，镜面区域由 PBR reflection 主导法向量约束。

---

## 七、IDIV 参数化

为保持与 Normal-GS 和 Scaffold-GS 思路一致，本文可以使用 anchor-based MLP 解码 IDIV：

$$
\mathbf{l}_{D,i}
=
\theta_l
(
\mathbf{f}_v,
\Delta \mathbf{x}_i,
\mathbf{d}
)
$$

其中 $\mathbf{f}_v$ 是 anchor feature，$\Delta \mathbf{x}_i$ 是 Gaussian 相对 anchor 的局部偏移，$\mathbf{d}$ 是 view direction 或其他已有视角编码。

也可以采用更保守的实现：直接为每个 Gaussian 维护一个可学习的 3D IDIV 向量，并加入轻量平滑约束。若以工程可实现性为优先，建议首先采用 per-Gaussian IDIV，验证核心假设成立后再切换到 anchor-MLP。

本文不建议将 metallic 直接作为 IDIV MLP 的强条件并要求高金属区域输出零向量。更合理的做法是让 metallic 只在最终颜色贡献中 gating IDIV，从而避免过强先验损害优化。

---

## 八、损失函数

整体损失函数为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rgb}}
+
\lambda_N\mathcal{L}_N
+
\lambda_{\mathrm{sm}}\mathcal{L}_{\mathrm{sm}}
+
\lambda_{\mathrm{vol}}\mathcal{L}_{\mathrm{vol}}
+
\lambda_{\mathrm{idiv}}\mathcal{L}_{\mathrm{idiv}}
$$

其中：

$$
\mathcal{L}_{\mathrm{rgb}}
=
(1-\lambda)L_1
+
\lambda L_{\mathrm{D\text{-}SSIM}}
$$

$$
\mathcal{L}_N
=
1-\tilde{\mathbf{N}}^T\bar{\mathbf{N}}
$$

$$
\mathcal{L}_{\mathrm{sm}}
=
\|\nabla\bar{\mathbf{N}}\|
\exp
\left(
-\|\nabla C_{\mathrm{gt}}\|
\right)
$$

$$
\mathcal{L}_{\mathrm{vol}}
=
\prod(\mathbf{s})
$$

IDIV 平滑损失可以写为：

$$
\mathcal{L}_{\mathrm{idiv}}
=
\sum_{(i,j)\in \mathcal{N}}
w_{ij}
\left\|
\mathbf{l}_{D,i}
-
\mathbf{l}_{D,j}
\right\|_1
$$

其中 $\mathcal{N}$ 表示空间邻域或 anchor 内邻域，$w_{ij}$ 可由距离或颜色边缘控制。该损失用于抑制 IDIV 过拟合局部噪声，但权重应保持较小，避免过度平滑真实光照变化。

推荐初始权重：

$$
\lambda_N = 0.05,\quad
\lambda_{\mathrm{sm}} = 1.0,\quad
\lambda_{\mathrm{vol}} = 0.001,\quad
\lambda_{\mathrm{idiv}} = 0.005
$$

---

## 九、训练策略

### 9.1 两阶段训练

本文沿用 Ref-Gaussian 的两阶段训练思想。

第一阶段：几何与基础材质初始化。

- 优化 Gaussian 几何、opacity、scale、rotation。
- 初始化 albedo、metallic、roughness、normal。
- 可同时开启弱 IDIV 项，但降低其权重，避免早期几何不稳定时 IDIV 吸收错误外观。

第二阶段：deferred material-aware IDIV 精化。

- 开启完整的 $L_D^{\mathrm{IDIV}} + L_S^{\mathrm{RefG}} + L_{\mathrm{ind}}^{\mathrm{RefG}}$ 渲染。
- 加强 normal consistency 与 IDIV 平滑。
- 周期性执行 TSDF mesh extraction 与 BVH update，用于 Ref-Gaussian inter-reflection。

### 9.2 几何与法向量传播

本文保留 Ref-Gaussian 的 material-aware normal propagation，用于增强高反射区域的法向量一致性。但不再引入“高金属区域 IDIV 置零”的硬规则。

更合理的策略是：

- 高金属低粗糙度区域：主要依赖 specular PBR 与 normal propagation。
- 低金属漫反射区域：主要依赖 diffuse-IDIV 与 normal consistency。
- 中间材质区域：两条路径共同优化。

---

## 十、关于不采用 IDE 与近远景插值的说明

### 10.1 不采用 Ref-NeRF IDE 主分支

Ref-NeRF IDE 适合在神经辐射场中编码粗糙度相关的反射方向，但 Ref-Gaussian 已经使用 split-sum PBR、prefiltered environment map 和 BRDF LUT 表达镜面项。

若再加入：

$$
L_S
=
\gamma L_S^{\mathrm{split\text{-}sum}}
+
(1-\gamma)L_S^{\mathrm{IDE}}
$$

会导致两个问题：

1. IDE 神经分支与物理镜面项竞争解释高频外观。
2. 材质、环境光和 residual appearance 的可分解性变差。

因此，本文不将 IDE 作为核心模块。若实验中确实需要，可作为极小权重 residual 分支放入附录消融，而不是主方法。

### 10.2 不采用全局近远景 $\beta$ 插值

原始近远景 IDIV-EnvMap 分解：

$$
\mathbf{l}_D
=
\beta \mathbf{l}_D^{\mathrm{near}}
+
(1-\beta)\mathbf{l}_D^{\mathrm{far}}
$$

不建议作为核心创新。原因是单个全局 $\beta$ 无法可靠表达像素级、空间位置级和材质级的近远光照变化，也无法处理遮挡和局部互反射。同时，Ref-Gaussian 已经具有环境光和 inter-reflection 建模。

本文将研究范围限定在 Ref-Gaussian 已覆盖的 reflective、glossy 和 general object-level scenes。Normal-GS 在大尺度室外远景中的退化问题可以作为未来工作，通过 spatially varying environment map 或 per-anchor illumination residual 进一步研究。

---

## 十一、实验设计

### 11.1 数据集

| 数据集 | 目的 |
|---|---|
| Shiny Blender Synthetic | 评估镜面反射重建 |
| Shiny Blender Real | 评估真实反射场景 |
| Glossy Synthetic | 评估光泽材质与法向量 |
| NeRF Synthetic | 评估非反射/一般物体表现 |
| Ref-Real dataset | 评估真实复杂反射 |

Mip-NeRF 360 可作为补充实验，但不应作为本文核心验证目标，因为本文不主张解决大尺度远景环境光分解问题。

### 11.2 对比方法

| 方法 | 目的 |
|---|---|
| 3DGS | 基础 Gaussian baseline |
| 2DGS | 几何表面 baseline |
| Normal-GS | normal-involved rendering baseline |
| GaussianShader | PBR Gaussian baseline |
| 3DGS-DR | deferred rendering baseline |
| Ref-Gaussian | 核心对比方法 |
| PhysNorm-GS | 本文方法 |

### 11.3 评估指标

| 指标 | 说明 |
|---|---|
| PSNR / SSIM / LPIPS | 新视角合成质量 |
| Normal MAE | 法向量精度，主要在有 GT normal 的合成数据上评估 |
| FPS | 渲染效率 |
| Training Time | 训练成本 |
| Material / Relighting metrics | 若沿用 Ref-Gaussian 协议，可评估材质分解与重光照质量 |

### 11.4 消融实验

| 变体 | 去除内容 | 预期现象 |
|---|---|---|
| Full | 完整方法 | 最佳综合表现 |
| w/o IDIV | 移除 deferred IDIV | 漫反射区域法向量质量下降 |
| w/o metallic gating | 去除 $(1-\bar{M})$ | 金属区域出现 diffuse 串扰 |
| Gaussian-level IDIV | 逐高斯 IDIV 着色 | 法向量和颜色更噪 |
| w/o IDIV smooth | 移除 IDIV 平滑 | IDIV 过拟合局部噪声 |
| w/o normal propagation | 移除 Ref-Gaussian normal propagation | 高反射区域法向量退化 |

---

## 十二、核心假设与可证伪预测

本文的核心假设是：

> 在 Ref-Gaussian 的 deferred PBR 框架中加入 material-aware IDIV，可以增强漫反射区域的 photometric normal gradient，从而缓解反射与非反射区域之间的法向量优化不平衡。

可证伪预测包括：

1. 在 NeRF Synthetic 或 Glossy Synthetic 的非金属区域，PhysNorm-GS 的 normal MAE 应优于 Ref-Gaussian。
2. 在 Shiny Blender 等反射场景，PhysNorm-GS 的 PSNR/SSIM/LPIPS 应接近或优于 Ref-Gaussian，不能因加入 IDIV 损害镜面表现。
3. 去除 metallic gating 后，高金属区域应出现 diffuse-IDIV 串扰，表现为反射模糊或材质分解变差。
4. 将 deferred IDIV 改为 Gaussian-level IDIV 后，法向量和颜色应更容易出现局部噪声。
5. 相比 Normal-GS，PhysNorm-GS 应在反射场景上明显更强；相比 Ref-Gaussian，PhysNorm-GS 应在非反射或混合材质区域提供更好的法向量质量。

---

## 十三、理论创新总结

本文的核心创新可以概括为三点：

1. **Deferred Material-Aware IDIV**  
   将 Normal-GS 的 IDIV 从前向逐高斯着色改造成 Ref-Gaussian deferred rendering 中的 G-buffer 属性，使漫反射区域也能获得显式 normal-color coupling。

2. **反射/非反射区域的法向量梯度平衡**  
   使用 metallic gating 在 diffuse-IDIV 与 specular-PBR 之间连续分配颜色解释能力，使非金属区域由 IDIV 提供法向量梯度，高反射区域由 Ref-Gaussian 的镜面反射路径提供法向量梯度。

3. **保持 Ref-Gaussian 的物理可分解性**  
   本文不引入 IDE 主分支、额外 IIV 分支或全局近远景插值，避免神经残差与物理材质项竞争解释外观，从而保留 Ref-Gaussian 在反射建模、互反射、重光照和材质编辑方面的优势。

---

## 十四、符号表

| 符号 | 含义 |
|---|---|
| $\Lambda_i, \bar{\Lambda}$ | Gaussian / pixel diffuse albedo |
| $m_i, \bar{M}$ | Gaussian / pixel metallic |
| $r_i, \bar{R}$ | Gaussian / pixel roughness |
| $\mathbf{n}_i, \bar{\mathbf{N}}$ | Gaussian / pixel normal |
| $\mathbf{l}_{D,i}, \bar{\mathbf{l}}_D$ | Gaussian / pixel IDIV |
| $\omega_o$ | view direction |
| $\omega_r$ | reflection direction |
| $E$ | environment map |
| $L_D^{\mathrm{IDIV}}$ | diffuse IDIV term |
| $L_S^{\mathrm{RefG}}$ | Ref-Gaussian specular term |
| $L_{\mathrm{ind}}^{\mathrm{RefG}}$ | Ref-Gaussian inter-reflection term |
