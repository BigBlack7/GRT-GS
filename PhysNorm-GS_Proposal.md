# PhysNorm-GS：物理感知法向量集成的延迟渲染高斯溅射

> **研究定位**：结合 Normal-GS 与 Ref-Gaussian 的互补优势，提出统一的物理渲染框架，同步解决法向量精度与反射建模的"跷跷板"困境。

---

## 一、问题分析与互补性洞察

### 1.1 两篇论文的核心矛盾

两篇论文的核心矛盾可以归结为同一个根源：**法向量与渲染管线的耦合深度不足**。

**Normal-GS 的突破与局限**

Normal-GS 将法向量显式嵌入前向渲染（IDIV 点积），使梯度信号可以回传给法向量。但它仍然在每个高斯基元级别执行着色，缺乏 Ref-Gaussian 延迟渲染的梯度平滑优势，也没有材质感知的分解能力。室外远景场景下，纯 IDIV 无法捕捉来自无穷远方向分布的环境光，导致法向量自正则效果欠佳。

**Ref-Gaussian 的突破与局限**

Ref-Gaussian 实现了像素级延迟着色与光线追踪互反射。但其法向量梯度仅来源于几何隐式约束，在延迟渲染的 alpha-blending 之后才进入渲染方程，缺乏 Normal-GS 那种法向量在色彩参数化中的显式前向参与。互反射建模中仅考虑反射方向可见性，对镜面叶片内的材质估计存在近似噪声。

### 1.2 两个核心理论空白

**空白 A**：在延迟渲染框架下，如何让 IDIV 也参与 alpha-blending，使法向量的梯度同时受益于延迟渲染的平滑效果？

**空白 B**：IDIV 如何与材质属性（metallic $m$，roughness $r$）联动，在高金属度场景自动退化为镜面主导，在低金属度场景主导漫反射，从而统一反射与非反射场景？

---

## 二、理论框架

### 2.1 统一渲染方程的重新参数化

从 Kajiya 渲染方程出发：

$$
L_{\text{out}}(\omega_o) = \int_{\Omega^+} L_i(\omega_i) \, f(\omega_i, \omega_o) \, (\omega_i \cdot \mathbf{n}) \, d\omega_i
$$

将 BRDF $f$ 按 Disney 模型分解为漫反射与镜面两项，并引入金属度 $m \in [0,1]$ 作为混合权重：

$$
f(\omega_i, \omega_o) = (1 - m) f_D + m \, f_S(\omega_i, \omega_o)
$$

代入渲染方程，漫反射项为：

$$
L_D = (1 - m) k_D \int_{\Omega^+} L_i(\omega_i) \, (\omega_i \cdot \mathbf{n}) \, d\omega_i
$$

**核心推导**——将 $\mathbf{n}$ 从积分中析出（与 Normal-GS 一致）：

$$
L_D = (1 - m) k_D \cdot \mathbf{n} \cdot \underbrace{\int_{\Omega^+} L_i(\omega_i) \, \omega_i \, d\omega_i}_{\mathbf{l}_D \;\text{（IDIV）}}
$$

$$
\boxed{L_D = (1 - m) \cdot k_D \cdot \mathbf{n} \cdot \mathbf{l}_D}
$$

这比 Normal-GS 的原始形式多了 $(1-m)$ 调制项，构成**材质感知 IDIV（MA-IDIV）**的核心：当 $m \to 1$ 时，漫反射项自动消失，完全交由镜面项主导，无需额外的硬切换逻辑。

### 2.2 延迟渲染空间中的 MA-IDIV

Ref-Gaussian 的延迟渲染在 alpha-blending 后得到像素级聚合量：

$$
\bar{X} = \sum_{i=1}^{N} x_i \, \alpha_i \prod_{j < i}(1 - \alpha_j), \quad \bar{X} \in \{\bar{\Lambda},\, \bar{M},\, \bar{R},\, \bar{\mathbf{N}}\}
$$

**提出的关键扩展**：将 IDIV $\mathbf{l}_D$ 作为额外的高斯属性参与 alpha-blending：

$$
\bar{\mathbf{l}}_D = \sum_{i=1}^{N} \mathbf{l}_{D,i} \, \alpha_i \prod_{j < i}(1 - \alpha_j)
$$

像素级漫反射颜色最终在延迟阶段计算：

$$
L_D = (1 - \bar{M}) \cdot \bar{\Lambda} \cdot \bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D
$$

**梯度分析**（理论正确性的关键）：

对聚合法向量 $\bar{\mathbf{N}}$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \bar{\mathbf{N}}} = \frac{\partial \mathcal{L}}{\partial L_D} \cdot (1 - \bar{M}) \bar{\Lambda} \bar{\mathbf{l}}_D + \frac{\partial \mathcal{L}}{\partial L_S} \cdot \frac{\partial L_S}{\partial \bar{\mathbf{N}}}
$$

对每个高斯基元法向量 $\mathbf{n}_i$ 的梯度由链式法则给出：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{n}_i} = \frac{\partial \mathcal{L}}{\partial \bar{\mathbf{N}}} \cdot \alpha_i \prod_{j < i}(1 - \alpha_j)
$$

与 Normal-GS 相比，梯度同时来自漫反射（经 IDIV 项）和镜面（经反射方向 $\omega_r$）两路，且经过 alpha-blending 的平均效应自然平滑。与 Ref-Gaussian 相比，法向量额外获得了来自 $\bar{\mathbf{l}}_D$ 的显式几何引导梯度，而不仅仅依赖深度法向量一致性损失。

### 2.3 MA-IDIV 的锚点-MLP 编码

IDIV 具有局部共享、低频变化的特性（继承自 Normal-GS）。在延迟渲染框架中，对每个锚点 $v$ 存储局部特征 $\mathbf{f}_v$，并用全局 MLP 解码，同时将金属度条件纳入解码器：

$$
\{\mathbf{l}_{D,k}^v\}_{k=1}^K = \theta_l(\mathbf{f}_v,\, m_v)
$$

关键创新是将**金属度条件**纳入 IDIV 解码器：当 $m_v$ 较高时，MLP 自动输出接近零向量的 $\mathbf{l}_D$（漫反射不重要），梯度无需人工截断。此设计使 MLP 在优化过程中自然学习到"高反射面不需要 IDIV"的归纳偏置。

### 2.4 IDIV 引导的间接光照

Ref-Gaussian 当前用球谐函数（SH）为每个高斯基元建模间接光 $\mathbf{l}_{\text{ind}}$，在几何噪声较大时精度受限。定义**间接光照向量（Indirect Illumination Vector, IIV）**：

$$
\mathbf{l}_{\text{ind}} = \theta_{\text{ind}}(\mathbf{f}_v,\, \omega_r,\, \bar{\mathbf{N}})
$$

最终间接光贡献为：

$$
L_{\text{ind}} = \bar{M} \cdot \left(\int_\Omega f_S(\omega_i, \omega_o)(\omega_i \cdot \bar{\mathbf{N}}) \, d\omega_i\right) \cdot \bar{\mathbf{N}} \cdot \mathbf{l}_{\text{ind}} \cdot (1 - V)
$$

这比 Ref-Gaussian 的 SH 参数化更精确，因为显式引入了法向量的几何约束。

### 2.5 自适应局部-全局光照分解

Normal-GS 在远距离室外场景失效，根本原因是 IDIV 只捕捉近场入射光，而环境光来自无穷远的方向分布。提出**自适应 IDIV-EnvMap 分解**，将入射光场分解为近场与远场：

$$
L_i(\omega_i) = L_i^{\text{near}}(\omega_i) + L_i^{\text{far}}(\omega_i)
$$

对应漫反射 IDIV 分解为：

$$
\mathbf{l}_D = \underbrace{\int_{\Omega^+} L_i^{\text{near}}(\omega_i) \omega_i \, d\omega_i}_{\bar{\mathbf{l}}_D^{\text{near}}} + \underbrace{\int_{\Omega^+} L_i^{\text{far}}(\omega_i) \omega_i \, d\omega_i}_{\mathbf{l}_D^{\text{far}}}
$$

其中 $\mathbf{l}_D^{\text{far}}$ 由预积分环境贴图查询得到，$\bar{\mathbf{l}}_D^{\text{near}}$ 仍由锚点 MLP 编码。两部分通过可学习的场景尺度参数 $\beta \in [0,1]$ 加权：

$$
\mathbf{l}_D = \beta \, \bar{\mathbf{l}}_D^{\text{near}} + (1 - \beta) \, \mathbf{l}_D^{\text{far}}
$$

当场景以近景为主（室内、桌面场景）时，优化自动使 $\beta \to 1$；当室外远景为主时，$\beta \to 0$，退化为纯 envmap 漫反射，与 Normal-GS 的室外失效问题彻底解耦。

---

## 三、镜面分量推导

镜面项延续 Ref-Gaussian 的 split-sum 近似，但法向量现在通过两个路径同时获得梯度：

$$
L_S \approx \underbrace{\int_\Omega f_S(\omega_i, \omega_o)(\omega_i \cdot \bar{\mathbf{N}}) \, d\omega_i}_{\text{BRDF lobe（预计算）}} \cdot \underbrace{\int_\Omega L_i(\omega_i) D(\omega_i, \omega_o)(\omega_i \cdot \bar{\mathbf{N}}) \, d\omega_i}_{\text{预滤波 envmap，以 } \omega_r \text{ 和 } \bar{R} \text{ 查询}}
$$

其中反射方向 $\omega_r = 2(\omega_o \cdot \bar{\mathbf{N}})\bar{\mathbf{N}} - \omega_o$ 依赖于聚合法向量 $\bar{\mathbf{N}}$，从而使镜面梯度也能回传至各高斯法向量。

此外，引入 Ref-NeRF 的 IDE（Integrated Directional Encoding）作为辅助编码，将反射方向 $\omega_r$ 与法向量一起编码，增强高频镜面效果的表达能力：

$$
L_S^{\text{IDE}} = \theta_S(\phi_{\text{IDE}}(\omega_r, \bar{R}),\, \bar{\mathbf{N}},\, \mathbf{f}_v)
$$

最终输出取两者的线性组合，由可学习标量 $\gamma$ 调制：

$$
L_S = \gamma \, L_S^{\text{split-sum}} + (1 - \gamma) \, L_S^{\text{IDE}}
$$

---

## 四、完整渲染方程

综合以上所有模块，最终像素颜色为：

$$
\boxed{
L_{\text{out}} = \underbrace{(1 - \bar{M}) \cdot \bar{\Lambda} \cdot \bar{\mathbf{N}} \cdot \bar{\mathbf{l}}_D}_{L_D \text{（MA-IDIV 漫反射）}}
+ \underbrace{\gamma L_S^{\text{split-sum}} + (1-\gamma) L_S^{\text{IDE}}}_{L_S \text{（镜面）}}
+ \underbrace{\bar{M} \cdot (\cdots) \cdot \bar{\mathbf{N}} \cdot \mathbf{l}_{\text{ind}} \cdot (1-V)}_{L_{\text{ind}} \text{（IIV 间接光）}}
}
$$

---

## 五、损失函数

$$
\mathcal{L} = \mathcal{L}_P + \lambda_N \mathcal{L}_N + \lambda_{\text{vol}} \mathcal{L}_{\text{vol}} + \lambda_{\text{sm}} \mathcal{L}_{\text{sm}} + \lambda_{\text{IDIV}} \mathcal{L}_{\text{IDIV}}
$$

| 损失项 | 表达式 | 说明 |
|--------|--------|------|
| $\mathcal{L}_P$ | $(1-\lambda)L_1 + \lambda L_{\text{D-SSIM}}$ | RGB 重建损失，$\lambda=0.2$ |
| $\mathcal{L}_N$ | $1 - \tilde{\mathbf{N}}^T \bar{\mathbf{N}}$ | 深度导出法向量与渲染法向量的余弦一致性 |
| $\mathcal{L}_{\text{vol}}$ | $\prod(\mathbf{s})$ | 体积正则化，抑制高斯基元过度膨胀 |
| $\mathcal{L}_{\text{sm}}$ | $\|\nabla\bar{\mathbf{N}}\| \exp(-\|\nabla C_{\text{gt}}\|)$ | 边缘感知法向量平滑，保留纹理边缘 |
| $\mathcal{L}_{\text{IDIV}}$ | $\|\bar{\mathbf{l}}_D - \bar{\mathbf{l}}_D^{\text{smooth}}\|$ | **新增**：IDIV 邻域平滑正则，防止过拟合局部噪声 |

推荐超参数初值：$\lambda_N = 0.05$，$\lambda_{\text{vol}} = 0.001$，$\lambda_{\text{sm}} = 1.0$，$\lambda_{\text{IDIV}} = 0.01$。

---

## 六、几何优化策略

### 6.1 两阶段训练

**阶段一（前 18k 步）——逐高斯初始化**

每个高斯基元使用局部材质属性和法向量直接计算出射辐射并 alpha-blending，确保几何快速收敛，同时让 IDIV 的 MLP 预训练到合理初值。

**阶段二（后 40k 步）——延迟着色精化**

切换为完整的 MA-IDIV + split-sum + IIV 管线，固定几何优化并精细化材质与光照分解。阶段切换时重置所有颜色与材质属性，仅保留高斯几何。

### 6.2 材质感知法向量传播（增强版）

对高金属度（$m \geq 0.02$）且低粗糙度（$r \leq 0.1$）的 2DGS 基元，定期扩大其尺度，同时将其 IDIV 近似零化（正则化），防止镜面区域的 IDIV 干扰法向量优化。

### 6.3 TSDF 周期网格提取

每 3000 步提取一次表面网格，用于光线追踪可见度计算，配合 BVH 加速结构实现高效互反射建模。

---

## 七、实验设计

### 7.1 数据集

| 数据集 | 场景类型 | 主要考察目标 |
|--------|----------|-------------|
| Shiny Blender | 合成镜面反射 | 高频镜面与互反射重建 |
| Glossy Synthetic | 合成光泽体 | 材质分解精度 |
| NeRF-Synthetic | 合成通用场景 | 非反射场景通用性 |
| Mip-NeRF 360 | 真实室外/室内 | 自适应近-远分解的室外性能 |
| Tanks & Temples | 大规模真实场景 | 泛化能力 |
| Ref-Real dataset | 真实反射场景 | 真实反射现象（球面镜、车窗） |

### 7.2 评估指标

- **渲染质量**：PSNR、SSIM、LPIPS
- **法向量精度**：MAE（Mean Angular Error）——在 Synthetic-NeRF 与 Glossy Synthetic 上使用 GT 法向量评测
- **材质分解质量**：环境贴图 SSIM/LPIPS（与 Ref-Gaussian 一致的评测协议）
- **效率**：训练时长（小时）、推理 FPS

### 7.3 对比基线

| 方法 | 核心特点 |
|------|---------|
| 3DGS | 原始基线 |
| Scaffold-GS | 锚点特征基线 |
| GaussianShader | 法向量 + envmap，无互反射 |
| SpecGaussian | 球谐高斯，无物理约束 |
| Normal-GS | 本文核心来源之一 |
| 3DGS-DR | 延迟渲染，无法向量显式耦合 |
| Ref-Gaussian | 本文核心来源之一 |
| **PhysNorm-GS（ours）** | 完整提案 |

### 7.4 消融实验设计

| 变体 | 去除模块 | 预期退化现象 |
|------|---------|-------------|
| A0（完整模型） | — | — |
| A1 | IDIV 延迟化（退化为高斯级 IDIV） | 法向量噪声增大，梯度平滑效果消失 |
| A2 | 材质调制 $(1-m)$（固定为 1） | 高金属度场景漫反射串扰，镜面失真 |
| A3 | 自适应近-远分解（固定 $\beta=1$） | 室外场景法向量 MAE 显著上升 |
| A4 | IIV 间接光照（退化为 SH） | 复杂互反射区域出现噪点 |
| A5 | IDIV 平滑正则 $\mathcal{L}_{\text{IDIV}}$ | IDIV 过拟合，法向量高频噪声 |

### 7.5 核心假设与可证伪预测

1. 在 Synthetic-NeRF 上，法向量 MAE 应低于 Normal-GS（20.71°），因为延迟渲染的梯度平滑减少了法向量噪声。
2. 在 Shiny Blender 的 ball 场景，PSNR 应高于 Normal-GS，接近或超过 Ref-Gaussian（37.01 dB）。
3. 在 Mip-NeRF 360 室外场景，PSNR 应高于 Normal-GS，验证近-远分解对室外问题的修复。
4. 渲染帧率应维持在 Ref-Gaussian（122 FPS）的同量级，因为 IDIV MLP 在推理前可 bake 为常量，无额外推理开销。

---

## 八、理论创新总结

本方法的核心创新不是两篇论文的简单拼接，而是提出了一个**新的理论命题**：

> *IDIV 可以作为高斯基元的一阶属性参与延迟渲染的 alpha-blending，并在像素级渲染方程中同时为漫反射和间接光照提供法向量耦合的梯度通路。*

这在数学上是自洽的（梯度推导见 §2.2），在计算上是高效的（仅增加 3 维 IDIV 向量的 alpha-blending，额外开销可忽略），在物理上是严格的（严格遵循 Kajiya 渲染方程，无额外近似引入的误差放大）。

材质感知的金属度调制 $(1-m)$ 提供了一个**从 Lambertian 到全镜面的连续插值**，使单一框架无需分支逻辑即可统一处理非反射（Normal-GS 擅长）和高反射（Ref-Gaussian 擅长）的场景。

自适应近-远光照分解则从光场理论角度系统性地解决了 IDIV 的室外退化问题，通过可学习参数 $\beta$ 在近场 IDIV 与远场 envmap 之间自动分配，而非通过启发式后处理修补。

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
| $\mathbf{n}$ / $\bar{\mathbf{N}}$ | 单个高斯法向量 / 像素级聚合法向量 |
| $\mathbf{l}_D$ / $\bar{\mathbf{l}}_D$ | IDIV / 像素级聚合 IDIV |
| $m$ / $\bar{M}$ | 高斯金属度 / 像素级聚合金属度 |
| $r$ / $\bar{R}$ | 高斯粗糙度 / 像素级聚合粗糙度 |
| $k_D$ / $\bar{\Lambda}$ | 漫反射率 / 像素级聚合反照率 |
| $\mathbf{f}_v$ | 锚点 $v$ 的局部特征向量 |
| $\omega_o$, $\omega_i$, $\omega_r$ | 出射、入射、反射方向 |
| $V$ | 可见度（0=被遮挡，1=可见） |
| $\beta$ | 近-远光照混合权重（可学习） |
| $\gamma$ | split-sum 与 IDE 混合权重（可学习） |
| $\theta_l$, $\theta_{\text{ind}}$, $\theta_S$ | IDIV、IIV、镜面 MLP 参数 |
