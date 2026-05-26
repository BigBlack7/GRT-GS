# PhysNorm-GS：面向混合反射场景的反射感知探针全局光照高斯逆渲染

## 摘要

本文研究 Gaussian Splatting 逆渲染在混合反射场景中的光照分解与稳定优化问题。现有反射高斯方法通常将高频环境贴图、BRDF 材质、法向和间接光放入同一端到端优化过程。该范式在高反射、低粗糙表面上可以从清晰镜面高光获得有效监督，但在漫反射物体、弱反射物体和大规模真实场景中，高频环境贴图缺乏可观测梯度，容易退化为吸收颜色残差的噪声变量。另一方面，仅依赖全局环境贴图也无法表达近场物体间反射和空间变化漫反射全局光照。

本文提出 **PhysNorm-GS**，一个面向混合反射场景的反射感知探针全局光照高斯逆渲染框架。核心观点是：不同材质区域对不同光照因子的可观测性不同，因此全局光照不应由单一环境贴图统一解释，而应根据反射强度、粗糙度、几何置信度和光度残差分解到远场镜面环境贴图、近场光线追踪辐射和空间漫反射辐照探针中。具体而言，本文提出 **Reflectance-Aware Probe Global Illumination, RAP-GI**：高频远场镜面反射由可靠镜面证据监督；近场镜面和互反射由局部 ray-traced transport 表达；低频漫反射全局光照由可学习 Gaussian irradiance probes 表示。训练上，本文进一步提出阶段一致连续优化以缓解 delayed rendering 阶段切换和材质重置带来的优化断裂。

本文贡献如下：

1. 提出 **RAP-GI：反射感知探针全局光照分解**。该方法将混合反射场景中的出射辐射分解为远场高频镜面环境、近场局部反射传输和空间低频漫反射辐照探针，并用材质可靠性决定各分量的监督责任。
2. 提出 **Material-aware Gaussian Irradiance Probes, MGIP**。该方法在高斯场景中构建可学习的低阶方向辐照探针场，使漫反射和粗糙表面主要监督局部低频全局光照，而不是污染高频环境贴图。
3. 提出 **Reflectance-Reliable Specular Factorization, R2SF**。该方法仅允许高反射、低粗糙、法向稳定且残差可信的区域强监督高频环境贴图，并将近场反射交给局部光线追踪传输。
4. 提出 **Phase-Consistent Continuation, PCC**。该方法将阶段切换从硬重置改为置信度软重置和连续过渡，缓解 delayed rendering 进入物理渲染后的指标突降。

---

## 1. 问题定义

### 1.1 单一环境贴图的可观测性缺陷

给定多视角图像集合 $\mathcal{I}=\{I_v\}$，目标是重建由高斯基元表示的几何、材质和光照。对像素 $\mathbf{p}$，物理渲染可写为：

$$
C(\mathbf{p})
=
D(\mathbf{p})
+
S(\mathbf{p})
+
I_{\mathrm{ind}}(\mathbf{p}),
$$

其中 $D$ 为漫反射项，$S$ 为镜面项，$I_{\mathrm{ind}}$ 为间接光或互反射项。镜面项通常通过反射方向查询环境贴图：

$$
\omega_r
=
2(\omega_o^\top \mathbf{n})\mathbf{n}
-
\omega_o,
$$

$$
S
=
f_s(\mathbf{n},\omega_o,r,\rho,E).
$$

其中 $\rho$ 是反射强度，$r$ 是粗糙度，$E$ 是环境贴图。对环境贴图 texel $E_k$ 的梯度近似为：

$$
\frac{\partial C}{\partial E_k}
=
w_s(\rho,r,\mathbf{n},\omega_o)
\frac{\partial E(\omega_r,r)}{\partial E_k}.
$$

当 $\rho\rightarrow 0$ 或 $r\rightarrow 1$ 时，镜面权重 $w_s$ 迅速衰减，高频环境贴图几乎不可观测。因此漫反射区域不应强监督高频环境贴图。若仍允许所有像素无差别优化 $E$，优化器会把 albedo、曝光、几何误差和未建模间接光写入环境贴图，形成彩色噪声。

### 1.2 漫反射全局光照的空间变化

漫反射颜色来自半球入射辐射积分：

$$
D(\mathbf{x},\mathbf{n})
=
A(\mathbf{x})
\int_{\Omega^+}
L_i(\mathbf{x},\omega)
\max(0,\mathbf{n}^{\top}\omega)
d\omega .
$$

该积分天然是低频的，并且依赖空间位置 $\mathbf{x}$。因此，大规模真实场景或多物体场景中的漫反射照明更适合由空间辐照场表达，而不是由单个全局环境贴图表达。传统图形学中的 irradiance volume、light probes 和 DDGI 正是针对这一问题：用空间探针缓存低频漫反射全局光照，并通过可见性降低漏光。

### 1.3 近场反射与远场反射的冲突

环境贴图适合描述天空、背景和远场照明，但无法表达局部物体在镜面表面中的近场反射。对低粗糙镜面表面，局部几何与遮挡会显著影响反射外观：

$$
S_{\mathrm{near}}
=
f_s(\mathbf{n},\omega_o,r,\rho,L_{\mathrm{local}}).
$$

若只使用全局环境贴图，近场物体反射会被错误写入 $E$，导致背景与物体反射互相污染。反过来，若只使用局部 ray tracing，又难以覆盖远场天空和不可见背景。因此需要远场与近场分解。

### 1.4 阶段切换导致的优化断裂

反射高斯逆渲染通常先使用稳定的高斯级着色获得几何，再切换到 delayed physical rendering 优化材质、法向、环境光和间接光。若阶段切换时全局重置颜色和材质，则优化目标从

$$
\min_{\mathcal{G},\theta_0}\mathcal{L}_{\mathrm{init}}
$$

突然变为

$$
\min_{\mathcal{G},\theta_{\mathrm{pbr}},E,\mathcal{P}}
\mathcal{L}_{\mathrm{pbr}},
$$

其中 $\mathcal{P}$ 为探针光照参数。该硬切换会造成训练曲线突降，部分场景即使继续训练也难以恢复早期指标。

---

## 2. 方法概览

PhysNorm-GS 使用 surface-aligned Gaussian primitives 表示几何和材质，并额外维护一个空间辐照探针场。每个 Gaussian $G_i$ 维护：

$$
G_i
=
\{
\mathbf{x}_i,
\alpha_i,
\mathbf{s}_i,
\mathbf{R}_i,
\mathbf{a}_i,
\rho_i,
r_i,
\mathbf{n}_i
\},
$$

其中 $\mathbf{a}_i$ 是基础颜色，$\rho_i$ 是反射强度，$r_i$ 是粗糙度。探针场为：

$$
\mathcal{P}
=
\{
(\mathbf{p}_j,\mathbf{c}_{j,lm},v_j)
\}_{j=1}^{N_p},
$$

其中 $\mathbf{p}_j$ 是 probe 位置，$\mathbf{c}_{j,lm}$ 是低阶球谐辐照系数，$v_j$ 是可选可见性或置信度统计。

最终颜色分解为：

$$
C
=
D_{\mathrm{probe}}
+
S_{\mathrm{far}}
+
S_{\mathrm{near}}
+
C_{\mathrm{bg}}.
$$

其中：

- $D_{\mathrm{probe}}$：由 Gaussian irradiance probes 提供的漫反射低频全局光照。
- $S_{\mathrm{far}}$：由高频环境贴图提供的远场镜面反射。
- $S_{\mathrm{near}}$：由局部 ray-traced transport 提供的近场镜面反射和互反射。
- $C_{\mathrm{bg}}$：背景与透明度混合项。

核心思想不是让所有像素同时监督所有变量，而是用材质责任权重分配梯度：

$$
q_{\mathrm{diff}}
=
(1-\rho)^\mu r^\nu q_{\mathrm{geo}}q_{\mathrm{photo}},
$$

$$
q_{\mathrm{spec}}
=
\rho^\alpha(1-r)^\beta q_{\mathrm{geo}}q_{\mathrm{photo}},
$$

$$
q_{\mathrm{rough}}
=
\rho^\alpha r^\eta q_{\mathrm{geo}}q_{\mathrm{photo}}.
$$

$q_{\mathrm{diff}}$ 高的区域主要监督探针辐照，$q_{\mathrm{spec}}$ 高的区域主要监督远场环境贴图和近场 ray-traced radiance，$q_{\mathrm{rough}}$ 对粗糙镜面提供低频反射混合。

---

## 3. RAP-GI：反射感知全局光照分解

### 3.1 三路光照分解

本文将入射光照分为三部分：

$$
L_i(\mathbf{x},\omega)
=
L_{\mathrm{probe}}(\mathbf{x},\omega)
+
L_{\mathrm{env}}(\omega)
+
L_{\mathrm{local}}(\mathbf{x},\omega).
$$

$L_{\mathrm{probe}}$ 是空间低频漫反射辐照，$L_{\mathrm{env}}$ 是远场高频环境光，$L_{\mathrm{local}}$ 是由局部几何产生的近场反射与互反射。

对漫反射项：

$$
D_{\mathrm{probe}}
=
A(\mathbf{x})
I_{\mathcal{P}}(\mathbf{x},\mathbf{n}),
$$

其中 $I_{\mathcal{P}}$ 是由探针插值得到的方向辐照。

对远场镜面项：

$$
S_{\mathrm{far}}
=
q_{\mathrm{spec}}
f_s(\mathbf{n},\omega_o,r,\rho,E_{\mathrm{HF}}).
$$

对近场镜面和互反射：

$$
S_{\mathrm{near}}
=
\gamma_t q_{\mathrm{spec}} q_{\mathrm{vis}}
f_s(\mathbf{n},\omega_o,r,\rho,L_{\mathrm{local}}),
$$

其中 $L_{\mathrm{local}}$ 由当前高斯表面或提取网格上的 ray tracing 得到，$\gamma_t$ 是训练中逐步启用的 ramp 权重。

### 3.2 可观测性驱动的梯度分配

对高频环境贴图，本文只允许可靠镜面区域强监督：

$$
\mathcal{L}_{E}
=
q_{\mathrm{spec}}
\rho_{\mathrm{rgb}}
\left(
C-I
\right)
+
\lambda_{\mathrm{tv}}
\|\nabla_{\mathbb{S}^2}E_{\mathrm{HF}}\|_1
+
\lambda_{\mathrm{eng}}
\mathcal{L}_{\mathrm{energy}}.
$$

对探针场，本文主要使用漫反射责任监督：

$$
\mathcal{L}_{\mathcal{P}}
=
q_{\mathrm{diff}}
\rho_{\mathrm{rgb}}
\left(
C-I
\right)
+
\lambda_{\mathrm{probe}}
\mathcal{R}(\mathcal{P}).
$$

这使低反射区域不再把残差写入高频环境贴图，而是更新空间低频辐照。

### 3.3 与传统 GI 的关系

传统 irradiance probes 和 DDGI 在前向渲染中通过探针缓存漫反射全局光照。本文将该思想转化为逆渲染中的可学习物理因子：probe 不再由已知场景和已知光源离线烘焙，而是由多视角图像、Gaussian 几何和材质责任共同优化。与传统 DDGI 相同，探针表达低频漫反射 GI；与传统 DDGI 不同，本文需要解决的是光照、材质和几何共同未知时的可观测性分配问题。

---

## 4. MGIP：材质感知 Gaussian Irradiance Probes

### 4.1 探针表示

每个 probe 维护低阶球谐辐照：

$$
I_j(\mathbf{n})
=
\sum_{l=0}^{L}
\sum_{m=-l}^{l}
\mathbf{c}_{j,lm}
Y_{lm}(\mathbf{n}).
$$

对任意高斯或像素位置 $\mathbf{x}$，选择邻近 probes 并插值得到：

$$
I_{\mathcal{P}}(\mathbf{x},\mathbf{n})
=
\sum_{j\in\mathcal{N}(\mathbf{x})}
w_j(\mathbf{x})
I_j(\mathbf{n}).
$$

最小实现中，probes 可放置在场景 bounding box 的规则网格中，使用 trilinear interpolation。进一步实现中，probes 可依据 Gaussian 密度和可见性自适应布置。

### 4.2 漫反射责任门控

探针只应由漫反射或粗糙表面强监督。定义：

$$
q_{\mathrm{diff}}
=
(1-\rho)^\mu r^\nu q_{\mathrm{n}}q_{\mathrm{vis}}q_{\mathrm{photo}}.
$$

其中 $q_{\mathrm{n}}$ 衡量法向一致性，$q_{\mathrm{vis}}$ 衡量多视角覆盖，$q_{\mathrm{photo}}$ 抑制异常残差。最终漫反射项为：

$$
D_{\mathrm{probe}}
=
A
\left[
(1-\tau q_{\mathrm{diff}}) I_0
+
\tau q_{\mathrm{diff}} I_{\mathcal{P}}(\mathbf{x},\mathbf{n})
\right],
$$

其中 $I_0$ 是基础低频照明或初始化颜色，$\tau$ 是 probe 启用强度。该形式保证 probe 在训练初期可退化，不会立即破坏已有 PBR 路径。

### 4.3 有界残差版本

为了避免探针成为任意颜色场，可将 probe 写成基础辐照上的有界残差：

$$
I_{\mathcal{P}}
=
I_{\mathrm{base}}
+
\Delta I_{\mathcal{P}},
$$

$$
\Delta I_{\mathcal{P}}
=
\tau q_{\mathrm{diff}}
\tanh
\left(
\sum_j w_j I_j(\mathbf{n})
\right).
$$

因此：

$$
\|\Delta I_{\mathcal{P}}\|_\infty
\le
\tau q_{\mathrm{diff}}.
$$

这一有界性使 probe 表达的是低频全局光照残差，而不是任意外观网络。

### 4.4 探针正则

本文使用空间平滑、能量约束和非负约束：

$$
\mathcal{R}(\mathcal{P})
=
\lambda_{\mathrm{smooth}}
\sum_{(i,j)\in\mathcal{E}_{p}}
\|\mathbf{c}_i-\mathbf{c}_j\|_1
+
\lambda_{\mathrm{energy}}
\left(
\overline{I}_{\mathcal{P}}-\overline{E}_{\mathrm{LF}}
\right)^2
+
\lambda_{\mathrm{neg}}
\|\min(I_{\mathcal{P}},0)\|_2^2.
$$

其中 $\mathcal{E}_{p}$ 是 probe 邻接图。该正则使探针在空间上平滑，并避免辐照能量无界漂移。

### 4.5 DDGI-lite 可见性扩展

完整版本可引入 DDGI 风格的可见性统计。每隔若干步，从 probe 向若干方向发射 rays，与 Gaussian 表面或提取 mesh 求交，记录 hit depth 的一阶和二阶矩：

$$
m_1(\omega)=\mathbb{E}[d],
\quad
m_2(\omega)=\mathbb{E}[d^2].
$$

查询时使用 Chebyshev 风格 visibility 降低漏光：

$$
V(\mathbf{x},\omega)
=
\mathrm{clip}
\left(
\frac{\sigma^2}{\sigma^2+(d-m_1)^2},
0,1
\right),
$$

其中 $\sigma^2=m_2-m_1^2$。该扩展用于大场景和遮挡复杂场景，但不作为首版必须实现项。

---

## 5. R2SF：可靠镜面因子分解

### 5.1 远场环境贴图可靠性

高频环境贴图只由可靠镜面证据监督：

$$
q_{\mathrm{spec}}
=
\rho^\alpha(1-r)^\beta
q_{\mathrm{n}}
q_{\mathrm{vis}}
q_{\mathrm{photo}}.
$$

当 $\rho$ 低或 $r$ 高时，$q_{\mathrm{spec}}$ 下降，环境贴图梯度被削弱。这样可避免非反射区域把 diffuse residual 写入 envmap。

实现上，R2SF 不直接压暗前向镜面颜色。否则高反射物体中的真实镜面能量会被错误削弱，导致渲染指标下降。本文采用 forward-preserving gradient gate：

$$
\tilde{S}_{\mathrm{far}}
=
q_{\mathrm{spec}}
S_{\mathrm{far}}
+
(1-q_{\mathrm{spec}})
\mathrm{sg}(S_{\mathrm{far}}),
$$

其中 $\mathrm{sg}(\cdot)$ 表示停止梯度。由于 $\tilde{S}_{\mathrm{far}}$ 与 $S_{\mathrm{far}}$ 在前向数值上相同，该形式不会改变渲染颜色；但反向传播时，不可靠区域对高频环境贴图和镜面分支的梯度被 $q_{\mathrm{spec}}$ 缩放。

### 5.2 远场和近场镜面拆分

本文将镜面项写成：

$$
S
=
\lambda_{\mathrm{far}}S_{\mathrm{far}}
+
\lambda_{\mathrm{near}}S_{\mathrm{near}}.
$$

其中：

$$
\lambda_{\mathrm{near}}
=
q_{\mathrm{spec}}q_{\mathrm{local}},
\quad
\lambda_{\mathrm{far}}
=
q_{\mathrm{spec}}(1-q_{\mathrm{local}}).
$$

$q_{\mathrm{local}}$ 可由 ray hit 是否存在、hit depth、几何置信度和 roughness 决定。镜面尖锐且命中局部表面时，近场传输占主导；粗糙或未命中局部表面时，远场环境贴图占主导。

---

## 6. PCC：阶段一致连续优化

### 6.1 置信度软重置

不再在 delayed rendering 阶段对所有材质执行硬重置，而是定义 Gaussian 级保留系数：

$$
q_i^{\mathrm{keep}}
=
q_i^{\mathrm{vis}}
q_i^{\mathrm{n}}
q_i^{\mathrm{photo}}
q_i^{\mathrm{opac}}.
$$

材质参数从旧值到初始值的软重置为：

$$
\theta_i^{+}
=
q_i^{\mathrm{keep}}\theta_i^{-}
+
(1-q_i^{\mathrm{keep}})\theta_0.
$$

可靠高斯保留已有外观和材质，不可靠高斯重新初始化。

### 6.2 连续过渡与蒸馏

在阶段边界 $[t_s,t_e]$ 定义：

$$
\eta_t
=
\mathrm{smoothstep}
\left(
\frac{t-t_s}{t_e-t_s}
\right).
$$

训练渲染为：

$$
C_t
=
(1-\eta_t)C_{\mathrm{init}}
+
\eta_t C_{\mathrm{phys}}.
$$

同时使用初始化阶段输出作为 teacher：

$$
\mathcal{L}_{\mathrm{phase}}
=
\|C_{\mathrm{phys}}-\mathrm{sg}(C_{\mathrm{init}})\|_1
+
\lambda_n
\left[
1-\langle N_{\mathrm{phys}},\mathrm{sg}(N_{\mathrm{init}})\rangle
\right]
+
\lambda_{\theta}
\|\Theta_{\mathrm{phys}}-\mathrm{sg}(\Theta_{\mathrm{init}})\|_1.
$$

该项只在过渡窗口内使用，并随训练逐渐衰减。

---

## 7. 总体优化目标

整体目标为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rgb}}
+
\lambda_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}
+
\lambda_E\mathcal{L}_E
+
\lambda_{\mathcal{P}}\mathcal{L}_{\mathcal{P}}
+
\lambda_{\mathrm{phase}}\mathcal{L}_{\mathrm{phase}}
+
\lambda_{\mathrm{local}}\mathcal{L}_{\mathrm{local}}.
$$

其中：

- $\mathcal{L}_{\mathrm{rgb}}$ 是 photometric reconstruction loss。
- $\mathcal{L}_{\mathrm{geo}}$ 包含法向、深度或 smoothness 正则。
- $\mathcal{L}_{E}$ 约束远场环境贴图。
- $\mathcal{L}_{\mathcal{P}}$ 约束探针辐照场。
- $\mathcal{L}_{\mathrm{phase}}$ 稳定阶段切换。
- $\mathcal{L}_{\mathrm{local}}$ 约束局部 ray-traced transport 的能量和置信度。

---

## 8. 分阶段实现策略

本文方法可以分阶段落地，避免一次性实现完整 DDGI 导致不可控：

1. **GIP-0：Per-Gaussian Irradiance Proxy。** 使用每个 Gaussian 的有界低频辐照残差作为探针代理，验证漫反射区域监督低频辐照是否有效。
2. **GIP-1：Learnable Grid Irradiance Probes。** 在场景 bbox 中放置规则 probe grid，使用 SH2/SH3 表示方向辐照，trilinear interpolation 查询。
3. **GIP-2：Gaussian-aware Adaptive Probes。** 根据 Gaussian 密度、可见性和场景尺度自适应布置 probes，提高大场景效率。
4. **GIP-3：DDGI-lite Visibility。** 利用已有 ray tracing 计算 probe visibility moments，缓解漏光和遮挡错误。
5. **Full RAP-GI。** 联合远场 envmap、近场 ray tracing 和 probe irradiance，形成完整混合反射全局光照逆渲染框架。

---

## 9. 与相关工作的关系

传统 irradiance volume、light probes 和 DDGI 证明了空间探针适合表达低频漫反射全局光照。本文不是将其直接用于前向渲染，而是将 probe 作为逆渲染中的可学习物理因子，并用材质可观测性控制其梯度来源。

基于 PBR 的高斯逆渲染和反射高斯方法通常强调环境贴图、BRDF 和 ray tracing 的联合优化。本文指出：环境贴图只适合可靠镜面证据，漫反射区域应监督空间辐照探针，近场反射应交给局部 ray-traced transport。该分解避免单一 envmap 同时解释背景、物体反射和漫反射残差。

本文与一般 appearance MLP 或颜色残差网络的区别在于：MGIP 是低阶方向辐照场，具有空间平滑、有界响应、材质责任门控和物理可解释性，不是任意外观拟合分支。

---

## 10. 预期优势

1. **非反射场景更稳。** 漫反射区域主要优化低频 probe irradiance，避免污染高频 envmap。
2. **反射场景更准。** 远场 envmap 与近场 ray-traced transport 分工明确，降低背景和局部物体反射冲突。
3. **真实大场景更适配。** 空间 probes 可表达局部照明变化，优于单个全局环境贴图。
4. **训练更稳定。** PCC 缓解 delayed rendering 阶段切换和材质重置导致的后期退化。
5. **创新主线更统一。** R2SF、MGIP、local ray tracing 和 PCC 都服务于同一命题：根据材质可观测性分配全局光照解释责任。

---

## 11. 最终论文叙事摘要

混合反射 Gaussian 逆渲染的核心困难并非材质参数不足，而是不同材质区域对不同光照因子的可观测性不同。高反射低粗糙区域能监督高频远场环境和近场镜面传输；漫反射和粗糙区域更适合监督空间低频全局辐照；真实大场景还需要处理局部照明变化与阶段优化不稳定。PhysNorm-GS 通过反射感知探针全局光照分解，将远场环境贴图、近场 ray-traced transport 和 Gaussian irradiance probes 统一到一个材质责任驱动的逆渲染框架中，从而减少环境贴图污染，提升混合反射场景的几何、材质和渲染质量。
