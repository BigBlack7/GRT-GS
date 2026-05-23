# PhysNorm-GS：面向混合反射场景的可靠照明因子分解与法向耦合高斯逆渲染

## 摘要

本文研究基于 Gaussian Splatting 的逆渲染在混合反射场景中的稳定重建问题。现有反射高斯方法通常将环境贴图、BRDF 材质、表面法向与间接光共同置于同一端到端优化框架中。该范式在高反射、低粗糙材质上能够从镜面高光中获得强约束，但在非反射物体、大规模真实场景或反射证据稀疏区域中，高频环境贴图缺乏可观测梯度，容易退化为吸收外观残差的高维噪声变量。与此同时，两阶段训练中从高斯级着色到延迟物理渲染的硬切换、材质属性的全局重置、以及基于网格可见性的互反射过早反馈，都会造成材质分解和几何优化的不稳定。

为此，本文提出 **PhysNorm-GS**，一个面向反射与漫反射统一重建的物理引导高斯逆渲染框架。本文的核心观点是：混合反射场景中的照明、法向、材质与间接光不应由固定迭代阶段和无差别全局变量共同解释，而应由反射可靠性、几何置信度和漫反射责任进行分配。具体而言，本文提出可靠反射引导的照明因子分解，将低频辐照场与高频镜面环境贴图解耦；提出法向耦合辐照场，为漫反射区域提供有界、零初始化、可退化的法向相关辐照残差；提出阶段一致连续优化，缓解从初始化渲染到延迟物理渲染的优化断裂；提出置信度门控互反射，使间接光反馈由几何和材质可靠性共同控制。

本文贡献如下：

1. 提出 **Reliable Reflectance-Guided Illumination Factorization, R2IF**。该方法根据反射强度、粗糙度、法向可靠性、可见性与重建残差估计高频环境贴图的可观测性，仅使用可靠镜面证据优化高频环境光，并以低频空间辐照场解释非反射与大规模真实场景中的主要照明。
2. 提出 **Normal-Coupled Irradiance Field, NCIF**。该方法以局部 Gaussian 辐照残差和空间 cell 低阶方向辐照共同建模漫反射照明，并通过零初始化、有界响应和漫反射责任门控为法向提供 photometric gradient，同时避免破坏镜面 PBR 分解。
3. 提出 **Phase-Consistent Continuation, PCC**。该方法将训练阶段切换从硬重置改为连续过渡，通过跨阶段 RGB、法向与材质蒸馏将初始化阶段的稳定几何和外观迁移到延迟物理渲染阶段。
4. 提出 **Confidence-Gated Interreflection, CGI**。该方法将互反射由固定迭代开关改为局部置信度驱动的物理反馈，降低不稳定网格可见性对材质和反射分解的污染。

---

## 1. 问题定义

### 1.1 混合反射场景中的可观测性不均衡

给定多视角图像集合 $\mathcal{I}=\{I_v\}$，目标是重建由高斯基元表示的几何、外观、材质和照明。对一个像素 $\mathbf{p}$，典型物理渲染形式可写为：

$$
C(\mathbf{p})
=
D(\mathbf{p})
+
S(\mathbf{p})
+
I_{\mathrm{ind}}(\mathbf{p}),
$$

其中 $D$ 为漫反射或基础颜色分量，$S$ 为由 BRDF 和环境光驱动的镜面分量，$I_{\mathrm{ind}}$ 为互反射或间接光项。镜面分量通常依赖反射方向：

$$
\omega_r
=
2(\omega_o^\top \mathbf{n})\mathbf{n}
-
\omega_o,
$$

并通过粗糙度 $r$ 和反射强度 $\rho$ 查询环境照明：

$$
S
=
f_s(\mathbf{n},\omega_o,r,\rho,E).
$$

当 $\rho$ 较大且 $r$ 较小时，图像中存在清晰的镜面证据，高频环境贴图 $E$、法向 $\mathbf{n}$ 和材质参数可以得到较强约束。相反，当 $\rho \rightarrow 0$ 或 $r \rightarrow 1$ 时，高频环境贴图对像素颜色的影响迅速衰减。此时强行优化高维环境贴图会导致如下问题：

1. 环境贴图梯度稀疏且病态，无法从图像中可靠恢复。
2. 高频环境贴图会吸收相机曝光、背景、albedo、几何误差和动态残差。
3. 非反射场景中，物理分解变量越多，越容易降低 RGB 重建稳定性。
4. 大规模真实场景中，单一全局环境贴图不足以解释空间变化照明。

这解释了一个常见现象：反射逆渲染方法在光滑反射物体上表现较好，但在非反射物体或真实开放场景中，环境贴图可能优化成彩色噪声，从而反过来损害 PSNR、SSIM 和 LPIPS。

### 1.2 阶段切换导致的优化断裂

许多反射高斯逆渲染流程采用两阶段训练：先用较稳定的高斯级着色获得几何与初始外观，再切换到延迟物理渲染以优化材质、法向和照明。若在阶段切换时全局重置颜色和材质，仅保留几何，则优化问题从

$$
\min_{\mathcal{G},\theta_0}
\mathcal{L}_{\mathrm{init}}
$$

突然变为

$$
\min_{\mathcal{G},\theta_{\mathrm{pbr}},E}
\mathcal{L}_{\mathrm{pbr}},
$$

其中 $\theta_0$ 和 $\theta_{\mathrm{pbr}}$ 的参数含义、渲染路径和梯度分布均不同。该硬切换会产生 optimization phase shift：训练曲线在阶段边界附近突然下降，部分场景即使继续训练也难以恢复早期指标。

### 1.3 本文目标

本文不把混合反射重建视为单纯增加材质变量或照明变量的问题，而将其表述为一个可靠性分配问题：

$$
\text{Which observation should supervise which physical factor?}
$$

本文希望满足四个原则：

1. 高频环境贴图只由可靠镜面证据监督。
2. 漫反射区域不应被迫通过高频环境贴图解释，而应由低频辐照和法向耦合残差解释。
3. 阶段切换应保持颜色、法向和材质的一致性。
4. 互反射只应在几何和材质均可靠时参与反馈。

---

## 2. 方法概览

PhysNorm-GS 使用表面高斯基元表示场景。每个 Gaussian $G_i$ 维护位置、尺度、旋转、不透明度、基础颜色、反射强度、粗糙度、法向相关参数以及局部辐照残差：

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
\mathbf{d}_i
\}.
$$

渲染时先通过 splatting 聚合 G-buffer，再执行像素级物理着色。最终颜色写为：

$$
C
=
D_{\mathrm{NCIF}}
+
S_{\mathrm{R2IF}}
+
I_{\mathrm{CGI}}
+
C_{\mathrm{bg}}.
$$

其中：

- $D_{\mathrm{NCIF}}$ 是法向耦合辐照场产生的漫反射分量。
- $S_{\mathrm{R2IF}}$ 是由可靠反射证据约束的镜面分量。
- $I_{\mathrm{CGI}}$ 是置信度门控的互反射项。
- $C_{\mathrm{bg}}$ 是背景和透明度混合项。

整体优化目标为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rgb}}
+
\lambda_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}
+
\lambda_{\mathrm{illum}}\mathcal{L}_{\mathrm{illum}}
+
\lambda_{\mathrm{ncif}}\mathcal{L}_{\mathrm{ncif}}
+
\lambda_{\mathrm{phase}}\mathcal{L}_{\mathrm{phase}}
+
\lambda_{\mathrm{ind}}\mathcal{L}_{\mathrm{ind}}.
$$

---

## 3. R2IF：可靠反射引导的照明因子分解

### 3.1 高频环境贴图不可观测性的梯度分析

考虑镜面项：

$$
S(\mathbf{p})
=
w_s(\mathbf{p})
E(\omega_r(\mathbf{p}),r(\mathbf{p})),
$$

其中 $w_s$ 是由 Fresnel、反射强度、粗糙度、alpha 和 BRDF 查表共同决定的权重。对环境贴图 texel $E_k$ 的梯度可写为：

$$
\frac{\partial C(\mathbf{p})}{\partial E_k}
=
w_s(\mathbf{p})
\frac{\partial E(\omega_r,r)}{\partial E_k}.
$$

当 $\rho \rightarrow 0$ 时，$w_s \rightarrow 0$；当 $r$ 较大时，环境查询被强低通滤波，单个高频 texel 的梯度也趋于弱化。因此：

$$
\left\|
\frac{\partial C}{\partial E_{\mathrm{HF}}}
\right\|
\rightarrow 0
\quad
\text{if}
\quad
\rho \rightarrow 0
\quad
\text{or}
\quad
r \rightarrow 1.
$$

这说明高频环境贴图在低反射或高粗糙区域不可辨识。若仍允许所有像素更新高频环境贴图，优化器会把无法由当前模型解释的颜色残差写入 $E_{\mathrm{HF}}$，形成噪声环境贴图。

### 3.2 可靠反射门控

本文定义高频环境贴图的可靠性权重：

$$
q_{\mathrm{spec}}
=
\rho^\alpha
\cdot
(1-r)^\beta
\cdot
q_{\mathrm{n}}
\cdot
q_{\mathrm{vis}}
\cdot
q_{\mathrm{photo}}.
$$

其中：

$$
q_{\mathrm{n}}
=
\exp
\left(
-
\kappa_n
\left[
1-
\langle
\mathbf{n}_{\mathrm{render}},
\mathbf{n}_{\mathrm{depth}}
\rangle
\right]
\right),
$$

$$
q_{\mathrm{photo}}
=
\exp
\left(
-
\kappa_p
\|C-I\|_1
\right),
$$

$$
q_{\mathrm{vis}}
=
\mathrm{clip}
\left(
\frac{m_i}{m_0},0,1
\right),
$$

$m_i$ 表示 Gaussian 或像素的多视角可见次数。直观上，只有高反射、低粗糙、法向稳定、视角覆盖充分且 photometric residual 不异常的区域，才允许强监督高频环境贴图。

### 3.3 低频和高频照明分解

本文将照明分解为低频空间辐照和高频镜面环境：

$$
L(\mathbf{x},\omega,\mathbf{n})
=
L_{\mathrm{LF}}(\mathbf{x},\mathbf{n})
+
q_{\mathrm{spec}}(\mathbf{x})
E_{\mathrm{HF}}(\omega).
$$

低频辐照场使用空间 cell 上的低阶球谐函数表示：

$$
L_{\mathrm{LF}}(\mathbf{x},\mathbf{n})
=
\sum_{l=0}^{L}
\sum_{m=-l}^{l}
\mathbf{c}_{c(\mathbf{x}),lm}
Y_{lm}(\mathbf{n}),
$$

其中 $c(\mathbf{x})$ 是空间 cell 索引，$Y_{lm}$ 是球谐基。该项主要解释非反射物体、大场景中的空间变化漫反射照明，以及真实图像中的低频光照差异。

高频环境贴图只用于镜面分量：

$$
S_{\mathrm{R2IF}}
=
q_{\mathrm{spec}}
f_s
(
\mathbf{n},
\omega_o,
r,
\rho,
E_{\mathrm{HF}}
).
$$

### 3.4 环境贴图正则

为了防止高频环境贴图成为残差噪声，本文加入三类约束：

$$
\mathcal{L}_{E}
=
\lambda_{\mathrm{tv}}
\|\nabla_{\mathbb{S}^2}E_{\mathrm{HF}}\|_1
+
\lambda_{\mathrm{eng}}
\left(
\overline{E}_{\mathrm{HF}}-\overline{L}_{\mathrm{LF}}
\right)^2
+
\lambda_{\mathrm{sat}}
\|\max(E_{\mathrm{HF}}-E_{\max},0)\|_2^2.
$$

其中第一项是球面 TV，第二项约束全局能量，第三项抑制颜色爆炸。训练早期或低反射场景中，$E_{\mathrm{HF}}$ 可保持冻结或只允许低学习率更新。

---

## 4. NCIF：法向耦合辐照场

### 4.1 设计动机

在低反射区域，镜面项弱，颜色对法向的梯度不足：

$$
\left\|
\frac{\partial S}{\partial \mathbf{n}}
\right\|
\approx 0.
$$

若漫反射项仅为颜色贴图或基础颜色：

$$
D=A,
$$

则 RGB loss 很难直接修正法向。本文提出法向耦合辐照场，使漫反射颜色与法向建立有界联系。

### 4.2 局部与空间多尺度辐照残差

定义漫反射责任：

$$
q_{\mathrm{diff}}
=
(1-\rho)^\mu
r^\nu
q_{\mathrm{n}}
q_{\mathrm{vis}}.
$$

其中低反射、高粗糙、法向可靠且可见充分的区域更依赖 NCIF。对像素 $\mathbf{p}$，聚合局部辐照残差：

$$
\bar{\mathbf{d}}(\mathbf{p})
=
\sum_{i\in \mathcal{R}(\mathbf{p})}
T_i\alpha_i
\mathbf{d}_i.
$$

空间 cell 的方向辐照残差为：

$$
H_{c(\mathbf{p})}(\mathbf{n})
=
\sum_{l=0}^{L'}
\sum_{m=-l}^{l}
\mathbf{h}_{c(\mathbf{p}),lm}
Y_{lm}(\mathbf{n}).
$$

NCIF 的响应定义为：

$$
R_{\mathrm{NCIF}}
=
\tau
q_{\mathrm{diff}}
\tanh
\left(
\mathbf{n}^{\top}\bar{\mathbf{d}}
+
H_{c(\mathbf{p})}(\mathbf{n})
\right).
$$

最终漫反射项为：

$$
D_{\mathrm{NCIF}}
=
A
L_{\mathrm{LF}}(\mathbf{x},\mathbf{n})
\left(
1+
R_{\mathrm{NCIF}}
\right).
$$

### 4.3 零初始化退化性

令所有 $\mathbf{d}_i$ 和 $\mathbf{h}_{c,lm}$ 初始化为零，则：

$$
R_{\mathrm{NCIF}}=0,
$$

因此：

$$
D_{\mathrm{NCIF}}
=
A
L_{\mathrm{LF}}.
$$

这说明 NCIF 在训练初期严格退化为普通低频辐照漫反射模型，不会像无约束颜色分支一样破坏已有物理分解。

### 4.4 有界性

由于：

$$
\tanh(x)\in[-1,1],
$$

可得：

$$
|R_{\mathrm{NCIF}}|
\le
\tau q_{\mathrm{diff}}
\le
\tau.
$$

因此 NCIF 最多只能以 $\tau$ 的比例调制漫反射能量，无法无限吸收颜色残差。这一性质保证它是辐照残差，而不是任意外观网络。

### 4.5 法向梯度

对法向求导：

$$
\frac{\partial D_{\mathrm{NCIF}}}{\partial \mathbf{n}}
=
A
\frac{\partial L_{\mathrm{LF}}}{\partial \mathbf{n}}
(1+R_{\mathrm{NCIF}})
+
A
L_{\mathrm{LF}}
\frac{\partial R_{\mathrm{NCIF}}}{\partial \mathbf{n}}.
$$

其中：

$$
\frac{\partial R_{\mathrm{NCIF}}}{\partial \mathbf{n}}
=
\tau q_{\mathrm{diff}}
\left(
1-\tanh^2(z)
\right)
\left(
\bar{\mathbf{d}}
+
\frac{\partial H_c(\mathbf{n})}{\partial \mathbf{n}}
\right)
+
\tau
\frac{\partial q_{\mathrm{diff}}}{\partial \mathbf{n}}
\tanh(z).
$$

该梯度为漫反射区域提供额外 photometric normal supervision。由于 $q_{\mathrm{diff}}$ 在高反射区域较小，NCIF 不会与镜面 PBR 路径竞争法向解释。

### 4.6 NCIF 正则

本文使用：

$$
\mathcal{L}_{\mathrm{NCIF}}
=
\lambda_d
\sum_i
\|\mathbf{d}_i\|_2^2
+
\lambda_h
\sum_c
\|\mathbf{h}_c\|_2^2
+
\lambda_{\mathrm{smooth}}
\sum_{(i,j)\in \mathcal{E}}
w_{ij}
\|\mathbf{d}_i-\mathbf{d}_j\|_1.
$$

邻接权重 $w_{ij}$ 可由空间距离、法向相似性和颜色边缘共同决定，使辐照残差在同一表面内平滑、在材质边界处保留变化。

---

## 5. PCC：阶段一致连续优化

### 5.1 阶段分布偏移

两阶段逆渲染中的硬切换会改变渲染函数：

$$
C_{\mathrm{init}}
=
F_{\mathrm{init}}(\mathcal{G},\theta_0),
$$

$$
C_{\mathrm{defer}}
=
F_{\mathrm{defer}}(\mathcal{G},\theta_{\mathrm{mat}},E,L).
$$

若在切换时直接重置 $\theta_{\mathrm{mat}}$，则优化器需要在短时间内重新解释颜色、法向、材质和照明。这会导致训练指标在切换点附近下降。

### 5.2 连续过渡渲染

本文在阶段边界设置过渡窗口 $[t_s,t_e]$，定义：

$$
\eta_t
=
\mathrm{smoothstep}
\left(
\frac{t-t_s}{t_e-t_s}
\right).
$$

训练渲染颜色为：

$$
C_t
=
(1-\eta_t)
C_{\mathrm{init}}
+
\eta_t
C_{\mathrm{defer}}.
$$

这样优化目标从初始化模型连续变形到物理延迟渲染模型，避免硬切换。

### 5.3 跨阶段蒸馏

使用初始化阶段的稳定预测作为 teacher：

$$
\mathcal{L}_{\mathrm{phase}}
=
\|C_{\mathrm{defer}}-\mathrm{sg}(C_{\mathrm{init}})\|_1
+
\lambda_n
\left[
1-
\langle
N_{\mathrm{defer}},
\mathrm{sg}(N_{\mathrm{init}})
\rangle
\right]
+
\lambda_\theta
\|\Theta_{\mathrm{defer}}-\mathrm{sg}(\Theta_{\mathrm{init}})\|_1.
$$

其中 $\Theta$ 可以包含 albedo、reflectance、roughness 和低频 irradiance。该项只在过渡窗口内使用，并随 $\eta_t$ 衰减。

### 5.4 置信度软重置

不再对全部高斯执行硬重置，而是定义 Gaussian 级保留系数：

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
q_i^{\mathrm{keep}}
\theta_i^{-}
+
(1-q_i^{\mathrm{keep}})
\theta_0.
$$

可靠的高斯保留已有外观和材质，不可靠的高斯重新初始化。这比全局重置更稳定，也更符合多视角优化中的置信度原则。

---

## 6. CGI：置信度门控互反射

### 6.1 问题

互反射需要依赖几何表面、可见性和反射方向。如果 mesh 或 surfel depth 尚不稳定，ray tracing 产生的 visibility 会把几何错误转化为错误间接光。该错误随后进入 specular shading，污染 reflectance、roughness、envmap 和法向。

### 6.2 门控互反射

本文将互反射写为：

$$
I_{\mathrm{CGI}}
=
\gamma_t
q_{\mathrm{geo}}
q_{\mathrm{mat}}
I_{\mathrm{ind}},
$$

其中 $\gamma_t$ 是全局渐进权重：

$$
\gamma_t
=
\mathrm{clip}
\left(
\frac{t-t_{\mathrm{ind}}}{T_{\mathrm{ramp}}},
0,
1
\right).
$$

几何置信度：

$$
q_{\mathrm{geo}}
=
q_{\mathrm{n}}
q_{\mathrm{depth}}
q_{\mathrm{vis}}
q_{\mathrm{mesh}}.
$$

材质置信度：

$$
q_{\mathrm{mat}}
=
\rho^\alpha
\cdot
(1-r)^\beta
\cdot
q_{\mathrm{photo}}.
$$

只有高反射、低粗糙、几何稳定且 residual 合理的区域才强启用互反射。对于非反射区域，互反射自然衰减，不再干扰漫反射重建。

### 6.3 间接光一致性正则

为避免间接光突变，加入：

$$
\mathcal{L}_{\mathrm{ind}}
=
\lambda_{\mathrm{ind\_smooth}}
\|\nabla I_{\mathrm{CGI}}\|_1
+
\lambda_{\mathrm{ind\_energy}}
\|\max(I_{\mathrm{CGI}}-\xi S_{\mathrm{direct}},0)\|_2^2.
$$

第一项抑制局部噪声，第二项限制间接光相对直接光的能量，避免错误 mesh 产生过强补偿。

---

## 7. 大规模真实场景外观校准

真实开放场景中的低指标通常不仅来自反射模型，还来自曝光、白平衡、背景、动态物体和尺度变化。本文引入轻量外观校准：

$$
\hat{I}_v
=
\mathbf{a}_v
\odot
I_v
+
\mathbf{b}_v,
$$

其中 $\mathbf{a}_v$ 和 $\mathbf{b}_v$ 是每张训练图的颜色仿射参数，并加入弱正则：

$$
\mathcal{L}_{\mathrm{exp}}
=
\|\mathbf{a}_v-\mathbf{1}\|_2^2
+
\|\mathbf{b}_v\|_2^2.
$$

背景和天空区域不应监督前景材质与环境贴图，因此定义背景门控 $q_{\mathrm{fg}}$：

$$
\mathcal{L}_{\mathrm{rgb}}
=
\rho_{\mathrm{robust}}
\left(
q_{\mathrm{fg}}
(C-\hat{I})
\right).
$$

其中 $\rho_{\mathrm{robust}}$ 可取 Charbonnier 或 Huber loss，以降低高光异常、遮挡和动态误差对材质分解的影响。

---

## 8. 总体训练流程

本文采用如下统一训练协议：

1. **几何与初始外观阶段**。使用稳定的高斯级着色优化几何、基础颜色、透明度与初始法向。
2. **PCC 过渡阶段**。在 $[t_s,t_e]$ 内同时计算初始化渲染和延迟物理渲染，通过连续权重 $\eta_t$ 与跨阶段蒸馏平滑迁移。
3. **R2IF 解耦照明阶段**。低频辐照场持续优化，高频环境贴图仅由高 $q_{\mathrm{spec}}$ 的可靠镜面证据监督。
4. **NCIF 法向耦合阶段**。在低反射和高粗糙区域逐步启用 NCIF residual，为漫反射法向提供 photometric gradient。
5. **CGI 互反射阶段**。在几何和材质置信度足够时逐步启用局部互反射。
6. **真实场景增强阶段**。对 Ref-Real 或大规模真实场景启用曝光校准、背景门控和空间低频辐照 cell。

---

## 9. 当前工程落地状态与测试计划

本节用于区分论文中的完整理论方案与当前工程已经实现的首版模块。当前实现遵循一个原则：保留 Ref-Gaussian 的主体渲染链路和材质参数体系，在其 surfel / volume 渲染路径上增量加入可靠性门控、NCIF 辐照残差、互反射门控和阶段切换稳定化。未完成的部分不会在实验结论中被默认视为已实现贡献。

### 9.1 当前已落地内容

- [x] **Ref-Gaussian 主链路保留**：保留反射强度、金属度、粗糙度、环境贴图、surfel 渲染、volume 渲染、delayed rendering 和 mesh-based indirect rendering 的主体流程。
- [x] **旧 IDIV 命名清理**：工程内旧 `idiv`、`eval_indirect`、`no_eval_indirect` 参数已从主训练与评估路径中移除，统一替换为 NCIF 语义。
- [x] **Per-Gaussian NCIF 向量**：每个 Gaussian 维护零初始化的 3D NCIF 辐照残差向量，并进入优化器、densification、pruning、PLY 保存与读取流程。
- [x] **NCIF 有界漫反射调制**：NCIF 通过 splatting 聚合到像素后，以 $\tanh(\mathbf{n}^{\top}\mathbf{d})$ 的有界响应调制 diffuse 分量，并使用 ramp 权重逐步启用。
- [x] **NCIF 漫反射责任门控**：当前实现使用反射强度和粗糙度构造 $q_{\mathrm{diff}}=(1-\rho)^\mu r^\nu$，使 NCIF 更偏向低反射、高粗糙区域。
- [x] **NCIF 正则首版**：已加入 NCIF map 平滑正则和 NCIF magnitude 正则，限制残差过度吸收颜色。
- [x] **R2IF 首版镜面可靠性门控**：当前实现使用反射强度和粗糙度构造 $q_{\mathrm{spec}}=\rho^\alpha(1-r)^\beta$，并在 surfel / volume 路径中调制 specular 强度。
- [x] **环境贴图稳定正则**：已加入环境贴图 TV 正则和双环境贴图能量一致性正则，以缓解低反射场景中 envmap 噪声化。
- [x] **CGI 首版互反射门控**：间接光不再由固定迭代后全量启用，而是结合训练 ramp 和镜面可靠性 gate 控制 indirect light 对 specular 的贡献。
- [x] **PCC 软重置首版**：阶段切换时可使用 soft reset 保留一部分已有材质与颜色，而不是完全硬重置所有材质属性。
- [x] **训练脚本统一**：`train.sh` 已统一到当前 NCIF/R2IF/PCC/CGI 参数体系，并保留部分官方稳定参数。
- [x] **中文 README**：README 已改写为中文，说明核心模块、训练命令、评估开关和参数含义。

### 9.2 当前尚未完整落地内容

- [ ] **完整 R2IF 可观测性估计**：当前 R2IF 只使用反射强度和粗糙度，尚未加入法向一致性 $q_{\mathrm{n}}$、可见性 $q_{\mathrm{vis}}$、photometric residual $q_{\mathrm{photo}}$ 和显式高频 envmap 梯度屏蔽。
- [ ] **低频空间辐照 cell / SH 场**：当前 NCIF 只实现 per-Gaussian 局部残差，尚未实现空间 cell 级低频方向辐照场。
- [ ] **完整 PCC 连续过渡渲染**：当前 PCC 是 soft reset 首版，尚未实现初始化渲染与 PBR 渲染的连续混合、RGB 蒸馏、法向蒸馏和材质蒸馏。
- [ ] **完整 CGI 几何置信度**：当前 CGI 使用训练 ramp 与反射可靠性门控，尚未加入 mesh 置信度、depth consistency、visibility consistency 和 indirect energy 正则。
- [ ] **真实场景外观校准**：尚未实现 per-image exposure / white-balance 仿射校准、背景门控和 robust photometric loss。
- [ ] **自动化实验汇总**：尚未实现面向 18k/20k/22k/30k/50k 的自动曲线对比、envmap 可视化汇总和模块消融表格生成。

### 9.3 创新点逐步落地测试流程

为了避免一次性叠加多个模块导致无法判断收益来源，实验应按照从稳定性到完整性的顺序推进。

**Step 0：Ref-Gaussian 复现基线。**

目标是确认当前清理后的工程仍能复现 Ref-Gaussian 主链路。训练时关闭所有新模块和新增正则：

$$
\texttt{--no\_use\_ncif --no\_use\_r2if --no\_use\_pcc --no\_use\_cgi --lambda\_env\_tv 0 --lambda\_env\_energy 0}
$$

优先测试已有突降嫌疑场景，例如 Bell、Toaster、Lego、Mic，并记录 18k、20k、22k、30k、50k 的 PSNR、SSIM、LPIPS、法向图、材质图和 envmap。

**Step 1：只测试 PCC 软重置。**

目标是验证 20k 附近指标突降是否主要来自硬重置和阶段切换。仅开启 PCC，关闭 R2IF、NCIF、CGI：

$$
\texttt{--use\_pcc --no\_use\_ncif --no\_use\_r2if --no\_use\_cgi --lambda\_env\_tv 0 --lambda\_env\_energy 0}
$$

若 20k 后 PSNR 曲线更平滑、30k 相比 baseline 回升，则说明阶段切换稳定化有效。此阶段重点调 `pcc_keep_ratio`，建议从 0.5、0.75、0.9 三档测试。

**Step 2：测试 R2IF 与环境贴图稳定正则。**

目标是验证低反射和高粗糙区域不应强监督高频 envmap。开启 R2IF 和 envmap 正则，仍关闭 NCIF 与 CGI：

$$
\texttt{--use\_pcc --use\_r2if --no\_use\_ncif --no\_use\_cgi}
$$

重点观察 NeRF Synthetic、Ref-Real 和低反射物体的 envmap 是否从彩色噪声变得更平滑，PSNR 是否至少不低于 baseline，LPIPS 是否改善。此阶段调 `r2if_specular_alpha`、`r2if_specular_beta`、`r2if_min_specular_gate`、`lambda_env_tv`。

**Step 3：测试 CGI 互反射门控。**

目标是验证间接光只在可靠镜面区域启用是否能减少错误 mesh feedback。开启 PCC、R2IF、CGI，关闭 NCIF：

$$
\texttt{--use\_pcc --use\_r2if --use\_cgi --no\_use\_ncif}
$$

优先测试高反射且存在互反射的场景，例如 Bell、Tbell、Teapot、Toaster。若 specular 边界更稳定、反射区域 LPIPS 改善，同时非反射区域不被间接光污染，则保留该模块。

**Step 4：测试 NCIF 漫反射法向耦合。**

目标是验证 NCIF 是否能提升低反射/漫反射区域的颜色和几何。开启 PCC、R2IF、NCIF，先关闭 CGI：

$$
\texttt{--use\_pcc --use\_r2if --use\_ncif --no\_use\_cgi}
$$

优先测试 NeRF Synthetic 中 Chair、Ficus、Hotdog、Ship，以及 Ref-Real 中 Gardenspheres、Toycar、Sedan。需要同时保存 `ncif_map`、`ncif_response`、`ncif_responsibility` 与法向图，确认 NCIF 只在 diffuse-dominant 区域工作，没有吸收镜面高光。

**Step 5：完整 PhysNorm-GS。**

目标是验证 PCC、R2IF、CGI、NCIF 共同作用后的最终性能。使用 `train.sh` 当前完整命令训练全数据集，并与 Step 0 基线、Ref-Gaussian 30k/50k、当前第一轮实验结果对比。

**Step 6：决定第二轮实现方向。**

若 Step 1 收益最大，则优先实现完整 PCC 蒸馏；若 Step 2 对 envmap 改善明显但指标不足，则优先实现空间低频 irradiance cell；若 Step 4 在 NeRF Synthetic 和 Ref-Real 有收益，则继续强化 NCIF 的多尺度低频辐照场；若 CGI 在部分场景负收益，则将其改为仅高可靠场景启用或加入更严格的 geometry confidence。

---

## 10. 消融实验设计

为了证明各模块有效性，本文设计如下消融：

| 设置 | 目的 | 预期现象 |
| --- | --- | --- |
| w/o R2IF | 验证可靠环境贴图优化 | 非反射场景 envmap 噪声增强，PSNR/LPIPS 下降 |
| w/o NCIF | 验证漫反射法向梯度 | NeRF Synthetic 与 Ref-Real 中漫反射区域法向和颜色下降 |
| w/o PCC | 验证阶段切换稳定性 | 18k-20k 附近更容易出现指标突降 |
| hard reset | 验证软重置必要性 | 材质重新收敛慢，部分场景后期退化 |
| w/o CGI | 验证互反射门控 | 高反射局部区域间接光错误，反射边界模糊 |
| global envmap only | 验证低频辐照分解 | 非反射和真实场景环境贴图混乱 |
| no exposure calibration | 验证真实图像外观校准 | Ref-Real 指标下降，材质颜色漂移 |
| NCIF without bound | 验证有界残差 | residual 吸收颜色，材质可解释性下降 |

关键可视化包括：

- 高频环境贴图 $E_{\mathrm{HF}}$。
- 低频辐照 cell。
- $q_{\mathrm{spec}}$ 可靠反射图。
- $q_{\mathrm{diff}}$ 漫反射责任图。
- NCIF residual response。
- CGI visibility 和 indirect map。
- 18k-25k 训练曲线与阶段切换前后材质图。

---

## 11. 与相关思想的关系

本文借鉴三类成熟思想，但方法目标和组合方式不同。

首先，deferred inverse rendering 和基于 BRDF 的环境光查询为反射重建提供了物理基础。然而，本文指出高频环境贴图并非在所有区域都可观测，因此提出 R2IF，以反射可靠性决定高频照明监督。

其次，surface-aligned Gaussian primitives 能缓解 3D Gaussian 的多视角几何不一致。本文进一步将其 normal-depth consistency 用作可靠性估计，参与照明、材质、重置和互反射的责任分配。

第三，传统图形学中的 continuation optimization、trust-region、球谐低频照明、鲁棒 photometric loss、曝光校准和尺度感知过滤，为大规模真实场景提供了稳定优化原则。本文将这些原则嵌入 Gaussian 逆渲染流程，形成统一的可靠性驱动物理优化框架。

本文不将某个已有模块直接作为主贡献，而是围绕一个统一问题展开：**在混合反射场景中，不同物理因子的可观测性高度不均衡，因此必须用可靠性来决定照明、法向、材质和互反射各自应承担的解释责任。**

---

## 12. 预期优势

PhysNorm-GS 的优势可以概括为：

1. **反射场景更稳定**。R2IF 和 CGI 保留镜面 PBR 与互反射优势，同时避免错误间接光污染。
2. **非反射场景更鲁棒**。低频辐照与 NCIF 避免高频环境贴图在低反射场景中退化为噪声。
3. **阶段训练更平滑**。PCC 缓解硬重置和渲染器切换造成的指标突降。
4. **大规模真实场景更适配**。空间辐照 cell、曝光校准和背景门控提高真实图像下的 photometric 稳定性。
5. **物理可解释性更强**。NCIF 有零初始化、有界响应和材质门控，不是任意颜色网络。

---

## 13. 最终论文叙事摘要

本文的最终叙事可概括为：

> 混合反射场景中的高斯逆渲染并非单纯的外观拟合问题，而是一个物理因子可观测性不均衡的问题。高反射区域可以监督镜面环境光和法向，低反射区域则缺乏可靠的高频环境光梯度，真实大场景还伴随曝光和空间照明变化。PhysNorm-GS 通过可靠反射引导照明因子分解、法向耦合辐照场、阶段一致连续优化和置信度门控互反射，将照明、材质、几何与间接光的解释责任显式分配到最可靠的观测区域，从而统一提升反射与非反射场景中的几何和渲染质量。
