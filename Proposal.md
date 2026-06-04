# GRT-GS：基于 DDGI 探针与 PRT 传输的物理一致反射高斯逆渲染

## 摘要

本文研究反射场景中 Gaussian Splatting 逆渲染的间接光照建模与训练稳定性问题。现有反射高斯方法通常使用可学习的高斯级视角相关颜色表示间接光，并将其与环境贴图、BRDF 材质、法向和局部光线追踪共同优化。该设计能够拟合训练视角，但间接光缺乏空间连续性和物理含义：每个 Gaussian 独立记忆颜色，难以表达跨区域能量传递；视角外推时容易出现反射漂移；在光照或材质编辑时也无法对应真实入射辐射。

本文提出 **GRT-GS, Gaussian Radiance Transfer for Reflective Gaussian Splatting**。核心思想是用传统图形学中的两类互补表示替代学习式间接光颜色：**DDGI-style probe grid** 表示空间连续入射辐射场，回答“场景中哪里有多少光”；**Gaussian PRT transfer** 表示每个 Gaussian 对方向入射光的接收和遮蔽能力，回答“该表面如何接收这些光”。二者在球谐基下统一为内积：

$$
L_{\mathrm{ind},i}
=
\left\langle
T_i,\,
C(\mathbf{x}_i)
\right\rangle
=
\sum_{k=1}^{K} T_{i,k}\,C_k(\mathbf{x}_i),
$$

其中 $C(\mathbf{x}_i)$ 是从探针网格三线性插值得到的局部入射辐射 SH 系数，$T_i$ 是第 $i$ 个 Gaussian 的 SH 传输系数。该间接光直接替换 Ref-Gaussian 中的 learned indirect color，并继续走原有 PBR、BVH visibility、BRDF 和 alpha blending 管线。

训练方面，本文保留 **Phase-Consistent Continuation, PCC**。PCC 将体渲染/延迟渲染阶段切换中的硬重置改为固定比例软连续更新，显著缓解部分场景在 20k 附近或后期出现的指标突降。最终方法由三个创新组成：DDGI Probe Radiance、Gaussian PRT Transfer、PCC。

## 1. 动机与问题

### 1.1 Ref-Gaussian learned indirect 的局限

Ref-Gaussian 的间接光可写为：

$$
L_{\mathrm{ind}}(\mathbf{p})
=
\sum_{i=1}^{N}
l_{\mathrm{ind},i}(\omega_r)
\alpha_i
T_i^{\alpha},
$$

其中 $l_{\mathrm{ind},i}$ 是每个 Gaussian 额外学习的视角相关间接光颜色，$\alpha_iT_i^\alpha$ 是 alpha compositing 权重。该表示有效但存在三个问题：

1. **局部记忆。** 每个 Gaussian 独立存储间接光颜色，没有显式跨区域能量传播。
2. **物理不可解释。** $l_{\mathrm{ind}}$ 不能直接映射为真实入射辐射 $L_i(\omega)$，难以用于光照编辑和材质编辑。
3. **视角外推不稳。** 视角相关颜色在未观测方向容易出现 reflection drift 或闪烁。

### 1.2 DDGI 与 PRT 的互补性

DDGI 和 PRT 都可使用球谐函数表达方向信息，因此可以自然组合。

DDGI-style probe grid 存储局部入射辐射：

$$
L_i(\mathbf{x},\omega)
\approx
\sum_{k=1}^{K}
C_k(\mathbf{x})Y_k(\omega).
$$

PRT 存储表面传输函数：

$$
T_i(\omega)
\approx
\sum_{k=1}^{K}
T_{i,k}Y_k(\omega).
$$

二者内积得到局部间接照明：

$$
L_{\mathrm{ind},i}
=
\int_{\Omega}
L_i(\mathbf{x}_i,\omega)T_i(\omega)d\omega
\approx
\sum_{k=1}^{K}
T_{i,k}C_k(\mathbf{x}_i).
$$

这正是本文的 Gaussian Radiance Transfer。

### 1.3 理论表述中的关键修正

需要区分两个量：

- $C(\mathbf{x})$：probe grid 插值得到的 SH 系数，可与 $T_i$ 做内积。
- $L_i(\mathbf{x},\omega_r)$：在反射方向 $\omega_r$ 上对 SH 做 evaluation 后得到的 RGB 辐射值。

因此，严格的 PRT 内积应写为 $\langle T_i,C(\mathbf{x}_i)\rangle$，不能把已经 evaluate 成 RGB 的 $R_g$ 再与 $T_g$ 内积。工程中保留三种可测试形式：

1. `dot`：$L_{\mathrm{ind}}=\langle T_i,C(\mathbf{x}_i)\rangle$，默认物理内积形式。
2. `directional`：$L_{\mathrm{ind}}=V_i(\omega_r)L_i(\mathbf{x}_i,\omega_r)$，更偏镜面方向查询。
3. `hybrid`：二者平均，用于混合粗糙/镜面场景的消融。

论文主叙事采用 `dot` 形式，`directional/hybrid` 作为实验参数或消融。

## 2. 方法概览

每个 Gaussian $G_i$ 维护几何、材质和传输属性：

$$
G_i=
\{\mathbf{x}_i,\alpha_i,\mathbf{s}_i,\mathbf{R}_i,
\mathbf{a}_i,\rho_i,r_i,\mathbf{n}_i,T_i\}.
$$

其中 $\rho_i$ 是反射强度，$r_i$ 是粗糙度，$T_i\in\mathbb{R}^{K}$ 是 Gaussian PRT transfer。场景中额外维护 DDGI-style probe grid：

$$
\mathcal{P}
=
\{(\mathbf{p}_j,C_j)\}_{j=1}^{M},
\qquad
C_j\in\mathbb{R}^{K\times 3}.
$$

渲染流程：

1. 通过 Ref-Gaussian / 2DGS 主链路获得 Gaussian 几何、法向、材质和 alpha compositing。
2. 对每个 Gaussian 查询 probe grid，得到 $C(\mathbf{x}_i)$。
3. 使用 Gaussian transfer $T_i$ 与 $C(\mathbf{x}_i)$ 计算 GRT 间接光。
4. 用 GRT 间接光替换 learned indirect color。
5. 保持原有 BRDF、环境贴图、BVH visibility、直接光与 alpha blending 不变。
6. 使用 PCC 稳定阶段切换。

## 3. DDGI Probe Radiance Field

### 3.1 Probe Grid

在场景包围盒内建立规则 probe grid。每个 probe 存储二阶 SH 系数：

$$
C_j=
\{c_{j,k}\}_{k=1}^{9},
\qquad
c_{j,k}\in\mathbb{R}^{3}.
$$

对任意 Gaussian 位置 $\mathbf{x}_i$，通过三线性插值得到：

$$
C(\mathbf{x}_i)
=
\mathrm{TriInterp}
\left(\mathcal{P},\mathbf{x}_i\right).
$$

二阶 SH 的低频性质适合表达漫反射和粗糙间接光；高频镜面反射仍由环境贴图和局部 BVH 反射路径负责。

### 3.2 性能设计

Probe 查询在 Gaussian 级别完成，而非像素级完成。得到的 GRT 间接光作为额外 Gaussian feature 进入 rasterizer，随后与其他属性一起 alpha blending。因此每次迭代的额外代价主要为：

$$
O(NK)
$$

其中 $N$ 是 Gaussian 数量，$K=9$ 是 SH 系数数。该复杂度远低于像素级 probe 查询或多次 ray marching。

## 4. Gaussian PRT Transfer

### 4.1 Transfer 表示

每个 Gaussian 学习一组 SH transfer 系数：

$$
T_i=
[T_{i,1},\ldots,T_{i,K}].
$$

它描述该 Gaussian 在不同方向上的入射光接收能力，包含法向、遮挡和局部几何对光照的影响。

### 4.2 TSDF/BVH 可见性初始化

为了让 $T_i$ 从训练开始就具有明确的几何含义，GRT-GS 不再仅使用均匀 DC 初始化，而是在 Ref-Gaussian delayed PBR 阶段首次抽取 TSDF surface mesh 后，直接利用该 mesh 构建 BVH，并为每个 Gaussian 拟合可见性 transfer。

对第 $i$ 个 Gaussian，以其 2DGS 几何法向 $n_i$ 为半球轴采样方向 $\omega_j\in\Omega^+(n_i)$。从 Gaussian 中心沿 $\omega_j$ 发射 BVH 查询光线，得到二值可见性：

$$
V_i(\omega_j)=
\begin{cases}
1,&\text{if no hit within scene range},\\
0,&\text{otherwise}.
\end{cases}
$$

然后用 SH 基 $\mathbf{Y}(\omega)$ 拟合 transfer：

$$
T_i^\star
=
\arg\min_{T_i}
\sum_j
\left(
\mathbf{Y}(\omega_j)^\top T_i
-
V_i(\omega_j)
\right)^2 .
$$

这一步把 TSDF/BVH 的显式几何遮挡转化为每个 Gaussian 的 PRT transfer 初值。后续训练继续优化 $T_i$，但初始化已经具备“哪里能接收光、哪个方向被遮挡”的物理语义。

工程中仍保留 DC 初始化作为回退：当 mesh/BVH 尚不可用，或显式关闭 `use_grt_visibility_init` 时，才使用 $T_{i,1}=1,T_{i,k>1}=0$。主实验默认启用 TSDF/BVH transfer 初始化。

### 4.3 GRT 间接光

默认形式：

$$
L_{\mathrm{GRT},i}
=
\sum_{k=1}^{K}
T_{i,k}C_k(\mathbf{x}_i).
$$

为了训练稳定，实际实现使用 ramp blend 逐步替换 learned indirect：

$$
\tilde L_{\mathrm{ind},i}
=
(1-\tau_t)L_{\mathrm{learned},i}
+
\tau_tL_{\mathrm{GRT},i},
$$

其中 $\tau_t$ 从 `grt_from_iter` 开始在 `grt_ramp_iters` 内线性增加到 `grt_tau`。当 `grt_tau=1` 时，GRT 完全替换 learned indirect。

## 5. Phase-Consistent Continuation

Ref-Gaussian 在从体渲染/初始化阶段切换到 delayed PBR 阶段时，会对颜色和材质属性进行重置。硬重置会导致优化目标突变：

$$
\theta_{\mathrm{old}}
\rightarrow
\theta_{\mathrm{reset}},
$$

从而造成 PSNR 曲线突降。PCC 使用固定保留率 $\kappa$ 进行软连续更新：

$$
\theta_{\mathrm{pcc}}
=
\kappa\theta_{\mathrm{old}}
+
(1-\kappa)\theta_{\mathrm{reset}}.
$$

本文使用固定 PCC 作为稳定训练模块。当前实验中 `PCC_KEEP=0.65` 已作为内部强基准。

## 6. 损失函数

总损失为：

$$
\mathcal{L}
=
\mathcal{L}_{c}
+
\lambda_n\mathcal{L}_{n}
+
\lambda_s\mathcal{L}_{smooth}
+
\lambda_g\mathcal{L}_{GRT}.
$$

颜色损失：

$$
\mathcal{L}_{c}
=
(1-\lambda)\mathcal{L}_1
+
\lambda(1-\mathrm{SSIM}).
$$

GRT 正则由三部分组成：

$$
\mathcal{L}_{GRT}
=
\lambda_{map}\mathcal{L}_{map}
+
\lambda_{probe}\mathcal{L}_{probe}
+
\lambda_{T}\mathcal{L}_{T}.
$$

其中：

- $\mathcal{L}_{map}$：GRT 间接光图的 edge-aware smoothness。
- $\mathcal{L}_{probe}$：probe grid 的能量和空间平滑正则。
- $\mathcal{L}_{T}$：transfer 非 DC 系数能量约束，避免 transfer 退化为任意颜色残差。

当前工程没有默认启用 kNN transfer 邻域正则，因为 kNN 图维护和 densification 后更新会显著增加训练成本。论文中可将其作为可选增强；主实验优先采用轻量正则保证效率。

## 7. 可视化与可解释性

GRT-GS 可直接渲染以下属性图：

1. `grt_map`：最终进入 PBR 间接光链路的 GRT 间接光。
2. `grt_probe_map`：反射方向上的 probe radiance 查询。
3. `grt_visibility`：由 Gaussian transfer 在反射方向上得到的可见性/接收强度。
4. `env1/env2`：远场环境贴图。
5. `specular_map/diffuse_map/roughness/refl_strength/normal`：Ref-Gaussian 原生材质与几何图。

这些图可用于证明方法不只是提高指标，还能给出可解释的光照-传输分解。

## 8. 实验计划

### 8.1 基准

外部基准：

- Ref-Gaussian 30k。

内部强基准：

- Fixed PCC 30k oracle，`PCC_KEEP=0.65`。

后续所有 GRT 实验均与 fixed PCC oracle 对比。

### 8.2 参数搜索

第一轮只跑关键场景，搜索：

1. `grt_mode`: `dot / directional / hybrid`。
2. `grt_tau`: `0.25 / 0.5 / 1.0`。
3. `probe_grid_res`: `4 / 8`。
4. `grt_from_iter`: synthetic 默认 `20000`，Ref-Real 默认 `10000`。
5. `grt_visibility_rays`: `32 / 64 / 96`。
6. `grt_transfer_refresh_interval`: `0 / 2000`，其中 `0` 表示只在首次可用 TSDF/BVH 时初始化一次。

重点场景：

- 反射合成：`GlossySynthetic/bell`, `GlossySynthetic/teapot`, `ShinyBlender/toaster`。
- 弱反射/漫反射：`NerfSynthetic/chair`, `NerfSynthetic/materials`, `NerfSynthetic/ship`。
- 真实场景：`RefReal/gardenspheres`, `RefReal/sedan`, `RefReal/toycar`。

### 8.3 判断标准

1. 平均 PSNR/SSIM/LPIPS 是否超过 fixed PCC oracle。
2. 反射场景不能破坏 PCC 已有收益，尤其 `bell/toaster`。
3. 非反射场景不能显著下降。
4. Ref-Real 若指标不提升，至少应观察到更干净的 `grt_map` 和更稳定的 envmap。
5. 记录训练时间变化，区分一次性 transfer 初始化与周期刷新带来的额外成本。

## 9. 贡献总结

本文贡献为：

1. 提出 **Gaussian Radiance Transfer (GRT)**，将 DDGI-style probe radiance 与 PRT transfer 统一到 Gaussian inverse rendering 中，用显式 transfer-radiance 内积替代 learned indirect color。
2. 提出 **DDGI Probe Radiance Field for 3DGS**，在 Gaussian 级别查询空间连续 SH 入射辐射场，以低成本表达大尺度低频全局光照。
3. 提出 **TSDF/BVH-initialized Gaussian PRT Transfer**，将显式几何可见性投影到每个 Gaussian 的 SH transfer 中，建模局部方向接收能力和遮蔽。
4. 提出并保留 **Phase-Consistent Continuation (PCC)**，以固定软重置缓解 Ref-Gaussian 阶段切换和后期退化问题。
