# DDGI x PRT：Gaussian Radiance Transfer 技术备忘

## 1. 核心关系

DDGI-style probe grid 与 PRT 的共同基础是球谐函数。Probe 存储空间位置处的入射辐射 SH 系数，PRT transfer 存储表面接收方向光的 SH 系数。二者可以通过内积得到间接光照：

$$
L_{\mathrm{ind},i}
=
\left\langle T_i,C(\mathbf{x}_i)\right\rangle
=
\sum_{k=1}^{K}T_{i,k}C_k(\mathbf{x}_i).
$$

- $C(\mathbf{x}_i)\in\mathbb{R}^{K\times3}$：由 probe grid 三线性插值得到的局部入射辐射 SH 系数。
- $T_i\in\mathbb{R}^{K}$：第 $i$ 个 Gaussian 的传输函数 SH 系数。
- $K=9$：当前工程默认使用二阶 SH。

需要注意：若已经在反射方向 $\omega_r$ 上 evaluate 得到 RGB 值 $L_i(\mathbf{x},\omega_r)$，它就不能再与 $T_i$ 做 SH 内积。方向形式应写为：

$$
L_{\mathrm{ind},i}^{dir}
=
V_i(\omega_r)L_i(\mathbf{x}_i,\omega_r).
$$

因此工程提供三种 `grt_mode`：

1. `dot`：PRT 内积形式，论文默认主公式。
2. `directional`：反射方向 probe radiance 乘 transfer visibility。
3. `hybrid`：二者平均，用于混合材质消融。

## 2. 为什么要替换 Ref-Gaussian 的 learned indirect

Ref-Gaussian 的间接光可写为：

$$
L_{\mathrm{ind}}
=
\sum_i l_{\mathrm{ind},i}\alpha_iT_i^\alpha.
$$

其中 $l_{\mathrm{ind},i}$ 是每个 Gaussian 额外学习的视角相关颜色。它的问题是：

1. 每个 Gaussian 独立记忆，缺少跨区域能量传播。
2. 无法对应真实入射辐射 $L_i(\omega)$，光照编辑困难。
3. 视角外推时容易产生 reflection drift。

GRT 的替换方式是：

$$
l_{\mathrm{ind},i}
\Rightarrow
\tilde L_{\mathrm{ind},i}
=
(1-\tau_t)l_{\mathrm{learned},i}
+
\tau_t\left\langle T_i,C(\mathbf{x}_i)\right\rangle.
$$

其余 Ref-Gaussian 管线保持不变，包括：

- 2DGS geometry；
- Disney/GGX BRDF；
- BVH visibility；
- direct/indirect specular composition；
- alpha blending。

## 3. 工程落地

### 3.1 DDGI Probe Radiance

Probe grid 存储：

```text
probe_grid: [R, R, R, K, 3]
```

对 Gaussian 位置 $\mathbf{x}_i$ 三线性插值得到：

```text
C_i: [K, 3]
```

### 3.2 Gaussian PRT Transfer

每个 Gaussian 存储：

```text
grt_transfer: [N, K]
```

默认初始化使用完整 TSDF/BVH visibility fit：

```text
1. Ref-Gaussian 抽取 TSDF mesh 并构建 BVH；
2. 每个 Gaussian 沿法向半球采样若干方向；
3. BVH trace 得到可见性 V_i(w)；
4. 最小二乘拟合 SH transfer T_i。
```

当 mesh/BVH 尚不可用或显式关闭 `use_grt_visibility_init` 时，工程回退到 DC 初始化 `T_i[0]=1, T_i[1:]=0`。

### 3.3 Gaussian 级计算

GRT 在 Gaussian 级别计算：

```text
grt_rgb_i = sum_k grt_transfer[i,k] * probe_coeff[i,k,:]
```

随后作为 Gaussian feature 进入 rasterizer。这样额外复杂度为 $O(NK)$，避免像素级 probe 查询导致训练时间大幅增加。

## 4. 可视化输出

工程会在训练和 eval 中输出：

- `grt_map`：最终进入 indirect light 的 GRT 间接光；
- `grt_probe_map`：反射方向 probe radiance；
- `grt_visibility`：transfer 在反射方向上的可见性/接收强度。

这些图用于判断 GRT 是否真的学到空间光照与传输结构，而不是退化为全灰或任意残差。

## 5. 当前实验入口

固定 PCC 基准：

```bash
SYNTH_ITERS=30000 PCC_KEEP=0.65 bash train_step_11_pcc_full.sh
```

GRT 关键场景参数搜索：

```bash
bash train_step_12_grt.sh
GRT_MODE=hybrid GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_MODE=directional GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_MODE=dot GRT_TAU=0.25 bash train_step_12_grt.sh
PROBE_RES=4 GRT_MODE=dot GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_VIS_RAYS=96 GRT_MODE=dot GRT_TAU=0.5 bash train_step_12_grt.sh
GRT_MODE=dot GRT_TAU=0.5 bash train_step_13_grt_full.sh
```
