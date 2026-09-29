# 变更记录（接口）

## 2026-09-29 变分压缩接口统一为三元输出 `(result, envs, info)`

### 变更内容

`mult` / `compress` / `hadamard` 及其 in-place 版本（`mult!` / `compress!` /
`hadamard!`）的返回值从二元组 `(result, info)` 统一改为三元组
`(result, envs, info)`：

```julia
mult(W, ψ, alg::Union{VOMPS,IDMRG})     -> (y::CanonicalIMPS, envs, info)
mult(W, W2, alg::Union{VOMPS,IDMRG})    -> (y::CanonicalIMPO, envs, info)
mult!(out, W, ψ, alg)                   -> (out, envs, info)
mult!(out, W, W2, alg)                  -> (out, envs, info)

compress(x::CanonicalIMPS, alg)         -> (x′::CanonicalIMPS, envs, info)
compress(W::CanonicalIMPO, alg)         -> (x′::CanonicalIMPO, envs, info)
compress(W::DenseIMPO, alg)             -> (x′::CanonicalIMPO, envs, info)
compress!(out, ψ/W, alg)                -> (out, envs, info)

hadamard(ψ₁, ψ₂, alg)                   -> (y::CanonicalIMPS, envs, info)
hadamard!(out, ψ₁, ψ₂, alg)             -> (out, envs, info)
```

- `envs`：变分引擎的最终环境缓存，对返回结果的射线解出——
  - `mult` / `mult!`：MPO 施加 / MPO 组合通道的 `MultCache`
    （rank-3 环境 `(below, w, above)` / `(above, w, below)`）；
  - `compress` / `compress!`：纯重叠通道的 `OverlapCache`
    （MPO 压缩在 `vectorize` 的 MPS 视图上演化，envs 持该 MPS 视图）；
  - `hadamard` / `hadamard!`：zip 通道的 `HadamardCache`。
- `info`：[`IterativeConvergenceInfo`](@ref)（`niter` 扫掠轮数、`losses`
  逐轮残差/漂移、`converged` 收敛标志），语义与上一版一致：
  - VOMPS 引擎：`losses = [初始 Galerkin 残差, 逐轮 Galerkin 残差...]`；
  - IDMRG 引擎：`losses = 逐轮 bond-0 中心矩阵漂移`（无初始假值）。

### 特殊路径

- `compress` 的 `D ≥ max_bonddim` 短路精确路径：0 轮、`converged = true`，
  `envs` 为恒等矩阵环境（AL/AR 规范下 ⟨x|x⟩ 通道的不动点）。
- in-place 版本的结果写回 `out` 后返回 `(out, envs, info)`：`envs` 为引擎
  对优化态（写回前的内部终态，与 `out` 同射线）解出的环境。
- `mult`（mpo·mps）的返回态经 `CanonicalIMPS(y.AL, y.C[end])` 重右正则化，
  `envs` 与该规范代表可能有规范差但同射线。

### 迁移指引

```julia
# 旧（二元组）
y, info = mult(W, ψ, alg)
# 新（三元组）
y, envs, info = mult(W, ψ, alg)
# 只要结果
y = first(mult(W, ψ, alg))
```

内部驱动函数 `_mult` / `_compress` / `_hadamard` 同步改返三元组；底层扫掠
引擎（`_vomps_sweeps` / `_idmrg_sweeps` / `_zip_vomps_sweeps` /
`_zip_idmrg_sweeps` / `_overlap_vomps_sweeps` / `_overlap_idmrg_sweeps`）
本就返回 `(x, envs, info)`，未变。

### 背景（同批次早前提交 28f9ab4）

- `IterativeConvergenceInfo(niter, losses, converged)` 取代 `iters::Ref` /
  标量 `ϵ` 返回约定；六个变分引擎去掉 keywords，改为接收
  `alg::VOMPS` / `alg::IDMRG` 位置参数；`find_groundstate`（VUMPS/IDMRG）
  返回 `(ψ, envs, info)`。
- 环境固定点核 `_zip_fixedpoints` → `hadamard_fixedpoints`、
  `_overlap_fixedpoints` → `overlap_fixedpoints`，内部统一为
  `fixedpoint(op, v0, :LM, alg)` 分派（与 `mixed_fixedpoints` 同款）。
