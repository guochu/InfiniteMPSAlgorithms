# 变更记录（接口）

## 2026-09-29 overlap.jl 并入 compress.jl

纯重叠通道（`OverlapCache`、`overlap_fixedpoints`、`Overlap_AC/C_Hamiltonian`、
`recalculate!`、`normalize_envs!`、`transfer_leftenv!/transfer_rightenv!` 的
OverlapCache 方法、`_galerkin_err` 的 OverlapCache 方法、
`_overlap_vomps_sweeps`/`_overlap_idmrg_sweeps`）整体并入
`algorithms/arithmetics/compress.jl`（文件末尾章节），`overlap.jl` 删除，
主文件相应去掉其 include。接口无变化，仅文件归位。

## 2026-09-29 OverlapCache 通道 Hamiltonian 化 + `CompressionEnvironments` 层次

- `_mapAC`/`_mapC`（纯重叠通道 rank-2 局部投影）由
  `Overlap_AC_Hamiltonian`/`Overlap_C_Hamiltonian`（只存 leftenv/rightenv，
  线性映射 `h(x) = GL·x·GR`）替代；装配入口 `AC_Hamiltonian(site,
  envs::OverlapCache)` / `C_Hamiltonian(site, envs::OverlapCache)`，作用对象为
  `ket.AC[site]` / `ket.C[site]`（C 的左环境取 `site + 1`，与 MPO 版约定一致）；
  扫掠局部更新、`_galerkin_err`、`_overlap_fixedpoints` 归一化、环境重标定全部
  改经该接口；
- `_normalize_overlap_envs!` 改名 `normalize_envs!`（与 DMRGCache 版同函数的
  分方法，3 参 `(envs, x, ket)`）；
- `OverlapCache` 构造器将 bra/ket 提升到环境标量类型（`_promote_scalar`），
  保证缓存内所有 fields 同一浮点类型；
- 新增 `algorithms/arithmetics/envs.jl`：
  `abstract type CompressionEnvironments <: Environments`，`OverlapCache`/
  `MultCache`/`HadamardCache` 改继承之（哈密顿量通道 DMRGCache 不变），
  已导出。

## 2026-09-29 AC 有效哈密顿量保 Schur 稀疏结构；OverlapCache 新增 `recalculate!`

- `AC_hamiltonian` 不再稠密化：`SparseIMPO` 的 Schur 张量原样存入
  `MPO_AC_Hamiltonian.operators`，AC 作用按 level 对 (i, j) 逐块收缩
  （Schur 上三角零块 `iszero` 跳过，语义与稠密收缩逐位一致）；稠密
  `DenseIMPO` 路径不变（两作用方法签名互斥）；
- 新增 `recalculate!(envs::OverlapCache, newbra, [alg_environments])`：
  为更新后的 bra 重解 ⟨bra|ket⟩ 不动点（ket 不变），边界 eigsolve 以当前
  `lefts[1]`/`rights[end]` 热启动（对标 MPSKit 原地 `recalculate!`）；
  `newbra` 与新不动点就地写回 `envs`（纯原地更新，返回 `envs` 本身）。
  `_overlap_vomps_sweeps` 的逐迭代环境重解与 `_overlap_idmrg_sweeps` 的收尾
  环境重解改为调用该函数。

## 2026-09-29 环境机制文件归位；`AC_hamiltonian`/`C_hamiltonian` 固化 `DMRGCache`

- `DMRGCache` 的定义与两个构造器（`DenseIMPO` 转移矩阵主本征向量版、
  `SparseIMPO` 逐 level 线性解版）及 `recalculate!` 从
  `algorithms/groundstates/idmrg.jl` 移到 `algorithms/groundstates/envs.jl`
  （与同属该缓存的 `_left/right_cyclethrough!`、`normalize_envs!`、
  `transfer_leftenv!/transfer_rightenv!` 同处）；
  `envs.jl` 中 `normalize_envs!`/`transfer_leftenv!`/`transfer_rightenv!`
  的 `envs::Environments` 固化为 `envs::DMRGCache`（operator 槽相应收窄为
  `Union{DenseIMPO,SparseIMPO}`，恒等分支删除）；
- `effective.jl` 从 `src/` 移到 `src/algorithms/groundstates/`，主文件在其
  紧跟 `envs.jl` 之后 include；`AC_hamiltonian`/`C_hamiltonian` 的
  `envs::Environments` 固化为 `envs::DMRGCache`（恒等通道的 rank-2 环境升维
  入口 `_to3` 随之删除）；恒等有效哈密顿量测试改用恒等 MPO 张量 +
  手工恒等矩阵环境的 `DMRGCache` 表达；
- 文件布局不变式：groundstates/ 现在承载「环境求解原语 + DMRGCache +
  有效哈密顿量 + VUMPS/IDMRG」，idmrg.jl 只剩 IDMRG 算法本体。

## 2026-09-29 IDMRG 加回 `alg_gauge`；`MPO_AC_Hamiltonian` 不再接受 `nothing`

### IDMRG 参数对象

`IDMRG` 新增 `alg_gauge` 字段（默认 `Defaults.alg_gauge()`，动态容差），**专用**
于收尾的 AR 混合规范重建（`updatetol(alg_gauge, iter, ϵ)` 适配，对标 MPSKit
`InfiniteMPS(mps.AR)`）——IDMRG 扫掠内无规范固定步，`alg_gauge` 不参与迭代。
`alg_environments` 保持上一条的语义：提供四个 IDMRG 引擎（groundstate
`find_groundstate`、mult/zip/overlap 通道 IDMRG 扫掠）的初始/收尾环境求解。

### 有效哈密顿量

`MPO_AC_Hamiltonian` 的 `operator` 槽不再允许 `nothing`
（`O<:Union{MPOTensor,SchurMPOTensor}`）：`AC_hamiltonian` 要求传入真实算符
（`SparseIMPO` Schur 张量照旧在构造时稠密化）。恒等通道的有效哈密顿量改用
恒等 MPO 表达：`AC_hamiltonian(site, ψ, identityimpo(T, cell), ψ, envs)`。
`MPO_C_Hamiltonian` 与 `C_hamiltonian`（5 参，`operator` 忽略）不变。

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
