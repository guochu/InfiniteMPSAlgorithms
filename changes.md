# 变更记录（接口）

## 2026-10-02 `svdguess_mult(W, W2, D)` 直接返回 `CanonicalIMPO`

- 修正包装层返回类型与输入域的不一致：mpo·mpo 的乘积是算符，原实现却以
  融合 (u·d) 物理腿的 `CanonicalIMPS`（vectorize 视图）返回，迫使调用方
  （`mult`）再 `devectorize`。现 `devectorize` 收进 `svdguess_mult` 内部——
  流式构造仍在融合视图上进行（键截断与 `gaugefix!` 都在该视图），规范化后
  按 `devectorize` 的互逆纯 reshape 拆回 (u, d)，原生 MPO 形态返回
  `CanonicalIMPO`（规范家族逐位携带，与旧表达式逐位相等）；`mult(W, W2, alg)`
  调用点相应简化；mpo·mps 方法（返回态 `CanonicalIMPS`）与裸张量串方法不变。

## 2026-10-02 `mult!`/`compress!`/`hadamard!` 类型提升兜底：通道解无法写回 `out` 时首个返回值为更新 bra

- `_copyinto!`（写回家族）加类型守卫：`out` 的标量类型无法表示 `y` 时（实
  `out` 遇复提升的通道解——复输入或 leading vector 复化使通道算术类型为复；
  缓存构造器把 bra 槽提升到环境 eltype，引擎内部恒同型演化，写回前才发现
  不可表示），不修改 `out`、直接返回 `y`（`envs.bra`）本身——
  `mult!`/`compress!`（MPS/MPO 版）/`hadamard!` 的第一个返回值相应为提升后的
  更新 bra（`envs.bra`），不再是原 `out`（`out` 仅保留 `changebond!` 预处理
  效果）；docstring 同步；
- 回归测试（arithmetics.jl）：实 out + 复通道的 `mult!`（mpo·mps 双引擎 /
  mpo·mpo）、`compress!`、`hadamard!` 断言 `result === envs.bra`、结果为复、
  `out` 存储保持实型、射线与严格参考一致。

## 2026-10-02 `randomimpo` 返回 `CanonicalIMPO`（与 `randomimps` 对齐）

- `randomimpo` 与 `randomimps` 同款：随机张量串构造后立即经 `CanonicalIMPO`
  构造器规范存储，返回类型由 `DenseIMPO` 改为 **`CanonicalIMPO`**（构造即混合
  规范）；`kwargs...` 透传 `gaugefix`（如 `tol`/`maxiter`）；
- 需要原始（未规范化、未定规范）张量串的测试改用新增测试助手
  **`rand_denseimpo`**（test/testhelpers.jl）；严格代数（`*`）、getindex、
  `_dense_mpo_repr` 等只有 Dense 表示参与的测试点相应改用该助手。

## 2026-09-30 IDMRG 扫描语义对齐 MPSKit（原地 AC 覆盖）；groundstate concordance 改固定小迭代行为一致性测试

- `_localupdate_sweep_idmrg!` 对齐 MPSKit 的两处扫描语义：分裂因子正对角化
  （`QRpos`，同 `left_orth!/right_orth!(; positive = true)`，本就一致）；分裂后把
  `AC[pos]` 原地覆盖为对应的 `AL`/`AR`（MPSKit `left_orth!`/`right_orth!` 原地
  语义；前向末站/反向首站保留 AC，同其非原地特例）——`AC` 是下一站 `:SR`
  eigsolve 的初值，动态容差松弛时初值差异直接进入迭代轨迹（Heisenberg k=1
  态偏差 3.4e-3 → 2.3e-3）；
- groundstate concordance 测试改**固定小迭代行为一致性**（不再跑收敛）：同初
  态、`tol = 0` 强制 k ∈ {1, 2, 5} 轮，断言能量与态一致。实测（k=1,2,5）：
  VUMPS 能量差 ~1e-14、态 ray 残差 ~1e-12（机器精度级严格一致，三通道
  TFIM/Heisenberg/DenseIMPO）；IDMRG 能量差 ≤ 3e-4、态残差 ≤ 4e-3 且随 k 收
  敛到同一不动点（Gauss–Seidel 顺序扫描 + 初始环境中间层非对角通道 linsolve
  解方向的 ⟨H⟩-等价差异；容差按实测给 1e-3/1e-2）；DenseIMPO 通道能量差
  ≤ 2e-6、态不比对（恒等层 bookkeeping 使转移矩阵主导本征空间简并，环境取向
  不由归一化唯一确定）；`niter` 断言放宽为 `≤ k`（Dense 通道 IDMRG 会在少数
  几轮内到达精确不动点——C 漂移恰为 0 合法提前收敛）。该 testset 从 ~100s
  降至 ~9s。

## 2026-09-30 `svdguess_*` kwargs；严格 hadamard 改重载 `⊙`；norm/fidelity/distance kwargs；`find_groundstate` 支持 DenseIMPO；`infinite_mpo` 删除

- `svdguess_compress`/`svdguess_mult`/`svdguess_hadamard` 的
  `CanonicalIMPS`/`CanonicalIMPO` 方法补 `kwargs...`（透传末端 `CanonicalIMPS`
  构造器 → `gaugefix!`，如 `tol`/`maxiter`）；bare-tensor 张量串方法不变；
- 严格 hadamard 运算 `hadamard(::DenseIMPS, ::DenseIMPS)` 改为重载
  FiniteMPSAlgorithms 的 unicode 算符 **`⊙(::DenseIMPS, ::DenseIMPS)`**（已
  再导出；lcm 单胞语义不变）；`hadamard` 名字保留给三参数变分压缩
  `hadamard(ψ1, ψ2, alg)`；
- `norm`（DenseIMPS）/`fidelity`/`infidelity`（DenseIMPS/CanonicalIMPS/
  CanonicalIMPO）/`distance`/`distance2`（DenseIMPS）补 `kwargs...`（透传内部
  `dot`，如 `krylovdim`）；
- `find_groundstate` 支持 `DenseIMPO` 哈密顿量输入（VUMPS/IDMRG，对标 MPSKit
  同样接受 `InfiniteMPO` 的设计：周期 trace 收缩 + 转移矩阵主本征向量环境；
  该通道的期望含恒等层 bookkeeping、非真实能量，求能量仍用 `SparseIMPO` 闭列
  公式）；原 ArgumentError 报错桩删除；与 MPSKit InfiniteMPO 通道的能量一致性
  由 concordance 测试固化（TFIM 稠密哈密顿量，VUMPS/IDMRG 双算法）；
- `infinite_mpo` 删除：单个 Schur bulk 的周期 DenseIMPO 用
  `DenseIMPO([tompotensor(bulk)])` 表达（原「D 并入恒等通道」布局与
  tompotensor 的 Schur 布局在含 NN 项的 bulk 上语义不同）；`models.jl` 三个
  模型的 `mpo` 字段与测试改用 tompotensor 表示。

## 2026-09-30 SparseIMPO 撤下家族视图（AL/AR/AC/C）

- 删除 `SparseIMPO` 的 `getproperty` 家族接口：`.AL`/`.AR`/`.AC` 不再按站稠密化
  （`tompotensor`），`.C` 不再提供 BondView 单位矩阵视图——Schur 块访问一律走
  `H[i][j, k]`（getindex）、内部遍历一律走 `H.Ws`；变分算术通道
  （compress/mult/hadamard）不支持 `SparseIMPO` 输入（需要 dense/canonical
  表示时显式 `DenseIMPO(H)`/`CanonicalIMPO(...)` 转换，或走其 Schur 专用通道：
  DMRGCache/expectationvalue/make_time_mpo）；`AbstractInfiniteMPO` 的 docstring
  同步（家族接口由 DenseIMPO/CanonicalIMPO 提供）；api.jl 家族接口测试相应调整
  （`H.AL` 现按 `TypeError` 断言）。

## 2026-09-30 `AbstractInfiniteMPS`/`AbstractInfiniteMPO` 家族接口；dense 类型直入变分通道；严格代数 lcm 单胞

- 新增抽象类型 **`AbstractInfiniteMPS{T<:Number}`**（`CanonicalIMPS`/`DenseIMPS`
  继承）与 **`AbstractInfiniteMPO{T<:Number}`**（`CanonicalIMPO`/`DenseIMPO`/
  `SparseIMPO` 继承）+ **`BondView{P,T} <: AbstractVector{Matrix{T}}`**，均已
  导出。**`scalartype` 泛型定义在抽象类型上**
  （`scalartype(::Type{<:AbstractInfiniteMPS{T}}) = T` 及 MPO 同款）——五个具体
  类型的 `scalartype` 定义随之删除；`BondView` 的 `parent` 字段持有**链/算符
  本身**，`getindex` 经 `parent.AL` 推断键维（`C[ℓ]` 为 `(site ℓ 右键维,
  site ℓ+1 左键维)` 的恒等方阵，周期取模下标，只读）；
- `DenseIMPS`/`DenseIMPO`/`SparseIMPO` 经 `getproperty` 增加 **`AL`/`AR`/`AC`/`C`
  家族访问**（规则同 [`CanonicalIMPS`](@ref)）：`AL`/`AR`/`AC` 即原始张量串
  本体（`SparseIMPO` 按站稠密化 `tompotensor`），`C` 返回 `BondView` 单位矩阵
  视图——三个类从此可与 `CanonicalIMPS`/`CanonicalIMPO` 同款消费（`ψ.AL[ℓ]`/
  `ψ.C[ℓ]` 周期下标）。**数据存储改为 `PeriodicVector`**（`DenseIMPS.As`/
  `DenseIMPO.Ws` 原 `Vector`，构造器收 `Vector`/`PeriodicVector` 皆可）；
  **行为变更**：`SparseIMPO` 的 `.C` 从「Schur C 块数组」改为该单位矩阵视图，
  且**单类型参数化** `SparseIMPO{T}`（field `Ws::PeriodicVector{SchurMPOTensor{T}}`，
  原 `SparseIMPO{TO}`）——`Base.parent(H)`/`.A`/`.B`/`.D` 属性删除，内部与
  块访问一律走 `H.Ws`/`H[i][j, k]`；
- `compress`/`compress!`/`mult`/`mult!`/`hadamard`/`hadamard!` 的输入放宽为
  `AbstractInfiniteMPO`/`AbstractInfiniteMPS`：`DenseIMPO`/`CanonicalIMPO` 与
  `DenseIMPS`/`CanonicalIMPS` 皆可直接输入，dense 类型经家族视图直接参与环境
  与局部映射（**不再强制转换**），**输出类型恒为 `CanonicalIMPO`/
  `CanonicalIMPS`**；缓存槽位（`MultCache` 的 operator/ket、`OverlapCache` 的
  ket、`HadamardCache` 的 ket1/ket2）同步放宽并在原类型上做标量提升
  （`_promote_scalar` 新增 Dense 方法）；原先的「raw 目标先规范化」包装方法
  （`compress(::DenseIMPS)`、`compress!(out, ::DenseIMPS/::DenseIMPO)`、
  `mult!(out, W, ::DenseIMPS)`）随之删除（泛型方法直接覆盖）；
  `svdguess_mult`/`svdguess_hadamard`/`svdguess_compress` 同步接受抽象类型
  （初态恒为 `CanonicalIMPS`）；`mult`/`hadamard` 的等长/整除检查不变；
- **kwargs 透传**：`randomimps(...; kwargs...)` 与
  `CanonicalIMPS(ψ::DenseIMPS; kwargs...)`（`CanonicalIMPO(W::DenseIMPO;
  kwargs...)` 已有）把 kwargs 透传末端 `gaugefix!`（如 `tol`/`maxiter`），
  不再强制默认参数；
- **严格代数的 lcm 单胞**：`DenseIMPO * DenseIMPS` 与严格
  `hadamard(::DenseIMPS, ::DenseIMPS)` 的输出单胞长度 = 两输入单胞的**最小
  公倍数**（逐周期平铺收缩；逐站物理维仍须一致）；同单胞输入行为不变；
  `DenseIMPO * DenseIMPO` 维持整除要求不变；
- 测试：家族接口/继承关系/BondView 单位矩阵/kwargs 透传（api.jl）、lcm 单胞
  与逐周期平铺语义（api.jl）、compress/mult/hadamard 的 dense ≡ canonical
  等价性（同射线输入 → 同变分结果、输出类型恒 Canonical，arithmetics.jl）。

## 2026-09-30 扫掠引擎改名 `compression_sweeps!`；`Environments` 的 `length` 契约

- `_compression_sweep!` 改名 **`compression_sweeps!`**（无下划线前缀；两个泛型
  方法 `(envs::CompressionEnvironments, alg::Union{VOMPS,IDMRG}) -> (envs,
  info)` 不变）。引擎收尾已保证 `envs.bra` 按包约定归一化且处于混合规范，
  公开层不再重复收尾：`mult`（mpo·mps）删除返回前的
  `CanonicalIMPS(y.AL, y.C[end])` 重右正则化 + `_global_normalize!`，直接返回
  `envs.bra` 本体；`mult!`（mpo·mps）相应改为 `_copyinto!(out, envs.bra)`；
  `compress`/`compress!`/`hadamard`/`hadamard!`/`mult`（mpo·mpo）此前已直连
  引擎，仅随改名；
- **`Environments` 的 `length` 契约**：每个子类必须定义
  `Base.length(envs)` = 环境的单胞长度 = 输入态/算符单胞长度的**最小公倍数**
  （环境数组 `lefts`/`rights` 的元素个数、以及该通道输出量 `out` 的长度都
  等于这个数）；泛型 `leftenv`/`rightenv` 的取模周期从 `length(envs.ket)` 改
  为 `length(envs)`（environments.jl），各缓存方法内部的
  `length(envs.ket)`/`length(envs.ket1)` 统一为 `length(envs)`；
  `HadamardCache` 的 `leftenv`/`rightenv` 覆盖删除（泛型方法已覆盖）；
  `DMRGCache` 补 `Base.length(envs) = length(envs.ket)`（operator 单胞恒为
  ket 的因子，lcm = `length(ket)`）；`CompressionEnvironments` 的泛型
  `length`（= `length(envs.bra)`）即 lcm 契约的实现（构造时已保证 bra/ket
  同长、operator 长度为其因子）；
- 测试固化该行为（test/algorithms/api.jl 新增 testset）：四个具体缓存
  （DMRGCache/OverlapCache/MultCache/HadamardCache，N=2 链 × L=1 算符 ⇒
  length = 2）、环境数组同长、N=L 不放大、N=1 为 1、`leftenv`/`rightenv`
  以 `length(envs)` 为周期取模、mult/compress 输出量长度 = 环境单胞。

## 2026-09-29 重叠通道局部映射回归 `_mapAC`/`_mapC`；删严格 `mult(W, ψ)`；`calc_galerkin` 改名；`_compression_sweep!` 移入 envs.jl

- 删除 `Overlap_AC_Hamiltonian`/`Overlap_C_Hamiltonian` 及其装配入口
  `AC_Hamiltonian(site, envs::OverlapCache)`/`C_Hamiltonian(site,
  envs::OverlapCache)`：无算符通道的局部投影回归更简单的 rank-2
  `_mapAC(GL, GR, ketac)`/`_mapC(GL, GR, ketc)`（与三元/zip 通道同名分派），
  `_local_AC`/`_local_C` 与 `overlap_fixedpoints` 的归一化直接调用之；
- 删除严格 `mult(W, ψ)`（二参数精确朴素构造）——等价表达
  `CanonicalIMPS(collect(W * DenseIMPS(collect(ψ.AL))))`（canonicalize + 归一）；
- `_galerkin_err(envs, x)` 改名 **`calc_galerkin(envs, x)`**（对齐 MPSKit
  命名；单站投影范数 `_galerkin` 不变）；
- `_compression_sweep!`（两个泛型方法）从 compress.jl 移到
  arithmetics/envs.jl（`CompressionEnvironments` 层次所在处）。

## 2026-09-29 统一扫掠引擎 `_compression_sweep!`；mult/hadamard 直连引擎

- `_overlap_vomps_sweeps!`/`_overlap_idmrg_sweeps!`（compress 通道）、
  `_vomps_sweeps`/`_idmrg_sweeps`（mult 通道）、`_zip_vomps_sweeps`/
  `_zip_idmrg_sweeps`（hadamard 通道）六个扫掠函数统一为
  **`_compression_sweep!(envs::CompressionEnvironments, alg::Union{VOMPS,IDMRG})
  -> (envs, info)`**（两个泛型方法，按缓存与 alg 分派；`(envs, alg)` 原地契约
  不变）：通道差异收敛到逐缓存的 `_local_AC`/`_local_C`（局部投影）、
  `recalculate!`（环境重解）、`transfer_leftenv!/transfer_rightenv!(envs, x,
  site)`、`normalize_envs!(envs, x)`、`_galerkin_err(envs, x)`、
  `_finalize_target(envs)`（finalize 回调的通道目标）上；
- `mult`/`mult!`/`hadamard`/`hadamard!` 直接构造缓存并调用
  `_compression_sweep!`（内部驱动 `_mult`/`_hadamard` 删除）；
  `mult!`/`hadamard!` 的 `changebond!` 改用 `D = alg.D`（与 `compress!` 一致，
  `max_bonddim` 不再用于此）；
- `MultCache`/`HadamardCache` 构造器把各槽位提升到通道标量类型（环境 eltype，
  同 OverlapCache）——扫掠对缓存 bra 的原地演化恒在同型算术上进行；
  `CompressionEnvironments` 新增泛型 `Base.length`/`scalartype`；
- IDMRG 收尾的 AR 混合规范重建统一改经 `_rebuild`（compress.jl 此前直接构造
  `CanonicalIMPS`）。

## 2026-09-29 compress/compress! 直连扫掠引擎；`_compress`/`_compress_ket` 删除

- 删除内部驱动 `_compress`（三种输入方法）与 `_compress_ket`（及其独占的
  `_truncate_bonddim` 初猜）：`compress`/`compress!` 直接基于
  `_overlap_vomps_sweeps!`/`_overlap_idmrg_sweeps!` 实现——构造
  `OverlapCache(初态, 目标链, alg.alg_environments)` 后调用对应扫掠，最终态即
  `envs.bra`（`compress` 用 `svdguess_compress(·, alg.D)` 作初态）；
- `compress!` 的 `changebond!` 改用 `D = alg.D`（此前取 `max_bonddim(out)`）
  ——`max_bonddim` 函数保留；
- `compress` 不再保留 `alg.D ≥ max_bonddim` 的短路精确路径（`D` 超出输入键
  时扫掠一步即收敛，行为不变）；初态对象不再被原地修改（引擎在缓存持有的
  副本上演化）。

## 2026-09-29 重叠通道扫掠改原地；OverlapCache 的 `length`/`scalartype`

- `_overlap_vomps_sweeps!`/`_overlap_idmrg_sweeps!` 改为原地版本：
  签名 `(envs::OverlapCache, alg) -> (envs, info)`——被优化的态即 `envs.bra`
  （构造缓存的初态），扫掠全程原地演化并写回缓存（环境经 `recalculate!`
  热启动重解）；`finalize` 回调返回的态写回缓存 bra 后继续使用；
- `OverlapCache` 新增 `Base.length`（bra/ket 单胞长度）与 `scalartype`
  （构造器统一提升后的缓存标量类型）。

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
