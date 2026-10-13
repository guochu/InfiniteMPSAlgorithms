# 变更记录（接口）

## 2026-10-13 zip_sweeps! 三通道化 + 严格等价：compress（OverlapCache）接入与 IDMRG2 轨迹级对齐

- **接口**：`zip_sweeps!` 两个方法（IDMRG/IDMRG2）的通道签名拓宽为
  `Union{MultCache, HadamardCache, OverlapCache}`——**compress 驱动经
  IDMRG/IDMRG2 自动走融合快路径**（OverlapCache 新增 8 个融合核：
  `_fuse_left`/`_local_AC_fused`/`_push_left_fused!`/`_fuse_right`/
  `_local_AC_fusedR`/`_push_right_fused!`/`_local_AC2_fused`/
  `_local_AC2_fusedR`，rank-2 矩阵环境 `(bra bond, ket bond)`、bra/键维
  可不同）；dispatch shim 同步拓宽（VOMPS 与 general 模板不受影响）；
- **严格等价（本轮核心）**：zip 与 general `compression_sweeps!` 的 push
  **活写序列逐位对齐**——
  - IDMRG：general 每 pass 本就单方向 push，zip 已严格一致；
  - IDMRG2：general 每窗口双向 push，其中对侧（同 pass 内）的写均为死写
    （值总在被读前覆盖），**唯二跨 pass 活写**为 L→R pass 窗口 1 的右推
    （`rights[1]`——被 L→R 窗口 N 的 map 与 R→L 窗口 N 消费）与 R→L pass
    窗口 N 的左推（`lefts[1]`——被窗口 1 消费）；zip 以逐 pass 单方向
    push + 这两处窗口特例 push 复现全部活写（此前缺这两处，未收敛轨迹
    会缓慢漂移、收敛轮数 ±2）；
- **验证**（[debug/zip_strict.jl]）：三通道 × {IDMRG, IDMRG2} × N ∈ {2, 3}
  共 14 个场景、tol=0 不收敛 6 轮——终态全槽位（AL/AR Gram + C）最大
  6.6e-14、逐轮 losses 最大 1.6e-15（纯浮点收缩序噪声）；收敛解
  fidelity = 1.0 且 **iters 全部严格一致**（compress 44/44、18/18 等，
  [debug/zip_sweeps_check.jl]）；小规模计时 compress IDMRG 1.7×/
  IDMRG2 与 general 持平、mult 1.1–3.3×、hadamard ~2×。

## 2026-10-13 zip_sweeps! 扩展 IDMRG2：MultCache/HadamardCache 的两站融合快路径

- **接口**：`zip_sweeps!` 新增 `(::Union{MultCache, HadamardCache}, ::IDMRG2)`
  方法（[src/algorithms/arithmetics/compress.jl] 末尾）+ dispatch shim
  `compression_sweeps!(::Union{MultCache, HadamardCache}, ::IDMRG2)`——
  mult/hadamard 驱动经 IDMRG2 自动走快路径（shim 与 IDMRG 的分列定义——
  合并为 `Union{IDMRG, IDMRG2}` 签名会与 general 模板产生方法歧义）；
- **机制**：IDMRG 融合的两站推广——L→R pass 每窗口把
  `(GL·operator.AL·ket.AL)`（zip 通道 `GL·A1.AL·A2.AL`）一次融合为 K0，
  折窗口键权重（AC = AL·C）后与 next 站 AR 因子、`rights[next]` 收缩出
  两站 `Θ`（rank-4，物理腿逐站保留），`_svd2!` 按 `alg.trunc` SVD 截断；
  R→L pass 镜像（S0 = `_fuse_right` 在 next 站融合，窗口 C 权重经
  **`_bondCT` 转置折叠**折入 S0 的 AR 基键腿，再与 pos 站 AL 因子收缩）。
  融合张量每窗口双用（map + 单方向 push）；general 模板的对侧 push 为
  死写（其值总在消费前被反向 pass 覆盖）——单方向化不改变任何 map 输入
  （60 轮未收敛轨迹与双 push 版逐位一致）；
- **正确性要点**：R→L 的 S0 键腿在 AR 左键基（= C 的**列**），`_fold_bond`
  消费行 ⇒ 须传 `Cᵀ`（新增 `_bondCT` 辅助，Dense 链恒 nothing）——单站
  `_local_AC_fusedR` 折的是 GL 的 AL 基腿故无需转置，两站镜像 map 折 S0
  侧则必须转置；hadamard 通道 next 站两因子共享开放物理腿 s2（R→L 为 s1）
  ⇒ per-s 切片 GEMM（同单站方案）；
- **等价性**：三通道单步 map/push rel-diff ~1e-16
  （[debug/zip2_step.jl]，mult mpo·mps Canonical/Dense、mpo·mpo、
  hadamard 双 pass 全核）；收敛解 fidelity = 1.0（iters 19/19、25/23、
  17/19——单方向 push 的轨迹差 ±1-2 轮，不动点相同）；
- **性能**（D=16/Dw=8 级 60 轮）：mpo·mps 1.0x、mpo·mpo 1.05x、
  hadamard 1.06x——IDMRG2 无固定点求解可省，收益上界 = 死写 push 消除，
  与 general 相当略优（hadamard 首版 6 腿中间张量 0.48x，重构 per-s
  切片后修复）。

## 2026-10-13 zip_sweeps!：MultCache/HadamardCache 的 IDMRG 融合快路径（InfiniteTEMPO get_left_xy 机制）

- **接口**：新增 `zip_sweeps!(envs::Union{MultCache, HadamardCache}, alg::IDMRG)`
  （[src/algorithms/arithmetics/compress.jl] 末尾，签名置于 cache struct
  定义之后——导入绑定约束）+ dispatch shim
  `compression_sweeps!(::Union{MultCache, HadamardCache}, ::IDMRG) = zip_sweeps!`
  ——mult/hadamard 驱动经 IDMRG 时自动走快路径；general
  `compression_sweeps!` 保留（`invoke` 可达，VOMPS/IDMRG2 与其余通道不变）；
- **机制**（InfiniteTEMPO 的 `get_left_xy`/`get_xy_right`）：每站把
  `(GL·operator.AL·ket.AL)`（或 hadamard 的 `GL·A1·A2`）一次融合为 K0——
  输入物理腿收缩、算符输出物理与右键开放——同一融合张量双用：map 侧折 C
  权重（`_fold_bond`，AC = AL·C / C·AR 分解；Dense 链无 C 恒等）后与 GR
  收缩出局部 AC，push 侧与 `conj(新 bra)` 单 GEMM 出下一站环境。融合 zip
  张量不落地，每站每方向省一次 AC 分解与固定点求解；
- **hadamard 融合核的共享物理腿**：zip 通道 A1/A2 三方共享物理腿 s
  （开放指标），不存在合法的二元收缩树（任意拆法 s 均出现 3 次）——
  `_fuse_left`/`_fuse_right` 按物理片 batched GEMM（键腿单步收缩 + per-s
  切片，同 `_mapAC_zip`/`_zip_push_*` 方案），最大中间张量 O(Dx·D1·D2·d)；
- **配套修复**：`_fold_bond` 的 perm 公式尾段错（`ntuple(i -> i + 1, N - leg)`
  应为 `ntuple(i -> leg + i, N - leg)`）——mult 通道只折末腿时恰好退化为
  恒等未暴露，hadamard 折中间腿即崩；
- **等价性**：全通道（mult mpo·mps Canonical/Dense 链、mult mpo·mpo、
  hadamard）fused vs general（`invoke` 强制）fidelity = 1.0、迭代数一致，
  单步 map/push rel-diff ~1e-16（[debug/zip_pair_step.jl]/
  [debug/zip_had_step.jl]/[debug/zip_sweeps_check.jl]，多轮扫描逐站一致）；
  小问题规模计时 1.2–5.1×；
- **数值注记**：融合 push 与 general push 的 `conj(bra)` 因子逐位对齐
  （push_left 经 adjoint 自带、push_right 显式 `reshape(conj(below), ...)`
  ——缺共轭是首版 R→L 数值分歧的根因）。

## 2026-10-12 forcereal：实输入的环境实数域投影开关（alg_environments）

- **接口**：`alg_environments`（NamedTuple 形态）新增 `forcereal::Bool = false`
  字段（[`Defaults.alg_environments`](@ref) 同步提供 kwarg；`DynamicTol`
  包装形态经内层 NamedTuple 携带）——`VOMPS`/`IDMRG`/`IDMRG2` 三算法的
  该字段直接透传，驱动层无需改动；
- **语义**：`forcereal = true` 且输入 scalartype 为实时，`mixed_fixedpoints`/
  `overlap_fixedpoints`/`hadamard_fixedpoints` 解出的复边界环境本征向量经
  `_realify!`（移植自 InfiniteTEMPO 同名函数：最大模元相位固定到实轴 +
  real 投影——相位固定使投影良定义）强投实数域，通道全程实算术、输出
  eltype 恒实；**复输入不起作用**（`T <: Real` 守卫）；默认 `false` 行为与
  既有完全一致（实输入的复 leading vector 照常升复）；
- **作用范围**：仅 arithmetics 的 compress/mult/hadamard 通道（三个
  `*_fixedpoints` 环境求解器消费）；groundstates 的 DMRG/VUMPS 等忽视该
  输入（其环境求解不经过这三个函数）；KrylovKit 算法对象形态的
  `alg_environments` 无 `forcereal` 字段（恒 `false`，需实数域投影时用
  NamedTuple 形态）；
- **测试**：arithmetics 新增 forcereal testset（三通道实输入输出恒 Float64
  且与默认解同射线、DynamicTol 穿透、复输入不受影响、`_forcereal` 穿透
  读取）；验证：全量 1322/1322（algorithms 610）。
- **recalculate! 类型守卫（配套）**：`recalculate!`（MultCache/OverlapCache/
  HadamardCache 三方法）新增类型检查——重解出的环境 eltype 宽于当前存储
  （实 cache 遇复不动点，`forcereal` 关闭的实输入通道可能出现）时，原地
  `copy!` 会让 cache 类型参数与实际元素类型脱节、下游实算术写回崩
  （InexactError）——此时返回**全槽位提升后的新 envs**（对齐各 cache
  构造器的 `_promote_scalar` 语义）；`compression_sweeps!`（VOMPS/IDMRG/
  IDMRG2）的调用点与全部驱动层改以 `envs = recalculate!(...)` /
  `envs, info = compression_sweeps!(...)` 接住返回值——**输入 envs 不再
  保证被原地改变，后续调用一律以返回的 envs 为准**（in-place 驱动的
  out 类型守卫既有语义不变）。

## 2026-10-12 内存审计：多因子 @tensor 收缩拆步（mult IDMRG2 大 D 内存爆炸根因修复）

- **背景**：D = 25/50/100、unit cell = 2 的 mult IDMRG2 内存压力测试
  （[debug/mult_idmrg2_memory.jl]）发现两个问题——mpo·mpo 通道 D = 100 直接
  `OutOfMemoryError`，以及 `_svd2!(::CanonicalIMPO)` 的
  `DimensionMismatch`；
- **根因 1（内存爆炸）**：`@tensor` 多因子收缩按**书写顺序左折叠**、每步用
  optimal 配对收缩——相邻因子无共享腿时产生**外积中间张量**：
  `push_env_right` 的 `(above·W)·conj(below)` 中间为
  `(al′, al, w′, w, bl′, bl)` 六腿，bra D=100 × 算符键 64 时单次申请
  ~334GiB 直接 OOM。修复：多因子热点全部拆成二元链式收缩（每步中间有界
  `O(D²·Dw·d)`）——transfermatrix 的 ternary push_env_left/right（MPO 与
  identity 通道）与单层 push_env_right 三方法（identity/matrix 右推的
  `A·conj(A)` 键⁴ 中间同步拆步）、`_push_slice_right`、
  `contract_mpo_expval`（5 因子，中间原为 D²·Dw²·D 级）、correlators 的
  传播步；其余 `@tensor`（effective.jl 的 AC/AC2 作用、`_mapAC`/`_mapC`、
  zip 切片核等）经审计相邻因子均有共享腿（中间 ≤4 腿有界），维持原状；
- **根因 2**：`_svd2!(::CanonicalIMPO)` 的 reshape 用**旧张量键维**作模板
  ——初猜胖于目标无损键（如 `truncdim(100)` > 目标 64）时窗口截断后的键
  profile 逐窗口动态变化，旧键维与新 SVD 输出形状不一致。修复：左/右键维
  取 SVD 输出自身维度（`size(u,1)`/`size(v,3)`），物理维 (u, d) 仍取自家族；
- **附带收益**：mpo·mps D=25 的吞吐分配 123GiB → 4.8GiB（26×）、耗时
  98.7s → 6.8s（外积巨块的分配/释放与 cache 污染消除）；
- **压力测试结论**（W 键 8，d=2，L=2）：D=25/50/100 的 mpo·mps 与 mpo·mpo
  全部收敛，驻留（终态+环境）MB 级、RSS 峰值增量 <250MB、无泄漏；W 键
  64 × ψ 键 64 压力段 RSS 稳定 ~1.9GiB（无爆炸），耗时为环境 Arnoldi 在
  密集谱上的计算量问题（非内存）。

## 2026-10-12 IDMRG2 开放 mult/hadamard 通道（三通道全支持）

- **驱动签名**：`mult`/`mult!`（mpo·mps 与 mpo·mpo）与 `hadamard`/`hadamard!`
  的 positional `alg` 从 `Union{VOMPS,IDMRG}` 放宽为
  `CompressionAlgorithm`（含 [`IDMRG2`](@ref)）；初猜键维统一走
  `_alg_guess_dim` 分派（IDMRG2 从 `alg.trunc` 提取，无维数语义方案回退
  目标链无损键维上界；in-place 的 `changebond!` 预处理同步条件化——与
  `compress!` 同款）；
- **引擎**：`compression_sweeps!(::IDMRG2)` 删除 OverlapCache 通道守卫；
  新增 `_local_AC2` 的通道分派——MultCache 双方法（mpo·mps 的 `GL·k1·k2·
  O1·O2·GR` 分步收缩，物理腿纯开放；mpo·mpo 的 `Θ` 在融合 `(u·d)` 视图
  上输出、站内收缩 `O.do ↔ ket.u` 两链键独立）与 HadamardCache（两因子
  `:ACAR` 两站张量先各自融合再 GL/GR 夹闭，共享物理腿按物理片切片
  batched GEMM）。收尾单站分解补全、Galerkin 收敛判据复用既有实现；
- **实测**：N=2 与 L=3 下 mult mpo·mps/mpo·mpo 与 hadamard 的 IDMRG2 与
  VOMPS 逐射线一致（fidelity ≡ 1.0，无损与有损）——当年记录的「wrap
  窗口环度量缺陷」（mult 0.97/hadamard 0.62-0.98）实测不复现（旧实现的
  环境取位问题）；已知限制：**大键维随机因子（如 6⊙3）的有损 L=3 zip**
  下 IDMRG2 可收敛到 VOMPS 之下的近优不动点（近简并谱 + 两站
  Gauss–Seidel 轨迹，f 0.78-0.83 vs 0.88-0.99，VOMPS 单站 ALS 不受影响）；
- **测试**：IDMRG2 testset 扩为三通道（compress 原有断言保留 + mult
  mpo·mps/mpo·mpo/hadamard 的 VOMPS 逐射线一致（无损 D=5 与有损 D=3）、
  in-place、L=3 zip（4⊙3→4 稳定配置））；验证：全量 states 177 /
  operators 113 / algorithms 新增后全过 / MPSKit concordance 422。

## 2026-10-12 双输入函数全面支持混合 scalartype（Float64 × ComplexF64 实测）

- **接口语义**：所有支持两个 MPS/MPO 输入的函数（`dot`/`fidelity`/`infidelity`、
  变分代数 `mult`/`hadamard`/`compress` 环境、严格代数 `*`/`⊙`、观测量
  `expectationvalue`）允许两输入 scalartype 不一致，输出按
  `promote_type` 提升（如实输入 × 复输入 → 复输出）；
- **放宽的核签名**（原单类型参数 `T` 要求两输入同型，混合输入 MethodError
  或 InexactError）：`TransferMatrix` 三个构造器（双参双层转移、标量对、
  MPO 三参——`dot`/`fidelity` 全族与基态环境求解的底层）、`fuse`（严格
  `W * ψ` 与 `svdguess_mult` 的站点融合核）、`_naive_mul_tensor`（严格
  `W1 * W2`）、`_naive_hadamard_tensor`（`⊙`；`similar` 缓冲同步
  promote）、`svdguess_mult` 两个裸张量方法与 `svdguess_hadamard` 裸张量
  方法（变分通道默认初猜；`svdguess_hadamard` 的 `B3` 缓冲类型此前漏了
  A2 的 scalartype，是 `hadamard(实, 复)` InexactError 的根因——签名不匹配
  时还会掉进 FMA 的宽泛 `PeriodicVector` 分派产生误导性错误）；
- **观测量**：`dominant_env` 的初猜类型与 `DMRGCache(ψ, ::DenseIMPO)` 的
  环境缓冲类型改按 `promote_type(scalartype(ψ), scalartype(operator))`
  ——`expectationvalue(实态, 复算符)` 此前 InexactError，现正常返回复能量；
- **新方法**：`dot(::CanonicalIMPO, ::CanonicalIMPO; kwargs...)`——MPO 的
  Hilbert–Schmidt 内积（vectorize 的 MPS 视图环重叠，`kwargs` 透传
  `krylovdim`），此前只有 `fidelity`（经 vectorize 转发）可用；
- 已 promote 无需改动：三个 `*_fixedpoints` 环境求解器与各 Compression
  缓存（bra/ket 类型参数独立）、`superoperator`、correlators；
- 验证：混合输入冒烟（[debug/mixed_scalartype.jl]）三通道变分、严格代数、
  dot/fidelity、观测量全部通过且输出类型 = promote_type；全量测试
  states 177 / operators 113 / algorithms 595 / MPSKit concordance 422。

## 2026-10-07 修复 svdguess_compress 规范链初猜质量（下游 InfiniteTEMPO 报告）

- **根因**：`svdguess_compress(x::CanonicalIMPS, D)` 对**左正交 `AL` 串**裸做
  流式 SVD 截断——混合规范下键谱/权重全在中心矩阵 `C` 里，`AL` 串的 SVD 谱
  与真实键谱无关，截断按「张量形状」而非「权重」选方向。MPS 通道谱衰减好时
  侥幸可用，**MPO 通道（vectorize 的 MPS 视图）暴露**：键 16、谱衰减好的
  演化算符上 D=6 初猜保真度仅 ~0.37（最优 ~1），变分压缩可能卡在次优不动点
  （下游实测 0.65 → 0.986）；
- **修复**：改经 [`truncate!`](@ref)（`truncdim(D)`：逐键中心矩阵的**真实
  键谱**截断 + 完全重正则化）——`CanonicalIMPS` 与 `CanonicalIMPO` 输入各自
  直调对应 `truncate!` 方法（MPO 走其 vectorize 的 MPS 视图通道），实测同
  场景初猜保真度 0.37 → > 0.99（变分最优水平）；**返回类型变化**：
  `CanonicalIMPO` 输入现返回 `CanonicalIMPO`（原返回 MPS 视图
  `CanonicalIMPS`）；裸张量串方法（下游底层入口）保留原流式实现并加
  docstring 警示（只对携带权重的串给出好初猜，勿对 `AL`/`AR` 家族直用）；
  测试新增回归用例（MPO 进出类型 + 初猜 > 0.99 + 变分收敛 > 0.9999）。

## 2026-10-07 TEBD 门类型改用 FMA + Schur 环境推进去稠密化

- 删除 tebd.jl 的本包门类型定义（AbstractGate/UnitaryGate/GeneralGate 的
  struct、positions/shift/adjoint/scalartype、NTuple 构造与 `_isunitary`
  ——与 FMA 侧本就是同一份代码的移植），经主文件 import 直接用 FMA 的
  （用户同时 using 两包时门类型不再同名冲突）；本包保留 Pair{Int,Int}
  矩阵约定的便捷构造（FMA 无，Kronecker 约定 permute 成张量约定后委托
  FMA 的 NTuple 构造）与 CanonicalIMPS 上的 apply!/swap!（Hastings 更新）；
- envs.jl 新增 `push_env_left/right(L, W::SchurMPOTensor, A)`——Schur 通道
  的环境推进逐 level 块收缩特化（`iszero` 零块跳过，复用 `_push_slice`
  kernel，语义与稠密版逐位一致、随机张量实测 ≈），`transfer_leftenv!/
  transfer_rightenv!` 据此去掉 `tompotensor` 稠密化分支、直接多分派
  （DenseIMPO 走稠密 kernel、SparseIMPO 走 Schur 特化）——IDMRG 扫掠的
  增量推进不再为 Schur 哈密顿量物化 (m·d·n·d) 稠密张量。

## 2026-10-07 FMA 同名函数族 import 合并（双包协同）

- 主文件 `import FiniteMPSAlgorithms: ...` 扩展 24 个同名函数（原有
  distance/distance2/⊙/truncate!/tompotensor/timeevompo/phydim 之外新增
  phydims/bonddim/vectorize/devectorize/superoperator/fidelity/infidelity/
  expectationvalue/entanglement_spectrum/changebond!/compress/compress!/mult/
  mult!/hadamard/hadamard!/tompotensors/svdguess_{mult,hadamard,compress}/
  apply!/swap!/positions/shift）：含义类似且类型分派不重叠（FMA 方法作用于
  有限链类型、本包作用于 infinite 类型）——合并为同一函数对象后，用户同时
  `using` 两包时统一函数名按类型自动分派，消除「仅 using 不 import 会静默
  创建同名本地函数、切裂方法表」的隐患（phydim 委托 bug 的教训）；实证
  分裂名单从 44 收敛到 20 个有意遮蔽项，双侧分派冒烟通过；
- **有意保持遮蔽（不 import，注释固化于主文件）**：σx/σy/σz/Sx/Sy/Sz
  （签名相同但缺省类型不同——FMA 全 ComplexF64、本包实矩阵缺省 Float64）、
  heisenberg_hamiltonian/tfim_hamiltonian/fermi_hubbard（FMA 带 L 位置参数
  返回有限 MPOHamiltonian vs 本包无限 SparseIMPO）、linsolve（FMA 的 ALS
  变分求解 vs 本包的 KrylovKit 稠密包装）、类型撞名 AbstractGate/
  UnitaryGate/GeneralGate 与 DMRGCache/MultCache/OverlapCache/HadamardCache
  （struct 无法 import 合并，cache 构造器语义已对齐 chain-first）、Defaults
  （模块撞名）；effective.jl 的 MPSTensor/MPOTensor 为内部 const 别名
  （不导出，故意取更宽的 AbstractArray），加注释说明。

## 2026-10-07 SparseIMPO 矩形 Schur 链；timeevompo(::SparseIMPO) 接口收敛；bonddim 全家族统一右键约定

- `SparseIMPO` 支持非方阵 SchurMPOTensor（对标 FMA `MPOHamiltonian` 的链式
  闭合）：构造器检查 `space_r(W[i]) == space_l(W[i+1])` + 周期闭合，取代
  「逐站方阵 + 全胞同层数」；unitcell > 1 时通道在不同键上开启/闭合的矩形
  张量合法（键层数逐键可不同）。`bonddim(H, ℓ)` 逐键、0 参 `bonddim(H)` 仅
  均匀链可用、新增 `max_bonddim(H::SparseIMPO)`；`isidentitylevel`/
  `isemptylevel` 矩形化（越界块 = 结构零、闭合单位角 (m,n) 识别）；
  `DMRGCache(SparseIMPO)` 逐键层结构求解：内层标号递增分轮、瞬态层（无环
  返回通道）单次扫描定值、闭合通道轮处理 corner 流（闭合层逐键换标号、
  角块恒 I ⇒ 环转移 = 纯 MPS 转移，须 regularize；均匀链上严格退化为原
  i = nl 恒等层轮）；AC 作用/期望值闭合列逐站取 `space_r`；
- **`make_time_mpo` → `timeevompo(::SparseIMPO)`**（breaking rename，直接重载
  FMA 同名函数）：只保留 `SparseIMPO` 入口（删除单 bulk 张量方法），删除
  `imaginary_evolution` keyword——`δ` 即指数系数本身（实时 `-im·t` / 虚时
  `-τ`，同 FMA 约定与本包 TDVP `integrate` 的约定），输出 `exp(δ·H)` 的
  `DenseIMPO`；主文件 `import FiniteMPSAlgorithms: timeevompo` 并再导出；
  同时删除 `SparseIMPO` 的 `Base.eltype`/`Base.iterate`/三参 `Base.getindex`
  定义（元素类型统一 `scalartype(H)`，后两者无包内消费者）；
- 删除 `src/operators/sparsempotensor.jl` 整个薄适配层：`nlvls`
  （= `space_l`，本包命名；层数请直接用 `space_l`/`space_r` 或 `size(W, 1)`）
  与 `*(λ, W)`（物理块缩放 B/C/D——与 FMA `lmul!` 的全块缩放语义不同）随
  功能一并退役（api.jl 的 `λ * W` 语义测试块同步删除）；export 列表移除
  `nlvls`，测试的层数断言改 `size(W, 1)`；
- 主文件 `import FiniteMPSAlgorithms: phydim` 合并函数族：Julia 向其他模块
  导出的函数加方法必须显式 import（仅 `using` 时定义会静默创建同名本地
  函数、切裂方法表），`phydim(H::SparseIMPO, i) = phydim(H[i])` 的 FMA
  `phydim(::SchurMPOTensor)` 委托由此修复；
- **`Base.:+(H::SparseIMPO, λs)` 退役**：能量平移不是算符代数（对标 MPSKit
  的 `H + λs` 仅为数值便利），且包内无消费者——同名 helper `shift_local_
  energies`（逐站注入 Schur D 角）移至 test/testhelpers.jl 供测试使用，
  反向 `λs + H` 一并移除；
- `SparseIMPO` 的 `phydim`/`phydims` 委托 FMA 的
  `phydim(::SchurMPOTensor)`；`isidentitylevel` 增加 `atol` keyword；
  `+ λs` 的 D 角注入改 `one(W.D)`；
- 删除 `src/operators/longrangeop.jl`：`SchurMPOTensor(::ExpDecayOpSum/
  ExpDecayOpTerm)` 便捷构造（hloc 缺省零）FMA 后端已提供（经 `using`
  再导出，对外 API 不变；顺带消除加载时的方法覆写告警）；
- **`bonddim` 全家族统一为 site 右键**：`bonddim(x, ℓ)` = 键 ℓ（site ℓ
  右侧）的维数——`DenseIMPS` 由 `size(ψ[ℓ], 1)`（左键）改为 `size(ψ[ℓ], 3)`、
  `DenseIMPO` 同、`SparseIMPO` 取 `space_r(H[ℓ])`；`CanonicalIMPS`/
  `CanonicalIMPO`（`size(C[ℓ], 1)`，C[ℓ] 本就在键 ℓ 上）不变——原 Dense
  家族左键 / Canonical 家族右键的错位在非均匀键链上不自洽（均匀链两者
  相等、此前未暴露）。`dot(::DenseIMPS, ::DenseIMPS)` 的转移矩阵初值相应
  改 `bonddim(·, 0)`（键 N）；均匀键链上所有断言数值不变；
- 测试：矩形链 [2×3,3×2,2×2] 与同算符均匀方形表示期望值逐位一致、VUMPS
  收敛解析能量；w1w2/twosite/tdvp/mpo/w1w2_concordance 的调用点与断言
  按新接口更新；docs/README 同步。

## 2026-10-07 SchurMPOTensor / longrangeop / w1w2 复用 FMA 后端（去重）；TEBD GeneralGate 的 gaugefix! 前移

- `src/operators/sparsempotensor.jl` 重写为薄适配层：SchurMPOTensor 类型本体
  （struct、逻辑矩阵构造器、块访问、稠密化 tompotensor、copy/scalartype/
  complex）全部由 FiniteMPSAlgorithms 提供（其实现由本包移植并扩展了矩形
  通道空间 `space_l ≠ space_r`；本包平移不变单胞张量恒方形）。本包侧保留：
  `nlvls`（= `space_l`，本包命名）与 `*(λ, W)`（物理块缩放 B/C/D、传播块 A
  与恒等角不动——FMA 无对应语义）。**`+` 语义变化**：删除本包的元素级加法
  （要求双方同通道结构），统一用 FMA 的**直和**（通道拼接、D 角相加，普适
  于不同通道结构，对标 MPSKit H1+H2 的块拼接语义）——`SparseIMPO +
  SparseIMPO` 随之普适化；`H + λs`（能量平移）改为直接注入各站 Schur 张量
  的 D 角（不引入新通道，键维不变，行为与旧元素级实现数值一致）；
- `src/operators/longrangeop.jl` 重写为薄适配层：ExpDecayOpTerm / ExpDecayOpSum
  类型与 W-form 构造由 FMA 提供；本包侧保留无 on-site 项的单参便捷构造
  `SchurMPOTensor(s)` / `SchurMPOTensor(t)`（= FMA `SchurMPOTensor(s, zeros)`）。
  通道约定差异（算符值等价、张量位形不同）：FMA 把衰减因子置于通道开启端
  （C = α·λ·a、B = b），本包旧版置于关闭端（C = α·a、B = λ·b）——
  距离 d 的路径两者都恰积累 α·λ^d；test/operators/longrangeop.jl 的块级
  断言按 FMA 约定更新（算符级断言两约定通用，不变）；
- `src/algorithms/timeevo/w1w2.jl` 重写为薄包装：步进器类型 `WI`/`WII`（本包
  再导出 FMA 类型）与张量级演化内核 `timeevompo`（WI 一阶展开、WII 的
  4d×4d 块传播矩阵 `LinearAlgebra.exp`、实/复 √δ 分解）全部由 FMA 提供；
  本包侧只保留 infinite 包装：δ 换算（`exp(-i·H·dt)` / `exp(-H·dt)`）与
  `DenseIMPO` 装配（`make_time_mpo` 对 bulk/SparseIMPO 两入口，逐站演化，
  支持任意单胞长）。删除本包重复的 `get_A/B/C/D`、`_sqrt2`、
  `_timempo_dense` 内核（w1w2 对标 MPSKit 的 distance/fidelity 断言保证
  内核切换数值不变，dt=0.1 全组合距离 0 ~ 6e-8 维持）；
- 主文件 `import FiniteMPSAlgorithms: ..., tompotensor`（FMA 未导出的 Schur
  稠密化；SchurMPOTensor/WI/WII/timeevompo/ExpDecayOp* 经既有 `using` 引入
  并再导出，对外 API 名不变）；
- `src/algorithms/timeevo/tebd.jl`：`apply!(::GeneralGate{2}, ...)` 的
  `gaugefix!(; order = :RL)` 从回程 swap 之后**前移**到非幺正门应用
  （`_nn_gate_apply!`）之后——回程 swap 的 Hastings 更新假设混合规范输入，
  在规范破缺的双站张量上作用会累积误差；门后立即重正则化使回程作用在
  规范态上；
- 验证：全量测试四块（states 168、operators 86、algorithms 542、
  MPSKit concordance 404）全部通过，共 1200/1200。

## 2026-10-06 `w1w2.jl` 移至 algorithms/timeevo；新增 `make_time_mpo`（WII）≡ MPSKit 对标测试

- `src/operators/w1w2.jl` → `src/algorithms/timeevo/w1w2.jl`（W^I/W^II 时间
  演化 MPO 属 timeevo 语义组；include 相应调整，主文件 timeevo 注释同步）；
- 新增 `test/mpskit/w1w2_concordance.jl`（此前 test/mpskit 无 WI/WII 对标）：
  **算子级严格一致**对齐标准——包内 `distance(::DenseIMPO, ::DenseIMPO)`（无
  相位/尺度自由），不做张量逐位比较——同一算符的 MPO 表示有键空间规范自由
  （实测逐张量相对差 ~0.2-0.3 而稠密周期 trace 的直接差 ~1e-16）。用例：
  TFIM（N=1，实时 + 虚时）与 Heisenberg XXX（N=1，多 Schur 通道）的
  `make_time_mpo(H, dt, WII())` vs `MPSKit.make_time_mpo(H_k, dt,
  MPSKit.WII())`，dt = 0.1 强步长（两侧相对严格解都不精确但对齐不受影响），
  距离断言 < 1e-7（实测 0 ~ 6e-8——本包 LinearAlgebra.exp 块指数与 MPSKit
  Arnoldi exponentiate 解同一块方程；阈值放宽因 Gram 消去 + eigsolve 容差给
  距离留 √ε 量级地板）；输出键维两侧一致断言。MPSKit 只实现 WII（无 WI），
  WI 无对标对象；
- 新增包 API `distance2`/`distance` 与 `fidelity`/`infidelity`
  （`::DenseIMPO, ::DenseIMPO`；`src/operators/linalg.jl`）：委托
  `vectorize(W₁)`/`vectorize(W₂)` 的同名 DenseIMPS 函数——HS 内积的转移矩阵
  主导本征值语义；`distance` 规范不变但对整体相位/尺度敏感（严格判"同一
  算符、同一尺度"），`fidelity` 允许整体复比例（射线语义，与 `CanonicalIMPO`
  版本互补）；
- 对标测试的"两 Infinite 对象一致性"判据一律改为双轨：包内 `fidelity`/
  `distance`（infinite 转移矩阵语义，**主判**）+ `_dense_mpo_repr`/
  `_dense_mps_repr` 的 periodic-repr 射线残差/直接差（**辅助**——periodic
  trace 把 Infinite MPO/MPS 当成有限环对象处理，概念上不完备，不能单独作
  为一致判据）：w1w2（distance + fidelity + 直接差）、mult（严格乘法
  distance + fidelity + ray；MPO·MPS 施加与变分复合 fidelity + ray）、
  compress（fidelity + ray）、groundstate（fidelity + ray）；
- `mpo_ray_residual`（testhelper）保留作辅助判据（变分不动点输出带归一化
  自由——mult_concordance 实测变分解尺度比 0.508 而射线残差 1e-14；严格
  乘法处 distance 的 eigsolve 地板 ~2e-6，阈值放宽到 1e-5）；
- `testhelpers_mpskit.jl` 的 `mpo_from_mpskit` 兼容 BlockTensorKit 的
  `SparseBlockTensorMap`（MPSKit `make_time_mpo` 的输出层，先 `TensorMap(t)`
  合并块）；
- 验证：全量测试四块（states 168、operators 86、algorithms 541、
  MPSKit concordance 363）全部通过，共 1158/1158。

## 2026-10-06 TDVP 接口清理：`dt` 即指数系数本身、删 `imaginary_evolution`/`verbosity` keyword、exponentiate 收敛告警；`alg_orth_trunc` 主文件 using；删冗余 scalartype 实例方法

- **`integrate(H, x, dt, alg)`**：`dt` 即指数的系数本身（不做任何 `-im`
  换算）——实时演化直接输入 `dt = -im·t`（`exp(-i·H·t)`）、虚时演化输入
  `dt = -τ`（`exp(-H·τ)`，cooling）；删除 `t` 与 `imaginary_evolution`
  输入；`exponentiate` 未收敛（`info.converged == 0`）时 `@warn`（原实现
  丢弃 info 不检查）；
- **`timestep(ψ, H, dt, [alg], [envs])`**：删 `t` 与 `imaginary_evolution`；
  **`time_evolve(ψ₀, H, t_span, [alg], [envs]; observer)`**：删
  `verbosity`/`imaginary_evolution` keyword——实时演化输入纯虚步长
  （`t_span = (-im) .* (0:0.01:1)`）、虚时输入负实步长
  （`t_span = -(0:0.05:20)`）；实数链仅在实时演化（`dt` 非实）时自动升复，
  虚时保持实数域；`TDVP` 类型新增 `verbosity::Int = Defaults.verbosity`
  field 控制迭代日志（对齐 VUMPS/IDMRG）；
- 主文件 `using FiniteMPSAlgorithms.Defaults: alg_orth_trunc`（只引入函数
  名、不引入 `Defaults` 模块名，避免与本包同名模块冲突），
  `truncate!`/TEBD gates 的 `trunc` 缺省处直接写 `alg_orth_trunc()`；
- 删除 `scalartype(W::SchurMPOTensor) = scalartype(typeof(W))` 冗余实例
  方法（VectorInterface 已有通用实例 fallback `scalartype(x) =
  scalartype(typeof(x))`；Type 方法保留——全 src 扫描无其他同类冗余）；
- 测试同步：tdvp/twosite/envs/api/finite_t_concordance 的实时用例改
  `(-im) .* tspan`、虚时用例改 `-(tspan)` / `dt = -dβ`；
  docs（index.md/algorithms.md）签名与示例更新；
- 验证：全量测试四块（states 168、operators 86、algorithms 541、
  MPSKit concordance 359）全部通过，共 1154/1154。

## 2026-10-06 `truncate!`/TEBD gate 接口的 `trunc` 默认改为 `FMA.Defaults.alg_orth_trunc()`；`truncate!` 的 `alg_gauge` keyword 改为 `kwargs...` 透传

- `truncate!(::CanonicalIMPS/::CanonicalIMPO)` 的 `trunc` 缺省从
  `DefaultTruncation`（`Defaults.D` 封顶 + 相对阈值 + `add_back = 1`）改为
  `FiniteMPSAlgorithms.Defaults.alg_orth_trunc()`（= `truncrelerr(ϵ =
  FMA.Defaults.tolgauge)` 相对阈值谱清理），对齐 FMA `truncate!`/
  `canonicalize!` 的默认方案；不再含键维封顶（需要 `D` 封顶时显式传
  `truncdim`/`truncdimcutoff`）；
- `truncate!` 的 `alg_gauge` keyword 删除，改为 `kwargs...` 直接透传内部的
  `gaugefix!(; order = :LR, kwargs...)`（如 `tol`/`maxiter`/`alg_orth`）；
- TEBD gate 接口族（`apply!(::UnitaryGate)`/`apply!(::GeneralGate)`/`swap!`
  及内部 `_hastings_update!`/`_nn_gate_apply!`）的 `trunc` 缺省从
  `NoTruncation()` 改为 `FiniteMPSAlgorithms.Defaults.alg_orth_trunc()`——
  默认只清理数值零谱方向（对随机态谱全显著时与 `NoTruncation` 无差），
  `NoTruncation()` 仍可显式传入得到逐位无损的门应用；docstring 同步；
- 验证：全量测试四块（states 168、operators 86、algorithms 541、
  MPSKit concordance 359）全部通过，共 1154/1154。

## 2026-10-06 `truncate!` 装配改为「截断 + 完全重正则化」（与 `gaugefix!(::InfiniteOrthogonalize)` 同构）

- 原实现的闭式装配（V 旋转 AR + `AL = AC/C` 右除）**强行保留**逐键截断谱：
  截断是键空间的非幺等投影、环形闭合方程超定，不自洽缺口 ~`err` 且被
  `cond(C)` 放大（谱含接近截断阈值的奇异值时失控，体现为 ϵ_left 偏差）；
- 新装配（全程无除法）：(1) 逐键 `tsvd(C)` 截断，保留子空间的行正交基 `Û`
  投影 `AL` 两侧键 `A'[ℓ] = Û_{ℓ-1}†·AL[ℓ]·Û_ℓ`——得到截断后的**原始张量串**
  （合法周期态）；(2) `gaugefix!(; order = :LR)` 完全重正则化：`AL` 由 QR
  装配、`AR` 由 LQ 重建（构造性正交）、`C` 由混合转移 fixed point **重解**
  收敛、`AC = AL·C` 闭式乘法。截断的不自洽只进射线精度（fidelity 损失
  ~丢弃权重），不进正则性：三项 `mixedcanonical_error` 均在 `alg_gauge.tol`
  （默认 1e-13）量级、与丢弃权重无关；
- 语义变化：输出 `C` 为重解的 fixed point（谱 ≈ 截断谱 + O(err) 自洽调整，
  不保证对角——需要谱时对 `C[ℓ]` 做一次 `tsvd`）；输出态归一（`norm = 1`，
  `:LR` 路径的 `C[N]` Frobenius 归一约定）；新增 keyword `alg_gauge`
  （默认 `Defaults.alg_gauge()`，重正则化的 `tol`/`maxiter`）；`CanonicalIMPO`
  版改为 vectorize 的 MPS 视图上执行同一装配后 devectorize 写回；
- 测试更新（states/mps.jl 的 truncate! testset）：强截断用例断言改为三项
  正则误差一致 `< 1e-10`（不再与 `err` 挂钩）；默认方案近无操作、秩亏清理
  用例断言不变（射线/键 profile/权重全保持）；代价：单次调用 ~一次完全
  正则化（fixed point + 双向扫掠），实测 states 块 ~2× 耗时；
- 验证：全量测试四块（states 165、operators 86、algorithms 541、
  MPSKit concordance 359）全部通过，共 1151/1151。

## 2026-10-06 新增 `gaugefix!(::InfiniteOrthogonalize)`（states/ortho_exact.jl）；`fixedpoint(::Arnoldi)` 走 schursolve；TDVP `alg_orth` field；清理 eltype/冗余判断

- **新增 `states/ortho_exact.jl`**：移植 InfiniteTEMPO 的
  `InfiniteOrthogonalize`（改为继承本包 `Algorithm`；fields
  `trunc`/`normalize`/`toleig`/`maxitereig`/`verbosity`）与
  `mixedcanonicalize2!`（Reference: PHYSICAL REVIEW B 78, 155117）——
  左右主导边界本征对（`_overlap_leading_boundaries`：随机秩 1 PSD 初值 +
  `fixedpoint`，迹归一）→ 谱截断 Cholesky 白化（`_chol_split`）→
  `tsvd(Y·X; trunc)` 给闭合键键谱 → 前向 QR（QRpos）扫掠 + 反向 SVD 扫掠
  （谱归一、`m = v·Diag(ss)` 吸收）→ site 1 左除 `Diag(S)`。两趟有限深度
  扫掠即收敛，与 `states/ortho.jl` 的迭代 power sweep 互补；
- **入口 `gaugefix!(ψ::CanonicalIMPS, As, alg::InfiniteOrthogonalize)` /
  `gaugefix!(W::CanonicalIMPO, As, alg::InfiniteOrthogonalize)`**（3 参，
  无 `C₀`；MPO 自动 `vectorize` 到 MPS 视图）：谱 `sv`（InfiniteTEMPO 的
  `x.s`：键在 site ℓ **左**侧）映射到我们的 `C[ℓ] = Diag(sv[ℓ+1])`，
  `AR = x`、`AC = C[ℓ-1]·AR`（行缩放）、`AL = AC/C`（右除）闭式装配
  （同 `truncate!` 模式）。`InfiniteOrthogonalize` 已 export
  （`mixedcanonicalize2!` 为内部 kernel 不导出）；
- `InfiniteOrthogonalize` 不设 `normalize` field（对齐本包其他 gauge 函数，
  输出恒归一）：InfiniteTEMPO 的 `normalize = false` 分支（保留 `√η` 射线
  尺度）不移植——其 `lmul!(ηs, x[1])`/`lmul!(ηs, x.s[1])` 在四族混合规范
  存储下会破坏 site 1 的正交性，且未归一态与本包 `dot`/`fidelity` 的转移谱
  语义（假定已归一）不兼容；
- `fixedpoint(::KrylovKit.Arnoldi)` 改走 `KrylovKit.schursolve`（对标
  MPSKit）：实算子 + 实初值保持在实数域（实 Schur 形式），结构上消除「实链
  非厄米混合转移返回复本征向量」问题；`uniform_leftorth!`/`uniform_rightorth!` 的
  gauge_eigsolve_step! 相应删除 real 分支。NamedTuple（环境通道，
  `alg_environments` 形态）**有意保持 `_eigsolve`**——mult/compress/hadamard
  的环境在实输入下 leading vector 可为复（通道升为复算术的既有 MPSKit 对齐
  行为；schursolve 的实 Schur 形式无法表示复本征对）；即 gauge 通道走
  schursolve、环境通道走 eigsolve；
- `TDVP` 增加 `alg_orth` field（默认 `Defaults.alg_orth()`，
  `timestep` 收尾 `regauge!` 的 QR/LQ 算法）；
- 删除冗余判断：`states/linalg.jl` 与 `states/canonicalmps.jl` 两处
  `λ isa Number ? λ : only(λ)`（`_eigsolve` 的 `vals[1]` 恒为 `Number`）；
- **`eltype` → `scalartype` 清扫**：凡「为取浮点/标量类型」而调用 `eltype`
  的一律改 `scalartype`（VectorInterface 递归实现，嵌套容器
  `scalartype(Vector{Matrix{T}}) = T` 等价 `eltype(eltype(x))`）——涉及
  `regauge!`（ortho.jl）、correlator、`mpohamiltonian`、
  `isidentitylevel`、`ExpDecayOpSum`、`_mpoham_scalar_type`、
  `_timempo_dense`、hadamard/mult/compress 的环境类型提升
  （`T = eltype(GLs[1])` → `scalartype(GLs[1])`、
  `promote_type(T, eltype(vL))` → `scalartype(vL)`）与
  `scalartype(::CompressionEnvironments)`；保留合法用途（容器元素类型
  `Vector{eltype(ψ.AL)}`、`Base.eltype` 定义、`v isa Number` 值分派）；
- 测试：`test/states/mps.jl` 新增「gaugefix!(InfiniteOrthogonalize)」
  testset——良态链 + 随机规范变换（射线不变、正则性 ~1e-14、实链标量类型
  保持、normalize 两分支）、相对阈值秩亏清理（零块链 → 键 profile [2,2]、
  射线不变）、CanonicalIMPO（HS 保真度）。

## 2026-10-02 `TDVP` 的 `tolgauge`/`gaugemaxiter` 合并为 `alg_gauge` field；gauge_eigsolve_step! 严格对齐 MPSKit

- `TDVP(; integrator, alg_gauge, finalize)`：与 `VUMPS`/`IDMRG`/`VOMPS` 的
  `alg_gauge` field 惯例完全一致，默认 `Defaults.alg_gauge()`（`(; tol,
  maxiter)` NamedTuple，动态容差下为 `DynamicTol` 包装）；`timestep` 收尾处
  解包后在 `gaugefix!` 调用中以 keyword 输入 `tol`/`maxiter`（经 2 参
  `CanonicalIMPS(ALs, C₀; kwargs...)` 构造器透传，`order = :R` 与 MPSKit 的
  `InfiniteMPS(AL, C₀)` 重建语义一致）；
- `gaugefix!`/`CanonicalIMPS(ALs, C₀)` 保持 MPSKit 形态：kwargs 透传 + 
  `order` 分发（不设 `alg_gauge` keyword；算法对象走四参 expert 分发）；
- `uniform_leftorth!`/`uniform_rightorth!` 的 gauge_eigsolve_step! 删除
  「复本征向量取实部」的非 MPSKit 分支（严格对齐 MPSKit
  `gauge_eigsolve_step!`：直接 `left_orth!(vec)`/`right_orth!(vec)`）；
- 测试：tdvp.jl 断言 `TDVP().alg_gauge` 解包后 `(tol, maxiter) ==
  (Defaults.tolgauge, Defaults.maxiter)`，显式 NamedTuple 用例与默认路径
  轨迹一致（~1e-10）。

## 2026-10-02 清理「operator might not be hermitian」告警（测试侧适配）；构造器缺省类型改 Float64；删除 `_mpo_from_mps`

- 告警诊断：测试中的海量 KrylovKit Lanczos 告警**不是哈密顿量构造错误**——
  DMRG/VUMPS/IDMRG 的约定就是输入哈密顿量厄米、局部求解默认 Lanczos（同
  MPSKit）。两个测试用例违反了该约定：(1) `test/algorithms/envs.jl` 异构
  物理维 testset 用随机复 onsite 的**非厄米**哈密顿量（Lanczos 上虚部高达
  1.0，8937 条）——改为对称化 `(h + h')/2` 的随机厄米 onsite，局部求解回归
  默认；(2) DenseIMPO 通道基态（`operators/mpo.jl`、groundstate concordance）：
  恒等层 bookkeeping 使转移矩阵主导空间简并，投影有效哈密顿量结构性非严格
  厄米（~1e-6）——该通道测试显式传 `alg_eigsolve =
  Defaults.alg_eigsolve(; ishermitian = false)`（Arnoldi；`compare_groundstate_k`
  增加 `alg_eigsolve` 转发关键字）。SparseIMPO 通道有效哈密顿量严格厄米
  （0 告警，对照）；源码层面 VUMPS/IDMRG 保持默认 Lanczos 不变；
- 构造器缺省标量类型统一为 Float64：`identityimpo(phydims)`（原
  ComplexF64）、`σx`/`σz`/`Sx`/`Sz`（原 ComplexF64）、`tfim_hamiltonian`/
  `tfim`/`fermi_hubbard` 的 `T` 缺省（原 ComplexF64）；σy/Sy 本征复矩阵、
  `heisenberg_hamiltonian`/`heisenberg_xxz`（含 Sy）保持 ComplexF64；
- 删除 `Defaults.eltype`（与标准库 `eltype` 同名的常量定义有遮蔽风险，且无
  使用方——concordance 的镜像断言同步移除）；
- 删除内部 reshape kernel `asmps_view`/`mps_view_to_mpo`/`_local_square_rdims`：
  包内约定 MPO 局域 `du == dd`（方算符），rank-3 ↔ rank-4 的互逆转换只保留
  `vectorize`/`devectorize`，二者新增 **raw-string（AbstractVector）重载**——
  `vectorize(Ws::AbstractVector{<:Array{T,4}})`（融合 `(wl,u,wr,d) →
  (wl,u·d,wr)`，逐站读入自身物理维）与
  `devectorize(As::AbstractVector{<:Array{T,3}})`（拆分 `f = r²`，完全平方数
  校验）；`CanonicalIMPO(Ws)` 构造器、`devectorize`/`vectorize` 的类型方法
  相应内联；
- 新增混合规范构造器 **`CanonicalIMPS(AL, C, AR[, AC])`** /
  **`CanonicalIMPO(AL, C, AR[, AC])`**（镜像 MPSKit 的 `InfiniteMPS(AL, C,
  AR)`）：`Vector`/`PeriodicVector` 皆可，`AC` 缺省由 `AC = AL·C` 闭式装配
  （rank-4 kernel `_mulAL`/批量 `_mul_ALC`），输入视为已处于相应规范、不做
  `gaugefix!` 重整；相应地 **4 参族序构造器统一为 `(AL, C, AR, AC)`**
  （原 `(AL, AR, C, AC)`，`CanonicalIMPO{T}` 内构造签名同步换序，字段存储序
  不变），全部内部调用点（copy/similar/circshift/dag/`CanonicalIMPO(Ws)`/
  `vectorize`/`devectorize`/mult `_promote_scalar`/tdvp 复数化）已随新序更新；
- 验证：states、operators（mpo/vectorize）、envs、groundstate concordance、
  compress 测试全过。

## 2026-10-02 新增 `truncate!(::CanonicalIMPS/::CanonicalIMPO; trunc)`

- 正则链的原地键截断（语义对齐 InfiniteTEMPO 的 `toiadt!`/`toipt!` finalize）：
  逐键对中心矩阵 `C[ℓ]` 做 SVD 截断（`trunc::TruncationScheme`，默认
  `DefaultTruncation` = `Defaults.D` 封顶 + `Defaults.tolgauge` 相对阈值 +
  `add_back = 1`），中心矩阵取对角谱（"C 对角" 规范），相邻键的 unitary
  因子把 `AR` 旋转到与对角 `C` 一致，`AL`/`AC` 闭式装配（`AC = diag(s)·AR'`、
  `AL = AC/C'`；MPO 版在 MPS 视图 `(wl, u·d, wr)` 上右除）；四族逐槽写回
  （原地），不做重正则化/重建规范；返回 `(ψ, err)`，`err` 为最大逐键 `tsvd`
  截断误差；
  - 正则性语义：截断丢弃的谱权重小时（相对阈值方案的谱清理——包括截掉
    秩亏的 ~0 方向），装配正则性偏差与丢弃权重同量级，`ismixedcanonical`
    在相应容差下成立；真截断（丢有限权重）时偏差随之增大（实测 ~err，受
    保留谱条件数放大）——正则性由调用方按 `err` 把控，不做强制重建；
  - `truncate!` 名字为 FiniteMPSAlgorithms 同名函数的方法扩展（`import`
    后新方法，原 `Vector`/有限 `CanonicalMPS` 方法不受影响）；已在导出列表
    （随 tensorops 层再导出）；
- 测试（states/mps.jl）：良态链默认方案近无操作（正则性/射线/键 profile/
  范数保持）、零块秩亏链的谱清理（键 4→2、正则性严格成立、err=0，MPS/MPO
  双版）、`truncdim(2)` 强截断（键 4→2、容差 = 丢弃权重下正则、混合一致性
  机器精确、`0 < fidelity < 1`）。

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
