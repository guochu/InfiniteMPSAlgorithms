# 变更记录（接口）

## 2026-10-20 TEBD 门 unitarity 改读字段（FMA 删除导出的 `isunitary(g)`）

- **背景**：FMA `43b0916` 让 `Gate` 构造器用
  `MatrixAlgebraKit.isunitary`（转发 `atol`）判定 unitarity，并删除内部
  `_isunitary_op` 与导出的 `isunitary(g::Gate)` 方法；调用侧直接读
  `g.isunitary`，`apply!` 的缺省 `canonicalize` 相应改为 `!g.isunitary`；
- **接口**：本包 import/export 移除 `isunitary`（不再从 FMA 引入不存在的
  绑定），保留 `AbstractGate, Gate, apply!, swap!, positions, shift`；
  单/两比特 `apply!` 的 `canonicalize` 缺省统一为 `!g.isunitary`；
- **测试/文档**：tebd.jl 的 `isunitary(g)` 断言全部改为 `g.isunitary`，
  testhelpers 注释、library.md（移除 `@docs isunitary`）、conventions.md、
  algorithms.md、README.md 同步；无行为变化。

## 2026-10-19 TEBD 接口跟进 FMA（`GateOp` → `Gate`）+ 单比特门支持

- **背景**：FMA 将 `GateOp` 更名为 **`Gate{N,T,M}`**，并新增矩阵构造
  `Gate(positions, op::AbstractMatrix, phydims; isunitary = nothing, atol)`；
  门类型现在在构造时按 `atol` 自动检测 unitarity（也可由关键字直接给定）；
- **接口**：import/export 由 `AbstractGate, GateOp, isunitary` 改为
  **`AbstractGate, Gate, isunitary`**；本包与测试/文档中的 `GateOp` 全部改名；
- **新增单比特门**：`apply!(g::AbstractGate{1}, ψ::CanonicalIMPS; trunc,
  canonicalize = !isunitary(g))`——门作用于中心张量 `AC[i]`（`1 <= i <= L`），
  键谱 `C[i-1]`、`C[i]` 保持不变（幺正门不改 Schmidt 值），`AL[i]`/`AR[i]`
  由规范恒等式经 `regauge!` 重算（幺正门时即 `G·AL[i]`、`G·AR[i]`）；
  非幺正门缺省经 `truncate!(ψ; trunc)` 重正则化，与 FMA 的 `canonicalize`
  语义一致；两比特门行为不变；
- **测试**：tebd.jl 新增"单比特门作用" testset（规范保持、`C` 不变、与两站
  `kron(U, I)` 等价、`g·g†` 往返、wrap 站点、非幺正门缺省 canonicalize、
  越界抛错），gate 类型 testset 补矩阵构造与单比特构造断言；
  testhelpers 的 `pair_gate` 改委托 `Gate`；envs.jl 两处 `GateOp` 改名；
- **docs**：library.md（`Gate` @docs）、conventions.md（构造时检测 unitarity）、
  algorithms.md（单比特门语义 + 双签名）、README.md、tdvp.jl docstring 同步。

## 2026-10-18 TEBD 门接口对齐 FMA 统一 GateOp（UnitaryGate/GeneralGate 退役）

- **背景**：FMA 的 TEBD 接口重做——`UnitaryGate`/`GeneralGate` 双类型删除，
  统一为单一 **`GateOp{N,T,M}`**（构造校验 rank/positions-distinct、support
  内部排序不再要求升序输入；**不校验 unitarity**，改由 `isunitary(g)` 显式
  检查；非幺正门的重正则化改由 `apply!(...; canonicalize = true)` 触发，
  finite 侧实现为 `truncate!(ψ; trunc)`）；本包相应调整；
- **接口**：import/export 从 `AbstractGate, UnitaryGate, GeneralGate` 换为
  **`AbstractGate, GateOp, isunitary`**（`apply!`/`swap!`/`positions`/`shift`
  不变）。infinite 侧 `apply!` 统一为
  `apply!(g::AbstractGate{2}, ψ::CanonicalIMPS; trunc, canonicalize = false)`
  ——不再按幺正/非幺正类型分派；`canonicalize = true` 时在 gather swaps 回移
  前以 `truncate!(ψ; trunc)`（逐键谱截断 + 重正则化的 infinite 版）恢复全
  规范（替代旧 GeneralGate 路径的 `gaugefix!`），对齐 FMA 语义。**门操作
  仍只支持两比特门**（`AbstractGate{2}`；FMA 的 N-site block sweep 与
  `fuse_gates`/`trotter_propagator` 为 finite/电路层设施，infinite 侧不实现）；
- **测试**：testhelpers 的 `pair_gate`（Pair 矩阵便捷构造）改委托 `GateOp`
  （构造不再拒绝非幺正矩阵）；tebd.jl 的"gate 类型" testset 重写为 GateOp
  语义（isunitary 显式断言、乱序 positions 内部排序归一、重复 positions
  拒绝），"GeneralGate" testset 改"非幺正门（canonicalize = true）"（含
  wrap 键非相邻门）；envs.jl 两处 `UnitaryGate(...)` 调用换 `GateOp`；
- **docs**：library.md（GateOp/isunitary 限定名 @docs）、conventions.md、
  algorithms.md（统一 apply! 签名 + canonicalize 语义）、README.md 同步。

## 2026-10-15 命名空间分层：Models/Defaults 子模块化 + 底层工具取消导出

- **Models 子模块**：models.jl 全部内容（`bulk_mpo`/`mpohamiltonian`/
  `heisenberg_hamiltonian`/`tfim_hamiltonian`/`heisenberg_xxz`/`tfim`/
  `fermi_hubbard` 与算符 `σx`/`σy`/`σz`/`Sx`/`Sy`/`Sz`）收进
  **`InfiniteMPSAlgorithms.Models`**（模块内 export，含依赖 `using ..`
  引入 scalartype/SchurMPOTensor/SparseIMPO/DenseIMPO/tompotensor）——
  用户经 `using InfiniteMPSAlgorithms.Models` 调用；主模块不再导出任何
  模型名（FMA 遮蔽层随之移除——模型名不再与 FMA 撞名）；
- **Defaults**：主模块 export 列表移除 `Defaults`——经
  `using InfiniteMPSAlgorithms.Defaults` 调用（src 内部引用不受影响；
  FMA Defaults 仍以 `FiniteMPSAlgorithms.Defaults` 限定区分）；
- **fuse → `_mult_site`**：站点融合核（`W * ψ` 的 kernel）私有化改名，
  不导出；定义（operators/linalg.jl）与内部调用（同文件 `W * ψ`、
  mult.jl 的 `svdguess_mult`）及注释/docstring 引用同步；
- **取消导出**：`integrate`（TDVP 局部指数核）、`isidentitylevel`/
  `isemptylevel`（Schur 层探测）——仍为公开函数，`InfiniteMPSAlgorithms.`
  限定访问（或 `using InfiniteMPSAlgorithms: xxx` 显式引入）；
- **测试**：runtests.jl 与 mpskit/testhelpers_mpskit.jl 顶部统一
  `using InfiniteMPSAlgorithms.Models` / `using InfiniteMPSAlgorithms.Defaults`；
  `fuse`/`integrate`/`isidentitylevel`/`isemptylevel` 的测试调用点改限定名
  或文件头显式 import（twosite.jl/mpo.jl）；
- **docs**：library.md 的 Models/Defaults 段改限定名 `@docs` 并补
  `TaylorCluster`；algorithms.md 的 timeevompo 段补 TaylorCluster 与
  integrate 不导出说明；index.md/README.md 的 quick-start 补
  `using InfiniteMPSAlgorithms.Models`、API 对照表同步；
- **验证**：全量分段跑（沙箱 fork 禁令下 libstdcxxprobe 启动竞态，
  `julia -e` 管道转发逐段完成）——states 177 / operators 113 /
  algorithms 626 全过（含 `_mult_site`/`integrate`/`isidentitylevel`
  限定名断言与 `using ...Models`/`using ...Defaults` 引入）；mpskit
  concordance 全部 8 个文件过（w1w2 段首跑暴露一处测试侧
  `InfiniteMPSAlgorithms.σx` 旧限定名，修为 `InfiniteMPSAlgorithms.Models.σx`
  后 50/50 全过——src 无改动）。

## 2026-10-15 TaylorCluster 时间演化 MPO（无限系统）+ debug/ 目录清空

- **接口**：`timeevompo(H::SparseIMPO, δ, alg::TaylorCluster) -> DenseIMPO`
  （[w1w2.jl]；[`TaylorCluster`](@ref) 类型本体与导出经 FiniteMPSAlgorithms
  再导出，本包 export 列表新增）。`N` 阶 Taylor cluster 展开的周期时间演化
  MPO（MPSKit `make_time_mpo` 家族对齐；Van Damme et al., SciPost Phys. 17,
  135 (2024)），`δ` 约定同 `WI`/`WII`（指数系数本身：实时 `-im·t` / 虚时
  `-τ`）。`N=1`（extension/compression 关闭）数学上 ≡ `WI`；
- **实现**：per-site 簇代数（`H^N` 的 N 折通道字典、`_site_pow`/`_taylor_*`
  全套）**逐字复用 FiniteMPSAlgorithms 的
  `timeevompo(::SchurMPOTensor, dt, ::TaylorCluster)` 单站方法**（本包
  `SparseIMPO.Ws` 即 FMA 的 `SchurMPOTensor`，`timeevompo` 经 import 合并
  方法表）；本方法只补无限侧装配——环上无 vacuum/closing 边界，每键存活
  通道 = 左站非零列 ∩ 右站非零行（MPSKit 整链 `remove_orphans!` 的逐键
  等价物），`sort!` 保原 column-major 空间序 ⇒ 输出键空间通道顺序与
  MPSKit 一致；
- **测试**：mpskit/w1w2_concordance.jl 新增 TaylorCluster testset——
  TFIM L=1（默认 N=2 与 extension/compression 开关组合 × 实时/虚时）、
  Heisenberg L=1（多通道 D=16）、L=2 单胞（异场 + 跨站 XX 的环闭合装配
  多站路径）、`N=1 ≡ 包内 WI` 自洽；判据同 WII 双轨（稠密周期 trace
  直接差 < 1e-10 主判——实测 ~1e-17 **逐位一致**、distance/fidelity 辅判）；
  testhelpers_mpskit.jl 的冲突消歧 import 列表加 `TaylorCluster`；
- **清理**：debug/ 目录全部内容删除（diag/bench/zip_* 等开发期脚本，
  历史结论已固化进 changes.md 与测试套件）；test/algorithms/compress.jl
  一处指向已删脚本的注释同步修剪；
- **验证**：w1w2_concordance.jl 单套件全过（WII + TaylorCluster 两
  testset；TaylorCluster 断言含逐位 dense repr ~1e-17）。全量测试当日被
  运行环境阻塞（沙箱 fork 禁令：julia 启动的 libstdcxxprobe 探测被拒、
  julia 无法启动，重试 ~20 分钟未恢复）；受影响测试面分析——本轮仅新增
  方法/导出/测试与注释修剪，既有测试的代码路径不变（w1w2_concordance
  已携全部改动单独全过），回归风险为零，环境恢复后可直接重跑全量。

## 2026-10-15 IDMRG2 压缩引擎 vs MPSKit 单迭代严格一致（R→L edge 的 AR[N] 一致性重构 + 去轮内单站重分割）+ 收敛判据换 bond-0 C 漂移

- **根因**（`debug/diag_idmrg2_iter.jl` 单迭代对照，identity 通道 ray residual
  0.085、窗口结构调整逐位不变 ⇒ 差异在窗口/环境管理层之外）：逐一排查
  MPSKit `approximate!(..., IDMRG2)`（src/algorithms/approximate/idmrg.jl）的
  `AC2_projection` kind 展开（`AC2_hamiltonian·AC2(above, site; kind)`——above
  为固定目标，kind 形态数学恒等）与环境 eigsolve 细节（`InfiniteEnvironments`
  的整胞 :LM 固定点 + GL 按 C 通道 overlap λ 缩放——与 `overlap_fixedpoints`
  同构）后，定位到两处**状态簿记**差异：
  1. MPSKit R→L edge 在窗口 SVD 后做一致性重构
     `ψ.AR[N] = C[N-1]⁻¹·(AL[N]·C[N])`（前除；两站分割只写 AL[N]/C[N]/AR[1]，
     AR[N] 从键权重反解）——该值被**出口 AR 串重建**消费；
  2. 本包每轮收尾的"混合规范补全"单站重分割（`leftsplit(AC[N])` 覆写
     AL[N]/C[N]、`rightsplit(AC[1])` 覆写 AR[1]）——MPSKit 无此步，覆写令
     出口 `_rebuild` 的 AR 串（AR[1] 取 rightsplit 值而非 edge v、AR[N] 取
     R→L 常规窗口 v 而非重构值）偏离 MPSKit，重建投影放大为 0.085 的态差。
- **修复**：融合引擎与 deprecated 模板同步——R→L edge 的 `_svd2!` 后新增
  `x.AR[N] = _frontdiv(x.C[N-1], _mulC(x.AL[N], x.C[N]))`（`_frontdiv` 新增
  CanonicalIMPO rank-4 分派：`(u, d)` MPS 视图上权重折左键）；删除每轮单站
  重分割（保留 AC 族刷新 `AC = AL·C` / `AC[1] = C[N]·AR[1]`——finalize 回调
  可见的族自洽，扫掠内 AC 无算子消费）。
- **窗口结构 MPSKit 对齐**（上一轮先行改动，与本修复同批提交）：两引擎扫掠
  重排为 MPSKit IDMRG2 形态——每 pass = 常规窗口 (1..N-1) + edge wrap 窗口
  (N,1)（L→R 常规 :ACAR / edge :ALAC，R→L 镜像）；`_local_AC2` 家族加
  `kind` 关键字（OverlapCache/MultCache/HadamardCache 三通道，:ACAR/:ALAC
  双形态——固定 ket 上数学恒等）；每窗口双向 transfer（MPSKit 每窗口双推，
  双方向环境窗口读时均新鲜）；每 pass 后 `normalize_envs!` 重标定；环境随
  窗口增量前推贯穿所有迭代（循环内不重解，`recalculate!` 留给出口）；出口
  `_rebuild` 规范恢复 + `recalculate!`（同 IDMRG）。
- **收敛判据**：压缩 IDMRG2 从最大逐站 Galerkin 残差换为 **bond-0 中心矩阵
  漂移** `ϵ = ‖C[0] − C_old‖`（MPSKit 同款；旧结构下 C[0] 不参与两站更新时
  判据恒零不可用，新结构 edge 窗口每 pass 更新 C[N] ≡ C[0] 良定义）。键维随
  `truncdim` 截断逐轮变化时比较前导 min 维子块（`_c_drift`，MPSKit 的
  infimum 投影语义——截断保留前导奇异方向、中心矩阵自身基下对角）。
  收敛标志 `ϵ < alg.tol`（MPSKit 语义）；`losses` 改为逐轮漂移（同 IDMRG
  引擎，无初始残差项）。
- **测试**：mpskit/compress_concordance.jl 新增 IDMRG2 两 testset——逐迭代
  对齐（k=1..5，maxiter=k/tol=0 强制恰好 k 轮，射线残差 ≤ 2e-10）与收敛
  轮数 + 终态对齐（`_mpskit_converged_iter` 加 kwargs 透传 `trscheme`，
  niter 严格相等）；arithmetics.jl 新增 `compress!（naive ⊙ 目标）≡
  hadamard!` testset（VOMPS/IDMRG/IDMRG2 三算法，niter 相等 + ray < 1e-10）
  并将 `mult! ≡ compress!` 循环扩展 IDMRG2（mpo·mps/mpo·mpo 两通道）；
  algdefs.jl 的 IDMRG2 docstring 与 docs/algorithms.md 的判据/签名描述同步。
- **验证**：全量 1351/1351（states 177 / operators 113 / algorithms 626 /
  MPSKit concordance 435）；`debug/diag_idmrg2_iter.jl` 单迭代输出+环境
  全槽位 ~1e-10（修复前 ray residual 0.085）、多迭代 k=1..5 ≤ 2e-10。

## 2026-10-13 in-place 驱动去除 changebond! 预处理：out 键维完全由调用者控制
