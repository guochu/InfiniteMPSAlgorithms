# 变更记录（接口）

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
