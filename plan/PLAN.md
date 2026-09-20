# InfiniteMPSAlgorithms — 设计与实现计划（v2）

无对称性的无限 MPS/MPO 张量网络算法包。张量 = 普通 `Array`，收缩 = `TensorOperations`，
低层张量操作复用 `TEMPO/src/tensorops`。本版按以下决定修订：

1. **不实现** GradientGrassmann、changebonds、激发态（penalty/准粒子）。
2. **VUMPS / IDMRG / TDVP 只实现 single-site 版本**。
3. **实现 MPOHamiltonian**（TEMPO 风格的稀疏/Schur 表示），并在其上实现
   **W^I / W^II**（含 `ComplexStepper`）时间演化与变分 `apply`，参考 TEMPO
   `src/mpohamiltonian/`（`arXiv:1407.1832`）。
4. **MPO site 张量指标与 TEMPO 对齐**：`(wl, u, wr, d)` ——
   与 MPSKit 的 `(左键, bra, ket, 右键)` 不同（键指标在 1、3 位）。
5. 数据结构：
   - `MixedCanonicalMPS`：混合规范存储，`ACs[ℓ] = ALs[ℓ]·Cs[ℓ]`，
     与 MPSKit 的 AC/C 布局一致（AC: `(Dl, s, Dr)`，C: `(Dl, Dr)`，C[ℓ] 在 bond ℓ→ℓ+1）。
   - `InfiniteMPO`：只存一串 4 维张量 `Ws[ℓ]::Array{T,4}`，指标 `(wl, u, wr, d)`。
   - `MixedCanonicalMPO`：与 `MixedCanonicalMPS` 同布局（ACs + Cs），
     只是张量为 4 维 `(Dl, u, wr, Dr)`；用途是**把 MPO 当 MPS 用**
     （MPO 压缩 = 对其 MPS 视图做重叠最大化变分）。

---

## 1. 技术栈

- Julia ≥ 1.10；TensorOperations（`@tensor`）、KrylovKit（eigsolve/exponentiate/linsolve）、
  MatrixAlgebraKit（tensorops 后端）、LinearAlgebra/Random/Printf。
- `src/tensorops/` 整体复制自 TEMPO（truncation / matrixalgebra / tensorfactorizations /
  distance），文件头注明来源；包内自带 `scalartype`（TEMPO 里来自 ImpurityModelBase）。
- W^II 的块算符矩阵指数：直接拼成稠密 `(4d)×(4d)` 矩阵调 `LinearAlgebra.exp`，
  与 TEMPO 经 ExpExp 的块指数结果等价，避免额外依赖。

## 2. 张量与指标约定（全部与 TEMPO 对齐）

```
MPS site 张量   A[a, s, b]          (左键, 物理, 右键)
MPO site 张量   W[w, u, w′, d]      (MPO左键, bra物理, MPO右键, ket物理)
恒等 MPO        W[1, u, 1, d] = δ(u,d)（键维 1）
混合规范        AC[ℓ] = AL[ℓ]·C[ℓ]，C[ℓ] 在 bond ℓ→ℓ+1（mod N）
左环境（恒等）  L[a, b]             (bra键, ket键)
左环境（MPO）   L[a, w, b]          (bra键, MPO键, ket键)
```

核心收缩（`@tensor` 实现时的指标样板）：

```julia
# 恒等通道推进
@tensor L′[a′, b′] := conj(A[a, s, a′]) * L[a, b] * A[b, s, b′]
# MPO 通道推进
@tensor L′[a′, w′, b′] := conj(Ā[a, ū, a′]) * L[a, w, b] * W[w, ū, w′, d] * A[b, d, b′]
# 有效哈密顿量 H_AC
@tensor AC′[b, d, b′] := FL[ā, w, b] * W[w, ū, w′, d] * FR[b′, w′, b̄] * conj(AC[ā, ū, b̄])
# 有效哈密顿量 H_C
@tensor C′[b, b′] := FL[ā, w, b] * conj(C[ā, b̄]) * FR[b′, w, b̄]
```

## 3. 包结构

```
src/
├── InfiniteMPSAlgorithms.jl
├── utils.jl                     # scalartype、Defaults、日志
├── tensorops/                   # 复制自 TEMPO + tensorops.jl
├── mps/
│   ├── mixedcanonicalmps.jl     # MixedCanonicalMPS（ACs + Cs）
│   ├── infinitempo.jl           # InfiniteMPO（4 维张量串）
│   ├── mixedcanonicalmpo.jl     # MixedCanonicalMPO + MPS 视图 + mpo_compress
│   ├── constructors.jl          # random/product、identity_mpo、random_mpo
│   ├── canonical.jl             # 周期 QR、C 固定点、canonicalize/fixgauge
│   └── transfer.jl              # 环境推进 kernel、TransferMatrix、dominant env
├── environments/
│   ├── environments.jl          # Environments（恒等 op=nothing / InfiniteMPO）
│   └── effective.jl             # H_AC / H_C 可调用有效映射
├── mpohamiltonian/              # 参照 TEMPO src/mpohamiltonian/
│   ├── sparsempotensor.jl       # SparseMPOTensor（算符矩阵，含 isid/phydim）
│   ├── schurmpotensor.jl        # SchurMPOTensor（上三角、dummy 键在第 1 行/末列）
│   ├── mpohamiltonian.jl        # MPOHamiltonian + tompotensors + infinite_mpo
│   └── w1w2.jl                  # timeevompo（WI/WII/ComplexStepper）→ InfiniteMPO
├── algorithms/
│   ├── common.jl                # Algorithm、默认参数、迭代日志
│   ├── vumps.jl                 # single-site VUMPS
│   ├── idmrg.jl                 # single-site IDMRG（SVD 分裂 + 环境推进）
│   ├── tdvp.jl                  # single-site TDVP（前后向半步扫描）
│   └── apply.jl                 # 重叠最大化变分 apply（供 W^I/W^II 时间演化与 MPO 压缩）
├── observables/
│   ├── expectation.jl           # expectation_value（恒等/MPO/单点/多点）
│   ├── correlation.jl           # correlation_matrix
│   └── entropy.jl               # 纠缠谱/熵（由 C 的奇异值）
└── models.jl                    # TFIM / XXZ / Hubbard（经 Schur bulk → InfiniteMPO）
```

## 4. API 一览

```julia
# ---- 数据结构 ----
MixedCanonicalMPS(ACs, Cs)          # 原始构造（带一致性检查）
MixedCanonicalMPS(As)               # 由普通 A 张量规范化（周期 QR + C 固定点）
InfiniteMPO(Ws)                     # 4 维张量串
MixedCanonicalMPO(ACs4, Cs)
# 访问: length/getindex/setindex!/copy/dag(conj)/scalartype/bond/maxbond/physicaldims
ALs(ψ)                              # 由 AC·C⁻¹ 极分解逐 site 导出（左正交）
ARs(ψ)                              # AC·C₊⁻¹ 导出（右正交）

# ---- 转移矩阵与环境 ----
push_env_left(L, A) / push_env_left(L, W, A)     # 环境推进 kernel（right 同理）
TransferMatrix(ψ) / TransferMatrix(W, ψ)         # 融合映射（size × mul! × call）
dominant_env(ψ; side, which) -> (λ, L)            # KrylovKit.eigsolve
environments(ψ) / environments(W, ψ) -> Environments
H_AC(envs, ℓ) / H_C(envs, ℓ)                      # 可调用有效哈密顿量

# ---- 基态（single-site）----
find_groundstate(ψ₀, H::Union{Nothing,InfiniteMPO}, alg) -> (; ψ, envs, E)
struct VUMPS;  tol; maxiter; verbosity; alg_eigsolve end
struct IDMRG;  trunc; tol; maxiter; verbosity end

# ---- 时间演化 ----
timeevompo(bulk::SchurMPOTensor, dt, alg::Union{WI,WII,ComplexStepper}) -> InfiniteMPO
time_evolve(ψ₀, H, dt, tfinal; alg::TDVP, observer) -> (; ψ, history)
apply(ψ₀, W::InfiniteMPO; alg::VariationalApply) -> (; ψ, overlap)   # W^I/W^II 逐步施加

# ---- MPO 当 MPS 用 ----
asmps_view(W::MixedCanonicalMPO) -> Vector{Array{T,3}}   # (wl, u*d, wr)
mpo_compress(W::InfiniteMPO, D; trunc, tol, maxiter) -> InfiniteMPO

# ---- 观测量 ----
expectation_value(ψ)                  # 单 site 范数因子 ⟨ψ|ψ⟩^{1/N}
expectation_value(ψ, W)               # 单 site ⟨ψ|W|ψ⟩^{1/N}（主特征值）
expectation_value(ψ, O, ℓ) / expectation_value(ψ, ops::Pair...)
correlation_matrix(ψ, O1, O2; range)  # C_k, k=0..range（连通）
entanglement_spectrum(ψ; ℓ) / entanglement_entropy(ψ; ℓ, α)

# ---- 模型（Schur bulk → InfiniteMPO）----
nn_term(a, b; coeff)                  # ExponentialDecayTerm(α=0)
tfim(; J, h, T) / heisenberg_xxz(; J, Δ, T) / fermi_hubbard(; t, U, μ, T)
infinite_mpo(bulk::SchurMPOTensor)    # 周期 bulk → InfiniteMPO
```

## 5. 核心算法要点（single-site）

### VUMPS
每轮迭代：
1. `envs = environments(H, ψ)`（全量重算固定点）。
2. 逐 site 解 `H_AC`/`H_C` 最小本征问题（KrylovKit `eigsolve(:SR, ishermitian)`，
   当前 AC/C 作热启动向量）。
3. 规范恢复：`AL[ℓ] = polarleft(AC′[ℓ]·C′[ℓ]⁻¹)`；
   `C` = 混合转移 `R ↦ Σ AL·R·AL†` 的主本征向量（PSD，tsvd 开方、相位吸收、归一化）；
   `AC = AL·C`。
4. 收敛：`max‖ΔAC‖, ‖ΔC‖ < tol`。能量 = Σ⟨H_AC⟩/(N·⟨AC|AC⟩)。

### IDMRG（single-site）
初始化固定点环境后逐 site 扫描：`AC′ = argmin H_AC` →
`tsvd(AC′)` 截断分裂 `AL·C`（trunc 生效）→ `AC = AL·C` → FL 增量推进；
每轮扫描结束全量重算环境修正漂移。无 C 子问题（与 VUMPS 的差别所在）。

### TDVP（single-site）
时间步 dt = 前向半步 + 后向半步。每 site：
`AC ← exp(−i·τ·H_AC)·AC` → QR 分裂 `AL·C` → `C ← exp(−i·(−τ)·H_C)·C`（投影分裂），
环境随扫描增量推进。τ = dt/2；实/虚时间由 dt 的复相位控制。
局部指数用 `KrylovKit.exponentiate(ishermitian=true)`。

### W^I / W^II（arXiv:1407.1832，参照 TEMPO `w1w2.jl`）
对 Schur bulk（块 `D=[1,1]`、`C=第1行`、`B=末列`、`A=内块`）：
- **W^I**：`[[I+dt·D, √dt·C],[√dt·B, A]]`（一阶）。
- **W^II**：逐块拼 `(4d)×(4d)` 下三角块矩阵
  `[[D,0,0,0],[δ₂C,D,0,0],[δ₁B,0,D,0],[A,δ₁B,δ₂C,D]]`（δ₁δ₂=dt，`_sqrt2` 分裂），
  稠密 `exp` 后取第 1..d 列的三段 → 新的 C/B/A 块（二阶）。
- **ComplexStepper**：`(1∓i)dt/2` 两步串联（二阶）。
- 输出经 `infinite_mpo` 转为周期 `InfiniteMPO`（键态 1 = identity 通道，
  on-site 项并入 `W[1,·,1,·]`）。

### apply（变分 MPO·MPS）
目标：`min ‖W|ψ⟩ − |x⟩‖` ⟺ 固定 ⟨x|x⟩=1 时 `max Re⟨x|B⟩`，
其中 `B_i = fuse(W_i, A_i)` 为融合 MPS 张量。局部解：
`v = LB·B_i·RB`，解 `(Lx⊗Rx)y = v`（CG），`AC = y/√(y†Py)`，
SVD 分裂（trunc 生效）+ 环境推进，扫描至重叠收敛。
`mpo_compress` 复用同一机制（B = MPO 的 MPS 视图）。

## 6. 测试与验收

| 项 | 基准 |
|---|---|
| 规范 | `‖ΣAL†AL−1‖≈0`；`AC = AL·C = C·AR`；`norm(ψ) = norm(ψ.AC[1])` |
| 固定点 | `‖L·T−λL‖` 残差；env 厄米性 |
| VUMPS | XXZ(Δ=1): `e₀ = 1/4−ln2 ≈ 0.44314718056`；TFIM(J=h=1): `e₀ = −4/π ≈ −1.27323954` |
| IDMRG | 与 VUMPS 同一能量（相对差 < 1e-6） |
| MPO | 恒等 MPO（和式语义）期望 = N；模型 MPO 期望 = VUMPS 能量 |
| W^I/W^II | W^I 能量误差 O(dt)；W^II O(dt²)；W^II 范数守恒 |
| apply | U=I（空哈密顿量）状态不变；W^II 演化能量守恒 |
| TDVP | 实时间能量守恒；虚时间收敛到基态能量 |
| 观测量 | 乘积态精确值；correlator 与显式收缩一致；熵（乘积态=0） |

---

## 9. 当前状态（2026-09-18，第二轮：严格对齐 MPSKit）

**方向变更**：不再追求稠密 `InfiniteMPO` 上 VUMPS/IDMRG 收敛（MPSKit 对稠密
`InfiniteMPO` 同样存在恒等通道污染问题，行为保持一致即可）。改为完整移植
MPSKit 的 Jordan 结构哈密顿量存储与逐层环境求解。

**本轮已完成**：
1. `JordanMPOTensor`（无对称性版本）：上三角块矩阵 `[[1,C,D],[0,A,B],[0,0,1]]`，
   块存储 `A(a,d,a,d)/B(a,d,d)/C(d,a,d)/D(d,d)`，支持 `W[i,j]` 块矩阵式访问、
   `Matrix{Union{Missing,Number,Matrix}}` 构造器、`tompotensor` 稠密化。
2. `MPOHamiltonian`（`FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` 别名）：
   `InfiniteMPOHamiltonian(Ws::Vector{<:Matrix})`、`isidentitylevel`/`isemptylevel`、
   `getproperty` 的 A/B/C/D 块、`H₁+H₂` 块 cat、`H+λs` 能量平移；
   旧 TEMPO 风格有限稀疏链改名 `SparseMPO`。
3. 环境对齐 MPSKit：单位层（level 1/nlvls）固定点 = ρ = I（本包规范）；
   中间层解 `(1 − T_diag)·x = RHS`（KrylovKit GMRES，`linsolve` 热启动）；
   **恒等层**用 `regularize!`（每次转移后投影 ρ 分量）解 `(1 − T_reg)x = RHS`，
   再逐 site 投影固定点分量（MPSKit 的 "hacky renormalization"）；
   **空层**（对角通道为零，如 Heisenberg/TFIM 的全部中间层）保留非齐次 RHS。
4. `H_AC`/`H_C` 改为 MPSKit 的严格线性算子形式（去掉对 x 的 conj；
   GR 环境改为 `(ket, w, bra)` 指标约定），复数态下亦正确。
5. `expectation_value(ψ, H::MPOHamiltonian)`：MPSKit 的**闭合列公式**
   （只收缩 `H[site][:, 1, 1, end]`：`D`/`B`/恒等簿记 + GR 的 level-nlvls 块）。
6. VUMPS 严格按 MPSKit 模板：`localupdate_step!`（AC/C 各自 `fixedpoint` + `regauge!`）→
   `gauge_step!`（`gaugefix!(ψ, ALs, C[end]; order=:R)` 后 `AC = AL·C`）→
   `envs_step!`（`recalculate!`）→ `calc_galerkin`。
7. IDMRG 严格按 MPSKit `_localupdate_sweep_idmrg!`：前向扫描（AC 解 + `left_orth` +
   `transfer_leftenv!`）、后向扫描（AC 解 + `right_orth` + `transfer_rightenv!`），
   `ϵ = ‖C[0] − C_old‖`，结束从 `AR` 重建混合规范（对标 `InfiniteMPS(mps.AR)`）。
8. 模型：`mpohamiltonian(h1, pairs)` 层矩阵构造 + `tfim_hamiltonian`/
   `heisenberg_hamiltonian`；`tfim`/`heisenberg_xxz`/`fermi_hubbard` 返回值
   增加 `hamiltonian` 字段；`make_time_mpo(H::MPOHamiltonian, ...)` 经
   `schurbulk` 转换走 W^I/W^II 机制。
9. `debug/` 对齐环境：`debug/Project.toml`（dev 本包 + MPSKit/TensorKit）、
   `mpsconvert.jl`（指标转换适配层：MPO 需置换 `(1,2,4,3)`，MPS/环境零置换）、
   `envs_alignment.jl`（稠密形式/环境/期望/H_AC/H_C 逐项对齐）、
   `vumps_idmrg_alignment.jl`（同一初态同一 H 的基态能量对齐）。

**待验证/待办（2026-09-18 会话末快照）**：

测试现状 64/81 通过。结构与单步验证全部成立：
- H_AC / H_C 厄米性 1e-16（随机复态、Jordan 环境）；
- TFIM 乘积态能量精确（D=1 时 E = -h 精确成立，验证闭合列公式与
  恒等层 regularize 语义自洽）；
- 规范不变量、Jordan 稠密化、块访问、`H+λs`、`make_time_mpo(H)` 全部通过。

**未决根因（唯一）**：VUMPS/IDMRG 迭代不收敛到精确基态。现象：
- 单步方向正确（localupdate 后 E 下降、H_AC 最小本征对被找到）；
- 但迭代中 E 读数漂移/振荡（IDMRG 增量推挤下每迭代漂移恰 −2·e₀；
  已加每迭代 `recalculate!` 后变为振荡）；
- 局部 Rayleigh 收敛到 −0.4 左右后停滞，galerkin ≈ 0.4 不再下降。

**判定手段（已就绪，待可运行的 Julia 环境）**：`debug/` 对齐脚本。
环境搭建（本会话被沙箱阻断，用户侧可直接执行）：
```bash
cd debug && julia --project=. -e 'using Pkg; Pkg.develop(path="..")'
julia --project=. debug/envs_alignment.jl        # 环境/H_AC/H_C/期望逐项对照
julia --project=. debug/vumps_idmrg_alignment.jl # VUMPS/IDMRG 基态能量对照
```
怀疑点（按优先级）：① 恒等层 `linsolve` 的 `regularize` 语义与 MPSKit 的
`RegTransferMatrix + 逐 site 减固定点` 仍有细微差别（lvec/rvec 配对或
`flip` 方向）；② 逐层 RHS/cyclethrough 的传播次序在 L>1 时的差异；
③ VUMPS gauge 恢复的 C 收敛准则。对齐脚本会直接打印两包环境张量与
每步能量差，可立即定位。

**其他已完成的小修正**：`push_env_right` 右环境改为 `(ket, w, bra)`
（MPSKit 约定，修复杂数正确性）；correlator 自包含 kernel；`scalartype`
补全（MPOHamiltonian）；旧 TEMPO 稀疏链改名 `SparseMPO`。
