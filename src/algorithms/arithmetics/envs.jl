# ---------------- 压缩/代数通道的环境缓存抽象 ----------------
#
# VOMPS/IDMRG 压缩引擎（mult / hadamard / compress 共用）的环境缓存层次：
# OverlapCache（恒等通道）、MultCache（mpo·mps 施加 / mpo·mpo 组合通道）、
# HadamardCache（zip 通道）——与哈密顿量通道的 DMRGCache
# （groundstates/envs.jl）相区分。

abstract type CompressionEnvironments <: Environments end

"压缩通道缓存的站数（`Environments` 的 `length` 契约）：输入单胞长度的最小
公倍数——构造时已保证 bra 与 ket 同长、operator（若存在）长度为其因子，故
= `length(envs.bra)`。"
Base.length(envs::CompressionEnvironments) = length(envs.bra)

"压缩通道缓存标量类型（环境张量的实际 eltype——实输入下复提升后的通道算术
类型；构造器已把各槽位提升到该类型）。"
scalartype(envs::CompressionEnvironments) = scalartype(envs.lefts[1])

# ---------------- 统一压缩扫掠引擎（compress / mult / hadamard 共用） ----------------

"`_mulFront(C, AR)`：中心矩阵前乘（`AC = C·AR`；rank-3 MPS 张量与 rank-4
CanonicalIMPO 张量（在 `(u, d)` MPS 视图上）分派）。"
function _mulFront(C::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    return reshape(reshape(C * reshape(A, size(A, 1), :), size(C, 1), size(A, 2), :),
                   size(C, 1), size(A, 2), size(A, 3))
end

function _mulFront(C::AbstractMatrix, A::AbstractArray{T,4}) where {T}
    wl, u, wr, d = size(A)
    v = reshape(C * reshape(_as_mps_view(A), size(C, 1), :), size(C, 1), u * d, wr)
    return permutedims(reshape(v, wl, u, d, wr), (1, 2, 4, 3))
end

"""
    compression_sweeps!(envs::CompressionEnvironments, alg::VOMPS) -> (envs, info)

压缩通道的统一变分扫掠引擎（`compress`/`mult`/`hadamard` 及其 in-place 版本的
共享底层；MPSKit `approximate` 的 `IterativeSolver` 管道）：被优化的态即
`envs.bra`（构造缓存的初态），目标由缓存持有——OverlapCache = ket（variational
compression）、MultCache = operator·ket（mpo·mps / mpo·mpo）、HadamardCache =
zip(ket1, ket2)。`alg::VOMPS` 跑 Jacobi 式 ALS 轮：`localupdate`
（`_local_AC`/`_local_C` → `regauge!`）→ `gauge_step!` → `recalculate!`
环境热启动重解 → 扫掠后检查 Galerkin 残差。环境写回 `lefts`/`rights`、
态写回 `envs.bra`——**但 `recalculate!` 的类型守卫（实 cache 遇复不动点）
可能返回提升后的新 envs，故输入 `envs` 不保证被原地改变：调用方一律以
返回值 `envs` 为准**。返回 `(envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = [初始残差,
逐轮 Galerkin 残差...]、`converged` 收敛标志）。返回时 `envs.bra` 已按包约定
归一化且处于混合规范——调用方无需再做任何收尾。
"""
function compression_sweeps!(envs::CompressionEnvironments, alg::VOMPS)
    x = envs.bra                       # 原地演化的态（缓存 bra 本体）
    N = length(envs)
    T = scalartype(envs)
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = calc_galerkin(envs, x)
    iter = 0
    losses = [ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # similar 仅提供 eltype 模板，形状由 regauge! 的输出决定）
        ALs = [similar(x.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            ALs[ℓ] = regauge!(_local_AC(envs, ℓ), _local_C(envs, ℓ);
                              alg = alg.alg_orth)
        end
        # gauge: restore the global right gauge（动态容差）
        alg_g = updatetol(alg.alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（bra 更新 + 热启动 + 动态环境容差；recalculate! 的类型
        # 守卫可能返回提升后的新 envs——后续以输出 envs 为准，x 重绑其 bra）
        alg_envs = updatetol(alg.alg_environments, iter - 1, ϵ)
        envs = recalculate!(envs, x, alg_envs)
        x = envs.bra
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        ϵ = calc_galerkin(envs, x)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    deprecated_compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG) -> (envs, info)

统一扫掠引擎的 IDMRG 模板（MPSKit `approximate(ψ₀, ..., IDMRG())`）：sequential
Gauss–Seidel double sweep with on-the-fly environment transfer
([`transfer_leftenv!`](@ref)/[`transfer_rightenv!`](@ref)), `leftorth`/
`rightorth` splits of the normalized local projections, per-double-sweep
environment rescaling ([`normalize_envs!`](@ref)) and center-matrix-drift
convergence `ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is
rebuilt from the `AR` string（[`_rebuild`](@ref)，容差取 `alg_gauge` 的动态
适配）and the environments are re-solved for the final state
([`recalculate!`](@ref))。全部原地：态写回 `envs.bra`、环境写回
`lefts`/`rights`，返回 `(envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = 逐轮中心
矩阵漂移、`converged` 收敛标志）。返回时 `envs.bra` 已按包约定归一化且处于
混合规范——调用方无需再做任何收尾。

**Deprecated**：驱动的 IDMRG 现走融合快路径 [`compression_sweeps!`](@ref)
（同一收缩图的分解重排，浮点舍入内严格等价）；本模板保留作对照基准（`invoke`
可达）。
"""
function deprecated_compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG)
    x = envs.bra                       # 原地演化的态（缓存 bra 本体）
    N = length(envs)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _local_AC(envs, ℓ)
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            transfer_leftenv!(envs, x, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _local_AC(envs, ℓ)
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            transfer_rightenv!(envs, x, ℓ - 1)
        end
        # 环境重标定
        normalize_envs!(envs, x)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建混合规范（_rebuild，容差取 alg_gauge 的动态适配），
    # 环境对终态重解。构造器的 gaugefix! 迭代自带 normalize!(C)（
    # uniform_leftorth! 的每轮不变量），扫掠内逐站 normalize!(AC) + 等距
    # 分解亦逐键保持 ‖C‖=1，无需额外全局归一化。
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = _rebuild([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol,
                 maxiter = alg_g.maxiter)
    envs = recalculate!(envs, x, alg.alg_environments)
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

# ---------------- compression_sweeps!（IDMRG/IDMRG2 融合引擎）的折叠 helpers ----------------

"链在键 ℓ 的中心矩阵（Canonical 链的 `AC = AL·C` / `AC = C·AR` 分解用）；
Dense 链的 AC ≡ AL/AR 纯视图（同一数据、无 C 分解）返回 `nothing`。"
_bondC(chain::Union{CanonicalIMPS,CanonicalIMPO}, ℓ::Int) = chain.C[ℓ]
_bondC(::Union{DenseIMPS,DenseIMPO}, ℓ::Int) = nothing

"`_bondCT(C)`：右环境侧键权重折叠的转置包装——AR 左键 = C 的**列**基
（C 行 = AL 右键），`_fold_bond` 消费被折腿对 C 的行，故折 S0/GR 侧的
AR 基键腿须传 `Cᵀ`（Dense 链恒 `nothing`）。"
_bondCT(::Nothing) = nothing
_bondCT(C::AbstractMatrix) = transpose(C)

"""
    _fold_bond(K0, Val(leg), C) -> K′

`K0` 的第 `leg` 条腿与键权重矩阵 `C` 收缩（`AC = AL·C` 的 C 折叠）：
`K′[...腿 leg 换成 C 的列指标...] = Σ K0[..., r, ...]·C[r, c]`（单个 GEMM）；
`C === nothing` 时原样返回 `K0`。
"""
function _fold_bond(K0::AbstractArray{T,N}, ::Val{leg}, ::Nothing) where {T,N,leg}
    return K0
end
function _fold_bond(K0::AbstractArray{T,N}, ::Val{leg}, C::AbstractMatrix) where {T,N,leg}
    perm = (ntuple(i -> i, leg - 1)..., ntuple(i -> leg + i, N - leg)..., leg)
    Kp = permutedims(K0, perm)                     # 其余腿在前、leg 最后
    rest = ntuple(i -> size(Kp, i), N - 1)
    Kf = reshape(Kp, prod(rest), size(C, 1)) * C
    return permutedims(reshape(Kf, (rest..., size(C, 2))), invperm(perm))
end

# `compression_sweeps!` 的 IDMRG/IDMRG2 融合引擎（InfiniteTEMPO 的
# get_left_xy 机制）定义于下——签名引用抽象类 CompressionEnvironments（本文件
# 已定义），无导入绑定求值顺序约束；通道分派核（`_fuse_left`/`_local_AC_fused`/
# `_push_*_fused!`/`_local_AC2_fused[R]`）在各通道文件（mult.jl/hadamard.jl/
# compress.jl），调用时运行时分派。general 逐站模板保留为
# `deprecated_compression_sweeps!`（对照基准，`invoke` 可达）。

"""
    compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG) -> (envs, info)

IDMRG 在压缩/算符/zip 三通道（[`OverlapCache`](@ref)/[`MultCache`](@ref)/
[`HadamardCache`](@ref)）上的融合引擎（InfiniteTEMPO 的 `get_left_xy`
机制）：每站把 `(GL·operator.AL·ket.AL)`（zip 通道为 `GL·A2.AL·A1.AL`、
重叠通道为 `GL·ket.AL`）**一次融合**为 K0——输入物理腿收缩、算符输出物理
`u` 与两条右键开放——同一融合张量双用：

- **map**：折 AC = AL·C 的键权重（Canonical 链；Dense 链 AC ≡ AL 视图恒等，
  `_bondC` 返回 nothing）后与 `rights[ℓ]` 收缩出局部 AC（逐腿 2 张量 GEMM）；
- **push**：与 `conj(新 bra)` 的单 GEMM 出下一站环境。

替代逐站 [`deprecated_compression_sweeps!`](@ref) 模板的两次独立四张量收缩
（`_local_AC` + `transfer_leftenv!`——两者各自重复收缩 (operator, ket) 对），
每站省一次 O(D²·Dw²·d) 级大收缩。扫掠结构（Gauss–Seidel 双向、逐站
normalize! + 正交分解）、`normalize_envs!` 重标定、bond-0 C 漂移收敛判据与
收尾（`_rebuild` + `recalculate!`）与 general IDMRG 模板同构；**push 序列与
general 逐位一致**（每 pass 单方向，无跨 pass 活写）——同一收缩图的分解
重排（浮点舍入内严格等价）。
"""
function compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG)
    x = envs.bra                       # 原地演化的态（缓存 bra 本体）
    N = length(envs)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：融合张量 K0 双用——map 出 AC、
        # push 出下一站左环境）
        for ℓ in 1:N
            K0 = _fuse_left(envs, ℓ)
            x.AC[ℓ] = _local_AC_fused(K0, envs, ℓ)
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            _push_left_fused!(envs, K0, x, ℓ)
        end
        # right to left sweep（S0 双用，镜像）
        for ℓ in N:-1:1
            S0 = _fuse_right(envs, ℓ)
            x.AC[ℓ] = _local_AC_fusedR(S0, envs, ℓ)
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            _push_right_fused!(envs, S0, x, ℓ)
        end
        # 环境重标定
        normalize_envs!(envs, x)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复 + 环境重解（同 general IDMRG 模板的收尾）
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = _rebuild([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol,
                 maxiter = alg_g.maxiter)
    envs = recalculate!(envs, x, alg.alg_environments)
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG2) -> (envs, info)

IDMRG2 在压缩/算符/zip 三通道上的融合引擎（[`compression_sweeps!`](@ref)
的 IDMRG 机制的两站推广，对应 [`deprecated_compression_sweeps!`](@ref) 的
IDMRG2 模板）：每窗口 `(pos, pos+1)` 把 `(GL·operator.AL·ket.AL)`（zip 通道为
`GL·A1.AL·A2.AL`、重叠通道为 `GL·ket.AL`）一次融合为 K0，经键权重折叠
（AC = AL·C 分解；Dense 链恒等）与 next 站 AR 因子、`rights[next]` 收缩出
两站张量 `Θ`（rank-4，物理腿逐站保留），按 `alg.trunc` SVD 截断分割
（`_svd2!`）。

push 序列与 deprecated 模板**严格一致**：其每窗口的双向 push 中，对侧
（同 pass 内）的写均为死写（值总在被读前覆盖），**唯二的跨 pass 活写**是
L→R pass 窗口 1 的右推（`rights[1]`——被 R→L pass 的 wrap 窗口 N 消费）与
R→L pass 窗口 N 的左推（`lefts[1]`——被窗口 1 消费），以及 L→R pass 窗口 N
的 map 读 `rights[1]`（即窗口 1 的右推值）；融合引擎以逐 pass 单方向 push +
这两处窗口特例 push 复现全部活写（未收敛轨迹与 deprecated 模板逐位一致）。
每轮收尾（单站分解补全规范、`recalculate!`、Galerkin 收敛判据、finalize）与
deprecated IDMRG2 模板逐行对应；数值上为同一收缩图的分解重排（浮点舍入内
严格等价）。
"""
function compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG2)
    N = length(envs)
    (N ≥ 2) || throw(ArgumentError("IDMRG2 requires unit cell length >= 2"))
    x = envs.bra
    ϵ = calc_galerkin(envs, x)
    iter = 0
    losses = Float64[ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # 两站双向扫掠（cyclic，融合张量双用——map + push，严格复现 general
        # 模板的活写序列：每 pass 单方向 + 跨 pass 的两处窗口特例）
        for pos in 1:N
            K0 = _fuse_left(envs, pos)
            Θ = _local_AC2_fused(K0, envs, pos)
            _svd2!(x, pos, Θ, alg.trunc)
            _push_left_fused!(envs, K0, x, pos)      # ≡ transfer_leftenv!(envs, x, next)
            if pos == 1
                # ≡ general 的 transfer_rightenv!(envs, x, 1)：rights[1] 被
                # L→R 窗口 N 的 map 与 R→L 窗口 N 的 map/push 消费（跨 pass 活写）
                S0 = _fuse_right(envs, _mod1(2, N))
                _push_right_fused!(envs, S0, x, _mod1(2, N))
            end
        end
        for pos in N:-1:1
            S0 = _fuse_right(envs, _mod1(pos + 1, N))
            Θ = _local_AC2_fusedR(S0, envs, pos)
            _svd2!(x, pos, Θ, alg.trunc)
            _push_right_fused!(envs, S0, x, _mod1(pos + 1, N))  # ≡ transfer_rightenv!(envs, x, pos)
            if pos == N
                # ≡ general 的 transfer_leftenv!(envs, x, 1)：lefts[1] 被
                # R→L 窗口 1 的 map 消费（跨 pass 活写）
                K0 = _fuse_left(envs, N)
                _push_left_fused!(envs, K0, x, N)
            end
        end
        # 混合规范补全（单站分解，秩亏链安全）+ 环境重解 + 收敛判据——
        # 与 general IDMRG2 模板逐行对应
        for ℓ in 1:(N-1)
            x.AC[ℓ] = _mulC(x.AL[ℓ], x.C[ℓ])
        end
        x.AC[N] = _mulFront(x.C[N-1], x.AR[N])
        x.AL[N], x.C[N] = _leftsplit(x.AC[N], alg.alg_orth)
        _, x.AR[1] = _rightsplit(x.AC[1], alg.alg_orth)
        alg_envs = updatetol(alg.alg_environments, iter, ϵ)
        envs = recalculate!(envs, x, alg_envs)
        x = envs.bra
        ϵ = calc_galerkin(envs, x)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG2", iter, ϵ)
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

"`_mulC(AL, C)`：中心矩阵左乘（`AC = AL·C`；rank-3 MPS 张量与 rank-4
CanonicalIMPO 张量（在 `(u, d)` MPS 视图上）分派）。"
function _mulC(A::AbstractArray{T,3}, C::AbstractMatrix) where {T}
    return reshape(reshape(A, :, size(C, 1)) * C, size(A, 1), size(A, 2), size(C, 2))
end

function _mulC(A::AbstractArray{T,4}, C::AbstractMatrix) where {T}
    wl, u, wr, d = size(A)
    v = reshape(_as_mps_view(A), :, size(C, 1)) * C          # (wl, u·d, wr)
    return permutedims(reshape(v, wl, u, d, wr), (1, 2, 4, 3))
end

"`_mulR(AR, C)`：中心矩阵右乘（权重挂到 AR 的右键；rank-3/rank-4 分派）。"
function _mulR(A::AbstractArray{T,3}, C::AbstractMatrix) where {T}
    return reshape(reshape(A, :, size(A, 3)) * C, size(A, 1), size(A, 2), size(C, 2))
end

function _mulR(A::AbstractArray{T,4}, C::AbstractMatrix) where {T}
    wl, u, wr, d = size(A)
    v = reshape(_as_mps_view(A), :, wr) * C                  # (wl, u·d, wr)
    return permutedims(reshape(v, wl, u, d, size(C, 2)), (1, 2, 4, 3))
end

"""
    deprecated_compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG2) -> (envs, info)

统一扫掠引擎的两站模板（MPSKit `approximate` 家族的 `IIDMRG2`；三个通道
[`OverlapCache`](@ref)/[`MultCache`](@ref)/[`HadamardCache`](@ref) 全开放，
窗口最优性由 [`_local_AC2`](@ref) 的通道分派给出）。
每轮为双向两站扫掠——窗口 `(pos, pos+1)`（cyclic），[`_local_AC2`](@ref)
以 `:ACAR` 形态（`ket.AC[pos]·ket.AR[pos+1]`）给出当前环境下的精确变分
最优，按 `alg.trunc` SVD 截断并前移规范边界（`AL[pos]`/`C[pos]`/`AR[pos+1]`
逐槽写回，环境双向前推）。每轮收尾以**单站分解**补全规范（无迭代，秩亏链
安全；迭代式 string 重建在零谱键上不保态）：AC 族由 `AL·C`/`C·AR` 重算，
`leftsplit(AC[N])` 补 `AL[N]` 与 bond N 权重、`rightsplit(AC[1])` 补 `AR[1]`。
收敛判据为最大逐站 Galerkin 残差（同 VOMPS）。返回值契约同
[`deprecated_compression_sweeps!`](@ref) 的 IDMRG 方法。

**Deprecated**：驱动的 IDMRG2 现走融合快路径 [`compression_sweeps!`](@ref)
（push 活写序列与本模板逐位对齐，未收敛轨迹逐位一致）；本模板保留作对照
基准（`invoke` 可达）。
"""
function deprecated_compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG2)
    N = length(envs)
    (N ≥ 2) || throw(ArgumentError("IDMRG2 requires unit cell length >= 2"))
    x = envs.bra
    ϵ = calc_galerkin(envs, x)
    iter = 0
    losses = Float64[ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # 两站双向扫掠（cyclic：pos = N 的窗口即 wrap 对 (N, 1)；wrap 的右环境
        # rights[1] 为上一轮解——标准 Gauss–Seidel 陈旧度，随收敛自洽）
        for pos in (1:N..., N:-1:1...)
            Θ = _local_AC2(envs, pos)
            _svd2!(x, pos, Θ, alg.trunc)
            transfer_leftenv!(envs, x, _mod1(pos + 1, N))
            transfer_rightenv!(envs, x, pos)
        end
        # 混合规范补全：两站窗口替换后 AL[N]/AR[1]/C[0]/AC 族陈旧——全部用
        # **单站分解**补全（无迭代，秩亏链安全；迭代式 string 重建在零谱键上
        # 不保态）：AC[ℓ≤N-1] = AL·C、AC[N] = C[N-1]·AR[N]（环上各权重一次），
        # 随后 leftsplit(AC[N]) 补 AL[N] 与 bond N 权重 C[N]≡C[0]、
        # rightsplit(AC[1]) 补 AR[1]（其 C 部分与 C[N] 同键同谱，取 AR 即可）
        for ℓ in 1:(N-1)
            x.AC[ℓ] = _mulC(x.AL[ℓ], x.C[ℓ])
        end
        x.AC[N] = _mulFront(x.C[N-1], x.AR[N])
        x.AL[N], x.C[N] = _leftsplit(x.AC[N], alg.alg_orth)
        _, x.AR[1] = _rightsplit(x.AC[1], alg.alg_orth)
        # 环境对终态重解（热启动 + 动态容差，同 VOMPS 的 envs_step!；类型
        # 守卫可能返回提升后的新 envs——后续以输出 envs 为准）
        alg_envs = updatetol(alg.alg_environments, iter, ϵ)
        envs = recalculate!(envs, x, alg_envs)
        x = envs.bra
        # 收敛判据：最大逐站 Galerkin 残差（同 VOMPS）
        ϵ = calc_galerkin(envs, x)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG2", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

"两站窗口 SVD 截断 + 规范边界前移（`Θ` 为逐站融合物理腿的 rank-4 两站张量
`(bl, s1, s2, br)`；分割 `(1, 2) | (3, 4)` 在两站**之间**——中心谱落在中间键
上，`u` = `(bl, s1, r)`、`v` = `(r, s2, br)`（`tsvd` 保留分区结构）。
`CanonicalIMPS` 与 `CanonicalIMPO` bra 各自装配 AL/C/AR 槽位，后者在 `(u, d)`
物理腿视图上拆分——同 `_leftsplit`/`_rightsplit` 的约定；中心矩阵归一同
MPSKit 的 `normalize!(c)`）。AC 槽位不在窗口内维护（写回需bond 权重的全局
约定，留给每轮收尾的 `AC = AL·C` / `C·AR` 重算）。"
function _svd2!(x::CanonicalIMPS, pos::Int, Θ::AbstractArray{<:Any,4},
                trunc::TruncationScheme)
    u, s, v, _ = tsvd(Θ, (1, 2), (3, 4); trunc)
    normalize!(s)
    next = _mod1(pos + 1, length(x))
    x.AL[pos] = u
    x.C[pos] = Matrix(Diagonal(s))
    x.AR[next] = v
    return x
end

function _svd2!(x::CanonicalIMPO, pos::Int, Θ::AbstractArray{<:Any,4},
                trunc::TruncationScheme)
    u, s, v, _ = tsvd(Θ, (1, 2), (3, 4); trunc)
    normalize!(s)
    next = _mod1(pos + 1, length(x))
    # 物理维 (u, d) 取自家族（不截断、恒定）；**键维随各窗口截断动态变化**
    #（两站融合键 400 截到 ≤100 等），左/右键取 SVD 输出自身的维度——
    # 旧张量的键维在初猜胖于目标无损键（如 truncdim(100) > 目标 64）时
    # 与截断后的窗口键不一致，不能作为 reshape 模板
    ALp, ARn = x.AL[pos], x.AR[next]
    x.AL[pos] = permutedims(reshape(u, size(u, 1), size(ALp, 2), size(ALp, 4), length(s)),
                            (1, 2, 4, 3))
    x.C[pos] = Matrix(Diagonal(s))
    x.AR[next] = permutedims(reshape(v, length(s), size(ARn, 2), size(ARn, 4), size(v, 3)),
                             (1, 2, 4, 3))
    return x
end

"`IDMRG2` 驱动函数的初猜键维：算法级分派——`VOMPS`/`IDMRG` 取固定 `alg.D`；
`IDMRG2` 从 `alg.trunc` 提取（维数语义的方案取其上限 `D`；无维数语义
（`NoTruncation`/`TruncateRelError`）返回 `nothing`——调用方回退到目标链的
无损键维上界（初猜不截断，截断完全由扫掠内的 SVD 方案执行））。"
_alg_guess_dim(alg::Union{VOMPS,IDMRG}) = alg.D
_alg_guess_dim(alg::IDMRG2) = _alg_guess_dim(alg.trunc)
_alg_guess_dim(trunc::TruncateDim) = trunc.D
_alg_guess_dim(trunc::TruncateDimCutoff) = trunc.D
_alg_guess_dim(::TruncationScheme) = nothing
