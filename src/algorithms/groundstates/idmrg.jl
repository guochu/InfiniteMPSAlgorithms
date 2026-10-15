# ---------------- IDMRG（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，single-site） ----------------
#
# 算法定义（IDMRG 参数对象）见 algdefs.jl；哈密顿量环境缓存（DMRGCache）的
# 定义与构造器见 groundstates/envs.jl——本文件只剩 IDMRG 算法本体。

"MPSKit's `_localupdate_sweep_idmrg!`: forward + backward sweep; returns
`(ψ, envs, C_old, E)`. 与 MPSKit 逐位对齐的两处扫描语义：分裂因子正对角化
（`alg_orth = QRpos`，同 `left_orth!/right_orth!(; positive = true)`）；分裂后
把 `AC[pos]` 原地覆盖为对应的 `AL`/`AR`（MPSKit 的 `left_orth!`/`right_orth!`
原地语义）——这决定下一站 `:SR` eigsolve 的初值（前向末站/反向首站保留 AC，
同 MPSKit 的非原地特例），动态容差松紧下初值差异会直接进入迭代轨迹。"
function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve, alg_orth)
    N = length(ψ)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:N
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = _leftsplit(ψ.AC[pos], alg_orth)
        pos == N || copyto!(ψ.AC[pos], ψ.AL[pos])
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end
    # right to left sweep
    for pos in N:-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.C[pos - 1], ψ.AR[pos] = _rightsplit(ψ.AC[pos], alg_orth)
        pos == 1 || copyto!(ψ.AC[pos], ψ.AR[pos])
        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end

"""
    find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                     alg::IDMRG, [envs]) -> (ψ, envs, info)

IDMRG ground-state search（`operator` 的两种形式同
[`find_groundstate`](@ref) 的 VUMPS 版说明：`SparseIMPO` 闭列公式求能量、
`DenseIMPO` 周期 trace 收缩对标 MPSKit 的 `InfiniteMPO` 通道）。The third
return is the [`IterativeConvergenceInfo`](@ref)（`niter` 迭代轮数、`losses` =
[初始 Galerkin 残差, 逐轮 bond-0 中心矩阵漂移...]、`converged` 收敛标志）。
"""
function find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                          alg::IDMRG,
                          envs::Environments = DMRGCache(ψ₀, operator))
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    E = expectationvalue(ψ, operator, envs)
    alg.verbosity > 0 && _logiter(stdout, "IDMRG", 0, ϵ, "f" => E)
    iter = 0
    losses = Float64[ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # MPSKit 第 iter 次扫描时 state.iter = iter-1
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter - 1, ϵ)
        ψ, envs, C_old, E_new = _localupdate_sweep_idmrg!(ψ, operator, envs,
                                                          alg_eigsolve, alg.alg_orth)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        ψ, envs = alg.finalize(iter, ψ, operator, envs)
        # error criterion（bond 0 中心矩阵之差）
        ϵ = norm(ψ.C[0] - C_old)
        push!(losses, ϵ)
        # new energy
        ΔE = (E_new - E) / 2
        E = E_new
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ, "f" => E, "ΔE" => ΔE)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建（对标 MPSKit 的 `InfiniteMPS(mps.AR)`，容差取
    # alg_gauge 的动态适配）
    N = length(ψ)
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = CanonicalIMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, IterativeConvergenceInfo(iter, losses, converged)
end

# ---------------- IDMRG2（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，two-site） ----------------
#
# 算法定义（IDMRG2 参数对象）见 algdefs.jl；两站有效哈密顿量
# （AC2 / AC2_hamiltonian）见 groundstates/effective.jl。

"`_div_tail(A, C)`：右除（`X·C = A` 的解，rank-3；MPSKit `A / C`）。"
_div_tail(A::AbstractArray{T,3}, C::AbstractMatrix) where {T} =
    reshape(reshape(A, :, size(C, 2)) / C, size(A, 1), size(A, 2), size(A, 3))

"`_frontdiv(C, A)`：前除（`C·X = A` 的解，rank-3；MPSKit `C \\ A`）。"
_frontdiv(C::AbstractMatrix, A::AbstractArray{T,3}) where {T} =
    reshape(C \ reshape(A, size(C, 1), :), size(C, 1), size(A, 2), size(A, 3))

"`_frontdiv(C, A)`：CanonicalIMPO rank-4 分派（在 `(u, d)` MPS 视图上、权重
折左键；压缩引擎 IDMRG2 的 R→L edge `AR[N] = C[N-1]⁻¹(AL[N]·C[N])` 用）。"
function _frontdiv(C::AbstractMatrix, A::AbstractArray{T,4}) where {T}
    wl, u, wr, d = size(A)
    v = reshape(C \ reshape(_as_mps_view(A), size(C, 1), :), size(C, 2), u * d, wr)
    return permutedims(reshape(v, size(C, 2), u, d, wr), (1, 2, 4, 3))
end

"两站窗口的 SVD 截断分割（rank-4 `:ACAR`/`:ALAC` 张量，分割 `(1, 2) | (3, 4)`
在两站之间；`tsvd` 保留分区结构 `u = (bl, s1, r)`、`v = (r, s2, br)`），中心
矩阵归一同 MPSKit 的 `normalize!(c)`。返回 `(al, c, ar)`。"
function _idmrg2_split(ac2::AbstractArray{T,4}, trunc::TruncationScheme) where {T}
    u, s, v, _ = tsvd(ac2, (1, 2), (3, 4); trunc)
    normalize!(s)
    return u, Matrix(Diagonal(s)), v
end

"MPSKit's `_localupdate_sweep_idmrg2!`: L→R 两站窗口（`:ACAR`）+ edge（中心
移位 + wrap 窗口 `:ALAC`）+ R→L（`:ALAC`）+ edge（`:ACAR`，同时记录能量）；
返回 `(ψ, envs, C_old, E)`。与 MPSKit 逐位对齐的两处扫描语义：分裂因子中心
矩阵正归一（`normalize!(c)`）；edge 处的 `AL / C`、`C \\ AC` 除法与 MPSKit 的
`/`、`\\` 同义（LU 求解）。"
function _localupdate_sweep_idmrg2!(ψ, H, envs, alg_eigsolve, alg_orth, trunc)
    N = length(ψ)
    local E
    # left to right：窗口 (pos, pos+1)，:ACAR
    for pos in 1:(N-1)
        ac2 = AC2(ψ, pos)
        h = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h, ac2, :SR, alg_eigsolve)
        al, c, ar = _idmrg2_split(ac2′, trunc)
        ψ.AL[pos] = al
        ψ.C[pos] = c
        ψ.AR[pos+1] = ar
        ψ.AC[pos+1] = _mulFront(c, ar)
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end
    # edge：中心从 bond N 移出（AL[N] = AC[N]/C[N]、AC[1] = AL[1]·C[1]），
    # wrap 窗口 (N, 1) 以 :ALAC（跨 bond 0）
    ψ.AL[N] = _div_tail(ψ.AC[N], ψ.C[N])
    ψ.AC[1] = _mulC(ψ.AL[1], ψ.C[1])
    ac2 = AC2(ψ, 0; kind = :ALAC)
    h = AC2_hamiltonian(0, ψ, H, ψ, envs)
    _, ac2′ = fixedpoint(h, ac2, :SR, alg_eigsolve)
    al, c, ar = _idmrg2_split(ac2′, trunc)
    ψ.AL[N] = al
    ψ.C[N] = c                              # ≡ C[0]
    ψ.AR[1] = ar
    ψ.AC[N] = _mulC(al, c)
    ψ.AC[1] = _mulFront(c, ar)
    ψ.AL[1] = _div_tail(ψ.AC[1], ψ.C[1])
    C_old = c
    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)
    # right to left：窗口 (pos, pos+1)，:ALAC
    for pos in (N-1):-1:1
        ac2 = AC2(ψ, pos; kind = :ALAC)
        h = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h, ac2, :SR, alg_eigsolve)
        al, c, ar = _idmrg2_split(ac2′, trunc)
        ψ.AL[pos] = al
        ψ.AC[pos] = _mulC(al, c)
        ψ.C[pos] = c
        ψ.AR[pos+1] = ar
        ψ.AC[pos+1] = _mulFront(c, ar)
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end
    # edge（:ACAR at 0）：中心移回 + 记录能量
    ψ.AC[N] = _mulFront(ψ.C[N-1], ψ.AR[N])
    ψ.AR[1] = _frontdiv(ψ.C[N], ψ.AC[1])
    ac2 = AC2(ψ, 0)
    h = AC2_hamiltonian(0, ψ, H, ψ, envs)
    E, ac2′ = fixedpoint(h, ac2, :SR, alg_eigsolve)
    al, c, ar = _idmrg2_split(ac2′, trunc)
    ψ.AL[N] = al
    ψ.C[N] = c
    ψ.AR[1] = ar
    ψ.AR[N] = _frontdiv(ψ.C[N-1], _mulC(al, c))
    ψ.AC[1] = _mulFront(c, ar)
    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)
    return ψ, envs, C_old, E
end

"""
    find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                     alg::IDMRG2, [envs]) -> (ψ, envs, info)

Two-site IDMRG ground-state search（严格对标 MPSKit 的同名 `IDMRG2`：
L→R/R→L 两站窗口 + 双 edge 的中心移位；两站窗口取有效哈密顿量 `:SR`
本征对后按 `alg.trunc` 截断——两站更新可跨过单站鞍点、键维随截断方案
动态增长）。`operator` 的两种形式同 [`find_groundstate`](@ref) 的 IDMRG
版说明。要求单胞长度 ≥ 2。The third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 迭代轮数、`losses` = [初始
Galerkin 残差, 逐轮 bond-0 中心矩阵漂移...]、`converged` 收敛标志）。
"""
function find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                          alg::IDMRG2,
                          envs::Environments = DMRGCache(ψ₀, operator))
    (length(ψ₀) ≥ 2) ||
        throw(ArgumentError("IDMRG2 requires unit cell length >= 2"))
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    E = expectationvalue(ψ, operator, envs)
    alg.verbosity > 0 && _logiter(stdout, "IDMRG2", 0, ϵ, "f" => E)
    iter = 0
    losses = Float64[ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # MPSKit 第 iter 次扫描时 state.iter = iter-1
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter - 1, ϵ)
        ψ, envs, C_old, E_new = _localupdate_sweep_idmrg2!(ψ, operator, envs,
                                                           alg_eigsolve,
                                                           alg.alg_orth, alg.trunc)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        ψ, envs = alg.finalize(iter, ψ, operator, envs)
        # error criterion（bond 0 中心矩阵之差）
        ϵ = norm(ψ.C[0] - C_old)
        push!(losses, ϵ)
        # new energy
        ΔE = (E_new - E) / 2
        E = E_new
        alg.verbosity > 0 && _logiter(stdout, "IDMRG2", iter, ϵ, "f" => E, "ΔE" => ΔE)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建（对标 MPSKit 的 `InfiniteMPS(mps.AR)`，容差取
    # alg_gauge 的动态适配）
    N = length(ψ)
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = CanonicalIMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, IterativeConvergenceInfo(iter, losses, converged)
end
