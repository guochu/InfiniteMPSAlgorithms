# ---------------- 纯重叠通道：OverlapCache 与 compress 的 VOMPS / IDMRG 引擎 ----------------
#
# ⟨below|above⟩（identity 通道，无算符）的环境缓存与变分扫掠——compress 的引擎
# （MPSKit `approximate(ψ₀, ϕ, ...)` 的无算符分支；mult 的 MPO 施加通道见 mult.jl
# 的 MultCache 版本）。环境张量为 rank-2 矩阵：`lefts[ℓ]` = `(below bond, above
# bond)`、`rights[ℓ]` = `(above bond, below bond)`，相比 rank-3 的 w=1 占位维
# 省一份冗余维。
#
# 注意：rank-2 双层 push（push_env_left/right(::AbstractMatrix, above, below)）
# 的参数序是 `(above, below)`——conj 落在 below 上；与 rank-3 三元方法的
# `(below, above)` 相反。下面的调用点均已按此对齐（conj 恒落在演动态 x 上）。

"`_mapAC(GL, GR, ketAC)`：恒等通道的 rank-2 局部 AC 投影。"
function _mapAC(GL::AbstractMatrix, GR::AbstractMatrix,
                ketAC::AbstractArray{Tk,3}) where {Tk}
    @tensor ACnew[aL, p, aR] := GL[aL, bL] * ketAC[bL, p, bR] * GR[bR, aR]
end

"`_mapC(GL, GR, ketC)`：恒等通道的 rank-2 局部 C 投影。"
function _mapC(GL::AbstractMatrix, GR::AbstractMatrix, ketC::AbstractMatrix)
    @tensor Cnew[a, a′] := GL[a, b] * ketC[b, b′] * GR[b′, a′]
end

"""
    OverlapCache(bra, ket, lefts, rights)
    OverlapCache(ψ) -> OverlapCache
    OverlapCache(below, above; tol, krylovdim, maxiter) -> OverlapCache

Environments of the pure overlap channel `⟨bra|ket⟩` (identity channel): the
left/right fixed points of the double-layer transfer, used by the VOMPS / IDMRG
compression sweeps of `compress` ([`_overlap_vomps_sweeps`](@ref) /
[`_overlap_idmrg_sweeps`](@ref)).

- `lefts[ℓ]`: rank-2 `(below bond, above bond)` left environment of site ℓ;
- `rights[ℓ]`: rank-2 `(above bond, below bond)` right environment of site ℓ;
- `OverlapCache(ψ)`: the AL/AR gauges make the fixed points identity matrices;
- `OverlapCache(below, above)`: the fixed points are obtained from the :LM
  eigenpairs of the fused transfer `T(above.AL, below.AL)`
  ([`overlap_fixedpoints`](@ref)); normalization
  mirrors MPSKit (unit-Frobenius GRs, GLs scaled by the local C-channel
  overlap λ).
"""
struct OverlapCache{B<:CanonicalIMPS,K<:CanonicalIMPS,T} <: Environments
    bra::B
    ket::K
    lefts::Vector{Array{T,2}}
    rights::Vector{Array{T,2}}
end

"Identity channel: in the AL/AR gauges the fixed points are identity matrices."
function OverlapCache(ψ::CanonicalIMPS; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Matrix{T}}(undef, N)
    rights = Vector{Matrix{T}}(undef, N)
    for ℓ in 1:N
        Dl, Dr = size(ψ.AL[ℓ], 1), size(ψ.AL[ℓ], 3)
        lefts[ℓ] = Matrix{T}(I, Dl, Dl)
        rights[ℓ] = Matrix{T}(I, Dr, Dr)
    end
    return OverlapCache(ψ, ψ, lefts, rights)
end

"rank-2 恒等通道固定点（环境核与 mult.jl 的 [`mixed_fixedpoints`](@ref) 统一——
左右不动点由 :LM 主本征对经 [`fixedpoint`](@ref) 解出（`alg` 分派 `tol`/`maxiter`：
NamedTuple / DynamicTol / KrylovKit 算法皆可），复环境按解的实际 eltype 存放；
本函数是三元通道的无算符分支，环境为 rank-2 矩阵）。"
function overlap_fixedpoints(below::CanonicalIMPS, above::CanonicalIMPS,
                             alg = Defaults.alg_environments();
                             GL0::Union{Nothing,AbstractArray} = nothing,
                             GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(below)
    T = promote_type(scalartype(below), scalartype(above))
    # 键 profile 逐站可变：环境定义在周期闭合的键 N 上（below/above 同长）
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Da)
        for ℓ in 1:N
            GL = push_env_left(GL, above.AL[ℓ], below.AL[ℓ])
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Da) : vec(copy(GL0))
    _, vL = fixedpoint(Tleft, v0L, :LM, alg)
    # 复环境提升（MPSKit 对齐：环境按 eigsolve 返回的实际 eltype 存放）
    TCL = promote_type(T, eltype(vL))
    GLs = Vector{Matrix{TCL}}(undef, N)
    GLs[1] = GL = reshape(vL, Dl, Da)
    for ℓ in 2:N
        GLs[ℓ] = GL = push_env_left(GL, above.AL[ℓ-1], below.AL[ℓ-1])
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dl)
        for ℓ in N:-1:1
            GR = push_env_right(GR, above.AR[ℓ], below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da * Dl) : vec(copy(GR0))
    _, vR = fixedpoint(Tright, v0R, :LM, alg)
    TCR = promote_type(T, eltype(vR))
    GRs = Vector{Matrix{TCR}}(undef, N)
    GRs[N] = GR = reshape(vR, Da, Dl)
    for ℓ in N-1:-1:1
        GRs[ℓ] = GR = push_env_right(GR, above.AR[ℓ+1], below.AR[ℓ+1])
    end

    # 归一化（MPSKit：GR Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）
    for ℓ in 1:N
        GRs[ℓ] ./= norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC(GLs[inext], GRs[ℓ], above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("overlap environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] ./= λ
    end
    return GLs, GRs
end

"Ternary overlap channel: left/right fixed points of ⟨below|above⟩（
[`overlap_fixedpoints`](@ref) 的 `alg` 分派 `tol`/`maxiter`）。"
function OverlapCache(below::CanonicalIMPS, above::CanonicalIMPS,
                      alg = Defaults.alg_environments();
                      GL0::Union{Nothing,AbstractArray} = nothing,
                      GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = overlap_fixedpoints(below, above, alg; GL0, GR0)
    return OverlapCache(below, above, GLs, GRs)
end

# ---- 纯重叠通道的增量环境推进（IDMRG 扫掠用；rank-2 参数序 (above, below)）----

function transfer_leftenv!(envs::OverlapCache, x::CanonicalIMPS,
                           ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], ket.AL[ℓm], x.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::OverlapCache, x::CanonicalIMPS,
                            ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ket.AR[ℓp], x.AR[ℓp])
    return envs
end

"纯重叠通道的环境重标定（MPSKit `normalize!(envs, below, above)` 语义：GR
Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）。"
function _normalize_overlap_envs!(envs::OverlapCache, x::CanonicalIMPS,
                                  ket::CanonicalIMPS)
    N = length(ket)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("overlap idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"纯重叠通道的最大逐站 Galerkin 残差（语义同 mult.jl 的
`_galerkin_err(operator, ket, x, envs)`，无算符插入）。"
function _galerkin_err(ket::CanonicalIMPS, x::CanonicalIMPS, envs::OverlapCache)
    N = length(ket)
    ϵ = 0.0
    for ℓ in 1:N
        ACmap = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], ACmap))
    end
    return ϵ
end

"""
    _overlap_vomps_sweeps(ket, x0, alg::VOMPS) -> (x, envs, info)

Overlap-maximizing VOMPS sweeps on the pure overlap channel (the operator-free
branch of MPSKit's `approximate(ψ₀, ϕ, VOMPS())`): find `x` approximating the
target chain `ket` itself — variational compression. Jacobi-style rounds, the
same `IterativeSolver` pipeline as the mult channel ([`_vomps_sweeps`](@ref)):
`localupdate`（`AC_new = GL·ket.AC·GR`、`C_new = GL₊·ket.C·GR` → `regauge!`）→
`gauge_step!` → warm-started environment re-solve → Galerkin residual checked
after the sweep. Returns `(x, envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = [初始残差,
逐轮 Galerkin 残差...]、`converged` 收敛标志）。
"""
function _overlap_vomps_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS, alg::VOMPS)
    N = length(ket)
    x = copy(x0)
    envs = OverlapCache(x, ket, alg.alg_environments)
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = _galerkin_err(ket, x, envs)
    iter = 0
    losses = [ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # 候选 AL 与 ket.AC 同形，eltype 提升到通道标量类型 T）
        ALs = [similar(ket.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            AC_new = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), ket.AC[ℓ])
            C_new = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            ALs[ℓ] = regauge!(AC_new, C_new; alg = alg.alg_orth)
        end
        # gauge: restore the global right gauge（动态容差）
        alg_g = updatetol(alg.alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（热启动 + 动态环境容差）
        alg_envs = updatetol(alg.alg_environments, iter - 1, ϵ)
        envs = OverlapCache(x, ket, alg_envs; GL0 = envs.lefts[1], GR0 = envs.rights[N])
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket, envs)
        ϵ = _galerkin_err(ket, x, envs)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    _global_normalize!(x)
    return x, envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    _overlap_idmrg_sweeps(ket, x0, alg::IDMRG) -> (x, envs, info)

IDMRG template on the pure overlap channel (the operator-free branch of
MPSKit's `approximate(ψ₀, ϕ, IDMRG())`): sequential Gauss–Seidel double sweep
with on-the-fly environment transfer, `leftorth`/`rightorth` splits of the
normalized local projections, per-double-sweep environment rescaling
([`_normalize_overlap_envs!`](@ref)) and center-matrix-drift convergence
`ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is rebuilt from
the `AR` string and the environments are re-solved for the final state.
Returns `(x, envs, info)`，`info` 为 [`IterativeConvergenceInfo`](@ref)
（`niter` = 扫掠轮数、`losses` = 逐轮中心矩阵漂移、`converged` 收敛标志）。
"""
function _overlap_idmrg_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS, alg::IDMRG)
    N = length(ket)
    x = copy(x0)
    # 初始环境：由初态解一次左右不动点，扫掠中只做增量 transfer 与重标定
    envs = OverlapCache(x, ket, alg.alg_environments)
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            transfer_leftenv!(envs, x, ket, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            transfer_rightenv!(envs, x, ket, ℓ - 1)
        end
        # 环境重标定
        _normalize_overlap_envs!(envs, x, ket)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket, envs)
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建混合规范（容差取 alg_gauge 的动态适配），环境对终态重解
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = CanonicalIMPS([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol,
                      maxiter = alg_g.maxiter)
    envs = OverlapCache(x, ket, alg.alg_environments)
    _global_normalize!(x)
    return x, envs, IterativeConvergenceInfo(iter, losses, converged)
end
