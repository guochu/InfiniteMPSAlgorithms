# ---------------- iterative MPO multiplication mult (variational application / compression of MPO·MPS and MPO·MPO) ----------------
#
# Goal: given (W, x), find y ≈ W·x (x an MPS: operator application; x an MPO:
# operator composition). The strict (compression-free) constructions are the
# typed operators `DenseIMPO * DenseIMPO` / `DenseIMPO * DenseIMPS` below;
# this file provides the iterative (variational) versions:
# - `VOMPS`: overlap-maximizing ALS sweeps (strictly mirrors MPSKit VOMPS,
#   src/algorithms/approximate/vomps.jl);
# - `IDMRG`: sequential Gauss–Seidel sweeps with on-the-fly environment
#   transfer and C-drift convergence (strictly mirrors MPSKit's approximate
#   IDMRG, src/algorithms/approximate/idmrg.jl; converges to the same fixed
#   point as VOMPS).
#
# No engine computes or returns an overlap: convergence is judged by the
# Galerkin residual alone (MPSKit `approximate` contract), and the interfaces
# return the optimized chain only.
#
# VOMPS template (mirroring MPSKit):
# 1. ternary fixed-point environments `MultCache(x, operator, ket)`
#    (below = bra, above = ket);
# 2. localupdate: local maps (no eigen solves, unlike the groundstate VUMPS)
#    `AC_new = AC_hamiltonian(ℓ)·ket.AC[ℓ]`, `C_new = C_hamiltonian(ℓ)·ket.C[ℓ]`
#    → `regauge!(AC_new, C_new)` yields candidate `AL`s;
# 3. gauge: `gaugefix!(; order = :R)` restores the right gauge;
# 4. convergence: `calc_galerkin` (tangent-space Galerkin residual).

# ---------------- environment cache of the MPO-application channel (MultCache) ----------------

"""
    MultCache(operator, bra, ket, lefts, rights)
    MultCache(below, operator::AbstractInfiniteMPO, above, [alg]) -> MultCache
    MultCache(below, W1::AbstractInfiniteMPO, W2::AbstractInfiniteMPO, [alg]) -> MultCache

Environments of the MPO-application channel: the left/right fixed points of
`⟨below|operator|above⟩`, used by `mult` (iterative MPO multiplication).
Two channels share the same cache format (rank-3 environment tensors):

- MPO application (`operator`, `above::AbstractInfiniteMPS`): the
  `⟨below|W|ψ⟩` channel; `lefts[ℓ]` = `(below bond, w, above bond)`,
  `rights[ℓ]` = `(above bond, w, below bond)` (MPSKit convention);
- MPO composition (`W1`, `W2::AbstractInfiniteMPO`; the operator slot
  holds W1 and the ket slot holds W2): the `⟨below|W1·W2⟩` channel with the
  product's fused tensors never materialized; the same leg layout reads
  `lefts[ℓ]` = `(below bond, wl1, wl2)`, `rights[ℓ]` = `(wr2, wr1, below bond)`
  (the two factor bond legs kept separate).

`operator`/`above`/`W1`/`W2` 槽接受 `DenseIMPO`/`CanonicalIMPO`/
`DenseIMPS`/`CanonicalIMPS`（[`AbstractInfiniteMPO`](@ref)/
[`AbstractInfiniteMPS`](@ref)——dense 类型经 `getproperty` 的 `AL`/`AR`/`AC`
周期视图与 `C` 单位矩阵视图直接参与，存储不做任何规范转换）。

Both channels obtain the fixed points from the :LM eigenpairs of the fused
transfer matrix (`mixed_fixedpoints`, shared by the two channels); normalization mirrors MPSKit's
`normalize!(::InfiniteEnvironments)`: each GR is Frobenius-normalized first,
then per site `λℓ = ⟨below.C[ℓ], C_map(ℓ)⟩` scales `GLs[ℓ+1]`, so that the
local contraction of every site is exactly 1 (identity-MPO expectation = N).
"""
struct MultCache{O<:AbstractInfiniteMPO,
                 B<:Union{CanonicalIMPS,CanonicalIMPO},
                 K<:Union{AbstractInfiniteMPS,AbstractInfiniteMPO},T} <: CompressionEnvironments
    operator::O
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

# ---- 三元通道固定点核（MultCache 构造器专用；mpo·mps / mpo·mpo 统一） ----

"Shared fixed-point solver of the ternary environments——mpo·mps 施加通道
`⟨below|operator|above⟩ = ⟨ψ|W|ψ⟩` 与 mpo·mpo 组合通道 `⟨below|W1·W2⟩`（operator
槽 = W1、above 槽 = W2，below 为 CanonicalIMPO）共用，环境 rank-3 且布局统一：
`lefts = (below 键, w 键, above 键)`、`rights = (above 键, w 键, below 键)`
（MPSKit 约定；mpo·mpo 通道即 `(bl, wl1, wl2)`/`(wr2, wr1, bl)`，两因子键腿分开
存放）。left/right dominant eigenvectors + MPSKit-style normalization。
`alg`（如 `Defaults.alg_environments()` 或动态容差适配后的副本，
DynamicTol/NamedTuple 皆可——`fixedpoint` 直接分派）提供环境的 `tol`/`maxiter`；
`krylovdim` 取 `Defaults.krylovdim`。`GL0`/`GR0` optionally warm start
the eigsolves with the previous environments: for block-degenerate targets the
fixed-point space is multi-dimensional and a continuous initial guess keeps the
ALS iteration stable."
function mixed_fixedpoints(below::Union{CanonicalIMPS,CanonicalIMPO},
                              operator::AbstractInfiniteMPO,
                              above::Union{AbstractInfiniteMPS,AbstractInfiniteMPO},
                              alg = Defaults.alg_environments();
                              GL0::Union{Nothing,AbstractArray} = nothing,
                              GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(below)
    L = length(operator)
    (N % L == 0 && length(above) == N) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    T = promote_type(scalartype(below), scalartype(operator), scalartype(above))
    Dw = size(operator.AL[1], 1)
    # 键 profile 逐站可变：环境张量一律定义在键 N 上（周期闭合处），
    # Dl/Da = below/above 链在键 N 上的键维，Dr = 同一键上 below 的键维。
    # 非均匀键下 size(below.AR[1],3) 是键 1 的键维，不能混用。
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)
    Dr = Dl
    Wopl = ℓ -> operator.AL[_mod1(ℓ, L)]   # 左推用 AL 家族（与 below/above.AL 一致）
    Wopr = ℓ -> operator.AR[_mod1(ℓ, L)]   # 右推用 AR 家族（与 below/above.AR 一致）

    # ---- left fixed point: dominant eigenvector of T_L(above.AL, operator, below.AL) ----
    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Dw, Da)
        for ℓ in 1:N
            GL = push_env_left(GL, below.AL[ℓ], Wopl(ℓ), above.AL[ℓ])
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Dw * Da) : vec(copy(GL0))
    _, vL = fixedpoint(Tleft, v0L, :LM, alg)
    # 复环境提升（MPSKit 对齐：环境张量按 eigsolve 返回的实际 eltype 存放；
    # 实输入下融合转移的 leading vector 可为复，通道随后整体升为复算术）
    TCL = promote_type(T, scalartype(vL))
    GLs = Vector{Array{TCL,3}}(undef, N)
    GLs[1] = reshape(vL, Dl, Dw, Da)
    for ℓ in 2:N
        GLs[ℓ] = push_env_left(GLs[ℓ-1], below.AL[ℓ-1], Wopl(ℓ - 1), above.AL[ℓ-1])
    end

    # ---- right fixed point: dominant eigenvector of T_R(above.AR, operator, below.AR) ----
    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dw, Dr)
        for ℓ in N:-1:1
            GR = push_env_right(GR, above.AR[ℓ], Wopr(ℓ), below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da * Dw * Dr) : vec(copy(GR0))
    _, vR = fixedpoint(Tright, v0R, :LM, alg)
    TCR = promote_type(T, scalartype(vR))
    GRs = Vector{Array{TCR,3}}(undef, N)
    GRs[N] = reshape(vR, Da, Dw, Dr)
    for ℓ in N-1:-1:1
        GRs[ℓ] = push_env_right(GRs[ℓ+1], above.AR[ℓ+1], Wopr(ℓ + 1), below.AR[ℓ+1])
    end

    # ---- normalization (mirroring MPSKit: GR Frobenius-normalized, GL scaled
    #      by the local overlap λ) ----
    for ℓ in 1:N
        GRs[ℓ] .= GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        GLn = GLs[inext]
        GR = GRs[ℓ]
        Cnew = _mapC(GLn, GR, above.C[ℓ], operator.C[_mod1(ℓ, L)])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("ternary environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] .= GLn ./ λ
    end
    return GLs, GRs
end

function MultCache(below::CanonicalIMPS, operator::AbstractInfiniteMPO,
                   above::AbstractInfiniteMPS, alg = Defaults.alg_environments();
                   GL0::Union{Nothing,AbstractArray} = nothing,
                   GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = mixed_fixedpoints(below, operator, above, alg; GL0, GR0)
    # 槽位提升到通道标量类型（环境 eltype）：[`compression_sweeps!`] 对缓存
    # bra 的原地演化恒在同型算术上进行（dense/canonical 槽各自原类型提升）
    T = scalartype(GLs[1])
    return MultCache(_promote_scalar(T, operator), _promote_scalar(T, below),
                     _promote_scalar(T, above), GLs, GRs)
end

"双 MPO 组合通道（mpo·mpo）：bra 槽 = 变分链（CanonicalIMPO，原生 MPO 形态）、
operator 槽 = W1、ket 槽 = W2，环境 rank-3
`(below, wl1, wl2)`/`(wr2, wr1, below)`（两个因子的键腿分开存放）。"
function MultCache(below::CanonicalIMPO, W1::AbstractInfiniteMPO, W2::AbstractInfiniteMPO,
                   alg = Defaults.alg_environments();
                   GL0::Union{Nothing,AbstractArray} = nothing,
                   GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = mixed_fixedpoints(below, W1, W2, alg; GL0, GR0)
    # 槽位提升到通道标量类型（同上）
    T = scalartype(GLs[1])
    return MultCache(_promote_scalar(T, W1), _promote_scalar(T, below),
                     _promote_scalar(T, W2), GLs, GRs)
end

"""
    recalculate!(envs::MultCache, newbra, [alg_environments]) -> envs

Recompute the `⟨bra|operator|ket⟩` fixed points for the updated bra `newbra`
（operator/ket 不变，[`mixed_fixedpoints`](@ref) 重解），边界 eigsolve 以当前
存储的 `lefts[1]`/`rights[end]` 热启动（对标 MPSKit 的原地 `recalculate!`）。
`newbra` 与新不动点就地写回 `envs`（原地更新，返回 `envs` 本身）。
"""
function recalculate!(envs::MultCache,
                      newbra::Union{CanonicalIMPS,CanonicalIMPO},
                      alg_environments = Defaults.alg_environments())
    GLs, GRs = mixed_fixedpoints(newbra, envs.operator, envs.ket,
                                 alg_environments;
                                 GL0 = envs.lefts[1], GR0 = envs.rights[end])
    copy!(envs.lefts, GLs)
    copy!(envs.rights, GRs)
    envs.bra ≡ newbra || _copyinto!(envs.bra, newbra)
    return envs
end

# 纯重叠通道（OverlapCache）与 compress 的 VOMPS/IDMRG 引擎见 compress.jl
# 严格乘法的 kernel（fuse / _naive_mul_tensor）见 operators/linalg.jl

"VOMPS local AC map（mpo·mps 施加通道；**统一 4 参约定 (GL, GR, O, ketAC)**，
两通道靠 `ketAC` 的秩分派——rank-3 = mpo·mps、rank-4 = mpo·mpo 的 W2 中心张量）：
`AC_new = GL·O·ketAC·GR`（O 为 operator 的原始张量 `.AC`，各参量允许不同标量
类型，自动提升）。"
function _mapAC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
                O::AbstractArray{To,4}, ketAC::AbstractArray{Tk,3}) where {Tg,To,Tgr,Tk}
    @tensor ACnew[aL, u, aR] := GL[aL, w, bL] * ketAC[bL, s, bR] * O[w, u, w′, s] * GR[bR, w′, aR]
    return ACnew
end

# 局部 C map 见下方 pair 通道的统一 4 参 `_mapC`（两通道同一收缩）

# ---------------- 正交分解 / 重建的 3、4 维重载（MPS 视图语义统一） ----------------

"`_leftsplit(AC, alg) -> (AL, C)`：MPS 视图 `(wl, u·d, wr)` 的左正交分解
（rank-3 MPS 中心张量 / rank-4 CanonicalIMPO 中心张量，按秩分派；`alg` 为
因式化算法，矩阵形态调用 `leftorth`——分腿形式仅接受 LQ 族，与 `regauge!`
同款）。`_rightsplit` 用 `alg'`（对偶因式化）。"
function _leftsplit(AC::AbstractArray{<:Any,3}, alg::FiniteMPSAlgorithms.OrthogonalFactorizationAlgorithm)
    wl, u, wr = size(AC)
    Q, C = leftorth(reshape(AC, wl * u, wr); alg = alg)
    return reshape(Q, wl, u, :), C
end
function _leftsplit(AC::AbstractArray{<:Any,4}, alg = Defaults.alg_orth())
    wl, u, wr, d = size(AC)
    ALv = _as_mps_view(AC)                                   # (wl, u·d, wr)
    Q, C = leftorth(reshape(ALv, wl * u * d, wr); alg = alg)
    AL4 = permutedims(reshape(Q, wl, u, d, :), (1, 2, 4, 3))   # (wl, u, d, r) → (wl, u, r, d)
    return AL4, C
end

"`_rightsplit(AC, alg) -> (C, AR)`：MPS 视图 `(wl, u·d, wr)` 的右正交分解。"
function _rightsplit(AC::AbstractArray{<:Any,3}, alg::FiniteMPSAlgorithms.OrthogonalFactorizationAlgorithm)
    wl, u, wr = size(AC)
    C, Q = rightorth(reshape(AC, wl, u * wr); alg = alg')
    return C, reshape(Q, :, u, wr)
end
function _rightsplit(AC::AbstractArray{<:Any,4}, alg = Defaults.alg_orth())
    wl, u, wr, d = size(AC)
    ACv = _as_mps_view(AC)                                   # (wl, u·d, wr)
    C, Q = rightorth(reshape(ACv, wl, u * d * wr); alg = alg')
    AR4 = permutedims(reshape(Q, :, u, d, wr), (1, 2, 4, 3))   # (r, u, d, wr) → (r, u, wr, d)
    return C, AR4
end

"`_rebuild(ARs; kwargs)`: 从 AR 串重建混合规范链（rank-3 → CanonicalIMPS、
rank-4 → CanonicalIMPO）。"
_rebuild(ARs::Vector{<:AbstractArray{<:Any,3}}; kwargs...) = CanonicalIMPS(ARs; kwargs...)
_rebuild(ARs::Vector{<:AbstractArray{<:Any,4}}; kwargs...) = CanonicalIMPO(ARs; kwargs...)

"Uniform norm normalization (AC and C are scaled together, preserving the
consistency of `AC = AL·C` and `AR = C[ℓ-1]⁻¹·AC`)."
function _global_normalize!(x::CanonicalIMPS)
    n = norm(x)
    n == 0 && error("mult: zero-norm state")
    for ℓ in 1:length(x)
        x.AC[ℓ] .= x.AC[ℓ] ./ n
        x.C[ℓ] .= x.C[ℓ] ./ n
    end
    return x
end

"通道标量类型提升（MPSKit 对齐）：实输入下融合转移的 leading vector 可为复，
环境按 eigsolve 的实际 eltype 存放（复）；随后的 ALS 扫掠在复算术上进行——
将演动态 `x` 提升到通道标量类型 `T`（已是 `T` 则原样返回）。"
function _promote_scalar(::Type{T}, ψ::CanonicalIMPS) where {T}
    scalartype(ψ) == T && return ψ
    cast = As -> PeriodicVector([T.(a) for a in As])
    return CanonicalIMPS(cast(ψ.AL), cast(ψ.C), cast(ψ.AR), cast(ψ.AC))
end

"VOMPS Galerkin residual (mirrors MPSKit's `calc_galerkin`): the norm of the
component of `normalize(AC_map)` orthogonal to the `AL` tangent space — the
overlap is insensitive to tangential drift, so convergence must be judged by
the residual rather than Δoverlap."
function _galerkin(AL::AbstractArray{Ta,3}, ACnew::AbstractArray{Tb,3}) where {Ta,Tb}
    ACn = normalize!(copy(ACnew))
    @tensor proj[b, b′] := conj(AL[a, s, b]) * ACn[a, s, b′]
    @tensor out[a, s, b′] := ACn[a, s, b′] - AL[a, s, b] * proj[b, b′]
    return norm(out)
end

"压缩通道的统一局部映射入口（[`compression_sweeps!`](@ref) 用；全部输入由缓存
持有，mpo·mps / mpo·mpo 两通道靠 ket 槽的秩分派统一）：`_local_AC(envs, ℓ)` 为
site ℓ 的 AC 局部投影、`_local_C(envs, ℓ)` 为 bond ℓ 的 C 局部投影。"
function _local_AC(envs::MultCache, ℓ::Int)
    return _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ),
                  envs.operator.AC[_mod1(ℓ, length(envs.operator))], envs.ket.AC[ℓ])
end

function _local_C(envs::MultCache, ℓ::Int)
    return _mapC(leftenv(envs, _mod1(ℓ + 1, length(envs))), rightenv(envs, ℓ),
                 envs.ket.C[ℓ], envs.operator.C[_mod1(ℓ, length(envs.operator))])
end

"扫掠 finalize 回调的通道目标（[`compression_sweeps!`](@ref) 用）：mult 通道 =
operator。"
_finalize_target(envs::MultCache) = envs.operator

"Maximum per-site Galerkin residual (mirrors MPSKit's
`calc_galerkin(below, operator, above, envs)`，语义同 `_galerkin(AL, AC_map)`：
local-map 输出在当前态 `x.AL` 切空间上的正交分量范数)。mpo·mps / mpo·mpo
两通道由 `_local_AC` 的秩分派统一。"
function calc_galerkin(envs::MultCache, x::Union{CanonicalIMPS,CanonicalIMPO})
    N = length(envs)
    ϵ = 0.0
    for ℓ in 1:N
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], _local_AC(envs, ℓ)))
    end
    return ϵ
end

# Ternary-channel incremental environment pushes for the IDMRG sweep (mirroring
# MPSKit's `transfer_leftenv!`/`transfer_rightenv!` for the
# `⟨below|operator|above⟩` channel: the below side is the state being optimized,
# the above side the target chain；mpo·mps / mpo·mpo 两通道由 `push_env_*` 的
# 秩分派统一).
function transfer_leftenv!(envs::MultCache,
                           x::Union{CanonicalIMPS,CanonicalIMPO}, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], x.AL[ℓm],
                                  envs.operator.AL[_mod1(ℓm, length(envs.operator))],
                                  envs.ket.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::MultCache,
                            x::Union{CanonicalIMPS,CanonicalIMPO}, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], envs.ket.AR[ℓp],
                                    envs.operator.AR[_mod1(ℓp, length(envs.operator))],
                                    x.AR[ℓp])
    return envs
end

"Ternary-channel environment rescaling during the sweep (mirrors MPSKit's
`normalize!(envs, below, operator, above)`): unit-Frobenius `GR`s; `GL[ℓ+1]`
scaled by `inv(λ)` with the local C-channel overlap
`λ = ⟨x.C[ℓ], C_map(ℓ)⟩`（C 通道恢复算符的 C 权重，与 envs 的规范家族约定
配套）。"
function normalize_envs!(envs::MultCache, x::Union{CanonicalIMPS,CanonicalIMPO})
    N = length(envs)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _local_C(envs, ℓ)
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

# ---------------- MPO·MPO composition channel (lazy fused-pair kernels) ----------------
#
# mult(W1, W2, alg) 的惰性 target：乘积算符 (W1·W2) 的 MPS 视图张量
# Ket[(wl2·wl1), (u·d), (wr2·wr1)] = Σ_m W1[wl1,u,wr1,m]·W2[wl2,m,wr2,d]。
# 乘积张量从不物化：环境收缩与局部投影直接消费 (W1, W2) 张量对（MultCache 的
# operator/ket 槽各存一个 CanonicalIMPO 因子），按「键指标优先、物理指标最后」
# 显式分步 GEMM，最大中间张量只有 O(D·D₁·D₂·d²)。
#
# 乘积因子天然保持规范一致性（W1、W2 各自混合规范 ⇒ 乘积 AL 左正交、AR 右正
# 交、AC·C 一致，C 的外积 kron 布局逐位对齐融合键序：wl1/wr1 为快指标），因此
# identity 通道固定点机制无需 gauge twist 直接适用。
#
# 与 mpo·mps 通道的分工：环境同为 rank-3 `Array{T,3}`，布局与 MPSKit 约定统一
# （lefts `(below, w, above)`、rights `(above, w, below)`）——本通道的两个因子键腿
# 分开存放 `(below, wl1, wl2)`/`(wr2, wr1, below)`；局部投影 `_mapAC` 的 pair
# 方法返回 rank-4 `(below, u, d, above)`（物理腿 (u, d) 不融合，调用方需要 MPS
# 视图时自行 reshape 成 (below, u·d, above)，u 快）。

"Pair-channel left push（rank-3 环境，bra 为 CanonicalIMPO rank-4 张量；重载
`push_env_left` 的 `(L, below, W, above)` 调用约定，与 mpo·mps 通道共用
[`mixed_fixedpoints`](@ref)）：
`L′[bl′, wr1, wr2] = Σ L[bl, wl1, wl2]·conj(below[bl, u, bl′, d])·W1[wl1, u, wr1, m]·W2[wl2, m, wr2, d]`。
显式三步二元收缩（键优先：wl1 → (wl2, m)，bra 物理 u/d 各随 W1/W2 收缩），
融合张量不落地。"
function push_env_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,4},
                       W1::AbstractArray{Tw1,4}, W2::AbstractArray{Tw2,4}) where {TL,Tb,Tw1,Tw2}
    wl1, _, _, _ = size(W1)
    wl2, _, _, _ = size(W2)
    # 步1（键 wl1；bra 物理 u 一并收缩）：Y[(wr1, m), wl2, u, bl]
    Y = @tensor Y[c1, mm, a2, u, jL] := L[jL, a1, a2] * W1[a1, u, c1, mm]
    # 步2（键 wl2 与桥 m、bra 物理 d 一并收缩）：T[(wr1, u, bl), (wr2, d)]
    T = @tensor T[c1, u, jL, c2, dd] := Y[c1, mm, a2, u, jL] * W2[a2, mm, c2, dd]
    # 步3（bra 剩余腿 (u, bl, d) 收缩）：输出 (bl′, wr1, wr2)
    @tensor L′[jR, c1, c2] := T[c1, u, jL, c2, dd] * conj(below[jL, u, jR, dd])
end

"Pair-channel right push（rank-3 环境，bra 为 CanonicalIMPO rank-4 张量；重载
`push_env_right` 的 `(R, above, W, below)` 调用约定，与 mpo·mps 通道共用
[`mixed_fixedpoints`](@ref)；环境布局统一为 `(above 键, w 键, below 键)`，
输入腿 = 站点右键 `(wr2, wr1, br)`、输出 = 站点左键 `(wl2, wl1, bl′)`）：
`R′[wl2, wl1, bl′] = Σ R[wr2, wr1, br]·W2[wl2, m, wr2, d]·W1[wl1, u, wr1, m]·conj(below[bl′, u, br, d])`。
显式三步二元收缩（键优先：wr2 → (wl2, m)，bra 物理 u/d 各随 W2/W1 收缩），
融合张量不落地。"
function push_env_right(R::AbstractArray{TR,3}, above::AbstractArray{Ta,4},
                        W1::AbstractArray{Tw1,4}, below::AbstractArray{Tb,4}) where {TR,Ta,Tw1,Tb}
    # 步1（键 wr2；ket 物理 d 一并收缩）：Y[(wl2, m, wr1, d, br)]
    Y = @tensor Y[a2, mm, c1, dd, jR] := R[c2, c1, jR] * above[a2, mm, c2, dd]
    # 步2（bra 物理 d、键 br 收缩）：Z[(wl2, m, wr1, u, bl′)]
    Z = @tensor Z[a2, mm, c1, u, jL] := Y[a2, mm, c1, dd, jR] * conj(below[jL, u, jR, dd])
    # 步3（键 wr1 与桥 m、bra 物理 u 一并收缩）：输出 (wl2, wl1, bl′)
    @tensor R′[a2, a1, jL] := Z[a2, mm, c1, u, jL] * W1[a1, u, c1, mm]
end

"Pair-channel local AC projection (identity channel, rank-3 环境；GR 布局
`(above 键, w 键, below 键)` = `(wl2, wr1, br)`，与 mpo·mps 通道统一)：
`AC4[bl, u, br, d] = Σ GL[bl, wl1, wl2]·W1[wl1, u, wr1, m]·W2[wl2, m, wr2, d]·GR[wr2, wr1, br]`。
显式三步键优先 GEMM；**直接返回 rank-4**（CanonicalIMPO 张量约定 `(wl, u, wr, d)`，
物理腿 (u, d) 不融合），调用方需要 MPS 视图时自行 permute+reshape。"
function _mapAC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
                W1::AbstractArray{Tw1,4}, W2::AbstractArray{Tw2,4}) where {Tg,Tgr,Tw1,Tw2}
    Y = @tensor Y[jL, w2, u, mm, r1] := GL[jL, w1, w2] * W1[w1, u, r1, mm]
    Z = @tensor Z[jL, u, dd, r1, r2] := Y[jL, w2, u, mm, r1] * W2[w2, mm, r2, dd]
    @tensor AC4[jL, u, jR, dd] := Z[jL, u, dd, r1, r2] * GR[r2, r1, jR]
end

"VOMPS local C projection——**两通道统一的 4 参约定 (GL, GR, C_ab, C_op)**：C
问题恢复融合键上的权重 = above 链的 C（`C_ab`：mpo·mps 为 ψ.C、mpo·mpo 为
W2.C）× operator 的 C（`C_op`：mpo·mps 为 W.C、mpo·mpo 为 W1.C）——
`Cnew[bl, br] = Σ GL[bl, w_op, w_ab]·C_ab[w_ab, r_ab]·C_op[w_op, r_op]·GR[r_ab, r_op, br]`
（GL 腿序 `(below, w_op, above)`、GR `(above, w_op, below)`，两通道一致）。"
function _mapC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
               C_ab::AbstractMatrix, C_op::AbstractMatrix) where {Tg,Tgr}
    Y = @tensor Y[jL, w1, r2] := GL[jL, w1, w2] * C_ab[w2, r2]
    Z = @tensor Z[jL, r1, r2] := Y[jL, w1, r2] * C_op[w1, r1]
    @tensor Cnew[jL, jR] := Z[jL, r1, r2] * GR[r2, r1, jR]
end

# ---------------- CanonicalIMPO-bra variants of the shared sweep helpers ----------------

"`_mpo_view(A)`: CanonicalIMPO 张量 `(wl, u, wr, d)` 的 MPS 视图 `(wl, u·d, wr)`。"
_as_mps_view(A::AbstractArray{T,4}) where {T} =
    reshape(permutedims(A, (1, 2, 4, 3)), size(A, 1), size(A, 2) * size(A, 4), size(A, 3))

function _galerkin(AL::AbstractArray{Ta,4}, ACnew::AbstractArray{Tb,4}) where {Ta,Tb}
    return _galerkin(_as_mps_view(AL), _as_mps_view(ACnew))
end

function _promote_scalar(::Type{T}, W::CanonicalIMPO) where {T}
    scalartype(W) == T && return W
    cast = As -> PeriodicVector([T.(a) for a in As])
    return CanonicalIMPO{T}(cast(W.AL), cast(W.C), cast(W.AR), cast(W.AC))
end

"`_promote_scalar` 的 DenseIMPO/DenseIMPS 版：按原类型逐张量提升（不转换规范
形态）。"
function _promote_scalar(::Type{T}, W::DenseIMPO) where {T}
    scalartype(W) == T && return W
    return DenseIMPO([T.(w) for w in W.Ws])
end

function _promote_scalar(::Type{T}, ψ::DenseIMPS) where {T}
    scalartype(ψ) == T && return ψ
    return DenseIMPS([T.(a) for a in ψ.As])
end

"Uniform norm normalization（CanonicalIMPO：AC 与 C 同除，保持 `AC = AL·C` 一致）。"
function _global_normalize!(W::CanonicalIMPO)
    n = norm(W)
    n == 0 && error("mult: zero-norm state")
    for ℓ in 1:length(W)
        W.AC[ℓ] .= W.AC[ℓ] ./ n
        W.C[ℓ] .= W.C[ℓ] ./ n
    end
    return W
end

"gauge_step! 的 CanonicalIMPO 版：候选 `AL` 串写入 `W.AL` 后
`gaugefix!(; order = :R)` 恢复全局右规范（`gaugefix!` 内部同步全部家族）。"
function gauge_step!(W::CanonicalIMPO, ALs::Vector, C₀; tol::Real, maxiter::Int)
    for ℓ in eachindex(ALs)
        W.AL[ℓ] = ALs[ℓ]
    end
    return gaugefix!(W, W.AL, C₀; order = :R, tol = tol, maxiter = maxiter)
end

# 双 MPO 组合通道（below::CanonicalIMPO，operator/ket 槽 = (W1, W2)）的环境
# 固定点与 mpo·mps 通道共用顶部的 [`mixed_fixedpoints`](@ref)：乘积因子各自
# 保持规范（W1、W2 各自混合规范 ⇒ 乘积 AL 左正交、AR 右正交、AC·C 一致），无需
# gauge twist。

"""
    mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPS, envs, info)
    mult(W, W2, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPO, envs, info)

The compute-on-the-fly version of the MPO multiplication: find `y ≈ W·ψ`
(operator application) or `y ≈ W·W2` (operator composition), variationally
compressed to the bond dimension `alg.D`. This method **never
materializes the naive target family**: for mpo·mps the local maps
`k = GL·W·ket·GR` are computed per site on the fly; for mpo·mpo the
factorized engine consumes the `(W1, W2)` tensor pairs directly with
bond-first contractions (the fused tensors are never formed). Intermediate
memory is O(single site) in both cases. Applying a time-evolution MPO is
`ψ′ = first(mult(timeevompo(H, -im * dt, WII()), ψ, alg))`.

- `W`/`ψ`/`W2`：`AbstractInfiniteMPO`/`AbstractInfiniteMPS`——`DenseIMPO`/
  `CanonicalIMPO` 与 `DenseIMPS`/`CanonicalIMPS` 皆可直接输入（dense 类型经
  `getproperty` 家族视图直接参与环境与局部映射，不做规范转换）；输出类型恒为
  `CanonicalIMPS`/`CanonicalIMPO`;
- `alg.D::Int`: target bond dimension of the variational compression
  (deterministic `svdguess_mult` initial state);
- both `alg` types share the same fixed point;
- the output is guaranteed to be in mixed-canonical form (mpo·mps →
  `CanonicalIMPS`, mpo·mpo → `CanonicalIMPO`) and normalized (幅值不携带
  信息，只有方向有意义);
- real-valued inputs whose fused transfer has complex leading eigenvalues
  are handled as in MPSKit (environments live on complex spaces there):
  the environments take the eigensolver's complex output and the channel
  continues in complex arithmetic, so the result may be complex-valued.

The second return is the engine's final environments（mpo·mps / mpo·mpo 通道的
[`MultCache`](@ref)，对返回态 `envs.bra` 的射线解出，envs 与其规范代表可能有
规范差但同射线）; the third return is the [`IterativeConvergenceInfo`](@ref)
（`niter` 扫掠轮数、`losses` 逐轮残差/漂移、`converged` 收敛标志）。
"""
function mult(W::AbstractInfiniteMPO, ψ::AbstractInfiniteMPS, alg::Union{VOMPS,IDMRG})
    (length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    # MPO-channel VOMPS/IDMRG (compute-on-the-fly, no naive target family)：
    # 初态 = [`svdguess_mult`](@ref)，扫掠经 [`compression_sweeps!`](@ref)
    envs = MultCache(svdguess_mult(W, ψ, alg.D), W, ψ, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    # 引擎收尾已保证 envs.bra 处于混合规范且按包约定归一化，直接返回其本体
    return envs.bra, envs, info
end

function mult(W::AbstractInfiniteMPO, W2::AbstractInfiniteMPO, alg::Union{VOMPS,IDMRG})
    (length(W2) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    # 双 MPO 组合通道：bra = 变分链（CanonicalIMPO，原生 MPO 形态）、
    # operator/ket 槽 = (W1, W2)——乘积张量从不物化（键优先分步 GEMM），
    # 因子按输入类型直接进入环境（dense 经 getproperty 家族视图）。
    envs = MultCache(svdguess_mult(W, W2, alg.D), W, W2,
                     alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return envs.bra, envs, info
end

# ---------------- svdguess_mult (deterministic initial guess) & mult! (in-place) ----------------

"""
    svdguess_mult(W, ψ, D) -> CanonicalIMPS
    svdguess_mult(W, W2, D) -> CanonicalIMPO
    svdguess_mult(Ws, ALs, D) -> Vector{Array{T,3}}          # bare-tensor entry
    svdguess_mult(Ws1, Ws2, D) -> Vector{Array{T,3}}         # bare-tensor entry

Deterministic initial guess of the iterative [`mult`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_mult`): the fusion/product's site tensors are
generated one at a time, **with the streaming carry absorbed during
construction** (contracting into the inputs before fusing — the naive product
tensor is never materialized), and streamed right→left through a truncating
right-orthogonalization with bond cap `D` ([`_lazy_svd_guess`](@ref)); the
ring's wrap bond is Schmidt-truncated at site 1. Every output bond is ≤ `D`.

The bare-tensor methods take the site-tensor strings directly (`PeriodicVector`
of rank-4 MPO tensors and rank-3 MPS tensors, e.g. `ψ.AL` / `ψ.AR` of a
[`CanonicalIMPS`](@ref)) and return the right-gauge tensor string — the
low-level entry point for downstream packages; the wrapper methods
re-canonicalize its output（`kwargs` 透传末端 `CanonicalIMPS` 构造器 →
`gaugefix!`）：mpo·mps 的乘积是态，返回 `CanonicalIMPS`；mpo·mpo 的乘积是
算符，流式构造在融合物理腿 (u·d) 的 rank-3 视图上进行，规范化后按
[`devectorize`](@ref) 的互逆纯 reshape 拆回 (u, d)，原生 MPO 形态返回
`CanonicalIMPO`（规范数据逐家族携带，混合正则性不变）。
"""
function svdguess_mult(W, ψ::AbstractInfiniteMPS, D::Int; kwargs...)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    return CanonicalIMPS(svdguess_mult(Wm.Ws, ψ.AL, D); kwargs...)
end

function svdguess_mult(Ws::PeriodicVector{<:Array{T,4}},
                       ALs::PeriodicVector{<:Array{T,3}}, D::Int) where {T}
    (length(ALs) % length(Ws) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    # carry 在构造时吸收（fuse 的融合序：左 (wl,bl)、右 (wr,br)）：
    # B'[(wl,bl), u, f] = Σ_{wr,br,d} W[wl,u,wr,d]·AL[bl,d,br]·carry[(wr,br), f]
    site = (ℓ, carry) -> begin
        W4 = Ws[ℓ]; A = ALs[ℓ]
        carry === nothing && return fuse(W4, A)
        wl, u, wr, dd = size(W4); bl, _, br = size(A)
        f = size(carry, 2)
        l4 = reshape(carry, wr, br, f)
        T4 = @tensor Tb[bl, dd, wr, ff] := A[bl, dd, br] * l4[wr, br, ff]
        B3 = @tensor B[wl, bl, u, ff] := W4[wl, u, wr, dd] * Tb[bl, dd, wr, ff]
        return reshape(B3, wl * bl, u, f)
    end
    return _lazy_svd_guess(site, length(ALs), D)
end

function svdguess_mult(W, W2::AbstractInfiniteMPO, D::Int; kwargs...)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    # 乘积是算符：融合 (u·d) 视图上规范化后拆回 (u, d)，原生 MPO 形态返回
    return devectorize(CanonicalIMPS(svdguess_mult(Wm.Ws, W2m.Ws, D); kwargs...))
end

function svdguess_mult(Ws1::PeriodicVector{<:Array{T,4}},
                       Ws2::PeriodicVector{<:Array{T,4}}, D::Int) where {T}
    (length(Ws2) % length(Ws1) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    # carry 在构造时吸收（_naive_mul_tensor 的融合序：左腿 (wl1 慢, wl2 快)、
    # 右腿 (wr1 慢, wr2 快)）：
    # B'[(wl1,wl2), u, d, f] =
    #     Σ_{m,wr1,wr2} W1[wl1,u,wr1,m]·W2[wl2,m,wr2,d]·carry[(wr1,wr2), f]
    # 显式矩阵乘（先吸收 carry 的 wr2 腿，再收缩 W1 的 m/wr1 腿），不落地大张量
    site = (ℓ, carry) -> begin
        W1 = Ws1[ℓ]; W2t = Ws2[ℓ]
        if carry === nothing
            W4 = _naive_mul_tensor(W1, W2t)
            wl, u, wr, dd = size(W4)
            return reshape(permutedims(W4, (1, 2, 4, 3)), wl, u * dd, wr)
        end
        wl1, u1, wr1, m = size(W1)
        wl2, _, wr2, dd = size(W2t)
        f = size(carry, 2)
        l4 = reshape(carry, wr1, wr2, f)                 # carry 行 = wr1 最快
        # Tb[wl2, m2, wr1, dd2, ff] = Σ_wr2 W2t[wl2, m2, wr2, dd2]·l4[wr1, wr2, ff]
        l4p = reshape(permutedims(l4, (2, 1, 3)), wr2, wr1 * f)
        W2p = reshape(permutedims(W2t, (1, 2, 4, 3)), wl2 * m * dd, wr2)
        T5 = reshape(W2p * l4p, wl2, m, dd, wr1, f)
        T5 = permutedims(T5, (1, 2, 4, 3, 5))            # (wl2, m, wr1, dd, f)
        # B5[wl1, wl2, u1, dd, ff] = Σ_{m,wr1} W1[wl1, u1, wr1, m]·T5[wl2, m, wr1, dd, ff]
        W1p = reshape(permutedims(W1, (1, 2, 4, 3)), wl1 * u1, m * wr1)
        T5p = reshape(permutedims(T5, (2, 3, 1, 4, 5)), m * wr1, wl2 * dd * f)
        B5 = W1p * T5p                                   # (wl1·u1, wl2·dd·f)
        B5 = reshape(permutedims(reshape(B5, wl1, u1, wl2, dd, f), (1, 3, 2, 4, 5)),
                     wl1 * wl2, u1 * dd, f)
        return B5
    end
    return _lazy_svd_guess(site, length(Ws2), D)
end

"""
    mult!(out, W, ψ, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)
    mult!(out, W, W2, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)

In-place [`mult`](@ref): `out` is the user-provided state/operator to be
optimized as the initial guess; its bond profile is first brought to uniform
`D = alg.D` with [`changebond!`](@ref). The optimized result is written back
into `out`——类型守卫：通道算术类型（构造时 bra 槽已提升到环境 eltype）宽于
`out` 的标量类型时（如实 `out` 遇复通道：复输入或 leading vector 复化），结果
无法原地表示，此时 `out` 不写回（仅 `changebond!` 预处理生效），第一个返回值
为提升后的更新 bra（`envs.bra`）本身。Returns `(out, envs, info)`——`envs` 为
引擎的最终环境、`info` 为 [`IterativeConvergenceInfo`](@ref)（首个返回值写回
后与 envs 的规范代表同射线）。
"""
function mult!(out::CanonicalIMPS, W::AbstractInfiniteMPO, ψ::AbstractInfiniteMPS,
               alg::Union{VOMPS,IDMRG})
    (length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    changebond!(out; D = alg.D)
    envs = MultCache(out, W, ψ, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return _copyinto!(out, envs.bra), envs, info
end

function mult!(out::CanonicalIMPO, W::AbstractInfiniteMPO, W2::AbstractInfiniteMPO,
               alg::Union{VOMPS,IDMRG})
    (length(W2) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    changebond!(out; D = alg.D)
    envs = MultCache(out, W, W2, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return _copyinto!(out, envs.bra), envs, info
end
