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
# 4. convergence: `_galerkin_err` (tangent-space Galerkin residual).

# ---------------- environment cache of the MPO-application channel (MultCache) ----------------

"""
    MultCache(operator, bra, ket, lefts, rights)
    MultCache(below, operator::DenseIMPO, above; tol, krylovdim, maxiter) -> MultCache
    MultCache(below, W1::CanonicalIMPO, W2::CanonicalIMPO; ...) -> MultCache

Environments of the MPO-application channel: the left/right fixed points of
`⟨below|operator|above⟩`, used by `mult` (iterative MPO multiplication).
Two channels share the same cache format (rank-3 environment tensors):

- MPO application (`operator::DenseIMPO`, `above::CanonicalIMPS`): the
  `⟨below|W|ψ⟩` channel; `lefts[ℓ]` = `(below bond, w, above bond)`,
  `rights[ℓ]` = `(above bond, w, below bond)` (MPSKit convention);
- MPO composition (`W1::CanonicalIMPO`, `W2::CanonicalIMPO`; the operator slot
  holds W1 and the ket slot holds W2): the `⟨below|W1·W2⟩` channel with the
  product's fused tensors never materialized; `lefts[ℓ]` =
  `(below bond, wl1, wl2)`, `rights[ℓ]` = `(wr1, wr2, below bond)` (the two
  factor bond legs kept separate).

Both channels obtain the fixed points from the :LM eigenpairs of the fused
transfer matrix (eigsolve); normalization mirrors MPSKit's
`normalize!(::InfiniteEnvironments)`: each GR is Frobenius-normalized first,
then per site `λℓ = ⟨below.C[ℓ], C_map(ℓ)⟩` scales `GLs[ℓ+1]`, so that the
local contraction of every site is exactly 1 (identity-MPO expectation = N).
"""
struct MultCache{O<:Union{DenseIMPO,CanonicalIMPO},
                 B<:Union{CanonicalIMPS,CanonicalIMPO},
                 K<:Union{CanonicalIMPS,CanonicalIMPO},T} <: Environments
    operator::O
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

function MultCache(below::CanonicalIMPS, operator::DenseIMPO,
                   above::CanonicalIMPS, alg = Defaults.alg_environments();
                   GL0::Union{Nothing,AbstractArray} = nothing,
                   GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = _ternary_fixedpoints(below, operator, above, alg; GL0, GR0)
    return MultCache(operator, below, above, GLs, GRs)
end

"双 MPO 组合通道（mpo·mpo）：bra 槽 = 变分链（CanonicalIMPO，原生 MPO 形态）、
operator 槽 = W1、ket 槽 = W2，环境 rank-3
`(below, wl1, wl2)`/`(wr1, wr2, below)`（两个因子的键腿分开存放）。"
function MultCache(below::CanonicalIMPO, W1::CanonicalIMPO, W2::CanonicalIMPO,
                   alg = Defaults.alg_environments();
                   GL0::Union{Nothing,AbstractArray} = nothing,
                   GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = _ternary_fixedpoints(below, W1, W2, alg; GL0, GR0)
    return MultCache(W1, below, W2, GLs, GRs)
end

# 纯重叠通道（OverlapCache）与 compress 的 VOMPS/IDMRG 引擎见 overlap.jl

"""
    fuse(W, AL) -> Array{T,3}

Per-site fusion of an MPO tensor with a left-orthogonal MPS tensor:
`B[(wl·bl), u, (wr·br)] = Σ_d W[wl, u, wr, d] · AL[bl, d, br]`.
"""
function fuse(W::AbstractArray{T,4}, AL::AbstractArray{T,3}) where {T}
    wl, u, wr, _ = size(W)
    bl, _, br = size(AL)
    @tensor B5[wl, bl, u, wr, br] := W[wl, u, wr, d] * AL[bl, d, br]
    return reshape(B5, wl * bl, u, wr * br)
end

"""
    _naive_mul_tensor(W1, W2) -> W12

Rank-4 bond fusion (MPO multiplication kernel, mirroring MPSKit's
`fuse_mul_mpo`): the middle physical index `m` = W1's d = W2's u, and the bond
dimension = product of the two bond dimensions (they need not be equal):

```julia
W12[(wl1, wl2), u, (wr1, wr2), d] = Σ_m W1[wl1, u, wr1, m] · W2[wl2, m, wr2, d]
```
"""
function _naive_mul_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    size(W1, 4) == size(W2, 2) ||
        throw(DimensionMismatch("MPO multiplication requires W1's physical in (d) to match W2's physical out (u)"))
    # this package's TensorOperations version does not support tuple composite
    # indices; use a flat intermediate tensor + reshape (as in fuse)
    @tensor W6[wl1, wl2, u, wr1, wr2, d] :=
        W1[wl1, u, wr1, m] * W2[wl2, m, wr2, d]
    return reshape(W6, size(W1, 1) * size(W2, 1), size(W1, 2),
                   size(W1, 3) * size(W2, 3), size(W2, 4))
end

"VOMPS local AC map (mirrors MPSKit `AC_hamiltonian·ket.AC`):
AC_new = (GL·O·GR)·ket.AC (各参量允许不同标量类型，自动提升)."
function _mapAC(GL::AbstractArray{Tg,3}, O::Union{Nothing,AbstractArray{To,4}},
                GR::AbstractArray{Tgr,3}, ketAC::AbstractArray{Tk,3}) where {Tg,To,Tgr,Tk}
    if O === nothing
        @tensor ACnew[aL, p, aR] := GL[aL, 1, bL] * ketAC[bL, p, bR] * GR[bR, 1, aR]
    else
        @tensor ACnew[aL, u, aR] := GL[aL, w, bL] * ketAC[bL, s, bR] * O[w, u, w′, s] * GR[bR, w′, aR]
    end
    return ACnew
end

"VOMPS local C map (mirrors MPSKit `C_hamiltonian·ket.C`): the channel
passes through with no W contraction."
function _mapC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
               ketC::AbstractMatrix{Tk}) where {Tg,Tgr,Tk}
    @tensor Cnew[a, a′] := GL[a, w, b] * ketC[b, b′] * GR[b′, w, a′]
    return Cnew
end

# ---------------- 统一局部映射入口（按 operator 槽类型分派 mpo·mps / mpo·mpo 核） ----------------

"`_mapAC(GL, GR, operator, ketAC, ℓ)`：统一局部 AC 投影入口——`operator::DenseIMPO`
（mpo·mps：单算符插入）与 `operator::CanonicalIMPO`（mpo·mpo：W1 因子，`ketAC`
为 W2 因子的 rank-4 中心张量）分派到各自 kernel。"
_mapAC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
       operator::DenseIMPO, ketAC::AbstractArray{Tk,3}, ℓ::Integer) where {Tg,Tgr,Tk} =
    _mapAC(GL, operator[_mod1(ℓ, length(operator))], GR, ketAC)
_mapAC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
       operator::CanonicalIMPO, ketAC::AbstractArray{Tk,4}, ℓ::Integer) where {Tg,Tgr,Tk} =
    _mapAC(GL, GR, operator.AC[_mod1(ℓ, length(operator))], ketAC)

"`_mapC(GL, GR, operator, ketC, ℓ)`：统一局部 C 投影入口（mpo·mps 的 C 通道无
算符收缩；mpo·mpo 需要两个因子的 C 外积）。"
_mapC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
      operator::DenseIMPO, ketC::AbstractMatrix{Tk}, ℓ::Integer) where {Tg,Tgr,Tk} =
    _mapC(GL, GR, ketC)
_mapC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
      operator::CanonicalIMPO, ketC::AbstractMatrix{Tk}, ℓ::Integer) where {Tg,Tgr,Tk} =
    _mapC(GL, GR, ketC, operator.C[_mod1(ℓ, length(operator))])

# ---------------- 正交分解 / 重建的 3、4 维重载（MPS 视图语义统一） ----------------

"`_leftsplit(AC) -> (AL, C)`：MPS 视图 `(wl, u·d, wr)` 的左正交分解
（rank-3 中心 / rank-4 CanonicalIMPO 中心重载）。"
_leftsplit(AC::AbstractArray{<:Any,3}) = leftorth(AC, (1, 2), (3,))
function _leftsplit(AC::AbstractArray{<:Any,4})
    AL4, C = leftorth(AC, (1, 2, 4), (3,))
    return permutedims(AL4, (1, 2, 4, 3)), C       # (wl, u, d, wr) → (wl, u, wr, d)
end

"`_rightsplit(AC) -> (C, AR)`：MPS 视图 `(wl, u·d, wr)` 的右正交分解。"
_rightsplit(AC::AbstractArray{<:Any,3}) = rightorth(AC, (1,), (2, 3))
function _rightsplit(AC::AbstractArray{<:Any,4})
    C, AR4 = rightorth(AC, (1,), (2, 4, 3))
    return C, permutedims(AR4, (1, 2, 4, 3))       # (wl, u, d, wr) → (wl, u, wr, d)
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
    return CanonicalIMPS(cast(ψ.AL), cast(ψ.AR), cast(ψ.C), cast(ψ.AC))
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

"Maximum per-site Galerkin residual (mirrors MPSKit's
`calc_galerkin(below, operator, above, envs)`, projecting the local-map output
in the current environments with the current state `x.AL`); the operator-free
overlap-channel version lives in overlap.jl."
function _galerkin_err(operator::DenseIMPO, ket::CanonicalIMPS,
                       x::CanonicalIMPS, envs)
    N = length(ket)
    ϵ = 0.0
    for ℓ in 1:N
        ACmap = _mapAC(leftenv(envs, ℓ), operator[_mod1(ℓ, length(operator))],
                       rightenv(envs, ℓ), ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], ACmap))
    end
    return ϵ
end

"""
    _vomps_sweeps(operator, ket, x0; tol, maxiter, verbosity, iters,
                  alg_gauge, alg_environments, alg_orth) -> (x, envs)

Overlap-maximizing variational sweeps of the MPO-application / MPO-composition
channels (strictly mirroring MPSKit's `approximate(ψ₀, (O, ϕ), VOMPS())`,
`src/algorithms/approximate/vomps.jl`): find `x` approximating `operator|ket⟩`.
按 `operator` 槽类型分派：`DenseIMPO` + `CanonicalIMPS` ket = mpo·mps 施加通道；
`CanonicalIMPO` + `CanonicalIMPO` = mpo·mpo 组合通道（bra 槽为 CanonicalIMPO，
原生 MPO 形态演化，乘积张量从不物化）。Each round (MPSKit `IterativeSolver`
pipeline):

1. `localupdate`: per-site local maps `AC_new = (GL·O·GR)·ket.AC`,
   `C_new = GL₊·ket.C·GR` → `regauge!` yields candidate `AL`s (all sites
   against the same, pre-sweep environments — Jacobi style);
2. `gauge_step!`: `gaugefix!(; order = :R)` restores the global right gauge,
   at the dynamically adapted gauge tolerance (`adapt_solver(alg_gauge)`);
3. `envs_step!`: environments recomputed (warm-started from the previous
   round's fixed points, dynamically adapted tolerance `alg_environments`);
4. convergence `ϵ = _galerkin_err ≤ tol` checked **after** the sweep (MPSKit
   `IterativeSolver` semantics: at least one sweep always runs).

`iters::Ref{Int}` optionally receives the number of sweeps performed. Returns
the optimized state and its final environments (MPSKit `approximate`
convention: `(ψ, envs, ϵ)`); callers that want a fidelity diagnostic can
compute it once from `envs` afterwards.
"""
function _vomps_sweeps(operator::Union{DenseIMPO,CanonicalIMPO},
                       ket::Union{CanonicalIMPS,CanonicalIMPO},
                       x0::Union{CanonicalIMPS,CanonicalIMPO};
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       iters::Union{Nothing,Base.RefValue{Int}} = nothing,
                       alg_gauge = Defaults.alg_gauge(),
                       alg_environments = Defaults.alg_environments(),
                       alg_orth = Defaults.alg_orth())
    N = length(ket)
    x = copy(x0)
    envs = MultCache(x, operator, ket, alg_environments)
    # 通道标量类型（MPSKit 对齐）：环境按 eigsolve 的实际 eltype 存放（实输入下
    # 融合转移的 leading vector 可为复），演动态随之提升，扫掠在提升后算术上进行
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    # 初始残差（MPSKit 用于日志与动态容差适配；收敛判定在扫掠之后）
    ϵ = _galerkin_err(operator, ket, x, envs)
    iter = 0
    for outer iter in 1:maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # 候选 AL 与 ket.AC 同形，eltype 提升到通道标量类型 T）
        ALs = [similar(ket.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            AC_new = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), operator, ket.AC[ℓ], ℓ)
            C_new = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ),
                          operator, ket.C[ℓ], ℓ)
            ALs[ℓ] = regauge!(AC_new, C_new; alg = alg_orth)
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!:
        # seeded with state.C[end], dynamically adapted gauge tolerance)
        alg_g = updatetol(alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（热启动：上一轮的不动点作为 eigsolve 初值 + 动态环境容差）
        alg_envs = updatetol(alg_environments, iter - 1, ϵ)
        envs = MultCache(x, operator, ket, alg_envs;
                         GL0 = envs.lefts[1], GR0 = envs.rights[N])
        ϵ = _galerkin_err(operator, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        ϵ ≤ tol && break
    end
    iters === nothing || (iters[] = iter)
    _global_normalize!(x)
    return x, envs
end

# ---------------- compression sweeps of the IDMRG template ----------------

# Ternary-channel incremental environment pushes for the IDMRG sweep (mirroring
# MPSKit's `transfer_leftenv!`/`transfer_rightenv!` for the
# `⟨below|operator|above⟩` channel: the below side is the state being optimized,
# the above side the target chain; the operator-free overlap-channel versions
# live in overlap.jl).
function transfer_leftenv!(envs::MultCache, x::CanonicalIMPS,
                           operator::DenseIMPO, ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], x.AL[ℓm],
                                  operator[_mod1(ℓm, length(operator))], ket.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::MultCache, x::CanonicalIMPS,
                            operator::DenseIMPO, ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ket.AR[ℓp],
                                    operator[_mod1(ℓp, length(operator))], x.AR[ℓp])
    return envs
end

"Ternary-channel environment rescaling during the sweep (mirrors MPSKit's
`normalize!(envs, below, operator, above)`): unit-Frobenius `GR`s; `GL[ℓ+1]`
scaled by `inv(λ)` with the local C-channel overlap
`λ = ⟨x.C[ℓ], _mapC(GL₊, GR, ket.C[ℓ])⟩`."
function _normalize_ternary_envs!(envs::MultCache, x::CanonicalIMPS,
                                  ket::CanonicalIMPS)
    N = length(ket)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"""
    _idmrg_sweeps(operator, ket, x0; tol, maxiter, verbosity, iters, alg_gauge) -> (x, envs)

IDMRG sweeps of the MPO-application / MPO-composition channels, strictly
mirroring MPSKit's `approximate(ψ₀, (O, ϕ), IDMRG())`
(`src/algorithms/approximate/idmrg.jl`): a sequential Gauss–Seidel double sweep
over the unit cell with on-the-fly environment transfer, converged on the
boundary center-matrix drift (converging to the same fixed point as the VOMPS
template of [`_vomps_sweeps`](@ref)):

1. left-to-right sweep: per site, the local projection
   `k = (GL·O·GR)·ket.AC`, normalized, split by `leftorth` into
   `AL[ℓ]/C[ℓ]`; `transfer_leftenv!` immediately pushes `GL[ℓ+1]` through the
   new `AL[ℓ]`, so site ℓ+1 is updated against environments that already
   contain it;
2. right-to-left sweep: per site, `k` again, split by `rightorth` into
   `C[ℓ-1]/AR[ℓ]`; `transfer_rightenv!` pushes `GR[ℓ-1]` through `AR[ℓ]`;
3. environments rescaled (MPSKit `normalize!`: unit-norm `GR`s, `GL[ℓ+1]`
   scaled by the local C-channel overlap);
4. convergence: `ϵ = ‖C[0]_new − C[0]_old‖` (boundary center-matrix drift).

Afterwards the mixed-canonical state is rebuilt from the `AR` string (MPSKit
`MultilineMPS(ψ.AR)` at the dynamically adapted `alg_gauge` tolerance) and the
environments are recomputed for the final state. （compress 的无算符版本见
overlap.jl 的 `_overlap_idmrg_sweeps`。）`iters::Ref{Int}` optionally receives
the sweep count.
"""
function _idmrg_sweeps(operator::Union{DenseIMPO,CanonicalIMPO},
                       ket::Union{CanonicalIMPS,CanonicalIMPO},
                       x0::Union{CanonicalIMPS,CanonicalIMPO};
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       iters::Union{Nothing,Base.RefValue{Int}} = nothing,
                       alg_gauge = Defaults.alg_gauge())
    N = length(ket)
    x = copy(x0)
    # 初始环境：由初态解一次左右不动点（MPSKit environments(ψ, toapprox...)），
    # 扫掠中只做增量 transfer 与重标定，不再整体重解（MPSKit IDMRG 语义）
    envs = MultCache(x, operator, ket, Defaults.alg_environments())
    # 通道标量类型（MPSKit 对齐，见 _vomps_sweeps 注释）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = 2 * tol
    iter = 0
    for outer iter in 1:maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), operator, ket.AC[ℓ], ℓ)
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ])
            transfer_leftenv!(envs, x, operator, ket, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), operator, ket.AC[ℓ], ℓ)
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ])
            transfer_rightenv!(envs, x, operator, ket, ℓ - 1)
        end
        # 环境重标定（MPSKit normalize!(envs, below, operator, above)）
        _normalize_ternary_envs!(envs, x, ket)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        ϵ < tol && break
    end
    iters === nothing || (iters[] = iter)
    # 规范恢复：从 AR 重建混合规范（MPSKit MultilineMPS(ψ.AR; alg_gauge...)），
    # 环境对终态重解（MPSKit recalculate!(envs, ψ, toapprox)）
    alg_g = updatetol(alg_gauge, iter, ϵ)
    x = _rebuild([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
    envs = MultCache(x, operator, ket, Defaults.alg_environments())
    _global_normalize!(x)
    return x, envs
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
# 与 mpo·mps 通道的分工：环境同为 rank-3 `Array{T,3}`——本通道的两个因子键腿
# 分开存放 `(below, wl1, wl2)`/`(wr1, wr2, below)`；局部投影 `_mapAC` 的 pair
# 方法返回 rank-4 `(below, u, d, above)`（物理腿 (u, d) 不融合，调用方需要 MPS
# 视图时自行 reshape 成 (below, u·d, above)，u 快）。

"Pair-channel left push（rank-3 环境，bra 为 CanonicalIMPO rank-4 张量）：
`L′[bl′, wr1, wr2] = Σ L[bl, wl1, wl2]·conj(below[bl, u, bl′, d])·W1[wl1, u, wr1, m]·W2[wl2, m, wr2, d]`。
显式三步二元收缩（键优先：wl1 → (wl2, m)，bra 物理 u/d 各随 W1/W2 收缩），
融合张量不落地。"
function _push_env_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,4},
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

"Pair-channel right push（rank-3 环境，bra 为 CanonicalIMPO rank-4 张量）：
`R′[wl1, wl2, bl′] = Σ R[wr1, wr2, bl]·W1[wl1, u, wr1, m]·W2[wl2, m, wr2, d]·conj(below[bl′, u, bl, d])`。
显式三步二元收缩（键优先：wr2 → (wl2, m)，bra 物理 u/d 各随 W1/W2 收缩），
融合张量不落地。"
function _push_env_right(R::AbstractArray{TR,3}, W1::AbstractArray{Tw1,4},
                         W2::AbstractArray{Tw2,4}, below::AbstractArray{Tb,4}) where {TR,Tb,Tw1,Tw2}
    # 步1（键 wr2；bra 物理 d 一并收缩）：Y[(wr1, m, wl2, d, br)]
    Y = @tensor Y[c1, a2, mm, dd, jR] := R[c1, c2, jR] * W2[a2, mm, c2, dd]
    # 步2（bra 物理 d、键 jR 收缩）：Z[(wr1, m, wl2, u, bl)]
    Z = @tensor Z[c1, a2, mm, jL, u] := Y[c1, a2, mm, dd, jR] * conj(below[jL, u, jR, dd])
    # 步3（键 wr1 与桥 m、bra 物理 u 一并收缩）
    @tensor R′[a1, a2, jL] := Z[c1, a2, mm, jL, u] * W1[a1, u, c1, mm]
end

"Pair-channel local AC projection (identity channel, rank-3 环境)：
`AC4[bl, u, br, d] = Σ GL[bl, wl1, wl2]·W1[wl1, u, wr1, m]·W2[wl2, m, wr2, d]·GR[wr1, wr2, br]`。
显式三步键优先 GEMM；**直接返回 rank-4**（CanonicalIMPO 张量约定 `(wl, u, wr, d)`，
物理腿 (u, d) 不融合），调用方需要 MPS 视图时自行 permute+reshape。"
function _mapAC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
                W1::AbstractArray{Tw1,4}, W2::AbstractArray{Tw2,4}) where {Tg,Tgr,Tw1,Tw2}
    Y = @tensor Y[jL, w2, u, mm, r1] := GL[jL, w1, w2] * W1[w1, u, r1, mm]
    Z = @tensor Z[jL, u, dd, r1, r2] := Y[jL, w2, u, mm, r1] * W2[w2, mm, r2, dd]
    @tensor AC4[jL, u, jR, dd] := Z[jL, u, dd, r1, r2] * GR[r1, r2, jR]
end

"Pair-channel local C projection：融合 C（外积 kron 布局，wl1/wr1 快）——
`Cnew[bl, br] = Σ GL[bl, wl1, wl2]·C1[wl1, wr1]·C2[wl2, wr2]·GR[wr1, wr2, br]`。"
function _mapC(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
               C2::AbstractMatrix, C1::AbstractMatrix) where {Tg,Tgr}
    Y = @tensor Y[jL, w1, r2] := GL[jL, w1, w2] * C2[w2, r2]
    Z = @tensor Z[jL, r1, r2] := Y[jL, w1, r2] * C1[w1, r1]
    @tensor Cnew[jL, jR] := Z[jL, r1, r2] * GR[r1, r2, jR]
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
    return CanonicalIMPO{T}(cast(W.AL), cast(W.AR), cast(W.C), cast(W.AC))
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

"""
    _ternary_fixedpoints(below::CanonicalIMPO, W1::CanonicalIMPO, W2::CanonicalIMPO;
                         tol, krylovdim, maxiter, GL0, GR0) -> (GLs, GRs)

双 MPO 组合通道（mpo·mpo，bra 为 CanonicalIMPO）的 identity 通道固定点：环境
推直接消费 `(W1, W2)` 因子对（[`_push_env_left`](@ref)/[`_push_env_right`](@ref)，
rank-3 环境 `(below, wl1, wl2)`/`(wr1, wr2, below)`）。乘积因子各自保持规范
（W1、W2 各自混合规范 ⇒ 乘积 AL 左正交、AR 右正交、AC·C 一致），无需 gauge
twist。
"""
function _ternary_fixedpoints(below::CanonicalIMPO, W1::CanonicalIMPO,
                              W2::CanonicalIMPO, alg = Defaults.alg_environments();
                              GL0::Union{Nothing,AbstractArray} = nothing,
                              GR0::Union{Nothing,AbstractArray} = nothing)
    alg = _envalg(alg)                       # 解开 DynamicTol 包装（.tol/.maxiter）
    N = length(below)
    NW = length(W1)
    (N % NW == 0 && length(W2) == N) ||
        throw(DimensionMismatch("incompatible unit-cell lengths"))
    T = promote_type(scalartype(below), scalartype(W1), scalartype(W2))
    Dl = size(below.AL[1], 1)
    D1 = size(W1.AL[1], 1)
    D2 = size(W2.AL[1], 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, D1, D2)
        for ℓ in 1:N
            GL = _push_env_left(GL, below.AL[ℓ], W1.AL[_mod1(ℓ, NW)], W2.AL[ℓ])
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * D1 * D2) : vec(copy(GL0))
    _, GL1 = _eigsolve(Tleft, v0L, 1, :LM; ishermitian = false, tol = alg.tol,
                       krylovdim = Defaults.krylovdim, maxiter = alg.maxiter)
    # 复环境提升（MPSKit 对齐：环境按 eigsolve 返回的实际 eltype 存放）
    TCL = promote_type(T, eltype(GL1[1]))
    GLs = Vector{Array{TCL,3}}(undef, N)
    GLs[1] = GL = reshape(GL1[1], Dl, D1, D2)
    for ℓ in 2:N
        GLs[ℓ] = GL = _push_env_left(GL, below.AL[ℓ-1],
                                     W1.AL[_mod1(ℓ - 1, NW)], W2.AL[ℓ-1])
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, D1, D2, Dl)
        for ℓ in N:-1:1
            GR = _push_env_right(GR, W1.AR[_mod1(ℓ, NW)], W2.AR[ℓ], below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, D1 * D2 * Dl) : vec(copy(GR0))
    _, GRN = _eigsolve(Tright, v0R, 1, :LM; ishermitian = false, tol = alg.tol,
                       krylovdim = Defaults.krylovdim, maxiter = alg.maxiter)
    TCR = promote_type(T, eltype(GRN[1]))
    GRs = Vector{Array{TCR,3}}(undef, N)
    GRs[N] = GR = reshape(GRN[1], D1, D2, Dl)
    for ℓ in N-1:-1:1
        GRs[ℓ] = GR = _push_env_right(GR, W1.AR[_mod1(ℓ + 1, NW)], W2.AR[ℓ+1],
                                      below.AR[ℓ+1])
    end

    # 归一化（MPSKit 约定：GR Frobenius、GL 乘局部 overlap λ）
    for ℓ in 1:N
        GRs[ℓ] ./= norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC(GLs[inext], GRs[ℓ], W2.C[ℓ], W1.C[_mod1(ℓ, NW)])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("ternary environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] ./= λ
    end
    return GLs, GRs
end

"双 MPO 组合通道的最大逐站 Galerkin 残差（bra 为 CanonicalIMPO；语义同
`_galerkin_err(operator::DenseIMPO, ...)`）。"
function _galerkin_err(operator::CanonicalIMPO, ket::CanonicalIMPO,
                       x::CanonicalIMPO, envs)
    N = length(ket)
    NW = length(operator)
    ϵ = 0.0
    for ℓ in 1:N
        W1t = operator.AC[_mod1(ℓ, NW)]
        k4 = _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), W1t, ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k4))
    end
    return ϵ
end

"双 MPO 组合通道的环境重标定（MPSKit `normalize!(envs, below, operator, above)`
语义：GR Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）。"
function _normalize_ternary_envs!(envs::MultCache{<:CanonicalIMPO,
                                                  <:Union{CanonicalIMPS,CanonicalIMPO},
                                                  <:CanonicalIMPO},
                                  x::CanonicalIMPO, ket::CanonicalIMPO)
    N = length(ket)
    NW = length(envs.operator)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ),
                     ket.C[ℓ], envs.operator.C[_mod1(ℓ, NW)])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

# 双 MPO 组合通道的增量环境推进（bra 为 CanonicalIMPO；rank-3 环境）
function transfer_leftenv!(envs::MultCache{<:CanonicalIMPO,
                                            <:Union{CanonicalIMPS,CanonicalIMPO},
                                            <:CanonicalIMPO},
                           x::CanonicalIMPO, operator::CanonicalIMPO,
                           ket::CanonicalIMPO, site::Int)
    N = length(ket)
    NW = length(operator)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = _push_env_left(envs.lefts[ℓm], x.AL[ℓm],
                                   operator.AL[_mod1(ℓm, NW)], ket.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::MultCache{<:CanonicalIMPO,
                                             <:Union{CanonicalIMPS,CanonicalIMPO},
                                             <:CanonicalIMPO},
                            x::CanonicalIMPO, operator::CanonicalIMPO,
                            ket::CanonicalIMPO, site::Int)
    N = length(ket)
    NW = length(operator)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = _push_env_right(envs.rights[ℓp],
                                     operator.AR[_mod1(ℓp, NW)], ket.AR[ℓp],
                                     x.AR[ℓp])
    return envs
end

"""
    mult(W, ψ) -> y::CanonicalIMPS
    mult(W, W2) -> y::CanonicalIMPO

Exact application/composition without compression: the naive construction
(fuse / MPO composition) is canonicalized into mixed-canonical storage. The
output bond dimension is the naive bond dimension (inherently large for large
inputs); the result is normalized (幅值不携带信息，只有方向有意义).
"""
function mult(W, ψ::CanonicalIMPS)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    K = [fuse(Wm[ℓ], ψ.AL[ℓ]) for ℓ in 1:length(ψ)]
    return _global_normalize!(CanonicalIMPS(collect(K)))
end

"""
    mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPS
    mult(W, W2, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPO

The compute-on-the-fly version of the MPO multiplication: find `y ≈ W·ψ`
(operator application) or `y ≈ W·W2` (operator composition), variationally
compressed to the bond dimension `alg.D`. This method **never
materializes the naive target family**: for mpo·mps the local maps
`k = GL·W·ket·GR` are computed per site on the fly; for mpo·mpo the
factorized engine consumes the `(W1, W2)` tensor pairs directly with
bond-first contractions (the fused tensors are never formed). Intermediate
memory is O(single site) in both cases. Applying a time-evolution MPO is
`ψ′ = mult(make_time_mpo(H, dt, WII()), ψ)`.

- `W`: an `DenseIMPO`, an `SparseIMPO` (densified into an `DenseIMPO`
  before application), or an `CanonicalIMPO`;
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

The internal [`_mult`](@ref) additionally returns the sweep count
(`(y, iters)`); the exported wrappers discard it.
"""
mult(W, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    first(_mult(W, ψ, alg, nothing; D = alg.D))

function _mult(W, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
               ψ₀::Union{Nothing,CanonicalIMPS}; D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    N = length(ψ)
    # MPO-channel VOMPS/IDMRG (compute-on-the-fly, no naive target family)
    x0 = ψ₀ !== nothing ? ψ₀ : svdguess_mult(Wm, ψ, D)
    iters = Ref(0)
    y, _ = if alg isa VOMPS
        _vomps_sweeps(Wm, ψ, x0; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, iters = iters,
                      alg_gauge = alg.alg_gauge,
                      alg_environments = alg.alg_environments,
                      alg_orth = alg.alg_orth)
    else
        _idmrg_sweeps(Wm, ψ, x0; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, iters = iters,
                      alg_gauge = alg.alg_gauge)
    end
    # guarantee the mixed canonical form: re-right-canonicalize from AL + C[end]
    # (preserving the ray), then normalize to the package norm convention
    y = CanonicalIMPS(collect(y.AL), y.C[end])
    return _global_normalize!(y), iters[]
end

function mult(W, W2::Union{DenseIMPO,CanonicalIMPO})
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    N = length(W2m)
    NW = length(Wm)
    # exact composition: naive fuse family → canonical storage (the output is
    # the target ray itself, normalized)
    K4 = [_naive_mul_tensor(Wm[_mod1(ℓ, NW)], W2m[ℓ]) for ℓ in 1:N]
    x = CanonicalIMPS(vectorize(DenseIMPO(K4)).As)
    _global_normalize!(x)
    return devectorize(x)
end

mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, alg::Union{VOMPS,IDMRG}) =
    first(_mult(W, W2, alg, nothing; D = alg.D))

function _mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, alg::Union{VOMPS,IDMRG},
               ψ₀::Union{Nothing,CanonicalIMPO}; D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    # 双 MPO 组合通道：bra = 变分链（CanonicalIMPO，原生 MPO 形态）、
    # operator/ket 槽 = (W1c, W2c)——乘积张量从不物化（键优先分步 GEMM），
    # 因子各自保持规范 ⇒ 无需 gauge twist。
    W1c = W isa CanonicalIMPO ? W : CanonicalIMPO(collect(Wm.Ws))
    W2c = W2 isa CanonicalIMPO ? W2 : CanonicalIMPO(collect(W2m.Ws))
    x0 = ψ₀ !== nothing ? ψ₀ : devectorize(svdguess_mult(Wm, W2m, D))
    iters = Ref(0)
    x, _ = if alg isa VOMPS
        _vomps_sweeps(W1c, W2c, x0; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, iters = iters,
                      alg_gauge = alg.alg_gauge,
                      alg_environments = alg.alg_environments,
                      alg_orth = alg.alg_orth)
    else
        _idmrg_sweeps(W1c, W2c, x0; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, iters = iters,
                      alg_gauge = alg.alg_gauge)
    end
    return x, iters[]
end

# ---------------- svdguess_mult (deterministic initial guess) & mult! (in-place) ----------------

"""
    svdguess_mult(W, ψ, D) -> CanonicalIMPS
    svdguess_mult(W, W2, D) -> CanonicalIMPS
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
low-level entry point for downstream packages; the `CanonicalIMPS` methods
are thin wrappers that re-canonicalize its output.
"""
function svdguess_mult(W, ψ::CanonicalIMPS, D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    return CanonicalIMPS(svdguess_mult(PeriodicVector(Wm.Ws), ψ.AL, D))
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

function svdguess_mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    return CanonicalIMPS(svdguess_mult(PeriodicVector(Wm.Ws), PeriodicVector(W2m.Ws), D))
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
    mult!(out, W, ψ, alg::Union{VOMPS,IDMRG}) -> out
    mult!(out, W, W2, alg::Union{VOMPS,IDMRG}) -> out

In-place [`mult`](@ref): `out` is the user-provided state/operator to be
optimized as the initial guess. The target bond dimension is taken from the
bond profile of `out` (its bond profile is first brought to uniform
`D = max_bonddim(out)` with [`changebond!`](@ref)); `alg.D` is ignored. The
optimized result is written back into `out`.
"""
function mult!(out::CanonicalIMPS, W, ψ::CanonicalIMPS,
               alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, _ = _mult(W, ψ, alg, out; D = D)
    return _copyinto!(out, y)
end

"Raw-ket variant: `mult!` of a strict-algebra input (`W * DenseIMPS`, not yet
canonicalized) — the input is canonicalized and the standard `mult!` runs."
function mult!(out::CanonicalIMPS, W, ψ::DenseIMPS,
               alg::Union{VOMPS,IDMRG}; kwargs...)
    return mult!(out, W, CanonicalIMPS(collect(ψ.As)), alg; kwargs...)
end

function mult!(out::CanonicalIMPO, W, W2::Union{DenseIMPO,CanonicalIMPO},
               alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, _ = _mult(W, W2, alg, out; D = D)
    return _copyinto!(out, y)
end
