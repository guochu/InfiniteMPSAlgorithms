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

Environments of the MPO-application channel: the left/right fixed points of
`⟨below|operator|above⟩` (`operator::DenseIMPO`), used by `mult` (iterative
MPO multiplication); below (bra) and above (ket) may be different states.

- `lefts[ℓ]`: the left environment of site ℓ, `(below bond, w, above bond)`;
- `rights[ℓ]`: the right environment of site ℓ, `(above bond, w, below bond)`
  (MPSKit convention);
- the fixed points are obtained from the :LM eigenpairs of the fused transfer
  matrix `T(above.AL, operator, below.AL)` (eigsolve);
- normalization mirrors MPSKit's `normalize!(::InfiniteEnvironments)`: each GR
  is Frobenius-normalized first, then per site `λℓ = ⟨below.C[ℓ], C_map(ℓ)⟩`
  scales `GLs[ℓ+1]`, so that the local contraction of every site is exactly 1
  (identity-MPO expectation = N).
"""
struct MultCache{O<:DenseIMPO,B<:CanonicalIMPS,K<:CanonicalIMPS,T} <: Environments
    operator::O
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

function MultCache(below::CanonicalIMPS, operator::DenseIMPO,
                   above::CanonicalIMPS;
                   tol::Real = Defaults.tol, krylovdim::Int = Defaults.krylovdim,
                   maxiter::Int = Defaults.maxiter,
                   GL0::Union{Nothing,AbstractArray} = nothing,
                   GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = _ternary_fixedpoints(below, operator, above; tol, krylovdim, maxiter, GL0, GR0)
    return MultCache(operator, below, above, GLs, GRs)
end

# ---------------- pure overlap-channel environment cache (OverlapCache) ----------------

"""
    OverlapCache(bra, ket, lefts, rights)
    OverlapCache(ψ) -> OverlapCache
    OverlapCache(below, above; tol, krylovdim, maxiter) -> OverlapCache

Environments of the pure overlap channel: the left/right fixed points of
`⟨bra|ket⟩` (identity channel, w dimension 1), used by the IDMRG compression
path of `mult` and the iterative compressions of `add`/`hadamard`.

- `OverlapCache(ψ)`: the identity channel uses the AL/AR gauges directly
  (fixed points = identity matrices);
- `OverlapCache(below, above)`: below and above may be different states; the
  fixed points are obtained from the :LM eigenpairs of the fused transfer
  matrix `T(above.AL, below.AL)` (eigsolve); normalization identical to
  [`MultCache`](@ref) (MPSKit convention).
- `lefts[ℓ]`: the left environment of site ℓ, `(bra bond, 1, ket bond)`;
- `rights[ℓ]`: the right environment of site ℓ, `(ket bond, 1, bra bond)`.
"""
struct OverlapCache{B<:CanonicalIMPS,K<:CanonicalIMPS,T} <: Environments
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"Identity channel: in the AL/AR gauges the fixed points are identity matrices."
function OverlapCache(ψ::CanonicalIMPS; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Array{T,3}}(undef, N)
    rights = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        Dl, Dr = size(ψ.AL[ℓ], 1), size(ψ.AL[ℓ], 3)
        lefts[ℓ] = _to3(Matrix{T}(I, Dl, Dl))
        rights[ℓ] = _to3(Matrix{T}(I, Dr, Dr))
    end
    return OverlapCache(ψ, ψ, lefts, rights)
end

"Ternary overlap channel: left/right fixed points of ⟨below|above⟩."
function OverlapCache(below::CanonicalIMPS, above::CanonicalIMPS;
                      tol::Real = Defaults.tol, krylovdim::Int = Defaults.krylovdim,
                      maxiter::Int = Defaults.maxiter,
                      GL0::Union{Nothing,AbstractArray} = nothing,
                      GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = _ternary_fixedpoints(below, nothing, above; tol, krylovdim, maxiter, GL0, GR0)
    return OverlapCache(below, above, GLs, GRs)
end

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
in the current environments with the current state `x.AL`)."
function _galerkin_err(operator::Union{Nothing,DenseIMPO}, ket::CanonicalIMPS,
                       x::CanonicalIMPS, envs)
    N = length(ket)
    ϵ = 0.0
    for ℓ in 1:N
        O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
        ACmap = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], ACmap))
    end
    return ϵ
end

"""
    _vomps_sweeps(operator, ket, x0, K; tol, maxiter, verbosity, iters) -> (x, envs)

Overlap-maximizing variational sweeps (strictly mirroring MPSKit's
`approximate(ψ₀, (O, ϕ), VOMPS())`, `src/algorithms/approximate/vomps.jl`):
find `x` approximating `operator|ket⟩` (with `operator = nothing`, approximating
the ket itself — variational compression). `K` is only kept in the interface
for backward compatibility; no overlap is computed anywhere in the sweep
(mirroring MPSKit's `approximate`, whose convergence measure is the Galerkin
residual alone). Each round (MPSKit `IterativeSolver` pipeline):

1. `localupdate`: per-site local maps `AC_new = (GL·O·GR)·ket.AC`,
   `C_new = GL₊·ket.C·GR` → `regauge!` yields candidate `AL`s (all sites
   against the same, pre-sweep environments — Jacobi style);
2. `gauge_step!`: `gaugefix!(; order = :R)` restores the global right gauge,
   at the dynamically adapted gauge tolerance (`adapt_solver(alg_gauge)`);
3. `envs_step!`: environments recomputed (warm-started from the previous
   round's fixed points, dynamically adapted tolerance);
4. convergence `ϵ = _galerkin_err ≤ tol` checked **after** the sweep (MPSKit
   `IterativeSolver` semantics: at least one sweep always runs).

`iters::Ref{Int}` optionally receives the number of sweeps performed. Returns
the optimized state and its final environments (MPSKit `approximate`
convention: `(ψ, envs, ϵ)`); callers that want a fidelity diagnostic can
compute it once from `envs` afterwards.
"""
function _vomps_sweeps(operator::Union{Nothing,DenseIMPO}, ket::CanonicalIMPS,
                         x0::CanonicalIMPS, K::Union{Nothing,<:Vector{<:Array}};
                         tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                         verbosity::Int = Defaults.verbosity,
                         iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    N = length(ket)
    x = copy(x0)
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    # 通道标量类型（MPSKit 对齐）：环境按 eigsolve 的实际 eltype 存放（实输入下
    # 融合转移的 leading vector 可为复），演动态随之提升，扫掠在提升后算术上进行
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    # 初始残差（MPSKit 用于日志与动态容差适配；收敛判定在扫掠之后）
    ϵ = _galerkin_err(operator, ket, x, envs)
    iter = 0
    for outer iter in 1:maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境）
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            AC_new = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            C_new = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            ALs[ℓ] = regauge!(AC_new, C_new; alg = Defaults.alg_orth())
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!:
        # seeded with state.C[end], dynamically adapted gauge tolerance)
        alg_gauge = updatetol(Defaults.alg_gauge(), iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
        # envs_step!（热启动：上一轮的不动点作为 eigsolve 初值 + 动态环境容差）
        alg_envs = updatetol(Defaults.alg_environments(), iter - 1, ϵ)
        envs = isnothing(operator) ?
               OverlapCache(x, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N],
                            tol = alg_envs.tol) :
               MultCache(x, operator, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N],
                         tol = alg_envs.tol)
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
# `⟨below|operator|above⟩` channel with distinct below/above: the below side is
# the state being optimized, the above side the target chain).
function transfer_leftenv!(envs::Union{MultCache,OverlapCache}, x::CanonicalIMPS,
                           operator::Union{Nothing,DenseIMPO}, ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = if isnothing(operator)
        push_env_left(envs.lefts[ℓm], x.AL[ℓm], ket.AL[ℓm])
    else
        push_env_left(envs.lefts[ℓm], x.AL[ℓm], operator[ℓm], ket.AL[ℓm])
    end
    return envs
end

function transfer_rightenv!(envs::Union{MultCache,OverlapCache}, x::CanonicalIMPS,
                            operator::Union{Nothing,DenseIMPO}, ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = if isnothing(operator)
        push_env_right(envs.rights[ℓp], ket.AR[ℓp], x.AR[ℓp])
    else
        push_env_right(envs.rights[ℓp], ket.AR[ℓp], operator[ℓp], x.AR[ℓp])
    end
    return envs
end

"Ternary-channel environment rescaling during the sweep (mirrors MPSKit's
`normalize!(envs, below, operator, above)`): unit-Frobenius `GR`s; `GL[ℓ+1]`
scaled by `inv(λ)` with the local C-channel overlap
`λ = ⟨x.C[ℓ], _mapC(GL₊, GR, ket.C[ℓ])⟩`."
function _normalize_ternary_envs!(envs::Environments, x::CanonicalIMPS,
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
    _idmrg_sweeps(ket, x0, K, [operator]; tol, maxiter, verbosity, iters) -> (x, envs)

Compression sweeps of the IDMRG template, strictly mirroring MPSKit's
`approximate(ψ₀, (O, ϕ), IDMRG())` (`src/algorithms/approximate/idmrg.jl`):
a sequential Gauss–Seidel double sweep over the unit cell with on-the-fly
environment transfer, converged on the boundary center-matrix drift
(converging to the same fixed point as the VOMPS template of
[`_vomps_sweeps`](@ref)):

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
`MultilineMPS(ψ.AR)` at the dynamically adapted gauge tolerance) and the
environments are recomputed for the final state. `K`/`alg_eigsolve` are kept
for interface compatibility. `iters::Ref{Int}` optionally receives the sweep
count.
"""
function _idmrg_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS,
                       K::Union{Nothing,<:Vector{<:Array}},
                       operator::Union{Nothing,DenseIMPO} = nothing;
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       alg_eigsolve = Defaults.alg_eigsolve(),
                       iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    N = length(ket)
    x = copy(x0)
    # 初始环境：由初态解一次左右不动点（MPSKit environments(ψ, toapprox...)），
    # 扫掠中只做增量 transfer 与重标定，不再整体重解（MPSKit IDMRG 语义）
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    # 通道标量类型（MPSKit 对齐，见 _vomps_sweeps 注释）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = 2 * tol
    iter = 0
    for outer iter in 1:maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = leftorth(x.AC[ℓ], (1, 2), (3,))
            transfer_leftenv!(envs, x, operator, ket, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            x.AC[ℓ] = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = rightorth(x.AC[ℓ], (1,), (2, 3))
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
    alg_gauge = updatetol(Defaults.alg_gauge(), iter, ϵ)
    x = CanonicalIMPS([x.AR[ℓ] for ℓ in 1:N]; tol = alg_gauge.tol,
                      maxiter = alg_gauge.maxiter)
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    _global_normalize!(x)
    return x, envs
end

# ---------------- factorized double-MPO lazy engine (bond-first contractions) ----------------
#
# mult(W, W2, alg) 的惰性 target：乘积算符 (W1·W2) 的 MPS 视图张量
# Ket[(wl2·wl1), (u·d), (wr2·wr1)] = Σ_m W1[wl1,u,wr1,m]·W2[wl2,m,wr2,d]。
# 经由 LazyKet + `_naive_mul_tensor` 闭包的做法在每次 ALf(ℓ) 时物化
# (D₁·D₂)²·(u·d) 的融合张量——lazy 只推迟了物化，构造路径本身仍是
# 「先缩物理桥指标 m、铺开全部辅助键」的最差顺序（融合键 D₁·D₂ 通常远大于
# 物理维度，(D₁·D₂)²·d² 很快不可行）。因子化引擎把 (W1, W2) 对一路保留到
# 环境收缩里，按「键指标优先、物理指标最后」显式分步 GEMM：最大中间张量
# 只有 O(D·D₁·D₂·d²)，融合张量从不落地。
#
# 乘积因子天然保持规范一致性（W1、W2 各自混合规范 ⇒ 乘积 AL 左正交、AR
# 右正交、AC·C 一致，C 的外积 kron 布局逐位对齐融合键序），因此 identity
# 通道固定点机制无需 gauge twist 直接适用。

"""
    _push_env_left(L, below, W1, W2) -> Matrix

Factorized identity-channel left push:
`L′[bl′, (wr2·wr1)] = Σ conj(below[bl, (u·d), bl′])·L[bl, (wl1·wl2)]·W1·W2`.
显式三步、每步一个二元收缩（TensorOperations 自动做多腿 GEMM），键指标
优先（bl → wl1 → (wl2, m, d)）、物理 (u, d, m) 最后——不物化 (D₁·D₂)²
的融合张量。
"""
function _push_env_left(L::AbstractMatrix, below::AbstractArray{Tb,3},
                        W1::AbstractArray{Tw1,4}, W2::AbstractArray{Tw2,4}) where {Tb,Tw1,Tw2}
    wl1, u1, wr1, m = size(W1)
    wl2, _, wr2, d2 = size(W2)
    bl, bl′ = size(below, 1), size(below, 3)
    L4 = permutedims(reshape(L, bl, wl1, wl2), (2, 3, 1))  # L[bl, (wl1·wl2)] → [wl1, wl2, bl]
    Cb = reshape(below, bl, u1, d2, bl′)           # 融合物理 (u·d)：u 快
    # 步1（键 bl）：Y[(wl1, wl2), (u, d), bl′]
    Y = @tensor Y[w1, w2, u, dd, j] := L4[w1, w2, bl] * conj(Cb[bl, u, dd, j])
    # 步2（键 wl1；物理 u 与 W1 一并收缩）：Z[(wl2, m, wr1, d), bl′]
    Z = @tensor Z[w2, mm, r1, dd, j] := Y[w1, w2, u, dd, j] * W1[w1, u, r1, mm]
    # 步3（键 wl2；桥 m 与物理 d 与 W2 一并收缩）
    Out = @tensor Out[j, r2, r1] := Z[w2, mm, r1, dd, j] * W2[w2, mm, r2, dd]
    return reshape(permutedims(Out, (1, 3, 2)), bl′, wr2 * wr1)
end

"Factorized identity-channel right push:
`R′[(wl1·wl2), bl′] = Σ R[(wr1·wr2), bl]·W1·W2·conj(below[bl′, (u·d), bl])`
（below = x.AR：第一维为新键、第三维为旧键，同 `push_env_right`）。
显式三步二元收缩（键优先：bl → wr2 → (wl1, m, u)；物理 (u, d, m) 最后）。"
function _push_env_right(R::AbstractMatrix, W1::AbstractArray{Tw1,4},
                         W2::AbstractArray{Tw2,4}, below::AbstractArray{Tb,3}) where {Tb,Tw1,Tw2}
    wl1, u1, wr1, m = size(W1)
    wl2, _, wr2, d2 = size(W2)
    bl′ = size(below, 1)                           # 新 below 键（输出列）
    bl = size(below, 3)                            # 旧 below 键（R 的列）
    R4 = reshape(R, wr1, wr2, bl)                  # 融合行 (wr1·wr2)：wr1 快；列 = below 右键
    Cb = reshape(below, bl′, u1, d2, bl)           # 融合物理 (u·d)：u 快
    # 步1（键 bl）：Y[(wr1, wr2), (u, d), bl′]
    Y = @tensor Y[r1, r2, u, dd, j] := R4[r1, r2, bl] * conj(Cb[j, u, dd, bl])
    # 步2（键 wr2；物理 d 与 W2 一并收缩）：Z[(wr1, u, bl′), (wl2, m)]
    Z = @tensor Z[r1, j, u, w2, mm] := Y[r1, r2, u, dd, j] * W2[w2, mm, r2, dd]
    # 步3（键 wr1；桥 m 与物理 u 与 W1 一并收缩）
    Out = @tensor Out[j, w2, w1] := Z[r1, j, u, w2, mm] * W1[w1, u, r1, mm]
    return reshape(permutedims(Out, (3, 2, 1)), wl1 * wl2, bl′)    # ((wl1·wl2), bl′)
end

"Factorized local AC map (identity channel):
`ACnew[aL, (u·d), aR] = Σ GL[aL, (wl1·wl2)]·W1AC·W2AC·GR[(wr2·wr1), aR]`.
显式三步二元收缩（键优先：wl1 → wl2 → (wr1, wr2)；物理 (u, d) 最后）。"
function _mapAC_fused(GL::AbstractMatrix, GR::AbstractMatrix,
                      W1::AbstractArray{Tw1,4}, W2::AbstractArray{Tw2,4}) where {Tw1,Tw2}
    wl1, u1, wr1, m = size(W1)
    wl2, _, wr2, d2 = size(W2)
    aL = size(GL, 1)
    aR = size(GR, 2)
    GL4 = reshape(GL, aL, wl1, wl2)                # 融合列 (wl1·wl2)：wl1 快
    # 步1（键 wl1；物理 u、桥 m 一并收缩）：Y[(aL, wl2, d? 自由), (wr1)]…
    Y = @tensor Y[jL, w2, u, mm, r1] := GL4[jL, w1, w2] * W1[w1, u, r1, mm]
    # 步2（键 wl2；桥 m 与物理 d 与 W2 一并收缩）：Z[(aL, u, d), (wr1·wr2)]
    Z = @tensor Z[jL, u, dd, r1, r2] := Y[jL, w2, u, mm, r1] * W2[w2, mm, r2, dd]
    # 步3（键 wr1、wr2 与 GR 融合行 (wr2·wr1) 收缩）
    Zp = reshape(Z, aL * u1 * d2, wr1 * wr2)
    ACn = Zp * GR                                  # (aL·u·d), aR
    return reshape(ACn, aL, u1 * d2, aR)
end

"Factorized local C map (identity channel):
`Cnew[aL, aR] = Σ GL[aL, (wl1·wl2)]·C2[wl2, wr2]·C1[wl1, wr1]·GR[(wr2·wr1), aR]`.
显式两步二元收缩（全辅助指标）。"
function _mapC_fused(GL::AbstractMatrix, GR::AbstractMatrix,
                     C2::AbstractMatrix, C1::AbstractMatrix)
    aL = size(GL, 1)
    wl2, wr2 = size(C2)
    wl1, wr1 = size(C1)
    GL3 = reshape(GL, aL, wl1, wl2)                # 融合列 (wl1·wl2)：wl1 快
    # 步1（键 wl2）：Y[(aL, wl1), wr2]
    Y = @tensor Y[jL, w1, r2] := GL3[jL, w1, w2] * C2[w2, r2]
    # 步2（键 wl1）：Z[(aL, wr1, wr2)]
    Z = @tensor Z[jL, r1, r2] := Y[jL, w1, r2] * C1[w1, r1]
    return reshape(Z, aL, wr1 * wr2) * GR          # (aL, (wr1·wr2) 列序 = GR 融合行) → (aL, aR)
end

"""
    FactorizedKet(ALf1, ALf2, ARf1, ARf2, ACf1, ACf2, Cf1, Cf2)

因子化双 MPO 惰性 target：乘积算符的融合张量 `Ket[(wl1·wl2), (u·d), (wr1·wr2)]
= W1 ⊗_m W2` **从不物化**，环境收缩直接消费 `(W1, W2)` 张量对（键优先的显式
分步 GEMM，见 [`_push_env_left`](@ref)）。`(AL·C = C·AR = AC)` 逐点成立。
"""
struct FactorizedKet{A1,A2,B1,B2,C1,C2,D1,D2}
    ALf1::A1; ALf2::A2
    ARf1::B1; ARf2::B2
    ACf1::C1; ACf2::C2
    Cf1::D1;  Cf2::D2
end

"""
    _lazy_ternary_fixedpoints(x, ket::FactorizedKet; kwargs...) -> (GLs, GRs)

因子化双 MPO target 的 identity 通道固定点：环境推直接消费 `(W1, W2)` 对
（[`_push_env_left`](@ref)/[`_push_env_right`](@ref)）。乘积因子各自保持
规范（W1、W2 各自混合规范 ⇒ 乘积 AL 左正交、AR 右正交、AC·C 一致），无需
gauge twist。`GL`/`GR` 为 rank-2 `(x 键, 融合键)` 矩阵。
"""
function _lazy_ternary_fixedpoints(x::CanonicalIMPS, ket::FactorizedKet;
                                   tol::Real = Defaults.tol,
                                   krylovdim::Int = Defaults.krylovdim,
                                   maxiter::Int = Defaults.maxiter,
                                   GL0::Union{Nothing,AbstractArray} = nothing,
                                   GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(x)
    T = scalartype(x)
    Dl = size(x.AL[1], 1)
    Dr = size(x.AR[1], 3)
    Da1 = size(ket.ALf1(1), 1) * size(ket.ALf2(1), 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Da1)
        for ℓ in 1:N
            GL = _push_env_left(GL, x.AL[ℓ], ket.ALf1(ℓ), ket.ALf2(ℓ))
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Da1) : vec(copy(GL0))
    _, GL1 = _eigsolve(Tleft, v0L, 1, :LM; ishermitian = false, tol = tol,
                       krylovdim = krylovdim, maxiter = maxiter)
    TCL = promote_type(T, eltype(GL1[1]))
    GLs = Vector{Matrix{TCL}}(undef, N)
    GLs[1] = GL = reshape(GL1[1], Dl, Da1)
    for ℓ in 2:N
        GL = _push_env_left(GL, x.AL[ℓ-1], ket.ALf1(ℓ-1), ket.ALf2(ℓ-1))
        GLs[ℓ] = GL
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da1, Dr)
        for ℓ in N:-1:1
            GR = _push_env_right(GR, ket.ARf1(ℓ), ket.ARf2(ℓ), x.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da1 * Dr) : vec(copy(GR0))
    _, GRN = _eigsolve(Tright, v0R, 1, :LM; ishermitian = false, tol = tol,
                       krylovdim = krylovdim, maxiter = maxiter)
    TCR = promote_type(T, eltype(GRN[1]))
    GRs = Vector{Matrix{TCR}}(undef, N)
    GRs[N] = GR = reshape(GRN[1], Da1, Dr)
    for ℓ in N-1:-1:1
        GR = _push_env_right(GR, ket.ARf1(ℓ+1), ket.ARf2(ℓ+1), x.AR[ℓ+1])
        GRs[ℓ] = GR
    end

    # 归一化（MPSKit 约定：GR Frobenius、GL 乘局部 overlap λ）
    for ℓ in 1:N
        GRs[ℓ] = GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC_fused(GLs[inext], GRs[ℓ], ket.Cf2(ℓ), ket.Cf1(ℓ))
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("factorized lazy environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] = GLs[inext] ./ λ
    end
    return GLs, GRs
end

"因子化 target 的 per-site Galerkin 残差（语义同 `_galerkin_err`）。"
function _lazy_galerkin_err(x::CanonicalIMPS, ket::FactorizedKet, GLs, GRs, N)
    ϵ = 0.0
    for ℓ in 1:N
        k = _mapAC_fused(GLs[ℓ], GRs[ℓ], ket.ACf1(ℓ), ket.ACf2(ℓ))
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k))
    end
    return ϵ
end

"因子化双 MPO 通道的环境重标定（MPSKit `normalize!(envs, below, operator, above)`
语义：GR Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）。"
function _normalize_lazy_mpo_envs!(GLs, GRs, x::CanonicalIMPS, ket::FactorizedKet)
    N = length(x)
    for ℓ in 1:N
        GRs[ℓ] = GRs[ℓ] ./ norm(GRs[ℓ])
        Cnew = _mapC_fused(GLs[_mod1(ℓ + 1, N)], GRs[ℓ], ket.Cf2(ℓ), ket.Cf1(ℓ))
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("factorized idmrg sweep: local overlap λ = 0 at site $ℓ")
        GLs[_mod1(ℓ + 1, N)] = GLs[_mod1(ℓ + 1, N)] ./ λ
    end
    return GLs, GRs
end

"""
    _vomps_mpo_sweeps(ket::FactorizedKet, x0, N; tol, maxiter, verbosity, iters) -> x

VOMPS template on the factorized double-MPO channel (mirroring
[`_vomps_sweeps`](@ref)): Jacobi-style rounds — all sites updated against the
same environments via the local maps `k = _mapAC_fused(...)` /
`ĉ = _mapC_fused(...)` → `regauge!` → `gauge_step!` (dynamically adapted gauge
tolerance) → environments re-solved warm-started (`_lazy_ternary_fixedpoints`,
dynamically adapted tolerance) → Galerkin residual `_lazy_galerkin_err`
checked **after** the sweep. The product factors preserve gauges, so no gauge
twist is needed and the fused tensors are never materialized.
`iters::Ref{Int}` optionally receives the sweep count.
"""
function _vomps_mpo_sweeps(ket::FactorizedKet, x0::CanonicalIMPS, N::Int;
                           tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                           verbosity::Int = Defaults.verbosity,
                           iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    T0 = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T0, eltype(GLs[1])), x)
    T = eltype(x.AL[1])
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
    iter = 0
    for outer iter in 1:maxiter
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = _mapAC_fused(GLs[ℓ], GRs[ℓ], ket.ACf1(ℓ), ket.ACf2(ℓ))
            ĉ = _mapC_fused(GLs[_mod1(ℓ + 1, N)], GRs[ℓ], ket.Cf2(ℓ), ket.Cf1(ℓ))
            ALs[ℓ] = regauge!(k, ĉ; alg = Defaults.alg_orth())
        end
        alg_gauge = updatetol(Defaults.alg_gauge(), iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
        alg_envs = updatetol(Defaults.alg_environments(), iter - 1, ϵ)
        GLs, GRs = _lazy_ternary_fixedpoints(x, ket; GL0 = GLs[1], GR0 = GRs[N],
                                             tol = alg_envs.tol)
        ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        ϵ ≤ tol && break
    end
    iters === nothing || (iters[] = iter)
    _global_normalize!(x)
    return x
end

"""
    _idmrg_mpo_sweeps(ket::FactorizedKet, x0, N; tol, maxiter, verbosity, iters) -> x

IDMRG template on the factorized double-MPO channel (mirroring
[`_idmrg_sweeps`](@ref)): sequential Gauss–Seidel double sweep with on-the-fly
environment transfer (`_push_env_left`/`_push_env_right` through the fresh
`AL`/`AR` and the `(W1, W2)` factor pairs), `leftorth`/`rightorth` splits of
the normalized local projections, per-double-sweep environment rescaling
([`_normalize_lazy_mpo_envs!`](@ref)), and center-matrix-drift convergence
`ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is rebuilt from
the `AR` string (MPSKit `MultilineMPS(ψ.AR)`). `iters::Ref{Int}` optionally
receives the sweep count.
"""
function _idmrg_mpo_sweeps(ket::FactorizedKet, x0::CanonicalIMPS, N::Int;
                           tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                           verbosity::Int = Defaults.verbosity,
                           iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    T0 = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T0, eltype(GLs[1])), x)
    ϵ = 2 * tol
    iter = 0
    for outer iter in 1:maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _mapAC_fused(GLs[ℓ], GRs[ℓ], ket.ACf1(ℓ), ket.ACf2(ℓ))
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = leftorth(x.AC[ℓ], (1, 2), (3,))
            GLs[_mod1(ℓ + 1, N)] = _push_env_left(GLs[ℓ], x.AL[ℓ],
                                                  ket.ALf1(ℓ), ket.ALf2(ℓ))
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _mapAC_fused(GLs[ℓ], GRs[ℓ], ket.ACf1(ℓ), ket.ACf2(ℓ))
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = rightorth(x.AC[ℓ], (1,), (2, 3))
            GRs[_mod1(ℓ - 1, N)] = _push_env_right(GRs[ℓ], ket.ARf1(ℓ),
                                                   ket.ARf2(ℓ), x.AR[ℓ])
        end
        # 环境重标定（MPSKit normalize!(envs, below, operator, above) 语义）
        _normalize_lazy_mpo_envs!(GLs, GRs, x, ket)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        ϵ < tol && break
    end
    iters === nothing || (iters[] = iter)
    # 规范恢复：从 AR 重建混合规范，环境对终态重解
    alg_gauge = updatetol(Defaults.alg_gauge(), iter, ϵ)
    x = CanonicalIMPS([x.AR[ℓ] for ℓ in 1:N]; tol = alg_gauge.tol,
                      maxiter = alg_gauge.maxiter)
    _global_normalize!(x)
    return x
end


# The strict (compression-free) constructions live in operators/linalg.jl as
# the typed operators `Base.:*(::DenseIMPO, ::DenseIMPO)` /
# `Base.:*(::DenseIMPO, ::DenseIMPS)`; this file provides the iterative
# (variational) versions:
# - `VOMPS`: overlap-maximizing ALS sweeps (strictly mirrors MPSKit VOMPS,
#   src/algorithms/approximate/vomps.jl);

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
        _vomps_sweeps(Wm, ψ, x0, nothing; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, iters = iters)
    else
        _idmrg_sweeps(ψ, x0, nothing, Wm; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve,
                      iters = iters)
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
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x = CanonicalIMPS(collect(K3))
    _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds)
end

mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, alg::Union{VOMPS,IDMRG}) =
    first(_mult(W, W2, alg, nothing; D = alg.D))

function _mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, alg::Union{VOMPS,IDMRG},
               ψ₀::Union{Nothing,CanonicalIMPS}; D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    N = length(W2m)
    NW = length(Wm)
    # 因子化惰性引擎（compute-on-the-fly）：乘积算符的融合张量从不物化，
    # 环境收缩直接消费 (W1, W2) 对（键优先显式分步 GEMM；乘积因子各自保持
    # 规范 ⇒ 无需 gauge twist）。
    W1c = W isa CanonicalIMPO ? W : CanonicalIMPO(collect(Wm.Ws))
    W2c = W2 isa CanonicalIMPO ? W2 : CanonicalIMPO(collect(W2m.Ws))
    dus = [size(W1c.AL[_mod1(ℓ, NW)], 2) for ℓ in 1:N]
    dds = [size(W2c.AL[ℓ], 4) for ℓ in 1:N]
    x0 = ψ₀ !== nothing ? ψ₀ : svdguess_mult(Wm, W2m, D)
    ket = FactorizedKet(
        ℓ -> W1c.AL[_mod1(ℓ, NW)], ℓ -> W2c.AL[ℓ],
        ℓ -> W1c.AR[_mod1(ℓ, NW)], ℓ -> W2c.AR[ℓ],
        ℓ -> W1c.AC[_mod1(ℓ, NW)], ℓ -> W2c.AC[ℓ],
        ℓ -> W1c.C[_mod1(ℓ, NW)], ℓ -> W2c.C[ℓ])
    iters = Ref(0)
    x = if alg isa VOMPS
        _vomps_mpo_sweeps(ket, x0, N; tol = alg.tol, maxiter = alg.maxiter,
                          verbosity = alg.verbosity, iters = iters)
    else
        _idmrg_mpo_sweeps(ket, x0, N; tol = alg.tol, maxiter = alg.maxiter,
                          verbosity = alg.verbosity, iters = iters)
    end
    _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds), iters[]
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
    ψ0 = CanonicalIMPS(asmps_view(collect(out.AC)))
    y, _ = _mult(W, W2, alg, ψ0; D = D)
    return _copyinto!(out, y)
end
