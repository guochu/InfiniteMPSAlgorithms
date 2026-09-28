# ---------------- iterative MPO multiplication mult (variational application / compression of MPO·MPS and MPO·MPO) ----------------
#
# Goal: given (W, x), find y ≈ W·x (x an MPS: operator application; x an MPO:
# operator composition). The naive exact construction is exact_mult in
# arithmetics.jl; this file provides the iterative (variational) versions:
# - `VOMPS`: overlap-maximizing ALS sweeps (mirrors MPSKit VOMPS,
#   src/algorithms/approximate/vomps.jl);
# - `IDMRG`: rank-1 effective-Hamiltonian local-map sweeps (converging to
#   the same fixed point as VOMPS).
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
    _overlap_sweeps(operator, ket, x0, K; tol, maxiter, verbosity) -> (x, envs)

Overlap-maximizing variational sweeps (MPSKit VOMPS template): find `x`
approximating `operator|ket⟩` (with `operator = nothing`, approximating the ket
itself — variational compression). `K` is only kept in the interface for
backward compatibility; no overlap is computed anywhere in the sweep (mirroring
MPSKit's `approximate`, whose convergence measure is the Galerkin residual
alone). Each round: ternary fp environments → per-site local maps + `regauge!`
→ `gaugefix!(:R)` → environment refresh + Galerkin-residual convergence
(mirroring MPSKit's `localupdate_step!`/`gauge_step!`/`envs_step!`/
`calc_galerkin` pipeline). Returns the optimized state and its final
environments (MPSKit `approximate` convention: `(ψ, envs, ϵ)`); callers that
want a fidelity diagnostic can compute it once from `envs` afterwards.
"""
function _overlap_sweeps(operator::Union{Nothing,DenseIMPO}, ket::CanonicalIMPS,
                         x0::CanonicalIMPS, K::Union{Nothing,<:Vector{<:Array}};
                         tol::Real = 1.0e-10, maxiter::Int = 100, verbosity::Int = 0)
    N = length(ket)
    x = copy(x0)
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    # 通道标量类型（MPSKit 对齐）：环境按 eigsolve 的实际 eltype 存放（实输入下
    # 融合转移的 leading vector 可为复），演动态随之提升，扫掠在提升后算术上进行
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = _galerkin_err(operator, ket, x, envs)
    for iter in 1:maxiter
        ϵ < tol && break
        # localupdate: per-site local maps + regauge (MPSKit VOMPS local step)
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            AC_new = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            C_new = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            ALs[ℓ] = regauge!(AC_new, C_new; alg = Defaults.alg_orth())
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!:
        # seeded with state.C[end])
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + calc_galerkin: tangent-space residual for the new state/environments
        # （环境热启动：上一轮的不动点作为 eigsolve 初值——逐轮不动点连续变化，
        #   热启动把每轮的支配本征对求解压到一两轮重启）
        envs = isnothing(operator) ?
               OverlapCache(x, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N]) :
               MultCache(x, operator, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N])
        ϵ = _galerkin_err(operator, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
    end
    _global_normalize!(x)
    return x, envs
end

# ---------------- compression sweeps of the IDMRG template ----------------

"""
    _idmrg_sweeps(ket, x0, K; tol, maxiter, verbosity, alg_eigsolve) -> (x, envs)

Compression sweeps of the IDMRG template (converging to the same fixed point
as the VOMPS template of [`_overlap_sweeps`](@ref)). Under the mixed-canonical
identity channel, VOMPS's local exact solution `k = GL·ket.AC·GR` is itself the
map output; a direct eigen solve of the map `x ↦ GL·x·GR` degenerates (the
dominant eigenspace contains arbitrary physical index combinations), so the
rank-1 effective Hamiltonian `ℋ = 𝕀 − |k⟩⟨k|/‖k‖²` is used. Its :SR smallest
eigenvector has the **closed form** `k/‖k‖` (ℋ is positive semidefinite and its
complementary spectrum is exactly 1), so the local update applies it directly
instead of running an iterative eigen solve — the eigensolver route costs
hundreds of matvecs per site per round for the same result. Each round
(MPSKit template):

1. `localupdate`: `k`, `ĉ = GL₊·ket.C·GR` → `AC = k/‖k‖`, `C = ĉ/‖ĉ‖` →
   `regauge!` yields candidate `AL`s;
2. `gauge_step!`: `gaugefix!(; order = :R)` restores the right gauge,
   `AC = AL·C`;
3. environments are recomputed (warm-started from the previous round's fixed
   points);
4. convergence criterion `_galerkin_err` (tangent-space Galerkin residual).

With `operator ≠ nothing` this is the MPO-channel version
(`k = GL·O·ket.AC·GR` computed per site on the fly, compute-on-the-fly).
No overlap is computed anywhere in the sweep (mirroring MPSKit's `approximate`);
the final environments are returned alongside the state, so callers that want a
fidelity diagnostic can compute it once from `envs`. `alg_eigsolve` is kept for
interface compatibility (the local problems no longer run eigen solves).
"""
function _idmrg_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS,
                       K::Union{Nothing,<:Vector{<:Array}},
                       operator::Union{Nothing,DenseIMPO} = nothing;
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       alg_eigsolve = Defaults.alg_eigsolve())
    N = length(ket)
    x = copy(x0)
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    # 通道标量类型（MPSKit 对齐，见 _overlap_sweeps 注释）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = _galerkin_err(operator, ket, x, envs)
    for iter in 1:maxiter
        ϵ < tol && break
        # localupdate: closed-form solution of the rank-1 AC and C subproblems
        # site by site (the :SR eigenvector of ℋ = 𝕀 − |k⟩⟨k|/⟨k,k⟩ is k/‖k‖)
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            k = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            ĉ = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            nk = norm(k)
            nk == 0 && error("rank-1 local map: zero vector at site $ℓ")
            ALs[ℓ] = regauge!(k ./ nk, ĉ ./ norm(ĉ); alg = Defaults.alg_orth())
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!)
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + convergence criterion (mirrors MPSKit calc_galerkin;
        # warm start, see _overlap_sweeps)
        envs = isnothing(operator) ?
               OverlapCache(x, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N]) :
               MultCache(x, operator, ket; GL0 = envs.lefts[1], GR0 = envs.rights[N])
        ϵ = _galerkin_err(operator, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
    end
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

"因子化 target 的变分压缩 sweep（VOMPS/IDMRG）：无 naive 融合张量、无
gauge twist（乘积因子保规范）；收敛判据为 Galerkin 残差；环境逐轮热启动。"
function _lazy_sweeps(ket::FactorizedKet, x0::CanonicalIMPS, N::Int;
                      alg::Union{VOMPS,IDMRG}, tol::Real = Defaults.tol,
                      maxiter::Int = Defaults.maxiter,
                      verbosity::Int = Defaults.verbosity)
    isvomps = alg isa VOMPS
    T = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T, eltype(GLs[1])), x)
    ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
    for iter in 1:maxiter
        ϵ < tol && break
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = _mapAC_fused(GLs[ℓ], GRs[ℓ], ket.ACf1(ℓ), ket.ACf2(ℓ))
            ĉ = _mapC_fused(GLs[_mod1(ℓ + 1, N)], GRs[ℓ], ket.Cf2(ℓ), ket.Cf1(ℓ))
            if isvomps
                ALs[ℓ] = regauge!(k, ĉ; alg = Defaults.alg_orth())
            else
                # rank-1 局部问题的闭式解（同 _idmrg_sweeps）
                nk = norm(k)
                nk == 0 && error("rank-1 local map: zero vector at site $ℓ")
                ALs[ℓ] = regauge!(k ./ nk, ĉ ./ norm(ĉ); alg = Defaults.alg_orth())
            end
        end
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        GLs, GRs = _lazy_ternary_fixedpoints(x, ket; GL0 = GLs[1], GR0 = GRs[N])
        ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
        verbosity > 0 && _logiter(stdout, isvomps ? "VOMPS" : "IDMRG", iter, ϵ)
    end
    _global_normalize!(x)
    return x
end

# ---------------- factorized zip engine (hadamard channel; bond-first contractions) ----------------
#
# hadamard 的惰性 target：`Ket[(c,a), s, (e,b)] = A2[c,s,e]·A1[a,s,b]`（ψ2 键为
# 主指标、ψ1 为次指标，与 `kron(A2, A1)` 融合序一致；物理腿 s 是 below/A1/A2
# 三方共享的收缩边）。经由 LazyKet 闭包的做法每次调用都先物化 (D1·D2, d, D1·D2)
# 的融合张量、再与环境收缩——等价于「先做乘法再压缩」的最差收缩路径（中间张量
# O((D1·D2)²·d)）。因子化引擎把 (A1, A2) 对一路保留到环境/局部映射的收缩里：
# 共享物理腿按站点分片（per-s 片 GEMM，同 `_naive_hadamard_tensor` 的 kron 方
# 案）、键指标优先——最大中间张量 O(Dx·D1·D2·d)，融合 zip 张量从不落地。

"""
    ZipKet(ALf1, ALf2, ARf1, ARf2, ACf1, ACf2, Cf1, Cf2)

因子化 zip（Hadamard/Schur 乘积）惰性 target：`Ket[(c,a), s, (e,b)] =
A2[c,s,e]·A1[a,s,b]`（ψ2 键为主指标、ψ1 为次指标；`AL·C = C·AR = AC` 逐点
成立，C 的融合序 = `kron(C2, C1)`）。环境与局部映射直接消费 `(A1, A2)` 因子
对——融合 zip 张量从不物化。
"""
struct ZipKet{A1,A2,B1,B2,C1,C2,D1,D2}
    ALf1::A1; ALf2::A2
    ARf1::B1; ARf2::B2
    ACf1::C1; ACf2::C2
    Cf1::D1;  Cf2::D2
end

"""
    _zip_push_left(L, below, A2, A1) -> Matrix

因子化 zip 的 identity 通道左推（环境 `L` 为 `(x 键, 融合键 (c·a))` 矩阵，
a 为快指标）：
`L′[bl′, (e·b)] = Σ conj(below[bl, s, bl′])·L[bl, (c·a)]·A2[c,s,e]·A1[a,s,b]`.
物理腿 s 为 below/A2/A1 三方共享边 ⇒ 按物理片收缩（per-s 切片 @tensor，同
`_naive_hadamard_tensor` 的 kron 方案）：最大中间张量 O(Dx·D1·D2·d)，融合
zip 张量从不物化。
"""
function _zip_push_left(L::AbstractMatrix, below::AbstractArray{Tb,3},
                        A2::AbstractArray{Ta,3}, A1::AbstractArray{T1,3}) where {Tb,Ta,T1}
    Dx = size(L, 1)
    bl′ = size(below, 3)
    D2, d = size(A2, 1), size(A2, 2)       # A2[c, s, e]
    D1 = size(A1, 1)                       # A1[a, s, b]
    T = promote_type(eltype(L), eltype(below), eltype(A2), eltype(A1))
    L3 = reshape(L, Dx, D1, D2)            # (bl, a, c)：融合 (c·a) 的 a 为快指标
    belowC = conj(below)                   # (bl, s, bl′)
    W = zeros(T, D2, D1, bl′)              # 各 s 片累加：(e, b, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl, bl′)
        A1s = @view A1[:, s, :]            # (a, b)
        A2s = @view A2[:, s, :]            # (c, e)
        @tensor W[e, b, bl′] += L3[bl, a, c] * A2s[c, e] * A1s[a, b] * below_s[bl, bl′]
    end
    return reshape(permutedims(W, (3, 2, 1)), bl′, D1 * D2)      # (bl′, (e·b))：b 为快指标
end

"""
    _zip_push_right(R, A2, A1, below) -> Matrix

因子化 zip 的 identity 通道右推（环境 `R` 为 `(融合键 (e·b), x 键)` 矩阵，
b 为快指标）：
`R′[(c·a), bl′] = Σ R[(e·b), bl]·A2[c,s,e]·A1[a,s,b]·conj(below[bl′, s, bl])`
（below = x.AR：第一维为新键、第三维为旧键）。物理腿按物理片收缩（per-s
切片 @tensor），最大中间张量 O(D2·D1·Dx·d)，融合 zip 张量从不物化。
"""
function _zip_push_right(R::AbstractMatrix, A2::AbstractArray{Ta,3},
                         A1::AbstractArray{T1,3}, below::AbstractArray{Tb,3}) where {Ta,T1,Tb}
    bl′ = size(below, 1)                   # 新 below 键（输出列）
    bl = size(below, 3)                    # 旧 below 键（R 的列）
    D2, d = size(A2, 1), size(A2, 2)       # A2ar[e, s, c]
    D1 = size(A1, 1)                       # A1ar[b, s, a]
    T = promote_type(eltype(R), eltype(below), eltype(A2), eltype(A1))
    R3 = reshape(R, D1, D2, bl)            # (b, e, bl)：GR 行 = b + (e-1)·D1，b 为快指标
    belowC = conj(below)                   # (bl′, s, bl)
    W = zeros(T, D1, D2, bl′)              # 各 s 片累加：(a, c, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl′, bl)
        A1s = @view A1[:, s, :]            # (a, b)：A1 的左键 a 为行
        A2s = @view A2[:, s, :]            # (c, e)：A2 的左键 c 为行
        @tensor W[a, c, bl′] += A2s[c, e] * A1s[a, b] * R3[b, e, bl] * below_s[bl′, bl]
    end
    return reshape(W, D1 * D2, bl′)                             # ((c, a), bl′)：a 为快指标
end

"""
    _mapAC_zip(GL, GR, A2ac, A1ac) -> Array{T,3}

因子化 zip 的 identity 通道局部 AC 映射：
`k[xL, p, xR] = Σ GL[xL, (c·a)]·A2ac[c,p,e]·A1ac[a,p,b]·GR[(e·b), xR]`.
物理腿 p 为两因子共享的开放指标（非收缩边），分两步 @tensor：键 c → 键 a →
融合右键 (e·b)，最大中间张量 O(Dx·D1·d·D2)，融合 zip AC 张量从不物化。
"""
function _mapAC_zip(GL::AbstractMatrix, GR::AbstractMatrix,
                    A2ac::AbstractArray{Ta,3}, A1ac::AbstractArray{T1,3}) where {Ta,T1}
    Dx = size(GL, 1)
    D2, d = size(A2ac, 1), size(A2ac, 2)   # A2ac[c, p, e]
    D1 = size(A1ac, 1)                     # A1ac[a, p, b]
    T = promote_type(eltype(GL), eltype(GR), eltype(A2ac), eltype(A1ac))
    GL3 = reshape(GL, Dx, D1, D2)          # (xL, a, c)：融合 (c·a) 的 a 为快指标
    GR3 = reshape(GR, D1, D2, size(GR, 2)) # (b, e, xR)：融合 (e·b) 的 b 为快指标
    # 步1（键 c）：Y[a, xL, p, e] = Σ_c GL3[xL, a, c]·A2ac[c, p, e]
    @tensor Y[a, xL, p, e] := GL3[xL, a, c] * A2ac[c, p, e]      # 中间 Dx·D1·d·D2
    # 步2a（键 a；p 逐片——p 为两因子共享的开放指标，@tensor 不支持批量共享，
    #      按物理片 batched GEMM）：Z[p, b, e, xL] = Σ_a Y[a, xL, p, e]·A1ac[a, p, b]
    Y3 = reshape(Y, D1, Dx, d, D2)                               # (a, xL, p, e)
    Z = zeros(T, d, D1, D2, Dx)
    for p in 1:d
        A1p = view(A1ac, :, p, :)                                # (a, b)
        Yp = @view Y3[:, :, p, :]                                # (a, xL, e)
        # GEMM → (b, (xL, e))，重排为 (b, e, xL) 后写入 Z[p, b, e, xL]
        Z[p, :, :, :] .= reshape(permutedims(
            reshape(transpose(A1p) * reshape(Yp, D1, Dx * D2), D1, Dx, D2), (1, 3, 2)), D1, D2, Dx)
    end
    # 步2b（键 b, e）：k[xL, p, xR] = Σ Z·GR
    xR = size(GR, 2)
    Zp = reshape(permutedims(Z, (1, 4, 2, 3)), d * Dx, D1 * D2)  # (p, xL, b, e)：列 = b + (e-1)·D1
    kR = Zp * reshape(GR3, D1 * D2, xR)                           # (d·D1, xR)
    return reshape(permutedims(reshape(kR, d, Dx, xR), (2, 1, 3)), Dx, d, xR)
end

"""
    _mapC_zip(GL, C2, C1, GR) -> Matrix

因子化 zip 的 identity 通道局部 C 映射：
`Cnew[xL, xR] = Σ GL[xL, (c·a)]·C2[c, e]·C1[a, b]·GR[(e·b), xR]`.
分两步 @tensor（键 c 与键 b），最大中间张量 O(Dx·D1·D2)，`kron(C2, C1)`
从不物化。
"""
function _mapC_zip(GL::AbstractMatrix, C2::AbstractMatrix,
                   C1::AbstractMatrix, GR::AbstractMatrix)
    D2, D1 = size(C2, 1), size(C1, 1)
    Dx = size(GL, 1)
    GL3 = reshape(GL, Dx, D1, D2)          # (xL, a, c)：融合 (c·a) 的 a 为快指标
    GR3 = reshape(GR, D1, D2, size(GR, 2)) # (b, e, xR)：融合 (e·b) 的 b 为快指标
    @tensor Y[a, xL, e] := GL3[xL, a, c] * C2[c, e]              # 中间 Dx·D1·D2
    @tensor S[a, e, xR] := C1[a, b] * GR3[b, e, xR]              # 中间 D1·D2·Dx
    @tensor Cnew[xL, xR] := Y[a, xL, e] * S[a, e, xR]
    return Cnew
end

"""
    _lazy_ternary_fixedpoints(x, ket::ZipKet; tol, krylovdim, maxiter, GL0, GR0) -> (GLs, GRs)

因子化 zip target 的 identity 通道固定点：环境推直接消费 `(A1, A2)` 因子对
（[`_zip_push_left`](@ref)/[`_zip_push_right`](@ref)）。`GL`/`GR` 为
`(x 键, 融合键 (c·a)/(e·b))` 矩阵。`GL0`/`GR0` 可选：用上一轮环境热启动
eigsolve（保持不动点在逐轮之间的连续性）。
"""
function _lazy_ternary_fixedpoints(x::CanonicalIMPS, ket::ZipKet;
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
            GL = _zip_push_left(GL, x.AL[ℓ], ket.ALf2(ℓ), ket.ALf1(ℓ))
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
        GL = _zip_push_left(GL, x.AL[ℓ-1], ket.ALf2(ℓ-1), ket.ALf1(ℓ-1))
        GLs[ℓ] = GL
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da1, Dr)
        for ℓ in N:-1:1
            GR = _zip_push_right(GR, ket.ARf2(ℓ), ket.ARf1(ℓ), x.AR[ℓ])
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
        GR = _zip_push_right(GR, ket.ARf2(ℓ+1), ket.ARf1(ℓ+1), x.AR[ℓ+1])
        GRs[ℓ] = GR
    end

    # 归一化（MPSKit 约定：GR Frobenius、GL 乘局部 overlap λ）
    for ℓ in 1:N
        GRs[ℓ] = GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC_zip(GLs[inext], ket.Cf2(ℓ), ket.Cf1(ℓ), GRs[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("factorized zip environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] = GLs[inext] ./ λ
    end
    return GLs, GRs
end

"因子化 zip target 的 per-site Galerkin 残差（语义同 `_galerkin_err`）。"
function _lazy_galerkin_err(x::CanonicalIMPS, ket::ZipKet, GLs, GRs, N)
    ϵ = 0.0
    for ℓ in 1:N
        k = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k))
    end
    return ϵ
end

"因子化 zip target 的变分压缩 sweep（VOMPS/IDMRG）：无融合 zip 张量、无
gauge twist；收敛判据为 Galerkin 残差；环境逐轮热启动。"
function _lazy_sweeps(ket::ZipKet, x0::CanonicalIMPS, N::Int;
                      alg::Union{VOMPS,IDMRG}, tol::Real = Defaults.tol,
                      maxiter::Int = Defaults.maxiter,
                      verbosity::Int = Defaults.verbosity)
    isvomps = alg isa VOMPS
    T = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T, eltype(GLs[1])), x)
    ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
    for iter in 1:maxiter
        ϵ < tol && break
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
            ĉ = _mapC_zip(GLs[_mod1(ℓ + 1, N)], ket.Cf2(ℓ), ket.Cf1(ℓ), GRs[ℓ])
            if isvomps
                ALs[ℓ] = regauge!(k, ĉ; alg = Defaults.alg_orth())
            else
                # rank-1 局部问题的闭式解（同 _idmrg_sweeps）
                nk = norm(k)
                nk == 0 && error("rank-1 local map: zero vector at site $ℓ")
                ALs[ℓ] = regauge!(k ./ nk, ĉ ./ norm(ĉ); alg = Defaults.alg_orth())
            end
        end
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        GLs, GRs = _lazy_ternary_fixedpoints(x, ket; GL0 = GLs[1], GR0 = GRs[N])
        ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
        verbosity > 0 && _logiter(stdout, isvomps ? "VOMPS" : "IDMRG", iter, ϵ)
    end
    _global_normalize!(x)
    return x
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
compressed to the bond dimension `alg.D`. Unlike [`naive_mult`](@ref) (naively
constructing the whole family first, then compressing), this method **never
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
"""
mult(W, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    _mult(W, ψ, alg, nothing; D = alg.D)

function _mult(W, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
               ψ₀::Union{Nothing,CanonicalIMPS}; D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    N = length(ψ)
    # MPO-channel VOMPS/IDMRG (compute-on-the-fly, no naive target family)
    x0 = ψ₀ !== nothing ? ψ₀ : svdguess_mult(Wm, ψ, D)
    y, _ = if alg isa VOMPS
        _overlap_sweeps(Wm, ψ, x0, nothing; tol = alg.tol, maxiter = alg.maxiter,
                        verbosity = alg.verbosity)
    else
        _idmrg_sweeps(ψ, x0, nothing, Wm; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    end
    # guarantee the mixed canonical form: re-right-canonicalize from AL + C[end]
    # (preserving the ray), then normalize to the package norm convention
    y = CanonicalIMPS(collect(y.AL), y.C[end])
    return _global_normalize!(y)
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
    _mult(W, W2, alg, nothing; D = alg.D)

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
    # 规范 ⇒ 无需 gauge twist）。fallback：naive 构造 + 压缩。
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
    x = _lazy_sweeps(ket, x0, N; alg = alg, tol = alg.tol,
                     maxiter = alg.maxiter, verbosity = alg.verbosity)
    _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds)
end

# ---------------- naive_mult (debug: naive family construction + optional compression) ----------------

"""
    naive_mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPS
    naive_mult(W, W2, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPO

Naive reference implementation of [`mult`](@ref) (debug only): first construct
the complete target family (`fuse` / MPO composition, memory O(N·D₁D₂)), then
compress to `alg.D` with the positional algorithm object `alg`
(VOMPS/IDMRG). No overlap is computed (same contract as [`mult`](@ref)).
Large input bond dimensions
produce huge intermediate families — use [`mult`](@ref) for production use.
"""
function naive_mult(W, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG})
    D = alg.D
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    N = length(ψ)
    K = [fuse(Wm[ℓ], ψ.AL[ℓ]) for ℓ in 1:N]        # naive target ray (fuse of W·ψ)
    ket = CanonicalIMPS(K)                   # canonical storage of the target (for the sweeps)
    x0 = svdguess_mult(Wm, ψ, D)
    y = _mult_compress(alg, ket, x0, K)
    # guarantee the mixed canonical form: re-right-canonicalize from AL + C[end]
    # (preserving the ray), then normalize to the package norm convention
    y = CanonicalIMPS(collect(y.AL), y.C[end])
    return _global_normalize!(y)
end

function naive_mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, alg::Union{VOMPS,IDMRG})
    D = alg.D
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    N = length(W2m)
    K4 = [_naive_mul_tensor(Wm[_mod1(ℓ, length(Wm))], W2m[ℓ]) for ℓ in 1:N]
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)                             # MPS-view target of the MPO product
    ket = CanonicalIMPS(K3)
    x0 = svdguess_mult(Wm, W2m, D)
    x = _mult_compress(alg, ket, x0, K3)
    _global_normalize!(x)                           # 归一化输出（MPSKit 约定）
    return _mpo_from_mps(x, dus, dds)               # → CanonicalIMPO (re-canonicalized)
end

"Compression engine dispatch for mult: `VOMPS` → overlap-maximizing sweeps,
`IDMRG` → rank-1 local-map sweeps."
function _mult_compress(alg::VOMPS, ket, x0, K)
    x, _ = _overlap_sweeps(nothing, ket, x0, K;
                           tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return x
end
function _mult_compress(alg::IDMRG, ket, x0, K)
    x, _ = _idmrg_sweeps(ket, x0, K;
                         tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity,
                         alg_eigsolve = alg.alg_eigsolve)
    return x
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
    y = _mult(W, ψ, alg, out; D = D)
    return _copyinto!(out, y)
end

function mult!(out::CanonicalIMPO, W, W2::Union{DenseIMPO,CanonicalIMPO},
               alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    ψ0 = CanonicalIMPS(asmps_view(collect(out.AC)))
    y = _mult(W, W2, alg, ψ0; D = D)
    return _copyinto!(out, y)
end
