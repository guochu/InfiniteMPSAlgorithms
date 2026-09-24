# ---------------- iterative MPO multiplication mult (variational application / compression of MPO·MPS and MPO·MPO) ----------------
#
# Goal: given (W, x), find y ≈ W·x (x an MPS: operator application; x an MPO:
# operator composition). The naive exact construction is exact_mult in
# arithmetics.jl; this file provides the iterative (variational) versions:
# - `VOMPS`: overlap-maximizing ALS sweeps (mirrors MPSKit VOMPS,
#   src/algorithms/approximate/vomps.jl);
# - `IDMRG`: rank-1 effective-Hamiltonian eigen-solver sweeps (converging to
#   the same fixed point as VOMPS).
#
# VOMPS template (mirroring MPSKit):
# 1. ternary fixed-point environments `MultCache(x, operator, ket)`
#    (below = bra, above = ket);
# 2. localupdate: local maps (no eigen solves, unlike the groundstate VUMPS)
#    `AC_new = AC_hamiltonian(ℓ)·ket.AC[ℓ]`, `C_new = C_hamiltonian(ℓ)·ket.C[ℓ]`
#    → `regauge!(AC_new, C_new)` yields candidate `AL`s;
# 3. gauge: `gaugefix!(; order = :R)` restores the right gauge;
# 4. overlap = N·Re⟨x|B⟩/Re⟨x|x⟩ (rank-3 composite-space ring closure).

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
                   tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200,
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
                      tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200,
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

"""
    _ring_overlap(ALs, K) -> Complex
    _ring_overlap(ALs, ketf, N) -> Complex
    _ring_overlap(ketf1, ketf2, N) -> Complex

Periodic ring trace of ⟨x|K⟩: on the composite space `(x bond, K bond)`,
`𝔈_ℓ[(a′,β′),(a,β)] = Σ_p conj(x.AL[ℓ][a,p,a′])·K[ℓ][β,p,β′]`; starting from
`E = I` propagate `E ← 𝔈_ℓ·E` and return `tr(∏𝔈_ℓ)` (the x-bond ring and the
K-bond ring close independently). In the lazy methods `K[ℓ] = ketf(ℓ)` is
generated on demand (only `E` is kept; the target family is never stored).
"""
function _ring_overlap(ALs::Vector{<:AbstractArray{Tx,3}}, K::Vector{<:AbstractArray{Tk,3}}) where {Tx,Tk}
    T = promote_type(Tx, Tk)
    N = length(K)
    Dx, DK = size(ALs[1], 1), size(K[1], 1)
    E = zeros(T, Dx, DK, Dx, DK)
    for a in 1:Dx, β in 1:DK
        E[a, β, a, β] = one(T)
    end
    for ℓ in 1:N
        A = ALs[ℓ]
        B = K[ℓ]
        E = @tensor E′[a′, β′, a₀, β₀] := conj(A[a, p, a′]) * B[β, p, β′] * E[a, β, a₀, β₀]
    end
    return tr(reshape(E, Dx * DK, Dx * DK))
end

function _ring_overlap(ALs::Vector{<:AbstractArray{Tx,3}}, ketf::F, N::Int) where {Tx,F}
    T = promote_type(Tx, eltype(ketf(1)))
    Dx, DK = size(ALs[1], 1), size(ketf(1), 1)
    E = zeros(T, Dx, DK, Dx, DK)
    for a in 1:Dx, β in 1:DK
        E[a, β, a, β] = one(T)
    end
    for ℓ in 1:N
        A = ALs[ℓ]
        B = ketf(ℓ)
        E = @tensor E′[a′, β′, a₀, β₀] := conj(A[a, p, a′]) * B[β, p, β′] * E[a, β, a₀, β₀]
    end
    return tr(reshape(E, Dx * DK, Dx * DK))
end

function _ring_overlap(ketf1::F, ketf2::G, N::Int) where {F,G}
    T = eltype(ketf1(1))
    D1, D2 = size(ketf1(1), 1), size(ketf2(1), 1)
    E = zeros(T, D1, D2, D1, D2)
    for a in 1:D1, β in 1:D2
        E[a, β, a, β] = one(T)
    end
    for ℓ in 1:N
        A = ketf1(ℓ)
        B = ketf2(ℓ)
        E = @tensor E′[a′, β′, a₀, β₀] := conj(A[a, p, a′]) * B[β, p, β′] * E[a, β, a₀, β₀]
    end
    return tr(reshape(E, D1 * D2, D1 * D2))
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

"Channel overlap: `Re⟨x|operator|ket⟩` (equal per-site contractions under the
closed-channel environment; site 1 is used; `operator = nothing` means
`Re⟨x|ket⟩`). With `x` globally normalized this is the real projection of the
target onto the ray of `x`."
function _channel_overlap(x::CanonicalIMPS, operator::Union{Nothing,DenseIMPO},
                          ket::CanonicalIMPS, envs)
    O = isnothing(operator) ? nothing : operator[1]
    return real(dot(x.AC[1], _mapAC(leftenv(envs, 1), O, rightenv(envs, 1), ket.AC[1])))
end

"Overlap report: channel inner product when `K = nothing`, otherwise the
ring-trace fidelity (in [0, N])."
function _report_overlap(x::CanonicalIMPS, operator::Union{Nothing,DenseIMPO},
                         ket::CanonicalIMPS, envs,
                         K::Union{Nothing,<:Vector{<:Array}})
    if K === nothing
        return _channel_overlap(x, operator, ket, envs)
    end
    N = length(ket)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    return N * abs(_ring_overlap(xALs, K)) /
           sqrt(real(_ring_overlap(xALs, xALs)) * real(_ring_overlap(K, K)))
end

"""
    _overlap_sweeps(operator, ket, x0, K; tol, maxiter, verbosity) -> (x, overlap)

Overlap-maximizing variational sweeps (MPSKit VOMPS template): find `x`
approximating `operator|ket⟩` (with `operator = nothing`, approximating the ket
itself — variational compression). `K` is the target tensor string used for the
overlap report; with `K = nothing` the overlap is reported via the channel
inner product and no naive target family is ever generated
(compute-on-the-fly). Each round: ternary fp environments → per-site local maps
+ `regauge!` → `gaugefix!(:R)` → environment refresh + Galerkin-residual
convergence (mirroring MPSKit's `localupdate_step!`/`gauge_step!`/
`envs_step!`/`calc_galerkin` pipeline).
overlap: ring-trace fidelity in [0, N] when `K` is given (same direction ⇔
= N); otherwise the channel inner product `Re⟨x|operator|ket⟩`.
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
    overlap = _report_overlap(x, operator, ket, envs, K)
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
        envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
        ϵ = _galerkin_err(operator, ket, x, envs)
        overlap = _report_overlap(x, operator, ket, envs, K)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ, "overlap" => overlap)
    end
    _global_normalize!(x)
    return x, overlap
end

# ---------------- compression sweeps of the IDMRG template ----------------

"Rank-1 effective Hamiltonian `ℋ = 𝕀 − |k⟩⟨k|/⟨k,k⟩` (Hermitian positive
semidefinite; its :SR smallest eigenvector is uniquely the k direction, no
degeneracy)."
function _rank1_hamiltonian(k::AbstractArray{T}) where {T}
    kn2 = dot(k, k)
    kn2 == 0 && error("rank-1 effective Hamiltonian: zero vector")
    return x -> x .- (dot(k, x) / kn2) .* k
end

"""
    _idmrg_sweeps(ket, x0, K; tol, maxiter, verbosity, alg_eigsolve) -> (x, overlap)

Compression sweeps of the IDMRG template (converging to the same fixed point
as the VOMPS template of [`_overlap_sweeps`](@ref) through a different code
path — the local update uses eigen solves rather than pure maps). Under the
mixed-canonical identity channel, VOMPS's local exact solution
`k = GL·ket.AC·GR` is itself the map output; a direct eigen solve of the map
`x ↦ GL·x·GR` degenerates (the dominant eigenspace contains arbitrary physical
index combinations), so the rank-1 effective Hamiltonian
`ℋ = 𝕀 − |k⟩⟨k|/‖k‖²` is used, whose :SR smallest eigenvector is uniquely the
k direction. Each round (MPSKit template):

1. `localupdate`: `k`, `ĉ = GL₊·ket.C·GR` → `fixedpoint(ℋ, x, :SR)` solves the
   AC and C subproblems → `regauge!` yields candidate `AL`s;
2. `gauge_step!`: `gaugefix!(; order = :R)` restores the right gauge,
   `AC = AL·C`;
3. environments are recomputed;
4. convergence criterion `_galerkin_err` (tangent-space Galerkin residual).

With `operator ≠ nothing` this is the MPO-channel version
(`k = GL·O·ket.AC·GR` computed per site on the fly, compute-on-the-fly);
with `K = nothing` the overlap is reported via the channel inner product.
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
        eigs_alg = updatetol(alg_eigsolve, iter, ϵ)
        # localupdate: solve the rank-1 AC and C subproblems site by site
        # (mirrors MPSKit VUMPS localupdate_step!)
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            k = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            _, AC = fixedpoint(_rank1_hamiltonian(k), x.AC[ℓ], :SR, eigs_alg)
            ĉ = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            _, C = fixedpoint(_rank1_hamiltonian(ĉ), x.C[ℓ], :SR, eigs_alg)
            ALs[ℓ] = regauge!(AC, C; alg = Defaults.alg_orth())
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!)
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + convergence criterion (mirrors MPSKit calc_galerkin)
        envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
        ϵ = _galerkin_err(operator, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
    end
    _global_normalize!(x)
    overlap = _report_overlap(x, operator, ket, envs, K)
    return x, overlap
end

# ---------------- generic compression engine for lazy (compute-on-the-fly) targets ----------------
#
# The four canonical families (AL/AR/AC/C) of the target are generated on
# demand per site by closures (the zip / blockdiag / fuse composite constructions
# all have exact closed forms); the naive target family is never stored — the
# naive-construction memory cost drops from O(N·Dᵏ) to a single-site transient
# footprint. The environments are ternary identity-channel fixed points (left
# environments use the target's AL, right environments its AR — matching the
# gauge convention of `_ternary_fixedpoints`).

"""
    LazyKet(ALf, ARf, ACf, Cf)

Canonical families of a lazy target: four closures `f(ℓ) -> tensor` generate
the `AL/AR/AC/C` of site ℓ on demand, with semantics identical to the same
fields of `CanonicalIMPS` (`AL·C = C·AR = AC` holds pointwise under the
closed forms, e.g. `zip(AL₁,AL₂)·zip(C₁,C₂) = zip(AC₁,AC₂)`).
"""
struct LazyKet{ALf,ARf,ACf,Cf}
    ALf::ALf
    ARf::ARf
    ACf::ACf
    Cf::Cf
end

"""
    _lazy_ternary_fixedpoints(x, ket::LazyKet; tol, krylovdim, maxiter, GL0, GR0) -> (GLs, GRs)

Identity-channel ternary fixed points with the target's AL/AR tensors generated
on demand. Conventions identical to `_ternary_fixedpoints` (mirroring MPSKit:
GR Frobenius-normalized, GL scaled by the local overlap λ). `GL0`/`GR0`
optionally warm start the eigsolves with the previous environments (keeps the
fixed-point choice continuous across sweeps for block-degenerate targets).
"""
function _lazy_ternary_fixedpoints(x::CanonicalIMPS, ket::LazyKet;
                                   tol::Real = 1.0e-13, krylovdim::Int = 12,
                                   maxiter::Int = 200,
                                   GL0::Union{Nothing,AbstractArray} = nothing,
                                   GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(x)
    T = scalartype(x)
    Dl = size(x.AL[1], 1)
    Dr = size(x.AR[1], 3)
    Da1 = size(ket.ALf(1), 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, 1, Da1)
        for ℓ in 1:N
            GL = push_env_left(GL, x.AL[ℓ], ket.ALf(ℓ))
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Da1) : vec(copy(GL0))
    _, GL1 = eigsolve(Tleft, v0L, 1, :LM; ishermitian = false, tol = tol, krylovdim = krylovdim,
                      maxiter = maxiter)
    TCL = promote_type(T, eltype(GL1[1]))   # 复环境提升（MPSKit 对齐）
    GLs = Vector{Array{TCL,3}}(undef, N)
    GL = reshape(GL1[1], Dl, 1, Da1)
    GLs[1] = GL
    for ℓ in 2:N
        GL = push_env_left(GL, x.AL[ℓ-1], ket.ALf(ℓ-1))
        GLs[ℓ] = GL
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da1, 1, Dr)
        for ℓ in N:-1:1
            GR = push_env_right(GR, ket.ARf(ℓ), x.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da1 * Dr) : vec(copy(GR0))
    _, GRN = eigsolve(Tright, v0R, 1, :LM; ishermitian = false, tol = tol, krylovdim = krylovdim,
                      maxiter = maxiter)
    TCR = promote_type(T, eltype(GRN[1]))
    GRs = Vector{Array{TCR,3}}(undef, N)
    GR = reshape(GRN[1], Da1, 1, Dr)
    GRs[N] = GR
    for ℓ in N-1:-1:1
        GR = push_env_right(GR, ket.ARf(ℓ+1), x.AR[ℓ+1])
        GRs[ℓ] = GR
    end

    # normalization (mirroring MPSKit: GR Frobenius-normalized, GL scaled by the
    # local overlap λ)
    for ℓ in 1:N
        GRs[ℓ] .= GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC(GLs[inext], GRs[ℓ], ket.Cf(ℓ))
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("lazy ternary environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] .= GLs[inext] ./ λ
    end
    return GLs, GRs
end

"Maximum per-site Galerkin residual of the lazy target (AC families, consistent
with `_galerkin_err`). Multi-target form: the local map is the **sum** over the
targets' partial maps (wavefunction sum; each channel has its own
non-degenerate environments)."
function _lazy_galerkin_err(x::CanonicalIMPS, kets::Vector{<:LazyKet},
                            GLset::Vector, GRset::Vector, N::Int)
    ϵ = 0.0
    for ℓ in 1:N
        k = reduce(+, (_mapAC(GLset[j][ℓ], nothing, GRset[j][ℓ], kets[j].ACf(ℓ))
                       for j in eachindex(kets)))
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k))
    end
    return ϵ
end

"Ring-trace fidelity of the lazy target
`N·|⟨x|ket⟩|/√(⟨x|x⟩⟨ket|ket⟩)` ∈ [0, N] (same direction ⇔ = N); same formula
as the K version of `_report_overlap`, with the target tensors generated on
demand by the closures. Multi-target form: the overlaps add linearly over the
targets, and `⟨ket|ket⟩` includes the pairwise ring overlaps."
function _lazy_overlap(x::CanonicalIMPS, kets::Vector{<:LazyKet}, N::Int)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    num = sum(_ring_overlap(xALs, kets[j].ALf, N) for j in eachindex(kets))
    norm2 = sum(_ring_overlap(kets[i].ALf, kets[j].ALf, N)
                for i in eachindex(kets), j in eachindex(kets))
    return N * abs(num) / sqrt(real(_ring_overlap(xALs, xALs)) * real(norm2))
end

function _lazy_overlap(x::CanonicalIMPS, ket::LazyKet, N::Int)
    return _lazy_overlap(x, [ket], N)
end

"`(X^{1/2}, X^{-1/2})` of a (numerically) positive definite matrix: phase-align
by tr first (the overall phase of the fixed point is arbitrary), Hermitianize,
and clip negative eigenvalues (round-off noise, or the not-fully-converged
components of a degenerate fixed-point space — a clipped approximate fp still
yields an effective twist whose residual the ALS absorbs)."
function _pd_sqrt_invsqrt(X::AbstractMatrix{T}) where {T}
    trx = tr(X)
    abs(trx) > sqrt(eps(real(T))) * norm(X) ||
        error("lazy gauge twist: gram fixed point has tr ≈ 0; phase undeterminable")
    Xh = (X * (conj(trx) / abs(trx)) + X' * (trx / abs(trx))) / 2
    vals, vecs = eigen(Hermitian(Xh))
    tol = sqrt(eps(real(T))) * maximum(abs, vals)
    rt = sqrt.(max.(vals, tol))   # clip negative eigenvalues to a tiny positive floor
    return vecs * Diagonal(rt) * vecs', vecs * Diagonal(inv.(rt)) * vecs'
end

"""
    _twist_lazy_ket(ket::LazyKet, N) -> LazyKet

Effective gauge twist of the lazy target: solve the gram fixed point
`X_{ℓ+1} = ALf(ℓ)†·X_ℓ·ALf(ℓ)` (dominant eigenpair of the periodic product
transfer, positive definite) and twist the four family closures as
`X_ℓ^{1/2}·f(ℓ)·X_{ℓ+1}^{-1/2}` (C as `X_{ℓ+1}^{1/2}·C·X_{ℓ+1}^{-1/2}`) into
the **effective mixed-canonical form**.

For non-canonical composite constructions (the shared-physical-index sums of
zip / fuse do not factorize, so the left/right orthogonality sums fail), the
ALS fixed points of the identity-channel machinery get distorted by the gram
twist — after the twist the transfer fixed points return to the identity, and
`AL·C = C·AR = AC` is preserved pointwise under the twist (the intermediate
`X^{±1/2}` factors cancel). For blockdiag (direct sums) `X = 𝕀` and the twist
is trivial. `X` is one D×D matrix per bond, memory O(N·D²), same order as the
environments — still compute-on-the-fly.
"""
function _twist_lazy_ket(ket::LazyKet, N::Int)
    Dk = size(ket.ALf(1), 1)
    T = eltype(ket.ALf(1))
    # gram fixed point: dominant eigenvector of the periodic product transfer
    # X_{ℓ+1} = ALf(ℓ)†·X_ℓ·ALf(ℓ)
    Tgram = function (v::AbstractVector)
        Xs = reshape(v, Dk, Dk, N)
        out = similar(Xs)
        for ℓ in 1:N
            A = ket.ALf(ℓ)
            Xl = Xs[:, :, ℓ]
            @tensor tmp[bl, br] := conj(A[bm, s, bl]) * Xl[bm, bp] * A[bp, s, br]
            out[:, :, _mod1(ℓ + 1, N)] = tmp
        end
        return vec(out)
    end
    _, X1 = eigsolve(Tgram, ones(T, N * Dk * Dk), 1, :LM; ishermitian = false,
                     tol = 1.0e-13, krylovdim = max(12, N * 2), maxiter = 200)
    Xs = reshape(X1[1], Dk, Dk, N)
    # Polish with a fixed-point iteration of the CP map: the gram transfer is a
    # direct sum for blockdiag-type targets, so the fp space is degenerate and
    # eigsolve may return a not-fully-converged vector inside it (which is not
    # positive definite after Hermitianization). The power iteration preserves
    # Hermiticity/PSD and exponentially suppresses the non-fp components.
    for _ in 1:200
        Xn = reshape(Tgram(vec(Xs)), Dk, Dk, N)
        λ = dot(vec(Xs), vec(Xn)) / dot(Xs, Xs)
        res = norm(vec(Xn) .- λ .* vec(Xs)) / norm(Xs)
        Xs = Xn ./ norm(Xn)
        res < 1.0e-12 && break
    end
    half = Vector{Matrix{T}}(undef, N + 1)
    ihalf = Vector{Matrix{T}}(undef, N + 1)
    for ℓ in 1:N
        half[ℓ], ihalf[ℓ] = _pd_sqrt_invsqrt(Xs[:, :, ℓ])
    end
    inext(ℓ) = _mod1(ℓ + 1, N)
    twist = (g, A, gi) -> @tensor B[bl, s, br] := g[bl, bm] * A[bm, s, bp] * gi[bp, br]
    return LazyKet(
        ℓ -> twist(half[ℓ], ket.ALf(ℓ), ihalf[inext(ℓ)]),
        ℓ -> twist(half[ℓ], ket.ARf(ℓ), ihalf[inext(ℓ)]),
        ℓ -> twist(half[ℓ], ket.ACf(ℓ), ihalf[inext(ℓ)]),
        ℓ -> half[inext(ℓ)] * ket.Cf(ℓ) * ihalf[inext(ℓ)],
    )
end

"""
    _lazy_sweeps(kets::Vector{<:LazyKet}, x0, N; alg, tol, maxiter, verbosity) -> (x, overlap)
    _lazy_sweeps(ket::LazyKet, x0, N; kwargs...) -> (x, overlap)

Generic compression engine for lazy (compute-on-the-fly) targets, shared by
the VOMPS and IDMRG templates: each target's AL/AR/AC/C families are generated
on demand per site by its closures; the local map
`k = Σ_j GL_j·kets[j].ACf(ℓ)·GR_j` sums the per-target partial maps (the
wavefunction sum — each channel `⟨x|ket_j⟩` has its own non-degenerate
environments, so block-degenerate sums such as `add` stay reliable); the naive
target families are never materialized.
overlap = ring-trace fidelity in [0, N] (= N means same direction).
"""
function _lazy_sweeps(kets::Vector{<:LazyKet}, x0::CanonicalIMPS, N::Int;
                      alg::Union{VOMPS,IDMRG}, tol::Real = Defaults.tol,
                      maxiter::Int = Defaults.maxiter,
                      verbosity::Int = Defaults.verbosity)
    isvomps = alg isa VOMPS
    kets = [_twist_lazy_ket(ket, N) for ket in kets]   # effective gauge twists
    T = promote_type(scalartype(x0), eltype(kets[1].ACf(1)))
    x = copy(x0)
    fps = [_lazy_ternary_fixedpoints(x, ket) for ket in kets]
    GLset = [fp[1] for fp in fps]
    GRset = [fp[2] for fp in fps]
    # 通道标量类型（MPSKit 对齐，见 _overlap_sweeps 注释）
    T = promote_type(T, eltype(GLset[1][1]))
    x = _promote_scalar(T, x)
    ϵ = _lazy_galerkin_err(x, kets, GLset, GRset, N)
    overlap = _lazy_overlap(x, kets, N)
    for iter in 1:maxiter
        ϵ < tol && break
        eigs_alg = isvomps ? nothing : updatetol(alg.alg_eigsolve, iter, ϵ)
        # localupdate: per-site local maps (VOMPS) or rank-1 eigen solves (IDMRG)
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = reduce(+, (_mapAC(GLset[j][ℓ], nothing, GRset[j][ℓ], kets[j].ACf(ℓ))
                           for j in eachindex(kets)))
            ĉ = reduce(+, (_mapC(GLset[j][_mod1(ℓ + 1, N)], GRset[j][ℓ], kets[j].Cf(ℓ))
                           for j in eachindex(kets)))
            if isvomps
                ALs[ℓ] = regauge!(k, ĉ; alg = Defaults.alg_orth())
            else
                _, AC = fixedpoint(_rank1_hamiltonian(k), x.AC[ℓ], :SR, eigs_alg)
                _, C = fixedpoint(_rank1_hamiltonian(ĉ), x.C[ℓ], :SR, eigs_alg)
                ALs[ℓ] = regauge!(AC, C; alg = Defaults.alg_orth())
            end
        end
        # gauge: restore the global right gauge (mirrors MPSKit gauge_step!)
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + calc_galerkin
        fps = [_lazy_ternary_fixedpoints(x, ket) for ket in kets]
        GLset = [fp[1] for fp in fps]
        GRset = [fp[2] for fp in fps]
        ϵ = _lazy_galerkin_err(x, kets, GLset, GRset, N)
        overlap = _lazy_overlap(x, kets, N)
        verbosity > 0 && _logiter(stdout, isvomps ? "VOMPS" : "IDMRG", iter, ϵ,
                                  "overlap" => overlap)
    end
    _global_normalize!(x)
    return x, overlap
end

function _lazy_sweeps(ket::LazyKet, x0::CanonicalIMPS, N::Int; kwargs...)
    return _lazy_sweeps([ket], x0, N; kwargs...)
end

"""
    _lazy_mpo_result(fams::Vector{<:Tuple}, x0, dus, dds, N; alg) -> (y, overlap)

Lazy assembly of MPO algebra results: each element of `fams` is a tuple
`(fAL, fAR, fAC, fC)` of rank-4 closures generating one target's AL/AR/AC/C
families per site on demand (entering [`_lazy_sweeps`](@ref) through the MPS
view; the local maps add up over the targets); after compression +
ring-trace amplitude alignment the result is mapped back to rank-4 canonical
storage (`CanonicalIMPO`) via [`_mpo_from_mps`](@ref).
"""
function _lazy_mpo_result(fams::Vector{<:Tuple}, x0::CanonicalIMPS,
                          dus::AbstractVector{Int}, dds::AbstractVector{Int},
                          N::Int; alg::Union{VOMPS,IDMRG})
    kets = [LazyKet(ℓ -> asmps_view([fam[1](ℓ)])[1],
                    ℓ -> asmps_view([fam[2](ℓ)])[1],
                    ℓ -> asmps_view([fam[3](ℓ)])[1],
                    fam[4]) for fam in fams]
    x, overlap = _lazy_sweeps(kets, x0, N; alg = alg, tol = alg.tol,
                              maxiter = alg.maxiter, verbosity = alg.verbosity)
    _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds), overlap
end

function _lazy_mpo_result(fAL, fAR, fAC, fC, x0::CanonicalIMPS,
                          dus::AbstractVector{Int}, dds::AbstractVector{Int},
                          N::Int; alg::Union{VOMPS,IDMRG})
    return _lazy_mpo_result([(fAL, fAR, fAC, fC)], x0, dus, dds, N; alg = alg)
end

"""
    mult(W, ψ) -> (y::CanonicalIMPS, overlap)
    mult(W, W2) -> (y::CanonicalIMPO, overlap)

Exact application/composition without compression: the naive construction
(fuse / MPO composition) is canonicalized into mixed-canonical storage. The
output bond dimension is the naive bond dimension (inherently large for large
inputs); `overlap = N` identically (the output is the target ray itself).
"""
function mult(W, ψ::CanonicalIMPS)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    N = length(ψ)
    T = promote_type(scalartype(Wm), scalartype(ψ))
    K = [fuse(Wm[ℓ], ψ.AL[ℓ]) for ℓ in 1:N]
    y = CanonicalIMPS(collect(K))
    return _global_normalize!(y), real(T)(N)
end

"""
    mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPS, overlap)
    mult(W, W2, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPO, overlap)

The compute-on-the-fly version of the MPO multiplication: find `y ≈ W·ψ`
(operator application) or `y ≈ W·W2` (operator composition), variationally
compressed to the bond dimension `alg.D`. Unlike [`naive_mult`](@ref) (naively
constructing the whole family first, then compressing), this method **never
materializes the naive target family**: the
local maps `k = GL·W·ket·GR` are computed per site on the fly under the
environments (MPO channel / lazy fuse target), with intermediate memory of
O(single site) only. Applying a time-evolution MPO is
`ψ′, _ = mult(make_time_mpo(H, dt, WII()), ψ)`.

- `W`: an `DenseIMPO`, an `SparseIMPO` (densified into an `DenseIMPO`
  before application), or an `CanonicalIMPO`;
- `alg.D::Int`: target bond dimension of the variational compression
  (deterministic `svdguess_mult` initial state);
- both `alg` types share the same fixed point;
- the output is guaranteed to be in mixed-canonical form (mpo·mps →
  `CanonicalIMPS`, mpo·mpo → `CanonicalIMPO`); `overlap` is the
  ring-trace fidelity in [0, N] (= N means same direction);
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
    _global_normalize!(y)
    # overlap = ring-trace fidelity in [0, N] (the target fuse tensors are
    # generated on demand; the naive family is not materialized)
    yALs = [y.AL[ℓ] for ℓ in 1:N]
    fusel = ℓ -> fuse(Wm[ℓ], ψ.AL[ℓ])
    overlap = N * abs(_ring_overlap(yALs, fusel, N)) /
              sqrt(real(_ring_overlap(yALs, yALs)) * real(_ring_overlap(fusel, fusel, N)))
    return y, overlap
end

function mult(W, W2::Union{DenseIMPO,CanonicalIMPO})
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    N = length(W2m)
    NW = length(Wm)
    T = promote_type(scalartype(Wm), scalartype(W2m))
    # exact composition: naive fuse family → canonical storage (the output is
    # the target ray itself → fidelity = N)
    K4 = [_naive_mul_tensor(Wm[_mod1(ℓ, NW)], W2m[ℓ]) for ℓ in 1:N]
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x = CanonicalIMPS(collect(K3))
    _global_normalize!(x)
    y = _mpo_from_mps(x, dus, dds)
    return y, real(T)(N)
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
    # VOMPS/IDMRG on the lazy fuse target (compute-on-the-fly).
    # The four family closures of the composite target (fuse preserves the
    # factor canonical-form consistency: `AL·C = C·AR = AC` pointwise); left
    # environments use AL, right environments AR (matching the
    # `_ternary_fixedpoints` gauge convention).
    W1c = W isa CanonicalIMPO ? W : CanonicalIMPO(collect(Wm.Ws))
    W2c = W2 isa CanonicalIMPO ? W2 : CanonicalIMPO(collect(W2m.Ws))
    dus = [size(W1c.AL[_mod1(ℓ, NW)], 2) for ℓ in 1:N]
    dds = [size(W2c.AL[ℓ], 4) for ℓ in 1:N]
    physdims = dus .* dds
    x0 = ψ₀ !== nothing ? ψ₀ : svdguess_mult(Wm, W2m, D)
    f = (fam1, fam2) -> ℓ -> _naive_mul_tensor(fam1[_mod1(ℓ, NW)], fam2[ℓ])
    # the C bond order matches the column-major reshape: composite bond
    # (wl2 major, wl1 minor)
    # lazy fuse engine + naive fallback (see _lazy_or_fallback; the naive family
    # is only materialized when the fallback fires)
    return _lazy_or_fallback(
        () -> _lazy_mpo_result(f(W1c.AL, W2c.AL), f(W1c.AR, W2c.AR), f(W1c.AC, W2c.AC),
                               ℓ -> kron(W2c.C[ℓ], W1c.C[_mod1(ℓ, NW)]),
                               x0, dus, dds, N; alg = alg),
        () -> _compress_mpo_result([_naive_mul_tensor(Wm[_mod1(ℓ, NW)], W2m[ℓ]) for ℓ in 1:N],
                                   D, alg),
        N)
end

"Fallback assembly of the lazy mpo·mpo path: naive construction + compression,
always returning `(result, overlap)`; the result is mixed-canonical
(`CanonicalIMPO`, satisfying `ismixedcanonical`)."
function _compress_mpo_result(K4::Vector{<:Array{T,4}}, D::Int, alg::Algorithm) where {T}
    naive = DenseIMPO(K4)
    dmax = max_bonddim(naive)
    D ≥ dmax && return CanonicalIMPO(collect(naive.Ws)), real(T)(length(K4))
    N = length(K4)
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x, overlap = _compress_ket(K3, [dus[ℓ] * dds[ℓ] for ℓ in 1:N], D, alg)
    _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds), overlap
end

# ---------------- naive_mult (debug: naive family construction + optional compression) ----------------

"""
    naive_mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> (y, overlap)
    naive_mult(W, W2, alg::Union{VOMPS,IDMRG}) -> (y, overlap)

Naive reference implementation of [`mult`](@ref) (debug only): first construct
the complete target family (`fuse` / MPO composition, memory O(N·D₁D₂)), then
compress to `alg.D` with the positional algorithm object `alg`
(VOMPS/IDMRG). `overlap` is the ring-trace
fidelity in [0, N] (= N means same direction). Large input bond dimensions
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
    y, overlap = _mult_compress(alg, ket, x0, K)
    # guarantee the mixed canonical form: re-right-canonicalize from AL + C[end]
    # (preserving the ray), then normalize to the package norm convention
    y = CanonicalIMPS(collect(y.AL), y.C[end])
    _global_normalize!(y)
    return y, overlap
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
    physdims = [size(K3[ℓ], 2) for ℓ in 1:N]
    x0 = svdguess_mult(Wm, W2m, D)
    x, overlap = _mult_compress(alg, ket, x0, K3)
    _global_normalize!(x)                           # 归一化输出（MPSKit 约定）
    y = _mpo_from_mps(x, dus, dds)                  # → CanonicalIMPO (re-canonicalized)
    return y, overlap
end

"Compression engine dispatch for mult: `VOMPS` → overlap-maximizing sweeps,
`IDMRG` → eigen-solver sweeps."
function _mult_compress(alg::VOMPS, ket, x0, K)
    return _overlap_sweeps(nothing, ket, x0, K;
                           tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
end
function _mult_compress(alg::IDMRG, ket, x0, K)
    return _idmrg_sweeps(ket, x0, K;
                         tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity,
                         alg_eigsolve = alg.alg_eigsolve)
end

# ---------------- svdguess_mult (deterministic initial guess) & mult! (in-place) ----------------

"""
    svdguess_mult(W, ψ, D) -> CanonicalIMPS
    svdguess_mult(W, W2, D) -> CanonicalIMPS

Deterministic initial guess of the iterative [`mult`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_mult`): the naive fuse/composition target (the
same tensor string as [`exact_mult`](@ref)) followed by the bond-wise SVD
truncation to `D` (the truncation factors act on orthogonality-protected
bonds, preserving the mixed-canonical form).
"""
function svdguess_mult(W, ψ::CanonicalIMPS, D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    K = [fuse(Wm[ℓ], ψ.AL[ℓ]) for ℓ in 1:length(ψ)]
    x = CanonicalIMPS(collect(K))
    return max_bonddim(x) ≤ D ? x : _truncate_bonddim(x, D)
end

function svdguess_mult(W, W2::Union{DenseIMPO,CanonicalIMPO}, D::Int)
    Wm = W isa DenseIMPO ? W : DenseIMPO(W)
    W2m = W2 isa DenseIMPO ? W2 : DenseIMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    K4 = [_naive_mul_tensor(Wm[_mod1(ℓ, length(Wm))], W2m[ℓ]) for ℓ in 1:length(W2m)]
    x = CanonicalIMPS(asmps_view(K4))
    return max_bonddim(x) ≤ D ? x : _truncate_bonddim(x, D)
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

function mult!(out::CanonicalIMPO, W, W2::Union{DenseIMPO,CanonicalIMPO},
               alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    ψ0 = CanonicalIMPS(asmps_view(collect(out.AC)))
    y, _ = _mult(W, W2, alg, ψ0; D = D)
    return _copyinto!(out, y)
end
