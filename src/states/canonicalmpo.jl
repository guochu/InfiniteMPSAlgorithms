"""
    CanonicalIMPO{T}

Mixed-canonical storage of an MPO with the same layout as
[`CanonicalIMPS`](@ref) (treating the MPO as an MPS; useful e.g. for
density matrices; storage and usage rules follow `CanonicalIMPS`):

- `AL[ℓ]::Array{T,4}`: MPO tensor `(wl, u, wr, d)`, left-orthogonal in the MPS view;
- `AR[ℓ]::Array{T,4}`: right-orthogonal in the MPS view;
- `C[ℓ]::Array{T,2}`: center matrix on bond ℓ;
- `AC[ℓ]::Array{T,4}`: center-canonical MPO tensor,
  `AC[ℓ] = AL[ℓ]·C[ℓ] = C[ℓ-1]·AR[ℓ]` (in the MPS view `(wl, u*d, wr)`;
  conventions identical to `CanonicalIMPS`).

Purpose: run MPS algorithms on MPOs (the dominant eigenvector is the optimal
smaller-bond approximation → MPO compression; canonical storage of
`hadamard(ψ, dag(ψ))`-type density matrices).
MPS view: `(wl, u, wr, d)` → permute `(1,2,4,3)` → reshape `(wl, u*d, wr)`.

Note: construction mixed-canonicalizes via `CanonicalIMPS`'s
`gaugefix!`. Gauge transformations telescope away in the periodic trace
representation (`tr(C⁻¹·X·C) = tr(X)`), so the operator's periodic
contraction is preserved up to the forced normalization scale of the MPS
view (`‖AC[1]‖ = 1` is not part of the gauge freedom) — the ray is exact.
"""
struct CanonicalIMPO{T}
    AL::PeriodicVector{Array{T,4}}
    AR::PeriodicVector{Array{T,4}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,4}}

    function CanonicalIMPO{T}(AL::PeriodicVector{Array{T,4}},
                                     AR::PeriodicVector{Array{T,4}},
                                     C::PeriodicVector{Array{T,2}},
                                     AC::PeriodicVector{Array{T,4}}) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            ℓ1 = _mod1(ℓ + 1, L)
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AC[ℓ], 2) == size(AC[ℓ], 4) ||
                throw(DimensionMismatch("physical (u,d) mismatch at site $ℓ"))
        end
        new{T}(AL, AR, C, AC)
    end
end

"""
    _mpo_from_mps(ψ::CanonicalIMPS, dus, dds) -> CanonicalIMPO

[`CanonicalIMPS`](@ref) → [`CanonicalIMPO`](@ref): the rank-3
families are mapped back to rank-4 via the inverse view transform
[`mps_view_to_mpo`](@ref) (`dus`/`dds` give the u/d physical dimensions per site).
"""
function _mpo_from_mps(ψ::CanonicalIMPS{T}, dus::AbstractVector{Int},
                       dds::AbstractVector{Int}) where {T}
    to4 = As -> mps_view_to_mpo(collect(As); dus = dus, dds = dds)
    return CanonicalIMPO{T}(PeriodicVector(to4(ψ.AL)), PeriodicVector(to4(ψ.AR)),
                                   copy(ψ.C), PeriodicVector(to4(ψ.AC)))
end

"""
    CanonicalIMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...)

Construct from plain MPO tensors (mirrors `CanonicalIMPS(As)`): convert
to an MPS via `asmps_view`, then mixed-canonicalize with `gaugefix!`
(the operator ray is preserved exactly; see the type docstring for the
normalization-scale caveat).
"""
function CanonicalIMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...) where {T}
    N = length(Ws)
    ψ = CanonicalIMPS(asmps_view(Ws); kwargs...)
    return _mpo_from_mps(ψ, [size(Ws[ℓ], 2) for ℓ in 1:N], [size(Ws[ℓ], 4) for ℓ in 1:N])
end

CanonicalIMPO(W::DenseIMPO; kwargs...) = CanonicalIMPO(W.Ws; kwargs...)

# ---------------- interface (mirrors CanonicalIMPS) ----------------

Base.length(W::CanonicalIMPO) = length(W.AL)
Base.size(W::CanonicalIMPO, args...) = size(W.AL, args...)
Base.getindex(W::CanonicalIMPO, ℓ::Integer) = W.AC[_mod1(ℓ, length(W))]
Base.setindex!(W::CanonicalIMPO, v::Array, ℓ::Integer) = (W.AC[ℓ] = v; W)
Base.firstindex(W::CanonicalIMPO) = 1
Base.lastindex(W::CanonicalIMPO) = length(W)
Base.iterate(W::CanonicalIMPO, args...) = iterate(W.AC, args...)
eachsite(W::CanonicalIMPO) = 1:length(W)

function Base.copy(W::CanonicalIMPO{T}) where {T}
    return CanonicalIMPO{T}(PeriodicVector([copy(a) for a in W.AL]),
                                   PeriodicVector([copy(a) for a in W.AR]),
                                   PeriodicVector([copy(c) for c in W.C]),
                                   PeriodicVector([copy(a) for a in W.AC]))
end
function Base.similar(W::CanonicalIMPO{T}) where {T}
    return CanonicalIMPO{T}(similar(W.AL), similar(W.AR), similar(W.C), similar(W.AC))
end
function Base.circshift(W::CanonicalIMPO, n)
    return CanonicalIMPO{T}(circshift(W.AL, n), circshift(W.AR, n),
                                   circshift(W.C, n), circshift(W.AC, n))
end

scalartype(::Type{CanonicalIMPO{T}}) where {T} = T
scalartype(W::CanonicalIMPO) = scalartype(typeof(W))

phydims(W::CanonicalIMPO) =
    [size(W.AL[ℓ], 2) * size(W.AL[ℓ], 4) for ℓ in 1:length(W)]
bonddim(W::CanonicalIMPO, ℓ::Integer) = size(W.C[_mod1(ℓ, length(W))], 1)
max_bonddim(W::CanonicalIMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"`dag(W)`: elementwise conjugation of every tensor (for overlap-type
contractions; not the operator-adjoint network)."
dag(W::CanonicalIMPO{T}) where {T} =
    CanonicalIMPO{T}(PeriodicVector(conj.(parent(W.AL))), PeriodicVector(conj.(parent(W.AR))),
                            PeriodicVector(conj.(parent(W.C))), PeriodicVector(conj.(parent(W.AC))))

"`LinearAlgebra.norm(W) = norm(W.AC[1])` (consistent with CanonicalIMPS)."
LinearAlgebra.norm(W::CanonicalIMPO) = norm(W.AC[1])

"""
    LinearAlgebra.normalize!(W) -> W

Mirror of MPSKit's `normalize!(ψ::InfiniteMPS)` on the MPS view: every bond
matrix `C[ℓ]` and every center tensor `AC[ℓ]` is normalized to unit Frobenius
norm. For a mixed-canonical operator the two normalizations coincide
(`‖AL·C‖ = ‖C‖` for left-orthogonal `AL`), so this is exactly the ring
normalization `⟨W, W⟩ = 1` of the MPS view.
"""
function LinearAlgebra.normalize!(W::CanonicalIMPO)
    normalize!.(W.C)
    normalize!.(W.AC)
    return W
end

"`DenseIMPO(W)`: convert back to a plain MPO using the left-canonical tensor
string `W.AL`. `tr(∏AL)` is the operator amplitude invariant under gauge
transformations (including phases) (= the construction input amplitude / a
positive real λ), whereas `tr(∏AC)` is C-matrix weighted and gauge dependent;
`AL` is used here to keep the conversion unique."
DenseIMPO(W::CanonicalIMPO) = DenseIMPO(collect(W.AL))

"""
    asmps_view(Ws::Vector{<:Array{T,4}}) -> Vector{Array{T,3}}

MPS view of an MPO tensor string: `(wl, u, wr, d)` → `(wl, u*d, wr)`.
"""
function asmps_view(Ws::Vector{<:Array{T,4}}) where {T}
    out = Vector{Array{T,3}}(undef, length(Ws))
    for (ℓ, W) in enumerate(Ws)
        wl, u, wr, d = size(W)
        out[ℓ] = reshape(permutedims(W, (1, 2, 4, 3)), wl, u * d, wr)
    end
    return out
end
asmps_view(W::DenseIMPO) = asmps_view(W.Ws)
asmps_view(W::CanonicalIMPO) = asmps_view(collect(W.AC))

"""
    mps_view_to_mpo(As::Vector{<:Array{T,3}}; dus, dds) -> Vector{Array{T,4}}

Inverse of [`asmps_view`](@ref): `(wl, u*d, wr)` → `(wl, u, wr, d)`.
`dus`/`dds` give the u/d physical dimensions per site.
"""
function mps_view_to_mpo(As::Vector{<:Array{T,3}}; dus::AbstractVector{Int}, dds::AbstractVector{Int}) where {T}
    length(As) == length(dus) == length(dds) || throw(DimensionMismatch())
    out = Vector{Array{T,4}}(undef, length(As))
    for (ℓ, A) in enumerate(As)
        wl, p, wr = size(A)
        (p == dus[ℓ] * dds[ℓ]) || throw(DimensionMismatch("physical dimension mismatch"))
        out[ℓ] = permutedims(reshape(A, wl, dus[ℓ], dds[ℓ], wr), (1, 2, 4, 3))
    end
    return out
end

# ---------------- operator-algebra transforms (vectorize / devectorize / superoperator) ----------------

"""
    vectorize(W::CanonicalIMPO) -> CanonicalIMPS
    vectorize(W::Union{DenseIMPO,SparseIMPO}) -> CanonicalIMPS

Vectorize an MPO into an MPS on the doubled (bra ⊗ ket) space: the two
physical legs of every tensor are fused into one composite index

    f = u + du·(d - 1)

(`u` the bra / operator-row leg is the fast index — the same MPS view as
[`asmps_view`](@ref), i.e. the row-major vectorization of the operator
matrix). The bond dimensions and the mixed-canonical gauges are carried over
verbatim, so the `CanonicalIMPO` conversion is exact (a pure reshape), and
`dot(vectorize(A), vectorize(B))` is the Hilbert–Schmidt inner product of the
operators. A `DenseIMPO`/`SparseIMPO` input is mixed-canonicalized first
(the operator value is preserved exactly in the periodic trace
representation).
"""
function vectorize(W::CanonicalIMPO)
    return CanonicalIMPS(PeriodicVector(asmps_view(collect(W.AL))),
                         PeriodicVector(asmps_view(collect(W.AR))),
                         copy(W.C),
                         PeriodicVector(asmps_view(collect(W.AC))))
end
vectorize(W::DenseIMPO) = vectorize(CanonicalIMPO(W))
vectorize(W::SparseIMPO) = vectorize(CanonicalIMPO(DenseIMPO(W)))

"""
    devectorize(ψ::CanonicalIMPS) -> CanonicalIMPO

Inverse of [`vectorize`](@ref): split the fused physical index `f` of every
site tensor back into the bra/ket pair `(u, d)`. All physical dimensions must
be equal perfect squares (square operators); the conversion is exact.
"""
function devectorize(ψ::CanonicalIMPS)
    N = length(ψ)
    p = size(ψ.AL[1], 2)
    r = isqrt(p)
    r^2 == p || throw(ArgumentError("physical dimension $p is not a perfect square"))
    for ℓ in 2:N
        size(ψ.AL[ℓ], 2) == p ||
            throw(ArgumentError("inhomogeneous physical dimensions at site $ℓ"))
    end
    return _mpo_from_mps(ψ, fill(r, N), fill(r, N))
end

"""
    superoperator(W; side = :left) -> DenseIMPO

The left/right multiplication superoperator of an MPO, as an MPO on the
doubled (bra ⊗ ket) space with the fused index convention of
[`vectorize`](@ref) (`f = u + du·(d - 1)`, `u` the bra / row leg fast):

- `side = :left` (`= kron(W, identityimpo(dus))`): `𝓦[bl, f', br, f] = W[bl, u', br, u]·δ[d', d]`,
  so that `𝓦 · vec(X)` is `vec(W·X)` (W multiplies from the left);
- `side = :right` (`= kron(identityimpo(dus), transpose(W))`): `𝓦[bl, f', br, f] = δ[u', u]·W[bl, d, br, d']`,
  so that `𝓦 · vec(X)` is `vec(X·W)` (W multiplies from the right).

The bond dimensions are unchanged (the spectator channel carries trivial
δ-bonds) and the physical dimension becomes `du·dd` (square operators only).
The output is a plain `DenseIMPO` (not canonical). Combined with
[`vectorize`](@ref) this turns operator–operator products into operator–state
problems, e.g. for `mult`:

    mult(superoperator(W1; side = :left),  vectorize(W2)) == vectorize(W1 * W2)
    mult(superoperator(W2; side = :right), vectorize(W1)) == vectorize(W1 * W2)

A typical finite-T purification generator is the sum of the two channel
superoperators, `𝓦_L(H) + 𝓦_R(H)` (= `H ⊗ I + I ⊗ Hᵀ`).
"""
function superoperator(W::DenseIMPO; side::Symbol = :left)
    dus = phydims(W)
    for ℓ in 1:length(W)
        size(W[ℓ], 4) == dus[ℓ] ||
            throw(ArgumentError("superoperator requires square operators (u == d) at site $ℓ"))
    end
    I = identityimpo(scalartype(W), dus)
    side === :left && return kron(W, I)
    side === :right && return kron(I, transpose(W))
    throw(ArgumentError("side must be :left or :right, got $side"))
end
superoperator(W::SparseIMPO; side::Symbol = :left) = superoperator(DenseIMPO(W); side)
superoperator(W::CanonicalIMPO; side::Symbol = :left) = superoperator(DenseIMPO(W); side)

"""
    fidelity(W₁, W₂) -> Real
    infidelity(W₁, W₂) -> Real

Hilbert–Schmidt fidelity of two operators stored as [`CanonicalIMPO`](@ref):
`|⟨W₁, W₂⟩_HS| / (‖W₁‖·‖W₂‖) ∈ [0, 1]`, computed as the
[`fidelity`](@ref) of the vectorized states (the ring overlap of the MPS
views is the C-weighted operator inner product). Invariant under overall
phases and scalings; `infidelity = 1 − fidelity`.
"""
fidelity(W₁::CanonicalIMPO, W₂::CanonicalIMPO) = fidelity(vectorize(W₁), vectorize(W₂))
infidelity(W₁::CanonicalIMPO, W₂::CanonicalIMPO) = 1 - fidelity(W₁, W₂)

"""
    mpo_compress(W::DenseIMPO, D; tol=1e-10, maxiter=100, verbosity=0) -> (; W, overlap)

Variationally compress an MPO to bond dimension `D`: view the MPO as an MPS
(`asmps_view`) and run VOMPS overlap-maximization sweeps on the identity
channel (equivalent to the bond-`D` variational approximation of the dominant
eigenvector of the double-layer transfer `W⊗W̄`). The output is the
**normalized** compressed state mapped back to left-canonical MPO tensors
(`norm = ‖AC[1]‖ = 1`, the package-wide norm convention; the absolute operator
amplitude is deliberately not restored — accuracy is measured with
[`fidelity`](@ref)/[`infidelity`](@ref), which are invariant under scale and
phase). Returns the compressed `DenseIMPO` and the final overlap
(normalized fidelity × N; see `_overlap_sweeps`).
"""
function mpo_compress(W::DenseIMPO, D::Int;
                      tol::Real = 1.0e-10, maxiter::Int = 100, verbosity::Int = 0)
    N = length(W)
    dus = [size(W[ℓ], 2) for ℓ in 1:N]
    dds = [size(W[ℓ], 4) for ℓ in 1:N]
    K = asmps_view(W.Ws)
    ket = CanonicalIMPS(K)       # canonicalize the MPS view of the MPO as the ket
    x0 = randomimps(scalartype(W), [dus[ℓ] * dds[ℓ] for ℓ in 1:N], D)
    x, overlap = _overlap_sweeps(nothing, ket, x0, K;
                                 tol = tol, maxiter = maxiter, verbosity = verbosity)
    _global_normalize!(x)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return (; W = DenseIMPO(ALs4), overlap = overlap)
end

"""
    mixedcanonical_error(W) -> (ϵ_left, ϵ_right, ϵ_mixed)
    ismixedcanonical(W; tol = 1e-8, verbosity = 0) -> Bool

Mixed-canonical diagnostics for [`CanonicalIMPO`](@ref): checked in the
MPS view `(wl, u·d, wr)` (kernel and conventions follow the
`CanonicalIMPS` methods).
"""
mixedcanonical_error(W::CanonicalIMPO) =
    _mixedcanonical_error(asmps_view(collect(W.AL)), asmps_view(collect(W.AR)), collect(W.C))

function ismixedcanonical(W::CanonicalIMPO; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(W)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end

# ---------------- gauge interface (delegation to the states/ortho.jl kernels in the MPO view) ----------------

function gaugefix!(W::CanonicalIMPO, A, C₀ = W.C[end]; order = :LR, kwargs...)
    N = length(W)
    dus = [size(W.AL[ℓ], 2) for ℓ in 1:N]
    dds = [size(W.AL[ℓ], 4) for ℓ in 1:N]
    # A: rank-4 MPO tensors → MPS view; rank-3 views are used directly
    Av = A isa AbstractVector{<:AbstractArray{<:Number,4}} ? asmps_view(collect(A)) : collect(A)
    # temporary MPS state over the view families (the gauge process only
    # reads/writes these families)
    ALv = asmps_view(collect(W.AL))
    ARv = asmps_view(collect(W.AR))
    Cv = collect(W.C)
    ACv = asmps_view(collect(W.AC))
    ψ = CanonicalIMPS(PeriodicVector(ALv), PeriodicVector(ARv), PeriodicVector(Cv),
                             PeriodicVector(ACv))
    gaugefix!(ψ, Av, C₀; order = order, kwargs...)
    # write back the rank-4 families
    AL4 = mps_view_to_mpo(collect(ψ.AL); dus = dus, dds = dds)
    AR4 = mps_view_to_mpo(collect(ψ.AR); dus = dus, dds = dds)
    AC4 = mps_view_to_mpo(collect(ψ.AC); dus = dus, dds = dds)
    for ℓ in 1:N
        W.AL[ℓ] = AL4[ℓ]
        W.AR[ℓ] = AR4[ℓ]
        W.C[ℓ] = ψ.C[ℓ]
        W.AC[ℓ] = AC4[ℓ]
    end
    return W
end

"""
    regauge!(AC::AbstractArray{T,4}, C::AbstractMatrix; alg) -> AL (rank-4)
    regauge!(CL::AbstractMatrix, AC::AbstractArray{T,4}; alg) -> AR (rank-4)

Rank-4 (MPO) tensor versions of [`regauge!](@ref): canonicalize in the MPS
view `(wl, u·d, wr)` and map back to rank-4; semantics identical to the
rank-3 methods.
"""
function regauge!(AC::AbstractArray{T,4}, C::AbstractMatrix{T}; alg = Defaults.alg_orth()) where {T}
    wl, u, wr, d = size(AC)
    ACv = reshape(permutedims(AC, (1, 2, 4, 3)), wl, u * d, wr)
    ALv = regauge!(ACv, C; alg = alg)
    return permutedims(reshape(ALv, wl, u, d, wr), (1, 2, 4, 3))
end

function regauge!(CL::AbstractMatrix{T}, AC::AbstractArray{T,4}; alg = Defaults.alg_orth()) where {T}
    wl, u, wr, d = size(AC)
    ACv = reshape(permutedims(AC, (1, 2, 4, 3)), wl, u * d, wr)
    ARv = regauge!(CL, ACv; alg = alg)
    return permutedims(reshape(ARv, wl, u, d, wr), (1, 2, 4, 3))
end
"""
    changebond!(W::CanonicalIMPO; D::Int, noise::Real = 1e-10) -> W

[`changebond!`](@ref) 的 MPO 版（MPS 视图 `(wl, u·d, wr)` 的键 profile 调整）：
`AL` 的左右键直接 `_resize_dim` 到 `D`（不足则扩容、超出则截取前导子块），再用
[`CanonicalIMPO`](@ref) 重新包装以恢复混合规范 —— 与 FiniteMPSAlgorithms 的同名
函数一致。各 bond 已等于 `D` 时直接返回、不做任何改动。

同 MPS 版：**infinite MPO 的键维不受物理维乘积限制**，忠实按用户给的 `D`，
不做 `min(D, ∏d)` 之类的截断。

`noise` 填充扩容出的新块（`0` 即零填充，态不变；`noise ≠ 0` 时填 `noise·randn`）：
零填充得到秩亏的态，规范不被唯一确定，单点 TDVP/VUMPS 等依赖规范的算法会因此
给出表示依赖的结果（FiniteMPSAlgorithms 的 `TDVP1` docstring 记录了同一现象）。
"""
function changebond!(W::CanonicalIMPO; D::Int, noise::Real = 1e-10)
    N = length(W)
    b = fill(D, N)
    # 各 bond 已等于 D ⇒ 无需改动，提前返回
    all(bonddim(W, ℓ) == b[ℓ] for ℓ in 1:N) && return W
    for ℓ in 1:N
        ℓm = _mod1(ℓ - 1, N)
        W.AL[ℓ] = _resize_dim(W.AL[ℓ], 1, b[ℓm]; noise = noise)
        W.AL[ℓ] = _resize_dim(W.AL[ℓ], 3, b[ℓ]; noise = noise)
    end
    y = CanonicalIMPO(collect(W.AL))
    copy!(W.AL, y.AL)
    copy!(W.AR, y.AR)
    copy!(W.C, y.C)
    copy!(W.AC, y.AC)
    return W
end
