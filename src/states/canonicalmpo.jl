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
representation (`tr(C⁻¹·X·C) = tr(X)`), so the operator value (the periodic
contraction) is preserved exactly.
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
to an MPS via `asmps_view`, then mixed-canonicalize with `gaugefix!` (the
operator value is preserved exactly in the periodic trace representation; see
the type docstring).
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

"Placeholder: the overall scale of an MPO carries physical meaning and its
normalization is controlled by the compression/algebra pipelines."
LinearAlgebra.normalize!(W::CanonicalIMPO) = W

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

"""
    _align_scale!(x::CanonicalIMPS, K::Vector{<:Array{T,3}}) -> x

Scale/phase alignment between the variational solution `x` and the target
tensor string `K` (used for the MPS results of the algebra operations):

- If the bond dimensions match `K` exactly: per-site Frobenius-optimal scalar
  `c_ℓ = ⟨x_AC|K_ℓ⟩/⟨x_AC|x_AC⟩` (restores the original scale and phase exactly);
- Otherwise: the ring overlap `⟨x|K⟩` is an N-th power in `x`; take the
  principal N-th root `c = (⟨x|K⟩/⟨x|x⟩)^(1/N)` and distribute it uniformly
  over the sites (making the ring overlap ⟨x|K⟩ real positive and equal in
  ring⟨x|x⟩ terms); the `C` chain is scaled in sync to preserve the
  `AC = AL·C = C·AR` gauge consistency.
"""
function _align_scale!(x::CanonicalIMPS, K::Vector{<:Array{T,3}}) where {T}
    N = length(x)
    if all(size(x.AC[ℓ]) == size(K[ℓ]) for ℓ in 1:N)
        for ℓ in 1:N
            c = dot(x.AC[ℓ], K[ℓ]) / dot(x.AC[ℓ], x.AC[ℓ])
            x.AC[ℓ] .= x.AC[ℓ] .* c
        end
    else
        xALs = [x.AL[ℓ] for ℓ in 1:N]
        c = (_ring_overlap(xALs, collect(K)) / _ring_overlap(xALs, xALs))^(1 / N)
        for ℓ in 1:N
            x.AC[ℓ] .= x.AC[ℓ] .* c
            x.C[ℓ] .= x.C[ℓ] .* c
        end
    end
    return x
end

"""
    mpo_compress(W::DenseIMPO, D; tol=1e-10, maxiter=100, verbosity=0) -> (; W, overlap)

Variationally compress an MPO to bond dimension `D`: view the MPO as an MPS
(`asmps_view`) and run VOMPS overlap-maximization sweeps on the identity
channel (equivalent to the bond-`D` variational approximation of the dominant
eigenvector of the double-layer transfer `W⊗W̄`). The output takes the
ring-trace alignment scalar `c` times the left-canonical tensors `AL`
([`_ring_scale`](@ref)): the output amplitude exactly restores the target's
projection onto the compressed ray (including phase; collecting `AC` would be
polluted by the C weighting). The per-site phase of the converged state is a
gauge freedom of the eigen solution (MPSKit likewise does not pin it down).
Returns the compressed `DenseIMPO` and the final overlap
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
    c = _ring_scale(x, K)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return (; W = DenseIMPO([c * A for A in ALs4]), overlap = overlap)
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
