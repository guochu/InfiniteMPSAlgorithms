"""
    CanonicalIMPS{T}

Mixed-canonical storage of an infinite (periodically tiled) MPS; the data
layout mirrors MPSKit's `InfiniteMPS`:

- `AL::PeriodicVector{Array{T,3}}`: left-canonical site tensors (`(Dl, s, Dr)`),
  `Σ AL†·AL = 1`;
- `AR::PeriodicVector{Array{T,3}}`: right-canonical site tensors, `Σ AR·AR† = 1`;
- `C::PeriodicVector{Array{T,2}}`: center matrices on bond ℓ (between sites
  ℓ and ℓ+1);
- `AC::PeriodicVector{Array{T,3}}`: center-canonical site tensors.

Convention (same as MPSKit): `AL[i] * C[i] = AC[i] = C[i-1] * AR[i]`.
All indices wrap around with period N.
"""
struct CanonicalIMPS{T}
    AL::PeriodicVector{Array{T,3}}
    AR::PeriodicVector{Array{T,3}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,3}}

    function CanonicalIMPS{T}(AL::PeriodicVector{Array{T,3}},
                                     AR::PeriodicVector{Array{T,3}},
                                     C::PeriodicVector{Array{T,2}},
                                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
        end
        new{T}(AL, AR, C, AC)
    end
end

_mul_ALC(AL::PeriodicVector{A}, C::PeriodicVector{B}) where {A<:Array{T,3},B<:Matrix{T}} where {T} =
    PeriodicVector(Array{T,3}[_mulAL(AL[ℓ], C[ℓ]) for ℓ in 1:length(AL)])
_mulAL(AL::AbstractArray{T,3}, C::AbstractMatrix{T}) where {T} =
    begin
        @tensor AC[a, s, c] := AL[a, s, b] * C[b, c]
    end

CanonicalIMPS(AL::PeriodicVector{Array{T,3}}, AR::PeriodicVector{Array{T,3}},
                     C::PeriodicVector{Array{T,2}},
                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T} =
    CanonicalIMPS{T}(AL, AR, C, AC)

"""
    CanonicalIMPS(As::AbstractVector{<:Array{T,3}}; kwargs...)

Construct from plain site tensors (mirrors MPSKit's `InfiniteMPS(A)`):
`AR = A`, then `gaugefix!(; order = :LR)` starting from `C₀ = I`, and `AC = AL·C`.
"""
function CanonicalIMPS(As::AbstractVector{<:Array{T,3}}; kwargs...) where {T}
    for ℓ in 1:length(As)-1
        size(As[ℓ], 3) == size(As[ℓ+1], 1) ||
            throw(DimensionMismatch("bond mismatch at site $ℓ"))
    end
    AR = PeriodicVector([copy(a) for a in As])
    AL = PeriodicVector([similar(a) for a in AR])
    AC = PeriodicVector([similar(a) for a in AR])
    D = size(AR[1], 1)
    C = PeriodicVector([similar(AR[1], D, size(AR[_mod1(ℓ + 1, length(AR))], 1)) for ℓ in 1:length(AR)])
    ψ = CanonicalIMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, As, Matrix{T}(I, D, D); kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

"""
    CanonicalIMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix; kwargs...)

Construct from left-canonical tensors plus an initial gauge matrix (mirrors
MPSKit's `InfiniteMPS(AL, C₀)`): `gaugefix!` to the right-canonical form
(`order = :R`), then `AC = AL·C`.
"""
function CanonicalIMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix;
                              kwargs...) where {T}
    AL = PeriodicVector([copy(a) for a in ALs])
    AR = PeriodicVector([similar(a) for a in AL])
    AC = PeriodicVector([similar(a) for a in AL])
    C = PeriodicVector([similar(AL[1], size(C₀, 1), size(C₀, 2)) for _ in 1:length(AL)])
    ψ = CanonicalIMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, ALs, C₀; order = :R, kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

# ---------------- interface ----------------

Base.length(ψ::CanonicalIMPS) = length(ψ.AL)
Base.size(ψ::CanonicalIMPS, args...) = size(ψ.AL, args...)
Base.getindex(ψ::CanonicalIMPS, ℓ::Integer) = ψ.AC[ℓ]
Base.setindex!(ψ::CanonicalIMPS, v::Array, ℓ::Integer) = (ψ.AC[ℓ] = v; ψ)
Base.firstindex(ψ::CanonicalIMPS) = 1
Base.lastindex(ψ::CanonicalIMPS) = length(ψ)
Base.iterate(ψ::CanonicalIMPS, args...) = iterate(ψ.AC, args...)
eachsite(ψ::CanonicalIMPS) = 1:length(ψ)

function Base.copy(ψ::CanonicalIMPS)
    return CanonicalIMPS(PeriodicVector([copy(a) for a in ψ.AL]),
                                PeriodicVector([copy(a) for a in ψ.AR]),
                                PeriodicVector([copy(c) for c in ψ.C]),
                                PeriodicVector([copy(a) for a in ψ.AC]))
end
function Base.similar(ψ::CanonicalIMPS{T}) where {T}
    return CanonicalIMPS{T}(similar(ψ.AL), similar(ψ.AR), similar(ψ.C), similar(ψ.AC))
end
function Base.circshift(ψ::CanonicalIMPS, n)
    return CanonicalIMPS(circshift(ψ.AL, n), circshift(ψ.AR, n),
                                circshift(ψ.C, n), circshift(ψ.AC, n))
end

scalartype(::Type{CanonicalIMPS{T}}) where {T} = T
scalartype(ψ::CanonicalIMPS) = scalartype(typeof(ψ))

phydims(ψ::CanonicalIMPS) = [size(ψ.AL[ℓ], 2) for ℓ in 1:length(ψ)]
bonddim(ψ::CanonicalIMPS, ℓ::Integer) = size(ψ.C[ℓ], 1)
max_bonddim(ψ::CanonicalIMPS) = maximum(bonddim(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`: elementwise conjugation of every tensor."
dag(ψ::CanonicalIMPS) =
    CanonicalIMPS(PeriodicVector(conj.(parent(ψ.AL))), PeriodicVector(conj.(parent(ψ.AR))),
                         PeriodicVector(conj.(parent(ψ.C))), PeriodicVector(conj.(parent(ψ.AC))))

"`LinearAlgebra.norm(ψ) = norm(ψ.AC[1])` (consistent with MPSKit)."
LinearAlgebra.norm(ψ::CanonicalIMPS) = norm(ψ.AC[1])

"""
    LinearAlgebra.normalize!(ψ::CanonicalIMPS)

Mirror of MPSKit's `normalize!(ψ::InfiniteMPS)` (`normalize!.(ψ.C);
normalize!.(ψ.AC)`): every bond matrix `C[ℓ]` and every center tensor `AC[ℓ]`
is normalized to unit Frobenius norm. For a mixed-canonical state the two
normalizations coincide (`‖AL·C‖ = ‖C‖` for left-orthogonal `AL`), so this is
exactly the ring normalization `⟨ψ, ψ⟩ = 1` (and
`norm(ψ) = norm(ψ.AC[1]) = 1`).
"""
function LinearAlgebra.normalize!(ψ::CanonicalIMPS)
    normalize!.(parent(ψ.C))
    normalize!.(parent(ψ.AC))
    return ψ
end

"""
    LinearAlgebra.dot(ψ₁, ψ₂; krylovdim = 30)

`⟨ψ₁|ψ₂⟩`: dominant eigenvalue of the double-layer `AL` transfer matrix
(KrylovKit Arnoldi).
"""
function LinearAlgebra.dot(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS; krylovdim::Int = 30)
    T = promote_type(scalartype(ψ₁), scalartype(ψ₂))
    v0 = vec(Matrix{T}(I, bonddim(ψ₁, 0), bonddim(ψ₂, 0)))
    tm = TransferMatrix(ψ₂.AL, ψ₁.AL)
    vals, vecs, _ = eigsolve(tm, v0, 1, :LM; krylovdim = krylovdim)
    λ = vals[1]
    return λ isa Number ? λ : only(λ)
end

"""
    fidelity(ψ₁, ψ₂) -> Real
    infidelity(ψ₁, ψ₂) -> Real

`fidelity = |⟨ψ₁|ψ₂⟩| / (‖ψ₁‖·‖ψ₂‖) ∈ [0, 1]`: the normalized ring overlap.
Invariant under independent overall phases **and** normalizations of the two
states, so it compares rays rather than representatives — the natural accuracy
measure for variational algebra results (which are only defined up to a global
phase). `infidelity = 1 − fidelity`.
"""
fidelity(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS) =
    abs(dot(ψ₁, ψ₂)) / (norm(ψ₁) * norm(ψ₂))
infidelity(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS) = 1 - fidelity(ψ₁, ψ₂)

# ---------------- mixed-canonical diagnostics (after InfiniteTEMPO's ismixedcanonical) ----------------

"Rank-3 shared kernel of the mixed-canonical error (`Cv[ℓ]` sits on the bond to
the right of site ℓ, closed periodically)."
function _mixedcanonical_error(ALv::AbstractVector{<:Array{T,3}},
                               ARv::AbstractVector{<:Array{T,3}},
                               Cv::AbstractVector{<:AbstractMatrix{T}}) where {T}
    L = length(ALv)
    (length(ARv) == length(Cv) == L) ||
        throw(ArgumentError("inconsistent cell lengths of AL, AR, C: $(length(ALv)), $(length(ARv)), $(length(Cv))"))
    ϵ_left = ϵ_right = ϵ_mixed = 0.0
    for ℓ in 1:L
        @tensor g[a, b] := conj(ALv[ℓ][x, s, a]) * ALv[ℓ][x, s, b]
        ϵ_left = max(ϵ_left, norm(g - I))
        @tensor g[a, b] := ARv[ℓ][a, s, x] * conj(ARv[ℓ][b, s, x])
        ϵ_right = max(ϵ_right, norm(g - I))
        @tensor ac1[a, s, c] := ALv[ℓ][a, s, b] * Cv[ℓ][b, c]
        @tensor ac2[a, s, c] := Cv[_mod1(ℓ - 1, L)][a, b] * ARv[ℓ][b, s, c]
        ϵ_mixed = max(ϵ_mixed, norm(ac1 - ac2))
    end
    return ϵ_left, ϵ_right, ϵ_mixed
end

"""
    mixedcanonical_error(ψ) -> (ϵ_left, ϵ_right, ϵ_mixed)
    ismixedcanonical(ψ; tol = 1e-8, verbosity = 0) -> Bool

Diagnostics of the mixed-canonical form (after InfiniteTEMPO's
`ismixedcanonical`, mainly for debugging):

- `ϵ_left`  = max_ℓ ‖Σ AL[ℓ]†·AL[ℓ] − I‖ (left orthogonality)
- `ϵ_right` = max_ℓ ‖Σ AR[ℓ]·AR[ℓ]† − I‖ (right orthogonality)
- `ϵ_mixed` = max_ℓ ‖AL[ℓ]·C[ℓ] − C[ℓ-1]·AR[ℓ]‖ (mixed-canonical consistency,
  with C closed periodically)

`CanonicalIMPO` is checked analogously in the MPS view
`(wl, u·d, wr)`. `ismixedcanonical` returns `true` when all three errors are
≤ `tol`; `verbosity > 0` prints the errors.
"""
mixedcanonical_error(ψ::CanonicalIMPS) =
    _mixedcanonical_error(parent(ψ.AL), parent(ψ.AR), parent(ψ.C))

function ismixedcanonical(ψ::CanonicalIMPS; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(ψ)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end
