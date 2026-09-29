"""
    DenseIMPS{T}

Infinite (periodically tiled) MPS storing a single string of rank-3 tensors:

- `As[ℓ]::Array{T,3}` with indices `(Dl, s, Dr)` — bonds in slots 1 and 3,
  the same layout as [`CanonicalIMPS`](@ref).

No canonicalization or normalization is performed — the raw tensor string is
stored exactly as given (the uncanonicalized counterpart of
[`CanonicalIMPS`](@ref), mirroring the [`DenseIMPO`](@ref) /
[`CanonicalIMPO`](@ref) split). The strict algebra operations
(`DenseIMPO * DenseIMPS`, `hadamard(::DenseIMPS, ::DenseIMPS)`) return this
type.
"""
struct DenseIMPS{T}
    As::Vector{Array{T,3}}

    function DenseIMPS{T}(As::Vector{Array{T,3}}) where {T}
        N = length(As)
        N > 0 || throw(DimensionMismatch("As must not be empty"))
        for ℓ in 1:N
            ℓ1 = _mod1(ℓ + 1, N)
            size(As[ℓ], 3) == size(As[ℓ1], 1) ||
                throw(DimensionMismatch("MPS bond mismatch: As[$ℓ] right bond $(size(As[ℓ],3)) vs As[$ℓ1] left bond $(size(As[ℓ1],1))"))
        end
        new{T}(As)
    end
end

DenseIMPS(As::Vector{Array{T,3}}) where {T} = DenseIMPS{T}(As)
DenseIMPS(As::PeriodicVector{<:Array{T,3}}) where {T} = DenseIMPS(collect(As))

Base.length(ψ::DenseIMPS) = length(ψ.As)
Base.getindex(ψ::DenseIMPS, ℓ::Integer) = ψ.As[_mod1(ℓ, length(ψ))]
Base.setindex!(ψ::DenseIMPS, v::Array, ℓ::Integer) = (ψ.As[_mod1(ℓ, length(ψ))] = v; ψ)
Base.firstindex(ψ::DenseIMPS) = 1
Base.lastindex(ψ::DenseIMPS) = length(ψ)
Base.iterate(ψ::DenseIMPS, args...) = iterate(ψ.As, args...)

function Base.copy(ψ::DenseIMPS)
    return DenseIMPS([copy(a) for a in ψ.As])
end

scalartype(::Type{DenseIMPS{T}}) where {T} = T
scalartype(ψ::DenseIMPS) = scalartype(typeof(ψ))

"`phydim(ψ, i)`: site `i` 的物理维（unit cell 内允许逐站不同）。"
phydim(ψ::DenseIMPS, i::Integer) = size(ψ[i], 2)
phydims(ψ::DenseIMPS) = [size(ψ[ℓ], 2) for ℓ in 1:length(ψ)]
"`bonddim(ψ, ℓ)`: the MPS bond dimension to the left of site ℓ."
bonddim(ψ::DenseIMPS, ℓ::Integer) = size(ψ[ℓ], 1)
max_bonddim(ψ::DenseIMPS) = maximum(bonddim(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`: elementwise conjugation of every tensor."
dag(ψ::DenseIMPS) = DenseIMPS(conj.(ψ.As))

# scalar multiplication / Hadamard product / overlap functions: see linalg.jl
