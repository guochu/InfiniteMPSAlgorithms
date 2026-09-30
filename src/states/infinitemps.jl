"""
    DenseIMPS{T}

Infinite (periodically tiled) MPS storing a single string of rank-3 tensors:

- `As[ℓ]::Array{T,3}` with indices `(Dl, s, Dr)` — bonds in slots 1 and 3,
  the same layout as [`CanonicalIMPS`](@ref).

No canonicalization or normalization is performed — the raw tensor string is
stored exactly as given (the uncanonicalized counterpart of
[`CanonicalIMPS`](@ref), mirroring the [`DenseIMPO`](@ref) /
[`CanonicalIMPO`](@ref) split). The strict algebra operations
(`DenseIMPO * DenseIMPS`, `⊙(::DenseIMPS, ::DenseIMPS)`) return this
type.
"""
struct DenseIMPS{T<:Number} <: AbstractInfiniteMPS{T}
    As::PeriodicVector{Array{T,3}}

    function DenseIMPS{T}(As::PeriodicVector{Array{T,3}}) where {T}
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

DenseIMPS(As::PeriodicVector{Array{T,3}}) where {T} = DenseIMPS{T}(As)
DenseIMPS(As::Vector{Array{T,3}}) where {T} = DenseIMPS{T}(PeriodicVector(As))

Base.propertynames(::DenseIMPS) = (:As, :AL, :AR, :AC, :C)
"`DenseIMPS` 的家族访问（[`AbstractInfiniteMPS`](@ref) 接口）：`AL`/`AR`/`AC`
即原始张量串本体（`PeriodicVector` 周期存储、无拷贝），`C` 返回
[`BondView`](@ref) 单位矩阵视图——使其可像 [`CanonicalIMPS`](@ref) 一样以
`ψ.AL[ℓ]` / `ψ.C[ℓ]`（周期下标）消费。"
function Base.getproperty(ψ::DenseIMPS, sym::Symbol)
    if sym === :AL || sym === :AR || sym === :AC
        return getfield(ψ, :As)
    elseif sym === :C
        return BondView(ψ)
    end
    return getfield(ψ, sym)
end

Base.length(ψ::DenseIMPS) = length(ψ.As)
Base.getindex(ψ::DenseIMPS, ℓ::Integer) = ψ.As[_mod1(ℓ, length(ψ))]
Base.setindex!(ψ::DenseIMPS, v::Array, ℓ::Integer) = (ψ.As[_mod1(ℓ, length(ψ))] = v; ψ)
Base.firstindex(ψ::DenseIMPS) = 1
Base.lastindex(ψ::DenseIMPS) = length(ψ)
Base.iterate(ψ::DenseIMPS, args...) = iterate(ψ.As, args...)

function Base.copy(ψ::DenseIMPS)
    return DenseIMPS(PeriodicVector([copy(a) for a in ψ.As]))
end

"`phydim(ψ, i)`: site `i` 的物理维（unit cell 内允许逐站不同）。"
phydim(ψ::DenseIMPS, i::Integer) = size(ψ[i], 2)
phydims(ψ::DenseIMPS) = [size(ψ[ℓ], 2) for ℓ in 1:length(ψ)]
"eachsite(ψ) = 1:length(ψ)（[`AbstractInfiniteMPS`](@ref) 泛型）。"
eachsite(ψ::AbstractInfiniteMPS) = 1:length(ψ)
"`bonddim(ψ, ℓ)`: the MPS bond dimension to the left of site ℓ."
bonddim(ψ::DenseIMPS, ℓ::Integer) = size(ψ[ℓ], 1)
max_bonddim(ψ::DenseIMPS) = maximum(bonddim(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`: elementwise conjugation of every tensor."
dag(ψ::DenseIMPS) = DenseIMPS(PeriodicVector([conj.(a) for a in ψ.As]))

# scalar multiplication / Hadamard product / overlap functions: see linalg.jl
