# ---------------- MPS constructors ----------------

"""
    randomimps([T=Float64,] phydims, D; rng=Random.default_rng()) -> CanonicalIMPS

Random MPS with uniform bond dimension `D`, canonicalized into the mixed
canonical form upon construction.
"""
function randomimps(::Type{T}, phydims::AbstractVector{Int}, D::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(phydims)
    (N >= 1) || throw(ArgumentError("at least one site is required"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        As[ℓ] = randn(rng, T, D, phydims[ℓ], D)
    end
    return CanonicalIMPS(As)
end
randomimps(phydims::AbstractVector{Int}, D::Int; kwargs...) = randomimps(Float64, phydims, D; kwargs...)

"""
    prodimps([T=Float64,] phydims, states) -> CanonicalIMPS

Product state with bond dimension 1; `states[ℓ]` gives the basis index at each
site (defaults to all `1`).
"""
function prodimps(::Type{T}, phydims::AbstractVector{Int},
                  states::AbstractVector{Int} = ones(Int, length(phydims))) where {T<:Number}
    N = length(phydims)
    (length(states) == N) || throw(DimensionMismatch("length of states must equal the number of sites"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        A = zeros(T, 1, phydims[ℓ], 1)
        A[1, states[ℓ], 1] = one(T)
        As[ℓ] = A
    end
    return CanonicalIMPS(As)
end
prodimps(phydims::AbstractVector{Int}, states::AbstractVector{Int} = ones(Int, length(phydims))) =
    prodimps(Float64, phydims, states)

# ---------------- MPO constructors ----------------

"""
    identityimpo([T=ComplexF64,] phydims) -> DenseIMPO

Identity MPO: bond dimension 1, `W[1, u, 1, d] = δ(u, d)`.
"""
function identityimpo(::Type{T}, phydims::AbstractVector{Int}) where {T<:Number}
    Ws = Vector{Array{T,4}}(undef, length(phydims))
    for (ℓ, dd) in enumerate(phydims)
        W = zeros(T, 1, dd, 1, dd)
        for s in 1:dd
            W[1, s, 1, s] = one(T)
        end
        Ws[ℓ] = W
    end
    return DenseIMPO(Ws)
end
identityimpo(phydims::AbstractVector{Int}) = identityimpo(ComplexF64, phydims)

"""
    randomimpo([T=ComplexF64,] phydims, Dw; rng=Random.default_rng()) -> DenseIMPO

Random MPO with uniform bond dimension `Dw` (generically non-Hermitian; for
physical models use the constructors in `models.jl`).
"""
function randomimpo(::Type{T}, phydims::AbstractVector{Int}, Dw::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(phydims)
    Ws = Vector{Array{T,4}}(undef, N)
    for ℓ in 1:N
        Ws[ℓ] = randn(rng, T, Dw, phydims[ℓ], Dw, phydims[ℓ])
    end
    return DenseIMPO(Ws)
end
randomimpo(phydims::AbstractVector{Int}, Dw::Int; kwargs...) = randomimpo(ComplexF64, phydims, Dw; kwargs...)
