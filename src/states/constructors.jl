# ---------------- MPS 构造器 ----------------

"""
    randomimps([T=Float64,] phydims, D; rng=Random.default_rng()) -> InfiniteCanonicalMPS

随机等键维 MPS，构造后规范化到混合规范形式。
"""
function randomimps(::Type{T}, phydims::AbstractVector{Int}, D::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(phydims)
    (N >= 1) || throw(ArgumentError("至少需要一个 site"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        As[ℓ] = randn(rng, T, D, phydims[ℓ], D)
    end
    return InfiniteCanonicalMPS(As)
end
randomimps(phydims::AbstractVector{Int}, D::Int; kwargs...) = randomimps(Float64, phydims, D; kwargs...)

"""
    prodimps([T=Float64,] phydims, states) -> InfiniteCanonicalMPS

键维 1 的乘积态，`states[ℓ]` 给出每个 site 的基矢（默认全部为 1）。
"""
function prodimps(::Type{T}, phydims::AbstractVector{Int},
                  states::AbstractVector{Int} = ones(Int, length(phydims))) where {T<:Number}
    N = length(phydims)
    (length(states) == N) || throw(DimensionMismatch("states 长度必须等于 site 数"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        A = zeros(T, 1, phydims[ℓ], 1)
        A[1, states[ℓ], 1] = one(T)
        As[ℓ] = A
    end
    return InfiniteCanonicalMPS(As)
end
prodimps(phydims::AbstractVector{Int}, states::AbstractVector{Int} = ones(Int, length(phydims))) =
    prodimps(Float64, phydims, states)

# ---------------- MPO 构造器 ----------------

"""
    identityimpo([T=ComplexF64,] phydims) -> InfiniteMPO

恒等 MPO：键维 1，`W[1, u, 1, d] = δ(u, d)`。
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
    return InfiniteMPO(Ws)
end
identityimpo(phydims::AbstractVector{Int}) = identityimpo(ComplexF64, phydims)

"""
    randomimpo([T=ComplexF64,] phydims, Dw; rng=Random.default_rng()) -> InfiniteMPO

随机等键维 MPO（一般非厄米；物理模型请用 `models.jl` 的构造器）。
"""
function randomimpo(::Type{T}, phydims::AbstractVector{Int}, Dw::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(phydims)
    Ws = Vector{Array{T,4}}(undef, N)
    for ℓ in 1:N
        Ws[ℓ] = randn(rng, T, Dw, phydims[ℓ], Dw, phydims[ℓ])
    end
    return InfiniteMPO(Ws)
end
randomimpo(phydims::AbstractVector{Int}, Dw::Int; kwargs...) = randomimpo(ComplexF64, phydims, Dw; kwargs...)
