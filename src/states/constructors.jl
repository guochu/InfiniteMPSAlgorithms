# ---------------- MPS 构造器 ----------------

"""
    random_mps([T=Float64,] physdims, D; rng=Random.default_rng()) -> MixedCanonicalMPS

随机等键维 MPS，构造后规范化到混合规范形式。
"""
function random_mps(::Type{T}, physdims::AbstractVector{Int}, D::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(physdims)
    (N >= 1) || throw(ArgumentError("至少需要一个 site"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        As[ℓ] = randn(rng, T, D, physdims[ℓ], D)
    end
    return MixedCanonicalMPS(As)
end
random_mps(physdims::AbstractVector{Int}, D::Int; kwargs...) = random_mps(Float64, physdims, D; kwargs...)

"""
    product_mps([T=Float64,] physdims, states) -> MixedCanonicalMPS

键维 1 的乘积态，`states[ℓ]` 给出每个 site 的基矢（默认全部为 1）。
"""
function product_mps(::Type{T}, physdims::AbstractVector{Int},
                     states::AbstractVector{Int} = ones(Int, length(physdims))) where {T<:Number}
    N = length(physdims)
    (length(states) == N) || throw(DimensionMismatch("states 长度必须等于 site 数"))
    As = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        A = zeros(T, 1, physdims[ℓ], 1)
        A[1, states[ℓ], 1] = one(T)
        As[ℓ] = A
    end
    return MixedCanonicalMPS(As)
end
product_mps(physdims::AbstractVector{Int}, states::AbstractVector{Int} = ones(Int, length(physdims))) =
    product_mps(Float64, physdims, states)

# ---------------- MPO 构造器 ----------------

"""
    identity_mpo([T=ComplexF64,] physdims) -> InfiniteMPO

恒等 MPO：键维 1，`W[1, u, 1, d] = δ(u, d)`。
"""
function identity_mpo(::Type{T}, physdims::AbstractVector{Int}) where {T<:Number}
    Ws = Vector{Array{T,4}}(undef, length(physdims))
    for (ℓ, d) in enumerate(physdims)
        W = zeros(T, 1, d, 1, d)
        for s in 1:d
            W[1, s, 1, s] = one(T)
        end
        Ws[ℓ] = W
    end
    return InfiniteMPO(Ws)
end
identity_mpo(physdims::AbstractVector{Int}) = identity_mpo(ComplexF64, physdims)

"""
    random_mpo([T=ComplexF64,] physdims, Dw; rng=Random.default_rng()) -> InfiniteMPO

随机等键维 MPO（一般非厄米；物理模型请用 `models.jl` 的构造器）。
"""
function random_mpo(::Type{T}, physdims::AbstractVector{Int}, Dw::Int;
                    rng::AbstractRNG = Random.default_rng()) where {T<:Number}
    N = length(physdims)
    Ws = Vector{Array{T,4}}(undef, N)
    for ℓ in 1:N
        Ws[ℓ] = randn(rng, T, Dw, physdims[ℓ], Dw, physdims[ℓ])
    end
    return InfiniteMPO(Ws)
end
random_mpo(physdims::AbstractVector{Int}, Dw::Int; kwargs...) = random_mpo(ComplexF64, physdims, Dw; kwargs...)
