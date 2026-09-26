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

# ---------------- changebond! (bond-profile adjustment; reference: FiniteMPSAlgorithms) ----------------

"""
`A` 沿维度 `dim` 调整到尺寸 `d`（首部子块保留；不足则扩容，超出则截取）。

新增块默认零填充（`noise = 0`，态不变）；`noise ≠ 0` 时填 `noise·randn`，用于
打破扩容后的键退化：零填充得到秩亏的态（奇异值含一批精确零），规范不被唯一
确定，单点 TDVP/VUMPS 等依赖规范的算法会给出表示依赖的结果。
"""
function _resize_dim(A::AbstractArray{T,N}, dim::Int, d::Int; noise::Real = 0) where {T,N}
    d0 = size(A, dim)
    d == d0 && return A
    sz = ntuple(i -> i == dim ? d : size(A, i), N)
    B = iszero(noise) ? zeros(T, sz) : noise * randn(T, sz)
    v = ntuple(i -> i == dim ? (1:min(d, d0)) : (1:size(A, i)), N)
    B[v...] = A[v...]
    return B
end

"""
    changebond!(ψ::CanonicalIMPS; D::Int, noise::Real = 1e-10) -> ψ

把每个 site tensor 的键维**强制**改到 `D`（每个 bond 都是 `D`；参考
FiniteMPSAlgorithms 的同名函数）：`AL` 的左右键直接 `_resize_dim` 到 `D`
（不足则扩容、超出则截取前导子块），随后用 [`CanonicalIMPS`](@ref) 重新包装，
恢复混合规范。各 bond 已等于 `D` 时直接返回、不做任何改动。
用于构造迭代压缩（`mult!` / `compress!`）的初猜。

注意：**infinite MPS 的键维不受物理维乘积限制**（`D` 可以任意大，周期链上每个
bond 都能取到 `D`），因此这里忠实按用户给的 `D`，不做 `min(D, ∏d)` 之类的截断；
这与有限链"最大键维 = 切开一侧的物理维乘积"不同。

`noise` 填充扩容出的新块（`0` 即零填充，态严格不变）：零填充得到秩亏的态
（`C` 的奇异值含一批精确零），规范不被唯一确定，单点 TDVP/VUMPS 等依赖规范的
算法会因此给出表示依赖的结果（FiniteMPSAlgorithms 的 `TDVP1` docstring 记录了
同一现象）；默认 `1e-10` 足以打破退化。
"""
function changebond!(ψ::CanonicalIMPS; D::Int, noise::Real = 1e-10)
    N = length(ψ)
    b = fill(D, N)
    # 各 bond 已等于 D ⇒ 无需改动，提前返回
    all(bonddim(ψ, ℓ) == b[ℓ] for ℓ in 1:N) && return ψ
    for ℓ in 1:N
        ℓm = _mod1(ℓ - 1, N)
        ψ.AL[ℓ] = _resize_dim(ψ.AL[ℓ], 1, b[ℓm]; noise = noise)
        ψ.AL[ℓ] = _resize_dim(ψ.AL[ℓ], 3, b[ℓ]; noise = noise)
    end
    y = CanonicalIMPS(collect(ψ.AL))
    copy!(ψ.AL, y.AL)
    copy!(ψ.AR, y.AR)
    copy!(ψ.C, y.C)
    copy!(ψ.AC, y.AC)
    return ψ
end
