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

"`A` 沿维度 `dim` 零填充（截取）到尺寸 `d`（首部子块保留；填充块恒为零）。"
function _resize_dim(A::AbstractArray{T,N}, dim::Int, d::Int) where {T,N}
    d0 = size(A, dim)
    d == d0 && return A
    sz = ntuple(i -> i == dim ? d : size(A, i), N)
    B = zeros(T, sz)
    v = ntuple(i -> i == dim ? (1:min(d, d0)) : (1:size(A, i)), N)
    B[v...] = A
    return B
end

"_bond_feasible_profile(phydims, D)：周期链的均匀可行键 profile（键维不超全环
物理维乘积；周期闭合要求各 bond 一致，无有限链的首尾边界约束）。"
function _bond_feasible_profile(phydims::AbstractVector{Int}, D::Int)
    return fill(min(D, prod(phydims)), length(phydims))
end

"""
    changebond!(ψ::CanonicalIMPS; D::Int) -> ψ

把键 profile 调到 `min(D, feasible)`（参考 FiniteMPSAlgorithms 的同名函数）：
大于目标的键截取前导键指标（混合规范下 `C` 的奇异值降序，前导子块即最优
截断，正交性由 U/V 的正交性保持），小于目标的键零填充（态不变）。
用于构造迭代压缩（`mult!` / `compress!`）的初猜。
"""
function changebond!(ψ::CanonicalIMPS; D::Int)
    N = length(ψ)
    b = _bond_feasible_profile(phydims(ψ), D)
    T = scalartype(ψ)
    # 截断超键：逐 bond 在原态上独立 SVD、统一应用（U/V 正交 ⇒ AR 串仍右规范），
    # 再从 AR 串重建混合规范
    svds = Dict{Int,Any}()
    for ℓ in 1:N
        bonddim(ψ, ℓ) > b[ℓ] || continue
        svds[ℓ] = tsvd(ψ.C[ℓ]; trunc = truncdim(b[ℓ]))
    end
    if !isempty(svds)
        for ℓ in 1:N
            ℓm = _mod1(ℓ - 1, N)
            if haskey(svds, ℓ)
                U, s, V, _ = svds[ℓ]
                ψ.AL[ℓ] = @tensor A[a, s2, c] := ψ.AL[ℓ][a, s2, bb] * U[bb, c]
                ψ.AR[ℓ] = @tensor A[a, s2, c] := ψ.AR[ℓ][a, s2, bb] * U[bb, c]
                ψ.C[ℓ] = Matrix{T}(Diagonal(s))
            end
            if haskey(svds, ℓm)
                _, _, Vm, _ = svds[ℓm]
                ψ.AL[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AL[ℓ][bb, s2, b]
                ψ.AR[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AR[ℓ][bb, s2, b]
            end
        end
        y = CanonicalIMPS(collect(ψ.AR))
        copy!(ψ.AL, y.AL)
        copy!(ψ.AR, y.AR)
        copy!(ψ.C, y.C)
        copy!(ψ.AC, y.AC)
    end
    # 零填充不足的键：在 AC 串（键位 1、3）上扩展（态不变）。零填充破坏
    # AL/AR 的正交性，随后从填充后的 AC 串整体重建混合规范（FiniteMPSAlgorithms
    # 同样以无截断 canonicalize! 收尾）。
    anypad = false
    for ℓ in 1:N
        b[ℓ] > bonddim(ψ, ℓ) || continue
        anypad = true
        ℓm = _mod1(ℓ - 1, N)
        ψ.AC[ℓ] = _resize_dim(ψ.AC[ℓ], 1, b[ℓm])
        ψ.AC[ℓ] = _resize_dim(ψ.AC[ℓ], 3, b[ℓ])
    end
    if anypad
        y = CanonicalIMPS(collect(ψ.AC))
        copy!(ψ.AL, y.AL)
        copy!(ψ.AR, y.AR)
        copy!(ψ.C, y.C)
        copy!(ψ.AC, y.AC)
    end
    return ψ
end
