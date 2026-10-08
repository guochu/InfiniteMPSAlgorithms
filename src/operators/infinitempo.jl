"""
    DenseIMPO{T}

Infinite (periodically tiled) MPO storing a single string of rank-4 tensors:

- `Ws[ℓ]::Array{T,4}` with indices `(wl, u, wr, d)` — **aligned with TEMPO**
  (MPO bonds in slots 1 and 3; `u` is the bra physical / operator-row index,
  `d` the ket physical / operator-column index).

The identity MPO has bond dimension 1: `Ws[ℓ][1, u, 1, d] = δ(u, d)`.
"""
struct DenseIMPO{T<:Number} <: AbstractInfiniteMPO{T}
    Ws::PeriodicVector{Array{T,4}}

    function DenseIMPO{T}(Ws::PeriodicVector{Array{T,4}}) where {T}
        N = length(Ws)
        N > 0 || throw(DimensionMismatch("Ws must not be empty"))
        for ℓ in 1:N
            W = Ws[ℓ]
            size(W, 2) == size(W, 4) ||
                throw(DimensionMismatch("site $ℓ physical dims: (u)=$(size(W,2)) vs (d)=$(size(W,4))"))
            ℓ1 = _mod1(ℓ + 1, N)
            size(W, 3) == size(Ws[ℓ1], 1) ||
                throw(DimensionMismatch("MPO bond mismatch: Ws[$ℓ] right bond $(size(W,3)) vs Ws[$ℓ1] left bond $(size(Ws[ℓ1],1))"))
        end
        new{T}(Ws)
    end
end

DenseIMPO(Ws::PeriodicVector{Array{T,4}}) where {T} = DenseIMPO{T}(Ws)
DenseIMPO(Ws::Vector{Array{T,4}}) where {T} = DenseIMPO{T}(PeriodicVector(Ws))

Base.propertynames(::DenseIMPO) = (:Ws, :AL, :AR, :AC, :C)
"`DenseIMPO` 的家族访问（[`AbstractInfiniteMPO`](@ref) 接口）：`AL`/`AR`/`AC`
即原始张量串本体（`PeriodicVector` 周期存储、无拷贝），`C` 返回
[`BondView`](@ref) 单位矩阵视图——使其可像 [`CanonicalIMPO`](@ref) 一样以
`W.AL[ℓ]` / `W.C[ℓ]`（周期下标）消费。"
function Base.getproperty(W::DenseIMPO, sym::Symbol)
    if sym === :AL || sym === :AR || sym === :AC
        return getfield(W, :Ws)
    elseif sym === :C
        return BondView(W)
    end
    return getfield(W, sym)
end

Base.length(W::DenseIMPO) = length(W.Ws)
Base.getindex(W::DenseIMPO, ℓ::Integer) = W.Ws[_mod1(ℓ, length(W))]
Base.setindex!(W::DenseIMPO, v::Array, ℓ::Integer) = (W.Ws[_mod1(ℓ, length(W))] = v; W)
Base.firstindex(W::DenseIMPO) = 1
Base.lastindex(W::DenseIMPO) = length(W)
Base.iterate(W::DenseIMPO, args...) = iterate(W.Ws, args...)

function Base.copy(W::DenseIMPO)
    return DenseIMPO(PeriodicVector([copy(w) for w in W.Ws]))
end

"`phydim(W, i)`: site `i` 的上物理维 `du`（unit cell 内允许逐站不同）。"
phydim(W::DenseIMPO, i::Integer) = size(W[i], 2)
phydims(W::DenseIMPO) = [size(W[ℓ], 2) for ℓ in 1:length(W)]
"`bonddim(W, ℓ)`: the MPO bond dimension on the bond right of site ℓ
（包内统一右键约定，见 `DenseIMPS` 版说明）。"
bonddim(W::DenseIMPO, ℓ::Integer) = size(W[ℓ], 3)
max_bonddim(W::DenseIMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"""
    changebond!(W::DenseIMPO; D::Int, noise::Real = 1e-10) -> W

[`changebond!`](@ref) 的 Dense MPO 版（语义同 [`CanonicalIMPO`](@ref) 版，
见 `states/canonicalmpo.jl` 的完整 docstring）：`Ws` 张量串逐站经
`resize_bonds`（InfiniteTEMPO 同名函数，两键一次调到 `D`，不足则扩容、
超出则截取前导子块）。Dense 家族不携带
规范数据，无需重新包装——键 profile 调整即全部工作。各 bond 已等于 `D`
时直接返回、不做任何改动。
"""
function changebond!(W::DenseIMPO; D::Int, noise::Real = 1e-10)
    all(bonddim(W, ℓ) == D for ℓ in 1:length(W)) && return W
    for ℓ in 1:length(W)
        W.Ws[ℓ] = resize_bonds(W.Ws[ℓ], D; noise = noise)
    end
    return W
end

"`dag(W)`: elementwise conjugation of every tensor (for overlap-type
contractions; not the operator-adjoint network)."
dag(W::DenseIMPO) = DenseIMPO(PeriodicVector([conj.(w) for w in W.Ws]))

"""
    Base.transpose(W::DenseIMPO) -> DenseIMPO

The operator transpose: the physical legs of every site tensor are exchanged
(`W[wl, u, wr, d] -> Wᵗ[wl, d, wr, u]`, bonds untouched), which transposes
the represented operator matrix. Combined with [`dag`](@ref) (elementwise
conjugation) this yields the operator-adjoint network.
"""
Base.transpose(W::DenseIMPO) = DenseIMPO([permutedims(w, (1, 4, 3, 2)) for w in W.Ws])

"""
    superoperator(a::DenseIMPO, b::DenseIMPO) -> DenseIMPO

两个等长 MPO 在加倍物理空间上的 Kronecker 型双通道乘积（命名刻意避开
`Base.kron`——那是矩阵/向量的 Kronecker 积约定，此处语义不同）：from
the site tensors `A[aL, po, aR, pin]` and `B[bL, q, bR, qin]` the result tensor is

    S[(aL,bL), (po,q), (aR,bR), (pin,qin)] = A[aL, po, aR, pin] · B[bL, q, bR, qin]

with the `a` legs the fastest on every fused index (the [`vectorize`](@ref)
fusion convention `f = u + du·(d - 1)`); the bond dimensions multiply. The
two multiplication channels are the special cases:

    superoperator(h; side = :left)  == superoperator(h, identityimpo(phydims(h)))
    superoperator(h; side = :right) == superoperator(identityimpo(phydims(h)), transpose(h))
"""
function superoperator(a::DenseIMPO, b::DenseIMPO)
    L = length(a)
    L == length(b) || throw(DimensionMismatch(
        "superoperator requires equal unit-cell lengths, got $L and $(length(b))"))
    T = promote_type(scalartype(a), scalartype(b))
    data = Vector{Array{T,4}}(undef, L)
    for ℓ in 1:L
        A, B = a[ℓ], b[ℓ]
        @tensor S[al, bl, po, q, ar, br, pin, qin] := A[al, po, ar, pin] * B[bl, q, br, qin]
        # merge (al, bl), (po, q), (ar, br), (pin, qin) column major (a-leg the fastest)
        data[ℓ] = reshape(S, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2),
                          size(A, 3) * size(B, 3), size(A, 4) * size(B, 4))
    end
    return DenseIMPO(data)
end
