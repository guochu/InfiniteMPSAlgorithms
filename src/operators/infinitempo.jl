"""
    InfiniteMPO{T}

无限（周期平铺）MPO，只存一串 4 维张量：

- `Ws[ℓ]::Array{T,4}`，指标 `(wl, u, wr, d)` —— **与 TEMPO 对齐**
  （MPO 键在 1、3 位；`u` 为 bra 物理/算符行，`d` 为 ket 物理/算符列）。

恒等 MPO 键维为 1：`Ws[ℓ][1, u, 1, d] = δ(u, d)`。
"""
struct InfiniteMPO{T}
    Ws::Vector{Array{T,4}}

    function InfiniteMPO{T}(Ws::Vector{Array{T,4}}) where {T}
        N = length(Ws)
        N > 0 || throw(DimensionMismatch("Ws 不能为空"))
        for ℓ in 1:N
            W = Ws[ℓ]
            size(W, 2) == size(W, 4) ||
                throw(DimensionMismatch("site $ℓ 物理：(u)=$(size(W,2)) 与 (d)=$(size(W,4)) 不匹配"))
            ℓ1 = _mod1(ℓ + 1, N)
            size(W, 3) == size(Ws[ℓ1], 1) ||
                throw(DimensionMismatch("MPO bond 不匹配: Ws[$ℓ] 右键 $(size(W,3)) vs Ws[$ℓ1] 左键 $(size(Ws[ℓ1],1))"))
        end
        new{T}(Ws)
    end
end

InfiniteMPO(Ws::Vector{Array{T,4}}) where {T} = InfiniteMPO{T}(Ws)
InfiniteMPO(Ws::PeriodicVector{<:Array{T,4}}) where {T} = InfiniteMPO(collect(Ws))

Base.length(W::InfiniteMPO) = length(W.Ws)
Base.getindex(W::InfiniteMPO, ℓ::Integer) = W.Ws[_mod1(ℓ, length(W))]
Base.setindex!(W::InfiniteMPO, v::Array, ℓ::Integer) = (W.Ws[_mod1(ℓ, length(W))] = v; W)
Base.firstindex(W::InfiniteMPO) = 1
Base.lastindex(W::InfiniteMPO) = length(W)
Base.iterate(W::InfiniteMPO, args...) = iterate(W.Ws, args...)

function Base.copy(W::InfiniteMPO)
    return InfiniteMPO([copy(w) for w in W.Ws])
end

scalartype(::Type{InfiniteMPO{T}}) where {T} = T
scalartype(W::InfiniteMPO) = scalartype(typeof(W))

phydims(W::InfiniteMPO) = [size(W[ℓ], 2) for ℓ in 1:length(W)]
"`bonddim(W, ℓ)`：site ℓ 左侧 MPO 键维。"
bonddim(W::InfiniteMPO, ℓ::Integer) = size(W[ℓ], 1)
max_bonddim(W::InfiniteMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"`dag(W)`：逐张量共轭（用于重叠型收缩；非算符伴随网络）。"
dag(W::InfiniteMPO) = InfiniteMPO(conj.(W.Ws))

# MPSKit 约定（scale!(first(mpo), α)）：标量乘只缩放第一个张量；
# 逐张量缩放会把算符值变成 α^N·W（N 为单胞长度），且偶数 N 时负号失效。
function Base.:*(α::Number, W::InfiniteMPO)
    out = [copy(w) for w in W.Ws]
    out[1] = α .* out[1]
    return InfiniteMPO(out)
end
Base.:*(W::InfiniteMPO, α::Number) = α * W
Base.:/(W::InfiniteMPO, α::Number) = (1 / α) * W
Base.:-(W::InfiniteMPO) = (-one(scalartype(W))) * W
