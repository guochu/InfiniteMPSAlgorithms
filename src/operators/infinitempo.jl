"""
    DenseIMPO{T}

Infinite (periodically tiled) MPO storing a single string of rank-4 tensors:

- `Ws[ℓ]::Array{T,4}` with indices `(wl, u, wr, d)` — **aligned with TEMPO**
  (MPO bonds in slots 1 and 3; `u` is the bra physical / operator-row index,
  `d` the ket physical / operator-column index).

The identity MPO has bond dimension 1: `Ws[ℓ][1, u, 1, d] = δ(u, d)`.
"""
struct DenseIMPO{T}
    Ws::Vector{Array{T,4}}

    function DenseIMPO{T}(Ws::Vector{Array{T,4}}) where {T}
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

DenseIMPO(Ws::Vector{Array{T,4}}) where {T} = DenseIMPO{T}(Ws)
DenseIMPO(Ws::PeriodicVector{<:Array{T,4}}) where {T} = DenseIMPO(collect(Ws))

Base.length(W::DenseIMPO) = length(W.Ws)
Base.getindex(W::DenseIMPO, ℓ::Integer) = W.Ws[_mod1(ℓ, length(W))]
Base.setindex!(W::DenseIMPO, v::Array, ℓ::Integer) = (W.Ws[_mod1(ℓ, length(W))] = v; W)
Base.firstindex(W::DenseIMPO) = 1
Base.lastindex(W::DenseIMPO) = length(W)
Base.iterate(W::DenseIMPO, args...) = iterate(W.Ws, args...)

function Base.copy(W::DenseIMPO)
    return DenseIMPO([copy(w) for w in W.Ws])
end

scalartype(::Type{DenseIMPO{T}}) where {T} = T
scalartype(W::DenseIMPO) = scalartype(typeof(W))

phydims(W::DenseIMPO) = [size(W[ℓ], 2) for ℓ in 1:length(W)]
"`bonddim(W, ℓ)`: the MPO bond dimension to the left of site ℓ."
bonddim(W::DenseIMPO, ℓ::Integer) = size(W[ℓ], 1)
max_bonddim(W::DenseIMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"`dag(W)`: elementwise conjugation of every tensor (for overlap-type
contractions; not the operator-adjoint network)."
dag(W::DenseIMPO) = DenseIMPO(conj.(W.Ws))

# MPSKit convention (scale!(first(mpo), α)): scalar multiplication scales only
# the first tensor; scaling every tensor would change the operator value to
# α^N·W (N = unit-cell length) and would break the minus sign for even N.
function Base.:*(α::Number, W::DenseIMPO)
    out = [copy(w) for w in W.Ws]
    out[1] = α .* out[1]
    return DenseIMPO(out)
end
Base.:*(W::DenseIMPO, α::Number) = α * W
Base.:/(W::DenseIMPO, α::Number) = (1 / α) * W
Base.:-(W::DenseIMPO) = (-one(scalartype(W))) * W
