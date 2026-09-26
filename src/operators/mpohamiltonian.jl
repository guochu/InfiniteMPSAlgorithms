# ---------------- SparseIMPO (ported from MPSKit src/operators/mpohamiltonian.jl) ----------------
#
# Hamiltonian MPO in Schur upper-triangular block-matrix form:
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# The first/last virtual levels are unit levels (identity channels);
# `isidentitylevel`/`isemptylevel` support the per-level linear solves of the
# DMRGCache (see algorithms/idmrg.jl).

"""
    SparseIMPO(Ws) -> SparseIMPO

Infinite Hamiltonian MPO in Schur (sparse) form, stored as a
`PeriodicVector` of [`SchurMPOTensor`](@ref)s (the periodic tiling is built
into the type; plain `Vector` inputs are converted automatically).
`Ws[i][j, k]` is the local operator at site `i` from left level `j` to right
level `k`; entries may be `Missing`, `Number`s, or `(d, d)` matrices.

约束：unit cell 内**层数必须一致**（Schur 上三角 + 周期闭合的要求），
但**各站物理维可以不同**（`phydims(H)` 逐站返回，例如 `[2, 3, 2]`）；
`models.jl` 的便捷构造（`tfim` / `heisenberg_xxz` / `mpohamiltonian(h1, …)`）
因为只给一个 `h1`，产出的模型各站物理维相同。
"""
struct SparseIMPO{TO<:SchurMPOTensor}
    W::PeriodicVector{TO}
    SparseIMPO{TO}(W::PeriodicVector{TO}) where {TO} = new{TO}(W)
end

Base.length(H::SparseIMPO) = length(H.W)
Base.getindex(H::SparseIMPO, i::Int) = getindex(H.W, i)
Base.getindex(H::SparseIMPO, i::Int, j::Int, k::Int) = H[i][j, k]
Base.firstindex(H::SparseIMPO) = firstindex(H.W)
Base.lastindex(H::SparseIMPO) = lastindex(H.W)
Base.parent(H::SparseIMPO) = H.W
Base.copy(H::SparseIMPO) = SparseIMPO(map(copy, parent(H)))
Base.iterate(H::SparseIMPO, args...) = iterate(H.W, args...)
Base.eltype(::Type{SparseIMPO{TO}}) where {TO} = TO

function SparseIMPO(Ws::PeriodicVector{TO}) where {TO<:SchurMPOTensor}
    return SparseIMPO{TO}(Ws)
end
function SparseIMPO(Ws::Vector{TO}) where {TO<:SchurMPOTensor}
    return SparseIMPO{TO}(PeriodicVector(Ws))
end
function SparseIMPO(Ws::Vector{<:Matrix})
    for W in Ws
        (size(W, 1) == size(W, 2)) || throw(ArgumentError("level matrices of an infinite Hamiltonian must be square"))
        (size(W, 1) == size(Ws[1], 1)) ||
            throw(ArgumentError("all level matrices must have the same number of levels " *
                                "(Schur 上三角 + 周期闭合要求层数一致；各站物理维可以不同)"))
    end
    return SparseIMPO(PeriodicVector([SchurMPOTensor(W) for W in Ws]))
end

"bonddim(H, ℓ): the number of Schur virtual levels at site ℓ (per-bond bond
dimension semantics, mirroring MPSKit's `size(mpo[i], 1)`)."
bonddim(H::SparseIMPO, ℓ::Integer) = nlvls(H[ℓ])
"bonddim(H): the uniform level count of the unit cell (the upper-triangular
block structure plus periodic closure require identical levels across sites,
guaranteed by the constructors)."
bonddim(H::SparseIMPO) = nlvls(H[1])

scalartype(::Type{SparseIMPO{TO}}) where {TO} = scalartype(TO)
scalartype(H::SparseIMPO) = scalartype(typeof(H))

phydims(H::SparseIMPO) = [size(H[ℓ].A, 2) for ℓ in 1:length(H)]

# MPSKit-style A/B/C/D block access (returns the corresponding block arrays per site)
function Base.getproperty(H::SparseIMPO, sym::Symbol)
    if sym === :A
        return [getfield(W, :A) for W in parent(H)]
    elseif sym === :B
        return [getfield(W, :B) for W in parent(H)]
    elseif sym === :C
        return [getfield(W, :C) for W in parent(H)]
    elseif sym === :D
        return [getfield(W, :D) for W in parent(H)]
    end
    return getfield(H, sym)
end

"""
    isidentitylevel(H, i) -> Bool

Whether level `i` is an identity level (its transfer contains only the physical
identity operator): always true for the first/last levels; middle levels
require the `(i,i)` diagonal block to be the identity on every site.
"""
function isidentitylevel(H::SparseIMPO, i::Int)
    n = bonddim(H)
    (i == 1 || i == n) && return true
    return all(parent(H)) do W
        block = W.A[i - 1, :, i - 1, :]
        return isapprox(block, Matrix{eltype(block)}(I, size(block)); atol = 1e-14)
    end
end

"""
    isemptylevel(H, i) -> Bool

Whether level `i` is a completely unused channel (mirroring MPSKit: a level is
empty if its diagonal block is structurally absent on any site). In this
package's dense representation this is equivalent to: on every site, the
diagonal block, the first-row C block, and the last-column B block are all
zero. Note that explicitly stored zero diagonal blocks (e.g. middle levels of
a strictly nearest-neighbor MPO) do not count as empty.
"""
function isemptylevel(H::SparseIMPO, i::Int)
    n = bonddim(H)
    (i == 1 || i == n) && return false
    return all(parent(H)) do W
        return iszero(W.A[i - 1, :, i - 1, :]) &&
               iszero(W.C[:, i - 1, :]) &&
               iszero(W.B[i - 1, :, :])
    end
end

# ---- linear algebra ----

function Base.:+(H₁::SparseIMPO, H₂::SparseIMPO)
    (length(H₁) == length(H₂)) || throw(DimensionMismatch("unit-cell lengths do not match"))
    W = [H₁[i] + H₂[i] for i in 1:length(H₁)]
    return SparseIMPO(typeof(H₁.W)(W))
end

"""
    H + λs::AbstractVector (or `λs + H`)

Add `λᵢ·I` per site (mirrors MPSKit's `H + λs`). 逐站取物理维（unit cell 内各站
物理维允许不同，例如 `phydims = [2, 3, 2]`）。
"""
function Base.:+(H::SparseIMPO, λs::AbstractVector{<:Number})
    (length(H) == length(λs)) || throw(DimensionMismatch("unit-cell lengths do not match"))
    Ws = Vector{Matrix{Any}}(undef, length(H))
    for i in 1:length(H)
        n = bonddim(H)
        d = size(H[i].A, 2)         # 逐站物理维
        W = Matrix{Any}(missing, n, n)
        W[1, 1] = one(scalartype(H))
        W[n, n] = one(scalartype(H))
        W[1, n] = λs[i] isa AbstractMatrix ? λs[i] : Matrix(λs[i] * I, d, d)
        Ws[i] = W
    end
    return H + SparseIMPO(Ws)
end
Base.:+(λs::AbstractVector{<:Number}, H::SparseIMPO) = H + λs

"""
    tompotensors(H::SparseIMPO) -> Vector{<:Array{T,4}}

Densify into the package's `(wl, u, wr, d)` MPO tensor string.
"""
tompotensors(H::SparseIMPO) = [tompotensor(H[i]) for i in 1:length(H)]

"""
    DenseIMPO(H::SparseIMPO) -> DenseIMPO

Dense periodic MPO conversion (mirrors MPSKit's `DenseMPO(H)`).
Note: the identity channel becomes explicit, so environments fall back to the
transfer-matrix dominant-eigenvector path.
"""
DenseIMPO(H::SparseIMPO) = DenseIMPO(tompotensors(H))

"""
    infinite_mpo(bulk::SchurMPOTensor) -> DenseIMPO

Schur tensor of a periodic bulk → `DenseIMPO` (bond state 1 = the identity
channel; the on-site term `D` is merged into the identity→identity channel):

- `W[1, ·, 1, ·] = I + D` (on-site term merged into identity→identity);
- `W[1, ·, j+1, ·] = C[:, j, :]`, `W[j+1, ·, 1, ·] = B[j, :, :]`,
  `W[i+1, ·, j+1, ·] = A[i, :, j, :]`.
"""
function infinite_mpo(bulk::SchurMPOTensor)
    T = scalartype(bulk)
    d = size(bulk.A, 2)
    a = size(bulk.A, 1)
    nb = a + 1                       # number of bond states: identity + middle operators
    W = zeros(T, nb, d, nb, d)
    W[1, :, 1, :] = Matrix{T}(I, d, d) + bulk.D
    for j in 1:a
        W[1, :, j+1, :] = bulk.C[:, j, :]           # aⱼ (channel opening)
        W[j+1, :, 1, :] = bulk.B[j, :, :]           # bⱼ (channel closing)
        for i in 1:a
            W[j+1, :, i+1, :] = bulk.A[j, :, i, :]  # channel propagation
        end
    end
    return DenseIMPO([W])
end
