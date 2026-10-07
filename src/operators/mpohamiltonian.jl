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
# DMRGCache (see algorithms/groundstates/envs.jl).

"""
    SparseIMPO(Ws) -> SparseIMPO

Infinite Hamiltonian MPO in Schur (sparse) form, stored as a
`PeriodicVector` of [`SchurMPOTensor`](@ref)s (the periodic tiling is built
into the type; plain `Vector` inputs are converted automatically).
`Ws[i][j, k]` is the local operator at site `i` from left level `j` to right
level `k`; entries may be `Missing`, `Number`s, or `(d, d)` matrices.

约束（对标 FMA 后端 `MPOHamiltonian` 的链式闭合）：相邻站
`space_r(Ws[i]) == space_l(Ws[i+1])`、周期 `space_r(Ws[end]) == space_l(Ws[1])`。
unitcell > 1 时各站 Schur 张量**可以为矩形**（通道在不同键上开启/闭合，
键层数逐键可不同）；各站物理维也可以不同（`phydims(H)`
逐站返回）。均匀（每站方形等层数）时 0 参 `bonddim(H)` 可用，一般情形请用
`bonddim(H, ℓ)` / `max_bonddim(H)`。
"""
struct SparseIMPO{T<:Number} <: AbstractInfiniteMPO{T}
    Ws::PeriodicVector{SchurMPOTensor{T}}

    function SparseIMPO{T}(Ws::PeriodicVector{SchurMPOTensor{T}}) where {T}
        _check_level_chain(Ws)
        new{T}(Ws)
    end
end

Base.length(H::SparseIMPO) = length(H.Ws)
Base.getindex(H::SparseIMPO, i::Int) = getindex(H.Ws, i)
Base.firstindex(H::SparseIMPO) = firstindex(H.Ws)
Base.lastindex(H::SparseIMPO) = lastindex(H.Ws)
Base.copy(H::SparseIMPO) = SparseIMPO(PeriodicVector([copy(w) for w in H.Ws]))
# 元素类型统一走 scalartype(H)（SparseIMPO{T} → T）；迭代/三参 getindex
# （H[i, j, k] = H[i][j, k]）无包内消费者，不提供。

SparseIMPO(Ws::PeriodicVector{SchurMPOTensor{T}}) where {T} = SparseIMPO{T}(Ws)
SparseIMPO(Ws::Vector{SchurMPOTensor{T}}) where {T} = SparseIMPO{T}(PeriodicVector(Ws))
SparseIMPO(Ws::Vector{<:Matrix}) =
    SparseIMPO(PeriodicVector([SchurMPOTensor(W) for W in Ws]))

"链式闭合检查（对标 FMA `MPOHamiltonian` 构造器）：相邻站
`space_r(W[i]) == space_l(W[i+1])`、周期 `space_r(W[end]) == space_l(W[1])`。
各站 Schur 张量可为矩形（键层数逐键不同）。"
function _check_level_chain(Ws)
    N = length(Ws)
    N > 0 || throw(ArgumentError("Ws must not be empty"))
    for ℓ in 1:N
        ℓ1 = _mod1(ℓ + 1, N)
        space_r(Ws[ℓ]) == space_l(Ws[ℓ1]) || throw(DimensionMismatch(
            "level chain mismatch: Ws[$ℓ] right levels $(space_r(Ws[ℓ])) vs Ws[$ℓ1] left levels $(space_l(Ws[ℓ1]))"))
    end
    return Ws
end

"bonddim(H, ℓ): the number of Schur virtual levels on the bond right of
site ℓ (= `space_r(H[ℓ])`，包内统一右键约定；矩形张量上左/右层数不同，
两侧经链式闭合衔接)."
bonddim(H::SparseIMPO, ℓ::Integer) = space_r(H[ℓ])

"bonddim(H): the uniform level count — only defined when every site tensor is
square with identical level counts (the translation-invariant case); throws an
error for rectangular chains（逐键层数请用 `bonddim(H, ℓ)` / `max_bonddim(H)`）."
function bonddim(H::SparseIMPO)
    nl = space_l(H[1])
    all(W -> space_l(W) == space_r(W) == nl, H.Ws) ||
        throw(ArgumentError("non-uniform level structure (rectangular Schur chain); " *
                            "use bonddim(H, ℓ) or max_bonddim(H)"))
    return nl
end
max_bonddim(H::SparseIMPO) = maximum(space_r(W) for W in H.Ws)

"`phydim(H, i)`: site `i` 的物理维（unit cell 内允许逐站不同）。"
phydim(H::SparseIMPO, i::Integer) = size(H[i].A, 2)
phydims(H::SparseIMPO) = [size(H[ℓ].A, 2) for ℓ in 1:length(H)]

"""
    isidentitylevel(H, i) -> Bool

Whether level `i` is an identity level (its transfer contains only the physical
identity operator): always true for the first level; the closing unit corner
(`i == m == n` of a square site) and identity `(i, i)` diagonal blocks on every
site also qualify. On rectangular sites the `(i, i)` diagonal block may be
structurally absent (`i` beyond the interior rows/columns) — such a cut makes
the level transfer nilpotent, not identity, so it counts as non-identity.
"""
function isidentitylevel(H::SparseIMPO, i::Int)
    i == 1 && return true
    return all(H.Ws) do W
        m, n = space_l(W), space_r(W)
        (i == m == n) && return true          # 该站的闭合单位角 (m, n)
        (i > m - 1 || i > n - 1) && return false  # 对角通道在该站缺失（矩形）
        block = W.A[i - 1, :, i - 1, :]
        return isapprox(block, Matrix{scalartype(block)}(I, size(block)); atol = 1e-14)
    end
end

"""
    isemptylevel(H, i) -> Bool

Whether level `i` is a completely unused channel (mirroring MPSKit: a level is
empty if its diagonal block is structurally absent on any site). In this
package's dense representation this is equivalent to: on every site, the
diagonal block, the first-row C block, and the last-column B block are all
zero (blocks beyond a rectangular site's logical shape count as absent/zero).
Note that explicitly stored zero diagonal blocks (e.g. middle levels of
a strictly nearest-neighbor MPO) do not count as empty.
"""
function isemptylevel(H::SparseIMPO, i::Int)
    i == 1 && return false
    # 某站的闭合单位层（右角）恒非空：闭合通道的终点，环境以 I 播种
    any(W -> i == space_r(W), H.Ws) && return false
    return all(H.Ws) do W
        m, n = space_l(W), space_r(W)
        dz = (i > m - 1 || i > n - 1) || iszero(W.A[i - 1, :, i - 1, :])
        cz = (i > n - 1) || iszero(W.C[:, i - 1, :])
        bz = (i > m - 1) || iszero(W.B[i - 1, :, :])
        return dz && cz && bz
    end
end

# ---- linear algebra ----

function Base.:+(H₁::SparseIMPO, H₂::SparseIMPO)
    (length(H₁) == length(H₂)) || throw(DimensionMismatch("unit-cell lengths do not match"))
    # 逐站 FMA 后端的 Schur 直和（通道拼接、D 角相加——普适于不同通道结构，
    # 对标 MPSKit H1+H2 的块拼接语义）
    W = [H₁[i] + H₂[i] for i in 1:length(H₁)]
    return SparseIMPO(PeriodicVector(W))
end

"""
    H + λs::AbstractVector (or `λs + H`)

Add `λᵢ·I` per site (mirrors MPSKit's `H + λs`). 逐站取物理维（unit cell 内各站
物理维允许不同，例如 `phydims = [2, 3, 2]`）。直接注入各站 Schur 张量的 `D`
角（on-site 项）——能量平移不引入新通道，键维保持不变。
"""
function Base.:+(H::SparseIMPO, λs::AbstractVector{<:Number})
    (length(H) == length(λs)) || throw(DimensionMismatch("unit-cell lengths do not match"))
    Ws = Vector{SchurMPOTensor{scalartype(H)}}(undef, length(H))
    for i in 1:length(H)
        W = copy(H[i])
        d = size(W.D, 1)               # 逐站物理维
        W.D .+= λs[i] * Matrix{scalartype(H)}(I, d, d)
        Ws[i] = W
    end
    return SparseIMPO(PeriodicVector(Ws))
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

# 单个 Schur bulk 的周期 DenseIMPO 表示：`DenseIMPO([tompotensor(bulk)])`
# （原 infinite_mpo 的「D 并入恒等通道」变体删除——与 tompotensor 的 Schur
# 布局重复，且在含 NN 项的 bulk 上语义不同）。
