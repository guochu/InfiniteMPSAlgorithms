# ---------------- SchurMPOTensor (ported from MPSKit src/operators/jordanmpotensor.jl) ----------------
#
# Upper-triangular block-matrix representation of an MPO (symmetry-free, plain
# Array version). The physical index order matches the rest of the package
# (TEMPO convention `W[wl, u, wr, d]`, bond indices in slots 1 and 3):
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# Block storage (virtual level count nlvls = a + 2, with unit levels first/last):
# - `A::Array{T,4}`: `(a, d, a, d)`, middle blocks (rows/columns 2..end-1);
# - `B::Array{T,3}`: `(a, d, d)`, middle → last column;
# - `C::Array{T,3}`: `(d, a, d)`, first row → middle;
# - `D::Array{T,2}`: `(d, d)`, first row / last column (the complete on-site term);
# - `[1,1]` and `[end,end]` are implied physical identities (the `BraidingTensor`
#   in MPSKit).

"""
    SchurMPOTensor{T}

Schur (upper-triangular) block-matrix tensor of an MPO (symmetry-free
version); see the module comment. Supports block-matrix style access `W[i, j]`
returning the `(d, d)` local operator.
"""
struct SchurMPOTensor{T}
    A::Array{T,4}
    B::Array{T,3}
    C::Array{T,3}
    D::Array{T,2}
end

Base.copy(W::SchurMPOTensor) = SchurMPOTensor(copy(W.A), copy(W.B), copy(W.C), copy(W.D))
scalartype(::Type{SchurMPOTensor{T}}) where {T} = T
scalartype(W::SchurMPOTensor) = scalartype(typeof(W))

"nlvls(W): the number of virtual levels of the Schur tensor (= bond channels + 2
unit levels)."
nlvls(W::SchurMPOTensor) = size(W.A, 1) + 2

# ---- block-matrix style access: W[i, j] → (d, d) local operator ----

function Base.getindex(W::SchurMPOTensor{T}, i::Int, j::Int) where {T}
    n = nlvls(W)
    d = size(W.A, 2)
    if (i == 1 && j == 1) || (i == n && j == n)
        return Matrix{T}(I, d, d)
    elseif i == 1 && j == n
        return W.D
    elseif i == 1
        return W.C[:, j - 1, :]
    elseif j == n
        return W.B[i - 1, :, :]
    elseif 1 < i < n && 1 < j < n
        return W.A[i - 1, :, j - 1, :]
    end
    return zeros(T, d, d)
end

function Base.setindex!(W::SchurMPOTensor{T}, O::AbstractMatrix, i::Int, j::Int) where {T}
    (size(O, 1) == size(O, 2) == size(W.A, 2)) ||
        throw(DimensionMismatch("local operator must be $(size(W.A, 2))×$(size(W.A, 2))"))
    n = nlvls(W)
    if (i == 1 && j == 1) || (i == n && j == n)
        # unit diagonal blocks: overwriting is not allowed (identity preserved)
        return W
    elseif i == 1 && j == n
        W.D .= O
    elseif i == 1
        W.C[:, j - 1, :] .= O
    elseif j == n
        W.B[i - 1, :, :] .= O
    elseif 1 < i < n && 1 < j < n
        W.A[i - 1, :, j - 1, :] .= O
    else
        throw(ArgumentError("lower-triangular Schur block ($i, $j) is identically zero and cannot be assigned"))
    end
    return W
end

# ---- constructors ----

function _mpoham_scalar_type(W::AbstractMatrix)
    T = Union{}
    for v in W
        v isa Missing && continue
        T = promote_type(T, v isa Number ? typeof(v) : eltype(v))
    end
    return T === Union{} ? Float64 : T
end

"""
    SchurMPOTensor(W::AbstractMatrix) -> SchurMPOTensor

Construct from an `n × n` operator matrix: `W[i, j]` is the `(d, d)` local
operator from row `i` (left level) to column `j` (right level); entries may be
`Missing`, `Number`s, or matrices. `[1,1]` and `[end,end]` are implied
identities (entries should be `1` or `Missing`).
"""
function SchurMPOTensor(W::AbstractMatrix)
    (size(W, 1) == size(W, 2)) || throw(ArgumentError("W must be a square matrix"))
    n = size(W, 1)
    (n >= 2) || throw(ArgumentError("W needs at least 2 levels (unit levels first/last)"))
    T = _mpoham_scalar_type(W)
    # physical dimension: taken from the first matrix entry
    d = 0
    for v in W
        if v isa AbstractMatrix
            d = size(v, 1)
            break
        end
    end
    (d > 0) || throw(ArgumentError("no (d, d) local operator entry found in W"))
    # validate the unit corners
    for (i, j) in ((1, 1), (n, n))
        v = W[i, j]
        (v isa Missing || v isa Number && isone(v)) ||
            throw(ArgumentError("W[$i, $j] must be 1 or Missing (unit levels imply identity)"))
    end
    J = SchurMPOTensor(zeros(T, n - 2, d, n - 2, d), zeros(T, n - 2, d, d),
                       zeros(T, d, n - 2, d), zeros(T, d, d))
    for i in 1:n, j in 1:n
        v = W[i, j]
        v isa Missing && continue
        iszero(v) && continue
        v isa Number ? J[i, j] = Matrix{T}(v * I, d, d) : (J[i, j] = v)
    end
    return J
end

"""
    tompotensor(W::SchurMPOTensor) -> Array{T,4}

Densify into the package's 4-index MPO tensor `(wl, u, wr, d)`: the identity
channels live at level `1` and level `nlvls`, i.e.
`Wd[1,:,1,:] = Wd[end,:,end,:] = I`.
"""
function tompotensor(W::SchurMPOTensor)
    T = scalartype(W)
    d = size(W.A, 2)
    n = nlvls(W)
    Wd = zeros(T, n, d, n, d)
    Wd[1, :, 1, :] .= Matrix{T}(I, d, d)
    Wd[n, :, n, :] .= Matrix{T}(I, d, d)
    Wd[1, :, n, :] .= W.D
    for j in 2:(n - 1)
        Wd[1, :, j, :] .= W.C[:, j - 1, :]
    end
    for i in 2:(n - 1)
        Wd[i, :, n, :] .= W.B[i - 1, :, :]
        for j in 2:(n - 1)
            Wd[i, :, j, :] .= W.A[i - 1, :, j - 1, :]
        end
    end
    return Wd
end

function Base.:+(W₁::SchurMPOTensor, W₂::SchurMPOTensor)
    n₁, n₂ = nlvls(W₁), nlvls(W₂)
    (n₁ == n₂ && size(W₁.A, 2) == size(W₂.A, 2)) ||
        throw(ArgumentError("Schur tensor level counts / physical dimensions do not match"))
    return SchurMPOTensor(W₁.A + W₂.A, W₁.B + W₂.B, W₁.C + W₂.C, W₁.D + W₂.D)
end

function Base.:*(λ::Number, W::SchurMPOTensor)
    # scale only the physical terms (the C, D blocks and the closed end of B);
    # the identity channel is left untouched
    return SchurMPOTensor(copy(W.A), λ .* W.B, λ .* W.C, λ .* W.D)
end
