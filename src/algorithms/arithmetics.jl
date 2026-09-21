# ---------------- naive exact constructors exact_* (benchmark references, debug only) ----------------
#
# MPSKit semantic references (src/states/finitemps.jl, src/operators/mpo.jl,
# abstractmpo.jl):
# - `FiniteMPS + FiniteMPS`, `FiniteMPO + FiniteMPO`, `DenseIMPO * DenseIMPO`:
#   naive exact construction (bond-dimension direct sum / fuse) with **no
#   variational compression**, requiring equal lengths (check_length);
# - `fuse_mul_mpo`: `(O1*O2)` has physical out = O1's u, physical in = O2's d,
#   with the middle physical index m = O1's d = O2's u, and
#   bond dimension = product of the two bond dimensions;
# - scalar multiplication scales only the first tensor (see infinitempo.jl).
#
# This file only contains the naive exact constructors: inputs/outputs are
# `PeriodicVector{Array}` periodic tensor strings (plain Vector inputs are
# accepted), with no normalization or compression. **Debug only** (as the
# reference for the iterative mult / add / hadamard).
# Iterative versions: mult in mult.jl, add in add.jl, hadamard in hadamard.jl.

# ---- naive exact construction kernels (direct sum / fuse / zip) ----

"Rank-3 bond direct sum (MPS addition kernel):
`A12[(al,bl), s, (ar,br)] = blockdiag(A1, A2)`."
function _naive_sum_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    size(A1, 2) == size(A2, 2) || throw(DimensionMismatch("MPS addition requires equal per-site physical dimensions"))
    out = zeros(T, size(A1, 1) + size(A2, 1), size(A1, 2), size(A1, 3) + size(A2, 3))
    out[1:size(A1, 1), :, 1:size(A1, 3)] .= A1
    out[size(A1, 1)+1:end, :, size(A1, 3)+1:end] .= A2
    return out
end

"Rank-4 bond direct sum (MPO addition kernel):
`W12[(wl1,wl2), u, (wr1,wr2), d] = blockdiag(W1, W2)`."
function _naive_sum_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    (size(W1, 2) == size(W2, 2) && size(W1, 4) == size(W2, 4)) ||
        throw(DimensionMismatch("MPO addition requires equal per-site physical (u,d) dimensions"))
    out = zeros(T, size(W1, 1) + size(W2, 1), size(W1, 2),
                size(W1, 3) + size(W2, 3), size(W1, 4))
    out[1:size(W1, 1), :, 1:size(W1, 3), :] .= W1
    out[size(W1, 1)+1:end, :, size(W1, 3)+1:end, :] .= W2
    return out
end

"""
    _naive_hadamard_tensor(A1, A2) -> A12

Rank-3 zip (Hadamard/Schur product kernel): the two virtual legs are zipped
together while the **physical leg is shared**,
`A12[(a,c), s, (b,e)] = A1[a,s,b]·A2[c,s,e]`. The two virtual chains are
independent, so the periodic trace factorizes:
`tr(∏A12) = tr(∏A1)·tr(∏A2)` — the MPS representation of the elementwise
waveform product `c12 = c1 .* c2` (physical dimension unchanged, bond dimension
= product of the two bond dimensions).
"""
function _naive_hadamard_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    a, s, b = size(A1)
    c, s2, e = size(A2)
    (s2 == s) || throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    # (a,c)/(b,e) fusion order: a and b are the major indices; Kronecker product
    # per physical slice (TensorOperations @tensor does not support the batched
    # form where s appears uncontracted in both operands)
    out = similar(A1, a * c, s, b * e)
    for k in 1:s
        out[:, k, :] = kron(A2[:, k, :], A1[:, k, :])
    end
    return out
end

# ---------------- exact_*: naive exact constructors (debug only) ----------------

"""
    exact_mult(W1, W2) -> PeriodicVector{Array{T,4}}
    exact_mult(W, ψ) -> PeriodicVector{Array{T,3}}

Naive exact MPO multiplication (mpo*mpo: bond dimension = product of the two
bond dimensions, mirroring MPSKit's `fuse_mul_mpo`; mpo*mps: bond dimension =
W bond × ψ bond). Inputs/outputs are `PeriodicVector{Array}` periodic tensor
strings; no normalization or compression is performed. **Debug only** (as the
reference for the variational `mult`); in the periodic trace representation
block diagonals telescope away, so
`tr(∏exact_mult(W1,W2)) = tr(∏W1)·tr(∏W2)` does NOT hold — compare operator
amplitudes of products via the dense representation (see the debug/ tests).
"""
function exact_mult(W1::AbstractVector{<:Array{T,4}}, W2::AbstractVector{<:Array{T,4}}) where {T}
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO multiplication requires equal lengths (mirrors MPSKit check_length)"))
    return PeriodicVector([_naive_mul_tensor(W1[ℓ], W2[ℓ]) for ℓ in 1:length(W1)])
end

function exact_mult(W::AbstractVector{<:Array{T,4}}, ψ::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    return PeriodicVector([fuse(W[_mod1(ℓ, length(W))], ψ[ℓ]) for ℓ in 1:length(ψ)])
end

"""
    exact_add(a1, a2) -> PeriodicVector

Naive exact addition (mps+mps / mpo+mpo): per-site bond-dimension direct sum
`blockdiag(a1[ℓ], a2[ℓ])`, requiring equal lengths and equal per-site physical
dimensions (mirrors MPSKit check_length). In the periodic trace representation
this is **exactly additive**:
`tr(∏exact_add(a1,a2)) = tr(∏a1) + tr(∏a2)`.
Inputs/outputs are `PeriodicVector{Array}` tensor strings; no normalization is
performed. **Debug only**.
"""
function exact_add(ψ1::AbstractVector{<:Array{T,3}}, ψ2::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("MPS addition requires equal lengths (mirrors MPSKit check_length)"))
    return PeriodicVector([_naive_sum_tensor(ψ1[ℓ], ψ2[ℓ]) for ℓ in 1:length(ψ1)])
end

function exact_add(W1::AbstractVector{<:Array{T,4}}, W2::AbstractVector{<:Array{T,4}}) where {T}
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO addition requires equal lengths (mirrors MPSKit check_length)"))
    return PeriodicVector([_naive_sum_tensor(W1[ℓ], W2[ℓ]) for ℓ in 1:length(W1)])
end

"""
    exact_hadamard(a1, a2) -> PeriodicVector{Array{T,3}}

Naive exact Hadamard product (the infinite-MPS generalization of the
elementwise product): per-site zipped virtual legs with a shared physical leg,
`A12[(a,c), s, (b,e)] = a1[a,s,b]·a2[c,s,e]`. Requires equal lengths and equal
per-site physical dimensions (the physical dimension is unchanged; the bond
dimension is the product of the two). The factorizing periodic trace yields the
**elementwise waveform product**:
`dense(exact_hadamard(a1,a2)) = dense(a1) .* dense(a2)`.
Inputs/outputs are `PeriodicVector{Array}` tensor strings; no normalization is
performed. **Debug only**.
"""
function exact_hadamard(ψ1::AbstractVector{<:Array{T,3}}, ψ2::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1[ℓ], 2) == size(ψ2[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    return PeriodicVector([_naive_hadamard_tensor(ψ1[ℓ], ψ2[ℓ]) for ℓ in 1:length(ψ1)])
end
