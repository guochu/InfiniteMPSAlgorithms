# ---------------- DenseIMPS linear algebra (strict operations & overlaps) ----------------
#
# 严格（无压缩、无规范化的）DenseIMPS 线性运算：scalar 乘法、Hadamard 乘积，
# 以及基于转移矩阵主导本征值的重叠泛函（dot / norm / fidelity / infidelity /
# distance / distance2）。这些运算只支持 `DenseIMPS`——`CanonicalIMPS` 不参与
# 分派（需要时显式 `CanonicalIMPS(collect(ψ.As))` 转换）。

# ---- scalar multiplication (MPSKit convention: scale only the first tensor;
#      scaling every tensor would change the state to αᴺ·ψ) ----

function Base.:*(α::Number, ψ::DenseIMPS)
    out = [copy(a) for a in ψ.As]
    out[1] = α .* out[1]
    return DenseIMPS(out)
end
Base.:*(ψ::DenseIMPS, α::Number) = α * ψ
Base.:/(ψ::DenseIMPS, α::Number) = (1 / α) * ψ
Base.:-(ψ::DenseIMPS) = (-one(scalartype(ψ))) * ψ

"""
    copyphyims(ψ::DenseIMPS) -> DenseIMPO

Reinterpret the MPS `ψ` as an operator chain on the same bond dimension
(reference: FiniteMPSAlgorithms' `copyphydims`): every site tensor
`A[aL, p, aR]` becomes the physical-space diagonal `W[aL, po, aR, pi] =
A[aL, po, aR]·δ_{po,pi}`, i.e. the physical dimensions are copied to both the
bra and the ket side. Applying the result to another MPS reproduces the
element-wise (Hadamard) product, so `copyphyims(ψ1) * ψ2 ≡ hadamard(ψ1, ψ2)`
(the alignment of `mult` and `hadamard`).
"""
function copyphyims(ψ::DenseIMPS)
    T = scalartype(ψ)
    data = Vector{Array{T,4}}(undef, length(ψ))
    for ℓ in 1:length(ψ)
        A = ψ[ℓ]
        d = size(A, 2)
        W = zeros(T, size(A, 1), d, size(A, 3), d)
        for p in 1:d
            W[:, p, :, p] .= A[:, p, :]
        end
        data[ℓ] = W
    end
    return DenseIMPO(data)
end

# ---- Hadamard (Schur) product ----

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
    # form where s appears uncontracted in both operands). The slices are
    # rank-3: FiniteMPSAlgorithms extends Base.kron to same-rank N-d arrays
    # (no such method in Base itself before Julia 1.11).
    out = similar(A1, a * c, s, b * e)
    for k in 1:s
        out[:, k, :] = kron(A2[:, k, :], A1[:, k, :])
    end
    return out
end

"""
    hadamard(ψ1::DenseIMPS, ψ2::DenseIMPS) -> DenseIMPS

Strict (compression-free) Hadamard/Schur product — the naive exact
construction as a typed operator: per-site zipped virtual legs with a shared
physical leg, `A12[(a,c), s, (b,e)] = ψ1[a,s,b]·ψ2[c,s,e]` (kernel
[`_naive_hadamard_tensor`](@ref)). Requires equal lengths and equal per-site
physical dimensions; the physical dimension is unchanged and the bond
dimension is the product of the two. The factorizing periodic trace yields the
**elementwise waveform product**, `dense(hadamard(a,b)) = dense(a) .* dense(b)`.
No canonicalization or normalization is performed — the raw zipped tensor
string is returned. For the variational compression use `hadamard(ψ1, ψ2, alg)`.
"""
function hadamard(ψ1::DenseIMPS, ψ2::DenseIMPS)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1[ℓ], 2) == size(ψ2[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    return DenseIMPS([_naive_hadamard_tensor(ψ1[ℓ], ψ2[ℓ]) for ℓ in 1:length(ψ1)])
end

# ---- overlaps (transfer-matrix dominant eigenvalues) ----

"""
    LinearAlgebra.dot(ψ1::DenseIMPS, ψ2::DenseIMPS; krylovdim = Defaults.krylovdim)

`⟨ψ1|ψ2⟩`: dominant eigenvalue of the double-layer transfer matrix
(KrylovKit Arnoldi; plain-array counterpart of the [`CanonicalIMPS`](@ref)
method — no canonicalization is performed on either side).
"""
function LinearAlgebra.dot(ψ1::DenseIMPS, ψ2::DenseIMPS;
                           krylovdim::Int = Defaults.krylovdim)
    T = promote_type(scalartype(ψ1), scalartype(ψ2))
    v0 = vec(Matrix{T}(I, bonddim(ψ1, 1), bonddim(ψ2, 1)))
    tm = TransferMatrix(ψ2.As, ψ1.As)
    vals, _, _ = _eigsolve(tm, v0, 1, :LM; ishermitian = false, krylovdim = krylovdim)
    λ = vals[1]
    return λ isa Number ? λ : only(λ)
end

"`LinearAlgebra.norm(ψ::DenseIMPS) = sqrt(|⟨ψ, ψ⟩|)`."
function LinearAlgebra.norm(ψ::DenseIMPS)
    return sqrt(abs(dot(ψ, ψ)))
end

"""
    fidelity(ψ1::DenseIMPS, ψ2::DenseIMPS) -> Real
    infidelity(ψ1::DenseIMPS, ψ2::DenseIMPS) -> Real

`fidelity = |⟨ψ1|ψ2⟩| / (‖ψ1‖·‖ψ2‖) ∈ [0, 1]`: the normalized ring overlap —
invariant under independent overall phases and normalizations of the two
states. `infidelity = 1 − fidelity`.
"""
fidelity(ψ1::DenseIMPS, ψ2::DenseIMPS) =
    abs(dot(ψ1, ψ2)) / (norm(ψ1) * norm(ψ2))
infidelity(ψ1::DenseIMPS, ψ2::DenseIMPS) = 1 - fidelity(ψ1, ψ2)

# 扩展 FiniteMPSAlgorithms 的 distance/distance2（保持其 plain-array 方法
# 与本模块 DenseIMPS 方法在同一个函数对象上，`using` 再导出行为不变）。
import FiniteMPSAlgorithms: distance, distance2

"""
    distance2(ψ1::DenseIMPS, ψ2::DenseIMPS) -> Real
    distance(ψ1::DenseIMPS, ψ2::DenseIMPS) -> Real

`‖ψ1 − ψ2‖² = ‖ψ1‖² + ‖ψ2‖² − 2·Re⟨ψ1|ψ2⟩` (the absolute value guards against
negative rounding; semantics aligned with FiniteMPSAlgorithms'
`distance(::CanonicalMPS, ::CanonicalMPS)`). `distance = sqrt(distance2)`.
"""
function distance2(ψ1::DenseIMPS, ψ2::DenseIMPS)
    sA = real(dot(ψ1, ψ1))
    sB = real(dot(ψ2, ψ2))
    c = dot(ψ1, ψ2)
    return abs(sA + sB - 2 * real(c))
end
distance(ψ1::DenseIMPS, ψ2::DenseIMPS) = sqrt(distance2(ψ1, ψ2))
