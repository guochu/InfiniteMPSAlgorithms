# ---------------- DenseIMPO strict algebra (compression-free `*`) ----------------
#
# 严格（无压缩、无规范化的）DenseIMPO 代数运算，直接重载 `Base.*`。这些运算
# 只支持 `DenseIMPO`/`DenseIMPS`——`CanonicalIMPO`/`CanonicalIMPS` 不参与分派
# （需要时显式转换：`CanonicalIMPO(collect(W.Ws))`、
# `CanonicalIMPS(collect(ψ.As))`）。

"""
    Base.:*(W::DenseIMPO, W2::DenseIMPO) -> DenseIMPO
    Base.:*(W::DenseIMPO, ψ::DenseIMPS) -> DenseIMPS

Strict (compression-free) MPO composition / application — the naive exact
construction as a typed operator, mirroring MPSKit's naive `*`:
`DenseIMPO * DenseIMPO` composes with kernel
[`_naive_mul_tensor`](@ref) (bond dimension = product of the two bond
dimensions), `DenseIMPO * DenseIMPS` applies with kernel
[`fuse`](@ref) (bond dimension = W bond × ψ bond). Unit-cell lengths must be
compatible (`length(target) % length(W) == 0`). No canonicalization or
normalization is performed — the raw fused tensor string is returned
(`DenseIMPO`/`DenseIMPS`). For the canonicalized, normalized representatives
use `mult(W, W2)` / `mult(W, ψ)`; for the variational compression use
`mult(W, ψ, alg)`.
"""
function Base.:*(W::DenseIMPO, W2::DenseIMPO)
    (length(W2) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    return DenseIMPO([_naive_mul_tensor(W[_mod1(ℓ, length(W))], W2[ℓ])
                      for ℓ in 1:length(W2)])
end

function Base.:*(W::DenseIMPO, ψ::DenseIMPS)
    (length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    return DenseIMPS([fuse(W[_mod1(ℓ, length(W))], ψ[ℓ]) for ℓ in 1:length(ψ)])
end
