# ---------------- DenseIMPO strict algebra (compression-free `*`) ----------------
#
# 严格（无压缩、无规范化的）DenseIMPO 代数运算，直接重载 `Base.*`。这些运算
# 只支持 `DenseIMPO`/`DenseIMPS`——`CanonicalIMPO`/`CanonicalIMPS` 不参与分派
# （需要时显式转换：`CanonicalIMPO(collect(W.Ws))`、
# `CanonicalIMPS(collect(ψ.As))`）。

"""
    fuse(W, AL) -> Array{T,3}

Per-site fusion of an MPO tensor with a left-orthogonal MPS tensor:
`B[(wl·bl), u, (wr·br)] = Σ_d W[wl, u, wr, d] · AL[bl, d, br]`.
"""
function fuse(W::AbstractArray{T,4}, AL::AbstractArray{T,3}) where {T}
    wl, u, wr, _ = size(W)
    bl, _, br = size(AL)
    @tensor B5[wl, bl, u, wr, br] := W[wl, u, wr, d] * AL[bl, d, br]
    return reshape(B5, wl * bl, u, wr * br)
end

"""
    _naive_mul_tensor(W1, W2) -> W12

Rank-4 bond fusion (MPO multiplication kernel, mirroring MPSKit's
`fuse_mul_mpo`): the middle physical index `m` = W1's d = W2's u, and the bond
dimension = product of the two bond dimensions (they need not be equal):

```julia
W12[(wl1, wl2), u, (wr1, wr2), d] = Σ_m W1[wl1, u, wr1, m] · W2[wl2, m, wr2, d]
```
"""
function _naive_mul_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    size(W1, 4) == size(W2, 2) ||
        throw(DimensionMismatch("MPO multiplication requires W1's physical in (d) to match W2's physical out (u)"))
    # this package's TensorOperations version does not support tuple composite
    # indices; use a flat intermediate tensor + reshape (as in fuse)
    @tensor W6[wl1, wl2, u, wr1, wr2, d] :=
        W1[wl1, u, wr1, m] * W2[wl2, m, wr2, d]
    return reshape(W6, size(W1, 1) * size(W2, 1), size(W1, 2),
                   size(W1, 3) * size(W2, 3), size(W2, 4))
end

"""
    Base.:*(W::DenseIMPO, W2::DenseIMPO) -> DenseIMPO
    Base.:*(W::DenseIMPO, ψ::DenseIMPS) -> DenseIMPS

Strict (compression-free) MPO composition / application — the naive exact
construction as a typed operator, mirroring MPSKit's naive `*`:
`DenseIMPO * DenseIMPO` composes with kernel
[`_naive_mul_tensor`](@ref) (bond dimension = product of the two bond
dimensions), `DenseIMPO * DenseIMPS` applies with kernel
[`fuse`](@ref) (bond dimension = W bond × ψ bond). The `*` output unit-cell
length is the **least common multiple** of the two inputs（单胞不同时逐周期
平铺收缩）; per-site physical dimensions must agree. No canonicalization or
normalization is performed — the raw fused tensor string is returned
(`DenseIMPO`/`DenseIMPS`). For the canonicalized representative use
`CanonicalIMPO(W * W2)` / `CanonicalIMPS(collect(W * ψ))`; for the variational
compression use `mult(W, ψ, alg)` / `mult(W, W2, alg)`.
"""
function Base.:*(W::DenseIMPO, W2::DenseIMPO)
    (length(W2) % length(W) == 0) ||
        throw(DimensionMismatch("incompatible MPO unit-cell lengths"))
    return DenseIMPO([_naive_mul_tensor(W[_mod1(ℓ, length(W))], W2[ℓ])
                      for ℓ in 1:length(W2)])
end

function Base.:*(W::DenseIMPO, ψ::DenseIMPS)
    L, N = length(W), length(ψ)
    P = lcm(L, N)
    all(size(W[_mod1(ℓ, L)], 4) == size(ψ[_mod1(ℓ, N)], 2) for ℓ in 1:P) ||
        throw(DimensionMismatch("incompatible per-site physical dimensions of MPS and MPO"))
    return DenseIMPS([fuse(W[_mod1(ℓ, L)], ψ[_mod1(ℓ, N)]) for ℓ in 1:P])
end

# ---------------- 标量代数 ----------------

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

# ---------------- Dense ↔ Canonical 转换与 Dense 视图（vectorize/devectorize/superoperator） ----------------

"""
    CanonicalIMPO(W::DenseIMPO; kwargs...) -> CanonicalIMPO

Mixed-canonicalize a plain MPO（等价于 `CanonicalIMPO(W.Ws; kwargs...)`）。
"""
CanonicalIMPO(W::DenseIMPO; kwargs...) = CanonicalIMPO(W.Ws; kwargs...)

"`DenseIMPO(W)`: convert back to a plain MPO using the left-canonical tensor
string `W.AL`. `tr(∏AL)` is the operator amplitude invariant under gauge
transformations (including phases) (= the construction input amplitude / a
positive real λ), whereas `tr(∏AC)` is C-matrix weighted and gauge dependent;
`AL` is used here to keep the conversion unique."
DenseIMPO(W::CanonicalIMPO) = DenseIMPO(collect(W.AL))

"`vectorize(W::DenseIMPO) -> DenseIMPS`：[`vectorize`](@ref) 的未规范化副本
（纯融合视图，与 `CanonicalIMPO` 输入的规范携带版共用同一融合约定）。"
vectorize(W::DenseIMPO) = DenseIMPS(asmps_view(W.Ws))
vectorize(W::SparseIMPO) = vectorize(DenseIMPO(W))

"`devectorize(ψ::DenseIMPS) -> DenseIMPO`：[`devectorize`](@ref) 的未规范化
副本（纯拆分，各站融合物理维须为完全平方数，`r` 可逐站不同）。"
function devectorize(ψ::DenseIMPS)
    rs = _local_square_rdims(phydims(ψ))
    return DenseIMPO(mps_view_to_mpo(collect(ψ.As); dus = rs, dds = rs))
end

"""
    superoperator(W; side = :left) -> DenseIMPO

The left/right multiplication superoperator of an MPO, as an MPO on the
doubled (bra ⊗ ket) space with the fused index convention of
[`vectorize`](@ref) (`f = u + du·(d - 1)`, `u` the bra / row leg fast):

- `side = :left` (`= superoperator(W, identityimpo(dus))`): `𝓦[bl, f', br, f] = W[bl, u', br, u]·δ[d', d]`,
  so that `𝓦 · vec(X)` is `vec(W·X)` (W multiplies from the left);
- `side = :right` (`= superoperator(identityimpo(dus), transpose(W))`): `𝓦[bl, f', br, f] = δ[u', u]·W[bl, d, br, d']`,
  so that `𝓦 · vec(X)` is `vec(X·W)` (W multiplies from the right).

The bond dimensions are unchanged (the spectator channel carries trivial
δ-bonds) and the physical dimension becomes `du·dd` (square operators only).
The output is a plain `DenseIMPO` (not canonical). Combined with
[`vectorize`](@ref) this turns operator–operator products into operator–state
problems, e.g. for `mult`:

    mult(superoperator(W1; side = :left),  vectorize(W2)) == vectorize(W1 * W2)
    mult(superoperator(W2; side = :right), vectorize(W1)) == vectorize(W1 * W2)

A typical finite-T purification generator is the sum of the two channel
superoperators, `𝓦_L(H) + 𝓦_R(H)` (= `H ⊗ I + I ⊗ Hᵀ`).
"""
function superoperator(W::DenseIMPO; side::Symbol = :left)
    dus = phydims(W)
    for ℓ in 1:length(W)
        size(W[ℓ], 4) == dus[ℓ] ||
            throw(ArgumentError("superoperator requires square operators (u == d) at site $ℓ"))
    end
    I = identityimpo(scalartype(W), dus)
    side === :left && return superoperator(W, I)
    side === :right && return superoperator(I, transpose(W))
    throw(ArgumentError("side must be :left or :right, got $side"))
end
superoperator(W::SparseIMPO; side::Symbol = :left) = superoperator(DenseIMPO(W); side)
superoperator(W::CanonicalIMPO; side::Symbol = :left) = superoperator(DenseIMPO(W); side)
