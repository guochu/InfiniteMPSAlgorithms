# ---------------- iterative hadamard (infinite-MPS generalization of the Hadamard/Schur product) ----------------
#
# Elementwise waveform product `c₁₂ = c₁ .* c2`: virtual legs zipped per site,
# physical leg shared (kernel `_naive_hadamard_tensor` in arithmetics.jl);
# the physical dimension is unchanged and the bond dimension becomes the product
# of the two. The iterative version runs the factorized zip engine (`_lazy_sweeps`
# on the `ZipKet` target — the fused zip tensors are never materialized, the
# environment/本地映射直接消费 (ψ1, ψ2) 因子对); the naive/debug reference is
# `naive_hadamard` (`_algebra_result` in compress.jl).

"""
    hadamard(ψ₁, ψ₂) -> CanonicalIMPS

Naive exact construction (no compression; the output bond
dimension = D₁·D₂, which is inherently large). The waveform is multiplied
pointwise, `c₁₂[(s…)] = c₁[(s…)] · c₂[(s…)]`, with the **physical dimension
unchanged**; the virtual legs are zipped per site with a shared physical leg,
and the two virtual chains are independent so the periodic trace factorizes,
`tr(∏A12) = tr(∏AL₁)·tr(∏AL₂)` exactly. The zip is generally **not** in
canonical form; the returned normalized state differs from `c₁.*c₂` by a
constructor-determined positive real scalar (a ray representative). Use
[`exact_hadamard`](@ref) for the raw tensor string with strict pointwise
amplitudes.
"""
function hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    K = [_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:length(ψ1)]
    return CanonicalIMPS(K)
end

"""
    hadamard(ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPS

Compute-on-the-fly variational compression of the Hadamard/Schur product to
the bond dimension `alg.D`: the zip target is consumed in **factorized form**
(the (ψ1, ψ2) tensor pairs go straight into the environment/局部映射收缩 — the
fused zip tensors are never materialized) and
variationally compressed with the positional algorithm
object `alg` (VOMPS/IDMRG), starting from the deterministic
`svdguess_hadamard` initial state. Convergence is judged by the Galerkin
residual alone (no overlap is computed — same contract as `mult`/`compress`).
For a naive reference implementation (construct the
whole family, then compress) see [`naive_hadamard`](@ref).
"""
hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    _hadamard(ψ1, ψ2, alg, nothing; D = alg.D)

function _hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    N = length(ψ1)
    # 因子化 zip target（compute-on-the-fly）：融合 zip 张量从不物化，环境与
    # 局部映射直接消费 (ψ1, ψ2) 因子对（先与环境收缩——键优先的显式分步 GEMM，
    # 见 `_zip_push_left`/`_mapAC_zip`）；C 的融合序 = `kron(C2, C1)` 与 zip
    # kernel 的 (ψ2 主, ψ1 次) 键序逐位对齐。
    ket = ZipKet(
        ℓ -> ψ1.AL[ℓ], ℓ -> ψ2.AL[ℓ],
        ℓ -> ψ1.AR[ℓ], ℓ -> ψ2.AR[ℓ],
        ℓ -> ψ1.AC[ℓ], ℓ -> ψ2.AC[ℓ],
        ℓ -> ψ1.C[ℓ],  ℓ -> ψ2.C[ℓ],
    )
    x0 = x0 === nothing ? svdguess_hadamard(ψ1, ψ2, D) : x0
    return _lazy_sweeps(ket, x0, N; alg = alg, tol = alg.tol,
                        maxiter = alg.maxiter, verbosity = alg.verbosity)
end

# ---------------- naive_hadamard (debug: naive family construction + optional compression) ----------------

"""
    naive_hadamard(ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> CanonicalIMPS

Naive reference implementation of [`hadamard`](@ref) (debug only): first
construct the complete zip family (memory O(N·D₁D₂)), then
compress to `alg.D` with the positional algorithm object `alg`. No overlap is
computed (same contract as [`hadamard`](@ref)). Large input
bond dimensions produce huge intermediate families — use [`hadamard`](@ref)
for production use.
"""
function naive_hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
                        alg::Union{VOMPS,IDMRG})
    D = alg.D
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    K = [_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:length(ψ1)]
    return _algebra_result(K, D, alg)
end

# ---------------- svdguess_hadamard (deterministic initial guess) & hadamard! (in-place) ----------------

"""
    svdguess_hadamard(ψ₁, ψ₂, D) -> CanonicalIMPS
    svdguess_hadamard(A1s::PeriodicVector{<:Array{T,3}},
                      A2s::PeriodicVector{<:Array{T,3}}, D) -> Vector{Array{T,3}}

Deterministic initial guess of the iterative [`hadamard`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_hadamard`): the pointwise product's site tensors
are generated one at a time, **with the streaming carry absorbed during
construction** (contracting into the zip inputs before fusing — the naive
product tensor is never materialized), and streamed right→left through a
truncating right-orthogonalization with bond cap `D`
([`_lazy_svd_guess`](@ref)); the ring's wrap bond is Schmidt-truncated at
site 1. Every output bond is ≤ `D`.

The bare-tensor method takes the site-tensor strings directly (e.g. `ψ.AL` /
`ψ.AR` of a [`CanonicalIMPS`](@ref)) and returns the right-gauge tensor
string — the low-level entry point for downstream packages; the
`CanonicalIMPS` method is a thin wrapper that re-canonicalizes its output.
"""
function svdguess_hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, D::Int)
    return CanonicalIMPS(svdguess_hadamard(ψ1.AL, ψ2.AL, D))
end

function svdguess_hadamard(A1s::PeriodicVector{<:Array{T,3}},
                           A2s::PeriodicVector{<:Array{T,3}}, D::Int) where {T}
    (length(A1s) == length(A2s)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    N = length(A1s)
    all(size(A1s[ℓ], 2) == size(A2s[ℓ], 2) for ℓ in 1:N) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    # carry 在构造时吸收（kron(A2, A1) 的融合序：左腿 (c 慢, a 快)、右腿 (e 慢, b 快)）：
    # B'[(c,a), s, f] = Σ_{e,b} A2[c,s,e]·A1[a,s,b]·carry[(e,b), f]
    # 先用 A1 的 b 腿吸收 carry、再用 A2 的 e 腿收缩 —— 两个矩阵乘，不落地大张量
    site = (ℓ, carry) -> begin
        A1 = A1s[ℓ]; A2 = A2s[ℓ]
        carry === nothing && return _naive_hadamard_tensor(A1, A2)
        a, s, b = size(A1); c, _, e = size(A2)
        f = size(carry, 2)
        l4 = reshape(carry, b, e, f)                     # carry 行 = (e-1)·b + b：b 最快
        B3 = Array{promote_type(eltype(A1), eltype(carry)),4}(undef, c, a, s, f)
        @inbounds for k in 1:s
            # Y[a,e,f] = Σ_b A1[a,s,b]·l4[b,e,f]
            Y = reshape(view(A1, :, k, :) * reshape(l4, b, e * f), a, e, f)
            # B'[c,a,f] = Σ_e A2[c,s,e]·Y[a,e,f]
            Bk = view(A2, :, k, :) * reshape(permutedims(Y, (2, 1, 3)), e, a * f)
            B3[:, :, k, :] = reshape(Bk, c, a, f)
        end
        # 左腿 flatten 与 naive 一致：行 = (c-1)·a + a（c 慢 a 快）
        return reshape(permutedims(B3, (2, 1, 3, 4)), a * c, s, f)
    end
    return _lazy_svd_guess(site, N, D)
end

"""
    hadamard!(out, ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> out

In-place [`hadamard`](@ref): `out` is the user-provided state to be optimized
as the initial guess. The target bond dimension is taken from the bond profile
of `out` (its bond profile is first brought to uniform `D = max_bonddim(out)`
with [`changebond!`](@ref)); `alg.D` is ignored. The optimized result is
written back into `out`.
"""
function hadamard!(out::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y = _hadamard(ψ1, ψ2, alg, out; D = D)
    return _copyinto!(out, y)
end
