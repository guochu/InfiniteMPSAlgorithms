# ---------------- compress (bond-dimension reduction of a single chain, mirroring MPSKit's approximate) ----------------
#
# Find `out ≈ x` with `max_bonddim(out) = D`: the overlap-maximizing
# variational approximation of a *given* chain (the target is the chain itself,
# unlike add/mult/hadamard whose targets are naive algebraic constructions).
# The engines are shared with the algebra compressions (`_overlap_sweeps` /
# `_idmrg_sweeps` on the identity channel).

"""
    svdguess_compress(x, D) -> CanonicalIMPS
    svdguess_compress(ALs::PeriodicVector{<:Array{T,3}}, D) -> Vector{Array{T,3}}

Deterministic initial guess of the iterative [`compress`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_compress`): the bond-wise SVD truncation of the
input to `D` (accurate but costly compared to [`changebond!`](@ref) zero
padding). `CanonicalIMPO` input returns the MPS-view guess (`CanonicalIMPS`
on the doubled space).

The bare-tensor method accepts the site-tensor string directly (e.g.
`ψ.AL` / `ψ.AR` of a [`CanonicalIMPS`](@ref)) and returns a plain right-gauge
tensor string truncated to `D` (wrap bond Schmidt-truncated at site 1,
[`_lazy_svd_guess`](@ref)) — the low-level entry point for downstream
packages; the `CanonicalIMPS`/`CanonicalIMPO` methods are thin wrappers that
re-canonicalize its output.
"""
function svdguess_compress(x::CanonicalIMPS, D::Int)
    max_bonddim(x) ≤ D && return copy(x)
    return CanonicalIMPS(svdguess_compress(x.AL, D))
end

function svdguess_compress(x::CanonicalIMPO, D::Int)
    return svdguess_compress(CanonicalIMPS(asmps_view(collect(x.AC))), D)
end

"Low-level tensor-string entry: streaming SVD truncation of a bare `AL`/`AR`
string to bond cap `D` (wrap bond Schmidt-truncated at site 1). The carry
(the previous site's truncated left basis) is absorbed into the next site
tensor on the fly — the carry-ignoring identity closure would break bond
consistency between the SVD outputs."
function svdguess_compress(ALs::PeriodicVector{<:Array{T,3}}, D::Int) where {T}
    site = (ℓ, carry) -> begin
        carry === nothing && return ALs[ℓ]
        A = ALs[ℓ]
        # B[a, s, f] = Σ_bb A[a, s, bb]·carry[bb, f]（carry 的行 = 上一站 SVD 截断
        # 出的右键基，与本站右键收缩；reshape 矩阵乘，@tensor 不接受开索引双侧出现）
        return reshape(reshape(A, :, size(A, 3)) * carry,
                       size(A, 1), size(A, 2), size(carry, 2))
    end
    return _lazy_svd_guess(site, length(ALs), D)
end

"""
    compress(x::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) -> (CanonicalIMPS, overlap)
    compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) -> (CanonicalIMPO, overlap)
    compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG}) -> (CanonicalIMPO, overlap)

Bond-dimension-`alg.D` variational approximation of a single chain (mirroring
MPSKit's `approximate`: maximize the overlap between the compressed chain and
the input chain, fixed point = the best rank-`D` approximation in the
ring-trace fidelity sense). `overlap` is the ring-trace fidelity in [0, N]
(= N means the input is exactly reproduced). `alg.D ≥ max_bonddim` of the
input short-circuits to the exact input. MPO results are always returned in
mixed-canonical storage (`CanonicalIMPO`, satisfying `ismixedcanonical`) — a
`DenseIMPO` input is canonicalized exactly (the operator value is preserved in
the periodic-trace representation) instead of being passed through. The
default initial guess is [`svdguess_compress`](@ref) (the input's own SVD
truncation). The positional `alg` dispatches [`VOMPS`](@ref) (ALS sweeps) or
[`IDMRG`](@ref) (eigen-solver sweeps), which share the same fixed point.
"""
compress(ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    _compress(ψ, alg, nothing; D = alg.D)

compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) =
    _compress(W, alg, nothing; D = alg.D)

compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG}) =
    _compress(W, alg, nothing; D = alg.D)

function _compress(ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    D >= max_bonddim(ψ) && return copy(ψ), real(length(ψ))
    K = collect(ψ.AC)
    x0 = x0 === nothing ? svdguess_compress(ψ, D) : x0
    x, overlap = if alg isa VOMPS
        _overlap_sweeps(nothing, ψ, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                        verbosity = alg.verbosity)
    else
        _idmrg_sweeps(ψ, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    end
    return _global_normalize!(x), overlap
end

function _compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    D >= max_bonddim(W) && return copy(W), real(length(W))
    N = length(W)
    dus = [size(W.AL[ℓ], 2) for ℓ in 1:N]
    dds = [size(W.AL[ℓ], 4) for ℓ in 1:N]
    ket = CanonicalIMPS(asmps_view(collect(W.AC)))
    K = collect(ket.AC)   # K 与 ket 同一规范表示（overlap 语义一致）
    x0 = x0 === nothing ? svdguess_compress(W, D) : x0
    x, overlap = if alg isa VOMPS
        _overlap_sweeps(nothing, ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                        verbosity = alg.verbosity)
    else
        _idmrg_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    end
    x = _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds), overlap
end

function _compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    D >= max_bonddim(W) && return CanonicalIMPO(collect(W.Ws)), real(length(W))
    N = length(W)
    dus = [size(W[ℓ], 2) for ℓ in 1:N]
    dds = [size(W[ℓ], 4) for ℓ in 1:N]
    ket = CanonicalIMPS(asmps_view(collect(W.Ws)))
    K = collect(ket.AC)
    x0 = x0 === nothing ? svdguess_compress(ket, D) : x0
    x, overlap = if alg isa VOMPS
        _overlap_sweeps(nothing, ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                        verbosity = alg.verbosity)
    else
        _idmrg_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    end
    x = _global_normalize!(x)
    return _mpo_from_mps(x, dus, dds), overlap
end

# ---------------- compress! (in-place) ----------------

"""
    compress!(out, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) -> out
    compress!(out, W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) -> out

In-place [`compress`](@ref): `out` is the user-provided chain to be optimized
as the initial guess. The target bond dimension is taken from the bond profile
of `out` (its bond profile is first brought to uniform `D = max_bonddim(out)`
with [`changebond!`](@ref)); `alg.D` is ignored. The optimized result is
written back into `out`.
"""
function compress!(out::CanonicalIMPS, ψ::CanonicalIMPS,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, _ = _compress(ψ, alg, out; D = D)
    return _copyinto!(out, y)
end

function compress!(out::CanonicalIMPO, W::CanonicalIMPO,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, _ = _compress(W, alg, CanonicalIMPS(asmps_view(collect(out.AC))); D = D)
    return _copyinto!(out, y)
end

# ---------------- 共享的代数压缩装配（原 add.jl；add 已删除） ----------------

"""
    _compress_ket(K, physdims, D, alg; x0 = nothing) -> (CanonicalIMPS, overlap)

Variationally compress the naively constructed target tensor string `K` (MPS
view, rank-3) to bond dimension `D`: `alg::VOMPS` runs the ALS sweeps
(`_overlap_sweeps`), `alg::IDMRG` the eigen-solver template (`_idmrg_sweeps`);
finally the result is globally normalized (norm convention `‖AC[1]‖ = 1`).
`x0` optionally provides the initial state (defaults to `svdguess`: the
target's own bond-wise SVD truncation, deterministic and inside the correct
basin).
"""
function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VOMPS; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? _truncate_bonddim(copy(ket), D) : x0
    x, overlap = _overlap_sweeps(nothing, ket, x0, K;
                                 tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return _global_normalize!(x), overlap
end

function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::IDMRG; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? _truncate_bonddim(copy(ket), D) : x0
    x, overlap = _idmrg_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                               verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    return _global_normalize!(x), overlap
end

_compress_ket(::Vector{<:Array{T,3}}, ::AbstractVector{Int}, ::Int,
              alg::Algorithm) where {T} =
    throw(ArgumentError("algebra compression only supports VOMPS() (DMRG-type) or IDMRG() algorithms; got $(typeof(alg))"))

"Result assembly of hadamard (MPS): `D ≥ max_bonddim(naive)` short-circuits to
the exact result; otherwise `(state truncated to bond dimension D, overlap)`.
The truncation is the
deterministic per-bond SVD truncation of the naive target itself
([`_truncate_bonddim`](@ref)): the identity-channel ALS is degenerate for
block-diagonal direct-sum targets, so the iterative engine is not used here."
function _algebra_result(K::Vector{<:Array{T,3}}, D::Int,
                         alg::Algorithm) where {T}
    naive = CanonicalIMPS(K)
    max_bonddim(naive) ≤ D && return naive
    x = _truncate_bonddim(copy(naive), D)
    _global_normalize!(x)
    N = length(K)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    overlap = N * abs(_ring_overlap(xALs, K)) /
              sqrt(real(_ring_overlap(xALs, xALs)) * real(_ring_overlap(K, K)))
    return x, overlap
end

"""
    _truncate_bonddim(ψ, D) -> CanonicalIMPS

Deterministic initial state for the compression of a block-degenerate target
(such as the direct sums of `add`): the SVD truncation of the naive target
itself to bond dimension `D` — always inside the correct ALS basin, unlike a
random initial state (the orthogonalities of `AL`/`AR` are preserved: the
truncation factors `U`/`V` act between orthogonality-protected bonds).
"""
function _truncate_bonddim(ψ::CanonicalIMPS{T}, D::Int) where {T}
    N = length(ψ)
    # 逐 bond 在原态上独立 SVD（不同 bond 的正交投影互易，可统一应用；
    # 串行"截一个再截下一个"会在已投影的态上用旧基因子，破坏混合规范）
    svds = Dict{Int,Any}()
    for ℓ in 1:N
        size(ψ.C[ℓ], 1) > D || continue
        svds[ℓ] = tsvd(ψ.C[ℓ]; trunc = truncdim(D))
    end
    isempty(svds) && return ψ
    # 应用：AL/AR[ℓ] 右键乘 U_ℓ、左键乘 V_{ℓ-1}；C[ℓ] = diag(s_ℓ)。
    # U/V 正交 ⇒ AR 串仍是右规范串，从它重建混合规范。
    for ℓ in 1:N
        ℓm = _mod1(ℓ - 1, N)
        if haskey(svds, ℓ)
            U, s, V, _ = svds[ℓ]
            ψ.AL[ℓ] = @tensor A[a, s2, c] := ψ.AL[ℓ][a, s2, bb] * U[bb, c]
            ψ.AR[ℓ] = @tensor A[a, s2, c] := ψ.AR[ℓ][a, s2, bb] * U[bb, c]
            ψ.C[ℓ] = Matrix{T}(Diagonal(s))
        end
        if haskey(svds, ℓm)
            _, _, Vm, _ = svds[ℓm]
            ψ.AL[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AL[ℓ][bb, s2, b]
            ψ.AR[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AR[ℓ][bb, s2, b]
        end
    end
    y = CanonicalIMPS(collect(ψ.AR))
    copy!(ψ.AL, y.AL)
    copy!(ψ.AR, y.AR)
    copy!(ψ.C, y.C)
    copy!(ψ.AC, y.AC)
    return ψ
end

# ---- lazy (on-the-fly) naive-SVD initial guess (shared by mult / hadamard) ----
#
# 参考 FiniteMPSAlgorithms 的 `_naive_svd_guess`，但收缩路径更省：naive 乘积的
# site tensor 由 `site(i, carry)` **在构造时就把 carry 吸收进收缩**（优先与 carry
# 收缩），而不是先 materialize 完整的 naive 大张量再乘 carry —— 后者会多出一个
# 键维 = 输入键维乘积 的大中间张量。自右向左流式做带截断的右正交化（键 ≤ `D`）：
# 右因子作为输出 site tensor，`U·diag(s)` 作为 carry 传给左侧下一站。整条 naive
# 乘积族从不落地，峰值中间内存只有「首个（无 carry 的）大张量 + 键 ≤ D 的输出」。
# 收尾在 site 1 上对 wrap 键（bond N）做 Schmidt 截断：扫掠是一条线扫，环形闭合
# 的键在扫掠中截不到；但此时 site 2:L 右规范 ⇒ site 1 张量的 (1,)|(2,3) SVD 恰是
# 该键的谱（截断精确最优），`uᵀ` 吸收回 site L（右规范保持：(uᵀA)(uᵀA)† = uᵀu）。
# 因此**输出的每个键都 ≤ D**，调用方无需再做收尾截断。

"""
    _lazy_svd_guess(site, L, D) -> Vector{Array{T,3}}

`site(i, carry)`（`i = 1:L`，`carry::Union{Nothing,AbstractMatrix}`）现算第 `i`
站张量并在构造中吸收 carry（见上方注释）；自右向左流式 SVD 截断到键 ≤ `D`，
收尾做 wrap 键截断。返回右规范串（site 2:L）＋携带余量的 site 1，
**所有键 ≤ `D`**。
"""
function _lazy_svd_guess(site::F, L::Int, D::Int) where {F}
    B = site(L, nothing)
    out = Vector{typeof(B)}(undef, L)
    carry = nothing
    for i in L:-1:2
        u, s, v, _ = tsvd(B, (1,), (2, 3); trunc = truncdim(D))
        out[i] = v
        carry = u * Diagonal(s)
        B = site(i - 1, carry)      # 为下一轮准备；i = 2 时即 site(1, carry)
    end
    # B = site(1, carry)：wrap 键（bond N）的 Schmidt 截断（bond N = site 1 的左键
    # = site L 的右键）
    u, s, v, _ = tsvd(B, (1,), (2, 3); trunc = truncdim(D))
    v3 = Diagonal(s) * reshape(v, length(s), :)             # v 是秩-3 (r, s, D)
    out[1] = reshape(v3, length(s), size(B, 2), size(B, 3))
    if L >= 2
        NL = out[L]
        out[L] = @tensor A[a, s2, b] := NL[a, s2, bb] * u[bb, b]    # uᵀ 吸收回 site L 右腿
    else
        # L = 1：没有独立的 wrap 键，对第二个键再做一次 SVD（初猜用途的近似）
        r1, s1, _ = size(out[1])
        u2, svals, _, _ = tsvd(out[1], (1, 2), (3,); trunc = truncdim(D))
        u2m = @tensor uu[p, q, f2] := u2[p, q, k] * Diagonal(svals)[k, f2]
        out[1] = reshape(u2m, r1, s1, length(svals))
    end
    return out
end

# ---- convergence fallback guard of the lazy (compute-on-the-fly) engines ----

"""
    _lazy_or_fallback(lazy_fn, fallback_fn, N) -> (result, overlap)

Shared guard of the lazy compute-on-the-fly paths: run the lazy engine first;
if the final ring fidelity clearly falls below `N` (the ALS may get stuck in a
wrong basin for block-degenerate targets), fall back to the naive construction
+ compression (correct, at the cost of materializing the naive family).
"""
function _lazy_or_fallback(lazy_fn::F1, fallback_fn::F2, N::Int) where {F1,F2}
    result, overlap = lazy_fn()
    real(overlap) < 0.9 * N && return fallback_fn()
    return result, overlap
end

"Copy the tensor families of `y` into `out` (both mixed-canonical)."
function _copyinto!(out::CanonicalIMPS, y::CanonicalIMPS)
    copy!(out.AL, y.AL)
    copy!(out.AR, y.AR)
    copy!(out.C, y.C)
    copy!(out.AC, y.AC)
    return out
end

function _copyinto!(out::CanonicalIMPO, y::CanonicalIMPO)
    copy!(out.AL, y.AL)
    copy!(out.AR, y.AR)
    copy!(out.C, y.C)
    copy!(out.AC, y.AC)
    return out
end

"DenseIMPO 结果写回 `CanonicalIMPO` 缓存：先转规范形式再逐家族复制。"
_copyinto!(out::CanonicalIMPO, y::DenseIMPO) = _copyinto!(out, CanonicalIMPO(collect(y.Ws)))
