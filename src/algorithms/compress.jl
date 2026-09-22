# ---------------- compress (bond-dimension reduction of a single chain, mirroring MPSKit's approximate) ----------------
#
# Find `out ≈ x` with `max_bonddim(out) = D`: the overlap-maximizing
# variational approximation of a *given* chain (the target is the chain itself,
# unlike add/mult/hadamard whose targets are naive algebraic constructions).
# The engines are shared with the algebra compressions (`_overlap_sweeps` /
# `_idmrg_sweeps` on the identity channel).

"""
    svdguess_compress(x, D) -> CanonicalIMPS

Deterministic initial guess of the iterative [`compress`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_compress`): the bond-wise SVD truncation of `x`
itself to `D` (accurate but costly compared to [`changebond!`](@ref) zero
padding; the mixed-canonical form is preserved — the truncation factors act on
orthogonality-protected bonds). `CanonicalIMPO` input returns the MPS-view
guess (`CanonicalIMPS` on the doubled space).
"""
function svdguess_compress(x::CanonicalIMPS, D::Int)
    return max_bonddim(x) ≤ D ? copy(x) : _truncate_bonddim(copy(x), D)
end

function svdguess_compress(x::CanonicalIMPO, D::Int)
    return max_bonddim(x) ≤ D ? CanonicalIMPS(asmps_view(collect(x.AC))) :
           _truncate_bonddim(CanonicalIMPS(asmps_view(collect(x.AC))), D)
end

"""
    compress(ψ::CanonicalIMPS; D, alg = VOMPS(), x0 = nothing) -> (CanonicalIMPS, overlap)
    compress(W::CanonicalIMPO; D, alg = VOMPS(), x0 = nothing) -> (CanonicalIMPO, overlap)
    compress(W::DenseIMPO; D, alg = VOMPS(), x0 = nothing) -> (DenseIMPO, overlap)

Bond-dimension-`D` variational approximation of a single chain (mirroring
MPSKit's `approximate`: maximize the overlap between the compressed chain and
the input chain, fixed point = the best rank-`D` approximation in the
ring-trace fidelity sense). `overlap` is the ring-trace fidelity in [0, N]
(= N means the input is exactly reproduced). The default initial guess is
[`svdguess_compress`](@ref) (the input's own SVD truncation); an explicit `x0`
overrides it. `alg` dispatches [`VOMPS`](@ref) (ALS sweeps) or [`IDMRG`](@ref)
(eigen-solver sweeps), which share the same fixed point.
"""
function compress(ψ::CanonicalIMPS; D::Int, alg::Union{VOMPS,IDMRG} = VOMPS(),
                  x0::Union{Nothing,CanonicalIMPS} = nothing)
    (D < max_bonddim(ψ)) ||
        return copy(ψ), real(length(ψ))
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

function compress(W::CanonicalIMPO; D::Int, alg::Union{VOMPS,IDMRG} = VOMPS(),
                  x0::Union{Nothing,CanonicalIMPS} = nothing)
    (D < max_bonddim(W)) ||
        return copy(W), real(length(W))
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

function compress(W::DenseIMPO; D::Int, alg::Union{VOMPS,IDMRG} = VOMPS(),
                  x0::Union{Nothing,CanonicalIMPS} = nothing)
    (D < max_bonddim(W)) ||
        return copy(W), real(length(W))
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
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return DenseIMPO(ALs4), overlap
end

# ---------------- compress! (in-place) ----------------

"""
    compress!(out, ψ::CanonicalIMPS; D, alg = VOMPS()) -> out
    compress!(out, W::CanonicalIMPO; D, alg = VOMPS()) -> out

In-place [`compress`](@ref): `out` is the user-provided chain to be optimized
as the initial guess (its bond profile is first brought to `D` with
[`changebond!`](@ref)); the optimized result is written back into `out`.
"""
function compress!(out::CanonicalIMPS, ψ::CanonicalIMPS; D::Int,
                   alg::Union{VOMPS,IDMRG} = VOMPS())
    changebond!(out; D = D)
    y, _ = compress(ψ; D = D, alg = alg, x0 = out)
    return _copyinto!(out, y)
end

function compress!(out::CanonicalIMPO, W::CanonicalIMPO; D::Int,
                   alg::Union{VOMPS,IDMRG} = VOMPS())
    changebond!(out; D = D)
    y, _ = compress(W; D = D, alg = alg, x0 = CanonicalIMPS(asmps_view(collect(out.AC))))
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

"Result assembly of add/hadamard (MPS): `D = nothing` → the naive exact state;
`D::Int` → `(state truncated to bond dimension D, overlap)` (short-circuits to
the exact result when `D ≥ max_bonddim(naive)`). The truncation is the
deterministic per-bond SVD truncation of the naive target itself
([`_truncate_bonddim`](@ref)): the identity-channel ALS is degenerate for
block-diagonal direct-sum targets, so the iterative engine is not used here."
function _algebra_result(K::Vector{<:Array{T,3}}, D::Union{Nothing,Int},
                         alg::Algorithm) where {T}
    naive = CanonicalIMPS(K)
    (D === nothing || max_bonddim(naive) ≤ D) && return naive
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

"MPO 版代数结果装配：naive rank-4 张量 → `D = nothing` 精确 / `D::Int` 压缩。
输出为基础归一化（`_global_normalize!`）后的左正则张量 `AL` 对应的
`DenseIMPO`（幅值语义已按 MPSKit 约定放弃，精度用 fidelity 检验）。"
function _mpo_algebra_result(K4::Vector{<:Array{T,4}}, D::Union{Nothing,Int},
                             alg::Algorithm) where {T}
    naive = DenseIMPO(K4)
    if D === nothing || max_bonddim(naive) ≤ D
        return naive
    end
    N = length(K4)
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x = _truncate_bonddim(CanonicalIMPS(K3), D)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    overlap = N * abs(_ring_overlap(xALs, K3)) /
              sqrt(real(_ring_overlap(xALs, xALs)) * real(_ring_overlap(K3, K3)))
    _global_normalize!(x)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return DenseIMPO(ALs4), overlap
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
