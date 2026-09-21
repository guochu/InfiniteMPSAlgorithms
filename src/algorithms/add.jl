# ---------------- iterative addition add (variational compression of mps+mps / mpo+mpo) ----------------
#
# The naive exact construction kernels live in arithmetics.jl
# (exact_add / _naive_sum_tensor); this file provides the iterative (variational
# compression) versions on top, sharing the compression engine of mult.jl's
# mult (_overlap_sweeps / _idmrg_sweeps):
# - `D = nothing`: naive exact construction (consistent with MPSKit, no compression);
# - `D::Int`: naive construction followed by compression to bond dimension `D`
#   (the direct-sum bond dimension grows only linearly as D₁+D₂+⋯, so the naive
#   family never blows up — unlike the fuse/zip constructions of mult/hadamard;
#   the identity-channel ALS is moreover degenerate for block-diagonal targets,
#   so the naive construction is the reliable route here).
#
# This file also hosts the variational-compression assembly shared by
# add / hadamard (_compress_ket / _algebra_result / _mpo_algebra_result) and the
# convergence fallback guard of the lazy engines (_lazy_or_fallback).

# ---- variational compression assembly shared by add / hadamard ----

"""
    _compress_ket(K, physdims, D, alg; x0 = nothing) -> (CanonicalIMPS, overlap)

Variationally compress the naively constructed target tensor string `K` (MPS
view, rank-3) to bond dimension `D`: `alg::VOMPS` runs the ALS sweeps
(`_overlap_sweeps`), `alg::IDMRG` the eigen-solver template (`_idmrg_sweeps`);
finally `_align_scale!` aligns scale/phase. `x0` optionally provides the
initial state (defaults to a random one).
"""
function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VOMPS; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? randomimps(T, physdims, D) : x0
    x, overlap = _overlap_sweeps(nothing, ket, x0, K;
                                 tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return _align_scale!(x, K), overlap
end

function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::IDMRG; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? randomimps(T, physdims, D) : x0
    x, overlap = _idmrg_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                               verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    return _align_scale!(x, K), overlap
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
    for ℓ in 1:N
        size(ψ.C[ℓ], 1) > D || continue
        U, s, V, _ = tsvd(ψ.C[ℓ]; trunc = truncdim(D))
        # bond ℓ truncation: the tensors to the left get U on their right bond,
        # the tensors to the right get V on their left bond (AL[ℓ+1]/AR[ℓ+1]
        # share the left bond; the orthogonalities are preserved because U/V
        # have orthonormal columns/rows)
        ψ.AL[ℓ] = @tensor A[a, s, c] := ψ.AL[ℓ][a, s, b] * U[b, c]
        ψ.AR[ℓ] = @tensor A[a, s, c] := ψ.AR[ℓ][a, s, b] * U[b, c]
        ψ.C[ℓ] = Matrix{T}(Diagonal(s))
        ℓ1 = _mod1(ℓ + 1, N)
        ψ.AL[ℓ1] = @tensor A[a, s, b] := V[a, c] * ψ.AL[ℓ1][c, s, b]
        ψ.AR[ℓ1] = @tensor A[a, s, b] := V[a, c] * ψ.AR[ℓ1][c, s, b]
    end
    for ℓ in 1:N
        ψ.AC[ℓ] = _mulAL(ψ.AL[ℓ], ψ.C[ℓ])
    end
    return ψ
end

"Result assembly of add/hadamard (MPO): naive rank-4 tensors → `D = nothing`
exact / `D::Int` compressed. The output takes the ring-trace alignment scalar
`c` times the left-canonical tensors `AL`: the output amplitude `c^N·tr(∏AL)`
exactly restores the target amplitude (including phase, for any `D`). The
per-site phase of the converged state is a gauge freedom of the eigen solution
(MPSKit likewise does not pin down the eigen-solution phase); the amplitude is
aligned in the ring-trace sense. Collecting `x.AC` is not possible: `AC = AL·C`
inserts a C weighting between bonds, and its periodic trace is a
gauge-dependent C-weighted quantity rather than the operator amplitude."
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
    c = _ring_scale(x, K3)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return DenseIMPO([c * A for A in ALs4]), overlap
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

# ---- variadic direct-sum helpers ----

"Common validation of the variadic add: at least two inputs, equal unit-cell
lengths."
function _add_check(ψs)
    (length(ψs) ≥ 2) || throw(ArgumentError("add requires at least 2 inputs (a single input is returned as is)"))
    N = length(ψs[1])
    all(length(ψ) == N for ψ in ψs) ||
        throw(DimensionMismatch("add requires equal unit-cell lengths (mirrors MPSKit check_length)"))
    return N
end

"Chain the per-site direct sums of N tensors (associativity of direct sums
keeps the bond dimension growing linearly: D₁+D₂+⋯)."
_sum_tensors(ts) = reduce(_naive_sum_tensor, ts)

# ---------------- public interface add (variadic) ----------------

"""
    add(ψ₁, ψ₂, ...; D = nothing, alg = VOMPS()) -> CanonicalIMPS
    add(W₁, W₂, ...; D = nothing, alg = VOMPS()) -> DenseIMPO

Iterative addition of any number ≥ 1 of MPSs (→ `CanonicalIMPS`) or MPOs
(→ `DenseIMPO`; inputs may be `CanonicalIMPO`). Equal unit-cell lengths and
equal per-site physical dimensions are required (mirroring MPSKit's
`check_length`). The direct-sum construction uses the left-canonical tensors
`AL` — amplitudes are exactly additive in the periodic trace representation
(`AC` would insert a C weighting and cannot be used):

- `D = nothing`: exact construction with the direct-sum bond dimension
  (consistent with MPSKit, no compression; the bond dimension grows linearly as
  D₁+D₂+⋯, with no intermediate blowup);
- `D::Int`: naive construction followed by variational compression to bond
  dimension `D` with VOMPS/IDMRG; `overlap` is the ring-trace fidelity in
  [0, N] (= N means same direction; returned as `(result, overlap)`).

A single input is returned unchanged. For the exact tensor strings see
[`exact_add`](@ref).
"""
function add(ψ1::CanonicalIMPS, ψs::CanonicalIMPS...;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (isempty(ψs)) && return ψ1
    allψ = (ψ1, ψs...)
    N = _add_check(allψ)
    all(size(allψ[j].AL[ℓ], 2) == size(allψ[1].AL[ℓ], 2)
        for j in 2:length(allψ), ℓ in 1:N) ||
        throw(DimensionMismatch("add requires equal per-site physical dimensions"))
    K = [_sum_tensors([ψ.AL[ℓ] for ψ in allψ]) for ℓ in 1:N]
    naive = CanonicalIMPS(K)
    D === nothing && return naive
    if max_bonddim(naive) ≤ D
        x = _global_normalize!(_truncate_bonddim(copy(naive), D))
        xALs = [x.AL[ℓ] for ℓ in 1:N]
        overlap = N * abs(_ring_overlap(xALs, K)) /
                  sqrt(real(_ring_overlap(xALs, xALs)) * real(_ring_overlap(K, K)))
        return x, overlap
    end
    # D::Int: multi-channel lazy sum (compute-on-the-fly) — each input forms its
    # own identity-channel ket; the local map `k = Σ_j GL_j·ψ_j·GR_j` implements
    # the wavefunction sum directly (each channel's environments are
    # non-degenerate, unlike a single block-diagonal target).
    kets = [LazyKet(ℓ -> ψ.AL[ℓ], ℓ -> ψ.AR[ℓ], ℓ -> ψ.AC[ℓ], ℓ -> ψ.C[ℓ])
            for ψ in allψ]
    T = promote_type(scalartype.(allψ)...)
    x0 = randomimps(T, phydims(ψ1), D)
    return _lazy_or_fallback(
        () -> _lazy_sweeps(kets, x0, N; alg = alg, tol = alg.tol,
                           maxiter = alg.maxiter, verbosity = alg.verbosity),
        () -> begin
            x = _global_normalize!(_truncate_bonddim(copy(naive), D))
            xALs = [x.AL[ℓ] for ℓ in 1:N]
            overlap = N * abs(_ring_overlap(xALs, K)) /
                      sqrt(real(_ring_overlap(xALs, xALs)) * real(_ring_overlap(K, K)))
            return x, overlap
        end,
        N)
end

"Common body of the MPO add (the inputs are kept RAW throughout: the direct sum
uses `W[ℓ]` with full operator amplitudes, so exact additivity holds)."
function _add_mpo(allW::Tuple, D::Union{Nothing,Int}, alg::Union{VOMPS,IDMRG})
    N = _add_check(allW)
    all(size(allW[j][ℓ], 2) == size(allW[1][ℓ], 2) &&
        size(allW[j][ℓ], 4) == size(allW[1][ℓ], 4)
        for j in 2:length(allW), ℓ in 1:N) ||
        throw(DimensionMismatch("add requires equal per-site physical (u,d) dimensions"))
    T = promote_type(scalartype.(allW)...)
    K4 = [_sum_tensors([W[ℓ] for W in allW]) for ℓ in 1:N]
    D === nothing && return DenseIMPO(K4)
    naive = DenseIMPO(K4)
    if max_bonddim(naive) ≤ D
        return naive, real(T)(N)
    end
    # D::Int: multi-channel lazy sum over the raw MPO tensors (isomorphic to the
    # MPS add; the raw tensors carry the full operator amplitudes so the sum is
    # exactly additive, the gram twist absorbs their non-canonical gauge, and
    # the C closures are unit calibrations), with the naive SVD truncation as
    # fallback.
    fams = [(ℓ -> W[ℓ], ℓ -> W[ℓ], ℓ -> W[ℓ],
             ℓ -> Matrix{T}(I, size(W[ℓ], 1), size(W[ℓ], 1))) for W in allW]
    dus = [size(allW[1][ℓ], 2) for ℓ in 1:N]
    dds = [size(allW[1][ℓ], 4) for ℓ in 1:N]
    physdims = dus .* dds
    x0 = randomimps(T, physdims, D)
    return _lazy_or_fallback(
        () -> _lazy_mpo_result(fams, x0, dus, dds, N; alg = alg),
        () -> _mpo_algebra_result(K4, D, alg),
        N)
end

function add(W1::CanonicalIMPO, Ws::CanonicalIMPO...;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (isempty(Ws)) && return W1
    return _add_mpo((W1, Ws...), D, alg)
end

function add(W1::DenseIMPO, Ws::DenseIMPO...;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (isempty(Ws)) && return W1
    return _add_mpo((W1, Ws...), D, alg)
end

"Direct sum of two rank-2 tensors (blockdiag); the exact bond matrix of the
direct-sum target."
function _blockdiag_matrix(C1::AbstractMatrix{T}, C2::AbstractMatrix{T}) where {T}
    d1, d2 = size(C1, 1), size(C2, 1)
    C = zeros(T, d1 + d2, d1 + d2)
    C[1:d1, 1:d1] .= C1
    C[d1+1:d1+d2, d1+1:d1+d2] .= C2
    return C
end

# ---------------- naive_add (debug: naive family construction + optional compression) ----------------

"""
    naive_add(ψ₁, ψ₂, ...; D = nothing, alg = VOMPS()) -> CanonicalIMPS / (CanonicalIMPS, overlap)
    naive_add(W₁, W₂, ...; D = nothing, alg = VOMPS()) -> DenseIMPO / (DenseIMPO, overlap)

Naive reference implementation of [`add`](@ref) (debug only): identical to
[`add`](@ref) (the direct sum is already naive; kept for interface symmetry
with [`naive_mult`](@ref) / [`naive_hadamard`](@ref)).
"""
naive_add(ψ1::CanonicalIMPS, ψs::CanonicalIMPS...; kwargs...) =
    add(ψ1, ψs...; kwargs...)
naive_add(W1::CanonicalIMPO, Ws::CanonicalIMPO...; kwargs...) =
    add(W1, Ws...; kwargs...)
naive_add(W1::DenseIMPO, Ws::DenseIMPO...; kwargs...) =
    add(W1, Ws...; kwargs...)
