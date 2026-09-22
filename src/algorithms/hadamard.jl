# ---------------- iterative hadamard (infinite-MPS generalization of the Hadamard/Schur product) ----------------
#
# Elementwise waveform product `c₁₂ = c₁ .* c2`: virtual legs zipped per site,
# physical leg shared (kernel `_naive_hadamard_tensor` in arithmetics.jl);
# the physical dimension is unchanged and the bond dimension becomes the product
# of the two. The compression assembly is shared with the algebra operations
# (_compress_ket / _algebra_result in compress.jl).

"""
    hadamard(ψ₁, ψ₂; D = nothing, alg = VOMPS()) -> CanonicalIMPS

Infinite-MPS generalization of the Hadamard/Schur product (elementwise
product): the waveform is multiplied pointwise,
`c₁₂[(s…)] = c₁[(s…)] · c₂[(s…)]`, with the **physical dimension unchanged**
and the bond dimension becoming the product of the two. Naive construction:
virtual legs zipped per site with a shared physical leg; the two virtual chains
are independent so the periodic trace factorizes,
`tr(∏A12) = tr(∏AL₁)·tr(∏AL₂)` exactly.

- `D = nothing`: naive exact construction (no compression; the output bond
  dimension = D₁·D₂, which is inherently large);
- `D::Int`: compute-on-the-fly compression — the zip target is generated site
  by site on demand (lazy zip; the zip family is never materialized) and
  variationally compressed to bond dimension `D` with VOMPS/IDMRG; `overlap`
  is the ring-trace fidelity in [0, N] (= N means same direction; returned as
  `(result, overlap)`).

Note: the naive zip is generally **not** in canonical form (sharing the
physical leg couples the orthogonality sums across the two virtual chains); the
`D = nothing` constructor's normalized output differs from `c₁.*c₂` by a
constructor-determined positive real scalar (a ray representative). Use
[`exact_hadamard`](@ref) for the raw tensor string with strict pointwise
amplitudes. For a naive reference implementation (construct the whole family,
then compress) see [`naive_hadamard`](@ref).
"""
function hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS;
                  D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS(),
                  x0::Union{Nothing,CanonicalIMPS} = nothing)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    N = length(ψ1)
    T = promote_type(scalartype(ψ1), scalartype(ψ2))
    if D === nothing
        K = [_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:N]
        return _algebra_result(K, nothing, alg)
    end
    # D::Int: lazy zip target (compute-on-the-fly). All four family closures
    # preserve the factor canonical forms (a zip of isometries is an isometry);
    # the C closure `C_zip = kron(C₂, C₁)` matches the bond order of the zip
    # kernel (`kron(A₂, A₁)`, A₂ major); left environments use AL, right
    # environments AR (matching the `_ternary_fixedpoints` gauge convention).
    ket = LazyKet(
        ℓ -> _naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]),
        ℓ -> _naive_hadamard_tensor(ψ1.AR[ℓ], ψ2.AR[ℓ]),
        ℓ -> _naive_hadamard_tensor(ψ1.AC[ℓ], ψ2.AC[ℓ]),
        ℓ -> kron(ψ2.C[ℓ], ψ1.C[ℓ]),
    )
    x0 = x0 === nothing ? svdguess_hadamard(ψ1, ψ2, D) : x0
    # lazy zip engine + naive fallback (see _lazy_or_fallback; the naive family
    # is only materialized when the fallback fires)
    return _lazy_or_fallback(
        () -> _lazy_sweeps(ket, x0, N; alg = alg, tol = alg.tol,
                           maxiter = alg.maxiter, verbosity = alg.verbosity),
        () -> _compress_ket([_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:N],
                            phydims(ψ1), D, alg),
        N)
end

# ---------------- naive_hadamard (debug: naive family construction + optional compression) ----------------

"""
    naive_hadamard(ψ₁, ψ₂; D = nothing, alg = VOMPS()) -> CanonicalIMPS

Naive reference implementation of [`hadamard`](@ref) (debug only): first
construct the complete zip family (memory O(N·D₁D₂)), then (optionally)
compress to `D`. `overlap` is the ring-trace fidelity in [0, N]. Large input
bond dimensions produce huge intermediate families — use [`hadamard`](@ref)
for production use.
"""
function naive_hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS;
                        D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
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

Deterministic initial guess of the iterative [`hadamard`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_hadamard`): the naive per-site zip (the same
tensor string as [`exact_hadamard`](@ref), with the exact pointwise product
amplitudes) followed by the bond-wise SVD truncation to `D`.
"""
function svdguess_hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, D::Int)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    K = [_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:length(ψ1)]
    x = CanonicalIMPS(K)
    return max_bonddim(x) ≤ D ? x : _truncate_bonddim(x, D)
end

"""
    hadamard!(out, ψ₁, ψ₂; D, alg = VOMPS()) -> out

In-place [`hadamard`](@ref): `out` is the user-provided state to be optimized
as the initial guess (its bond profile is first brought to `D` with
[`changebond!`](@ref)); the optimized result is written back into `out`.
"""
function hadamard!(out::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS;
                   D::Int, alg::Union{VOMPS,IDMRG} = VOMPS())
    changebond!(out; D = D)
    y, _ = hadamard(ψ1, ψ2; D = D, alg = alg, x0 = out)
    return _copyinto!(out, y)
end
