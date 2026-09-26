"""
    CanonicalIMPS{T}

Mixed-canonical storage of an infinite (periodically tiled) MPS; the data
layout mirrors MPSKit's `InfiniteMPS`:

- `AL::PeriodicVector{Array{T,3}}`: left-canonical site tensors (`(Dl, s, Dr)`),
  `Σ AL†·AL = 1`;
- `AR::PeriodicVector{Array{T,3}}`: right-canonical site tensors, `Σ AR·AR† = 1`;
- `C::PeriodicVector{Array{T,2}}`: center matrices on bond ℓ (between sites
  ℓ and ℓ+1);
- `AC::PeriodicVector{Array{T,3}}`: center-canonical site tensors.

Convention (same as MPSKit): `AL[i] * C[i] = AC[i] = C[i-1] * AR[i]`.
All indices wrap around with period N.
"""
struct CanonicalIMPS{T}
    AL::PeriodicVector{Array{T,3}}
    AR::PeriodicVector{Array{T,3}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,3}}

    function CanonicalIMPS{T}(AL::PeriodicVector{Array{T,3}},
                                     AR::PeriodicVector{Array{T,3}},
                                     C::PeriodicVector{Array{T,2}},
                                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
        end
        new{T}(AL, AR, C, AC)
    end
end

_mul_ALC(AL::PeriodicVector{A}, C::PeriodicVector{B}) where {A<:Array{T,3},B<:Matrix{T}} where {T} =
    PeriodicVector(Array{T,3}[_mulAL(AL[ℓ], C[ℓ]) for ℓ in 1:length(AL)])
_mulAL(AL::AbstractArray{T,3}, C::AbstractMatrix{T}) where {T} =
    begin
        @tensor AC[a, s, c] := AL[a, s, b] * C[b, c]
    end

CanonicalIMPS(AL::PeriodicVector{Array{T,3}}, AR::PeriodicVector{Array{T,3}},
                     C::PeriodicVector{Array{T,2}},
                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T} =
    CanonicalIMPS{T}(AL, AR, C, AC)

"""
    _check_bond_consistency(As) -> nothing

逐站校验张量串的虚拟键一致性，**包含周期闭合那一条键**（`size(As[N],3) ==
size(As[1],1)`，对齐 MPSKit `InfiniteMPS(A)` 的 `circshift` 全查）。键维允许
逐站不同（非均匀键 profile）。
"""
function _check_bond_consistency(As::AbstractVector)
    N = length(As)
    for ℓ in 1:N
        size(As[ℓ], 3) == size(As[_mod1(ℓ + 1, N)], 1) ||
            throw(DimensionMismatch("bond mismatch at site $ℓ"))
    end
    return nothing
end

"""
    _bond_profile(As) -> Vector{Int}

`χ[ℓ] = size(As[ℓ], 3)`：第 ℓ 条键（site ℓ 与 ℓ+1 之间）的键维。非均匀键下各值
可不同，需要全局键维的地方一律改用这个 profile 而不是单个 `D`。
"""
_bond_profile(As::AbstractVector) = [size(As[ℓ], 3) for ℓ in 1:length(As)]

# ---- 满秩（可行键 profile）化：对齐 MPSKit 的 makefullrank! ----

"""
`size(A,3) ≤ size(A,1)·size(A,2)`：左→右映射单射，即 `Dr ≤ Dl·d`
（MPSKit `isfullrank(A; side = :right)`）。
"""
_isfullrank_right(A::AbstractArray{T,3}) where {T} = size(A, 3) <= size(A, 1) * size(A, 2)

"""
`size(A,1) ≤ size(A,3)·size(A,2)`：右→左映射单射，即 `Dl ≤ Dr·d`
（MPSKit `isfullrank(A; side = :left)`）。
"""
_isfullrank_left(A::AbstractArray{T,3}) where {T} = size(A, 1) <= size(A, 3) * size(A, 2)

"""
    _makefullrank!(As; alg = Defaults.alg_orth()) -> As

MPSKit `makefullrank!(A::PeriodicVector)` 的 plain-array 版：把**不可行**的键维
（`Dr > Dl·d` 或 `Dl > Dr·d`）用 QR/LQ 把冗余方向投影掉，因子吸收进邻居。

`A[i] = Q`、`A[i±1] = R·A[i±1]` 保持周期 trace（乘积）不变，因此**态严格不变**，
只是把用不到的键维删掉。反复执行到每个站点两个方向都可行。对齐 MPSKit
`InfiniteMPS(A)` 的行为（例如 [2,5,2] 在 d=2 下被降到 [2,4,2]）。

作用：保证后续 `gaugefix!` 扫掠里的 QR **永不截断**，从而 `C[ℓ]` 始终是方阵
（MPSKit 的 gauge 加速步 `gauge_eigsolve_step!` 正是依赖这一点而没有方阵守卫）。
"""
function _makefullrank!(As::AbstractVector{<:Array{T,3}};
                        alg = Defaults.alg_orth()) where {T}
    N = length(As)
    while true
        i = findfirst(ℓ -> !(_isfullrank_left(As[ℓ]) && _isfullrank_right(As[ℓ])), 1:N)
        i === nothing && return As
        A = As[i]
        Dl, d, Dr = size(A)
        if !_isfullrank_right(A)
            # Dr > Dl·d：丢掉右键的冗余方向，余因子 R 吸收进 site i+1 的左键
            Q, Rf = leftorth(reshape(A, Dl * d, Dr); alg = alg)
            As[i] = reshape(Q, Dl, d, size(Q, 2))
            j = _mod1(i + 1, N)
            Aj = As[j]
            As[j] = reshape(Rf * reshape(Aj, size(Aj, 1), size(Aj, 2) * size(Aj, 3)),
                            size(Rf, 1), size(Aj, 2), size(Aj, 3))
        else
            # Dl > Dr·d：丢掉左键的冗余方向，余因子 L 吸收进 site i-1 的右键
            Lf, Q = rightorth(reshape(A, Dl, d * Dr); alg = alg')
            As[i] = reshape(Q, size(Lf, 2), d, Dr)
            j = _mod1(i - 1, N)
            Aj = As[j]
            As[j] = reshape(reshape(Aj, size(Aj, 1) * size(Aj, 2), size(Aj, 3)) * Lf,
                            size(Aj, 1), size(Aj, 2), size(Lf, 2))
        end
    end
end

"""
    CanonicalIMPS(As::AbstractVector{<:Array{T,3}}; kwargs...)

Construct from plain site tensors (mirrors MPSKit's `InfiniteMPS(A)`):
`AR = A`, then `gaugefix!(; order = :LR)` starting from `C₀ = I`, and `AC = AL·C`.
键维允许逐站不同（非均匀键 profile）。
"""
function CanonicalIMPS(As::AbstractVector{<:Array{T,3}}; kwargs...) where {T}
    N = length(As)
    _check_bond_consistency(As)
    Araw = [copy(a) for a in As]
    # 对齐 MPSKit `InfiniteMPS(A)`：先把不可行的键维（Dr > Dl·d 或 Dl > Dr·d）删掉，
    # 态严格不变；这样 gaugefix! 扫掠里的 QR 永不截断、C[ℓ] 恒为方阵。
    _makefullrank!(Araw)
    AR = PeriodicVector(Araw)
    AL = PeriodicVector([similar(a) for a in AR])
    AC = PeriodicVector([similar(a) for a in AR])
    # C[ℓ] 作用在第 ℓ 条键上：(site ℓ 的右键维, site ℓ+1 的左键维)；两者相等时即
    # (χ_ℓ, χ_ℓ)。非均匀键下不能用单个全局 D。
    C = PeriodicVector([similar(AR[ℓ], size(AR[ℓ], 3), size(AR[_mod1(ℓ + 1, N)], 1))
                        for ℓ in 1:N])
    D = size(AR[1], 1)          # C₀ 作用在键 N 上，形状 (χ_N, χ_N)
    ψ = CanonicalIMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, parent(AR), Matrix{T}(I, D, D); kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

"""
    CanonicalIMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix; kwargs...)

Construct from left-canonical tensors plus an initial gauge matrix (mirrors
MPSKit's `InfiniteMPS(AL, C₀)`): `gaugefix!` to the right-canonical form
(`order = :R`), then `AC = AL·C`. `C₀` 作用在键 N 上，必须是 `(χ_N, χ_N)`。
"""
function CanonicalIMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix;
                              kwargs...) where {T}
    N = length(ALs)
    _check_bond_consistency(ALs)
    (size(C₀, 1) == size(C₀, 2) == size(ALs[N], 3)) ||
        throw(DimensionMismatch("C₀ must be $(size(ALs[N],3))×$(size(ALs[N],3))"))
    AL = PeriodicVector([copy(a) for a in ALs])
    AR = PeriodicVector([similar(a) for a in AL])
    AC = PeriodicVector([similar(a) for a in AL])
    C = PeriodicVector([similar(AL[ℓ], size(AL[ℓ], 3), size(AL[_mod1(ℓ + 1, N)], 1))
                        for ℓ in 1:N])
    ψ = CanonicalIMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, ALs, C₀; order = :R, kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

# ---------------- interface ----------------

Base.length(ψ::CanonicalIMPS) = length(ψ.AL)
Base.size(ψ::CanonicalIMPS, args...) = size(ψ.AL, args...)
Base.getindex(ψ::CanonicalIMPS, ℓ::Integer) = ψ.AC[ℓ]
Base.setindex!(ψ::CanonicalIMPS, v::Array, ℓ::Integer) = (ψ.AC[ℓ] = v; ψ)
Base.firstindex(ψ::CanonicalIMPS) = 1
Base.lastindex(ψ::CanonicalIMPS) = length(ψ)
Base.iterate(ψ::CanonicalIMPS, args...) = iterate(ψ.AC, args...)
eachsite(ψ::CanonicalIMPS) = 1:length(ψ)

function Base.copy(ψ::CanonicalIMPS)
    return CanonicalIMPS(PeriodicVector([copy(a) for a in ψ.AL]),
                                PeriodicVector([copy(a) for a in ψ.AR]),
                                PeriodicVector([copy(c) for c in ψ.C]),
                                PeriodicVector([copy(a) for a in ψ.AC]))
end
function Base.similar(ψ::CanonicalIMPS{T}) where {T}
    return CanonicalIMPS{T}(similar(ψ.AL), similar(ψ.AR), similar(ψ.C), similar(ψ.AC))
end
function Base.circshift(ψ::CanonicalIMPS, n)
    return CanonicalIMPS(circshift(ψ.AL, n), circshift(ψ.AR, n),
                                circshift(ψ.C, n), circshift(ψ.AC, n))
end

scalartype(::Type{CanonicalIMPS{T}}) where {T} = T
scalartype(ψ::CanonicalIMPS) = scalartype(typeof(ψ))

phydims(ψ::CanonicalIMPS) = [size(ψ.AL[ℓ], 2) for ℓ in 1:length(ψ)]
bonddim(ψ::CanonicalIMPS, ℓ::Integer) = size(ψ.C[ℓ], 1)
max_bonddim(ψ::CanonicalIMPS) = maximum(bonddim(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`: elementwise conjugation of every tensor."
dag(ψ::CanonicalIMPS) =
    CanonicalIMPS(PeriodicVector(conj.(parent(ψ.AL))), PeriodicVector(conj.(parent(ψ.AR))),
                         PeriodicVector(conj.(parent(ψ.C))), PeriodicVector(conj.(parent(ψ.AC))))

"`LinearAlgebra.norm(ψ) = norm(ψ.AC[1])` (consistent with MPSKit)."
LinearAlgebra.norm(ψ::CanonicalIMPS) = norm(ψ.AC[1])

"""
    LinearAlgebra.normalize!(ψ::CanonicalIMPS)

Mirror of MPSKit's `normalize!(ψ::InfiniteMPS)` (`normalize!.(ψ.C);
normalize!.(ψ.AC)`): every bond matrix `C[ℓ]` and every center tensor `AC[ℓ]`
is normalized to unit Frobenius norm. For a mixed-canonical state the two
normalizations coincide (`‖AL·C‖ = ‖C‖` for left-orthogonal `AL`), so this is
exactly the ring normalization `⟨ψ, ψ⟩ = 1` (and
`norm(ψ) = norm(ψ.AC[1]) = 1`).
"""
function LinearAlgebra.normalize!(ψ::CanonicalIMPS)
    normalize!.(parent(ψ.C))
    normalize!.(parent(ψ.AC))
    return ψ
end

"""
    LinearAlgebra.dot(ψ₁, ψ₂; krylovdim = 30)

`⟨ψ₁|ψ₂⟩`: dominant eigenvalue of the double-layer `AL` transfer matrix
(KrylovKit Arnoldi).
"""
function LinearAlgebra.dot(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS; krylovdim::Int = 30)
    T = promote_type(scalartype(ψ₁), scalartype(ψ₂))
    v0 = vec(Matrix{T}(I, bonddim(ψ₁, 0), bonddim(ψ₂, 0)))
    tm = TransferMatrix(ψ₂.AL, ψ₁.AL)
    vals, vecs, _ = eigsolve(tm, v0, 1, :LM; krylovdim = krylovdim)
    λ = vals[1]
    return λ isa Number ? λ : only(λ)
end

"""
    fidelity(ψ₁, ψ₂) -> Real
    infidelity(ψ₁, ψ₂) -> Real

`fidelity = |⟨ψ₁|ψ₂⟩| / (‖ψ₁‖·‖ψ₂‖) ∈ [0, 1]`: the normalized ring overlap.
Invariant under independent overall phases **and** normalizations of the two
states, so it compares rays rather than representatives — the natural accuracy
measure for variational algebra results (which are only defined up to a global
phase). `infidelity = 1 − fidelity`.
"""
fidelity(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS) =
    abs(dot(ψ₁, ψ₂)) / (norm(ψ₁) * norm(ψ₂))
infidelity(ψ₁::CanonicalIMPS, ψ₂::CanonicalIMPS) = 1 - fidelity(ψ₁, ψ₂)

# ---------------- mixed-canonical diagnostics (after InfiniteTEMPO's ismixedcanonical) ----------------

"Rank-3 shared kernel of the mixed-canonical error (`Cv[ℓ]` sits on the bond to
the right of site ℓ, closed periodically)."
function _mixedcanonical_error(ALv::AbstractVector{<:Array{T,3}},
                               ARv::AbstractVector{<:Array{T,3}},
                               Cv::AbstractVector{<:AbstractMatrix{T}}) where {T}
    L = length(ALv)
    (length(ARv) == length(Cv) == L) ||
        throw(ArgumentError("inconsistent cell lengths of AL, AR, C: $(length(ALv)), $(length(ARv)), $(length(Cv))"))
    ϵ_left = ϵ_right = ϵ_mixed = 0.0
    for ℓ in 1:L
        @tensor g[a, b] := conj(ALv[ℓ][x, s, a]) * ALv[ℓ][x, s, b]
        ϵ_left = max(ϵ_left, norm(g - I))
        @tensor g[a, b] := ARv[ℓ][a, s, x] * conj(ARv[ℓ][b, s, x])
        ϵ_right = max(ϵ_right, norm(g - I))
        @tensor ac1[a, s, c] := ALv[ℓ][a, s, b] * Cv[ℓ][b, c]
        @tensor ac2[a, s, c] := Cv[_mod1(ℓ - 1, L)][a, b] * ARv[ℓ][b, s, c]
        ϵ_mixed = max(ϵ_mixed, norm(ac1 - ac2))
    end
    return ϵ_left, ϵ_right, ϵ_mixed
end

"""
    mixedcanonical_error(ψ) -> (ϵ_left, ϵ_right, ϵ_mixed)
    ismixedcanonical(ψ; tol = 1e-8, verbosity = 0) -> Bool

Diagnostics of the mixed-canonical form (after InfiniteTEMPO's
`ismixedcanonical`, mainly for debugging):

- `ϵ_left`  = max_ℓ ‖Σ AL[ℓ]†·AL[ℓ] − I‖ (left orthogonality)
- `ϵ_right` = max_ℓ ‖Σ AR[ℓ]·AR[ℓ]† − I‖ (right orthogonality)
- `ϵ_mixed` = max_ℓ ‖AL[ℓ]·C[ℓ] − C[ℓ-1]·AR[ℓ]‖ (mixed-canonical consistency,
  with C closed periodically)

`CanonicalIMPO` is checked analogously in the MPS view
`(wl, u·d, wr)`. `ismixedcanonical` returns `true` when all three errors are
≤ `tol`; `verbosity > 0` prints the errors.
"""
mixedcanonical_error(ψ::CanonicalIMPS) =
    _mixedcanonical_error(parent(ψ.AL), parent(ψ.AR), parent(ψ.C))

function ismixedcanonical(ψ::CanonicalIMPS; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(ψ)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end
