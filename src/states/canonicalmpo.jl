"""
    CanonicalIMPO{T}

Mixed-canonical storage of an MPO with the same layout as
[`CanonicalIMPS`](@ref) (treating the MPO as an MPS; useful e.g. for
density matrices; storage and usage rules follow `CanonicalIMPS`):

- `AL[ℓ]::Array{T,4}`: MPO tensor `(wl, u, wr, d)`, left-orthogonal in the MPS view;
- `AR[ℓ]::Array{T,4}`: right-orthogonal in the MPS view;
- `C[ℓ]::Array{T,2}`: center matrix on bond ℓ;
- `AC[ℓ]::Array{T,4}`: center-canonical MPO tensor,
  `AC[ℓ] = AL[ℓ]·C[ℓ] = C[ℓ-1]·AR[ℓ]` (in the MPS view `(wl, u*d, wr)`;
  conventions identical to `CanonicalIMPS`).

Purpose: run MPS algorithms on MPOs (the dominant eigenvector is the optimal
smaller-bond approximation → MPO compression; canonical storage of
`hadamard(ψ, dag(ψ))`-type density matrices).
MPS view: `(wl, u, wr, d)` → permute `(1,2,4,3)` → reshape `(wl, u*d, wr)`.

Note: construction mixed-canonicalizes via `CanonicalIMPS`'s
`gaugefix!`. Gauge transformations telescope away in the periodic trace
representation (`tr(C⁻¹·X·C) = tr(X)`), so the operator's periodic
contraction is preserved up to the forced normalization scale of the MPS
view (`‖AC[1]‖ = 1` is not part of the gauge freedom) — the ray is exact.
"""
struct CanonicalIMPO{T<:Number} <: AbstractInfiniteMPO{T}
    AL::PeriodicVector{Array{T,4}}
    AR::PeriodicVector{Array{T,4}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,4}}

    function CanonicalIMPO{T}(AL::PeriodicVector{Array{T,4}},
                                     C::PeriodicVector{Array{T,2}},
                                     AR::PeriodicVector{Array{T,4}},
                                     AC::PeriodicVector{Array{T,4}}) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            ℓ1 = _mod1(ℓ + 1, L)
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond mismatch at site $ℓ"))
            size(AC[ℓ], 2) == size(AC[ℓ], 4) ||
                throw(DimensionMismatch("physical (u,d) mismatch at site $ℓ"))
        end
        new{T}(AL, AR, C, AC)
    end
end

"""
    CanonicalIMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...)

Construct from plain MPO tensors (mirrors `CanonicalIMPS(As)`): fuse the
physical legs into the vectorized MPS view (the [`vectorize`](@ref)
convention), then mixed-canonicalize with `gaugefix!`
(the operator ray is preserved exactly; see the type docstring for the
normalization-scale caveat).
"""
function CanonicalIMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...) where {T}
    ψ = CanonicalIMPS(vectorize(Ws); kwargs...)
    to4 = As -> devectorize(collect(As))
    return CanonicalIMPO{T}(PeriodicVector(to4(ψ.AL)), copy(ψ.C),
                            PeriodicVector(to4(ψ.AR)), PeriodicVector(to4(ψ.AC)))
end

"`_mulAL(W, C)`（rank-4）：MPS 视图上的 `AC = AL·C`（`CanonicalIMPO` 构造器的
`AC` 闭式装配 kernel；rank-3 版见 `canonicalmps.jl`）。"
function _mulAL(W::AbstractArray{T,4}, C::AbstractMatrix{T}) where {T}
    wl, u, wr, d = size(W)
    ALm = reshape(permutedims(W, (1, 2, 4, 3)), wl * u * d, wr)
    ACv = reshape(ALm * C, wl, u, d, size(C, 2))
    return permutedims(ACv, (1, 2, 4, 3))
end
_mul_ALC(AL::AbstractVector{<:Array{T,4}},
         C::AbstractVector{<:AbstractMatrix{T}}) where {T} =
    Array{T,4}[_mulAL(AL[ℓ], C[ℓ]) for ℓ in eachindex(AL)]

"""
    CanonicalIMPO(AL, C, AR, [AC]) -> CanonicalIMPO

Mirror of the `CanonicalIMPS` four-argument method：`AL`/`C`/`AR`/`AC` 分别为
左正交算符串、键中心矩阵、右正交算符串与中心串（`Vector` 或 `PeriodicVector`
均可）；`AC` 缺省由 `AC = AL·C` 闭式装配。输入即视为已处于相应规范，不做
`gaugefix!` 重整。
"""
CanonicalIMPO(AL::AbstractVector{<:Array{T,4}}, C::AbstractVector{<:Array{T,2}},
              AR::AbstractVector{<:Array{T,4}},
              AC::AbstractVector{<:Array{T,4}} = _mul_ALC(AL, C)) where {T} =
    CanonicalIMPO{T}(PeriodicVector(collect(AL)), PeriodicVector(collect(C)),
                     PeriodicVector(collect(AR)), PeriodicVector(collect(AC)))

# ---------------- interface (mirrors CanonicalIMPS) ----------------

Base.length(W::CanonicalIMPO) = length(W.AL)
Base.size(W::CanonicalIMPO, args...) = size(W.AL, args...)
eachsite(W::CanonicalIMPO) = 1:length(W)

function Base.copy(W::CanonicalIMPO{T}) where {T}
    return CanonicalIMPO{T}(PeriodicVector([copy(a) for a in W.AL]),
                                   PeriodicVector([copy(c) for c in W.C]),
                                   PeriodicVector([copy(a) for a in W.AR]),
                                   PeriodicVector([copy(a) for a in W.AC]))
end
function Base.similar(W::CanonicalIMPO{T}) where {T}
    return CanonicalIMPO{T}(similar(W.AL), similar(W.C), similar(W.AR), similar(W.AC))
end
function Base.circshift(W::CanonicalIMPO, n)
    return CanonicalIMPO{T}(circshift(W.AL, n), circshift(W.C, n),
                                   circshift(W.AR, n), circshift(W.AC, n))
end

"phydim(W, i): site `i` 的物理维（unit cell 内允许逐站不同；包内约定并强制
方算符 `du == dd`）。"
phydim(W::CanonicalIMPO, i::Integer) = size(W.AL[i], 2)
phydims(W::CanonicalIMPO) = [phydim(W, ℓ) for ℓ in 1:length(W)]
bonddim(W::CanonicalIMPO, ℓ::Integer) = size(W.C[_mod1(ℓ, length(W))], 1)
max_bonddim(W::CanonicalIMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"`dag(W)`: elementwise conjugation of every tensor (for overlap-type
contractions; not the operator-adjoint network)."
dag(W::CanonicalIMPO{T}) where {T} =
    CanonicalIMPO{T}(PeriodicVector(conj.(parent(W.AL))), PeriodicVector(conj.(parent(W.C))),
                            PeriodicVector(conj.(parent(W.AR))),
                            PeriodicVector(conj.(parent(W.AC))))

"`LinearAlgebra.norm(W) = norm(W.AC[1])` (consistent with CanonicalIMPS)."
LinearAlgebra.norm(W::CanonicalIMPO) = norm(W.AC[1])

"""
    LinearAlgebra.normalize!(W) -> W

Mirror of MPSKit's `normalize!(ψ::InfiniteMPS)` on the MPS view: every bond
matrix `C[ℓ]` and every center tensor `AC[ℓ]` is normalized to unit Frobenius
norm. For a mixed-canonical operator the two normalizations coincide
(`‖AL·C‖ = ‖C‖` for left-orthogonal `AL`), so this is exactly the ring
normalization `⟨W, W⟩ = 1` of the MPS view.
"""
function LinearAlgebra.normalize!(W::CanonicalIMPO)
    normalize!.(W.C)
    normalize!.(W.AC)
    return W
end

"""
    vectorize(Ws::AbstractVector{<:Array{T,4}}) -> Vector{Array{T,3}}

[`vectorize`](@ref) 的 raw-string 重载：MPO 张量串的 MPS 视图，
`(wl, u, wr, d)` → `(wl, u*d, wr)`（纯 reshape，逐站读入自身物理维）。
"""
function vectorize(Ws::AbstractVector{<:Array{T,4}}) where {T}
    out = Vector{Array{T,3}}(undef, length(Ws))
    for (ℓ, W) in enumerate(Ws)
        wl, u, wr, d = size(W)
        out[ℓ] = reshape(permutedims(W, (1, 2, 4, 3)), wl, u * d, wr)
    end
    return out
end

"""
    devectorize(As::AbstractVector{<:Array{T,3}}) -> Vector{Array{T,4}}

[`devectorize`](@ref) 的 raw-string 重载（[`vectorize`](@ref) 的逆）：拆分各站
融合物理维 `f` 为 `(u, d) = (r, r)`——包约定局域 `du == dd`，`f` 须为完全
平方数（`r` 可逐站不同）。
"""
function devectorize(As::AbstractVector{<:Array{T,3}}) where {T}
    out = Vector{Array{T,4}}(undef, length(As))
    for (ℓ, A) in enumerate(As)
        wl, f, wr = size(A)
        r = isqrt(f)
        r^2 == f || throw(ArgumentError(
            "fused physical dimension $f is not a perfect square (local du == dd required)"))
        out[ℓ] = permutedims(reshape(A, wl, r, r, wr), (1, 2, 4, 3))
    end
    return out
end

# ---------------- operator-algebra transforms (vectorize / devectorize / superoperator) ----------------

"""
    vectorize(W::CanonicalIMPO) -> CanonicalIMPS
    vectorize(W::Union{DenseIMPO,SparseIMPO}) -> DenseIMPS

Vectorize an MPO into an MPS on the doubled (bra ⊗ ket) space: the two
physical legs of every tensor are fused into one composite index

    f = u + du·(d - 1)

(`u` the bra / operator-row leg is the fast index — i.e. the row-major
vectorization of the operator matrix). The `CanonicalIMPO` method carries the
mixed-canonical gauge families (including `C`) over verbatim — a pure
reshape, so the conversion is exact and `dot(vectorize(A), vectorize(B))` is
the Hilbert–Schmidt inner product of the operators. The
`DenseIMPO`/`SparseIMPO` method is the uncanonicalized counterpart: a pure
fused view of the raw tensors with no canonicalization (mirror of the
`DenseIMPS`/`CanonicalIMPS` split).
"""
function vectorize(W::CanonicalIMPO)
    return CanonicalIMPS(PeriodicVector(vectorize(collect(W.AL))),
                         copy(W.C),
                         PeriodicVector(vectorize(collect(W.AR))),
                         PeriodicVector(vectorize(collect(W.AC))))
end
# DenseIMPO/SparseIMPO/DenseIMPS 方法（纯融合视图、互逆转换）见 operators/linalg.jl

"""
    devectorize(ψ::CanonicalIMPS) -> CanonicalIMPO
    devectorize(ψ::DenseIMPS) -> DenseIMPO

Inverse of [`vectorize`](@ref): split the fused physical index `f` of every
site tensor back into the bra/ket pair `(u, d)`. By the package convention the
local input/output dimensions agree (`du == dd` at every site), so each site's
fused dimension must be a perfect square `r[ℓ]²` — with `r` allowed to differ
from site to site (inhomogeneous unit cells). The conversion is exact: the
families are split by a pure reshape and the canonical gauge data (including
`C`) is carried over verbatim, without any re-canonicalization.
"""
function devectorize(ψ::CanonicalIMPS)
    T = scalartype(ψ)
    to4 = As -> devectorize(collect(As))
    return CanonicalIMPO{T}(PeriodicVector(to4(ψ.AL)), copy(ψ.C),
                            PeriodicVector(to4(ψ.AR)), PeriodicVector(to4(ψ.AC)))
end

"""
    fidelity(W₁, W₂) -> Real
    infidelity(W₁, W₂) -> Real

Hilbert–Schmidt fidelity of two operators stored as [`CanonicalIMPO`](@ref):
`|⟨W₁, W₂⟩_HS| / (‖W₁‖·‖W₂‖) ∈ [0, 1]`, computed as the
[`fidelity`](@ref) of the vectorized states (the ring overlap of the MPS
views is the C-weighted operator inner product). Invariant under overall
phases and scalings; `kwargs` 透传 vectorized 态的 `dot`（如 `krylovdim`）;
`infidelity = 1 − fidelity`.
"""
fidelity(W₁::CanonicalIMPO, W₂::CanonicalIMPO; kwargs...) =
    fidelity(vectorize(W₁), vectorize(W₂); kwargs...)
infidelity(W₁::CanonicalIMPO, W₂::CanonicalIMPO; kwargs...) =
    1 - fidelity(W₁, W₂; kwargs...)

"""
    mixedcanonical_error(W) -> (ϵ_left, ϵ_right, ϵ_mixed)
    ismixedcanonical(W; tol = 1e-8, verbosity = 0) -> Bool

Mixed-canonical diagnostics for [`CanonicalIMPO`](@ref): checked in the
vectorized MPS view `(wl, u·d, wr)` (kernel and conventions follow the
`CanonicalIMPS` methods).
"""
mixedcanonical_error(W::CanonicalIMPO) =
    (vψ = vectorize(W); _mixedcanonical_error(vψ.AL, vψ.AR, vψ.C))

function ismixedcanonical(W::CanonicalIMPO; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(W)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end

# ---------------- gauge interface (delegation to the states/ortho.jl kernels in the MPO view) ----------------

function gaugefix!(W::CanonicalIMPO, A, C₀ = W.C[end]; order = :LR, kwargs...)
    # 规范族逐位携带的 MPS 视图（gaugefix 只读写这些家族）
    ψ = vectorize(W)
    # A: rank-4 MPO 张量 → 融合视图；rank-3 视图直接使用
    Av = A isa AbstractVector{<:AbstractArray{<:Number,4}} ?
         vectorize(DenseIMPO(collect(A))).As : collect(A)
    gaugefix!(ψ, Av, C₀; order = order, kwargs...)
    # 写回 rank-4 家族
    W4 = devectorize(ψ)
    for ℓ in 1:length(W)
        W.AL[ℓ] = W4.AL[ℓ]
        W.AR[ℓ] = W4.AR[ℓ]
        W.C[ℓ] = ψ.C[ℓ]
        W.AC[ℓ] = W4.AC[ℓ]
    end
    return W
end

"""
    regauge!(AC::AbstractArray{T,4}, C::AbstractMatrix; alg) -> AL (rank-4)
    regauge!(CL::AbstractMatrix, AC::AbstractArray{T,4}; alg) -> AR (rank-4)

Rank-4 (MPO) tensor versions of [`regauge!](@ref): canonicalize in the MPS
view `(wl, u·d, wr)` and map back to rank-4; semantics identical to the
rank-3 methods.
"""
function regauge!(AC::AbstractArray{T,4}, C::AbstractMatrix{T}; alg = Defaults.alg_orth()) where {T}
    wl, u, wr, d = size(AC)
    ACv = reshape(permutedims(AC, (1, 2, 4, 3)), wl, u * d, wr)
    ALv = regauge!(ACv, C; alg = alg)
    return permutedims(reshape(ALv, wl, u, d, wr), (1, 2, 4, 3))
end

function regauge!(CL::AbstractMatrix{T}, AC::AbstractArray{T,4}; alg = Defaults.alg_orth()) where {T}
    wl, u, wr, d = size(AC)
    ACv = reshape(permutedims(AC, (1, 2, 4, 3)), wl, u * d, wr)
    ARv = regauge!(CL, ACv; alg = alg)
    return permutedims(reshape(ARv, wl, u, d, wr), (1, 2, 4, 3))
end
"""
    changebond!(W::CanonicalIMPO; D::Int, noise::Real = 1e-10) -> W

[`changebond!`](@ref) 的 MPO 版（MPS 视图 `(wl, u·d, wr)` 的键 profile 调整）：
`AL` 的左右键直接 `_resize_dim` 到 `D`（不足则扩容、超出则截取前导子块），再用
[`CanonicalIMPO`](@ref) 重新包装以恢复混合规范 —— 与 FiniteMPSAlgorithms 的同名
函数一致。各 bond 已等于 `D` 时直接返回、不做任何改动。

同 MPS 版：**infinite MPO 的键维不受物理维乘积限制**，忠实按用户给的 `D`，
不做 `min(D, ∏d)` 之类的截断。

`noise` 填充扩容出的新块（`0` 即零填充，态不变；`noise ≠ 0` 时填 `noise·randn`）：
零填充得到秩亏的态，规范不被唯一确定，单点 TDVP/VUMPS 等依赖规范的算法会因此
给出表示依赖的结果（FiniteMPSAlgorithms 的 `TDVP1` docstring 记录了同一现象）。
"""
function changebond!(W::CanonicalIMPO; D::Int, noise::Real = 1e-10)
    N = length(W)
    b = fill(D, N)
    # 各 bond 已等于 D ⇒ 无需改动，提前返回
    all(bonddim(W, ℓ) == b[ℓ] for ℓ in 1:N) && return W
    for ℓ in 1:N
        ℓm = _mod1(ℓ - 1, N)
        W.AL[ℓ] = _resize_dim(W.AL[ℓ], 1, b[ℓm]; noise = noise)
        W.AL[ℓ] = _resize_dim(W.AL[ℓ], 3, b[ℓ]; noise = noise)
    end
    y = CanonicalIMPO(collect(W.AL))
    copy!(W.AL, y.AL)
    copy!(W.AR, y.AR)
    copy!(W.C, y.C)
    copy!(W.AC, y.AC)
    return W
end

# ---------------- truncate!（逐键 C 截断；语义对齐 InfiniteTEMPO 的 toipt!） ----------------

"""
    truncate!(W::CanonicalIMPO; trunc = DefaultTruncation) -> (W, err)

[`truncate!`](@ref) 的 `CanonicalIMPO` 版（InfiniteTEMPO `toipt!` finalize 语义）：
逐键 SVD 截断 `C[ℓ]`、中心矩阵取对角谱，相邻键的 unitary 因子把 `AR`（rank-4）
旋转到与对角 `C` 一致，`AL` 在 MPS 视图 `(wl, u·d, wr)` 上右除装配
（`AL'·C' = AC'`）。四族逐槽写回（原地），不做重正则化/重建规范；正则性
偏差与丢弃谱权重同量级（`err`），完整约定见 `CanonicalIMPS` 方法。
"""
function truncate!(W::CanonicalIMPO; trunc::TruncationScheme = DefaultTruncation)
    N = length(W)
    T = scalartype(W)
    # 逐键 SVD 截断 C：中心矩阵取对角谱，右因子（行正交的 V†）留作规范旋转
    Cs = Vector{Matrix{T}}(undef, N)
    Sb = Vector{Vector{Float64}}(undef, N)
    Vs = Vector{Matrix{T}}(undef, N)
    err = 0.0
    for ℓ in 1:N
        _, S, V, e = tsvd(W.C[ℓ], (1,), (2,); trunc)
        Cs[ℓ] = Matrix{T}(Diagonal(S))
        Sb[ℓ] = S
        Vs[ℓ] = V
        err = max(err, e)
    end
    # 相邻键的 unitary 因子把 AR 旋转到与对角 C 一致；AL 在 MPS 视图上右除装配
    ALs = Vector{Array{T,4}}(undef, N)
    ARs = Vector{Array{T,4}}(undef, N)
    ACs = Vector{Array{T,4}}(undef, N)
    for ℓ in 1:N
        @tensor Ar[a, u, c, d] := Vs[_mod1(ℓ - 1, N)][a, b] * W.AR[ℓ][b, u, e, d] *
                                 conj(Vs[ℓ][c, e])
        AC = Ar .* reshape(Sb[_mod1(ℓ - 1, N)], :, 1, 1, 1)    # AC = diag(s)·AR（行缩放）
        wl, u, wr, dd = size(AC)
        # AL·C = AC 在 MPS 视图 (wl, u·d, wr) 上求解：AL_view = AC_view / C
        ACview = reshape(permutedims(AC, (1, 2, 4, 3)), wl * u * dd, wr)
        ALs[ℓ] = permutedims(reshape(ACview / Cs[ℓ], wl, u, dd, wr), (1, 2, 4, 3))
        ARs[ℓ] = Ar
        ACs[ℓ] = AC
    end
    copy!(W.AL, ALs)
    copy!(W.AR, ARs)
    copy!(W.C, Cs)
    copy!(W.AC, ACs)
    return W, err
end
