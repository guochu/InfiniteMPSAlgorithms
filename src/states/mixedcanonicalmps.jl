"""
    InfiniteCanonicalMPS{T}

无限（周期平铺）MPS 的混合规范存储，数据布局与 MPSKit 的 `InfiniteMPS` 一致：

- `AL::PeriodicVector{Array{T,3}}`：左规范 site 张量（`(Dl, s, Dr)`），`Σ AL†·AL = 1`；
- `AR::PeriodicVector{Array{T,3}}`：右规范 site 张量，`Σ AR·AR† = 1`；
- `C::PeriodicVector{Array{T,2}}`：bond ℓ（site ℓ 与 ℓ+1 之间）的中心矩阵；
- `AC::PeriodicVector{Array{T,3}}`：中心规范 site 张量。

约定（与 MPSKit 相同）：`AL[i] * C[i] = AC[i] = C[i-1] * AR[i]`。
所有下标按 mod N 循环。
"""
struct InfiniteCanonicalMPS{T}
    AL::PeriodicVector{Array{T,3}}
    AR::PeriodicVector{Array{T,3}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,3}}

    function InfiniteCanonicalMPS{T}(AL::PeriodicVector{Array{T,3}},
                                     AR::PeriodicVector{Array{T,3}},
                                     C::PeriodicVector{Array{T,2}},
                                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
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

InfiniteCanonicalMPS(AL::PeriodicVector{Array{T,3}}, AR::PeriodicVector{Array{T,3}},
                     C::PeriodicVector{Array{T,2}},
                     AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T} =
    InfiniteCanonicalMPS{T}(AL, AR, C, AC)

"""
    InfiniteCanonicalMPS(As::AbstractVector{<:Array{T,3}}; kwargs...)

由普通 site 张量构造（对标 MPSKit 的 `InfiniteMPS(A)`）：
`AR = A`，从 `C₀ = I` 出发 `gaugefix!(; order = :LR)`，`AC = AL·C`。
"""
function InfiniteCanonicalMPS(As::AbstractVector{<:Array{T,3}}; kwargs...) where {T}
    for ℓ in 1:length(As)-1
        size(As[ℓ], 3) == size(As[ℓ+1], 1) ||
            throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
    end
    AR = PeriodicVector([copy(a) for a in As])
    AL = PeriodicVector([similar(a) for a in AR])
    AC = PeriodicVector([similar(a) for a in AR])
    D = size(AR[1], 1)
    C = PeriodicVector([similar(AR[1], D, size(AR[_mod1(ℓ + 1, length(AR))], 1)) for ℓ in 1:length(AR)])
    ψ = InfiniteCanonicalMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, As, Matrix{T}(I, D, D); kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

"""
    InfiniteCanonicalMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix; kwargs...)

由左规范张量 + 初始规范矩阵构造（对标 MPSKit 的 `InfiniteMPS(AL, C₀)`）：
`gaugefix!` 右规范（`order = :R`），`AC = AL·C`。
"""
function InfiniteCanonicalMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix;
                              kwargs...) where {T}
    AL = PeriodicVector([copy(a) for a in ALs])
    AR = PeriodicVector([similar(a) for a in AL])
    AC = PeriodicVector([similar(a) for a in AL])
    C = PeriodicVector([similar(AL[1], size(C₀, 1), size(C₀, 2)) for _ in 1:length(AL)])
    ψ = InfiniteCanonicalMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, ALs, C₀; order = :R, kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

# ---------------- 接口 ----------------

Base.length(ψ::InfiniteCanonicalMPS) = length(ψ.AL)
Base.size(ψ::InfiniteCanonicalMPS, args...) = size(ψ.AL, args...)
Base.getindex(ψ::InfiniteCanonicalMPS, ℓ::Integer) = ψ.AC[ℓ]
Base.setindex!(ψ::InfiniteCanonicalMPS, v::Array, ℓ::Integer) = (ψ.AC[ℓ] = v; ψ)
Base.firstindex(ψ::InfiniteCanonicalMPS) = 1
Base.lastindex(ψ::InfiniteCanonicalMPS) = length(ψ)
Base.iterate(ψ::InfiniteCanonicalMPS, args...) = iterate(ψ.AC, args...)
eachsite(ψ::InfiniteCanonicalMPS) = 1:length(ψ)

function Base.copy(ψ::InfiniteCanonicalMPS)
    return InfiniteCanonicalMPS(PeriodicVector([copy(a) for a in ψ.AL]),
                                PeriodicVector([copy(a) for a in ψ.AR]),
                                PeriodicVector([copy(c) for c in ψ.C]),
                                PeriodicVector([copy(a) for a in ψ.AC]))
end
function Base.similar(ψ::InfiniteCanonicalMPS{T}) where {T}
    return InfiniteCanonicalMPS{T}(similar(ψ.AL), similar(ψ.AR), similar(ψ.C), similar(ψ.AC))
end
function Base.circshift(ψ::InfiniteCanonicalMPS, n)
    return InfiniteCanonicalMPS(circshift(ψ.AL, n), circshift(ψ.AR, n),
                                circshift(ψ.C, n), circshift(ψ.AC, n))
end

scalartype(::Type{InfiniteCanonicalMPS{T}}) where {T} = T
scalartype(ψ::InfiniteCanonicalMPS) = scalartype(typeof(ψ))

phydims(ψ::InfiniteCanonicalMPS) = [size(ψ.AL[ℓ], 2) for ℓ in 1:length(ψ)]
bonddim(ψ::InfiniteCanonicalMPS, ℓ::Integer) = size(ψ.C[ℓ], 1)
max_bonddim(ψ::InfiniteCanonicalMPS) = maximum(bonddim(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`：逐张量共轭。"
dag(ψ::InfiniteCanonicalMPS) =
    InfiniteCanonicalMPS(PeriodicVector(conj.(parent(ψ.AL))), PeriodicVector(conj.(parent(ψ.AR))),
                         PeriodicVector(conj.(parent(ψ.C))), PeriodicVector(conj.(parent(ψ.AC))))

"`LinearAlgebra.norm(ψ) = norm(ψ.AC[1])`（与 MPSKit 一致）。"
LinearAlgebra.norm(ψ::InfiniteCanonicalMPS) = norm(ψ.AC[1])

"""
    LinearAlgebra.normalize!(ψ::InfiniteCanonicalMPS)

归一化 `C` 与 `AC`（与 MPSKit 一致：规范形状由 AL/AR 保持）。
"""
function LinearAlgebra.normalize!(ψ::InfiniteCanonicalMPS)
    normalize!.(parent(ψ.C))
    normalize!.(parent(ψ.AC))
    return ψ
end

"""
    LinearAlgebra.dot(ψ₁, ψ₂; krylovdim = 30)

`⟨ψ₁|ψ₂⟩`：`AL` 双层转移矩阵的主本征值（KrylovKit Arnoldi）。
"""
function LinearAlgebra.dot(ψ₁::InfiniteCanonicalMPS, ψ₂::InfiniteCanonicalMPS; krylovdim::Int = 30)
    T = promote_type(scalartype(ψ₁), scalartype(ψ₂))
    v0 = vec(Matrix{T}(I, bonddim(ψ₁, 0), bonddim(ψ₂, 0)))
    tm = TransferMatrix(ψ₂.AL, ψ₁.AL)
    vals, vecs, _ = eigsolve(tm, v0, 1, :LM; krylovdim = krylovdim)
    λ = vals[1]
    return λ isa Number ? λ : only(λ)
end

# ---------------- 混合规范诊断（参考 InfiniteTEMPO 的 ismixedcanonical） ----------------

"混合规范误差的 rank-3 共享核（`Cv[ℓ]` 位于 site ℓ 右键，周期闭合）。"
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

混合规范形式的诊断（参考 InfiniteTEMPO 的 `ismixedcanonical`，主要用于 debug）：

- `ϵ_left`  = max_ℓ ‖Σ AL[ℓ]†·AL[ℓ] − I‖（左正交性）
- `ϵ_right` = max_ℓ ‖Σ AR[ℓ]·AR[ℓ]† − I‖（右正交性）
- `ϵ_mixed` = max_ℓ ‖AL[ℓ]·C[ℓ] − C[ℓ-1]·AR[ℓ]‖（混合规范一致性，C 周期闭合）

`InfiniteCanonicalMPO` 在 MPS 视图 `(wl, u·d, wr)` 意义下同样检查。
`ismixedcanonical` 在三个误差均 ≤ `tol` 时返回 `true`；`verbosity > 0` 时打印误差。
"""
mixedcanonical_error(ψ::InfiniteCanonicalMPS) =
    _mixedcanonical_error(parent(ψ.AL), parent(ψ.AR), parent(ψ.C))

function ismixedcanonical(ψ::InfiniteCanonicalMPS; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(ψ)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end
