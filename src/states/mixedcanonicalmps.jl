"""
    MixedCanonicalMPS{T}

无限（周期平铺）MPS 的混合规范存储，数据布局与 MPSKit 的 `InfiniteMPS` 一致：

- `AL::PeriodicVector{Array{T,3}}`：左规范 site 张量（`(Dl, s, Dr)`），`Σ AL†·AL = 1`；
- `AR::PeriodicVector{Array{T,3}}`：右规范 site 张量，`Σ AR·AR† = 1`；
- `C::PeriodicVector{Array{T,2}}`：bond ℓ（site ℓ 与 ℓ+1 之间）的中心矩阵；
- `AC::PeriodicVector{Array{T,3}}`：中心规范 site 张量。

约定（与 MPSKit 相同）：`AL[i] * C[i] = AC[i] = C[i-1] * AR[i]`。
所有下标按 mod N 循环。
"""
struct MixedCanonicalMPS{T}
    AL::PeriodicVector{Array{T,3}}
    AR::PeriodicVector{Array{T,3}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,3}}

    function MixedCanonicalMPS{T}(AL::PeriodicVector{Array{T,3}},
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

MixedCanonicalMPS(AL::PeriodicVector{Array{T,3}}, AR::PeriodicVector{Array{T,3}},
                  C::PeriodicVector{Array{T,2}},
                  AC::PeriodicVector{Array{T,3}} = _mul_ALC(AL, C)) where {T} =
    MixedCanonicalMPS{T}(AL, AR, C, AC)

"""
    MixedCanonicalMPS(As::AbstractVector{<:Array{T,3}}; kwargs...)

由普通 site 张量构造（对标 MPSKit 的 `InfiniteMPS(A)`）：
`AR = A`，从 `C₀ = I` 出发 `gaugefix!(; order = :LR)`，`AC = AL·C`。
"""
function MixedCanonicalMPS(As::AbstractVector{<:Array{T,3}}; kwargs...) where {T}
    for ℓ in 1:length(As)-1
        size(As[ℓ], 3) == size(As[ℓ+1], 1) ||
            throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
    end
    AR = PeriodicVector([copy(a) for a in As])
    AL = PeriodicVector([similar(a) for a in AR])
    AC = PeriodicVector([similar(a) for a in AR])
    D = size(AR[1], 1)
    C = PeriodicVector([similar(AR[1], D, size(AR[_mod1(ℓ + 1, length(AR))], 1)) for ℓ in 1:length(AR)])
    ψ = MixedCanonicalMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, As, Matrix{T}(I, D, D); kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

"""
    MixedCanonicalMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix; kwargs...)

由左规范张量 + 初始规范矩阵构造（对标 MPSKit 的 `InfiniteMPS(AL, C₀)`）：
`gaugefix!` 右规范（`order = :R`），`AC = AL·C`。
"""
function MixedCanonicalMPS(ALs::AbstractVector{<:Array{T,3}}, C₀::AbstractMatrix;
                           kwargs...) where {T}
    AL = PeriodicVector([copy(a) for a in ALs])
    AR = PeriodicVector([similar(a) for a in AL])
    AC = PeriodicVector([similar(a) for a in AL])
    C = PeriodicVector([similar(AL[1], size(C₀, 1), size(C₀, 2)) for _ in 1:length(AL)])
    ψ = MixedCanonicalMPS{T}(AL, AR, C, AC)
    gaugefix!(ψ, ALs, C₀; order = :R, kwargs...)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end

# ---------------- 接口 ----------------

Base.length(ψ::MixedCanonicalMPS) = length(ψ.AL)
Base.size(ψ::MixedCanonicalMPS, args...) = size(ψ.AL, args...)
Base.getindex(ψ::MixedCanonicalMPS, ℓ::Integer) = ψ.AC[ℓ]
Base.setindex!(ψ::MixedCanonicalMPS, v::Array, ℓ::Integer) = (ψ.AC[ℓ] = v; ψ)
Base.firstindex(ψ::MixedCanonicalMPS) = 1
Base.lastindex(ψ::MixedCanonicalMPS) = length(ψ)
Base.iterate(ψ::MixedCanonicalMPS, args...) = iterate(ψ.AC, args...)
eachsite(ψ::MixedCanonicalMPS) = 1:length(ψ)

function Base.copy(ψ::MixedCanonicalMPS)
    return MixedCanonicalMPS(PeriodicVector([copy(a) for a in ψ.AL]),
                             PeriodicVector([copy(a) for a in ψ.AR]),
                             PeriodicVector([copy(c) for c in ψ.C]),
                             PeriodicVector([copy(a) for a in ψ.AC]))
end
function Base.similar(ψ::MixedCanonicalMPS{T}) where {T}
    return MixedCanonicalMPS{T}(similar(ψ.AL), similar(ψ.AR), similar(ψ.C), similar(ψ.AC))
end
function Base.circshift(ψ::MixedCanonicalMPS, n)
    return MixedCanonicalMPS(circshift(ψ.AL, n), circshift(ψ.AR, n),
                             circshift(ψ.C, n), circshift(ψ.AC, n))
end

scalartype(::Type{MixedCanonicalMPS{T}}) where {T} = T
scalartype(ψ::MixedCanonicalMPS) = scalartype(typeof(ψ))

physicaldims(ψ::MixedCanonicalMPS) = [size(ψ.AL[ℓ], 2) for ℓ in 1:length(ψ)]
bond(ψ::MixedCanonicalMPS, ℓ::Integer) = size(ψ.C[ℓ], 1)
maxbond(ψ::MixedCanonicalMPS) = maximum(bond(ψ, ℓ) for ℓ in 1:length(ψ))

"`dag(ψ)`：逐张量共轭。"
dag(ψ::MixedCanonicalMPS) =
    MixedCanonicalMPS(PeriodicVector(conj.(parent(ψ.AL))), PeriodicVector(conj.(parent(ψ.AR))),
                      PeriodicVector(conj.(parent(ψ.C))), PeriodicVector(conj.(parent(ψ.AC))))

"`LinearAlgebra.norm(ψ) = norm(ψ.AC[1])`（与 MPSKit 一致）。"
LinearAlgebra.norm(ψ::MixedCanonicalMPS) = norm(ψ.AC[1])

"""
    LinearAlgebra.normalize!(ψ::MixedCanonicalMPS)

归一化 `C` 与 `AC`（与 MPSKit 一致：规范形状由 AL/AR 保持）。
"""
function LinearAlgebra.normalize!(ψ::MixedCanonicalMPS)
    normalize!.(parent(ψ.C))
    normalize!.(parent(ψ.AC))
    return ψ
end

"""
    LinearAlgebra.dot(ψ₁, ψ₂; krylovdim = 30)

`⟨ψ₁|ψ₂⟩`：`AL` 双层转移矩阵的主本征值（KrylovKit Arnoldi）。
"""
function LinearAlgebra.dot(ψ₁::MixedCanonicalMPS, ψ₂::MixedCanonicalMPS; krylovdim::Int = 30)
    T = promote_type(scalartype(ψ₁), scalartype(ψ₂))
    v0 = vec(Matrix{T}(I, bond(ψ₁, 0), bond(ψ₂, 0)))
    tm = TransferMatrix(ψ₂.AL, ψ₁.AL)
    vals, vecs, _ = eigsolve(tm, v0, 1, :LM; krylovdim = krylovdim)
    λ = vals[1]
    return λ isa Number ? λ : only(λ)
end
