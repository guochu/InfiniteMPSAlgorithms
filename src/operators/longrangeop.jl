# ---------------- exponentially decaying long-range operators ----------------
#
# Mirrors TEMPO's `schurmpo` layer: exponentially decaying long-range
# interaction terms of the form `α·λ^d·(â ⊗ m̂^⊗(d-1) ⊗ b̂)` (distance d ≥ 1,
# where the operators m̂ sit on the d − 1 intermediate sites), converted
# directly into a [`SchurMPOTensor`](@ref) via the Schur (upper-triangular
# block) construction of arXiv:1407.1832:
#
# ```math
# \begin{pmatrix}
# 1 & α·a & 0 \\
# 0 & λ·m̂ & λ·b \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# A sum of terms with different `(α_p, λ_p)` opens one channel per term.

"""
    ExpDecayOpTerm(a, m, b, α = 1.0, λ = 1.0)

A single exponentially decaying long-range interaction term
`α·λ^d·(â ⊗ m̂^⊗(d-1) ⊗ b̂)`: `a` opens the channel on site `i`, the operators
`m` propagate through the `d − 1` intermediate sites, `b` closes it on site
`i + d`, and `λ` is the per-step decay factor. Convertible to a
[`SchurMPOTensor`](@ref) via `SchurMPOTensor(term)`.
"""
struct ExpDecayOpTerm{M1<:AbstractMatrix,M<:AbstractMatrix,M2<:AbstractMatrix,T<:Number}
    a::M1
    m::M
    b::M2
    α::T
    λ::T
end

function ExpDecayOpTerm(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
                        α::Number = 1.0, λ::Number = 1.0)
    T = promote_type(typeof(α), typeof(λ))
    return ExpDecayOpTerm(a, m, b, convert(T, α), convert(T, λ))
end

scalartype(::Type{ExpDecayOpTerm{M1,M,M2,T}}) where {M1,M,M2,T} =
    promote_type(scalartype(M1), scalartype(M), scalartype(M2), T)

"""
    ExpDecayOpSum(a, m, b, αs, λs)

A sum of exponentially decaying long-range interaction terms sharing the same
operator triple `(a, m, b)`: `Σ_p αs[p]·λs[p]^d·(â ⊗ m̂^⊗(d-1) ⊗ b̂)` — e.g.
the exponential (Prony) expansion of a power-law decay. Convertible to a
[`SchurMPOTensor`](@ref) via `SchurMPOTensor(sum)` (one Schur channel per
term).
"""
struct ExpDecayOpSum{M1<:AbstractMatrix,M<:AbstractMatrix,M2<:AbstractMatrix,T<:Number}
    a::M1
    m::M
    b::M2
    αs::Vector{T}
    λs::Vector{T}
end

function ExpDecayOpSum(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
                       αs::Vector{<:Number}, λs::Vector{<:Number})
    (length(αs) == length(λs)) ||
        throw(DimensionMismatch("αs and λs must have equal lengths"))
    T = promote_type(eltype(αs), eltype(λs))
    return ExpDecayOpSum(a, m, b, convert(Vector{T}, αs), convert(Vector{T}, λs))
end

scalartype(::Type{ExpDecayOpSum{M1,M,M2,T}}) where {M1,M,M2,T} =
    promote_type(scalartype(M1), scalartype(M), scalartype(M2), T)

"""
    SchurMPOTensor(t::ExpDecayOpTerm) -> SchurMPOTensor
    SchurMPOTensor(s::ExpDecayOpSum) -> SchurMPOTensor

Convert an exponentially decaying long-range operator into a
[`SchurMPOTensor`](@ref) with `N` channels (`N = 1` for a single term,
`N = length(αs)` for a sum), following the Schur construction:

- `cell[i+1, i+1] = λs[i]·m` (channel self-propagation with decay),
- `cell[1, i+1] = αs[i]·a` (channel opening),
- `cell[i+1, end] = λs[i]·b` (channel closing),
- `cell[1, 1] = cell[end, end] = 1`, `cell[1, end] = 0`.
"""
function SchurMPOTensor(s::ExpDecayOpSum)
    isempty(s.αs) && throw(ArgumentError("ExpDecayOpSum needs at least one term"))
    N = length(s.αs)
    T = scalartype(s)
    d = size(s.m, 1)
    (size(s.a, 1) == size(s.a, 2) == d && size(s.b, 1) == size(s.b, 2) == d) ||
        throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
    cell = Matrix{Any}(undef, N + 2, N + 2)
    cell .= zero(T)
    cell[1, 1] = one(T)
    cell[end, end] = one(T)
    for i in 1:N
        cell[i+1, i+1] = s.λs[i] * s.m
        cell[1, i+1] = s.αs[i] * s.a
        cell[i+1, end] = s.λs[i] * s.b
    end
    return SchurMPOTensor(cell)
end

function SchurMPOTensor(t::ExpDecayOpTerm)
    return SchurMPOTensor(ExpDecayOpSum(t.a, t.m, t.b, [t.α], [t.λ]))
end
