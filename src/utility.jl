const TO = TensorOperations

# scalartype reuses the existing TensorOperations (VectorInterface) function:
# methods for `Number` and `AbstractArray` already exist; we import it here and
# only extend it for the package's custom types.
import TensorOperations: scalartype

_mod1(ℓ::Integer, N::Integer) = mod1(Int(ℓ), Int(N))

# ---------------- periodic arrays (mirroring MPSKit's PeriodicArray/PeriodicVector) ----------------

"""
    PeriodicVector{T}(v::Vector{T})

Periodic vector: out-of-range indices wrap around via `mod1`
(mirrors MPSKit's `PeriodicVector`).
"""
struct PeriodicVector{T} <: AbstractVector{T}
    data::Vector{T}
end
# Note: the struct auto-generates the outer constructor PeriodicVector(v::Vector{T})

Base.size(p::PeriodicVector) = size(p.data)
Base.length(p::PeriodicVector) = length(p.data)
Base.getindex(p::PeriodicVector, i::Integer) = p.data[_mod1(i, length(p.data))]
Base.setindex!(p::PeriodicVector, v, i::Integer) = (p.data[_mod1(i, length(p.data))] = v; p)
Base.getindex(p::PeriodicVector, i::Colon) = p.data
Base.firstindex(p::PeriodicVector) = 1
Base.lastindex(p::PeriodicVector) = length(p.data)
Base.iterate(p::PeriodicVector, args...) = iterate(p.data, args...)
Base.similar(p::PeriodicVector{T}) where {T} = PeriodicVector(similar(p.data))
Base.copy(p::PeriodicVector) = PeriodicVector(copy(p.data))
Base.circshift(p::PeriodicVector, n) = PeriodicVector(circshift(p.data, n))
Base.:(==)(a::PeriodicVector, b::PeriodicVector) = a.data == b.data
_parent(p::PeriodicVector) = p.data

"""
    PeriodicArray{T,N}(a::Array{T,N})

Periodic array: the first dimension wraps around via `mod1`
(mirrors MPSKit's `PeriodicArray`).
"""
struct PeriodicArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
# Note: the struct auto-generates the outer constructor PeriodicArray(a::Array{T,N})

Base.size(p::PeriodicArray) = size(p.data)
Base.getindex(p::PeriodicArray, i::Integer, args...) = p.data[_mod1(i, size(p.data, 1)), args...]
Base.setindex!(p::PeriodicArray, v, i::Integer, args...) =
    (p.data[_mod1(i, size(p.data, 1)), args...] = v; p)
Base.getindex(p::PeriodicArray, i::Colon, args...) = p.data[args...]
Base.iterate(p::PeriodicArray, args...) = iterate(p.data, args...)
Base.copy(p::PeriodicArray) = PeriodicArray(copy(p.data))
_parent(p::PeriodicArray) = p.data

# ---------------- algorithm abstractions and default parameters ----------------

"""
    Algorithm

Abstract supertype of all algorithm parameter types (`VUMPS`, `IDMRG`, `TDVP`,
`VOMPS`, `WI`, `WII`, `LeftCanonical`, ...), aligned with MPSKit's `Algorithm`.
"""
abstract type Algorithm end

# ---------------- DynamicTol (mirroring MPSKit's DynamicTols) ----------------

"""
    DynamicTol(alg, tol_min, tol_max, tol_factor)

Dynamic tolerance wrapper for iterative algorithms (mirrors MPSKit's `DynamicTol`):
at each iteration the inner solver tolerance is updated from the current error
ϵ and the iteration count via

    new_tol = clamp(ϵ · tol_factor / √iter, tol_min, tol_max)

[`updatetol`](@ref) returns the inner algorithm object with its `tol` field updated.
"""
struct DynamicTol{A}
    alg::A
    tol_min::Float64
    tol_max::Float64
    tol_factor::Float64
    function DynamicTol(alg::A, tol_min::Real, tol_max::Real, tol_factor::Real) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("DynamicTol requires 0 ≤ tol_min ≤ tol_max"))
        return new{A}(alg, tol_min, tol_max, tol_factor)
    end
end
DynamicTol(alg; tol_min::Real = 1.0e-6, tol_max::Real = 1.0e-2, tol_factor::Real = 0.1) =
    DynamicTol(alg, tol_min, tol_max, tol_factor)

"Unwrapped algorithm: dynamic tolerance is a no-op (mirrors MPSKit)."
updatetol(alg, iter::Integer, ϵ::Real) = alg

function updatetol(alg::DynamicTol, iter::Integer, ϵ::Real)
    iter = max(iter, one(iter))
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
    return _updatetol(alg.alg, new_tol)
end

"Update the `tol` field of a KrylovKit Lanczos/Arnoldi solver (rebuilds the object,
equivalent to MPSKit's Accessors.@set)."
function _updatetol(alg::KrylovKit.Lanczos, tol::Real)
    return KrylovKit.Lanczos(; tol = tol, maxiter = alg.maxiter,
                             krylovdim = alg.krylovdim, eager = alg.eager,
                             verbosity = alg.verbosity)
end
function _updatetol(alg::KrylovKit.Arnoldi, tol::Real)
    return KrylovKit.Arnoldi(; tol = tol, maxiter = alg.maxiter,
                             krylovdim = alg.krylovdim, eager = alg.eager,
                             verbosity = alg.verbosity)
end
_updatetol(alg::NamedTuple, tol::Real) = merge(alg, (tol = tol,))

"""
    module Defaults

Default parameters and default algorithm constructors; fields align with
MPSKit's `Defaults`.
"""
module Defaults

using ..KrylovKit
using ..InfiniteMPSAlgorithms: DynamicTol

const VERBOSE_NONE = 0
const VERBOSE_WARN = 1
const VERBOSE_CONV = 2
const VERBOSE_ITER = 3
const VERBOSE_ALL = 4

const eltype = ComplexF64
const maxiter = 200
const tolgauge = 1.0e-13
const tol = 1.0e-10
const verbosity = 0
const krylovdim = 30
# dynamic tolerances (mirroring MPSKit Defaults)
const dynamic_tols = true
const tol_min = 1.0e-14
const tol_max = 1.0e-4
const eigs_tolfactor = 1.0e-3
const gauge_tolfactor = 1.0e-6
const envs_tolfactor = 1.0e-4

_finalize(iter, state, opp, envs) = (state, envs)

# stubs for the default orthonalization / SVD algorithms (attached outside the module)
function alg_orth end
function alg_svd end

"Eigenvalue solver (a KrylovKit algorithm object). `ishermitian=true` uses
Lanczos, otherwise Arnoldi."
function alg_eigsolve(; ishermitian = true, tol = tol, maxiter = maxiter,
                      eager = true, krylovdim = krylovdim,
                      dynamic_tols′ = dynamic_tols, tol_min′ = tol_min,
                      tol_max′ = tol_max, tol_factor = eigs_tolfactor)
    alg = ishermitian ?
          KrylovKit.Lanczos(; tol = tol, maxiter = maxiter, eager = eager,
                            krylovdim = krylovdim) :
          KrylovKit.Arnoldi(; tol = tol, maxiter = maxiter, eager = eager,
                            krylovdim = krylovdim)
    return dynamic_tols′ ? DynamicTol(alg, tol_min′, tol_max′, tol_factor) : alg
end

"Exponential solver (for local time evolution in TDVP)."
function alg_expsolve(; ishermitian = true, tol = tol, maxiter = maxiter, krylovdim = krylovdim)
    return ishermitian ?
           KrylovKit.Lanczos(; tol = tol, maxiter = maxiter, krylovdim = krylovdim) :
           KrylovKit.Arnoldi(; tol = tol, maxiter = maxiter, krylovdim = krylovdim)
end

alg_gauge(; tol = tolgauge, maxiter = maxiter,
          dynamic_tols′ = dynamic_tols, tol_min′ = tol_min, tol_max′ = tol_max,
          tol_factor = gauge_tolfactor) =
    dynamic_tols′ ? DynamicTol((; tol = tol, maxiter = maxiter), tol_min′, tol_max′, tol_factor) :
    (; tol = tol, maxiter = maxiter)
alg_environments(; tol = tol, maxiter = maxiter,
                 dynamic_tols′ = dynamic_tols, tol_min′ = tol_min, tol_max′ = tol_max,
                 tol_factor = envs_tolfactor) =
    dynamic_tols′ ? DynamicTol((; tol = tol, maxiter = maxiter), tol_min′, tol_max′, tol_factor) :
    (; tol = tol, maxiter = maxiter)
end

# default orthogonalization / SVD algorithms (QRpos/SDD are defined in tensorops,
# attached to Defaults here)
Defaults.alg_orth() = QRpos()
Defaults.alg_svd() = SDD()

# ---------------- iteration logging ----------------

function _logiter(io::IO, name::AbstractString, iter::Int, err::Real, extra::Pair...)
    str = join(["$k = $(repr(round(v; sigdigits = 8)))" for (k, v) in extra], ", ")
    @printf(io, "%s iter %4d : ϵ = %.3e %s\n", name, iter, err, str)
    return nothing
end
