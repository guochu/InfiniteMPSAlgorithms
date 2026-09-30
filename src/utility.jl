const TO = TensorOperations

# scalartype 复用 TensorOperations（VectorInterface）的既有函数：`Number` 与
# `AbstractArray` 的方法已存在，本包只为其自定义类型扩展；`import` 集中在主文件。

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
Base.circshift(p::PeriodicVector, n::Integer) = PeriodicVector(circshift(p.data, n))
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

# ---------------- infinite chain / operator abstract types ----------------

"""
    AbstractInfiniteMPS{T<:Number}

Abstract supertype of the infinite (periodically tiled) MPS representations:
[`CanonicalIMPS`](@ref)（混合规范存储）与 [`DenseIMPS`](@ref)（原始张量串）。
两者共享 `AL`/`AR`/`AC`/`C` 的家族访问接口（`DenseIMPS` 经 `getproperty` 视图
提供：`AL`/`AR`/`AC` 返回数据的周期视图、`C` 返回 [`BondView`](@ref) 单位矩阵
视图），因此以 `ψ.AL[ℓ]` / `ψ.C[ℓ]`（周期下标）消费家族数据的算法对两者皆可用。
"""
abstract type AbstractInfiniteMPS{T<:Number} end

"""
    AbstractInfiniteMPO{T<:Number}

Abstract supertype of the infinite MPO representations: [`CanonicalIMPO`](@ref)
（混合规范存储）、[`DenseIMPO`](@ref)（原始张量串）与 [`SparseIMPO`](@ref)
（Schur 稀疏哈密顿量容器）。`AL`/`AR`/`AC` 周期视图 + `C` 单位矩阵视图的家族
接口由 `DenseIMPO`/`CanonicalIMPO` 提供（变分算术通道 compress/mult/hadamard
据此直接消费两类输入）；`SparseIMPO` 不提供家族视图、不参与这些通道（需要
dense/canonical 表示时显式 `DenseIMPO(H)`/`CanonicalIMPO(...)` 转换，或走其
Schur 专用通道：DMRGCache/expectationvalue/make_time_mpo）。
"""
abstract type AbstractInfiniteMPO{T<:Number} end

"`scalartype`（家族标量类型）：由抽象类型的标量参数直接给出——五个具体
链/算符类型（`DenseIMPS`/`CanonicalIMPS`/`DenseIMPO`/`CanonicalIMPO`/
`SparseIMPO`）共用，子类无需再各自定义。"
scalartype(::Type{<:AbstractInfiniteMPS{T}}) where {T} = T
scalartype(::Type{<:AbstractInfiniteMPO{T}}) where {T} = T

"""
    BondView(parent) <: AbstractVector{Matrix{scalartype(parent)}}

`C` 家族的单位矩阵视图（[`DenseIMPS`](@ref)/[`DenseIMPO`](@ref)/[`SparseIMPO`](@ref)
的 `getproperty` 接口用）：无中心矩阵数据的链上，`C[ℓ]` 按约定取**单位矩阵**，
维度由 `parent.AL` 的键维推断（`C[ℓ]` 作用在键 ℓ 上，为
`(site ℓ 右键维, site ℓ+1 左键维)` 的恒等方阵）。只读视图：`getindex` 以
`length(parent)` 为周期取模（与 `PeriodicVector` 一致），`parent` 字段持有
链/算符本身。
"""
struct BondView{P,T<:Number} <: AbstractVector{Matrix{T}}
    parent::P
end
BondView(parent::Union{AbstractInfiniteMPS,AbstractInfiniteMPO}) =
    BondView{typeof(parent),scalartype(parent)}(parent)

Base.size(v::BondView) = (length(v.parent),)
Base.length(v::BondView) = length(v.parent)
function Base.getindex(v::BondView{P,T}, i::Integer) where {P,T}
    N = length(v.parent)
    ℓ = _mod1(i, N)
    dr = size(v.parent.AL[ℓ], 3)
    dl = size(v.parent.AL[_mod1(ℓ + 1, N)], 1)
    return Matrix{T}(I, dr, dl)
end

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
    fixedpoint(operator, x₀, which, alg) -> (λ, v)

Dominant eigenpair of the transfer map (mirrors MPSKit's `fixedpoint`;
internally KrylovKit.eigsolve). `alg` is a KrylovKit `Lanczos`/`Arnoldi`
algorithm object.
"""
function fixedpoint(operator, x₀, which::Symbol, alg::KrylovKit.KrylovAlgorithm)
    isherm = alg isa KrylovKit.Lanczos
    vals, vecs, _ = _eigsolve(operator, x₀, 1, which;
                              ishermitian = isherm, tol = alg.tol,
                              krylovdim = alg.krylovdim, maxiter = alg.maxiter,
                              eager = true)
    return vals[1], vecs[1]
end

"DynamicTol wrapper: uses the inner Krylov algorithm's initial tolerance."
fixedpoint(operator, x₀, which::Symbol, alg::DynamicTol) =
    fixedpoint(operator, x₀, which, alg.alg)

"NamedTuple algorithm（环境通道的 `alg_environments` 形态，DynamicTol 解包后
落到这里）：非厄米迁移矩阵的 `which` 主本征对；`krylovdim` 取
`Defaults.krylovdim`。"
function fixedpoint(operator, x₀, which::Symbol, alg::NamedTuple;
                    krylovdim::Int = Defaults.krylovdim)
    vals, vecs, _ = _eigsolve(operator, x₀, 1, which; ishermitian = false,
                              tol = alg.tol, krylovdim = krylovdim,
                              maxiter = alg.maxiter, eager = true)
    return vals[1], vecs[1]
end

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
const D = 64
const maxiter = 300
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

"""
内部 [`KrylovKit.eigsolve`](@ref) 封装：默认参数一律取自 [`Defaults`](@ref)
（`tol = Defaults.tol`、`maxiter = Defaults.maxiter`、`krylovdim = Defaults.krylovdim`），
并在收敛数不足（`info.converged < howmany`）时 `@warn`。包内所有特征值求解都
经由这里，保证默认行为一致；`warn = false` 可关闭告警（用于收敛失败属预期、
且结果会被后续处理修复的场合）。
"""
function _eigsolve(f, x₀, howmany::Integer, which::Symbol;
                   ishermitian::Bool = false, tol::Real = Defaults.tol,
                   maxiter::Int = Defaults.maxiter, krylovdim::Int = Defaults.krylovdim,
                   eager::Bool = true, warn::Bool = true, kwargs...)
    vals, vecs, info = KrylovKit.eigsolve(f, x₀, howmany, which;
                                          ishermitian = ishermitian, tol = tol,
                                          maxiter = maxiter, krylovdim = krylovdim,
                                          eager = eager, kwargs...)
    if warn && info.converged < howmany
        @warn "KrylovKit.eigsolve 未完全收敛" nconv = info.converged howmany =
              Int(howmany) normres = info.normres numiter = info.numiter tol = tol maxiter =
              maxiter krylovdim = krylovdim
    end
    return vals, vecs, info
end

# 包级默认截断方案：以 Defaults.D 封顶、Defaults.tolgauge 为相对截断阈值、
# 至少保留一个奇异值（对齐 FiniteMPSAlgorithms 的 DefaultTruncation）。
const DefaultTruncation = truncdimcutoff(D = Defaults.D, ϵ = Defaults.tolgauge, add_back = 1)

# ---------------- iteration logging ----------------

function _logiter(io::IO, name::AbstractString, iter::Int, err::Real, extra::Pair...)
    str = join(["$k = $(repr(round(v; sigdigits = 8)))" for (k, v) in extra], ", ")
    @printf(io, "%s iter %4d : ϵ = %.3e %s\n", name, iter, err, str)
    return nothing
end
