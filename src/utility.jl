const TO = TensorOperations

# scalartype 复用 TensorOperations（VectorInterface）的既有函数：`Number` 与
# `AbstractArray` 的方法已存在，这里仅 import 后为包内自定义类型扩展方法。
import TensorOperations: scalartype

_mod1(ℓ::Integer, N::Integer) = mod1(Int(ℓ), Int(N))

# ---------------- 周期数组（对标 MPSKit 的 PeriodicArray/PeriodicVector） ----------------

"""
    PeriodicVector{T}(v::Vector{T})

周期向量：越界下标按 `mod1` 循环（对标 MPSKit 的 `PeriodicVector`）。
"""
struct PeriodicVector{T} <: AbstractVector{T}
    data::Vector{T}
end
# 注意：struct 已自动生成 PeriodicVector(v::Vector{T}) 外部构造器

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

周期数组：第 1 维按 `mod1` 循环（对标 MPSKit 的 `PeriodicArray`）。
"""
struct PeriodicArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
# 注意：struct 已自动生成 PeriodicArray(a::Array{T,N}) 外部构造器

Base.size(p::PeriodicArray) = size(p.data)
Base.getindex(p::PeriodicArray, i::Integer, args...) = p.data[_mod1(i, size(p.data, 1)), args...]
Base.setindex!(p::PeriodicArray, v, i::Integer, args...) =
    (p.data[_mod1(i, size(p.data, 1)), args...] = v; p)
Base.getindex(p::PeriodicArray, i::Colon, args...) = p.data[args...]
Base.iterate(p::PeriodicArray, args...) = iterate(p.data, args...)
Base.copy(p::PeriodicArray) = PeriodicArray(copy(p.data))
_parent(p::PeriodicArray) = p.data

# ---------------- 算法抽象与默认参数 ----------------

"""
    Algorithm

所有算法参数类型的抽象父类（VUMPS、IDMRG、TDVP、VOMPS、WI、WII、LeftCanonical 等），
与 MPSKit 的 `Algorithm` 对齐。
"""
abstract type Algorithm end

# ---------------- DynamicTol（对标 MPSKit DynamicTols） ----------------

"""
    DynamicTol(alg, tol_min, tol_max, tol_factor)

迭代算法的动态容差包装器（对标 MPSKit 的 `DynamicTol`）：
每轮按当前误差 ϵ 与迭代数更新内部求解器容差

    new_tol = clamp(ϵ · tol_factor / √iter, tol_min, tol_max)

经 [`updatetol`](@ref) 作用后返回更新了 `tol` 字段的内部算法对象。
"""
struct DynamicTol{A}
    alg::A
    tol_min::Float64
    tol_max::Float64
    tol_factor::Float64
    function DynamicTol(alg::A, tol_min::Real, tol_max::Real, tol_factor::Real) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("tol_min 必须满足 0 ≤ tol_min ≤ tol_max"))
        return new{A}(alg, tol_min, tol_max, tol_factor)
    end
end
DynamicTol(alg; tol_min::Real = 1.0e-6, tol_max::Real = 1.0e-2, tol_factor::Real = 0.1) =
    DynamicTol(alg, tol_min, tol_max, tol_factor)

"未包装的算法：动态容差为空操作（对标 MPSKit）。"
updatetol(alg, iter::Integer, ϵ::Real) = alg

function updatetol(alg::DynamicTol, iter::Integer, ϵ::Real)
    iter = max(iter, one(iter))
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
    return _updatetol(alg.alg, new_tol)
end

"更新 KrylovKit Lanczos/Arnoldi 的 `tol` 字段（重建对象，等价于 MPSKit 的 Accessors.@set）。"
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

默认参数与默认算法构造器，字段与 MPSKit 的 `Defaults` 对齐。
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
# 动态容差（对标 MPSKit Defaults）
const dynamic_tols = true
const tol_min = 1.0e-14
const tol_max = 1.0e-4
const eigs_tolfactor = 1.0e-3
const gauge_tolfactor = 1.0e-6
const envs_tolfactor = 1.0e-4

_finalize(iter, state, opp, envs) = (state, envs)

# 正交化 / SVD 默认算法的桩（具体实现在模块外挂载）
function alg_orth end
function alg_svd end

"本征求解器（KrylovKit 算法对象）。`ishermitian=true` 用 Lanczos，否则 Arnoldi。"
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

"指数求解器（TDVP 局部时间演化用）。"
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

# 正交化 / SVD 默认算法（QRpos/SDD 在 tensorops 中定义，挂到 Defaults 上）
Defaults.alg_orth() = QRpos()
Defaults.alg_svd() = SDD()

# ---------------- 迭代日志 ----------------

function _logiter(io::IO, name::AbstractString, iter::Int, err::Real, extra::Pair...)
    str = join(["$k = $(repr(round(v; sigdigits = 8)))" for (k, v) in extra], ", ")
    @printf(io, "%s iter %4d : ϵ = %.3e %s\n", name, iter, err, str)
    return nothing
end
