# ---------------- 迭代加法 add（mps+mps / mpo+mpo 的变分压缩） ----------------
#
# 朴素精确构造核见 arithmetics.jl（exact_add / _naive_sum_tensor）；本文件在
# 其上提供迭代（变分压缩）版本，与 mult.jl 的 mult 共用同一压缩引擎
# （_overlap_sweeps / _idmrg_sweeps）：
# - `D = nothing`：朴素精确（与 MPSKit 一致，不压缩）；
# - `D::Int`：朴素构造后压缩到键维 `D`。
#
# 本文件同时存放 add / hadamard 共用的变分压缩装配
# （_compress_ket / _algebra_result / _mpo_algebra_result）。

# ---- add / hadamard 共用的变分压缩装配 ----

"""
    _compress_ket(K, physdims, D, alg) -> InfiniteCanonicalMPS

把朴素构造的目标张量串 `K`（MPS 视图，rank-3）变分压缩到键维 `D`：
`alg::VOMPS` 走 ALS 扫描（`_overlap_sweeps`），`alg::IDMRG` 走本征求解模板
（`_idmrg_sweeps`），最后 `_align_scale!` 对齐 scale/相位。
"""
function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VOMPS) where {T}
    ket = InfiniteCanonicalMPS(K)
    x0 = randomimps(T, physdims, D)
    x, _ = _overlap_sweeps(nothing, ket, x0, K;
                           tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return _align_scale!(x, K)
end

function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::IDMRG) where {T}
    ket = InfiniteCanonicalMPS(K)
    x0 = randomimps(T, physdims, D)
    x, _ = _idmrg_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                         verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    return _align_scale!(x, K)
end

_compress_ket(::Vector{<:Array{T,3}}, ::AbstractVector{Int}, ::Int,
              alg::Algorithm) where {T} =
    throw(ArgumentError("代数运算压缩仅支持 VOMPS()（DMRG 型）或 IDMRG() 算法，收到 $(typeof(alg))"))

"add/hadamard 的结果装配（MPS）：`D = nothing` → 朴素精确；`D::Int` → 压缩到键维 D
（`D ≥ max_bonddim(朴素)` 时短路返回精确结果）。"
function _algebra_result(K::Vector{<:Array{T,3}}, D::Union{Nothing,Int},
                         alg::Algorithm) where {T}
    naive = InfiniteCanonicalMPS(K)
    (D === nothing || max_bonddim(naive) ≤ D) && return naive
    physdims = [size(K[ℓ], 2) for ℓ in 1:length(K)]
    return _compress_ket(K, physdims, D, alg)
end

"add/hadamard 的结果装配（MPO）：朴素 rank-4 张量串 → `D = nothing` 精确 /
`D::Int` 压缩。输出取环迹对齐标量 `c` 乘左规范张量 `AL`：输出幅值 `c^N·tr(∏AL)`
精确恢复目标幅值（含相位、任意 `D`）。压缩收敛态的逐 site 相位是本征解的规范
自由度（MPSKit 同样不钉定本征解相位），环迹意义下幅值已对齐。不能收集 `x.AC`：
`AC = AL·C` 在键间插入 `C` 加权，其周期 trace 是规范依赖的 C 加权量而非算符幅值。"
function _mpo_algebra_result(K4::Vector{<:Array{T,4}}, D::Union{Nothing,Int},
                             alg::Algorithm) where {T}
    naive = InfiniteMPO(K4)
    if D === nothing || max_bonddim(naive) ≤ D
        return naive
    end
    N = length(K4)
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x = _compress_ket(K3, [dus[ℓ] * dds[ℓ] for ℓ in 1:N], D, alg)
    c = _ring_scale(x, K3)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return InfiniteMPO([c * A for A in ALs4])
end

# ---------------- 公开接口 add ----------------

"""
    add(ψ₁, ψ₂; D = nothing, alg = VOMPS()) -> InfiniteCanonicalMPS
    add(W₁, W₂; D = nothing, alg = VOMPS()) -> InfiniteMPO

迭代加法：mps+mps（→ `InfiniteCanonicalMPS`）与 mpo+mpo（→ `InfiniteMPO`）。
朴素精确构造对标 MPSKit `FiniteMPS + FiniteMPS`（要求长度与逐 site 物理维相等）：
朴素构造取左规范张量 `AL`——周期 trace 表示下直和的幅值
`tr(∏blockdiag(AL₁,AL₂)) = tr(∏AL₁) + tr(∏AL₂)` 精确可加（`AC` 会插入 `C`
加权、改变射线，不能用）：

- `D = nothing`：键维直和精确构造（与 MPSKit 一致，不压缩）；
- `D::Int`：朴素构造后变分压缩到键维 `D`（`alg` 支持 `VOMPS()` 与 `IDMRG()`）。
"""
function add(ψ1::InfiniteCanonicalMPS, ψ2::InfiniteCanonicalMPS;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("MPS 加法要求长度相等（对标 MPSKit check_length）"))
    K = [_naive_sum_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:length(ψ1)]
    return _algebra_result(K, D, alg)
end

function add(W1::InfiniteCanonicalMPO, W2::InfiniteCanonicalMPO;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO 加法要求长度相等（对标 MPSKit check_length）"))
    K4 = [_naive_sum_tensor(W1.AL[ℓ], W2.AL[ℓ]) for ℓ in 1:length(W1)]
    return _mpo_algebra_result(K4, D, alg)
end

function add(W1::InfiniteMPO, W2::InfiniteMPO;
             D::Union{Nothing,Int} = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO 加法要求长度相等（对标 MPSKit check_length）"))
    K4 = [_naive_sum_tensor(W1[ℓ], W2[ℓ]) for ℓ in 1:length(W1)]
    return _mpo_algebra_result(K4, D, alg)
end
