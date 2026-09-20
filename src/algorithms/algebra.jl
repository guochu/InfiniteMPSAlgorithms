# ---------------- 状态/MPO 的迭代代数运算（朴素精确构造 + 变分压缩） ----------------
#
# MPSKit 语义基准（src/states/finitemps.jl、src/operators/mpo.jl、abstractmpo.jl）：
# - `FiniteMPS + FiniteMPS`、`FiniteMPO + FiniteMPO`、`InfiniteMPO * InfiniteMPO`：
#   朴素精确构造（键维直和 / fuse），**不做变分压缩**，且要求长度相等（check_length）；
# - `fuse_mul_mpo`：`(O1*O2)` 物理出 = O1 的 u、物理入 = O2 的 d，
#   中间物理指标 m = O1 的 d = O2 的 u，键维 = 两键维乘积；
# - 标量乘只缩放第一个张量（见 infinitempo.jl）。
#
# 本包在朴素构造之上提供迭代（变分压缩）版本：
# - `D = nothing`：朴素精确（与 MPSKit 一致）；
# - `D::Int`：朴素构造后压缩到键维 `D`，`alg` 支持
#   `VOMPS()`（DMRG 型 1-site ALS 扫描，`_overlap_sweeps`）与
#   `VUMPS()`（本征求解模板，`_vumps_sweeps`），两者不动点相同。
#
# hadamard（逐 site 物理指标外积，MPSKit 无此接口）：
# `ρ = hadamard(ψ, dag(ψ))` 即纯态密度矩阵 `|ψ⟩⟨ψ|`。

# ---- 朴素精确构造（键维直和 / fuse / zip）----

"rank-3 键直和（MPS 加法）：`A12[(al,bl), s, (ar,br)] = blockdiag(A1, A2)`。"
function _naive_sum_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    size(A1, 2) == size(A2, 2) || throw(DimensionMismatch("MPS 加法要求逐 site 物理维相等"))
    out = zeros(T, size(A1, 1) + size(A2, 1), size(A1, 2), size(A1, 3) + size(A2, 3))
    out[1:size(A1, 1), :, 1:size(A1, 3)] .= A1
    out[size(A1, 1)+1:end, :, size(A1, 3)+1:end] .= A2
    return out
end

"rank-4 键直和（MPO 加法）：`W12[(wl1,wl2), u, (wr1,wr2), d] = blockdiag(W1, W2)`。"
function _naive_sum_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    (size(W1, 2) == size(W2, 2) && size(W1, 4) == size(W2, 4)) ||
        throw(DimensionMismatch("MPO 加法要求逐 site 物理 (u,d) 相等"))
    out = zeros(T, size(W1, 1) + size(W2, 1), size(W1, 2),
                size(W1, 3) + size(W2, 3), size(W1, 4))
    out[1:size(W1, 1), :, 1:size(W1, 3), :] .= W1
    out[size(W1, 1)+1:end, :, size(W1, 3)+1:end, :] .= W2
    return out
end

"""
    _naive_mul_tensor(W1, W2) -> W12

rank-4 键 fuse（MPO 乘法，对标 MPSKit `fuse_mul_mpo`）：中间物理指标
`m = W1 的 d = W2 的 u`，键维 = 两键维乘积（无需相等）：

```julia
W12[(wl1, wl2), u, (wr1, wr2), d] = Σ_m W1[wl1, u, wr1, m] · W2[wl2, m, wr2, d]
```
"""
function _naive_mul_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    size(W1, 4) == size(W2, 2) ||
        throw(DimensionMismatch("MPO 乘法要求 W1 的物理入 (d) 与 W2 的物理出 (u) 维相等"))
    # 本包 TensorOperations 版本不支持元组复合指标，用平坦中间张量 + reshape
    # （与 apply.jl fuse 的写法一致）
    @tensor W6[wl1, wl2, u, wr1, wr2, d] :=
        W1[wl1, u, wr1, m] * W2[wl2, m, wr2, d]
    return reshape(W6, size(W1, 1) * size(W2, 1), size(W1, 2),
                   size(W1, 3) * size(W2, 3), size(W2, 4))
end

"""
    _naive_hadamard_tensor(A1, A2) -> W

rank-4 zip（hadamard 逐 site 外积）：`W[(a,c), s1, (b,e), s2] = A1[a,s1,b]·A2[c,s2,e]`，
物理指标 `u = s1`、`d = s2`（密度矩阵 `hadamard(ψ, dag(ψ))` 的 MPO 表示）。
"""
function _naive_hadamard_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    # 同上：平坦中间张量 + reshape
    @tensor W6[a, c, s1, b, e, s2] := A1[a, s1, b] * A2[c, s2, e]
    return reshape(W6, size(A1, 1) * size(A2, 1), size(A1, 2),
                   size(A1, 3) * size(A2, 3), size(A2, 2))
end

# ---- VUMPS 模板的压缩迭代 ----

"秩-1 有效哈密顿量 `ℋ = 𝕀 − |k⟩⟨k|/⟨k,k⟩`（Hermitian 半正定；:SR 最小本征向量
唯一 = k 方向，无简并）。"
function _rank1_hamiltonian(k::AbstractArray{T}) where {T}
    kn2 = dot(k, k)
    kn2 == 0 && error("秩-1 有效哈密顿量：零向量")
    return x -> x .- (dot(k, x) / kn2) .* k
end

"""
    _vumps_sweeps(ket, x0, K; tol, maxiter, verbosity, alg_eigsolve) -> x

VUMPS 模板的压缩迭代（与 [`_overlap_sweeps`](@ref) 的 VOMPS 模板收敛到同一不动点，
代码路径不同——局部更新走本征求解而非纯映射）。混合规范恒等通道下，VOMPS 的
局部精确解 `k = GL·ket.AC·GR` 既是映射输出；直接对映射 `x ↦ GL·x·GR` 做本征求解
会退化（主本征空间含任意物理指标组合），故取秩-1 有效哈密顿量
`ℋ = 𝕀 − |k⟩⟨k|/‖k‖²`，其 :SR 最小本征向量唯一为 k 方向。每轮（MPSKit 模板）：

1. `localupdate`：`k`、`ĉ = GL₊·ket.C·GR` → `fixedpoint(ℋ, x, :SR)` 解 AC 与 C
   子问题 → `regauge!` 得候选 `AL`；
2. `gauge_step!`：`gaugefix!(; order = :R)` 恢复右规范、`AC = AL·C`；
3. 环境重算 `environments(x, nothing, ket)`；
4. 收敛判据 `_galerkin_err`（切空间 Galerkin 残差）。
"""
function _vumps_sweeps(ket::MixedCanonicalMPS, x0::MixedCanonicalMPS,
                       K::Vector{<:Array{T,3}};
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       alg_eigsolve = Defaults.alg_eigsolve()) where {T}
    N = length(ket)
    x = copy(x0)
    envs = environments(x, nothing, ket)
    ϵ = _galerkin_err(nothing, ket, x, envs)
    for iter in 1:maxiter
        ϵ < tol && break
        eigs_alg = updatetol(alg_eigsolve, iter, ϵ)
        # localupdate：逐 site 解秩-1 AC 与 C 子问题（对标 MPSKit VUMPS localupdate_step!）
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = _mapAC(leftenv(envs, ℓ), nothing, rightenv(envs, ℓ), ket.AC[ℓ])
            _, AC = fixedpoint(_rank1_hamiltonian(k), x.AC[ℓ], :SR, eigs_alg)
            ĉ = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            _, C = fixedpoint(_rank1_hamiltonian(ĉ), x.C[ℓ], :SR, eigs_alg)
            ALs[ℓ] = regauge!(AC, C; alg = Defaults.alg_orth())
        end
        # gauge：恢复整体右规范（对标 MPSKit gauge_step!）
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + 收敛判据（对标 MPSKit calc_galerkin）
        envs = environments(x, nothing, ket)
        ϵ = _galerkin_err(nothing, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "VUMPS", iter, ϵ)
    end
    _global_normalize!(x)
    return x
end

# ---- 压缩装配 ----

"""
    _compress_ket(K, physdims, D, alg) -> MixedCanonicalMPS

把朴素构造的目标张量串 `K`（MPS 视图，rank-3）变分压缩到键维 `D`：
`alg::VOMPS` 走 ALS 扫描（`_overlap_sweeps`），`alg::VUMPS` 走本征求解模板
（`_vumps_sweeps`），最后 `_align_scale!` 对齐 scale/相位。
"""
function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VOMPS) where {T}
    ket = MixedCanonicalMPS(K)
    x0 = random_mps(T, physdims, D)
    x, _ = _overlap_sweeps(nothing, ket, x0, K;
                           tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return _align_scale!(x, K)
end

function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VUMPS) where {T}
    ket = MixedCanonicalMPS(K)
    x0 = random_mps(T, physdims, D)
    x = _vumps_sweeps(ket, x0, K; tol = alg.tol, maxiter = alg.maxiter,
                      verbosity = alg.verbosity, alg_eigsolve = alg.alg_eigsolve)
    return _align_scale!(x, K)
end

_compress_ket(::Vector{<:Array{T,3}}, ::AbstractVector{Int}, ::Int,
              alg::Algorithm) where {T} =
    throw(ArgumentError("代数运算压缩仅支持 VOMPS()（DMRG 型）或 VUMPS() 算法，收到 $(typeof(alg))"))

"""
    _ring_scale(x::MixedCanonicalMPS, K) -> 标量

环迹对齐标量 `c = (⟨x.AL|K⟩ / ⟨x.AL|x.AL⟩)^(1/N)`（[`_ring_overlap`](@ref) 的
N 次型主值根）。`c^N` 是目标 `K` 在 `x` 射线上的投影系数（含相位）：逐 site 乘
`c` 后，输出幅值 `c^N·tr(∏x.AL)` 精确恢复 `K` 的幅值（任意键维、gauge 鲁棒）。
供压缩 MPO 输出与 [`_align_scale!`](@ref) 使用。
"""
function _ring_scale(x::MixedCanonicalMPS, K::Vector{<:Array{T,3}}) where {T}
    N = length(x)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    return (_ring_overlap(xALs, collect(K)) / _ring_overlap(xALs, xALs))^(1 / N)
end

"代数运算结果装配（MPS）：`D = nothing` → 朴素精确；`D::Int` → 压缩到键维 D
（`D ≥ maxbond(朴素)` 时短路返回精确结果）。"
function _algebra_result(K::Vector{<:Array{T,3}}, D::Union{Nothing,Int},
                         alg::Algorithm) where {T}
    naive = MixedCanonicalMPS(K)
    (D === nothing || maxbond(naive) ≤ D) && return naive
    physdims = [size(K[ℓ], 2) for ℓ in 1:length(K)]
    return _compress_ket(K, physdims, D, alg)
end

"代数运算结果装配（MPO）：朴素 rank-4 张量串 → `D = nothing` 精确 / `D::Int` 压缩。"
function _mpo_algebra_result(K4::Vector{<:Array{T,4}}, D::Union{Nothing,Int},
                             alg::Algorithm) where {T}
    naive = InfiniteMPO(K4)
    if D === nothing || maxbond(naive) ≤ D
        return naive
    end
    N = length(K4)
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)
    x = _compress_ket(K3, [dus[ℓ] * dds[ℓ] for ℓ in 1:N], D, alg)
    # 输出取左规范张量 `AL` 乘环迹对齐标量 `c`（`c^N = ⟨x.AL|K⟩/⟨x.AL|x.AL⟩` 为
    # 目标在 `x` 射线上的投影系数）：输出幅值 `c^N·tr(∏x.AL)` 精确恢复目标幅值
    # （含相位、任意 `D`）。压缩收敛态的逐 site 相位是本征解的规范自由度
    # （MPSKit 同样不钉定本征解相位），环迹意义下幅值已对齐。不能收集 `x.AC`：
    # `AC = AL·C` 在键间插入 `C` 加权，其周期 trace 是规范依赖的 C 加权量而非算符幅值。
    c = _ring_scale(x, K3)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return InfiniteMPO([c * A for A in ALs4])
end

# ---------------- 公开 API ----------------

"""
    Base.:+(ψ₁::MixedCanonicalMPS, ψ₂::MixedCanonicalMPS; D = nothing, alg = VOMPS())

MPS 加法（对标 MPSKit `FiniteMPS + FiniteMPS` 的朴素精确构造，要求长度与逐 site
物理维相等）：朴素构造取左规范张量 `AL`——周期 trace 表示下直和的幅值
`tr(∏blockdiag(AL₁,AL₂)) = tr(∏AL₁) + tr(∏AL₂)` 精确可加（`AC` 会插入 `C`
加权、改变射线，不能用）：

- `D = nothing`：键维直和精确构造（与 MPSKit 一致，不压缩）；
- `D::Int`：朴素构造后变分压缩到键维 `D`（`alg` 支持 `VOMPS()` 与 `VUMPS()`）。
"""
function Base.:+(ψ₁::MixedCanonicalMPS, ψ₂::MixedCanonicalMPS;
                 D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
    (length(ψ₁) == length(ψ₂)) ||
        throw(DimensionMismatch("MPS 加法要求长度相等（对标 MPSKit check_length）"))
    K = [_naive_sum_tensor(ψ₁.AL[ℓ], ψ₂.AL[ℓ]) for ℓ in 1:length(ψ₁)]
    return _algebra_result(K, D, alg)
end

"""
    Base.:+(W₁::InfiniteMPO, W₂::InfiniteMPO; D = nothing, alg = VOMPS())

MPO 加法（对标 MPSKit `FiniteMPO + FiniteMPO`）：键维直和精确构造，
要求长度与逐 site 物理 `(u,d)` 相等。压缩语义同上。
"""
function Base.:+(W₁::InfiniteMPO, W₂::InfiniteMPO;
                 D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
    (length(W₁) == length(W₂)) ||
        throw(DimensionMismatch("MPO 加法要求长度相等（对标 MPSKit check_length）"))
    K4 = [_naive_sum_tensor(W₁[ℓ], W₂[ℓ]) for ℓ in 1:length(W₁)]
    return _mpo_algebra_result(K4, D, alg)
end

"""
    Base.:-(W₁::InfiniteMPO, W₂::InfiniteMPO; D = nothing, alg = VOMPS())

MPO 减法：符号经 `(-1)*W₂`（首张量缩放，MPSKit 标量乘约定）折入加法的
朴素构造，不依赖逐张量缩放。
"""
function Base.:-(W₁::InfiniteMPO, W₂::InfiniteMPO;
                 D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
    return +(W₁, -W₂; D = D, alg = alg)
end

"""
    Base.:*(W₁::InfiniteMPO, W₂::InfiniteMPO; D = nothing, alg = VOMPS())

MPO 乘法（算符复合，对标 MPSKit `fuse_mul_mpo` 朴素逐 site fuse：键维 = 两键维
乘积、中间物理指标 `m = W1 的 d = W2 的 u`，要求长度与物理维相等）。
压缩语义同上。
"""
function Base.:*(W₁::InfiniteMPO, W₂::InfiniteMPO;
                 D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
    (length(W₁) == length(W₂)) ||
        throw(DimensionMismatch("MPO 乘法要求长度相等（对标 MPSKit check_length）"))
    K4 = [_naive_mul_tensor(W₁[ℓ], W₂[ℓ]) for ℓ in 1:length(W₁)]
    return _mpo_algebra_result(K4, D, alg)
end

"""
    hadamard(ψ₁::MixedCanonicalMPS, ψ₂::MixedCanonicalMPS; D = nothing, alg = VOMPS()) -> InfiniteMPO

逐 site 物理指标外积（Hadamard 积）：`W[(a,c), s1, (b,e), s2] = ψ₁[a,s1,b]·ψ₂[c,s2,e]`，
返回键维为两态键维乘积的 `InfiniteMPO`（物理 `u = s1`、`d = s2`，要求长度与
逐 site 物理维相等）。朴素构造取左规范张量 `AL`（理由同 `+`）：算符幅值
`O[(u…),(d…)] = c₁[u…]·c₂[d…]`（两态周期 trace 波幅的外积）。
`ρ = hadamard(ψ, dag(ψ))` 即纯态密度矩阵 `|ψ⟩⟨ψ|`。
压缩语义与 `+` 相同。
"""
function hadamard(ψ₁::MixedCanonicalMPS, ψ₂::MixedCanonicalMPS;
                  D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
    (length(ψ₁) == length(ψ₂)) ||
        throw(DimensionMismatch("hadamard 要求长度相等"))
    all(size(ψ₁.AL[ℓ], 2) == size(ψ₂.AL[ℓ], 2) for ℓ in 1:length(ψ₁)) ||
        throw(DimensionMismatch("hadamard 要求逐 site 物理维相等（u = d 方阵）"))
    K4 = [_naive_hadamard_tensor(ψ₁.AL[ℓ], ψ₂.AL[ℓ]) for ℓ in 1:length(ψ₁)]
    return _mpo_algebra_result(K4, D, alg)
end

# ---------------- MixedCanonicalMPO 版本（经 InfiniteMPO 委托） ----------------

for op in (:+, :-, :*)
    @eval function Base.$op(W₁::MixedCanonicalMPO, W₂::MixedCanonicalMPO;
                            D::Union{Nothing,Int} = nothing, alg::Algorithm = VOMPS())
        return Base.$op(InfiniteMPO(W₁), InfiniteMPO(W₂); D = D, alg = alg)
    end
end
