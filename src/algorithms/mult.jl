# ---------------- MPO 迭代乘法 mult（MPO·MPS 与 MPO·MPO 的变分施加/压缩） ----------------
#
# 目标：给定 (W, x)，求 y ≈ W·x（x 为 MPS：态施加；x 为 MPO：算符复合）。
# 朴素精确构造见 arithmetics.jl 的 exact_mult；本文件提供迭代（变分）版本：
# - `VOMPS`：重叠最大化 ALS 扫描（对标 MPSKit VOMPS，src/algorithms/approximate/vomps.jl）；
# - `IDMRG`：秩-1 有效哈密顿量本征求解扫描（与 VOMPS 收敛到同一不动点）。
#
# VOMPS 模板（对标 MPSKit）：
# 1. 三元固定点环境 `MultCache(x, operator, ket)`（below = bra、above = ket）；
# 2. localupdate：局部映射（无本征求解，区别于 groundstate VUMPS）
#    `AC_new = AC_hamiltonian(ℓ)·ket.AC[ℓ]`、`C_new = C_hamiltonian(ℓ)·ket.C[ℓ]`
#    → `regauge!(AC_new, C_new)` 得候选 `AL`；
# 3. gauge：`gaugefix!(; order = :R)` 恢复右规范；
# 4. overlap = N·Re⟨x|B⟩/Re⟨x|x⟩（rank-3 复合空间环迹闭合）。

# ---------------- MPO 施加通道环境缓存（MultCache） ----------------

"""
    MultCache(operator, bra, ket, lefts, rights)
    MultCache(below, operator::InfiniteMPO, above; tol, krylovdim, maxiter) -> MultCache

MPO 施加通道环境：`⟨below|operator|above⟩`（`operator::InfiniteMPO`）的左右固定点，
供 `mult`（迭代 MPO 乘法）使用，below（bra）与 above（ket）可为不同态。

- `lefts[ℓ]`：site ℓ 左环境 `(below键, w, above键)`；
- `rights[ℓ]`：site ℓ 右环境 `(above键, w, below键)`（MPSKit 约定）；
- 固定点由复合转移矩阵 `T(above.AL, operator, below.AL)` 的 :LM 本征对求解
  （eigsolve）；
- 归一化对标 MPSKit `normalize!(::InfiniteEnvironments)`：每个 GR 先 Frobenius
  归一，再逐 site 用 `λℓ = ⟨below.C[ℓ], C_map(ℓ)⟩` 缩放 `GLs[ℓ+1]`，
  使每个 site 的局部收缩恰为 1（恒等 MPO 期望 = N）。
"""
struct MultCache{O<:InfiniteMPO,B<:InfiniteCanonicalMPS,K<:InfiniteCanonicalMPS,T} <: Environments
    operator::O
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

function MultCache(below::InfiniteCanonicalMPS, operator::InfiniteMPO,
                   above::InfiniteCanonicalMPS;
                   tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200)
    GLs, GRs = _ternary_fixedpoints(below, operator, above; tol, krylovdim, maxiter)
    return MultCache(operator, below, above, GLs, GRs)
end

# ---------------- 纯重叠通道环境缓存（OverlapCache） ----------------

"""
    OverlapCache(bra, ket, lefts, rights)
    OverlapCache(ψ) -> OverlapCache
    OverlapCache(below, above; tol, krylovdim, maxiter) -> OverlapCache

纯重叠通道环境：`⟨bra|ket⟩` 的左右固定点（恒等通道，w 维为 1），
供 `mult` 的 IDMRG 压缩路径与 `add`/`hadamard` 的迭代压缩等使用。

- `OverlapCache(ψ)`：恒等通道直接用 AL/AR 规范（固定点 = 恒等矩阵）；
- `OverlapCache(below, above)`：below 与 above 可为不同态，固定点由复合转移矩阵
  `T(above.AL, below.AL)` 的 :LM 本征对求解（eigsolve），归一化与
  [`MultCache`](@ref) 相同（MPSKit 约定）。
- `lefts[ℓ]`：site ℓ 左环境 `(bra键, 1, ket键)`；
- `rights[ℓ]`：site ℓ 右环境 `(ket键, 1, bra键)`。
"""
struct OverlapCache{B<:InfiniteCanonicalMPS,K<:InfiniteCanonicalMPS,T} <: Environments
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"恒等通道：AL/AR 规范下固定点 = 恒等矩阵。"
function OverlapCache(ψ::InfiniteCanonicalMPS; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Array{T,3}}(undef, N)
    rights = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        Dl, Dr = size(ψ.AL[ℓ], 1), size(ψ.AL[ℓ], 3)
        lefts[ℓ] = _to3(Matrix{T}(I, Dl, Dl))
        rights[ℓ] = _to3(Matrix{T}(I, Dr, Dr))
    end
    return OverlapCache(ψ, ψ, lefts, rights)
end

"三元重叠通道：⟨below|above⟩ 的左右固定点。"
function OverlapCache(below::InfiniteCanonicalMPS, above::InfiniteCanonicalMPS;
                      tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200)
    GLs, GRs = _ternary_fixedpoints(below, nothing, above; tol, krylovdim, maxiter)
    return OverlapCache(below, above, GLs, GRs)
end

"""
    fuse(W, AL) -> Array{T,3}

MPO 张量与 MPS 左正交张量的逐 site 融合：
`B[(wl·bl), u, (wr·br)] = Σ_d W[wl, u, wr, d] · AL[bl, d, br]`。
"""
function fuse(W::AbstractArray{T,4}, AL::AbstractArray{T,3}) where {T}
    wl, u, wr, _ = size(W)
    bl, _, br = size(AL)
    @tensor B5[wl, bl, u, wr, br] := W[wl, u, wr, d] * AL[bl, d, br]
    return reshape(B5, wl * bl, u, wr * br)
end

"""
    _naive_mul_tensor(W1, W2) -> W12

rank-4 键 fuse（MPO 乘法核，对标 MPSKit `fuse_mul_mpo`）：中间物理指标
`m = W1 的 d = W2 的 u`，键维 = 两键维乘积（无需相等）：

```julia
W12[(wl1, wl2), u, (wr1, wr2), d] = Σ_m W1[wl1, u, wr1, m] · W2[wl2, m, wr2, d]
```
"""
function _naive_mul_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    size(W1, 4) == size(W2, 2) ||
        throw(DimensionMismatch("MPO 乘法要求 W1 的物理入 (d) 与 W2 的物理出 (u) 维相等"))
    # 本包 TensorOperations 版本不支持元组复合指标，用平坦中间张量 + reshape
    # （与 fuse 的写法一致）
    @tensor W6[wl1, wl2, u, wr1, wr2, d] :=
        W1[wl1, u, wr1, m] * W2[wl2, m, wr2, d]
    return reshape(W6, size(W1, 1) * size(W2, 1), size(W1, 2),
                   size(W1, 3) * size(W2, 3), size(W2, 4))
end

"VOMPS 局部 AC 映射（对标 MPSKit `AC_hamiltonian·ket.AC`）：AC_new = (GL·O·GR)·ket.AC。"
function _mapAC(GL::AbstractArray{T,3}, O::Union{Nothing,AbstractArray{T,4}},
                GR::AbstractArray{T,3}, ketAC::AbstractArray{T,3}) where {T}
    if O === nothing
        @tensor ACnew[aL, p, aR] := GL[aL, 1, bL] * ketAC[bL, p, bR] * GR[bR, 1, aR]
    else
        @tensor ACnew[aL, u, aR] := GL[aL, w, bL] * ketAC[bL, s, bR] * O[w, u, w′, s] * GR[bR, w′, aR]
    end
    return ACnew
end

"VOMPS 局部 C 映射（对标 MPSKit `C_hamiltonian·ket.C`）：通道贯穿、无 W 收缩。"
function _mapC(GL::AbstractArray{T,3}, GR::AbstractArray{T,3}, ketC::AbstractMatrix{T}) where {T}
    @tensor Cnew[a, a′] := GL[a, w, b] * ketC[b, b′] * GR[b′, w, a′]
    return Cnew
end

"""
    _ring_overlap(ALs, K) -> 复数

⟨x|K⟩ 的周期环迹：复合空间 `(x键, K键)` 上
`𝔈_ℓ[(a′,β′),(a,β)] = Σ_p conj(x.AL[ℓ][a,p,a′])·K[ℓ][β,p,β′]`，
从 `E = I` 出发 `E ← 𝔈_ℓ·E` 传播，返回 `tr(∏𝔈_ℓ)`（x 键环与 K 键环独立闭合）。
"""
function _ring_overlap(ALs::Vector{<:AbstractArray{T,3}}, K::Vector{<:AbstractArray{T,3}}) where {T}
    N = length(K)
    Dx, DK = size(ALs[1], 1), size(K[1], 1)
    E = zeros(T, Dx, DK, Dx, DK)
    for a in 1:Dx, β in 1:DK
        E[a, β, a, β] = one(T)
    end
    for ℓ in 1:N
        A = ALs[ℓ]
        B = K[ℓ]
        E = @tensor E′[a′, β′, a₀, β₀] := conj(A[a, p, a′]) * B[β, p, β′] * E[a, β, a₀, β₀]
    end
    return tr(reshape(E, Dx * DK, Dx * DK))
end

"""
    _ring_scale(x::InfiniteCanonicalMPS, K) -> 标量

环迹对齐标量 `c = (⟨x.AL|K⟩ / ⟨x.AL|x.AL⟩)^(1/N)`（[`_ring_overlap`](@ref) 的
N 次型主值根）。`c^N` 是目标 `K` 在 `x` 射线上的投影系数（含相位）：逐 site 乘
`c` 后，输出幅值 `c^N·tr(∏x.AL)` 精确恢复 `K` 的幅值（任意键维、gauge 鲁棒）。
供 `mult` 的 mpo*mpo 输出、`_mpo_algebra_result` 与 `mpo_compress` 使用。
"""
function _ring_scale(x::InfiniteCanonicalMPS, K::Vector{<:Array{T,3}}) where {T}
    N = length(x)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    return (_ring_overlap(xALs, collect(K)) / _ring_overlap(xALs, xALs))^(1 / N)
end

"统一范数归一（AC 与 C 同缩放，保持 `AC = AL·C` 与 `AR = C[ℓ-1]⁻¹·AC` 的一致性）。"
function _global_normalize!(x::InfiniteCanonicalMPS)
    n = norm(x)
    n == 0 && error("mult: 零范数态")
    for ℓ in 1:length(x)
        x.AC[ℓ] .= x.AC[ℓ] ./ n
        x.C[ℓ] .= x.C[ℓ] ./ n
    end
    return x
end

"VOMPS Galerkin 残差（对标 MPSKit `calc_galerkin`）：`normalize(AC_map)` 垂直于
`AL` 切空间的分量范数——overlap 对切向漂移不敏感，判敛必须用残差而非 Δoverlap。"
function _galerkin(AL::AbstractArray{T,3}, ACnew::AbstractArray{T,3}) where {T}
    ACn = normalize!(copy(ACnew))
    @tensor proj[b, b′] := conj(AL[a, s, b]) * ACn[a, s, b′]
    @tensor out[a, s, b′] := ACn[a, s, b′] - AL[a, s, b] * proj[b, b′]
    return norm(out)
end

"逐 site 的最大 Galerkin 残差（对标 MPSKit `calc_galerkin(below, operator, above, envs)`，
用当前状态 `x.AL` 投影当前环境下的局部映射输出）。"
function _galerkin_err(operator::Union{Nothing,InfiniteMPO}, ket::InfiniteCanonicalMPS,
                       x::InfiniteCanonicalMPS, envs)
    N = length(ket)
    ϵ = 0.0
    for ℓ in 1:N
        O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
        ACmap = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], ACmap))
    end
    return ϵ
end

"""
    _overlap_sweeps(operator, ket, x0, K; tol, maxiter, verbosity) -> (x, overlap)

重叠最大化变分扫描（MPSKit VOMPS 模板）：求 `x` 逼近 `operator|ket⟩`（`operator = nothing`
时逼近 ket 本身，用于变分压缩）。`K` 为目标态张量串（overlap 闭合用）。
每轮：三元 fp 环境 → 逐 site 局部映射 + `regauge!` → `gaugefix!(:R)` →
环境重算 + Galerkin 残差判敛（对标 MPSKit `localupdate_step!`/`gauge_step!`/
`envs_step!`/`calc_galerkin` 流程）。
overlap = N·Re⟨x|K⟩_ring/√(⟨x|x⟩_ring·⟨K|K⟩_ring)：环迹即周期闭链波函数内积
（Cauchy–Schwarz 严格成立），故 overlap ∈ [0, N] 且对 x、K 的整体 scale 均不变，
方向一致 ⇔ overlap = N。
"""
function _overlap_sweeps(operator::Union{Nothing,InfiniteMPO}, ket::InfiniteCanonicalMPS,
                         x0::InfiniteCanonicalMPS, K::Vector{<:Array{T,3}};
                         tol::Real = 1.0e-10, maxiter::Int = 100, verbosity::Int = 0) where {T}
    N = length(ket)
    Knorm = real(_ring_overlap(K, K))
    x = copy(x0)
    envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
    ϵ = _galerkin_err(operator, ket, x, envs)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    overlap = N * abs(_ring_overlap(xALs, K)) /
              sqrt(real(_ring_overlap(xALs, xALs)) * Knorm)
    for iter in 1:maxiter
        ϵ < tol && break
        # localupdate：逐 site 局部映射 + regauge（对标 MPSKit VOMPS 局部步）
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            O = isnothing(operator) ? nothing : operator[_mod1(ℓ, length(operator))]
            AC_new = _mapAC(leftenv(envs, ℓ), O, rightenv(envs, ℓ), ket.AC[ℓ])
            C_new = _mapC(leftenv(envs, _mod1(ℓ + 1, N)), rightenv(envs, ℓ), ket.C[ℓ])
            ALs[ℓ] = regauge!(AC_new, C_new; alg = Defaults.alg_orth())
        end
        # gauge：恢复整体右规范（对标 MPSKit gauge_step!：播种 state.C[end]）
        gauge_step!(x, ALs, x.C[N]; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
        # envs_step! + calc_galerkin：新状态、新环境下的切空间残差
        envs = isnothing(operator) ? OverlapCache(x, ket) : MultCache(x, operator, ket)
        ϵ = _galerkin_err(operator, ket, x, envs)
        # overlap 报告值：归一化保真度对整体 scale（含相位）不变
        xALs = [x.AL[ℓ] for ℓ in 1:N]
        overlap = N * abs(_ring_overlap(xALs, K)) /
                  sqrt(real(_ring_overlap(xALs, xALs)) * Knorm)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ, "overlap" => overlap)
    end
    _global_normalize!(x)
    return x, overlap
end

# ---------------- IDMRG 模板的压缩迭代 ----------------

"秩-1 有效哈密顿量 `ℋ = 𝕀 − |k⟩⟨k|/⟨k,k⟩`（Hermitian 半正定；:SR 最小本征向量
唯一 = k 方向，无简并）。"
function _rank1_hamiltonian(k::AbstractArray{T}) where {T}
    kn2 = dot(k, k)
    kn2 == 0 && error("秩-1 有效哈密顿量：零向量")
    return x -> x .- (dot(k, x) / kn2) .* k
end

"""
    _idmrg_sweeps(ket, x0, K; tol, maxiter, verbosity, alg_eigsolve) -> (x, overlap)

IDMRG 模板的压缩迭代（与 [`_overlap_sweeps`](@ref) 的 VOMPS 模板收敛到同一不动点，
代码路径不同——局部更新走本征求解而非纯映射）。混合规范恒等通道下，VOMPS 的
局部精确解 `k = GL·ket.AC·GR` 既是映射输出；直接对映射 `x ↦ GL·x·GR` 做本征求解
会退化（主本征空间含任意物理指标组合），故取秩-1 有效哈密顿量
`ℋ = 𝕀 − |k⟩⟨k|/‖k‖²`，其 :SR 最小本征向量唯一为 k 方向。每轮（MPSKit 模板）：

1. `localupdate`：`k`、`ĉ = GL₊·ket.C·GR` → `fixedpoint(ℋ, x, :SR)` 解 AC 与 C
   子问题 → `regauge!` 得候选 `AL`；
2. `gauge_step!`：`gaugefix!(; order = :R)` 恢复右规范、`AC = AL·C`；
3. 环境重算 `OverlapCache(x, ket)`；
4. 收敛判据 `_galerkin_err`（切空间 Galerkin 残差）。
"""
function _idmrg_sweeps(ket::InfiniteCanonicalMPS, x0::InfiniteCanonicalMPS,
                       K::Vector{<:Array{T,3}};
                       tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                       verbosity::Int = Defaults.verbosity,
                       alg_eigsolve = Defaults.alg_eigsolve()) where {T}
    N = length(ket)
    Knorm = real(_ring_overlap(K, K))
    x = copy(x0)
    envs = OverlapCache(x, ket)
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
        envs = OverlapCache(x, ket)
        ϵ = _galerkin_err(nothing, ket, x, envs)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
    end
    _global_normalize!(x)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    overlap = N * abs(_ring_overlap(xALs, K)) /
              sqrt(real(_ring_overlap(xALs, xALs)) * Knorm)
    return x, overlap
end

# ---------------- 公开接口 mult ----------------

"""
    mult(W, ψ; [ψ₀], alg = VOMPS()) -> (y::InfiniteCanonicalMPS, overlap)
    mult(W, W2; [ψ₀], alg = VOMPS()) -> (y::InfiniteCanonicalMPO, overlap)

MPO 乘法的迭代（变分）版本：求 `y ≈ W·ψ`（态施加）或 `y ≈ W·W2`（算符复合）。
朴素精确构造见 [`exact_mult`](@ref)；时间演化 MPO 的施加即
`ψ′, _ = mult(make_time_mpo(H, dt, WII()), ψ)`。

- `W`：`InfiniteMPO`、`MPOHamiltonian`（稠密化为 InfiniteMPO 后施加）或
  `InfiniteCanonicalMPO`；
- `alg`：[`VOMPS`](@ref)（重叠最大化 ALS 扫描）或 [`IDMRG`](@ref)
  （秩-1 有效哈密顿量本征求解），两者不动点相同；输出键维
  `D = bondD(alg.trunc)`；
- `ψ₀`：初态（缺省时以键维 `D` 的随机态出发；`D ≥ 朴素构造键维` 时短路，
  直接返回精确施加/复合的规范存储）；
- 输出保证为混合规范形式（mpo·mps → `InfiniteCanonicalMPS`，mpo·mpo →
  `InfiniteCanonicalMPO`；算符输出的幅值遵循 `InfiniteCanonicalMPO` 的射线
  规范约定，保真度由 `overlap` 给出，overlap ∈ [0, N]，= N 即方向一致）。
"""
function mult(W, ψ::InfiniteCanonicalMPS; ψ₀ = nothing, alg::Union{VOMPS,IDMRG} = VOMPS())
    Wm = W isa InfiniteMPO ? W : InfiniteMPO(W)
    (length(ψ) % length(Wm) == 0) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    N = length(ψ)
    T = promote_type(scalartype(Wm), scalartype(ψ))
    K = [fuse(Wm[ℓ], ψ.AL[ℓ]) for ℓ in 1:N]        # 朴素目标射线（W·ψ 的 fuse）
    D = bondD(alg.trunc)
    if isnothing(ψ₀) && all(size(K[ℓ], 1) ≤ D for ℓ in eachindex(K))
        return InfiniteCanonicalMPS(collect(K)), float(N)   # 精确施加短路（overlap = N）
    end
    ket = InfiniteCanonicalMPS(K)                   # 目标的规范存储（迭代用）
    x0 = isnothing(ψ₀) ? randommps(T, phydims(ψ), D) : ψ₀
    y, overlap = _mult_compress(alg, ket, x0, K)
    # 保证混合规范：从 AL + C[end] 重新右规范化（保持射线）
    y = InfiniteCanonicalMPS(collect(y.AL), y.C[end])
    return y, overlap
end

function mult(W, W2::Union{InfiniteMPO,InfiniteCanonicalMPO}; ψ₀ = nothing,
              alg::Union{VOMPS,IDMRG} = VOMPS())
    Wm = W isa InfiniteMPO ? W : InfiniteMPO(W)
    W2m = W2 isa InfiniteMPO ? W2 : InfiniteMPO(W2)
    (length(W2m) % length(Wm) == 0) ||
        throw(DimensionMismatch("MPO 单胞长度不兼容"))
    N = length(W2m)
    T = promote_type(scalartype(Wm), scalartype(W2m))
    K4 = [_naive_mul_tensor(Wm[_mod1(ℓ, length(Wm))], W2m[ℓ]) for ℓ in 1:N]
    dus = [size(K4[ℓ], 2) for ℓ in 1:N]
    dds = [size(K4[ℓ], 4) for ℓ in 1:N]
    K3 = asmps_view(K4)                             # MPO 乘积的 MPS 视图目标
    D = bondD(alg.trunc)
    if isnothing(ψ₀) && all(size(K3[ℓ], 1) ≤ D for ℓ in eachindex(K3))
        # 精确复合短路：朴素乘积的规范存储（overlap = N）
        y = _mpo_from_mps(_align_scale!(InfiniteCanonicalMPS(K3), K3), dus, dds)
        return y, float(N)
    end
    ket = InfiniteCanonicalMPS(K3)
    x0 = isnothing(ψ₀) ? randommps(T, [size(K3[ℓ], 2) for ℓ in 1:N], D) : ψ₀
    x, overlap = _mult_compress(alg, ket, x0, K3)
    x = _align_scale!(x, K3)                        # 幅值/相位对齐到目标射线
    y = _mpo_from_mps(x, dus, dds)                  # → InfiniteCanonicalMPO（重新规范）
    return y, overlap
end

"mult 的压缩引擎分派：`VOMPS` → 重叠最大化扫描，`IDMRG` → 本征求解扫描。"
function _mult_compress(alg::VOMPS, ket, x0, K)
    return _overlap_sweeps(nothing, ket, x0, K;
                           tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
end
function _mult_compress(alg::IDMRG, ket, x0, K)
    return _idmrg_sweeps(ket, x0, K;
                         tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity,
                         alg_eigsolve = alg.alg_eigsolve)
end
