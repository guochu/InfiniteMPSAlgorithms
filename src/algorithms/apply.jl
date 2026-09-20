# ---------------- MPO·MPS 变分施加（对标 MPSKit 的 apply / approximate + VOMPS 命名） ----------------
#
# 目标：min_x ‖W|ψ⟩ − |x‖² ⟺ 固定 ⟨x|x⟩ = 1 时 max Re⟨x|B⟩，
# 其中 B 为融合目标 MPS 张量串（W·ψ 或被压缩 MPO 的 MPS 视图）。
#
# 对标 MPSKit VOMPS 模板（src/algorithms/approximate/vomps.jl）：
# 1. 三元固定点环境 `environments(x, operator, ket)`（below = bra、above = ket）；
# 2. localupdate：局部映射（无本征求解，区别于 groundstate VUMPS）
#    `AC_new = AC_hamiltonian(ℓ)·ket.AC[ℓ]`、`C_new = C_hamiltonian(ℓ)·ket.C[ℓ]`
#    → `regauge!(AC_new, C_new)` 得候选 `AL`；
# 3. gauge：`gaugefix!(; order = :R)` 恢复右规范；
# 4. overlap = N·Re⟨x|B⟩/Re⟨x|x⟩（rank-3 复合空间环迹闭合）。

"""
    VOMPS(; tol, maxiter, verbosity)

MPO·MPS 变分压缩/施加算法参数（命名对标 MPSKit 的 `VOMPS` 家族）。
键维由初态决定（变分流形上最大化重叠，与 MPSKit 一致）。
"""
@kwdef struct VOMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
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

"统一范数归一（AC 与 C 同缩放，保持 `AC = AL·C` 与 `AR = C[ℓ-1]⁻¹·AC` 的一致性）。"
function _global_normalize!(x::MixedCanonicalMPS)
    n = norm(x)
    n == 0 && error("apply: 零范数态")
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
function _galerkin_err(operator::Union{Nothing,InfiniteMPO}, ket::MixedCanonicalMPS,
                       x::MixedCanonicalMPS, envs)
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
时逼近 ket 本身，用于 MPO 压缩）。`K` 为目标态张量串（overlap 闭合用）。
每轮：三元 fp 环境 → 逐 site 局部映射 + `regauge!` → `gaugefix!(:R)` →
环境重算 + Galerkin 残差判敛（对标 MPSKit `localupdate_step!`/`gauge_step!`/
`envs_step!`/`calc_galerkin` 流程）。
overlap = N·Re⟨x|K⟩_ring/√(⟨x|x⟩_ring·⟨K|K⟩_ring)：环迹即周期闭链波函数内积
（Cauchy–Schwarz 严格成立），故 overlap ∈ [0, N] 且对 x、K 的整体 scale 均不变，
方向一致 ⇔ overlap = N。
"""
function _overlap_sweeps(operator::Union{Nothing,InfiniteMPO}, ket::MixedCanonicalMPS,
                         x0::MixedCanonicalMPS, K::Vector{<:Array{T,3}};
                         tol::Real = 1.0e-10, maxiter::Int = 100, verbosity::Int = 0) where {T}
    N = length(ket)
    Knorm = real(_ring_overlap(K, K))
    x = copy(x0)
    envs = environments(x, operator, ket)
    ϵ = _galerkin_err(operator, ket, x, envs)
    xALs = [x.AL[ℓ] for ℓ in 1:N]
    overlap = N * real(_ring_overlap(xALs, K)) /
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
        envs = environments(x, operator, ket)
        ϵ = _galerkin_err(operator, ket, x, envs)
        # overlap 报告值：归一化保真度对整体 scale 不变
        xALs = [x.AL[ℓ] for ℓ in 1:N]
        overlap = N * real(_ring_overlap(xALs, K)) /
                  sqrt(real(_ring_overlap(xALs, xALs)) * Knorm)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ, "overlap" => overlap)
    end
    _global_normalize!(x)
    return x, overlap
end

"""
    apply(ψ₀, W, [alg::VOMPS]) -> (ψ, overlap)

变分施加 MPO：求 `x` 最小化 `‖W·ψ₀ − x‖`（固定 ⟨x|x⟩ = 1）。
W^I/W^II 时间演化即 `ψ′, _ = apply(ψ₀, make_time_mpo(H, dt, WII()), VOMPS())`。
"""
function apply(ψ₀::MixedCanonicalMPS, W::InfiniteMPO, alg::VOMPS = VOMPS())
    (length(ψ₀) % length(W) == 0) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    return approximate(ψ₀, W, ψ₀, alg)
end

"""
    approximate(ψ₀, W, ψ, [alg::VOMPS]) -> (ψ₀′, overlap)

变分逼近 `W|ψ⟩`（MPSKit `approximate` 语义）：求 `ψ₀′` 最小化 `‖W·ψ − ψ₀′‖`。
要求 `length(ψ₀) == length(ψ)` 且为 `length(W)` 的整数倍。
"""
function approximate(ψ₀::MixedCanonicalMPS, W::InfiniteMPO, ψ::MixedCanonicalMPS,
                     alg::VOMPS = VOMPS())
    (length(ψ) == length(ψ₀) && length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    N = length(ψ)
    K = [fuse(W[_mod1(ℓ, length(W))], ψ.AL[ℓ]) for ℓ in 1:N]
    x, overlap = _overlap_sweeps(W, ψ, ψ₀, K;
                                 tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
    return x, overlap
end
