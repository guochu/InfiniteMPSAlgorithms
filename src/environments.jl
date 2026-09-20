# ---------------- 环境缓存（对标 MPSKit src/environments/infinite_envs.jl） ----------------
#
# 按用途拆分为三种具体缓存（抽象父类型 `Environments`）：
# - `OverlapCache`：纯重叠通道 ⟨bra|ket⟩（`environments(ψ)`、三元
#   `environments(below, nothing, above)`）；
# - `MultCache`：MPO 施加通道 ⟨bra|W|ket⟩（三元 `environments(below, W, above)`，
#   供变分施加 MPO / mpo_compress）；
# - `DMRGCache`：哈密顿量通道（`environments(ψ, H)`，H 可为 `MPOHamiltonian`
#   或 `InfiniteMPO`），供基态搜索（VUMPS / IDMRG / TDVP）与能量计算。

"""
    Environments

⟨below|operator|ket⟩ 型双层网络左右固定点环境的抽象父类型（对标 MPSKit 的
InfiniteEnvironments）；具体类型：[`OverlapCache`](@ref)（纯重叠）、
[`MultCache`](@ref)（MPO 施加）与 [`DMRGCache`](@ref)（哈密顿量）。
"""
abstract type Environments end

"""
    OverlapCache(bra, ket, lefts, rights)

纯重叠通道环境：`⟨bra|ket⟩` 的左右固定点（恒等通道，w 维为 1）。

- `lefts[ℓ]`：site ℓ 左环境 `(bra键, 1, ket键)`；
- `rights[ℓ]`：site ℓ 右环境 `(ket键, 1, bra键)`（MPSKit 约定）。
"""
struct OverlapCache{B<:MixedCanonicalMPS,K<:MixedCanonicalMPS,T} <: Environments
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"""
    MultCache(operator, bra, ket, lefts, rights)

MPO 施加通道环境：`⟨bra|operator|ket⟩`（`operator::InfiniteMPO`）的左右固定点，
供变分施加 MPO（`apply`）与 `mpo_compress` 使用。

- `lefts[ℓ]`：site ℓ 左环境 `(bra键, w, ket键)`；
- `rights[ℓ]`：site ℓ 右环境 `(ket键, w, bra键)`（MPSKit 约定）。
"""
struct MultCache{O<:InfiniteMPO,B<:MixedCanonicalMPS,K<:MixedCanonicalMPS,T} <: Environments
    operator::O
    bra::B
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"""
    DMRGCache(operator, ket, lefts, rights)

哈密顿量通道环境缓存（基态搜索 VUMPS / IDMRG / TDVP 与能量计算）：
`operator::Union{MPOHamiltonian,InfiniteMPO}`——`MPOHamiltonian` 的 w 维 =
Jordan 层数（逐 level 求解）；`InfiniteMPO` 为稠密 MPO 哈密顿量
（转移矩阵主本征向量）。
"""
struct DMRGCache{H<:Union{MPOHamiltonian,InfiniteMPO},K<:MixedCanonicalMPS,T} <: Environments
    operator::H
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"leftenv(envs, ℓ)：site ℓ 左环境。"
leftenv(envs::Environments, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs.ket))]
"rightenv(envs, ℓ)：site ℓ 右环境。"
rightenv(envs::Environments, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs.ket))]

_to3(L::AbstractMatrix{T}) where {T} = reshape(L, size(L, 1), 1, size(L, 2))
_to3(L::AbstractArray{T,3}) where {T} = L

"""
    environments(ψ::MixedCanonicalMPS) -> OverlapCache
    environments(ψ, operator) -> Union{OverlapCache,DMRGCache}

- `operator = nothing`：恒等通道直接用 AL/AR 规范（固定点 = 恒等矩阵），
  返回 [`OverlapCache`](@ref)（⟨ψ|ψ⟩ 纯重叠）；
- `operator::InfiniteMPO`：稠密 MPO 哈密顿量通道，转移矩阵主本征向量
  （Krylov Arnoldi），返回 [`DMRGCache`](@ref)（InfiniteMPO 可直接作为
  基态搜索的哈密顿量，也用于 `expectation_value(ψ, W)`）；
- `operator::MPOHamiltonian`：Jordan 结构逐层求解（对标 MPSKit）——见
  [`environments(ψ, H::MPOHamiltonian)`](@ref)，返回 [`DMRGCache`](@ref)。
"""
function environments(ψ::MixedCanonicalMPS, operator::Nothing = nothing; kwargs...)
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

function environments(ψ::MixedCanonicalMPS, operator::InfiniteMPO; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Array{T,3}}(undef, N)
    rights = Vector{Array{T,3}}(undef, N)
    _, L0 = dominant_env(operator, ψ; side = :left, kwargs...)
    _, R0 = dominant_env(operator, ψ; side = :right, kwargs...)
    lefts[1] = _to3(L0)
    for ℓ in 2:N
        lefts[ℓ] = push_env_left(lefts[ℓ-1], operator[ℓ-1], ψ.AL[ℓ-1])
    end
    rights[N] = _to3(R0)
    for ℓ in N-1:-1:1
        rights[ℓ] = push_env_right(rights[ℓ+1], operator[ℓ+1], ψ.AR[ℓ+1])
    end
    # 对标 MPSKit normalize!(::InfiniteMPO)：先把每个 GR 做 Frobenius 归一，
    # 再逐 site 用配分函数 λ_i = ⟨AC_i | GL_{i+1}·W_i·GR_i | AC_i⟩ 缩放 GL_{i+1}，
    # 使每个 site 的局部收缩恰好为 1（恒等 MPO 期望 = N）。
    for ℓ in 1:N
        rights[ℓ] .= rights[ℓ] ./ norm(rights[ℓ])
    end
    for i in 1:N
        inext = i == N ? 1 : i + 1
        λi = contract_mpo_expval(ψ.AC[i], lefts[inext], operator[i], rights[i])
        lefts[inext] .= lefts[inext] ./ λi
    end
    return DMRGCache(operator, ψ, lefts, rights)
end

"""
    environments(below, operator::Nothing, above) -> OverlapCache
    environments(below, operator::InfiniteMPO, above) -> MultCache

三元环境（对标 MPSKit `environments(below, operator, above)`）：求
⟨below|operator|above⟩ 网络的左右固定点，below（bra）与 above（ket）可为不同态。
`operator = nothing` 为纯重叠通道（[`OverlapCache`](@ref)）；`operator::InfiniteMPO`
为 MPO 施加通道（[`MultCache`](@ref)）。

- `lefts[ℓ]::(below左键, w, above左键)`、`rights[ℓ]::(above右键, w, below右键)`；
- 固定点由复合转移矩阵 `T(above.AL, operator, below.AL)` 的 :LM 本征对求解
  （eigsolve；恒等通道即 `T(above.AL, below.AL)`）；
- 归一化对标 MPSKit `normalize!(::InfiniteEnvironments)`：每个 GR 先 Frobenius
  归一，再逐 site 用 `λℓ = ⟨below.C[ℓ], C_map(ℓ)⟩` 缩放 `GLs[ℓ+1]`，
  使每个 site 的局部收缩恰为 1（恒等 MPO 期望 = N）。
"""
function environments(below::MixedCanonicalMPS, operator::Nothing,
                      above::MixedCanonicalMPS;
                      tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200)
    GLs, GRs = _ternary_fixedpoints(below, nothing, above; tol, krylovdim, maxiter)
    return OverlapCache(below, above, GLs, GRs)
end

function environments(below::MixedCanonicalMPS, operator::InfiniteMPO,
                      above::MixedCanonicalMPS;
                      tol::Real = 1.0e-12, krylovdim::Int = 12, maxiter::Int = 200)
    GLs, GRs = _ternary_fixedpoints(below, operator, above; tol, krylovdim, maxiter)
    return MultCache(operator, below, above, GLs, GRs)
end

"三元环境共享的固定点求解（左右主本征向量 + MPSKit 型归一化）。"
function _ternary_fixedpoints(below::MixedCanonicalMPS, operator, above::MixedCanonicalMPS;
                              tol::Real, krylovdim::Int, maxiter::Int)
    N = length(below)
    L = isnothing(operator) ? N : length(operator)
    (N % L == 0 && length(above) == N) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    T = promote_type(scalartype(below), scalartype(above))
    Dw = isnothing(operator) ? 1 : size(operator[1], 1)
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)
    Dr = size(below.AR[1], 3)
    Wop = isnothing(operator) ? (ℓ -> nothing) : (ℓ -> operator[_mod1(ℓ, L)])

    # ---- 左固定点：T_L(above.AL, operator, below.AL) 的主本征向量 ----
    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Dw, Da)
        for ℓ in 1:N
            W = Wop(ℓ)
            GL = isnothing(W) ? push_env_left(GL, below.AL[ℓ], above.AL[ℓ]) :
                 push_env_left(GL, below.AL[ℓ], W, above.AL[ℓ])
        end
        return vec(GL)
    end
    _, GL1 = eigsolve(Tleft, ones(T, Dl * Dw * Da), 1, :LM;
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    GLs = Vector{Array{T,3}}(undef, N)
    GLs[1] = reshape(GL1[1], Dl, Dw, Da)
    for ℓ in 2:N
        W = Wop(ℓ - 1)
        GLs[ℓ] = isnothing(W) ? push_env_left(GLs[ℓ-1], below.AL[ℓ-1], above.AL[ℓ-1]) :
                 push_env_left(GLs[ℓ-1], below.AL[ℓ-1], W, above.AL[ℓ-1])
    end

    # ---- 右固定点：T_R(above.AR, operator, below.AR) 的主本征向量 ----
    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dw, Dr)
        for ℓ in N:-1:1
            W = Wop(ℓ)
            GR = isnothing(W) ? push_env_right(GR, above.AR[ℓ], below.AR[ℓ]) :
                 push_env_right(GR, above.AR[ℓ], W, below.AR[ℓ])
        end
        return vec(GR)
    end
    _, GRN = eigsolve(Tright, ones(T, Da * Dw * Dr), 1, :LM;
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    GRs = Vector{Array{T,3}}(undef, N)
    GRs[N] = reshape(GRN[1], Da, Dw, Dr)
    for ℓ in N-1:-1:1
        W = Wop(ℓ + 1)
        GRs[ℓ] = isnothing(W) ? push_env_right(GRs[ℓ+1], above.AR[ℓ+1], below.AR[ℓ+1]) :
                 push_env_right(GRs[ℓ+1], above.AR[ℓ+1], W, below.AR[ℓ+1])
    end

    # ---- 归一化（对标 MPSKit：GR Frobenius 归一、GL 按局部重叠 λ 缩放）----
    for ℓ in 1:N
        GRs[ℓ] .= GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        GLn = GLs[inext]
        GR = GRs[ℓ]
        Cnew = _mapC(GLn, GR, above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("三元环境：site $ℓ 局部重叠 λ = 0")
        GLs[inext] .= GLn ./ λ
    end
    return GLs, GRs
end

# ---- Hamiltonian 路径的逐层 kernel ----

"通道切片 (l → i) 的左推进：L′ = Σ conj(A[a,ū,a′])·L[a,b]·Wl[ū,d]·A[b,d,b′]。"
function _push_slice_left(L::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, ū, a′]) * L[a, b] * Wl[ū, d] * A[b, d, b′]
    return L′
end

"通道切片 (i → l) 的右推进：对标 MPSKit transfer_right，A 与 ket(d) 收缩、conj(A) 与 bra(u) 收缩。"
function _push_slice_right(R::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, d, a] * Wl[ū, d] * conj(A[b′, ū, b]) * R[a, b]
end

"""
    environments(ψ, H::MPOHamiltonian; tol, maxiter) -> DMRGCache

Jordan 哈密顿量环境：逐 level 线性求解（对标 MPSKit 的
`compute_leftenvs!/compute_rightenvs!(::InfiniteMPOHamiltonian)`）。
"""
function environments(ψ::MixedCanonicalMPS, H::MPOHamiltonian;
                      tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                      krylovdim::Int = Defaults.krylovdim,
                      init_lefts::Union{Nothing,Vector{<:AbstractArray}} = nothing,
                      init_rights::Union{Nothing,Vector{<:AbstractArray}} = nothing)
    N = length(ψ)
    nl = mpobond(H)
    T = promote_type(scalartype(ψ), scalartype(H))
    Ds = [size(ψ.AL[ℓ], 1) for ℓ in 1:N]
    lefts = [zeros(T, Ds[ℓ], nl, Ds[ℓ]) for ℓ in 1:N]
    rights = [zeros(T, Ds[ℓ], nl, Ds[ℓ]) for ℓ in 1:N]
    Wds = [tompotensor(H[ℓ]) for ℓ in 1:N]      # (nl, d, nl, d)
    Ids = [Matrix{T}(I, Ds[ℓ], Ds[ℓ]) for ℓ in 1:N]

    # 单位层：level 1（左）与 level nl（右）= ρ = I（AL/AR 规范固定点）
    for ℓ in 1:N
        lefts[ℓ][:, 1, :] .= Ids[ℓ]
        rights[ℓ][:, nl, :] .= Ids[ℓ]
    end

    # 对标 MPSKit environment_alg：krylovdim 截断到环境向量空间维数 D·D
    max_krylovdim = Ds[1] * Ds[1]
    linalg = KrylovKit.GMRES(; tol = tol, maxiter = maxiter,
                            krylovdim = min(max_krylovdim, krylovdim))

    # ---- 左环境：level 2..nl ----
    # 对标 MPSKit compute_leftenvs!：每 level 先 cyclethrough 全胞扫描（通道跳转
    # 可发生在任意中间 site），在 site 1 解 (1 − T)·x = RHS 后再扫描一次展开。
    for i in 2:nl
        D = Ds[1]
        # 热启动初值（对标 MPSKit：复用上一轮环境的第 i 层）
        prev = if init_lefts !== nothing && size(init_lefts[1]) == size(lefts[1])
            vec(copy(init_lefts[1][:, i, :]))
        else
            vec(copy(lefts[1][:, i, :]))
        end
        # 第一次全胞扫描：RHS 落在 lefts[1][i]（写 site+1，读 site，顺序覆盖）
        _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Ds, T)
        RHS = copy(lefts[1][:, i, :])
        if isidentitylevel(H, i)
            # MPSKit：T=regularize(Tm, l_LL=I, r_LL=C[N]C[N]')，linsolve 用 flip(T)，
            # 而 flip(RegTM) 交换 l/r 参数（transfermatrix.jl:40），
            # 故实际作用为 T(v) − tr(r_LL·v)·l_LL = T(v) − tr(ρr·v)·I。
            I1 = Ids[1]
            ρr = ψ.C[N] * ψ.C[N]'
            op = function (v::AbstractVector)
                X = reshape(v, D, D)
                for ℓ in 1:N
                    X = push_env_left(X, ψ.AL[ℓ])
                end
                regularize!(X, ρr, I1)
                return vec(X)
            end
            x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
            lefts[1][:, i, :] .= reshape(x, D, D)
            # 第二次扫描：把修正后的 site 1 展开到 site 2..N（MPSKit 仅 L>1 时执行）
            if N > 1
                _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Ds, T)
            end
            # 恒等层：逐 site 投影掉固定点分量
            for ℓ in 1:N
                ρr_ℓ = ψ.C[ℓ - 1] * ψ.C[ℓ - 1]'
                regularize!(@view(lefts[ℓ][:, i, :]), ρr_ℓ, Ids[ℓ])
            end
        else
            if !isemptylevel(H, i)
                op = function (v::AbstractVector)
                    X = reshape(v, D, D)
                    for ℓ in 1:N
                        X = _push_slice_left(X, view(Wds[ℓ], i, :, i, :), ψ.AL[ℓ])
                    end
                    return vec(X)
                end
                x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
                lefts[1][:, i, :] .= reshape(x, D, D)
            end
            if N > 1
                _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Ds, T)
            end
        end
    end

    # ---- 右环境：level nl-1..1（反向全胞扫描） ----
    for i in nl-1:-1:1
        D = Ds[N]
        # 热启动初值（对标 MPSKit：复用上一轮环境的第 i 层）
        prev = if init_rights !== nothing && size(init_rights[N]) == size(rights[N])
            vec(copy(init_rights[N][:, i, :]))
        else
            vec(copy(rights[N][:, i, :]))
        end
        # 第一次反向全胞扫描：RHS 落在 rights[N][i]
        _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Ds, T)
        RHS = copy(rights[N][:, i, :])
        if isidentitylevel(H, i)
            # l_RR(ψ, 1) = C[N]'·C[N]（默认 loc=1，site 0 即 site N），r_RR = I
            IN = Ids[N]
            ρl = ψ.C[N]' * ψ.C[N]
            op = function (v::AbstractVector)
                X = reshape(v, D, D)
                for ℓ in N:-1:1
                    X = push_env_right(X, ψ.AR[ℓ])
                end
                regularize!(X, ρl, IN)
                return vec(X)
            end
            x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
            rights[N][:, i, :] .= reshape(x, D, D)
            if N > 1
                _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Ds, T)
            end
            # 恒等层：逐 site 投影
            for ℓ in 1:N
                ρl_ℓ = ψ.C[ℓ]' * ψ.C[ℓ]
                regularize!(@view(rights[ℓ][:, i, :]), ρl_ℓ, Ids[ℓ])
            end
        else
            if !isemptylevel(H, i)
                op = function (v::AbstractVector)
                    X = reshape(v, D, D)
                    for ℓ in N:-1:1
                        X = _push_slice_right(X, view(Wds[ℓ], i, :, i, :), ψ.AR[ℓ])
                    end
                    return vec(X)
                end
                x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
                rights[N][:, i, :] .= reshape(x, D, D)
            end
            if N > 1
                _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Ds, T)
            end
        end
    end

    return DMRGCache(H, ψ, lefts, rights)
end

"全胞左扫描（对标 MPSKit left_cyclethrough!）：`GL[site+1, i] = Σ_{l≤i} W_site[l→i]·GL[site, l]`。"
function _left_cyclethrough!(lefts, Wds, ALs, i::Int, N::Int, Ds, T)
    for site in 1:N
        snext = site == N ? 1 : site + 1
        tgt = zeros(T, Ds[snext], Ds[snext])
        for l in 1:i
            tgt .+= _push_slice_left(lefts[site][:, l, :],
                                     view(Wds[site], l, :, i, :), ALs[site])
        end
        lefts[snext][:, i, :] .= tgt
    end
    return lefts
end

"全胞右扫描（对标 MPSKit right_cyclethrough!）：`GR[site−1, i] = Σ_{l≥i} W_site[i→l]·GR[site, l]`。"
function _right_cyclethrough!(rights, Wds, ARs, i::Int, N::Int, Ds, T)
    nl = size(Wds[1], 1)
    for site in N:-1:1
        sprev = site == 1 ? N : site - 1
        tgt = zeros(T, Ds[sprev], Ds[sprev])
        for l in i:nl
            tgt .+= _push_slice_right(rights[site][:, l, :],
                                      view(Wds[site], i, :, l, :), ARs[site])
        end
        rights[sprev][:, i, :] .= tgt
    end
    return rights
end

"""
    recalculate!(envs::DMRGCache, ψ, operator = envs.operator; kwargs...) -> envs

按（可能已更新的）`ψ` 从头重算固定点并重新展开局部环境（热启动：旧环境作为
逐 level linsolve 的初值，对标 MPSKit 原地 recalculate!）。
`kwargs`（如 `tol`、`maxiter`、`krylovdim`）透传给 `environments`，
对标 MPSKit `recalculate!(...; alg_environments.tol)` 的动态环境容差。
"""
function recalculate!(envs::DMRGCache, ψ::MixedCanonicalMPS,
                      operator::Union{MPOHamiltonian,InfiniteMPO} = envs.operator;
                      kwargs...)
    # 热启动：把旧环境作为逐 level linsolve 的初值（对标 MPSKit 原地 recalculate!）
    old_lefts = envs.lefts
    old_rights = envs.rights
    if operator isa MPOHamiltonian
        envs2 = environments(ψ, operator;
                             init_lefts = old_lefts, init_rights = old_rights, kwargs...)
    else
        envs2 = environments(ψ, operator; kwargs...)
    end
    copy!(envs.lefts, envs2.lefts)
    copy!(envs.rights, envs2.rights)
    envs.ket ≡ ψ || (envs = envs2)
    return envs
end

"OverlapCache / MultCache：无热启动通道，按更新后的 `ψ` 直接重建并返回新缓存
（`operator` 缺省按缓存类型取值：MultCache 用其算符、OverlapCache 为 nothing）。"
function recalculate!(envs::Union{OverlapCache,MultCache}, ψ::MixedCanonicalMPS,
                      operator = envs isa MultCache ? envs.operator : nothing;
                      kwargs...)
    return environments(ψ, operator; kwargs...)
end

"""
    normalize_envs!(envs, below, operator, above) -> envs

对标 MPSKit `normalize!`：
- 右环境归一化到 norm 1；
- 左环境缩放使得 `dot(C, C_hamiltonian * C) = 1`。

避免 IDMRG 扫描中环境范数爆炸。
"""
function normalize_envs!(envs::Environments, below::MixedCanonicalMPS,
                    operator::Union{Nothing,InfiniteMPO,MPOHamiltonian},
                    above::MixedCanonicalMPS)
    N = length(below)
    for i in 1:N
        # 右环境归一化到 norm 1
        r = envs.rights[i]
        nr = norm(r)
        if nr > 0
            r ./= nr
        end
        # λ = dot(C[i], C_hamiltonian(i) * C[i])
        hC = C_hamiltonian(i, below, operator, above, envs)
        Cproj = hC(below.C[i])
        λ = dot(below.C[i], Cproj)
        if abs(λ) > 0
            # GL[i+1] *= inv(λ)
            l = envs.lefts[_mod1(i + 1, N)]
            l ./= λ
        end
    end
    return envs
end

"""
    transfer_leftenv!(envs, ψ, operator, ψ2, site) -> envs

把左环境从 `site − 1` 推进到 `site`（IDMRG 扫描的增量更新，对标 MPSKit 的
`transfer_leftenv!`）。
"""
function transfer_leftenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓm = _mod1(site - 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa MPOHamiltonian
        tompotensor(operator[ℓm])
    else
        operator[ℓm]
    end
    if isnothing(W)
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], ψ.AL[ℓm])
    else
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], W, ψ.AL[ℓm])
    end
    return envs
end

"把右环境从 `site + 1` 推进到 `site`。"
function transfer_rightenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓp = _mod1(site + 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa MPOHamiltonian
        tompotensor(operator[ℓp])
    else
        operator[ℓp]
    end
    if isnothing(W)
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ψ.AR[ℓp])
    else
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], W, ψ.AR[ℓp])
    end
    return envs
end
