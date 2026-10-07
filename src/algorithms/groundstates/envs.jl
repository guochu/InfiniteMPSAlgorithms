# ---------------- 环境求解 kernel + 哈密顿量环境缓存 DMRGCache ----------------
#
# 哈密顿量通道（DMRGCache）的环境求解原语：非齐次线性解 `linsolve`（Schur 逐
# level 通道）、恒等层固定点投影 `regularize!`、DenseMPO 通道的转移矩阵主本征
# 向量 `dominant_env`；随后是 DMRGCache 的定义与构造器，以及环境的归一化与
# 增量推进（`normalize_envs!`/`transfer_leftenv!`/`transfer_rightenv!`）。

"""
    linsolve(operator, b, x₀, [alg]; a₀ = 1, a₁ = 1) -> (x, info)

Solve the linear system `a₀·x + a₁·A·x = b` (mirrors MPSKit's `linsolve`;
internally KrylovKit.linsolve; `alg` is `GMRES`/`BiCGStab`/`CG`).
"""
function linsolve(operator, b::AbstractVector, x₀::AbstractVector,
                  alg::KrylovKit.KrylovAlgorithm = KrylovKit.GMRES();
                  a₀ = 1, a₁ = 1)
    x, info = KrylovKit.linsolve(v -> operator(v), b, x₀, alg, a₀, a₁)
    return x, info
end

"""
    regularize!(v, lvec, rvec) -> v

Project out the identity-channel fixed-point component (mirrors MPSKit's
`regularize!`): `v ← v − rvec·⟨lvec, v⟩`. In this package's gauge
`lvec = rvec = I`, i.e. `v ← v − tr(v)·I`.
"""
function regularize!(v::AbstractMatrix, lvec::AbstractMatrix, rvec::AbstractMatrix)
    c = sum(lvec .* transpose(v))   # MPSKit semantics: Σ lvec[a,b]·v[b,a] (no conjugation)
    v .-= c .* rvec
    return v
end

function _dominant_env_matvec(op::Union{Nothing,DenseIMPO}, ψ::CanonicalIMPS, side::Symbol)
    N = length(ψ)
    identity = isnothing(op)
    return function matvec(v::AbstractVector)
        if identity
            L = reshape(v, size(ψ.AL[1], 1), size(ψ.AL[1], 1))
            if side === :left
                for ℓ in 1:N
                    L = push_env_left(L, ψ.AL[ℓ])
                end
            else
                R = L
                for ℓ in N:-1:1
                    R = push_env_right(R, ψ.AR[ℓ])
                end
                L = R
            end
            return vec(L)
        else
            W1 = op[1]
            L = reshape(v, size(ψ.AL[1], 1), size(W1, 1), size(ψ.AL[1], 1))
            if side === :left
                for ℓ in 1:N
                    L = push_env_left(L, op[ℓ], ψ.AL[ℓ])
                end
            else
                R = L
                for ℓ in N:-1:1
                    R = push_env_right(R, op[ℓ], ψ.AR[ℓ])
                end
                L = R
            end
            return vec(L)
        end
    end
end

"""
    dominant_env(ψ; side=:left, which=:LM, kwargs...) -> (λ, L)
    dominant_env(W, ψ; side=:left, which=:LM, kwargs...) -> (λ, L)

Dominant eigenvector of the identity/MPO-channel transfer matrix (tiled over
one unit cell), rank-3 `(bond, w, bond)`（identity 通道 w = 1）. The identity
channel uses AL/AR (strictly canonical) and returns `λ ≈ 1`.
"""
function dominant_env(ψ::CanonicalIMPS; side::Symbol = :left, which::Symbol = :LM, kwargs...)
    return dominant_env(nothing, ψ; side = side, which = which, kwargs...)
end

function dominant_env(op::Union{Nothing,DenseIMPO}, ψ::CanonicalIMPS;
                      side::Symbol = :left, which::Symbol = :LM,
                      tol::Real = Defaults.tol, krylovdim::Int = Defaults.krylovdim,
                      maxiter::Int = Defaults.maxiter)
    identity = isnothing(op)
    D = size(ψ.AL[1], 1)
    dim = identity ? D * D : D * size(op[1], 1) * D
    T = scalartype(ψ)
    matvec = _dominant_env_matvec(op, ψ, side)
    v0 = ones(T, dim)
    λs, vs, _ = _eigsolve(matvec, v0, 1, which; ishermitian = false, tol = tol,
                          krylovdim = krylovdim, maxiter = maxiter)
    λ = λs[1]
    L = identity ? reshape(vs[1], D, 1, D) : reshape(vs[1], D, size(op[1], 1), D)
    L ./= norm(L)
    return λ, L
end

# ---- per-level kernels of the Hamiltonian path (exclusive to the DMRGCache constructor) ----

"Left push of the channel slice (l → i): L′ = Σ conj(A[a,ū,a′])·L[a,b]·Wl[ū,d]·A[b,d,b′]."
function _push_slice_left(L::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, ū, a′]) * L[a, b] * Wl[ū, d] * A[b, d, b′]
    return L′
end

"Right push of the channel slice (i → l): mirrors MPSKit transfer_right; A
contracts with the ket (d), conj(A) with the bra (u)."
function _push_slice_right(R::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, d, a] * Wl[ū, d] * conj(A[b′, ū, b]) * R[a, b]
end

"Full-cell left sweep of an **interior** level (mirrors MPSKit left_cyclethrough!):
`GL[site+1, i] = Σ_{l≤i} W_site[l→i]·GL[site, l]`. 矩形 Schur 链：目标层 i
在下一键不是内层（`i ≥ space_r(site)`，含闭合位）时跳过——闭合层由
[`_left_closing_sweep!`](@ref) 专责；读取上限 `min(i, space_l(site))`
（左侧键的层数，闭合位的读取块结构为零，不贡献）。"
function _left_cyclethrough!(lefts, Wds, ALs, i::Int, N::Int, Ds, T)
    for site in 1:N
        snext = site == N ? 1 : site + 1
        # 目标键 = site 的右键（逐站键维不同，取张量自身的实际尺寸）
        χs = size(ALs[site], 3)
        ms, ns = size(Wds[site], 1), size(Wds[site], 3)
        i <= ns - 1 || continue
        tgt = zeros(T, χs, χs)
        for l in 1:min(i, ms)
            tgt .+= _push_slice_left(lefts[site][:, l, :],
                                     view(Wds[site], l, :, i, :), ALs[site])
        end
        lefts[snext][:, i, :] .= tgt
    end
    return lefts
end

"Full-cell left sweep of the **closing** level: at every site the whole closing
column `W_site[:, n_site]` is contracted（on-site `D`、内层通道的 `B` 闭合块、
上一键闭合层的 corner 延续）——矩形 Schur 链上闭合层逐键换标号（corner 流：
`closing(b) → last row → corner → closing(b+1)`，角块恒为 I，故其环转移 =
纯 MPS 转移，在 DMRGCache 里走 regularize 路径）。"
function _left_closing_sweep!(lefts, Wds, ALs, N::Int, T)
    for site in 1:N
        snext = site == N ? 1 : site + 1
        χs = size(ALs[site], 3)
        ms, ns = size(Wds[site], 1), size(Wds[site], 3)
        tgt = zeros(T, χs, χs)
        for l in 1:ms
            tgt .+= _push_slice_left(lefts[site][:, l, :],
                                     view(Wds[site], l, :, ns, :), ALs[site])
        end
        lefts[snext][:, ns, :] .= tgt
    end
    return lefts
end

"Full-cell right sweep (mirrors MPSKit right_cyclethrough!):
`GR[site−1, i] = Σ_{l≥i} W_site[i→l]·GR[site, l]`. 矩形 Schur 链：目标层 i
在上一键不存在（`i > space_l(site)`）时跳过；读取范围 `i:space_r(site)`
（含闭合位的 corner 读取——闭合右环境恒为 I 播种，corner 推进保持不变）。"
function _right_cyclethrough!(rights, Wds, ARs, i::Int, N::Int, Ds, T)
    for site in N:-1:1
        sprev = site == 1 ? N : site - 1
        # 目标键 = site 的左键（逐站键维不同，取张量自身的实际尺寸）
        χs = size(ARs[site], 1)
        ms, ns = size(Wds[site], 1), size(Wds[site], 3)
        i <= ms || continue
        tgt = zeros(T, χs, χs)
        for l in i:ns
            tgt .+= _push_slice_right(rights[site][:, l, :],
                                      view(Wds[site], i, :, l, :), ARs[site])
        end
        rights[sprev][:, i, :] .= tgt
    end
    return rights
end

"对角通道切片 `Wd[i, i]`（逐层 linsolve 的转移算子）。矩形 Schur 链上 i 可
超出该站的内层数（通道未跨越此站）：通道缺失 = 零块，整个环上的对角转移
为 0，`(1 − T)` 平凡可逆、linsolve 一步收敛到 `x = RHS`。"
@inline function _diag_channel(Wd::AbstractArray{T,4}, i::Int) where {T}
    return if i <= min(size(Wd, 1), size(Wd, 3))
        view(Wd, i, :, i, :)
    else
        zeros(T, size(Wd, 2), size(Wd, 4))
    end
end

# ---------------- 哈密顿量环境缓存（DMRGCache） ----------------

"""
    DMRGCache(operator, ket, lefts, rights)
    DMRGCache(ψ, operator) -> DMRGCache

Environment cache of the Hamiltonian channel (ground-state VUMPS / IDMRG /
TDVP and energy evaluation): `operator::Union{SparseIMPO,DenseIMPO}` —
the `w` dimension of an `SparseIMPO` equals the number of Schur levels
(per-level solves); an `DenseIMPO` is a dense MPO Hamiltonian
(transfer-matrix dominant eigenvector).

- `lefts[ℓ]`: the left environment of site ℓ, `(ket bond, w, ket bond)`;
- `rights[ℓ]`: the right environment of site ℓ, `(ket bond, w, ket bond)`.
"""
struct DMRGCache{H<:Union{SparseIMPO,DenseIMPO},K<:CanonicalIMPS,T} <: Environments
    operator::H
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"哈密顿量通道缓存的站数（`Environments` 的 `length` 契约）：ket 的单胞长度——
operator 的单胞长度恒为其因子（N 是它的倍数），故最小公倍数 = `length(envs.ket)`。"
Base.length(envs::DMRGCache) = length(envs.ket)

"""
    DMRGCache(ψ, W::DenseIMPO; kwargs...) -> DMRGCache

Dense MPO Hamiltonian channel: transfer-matrix dominant eigenvector
(Krylov Arnoldi). An `DenseIMPO` can be used directly as the Hamiltonian of
a ground-state search, as well as in `expectationvalue(ψ, W)`.

!!! note "与 `SparseIMPO` 通道的语义差别（MPSKit 对齐）"
    `DenseIMPO` 通道算的是**周期 trace 的完整收缩**（环境吃满 MPO 的 level
    指标），环境标度约定为 MPSKit `normalize!(::InfiniteEnvironments{DenseMPO})`：
    每个右环境做 Frobenius 归一、每个左环境按 C 通道投影
    `λ_i = ⟨C_i|H_C(i)|C_i⟩` 缩放。

    对**Hamiltonian 型** MPO（带闭合层结构），完整 trace 会额外计入恒等层
    bookkeeping，因此 `expectationvalue(ψ, W::DenseIMPO)` 的值与真实能量相差一个
    与表示/态都有关的项（MPSKit 的 `expectation_value(ψ, ::InfiniteMPO)` 行为完全
    相同，N=1 时两者逐位一致）；**能量请走 `SparseIMPO` 通道**（闭列公式，
    `tfim_hamiltonian` 等），它对齐 MPSKit 的 `InfiniteMPOHamiltonian` 且精确。

    另外该通道的环境由算子通道转移矩阵的**主本征向量**给出：当态存在（近似）
    退耦合键时该本征空间（近似）退化，选取不由归一化唯一确定，因此对同一物理态的
    不同（例如零填充扩键的）表示不严格不变；`SparseIMPO` 通道用非齐次线性解，
    无此歧义（零填充下逐位不变）。时间推进用的 `make_time_mpo` 生成元接近恒等，
    该标度约定与 1 的偏差为 O(dt)，实际使用不受影响。
"""
function DMRGCache(ψ::CanonicalIMPS, operator::DenseIMPO; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Array{T,3}}(undef, N)
    rights = Vector{Array{T,3}}(undef, N)
    _, L0 = dominant_env(operator, ψ; side = :left, kwargs...)
    _, R0 = dominant_env(operator, ψ; side = :right, kwargs...)
    lefts[1] = L0
    for ℓ in 2:N
        lefts[ℓ] = push_env_left(lefts[ℓ-1], operator[ℓ-1], ψ.AL[ℓ-1])
    end
    rights[N] = R0
    for ℓ in N-1:-1:1
        rights[ℓ] = push_env_right(rights[ℓ+1], operator[ℓ+1], ψ.AR[ℓ+1])
    end
    # 对标 MPSKit `normalize!(::InfiniteEnvironments{DenseMPO})`：先把每个 GR 做
    # Frobenius 归一，再逐 site 用 `C_hamiltonian(i)` 的 C 通道投影
    # λ_i = ⟨C_i|H_C(i)|C_i⟩ 缩放 GL_{i+1}（= 键 i 上的左环境）。逐站取环境，
    # 非均匀键 profile 下形状自动一致。
    return normalize_envs!(DMRGCache(operator, ψ, lefts, rights), ψ, operator, ψ)
end

"""
    DMRGCache(ψ, H::SparseIMPO; tol, maxiter, krylovdim, init_lefts, init_rights)
        -> DMRGCache

Schur Hamiltonian environments: per-level linear solves (mirroring MPSKit's
`compute_leftenvs!/compute_rightenvs!(::SparseIMPO)`).
`init_lefts`/`init_rights` provide warm-started initial values (see
[`recalculate!`](@ref)).

矩形 Schur 链（unitcell > 1、键层数逐键不同）：`lefts[ℓ]` 的 w 维 =
`space_l(H[ℓ])`（键 ℓ-1 的层数）、`rights[ℓ]` 的 w 维 = `space_r(H[ℓ])`；
左环境在键 N 上按其层数逐层求解、右环境同理，通道缺失的层（该键无此层）
在对角转移中为零块（见 [`_diag_channel`](@ref)），`(1 − T)` 平凡可逆。
要求 ket 的单胞长度是算符单胞长度的整数倍（键结构在 ket 单胞上闭合）。
"""
function DMRGCache(ψ::CanonicalIMPS, H::SparseIMPO;
                   tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                   krylovdim::Int = Defaults.krylovdim,
                   init_lefts::Union{Nothing,Vector{<:AbstractArray}} = nothing,
                   init_rights::Union{Nothing,Vector{<:AbstractArray}} = nothing)
    N = length(ψ)
    T = promote_type(scalartype(ψ), scalartype(H))
    # 逐站键维：lefts[ℓ] 在键 ℓ-1（键维 χ_{ℓ-1} = size(AL[ℓ],1)），
    # rights[ℓ] 在键 ℓ（键维 χ_ℓ = size(AL[ℓ],3)）。非均匀键下两者不再相同。
    Dl = [size(ψ.AL[ℓ], 1) for ℓ in 1:N]
    Dr = [size(ψ.AL[ℓ], 3) for ℓ in 1:N]
    Wds = [tompotensor(H[ℓ]) for ℓ in 1:N]      # (m_ℓ, d, n_ℓ, d)，矩形链逐站形状
    (size(Wds[N], 3) == size(Wds[1], 1)) || throw(DimensionMismatch(
        "operator level chain does not close over the ket unit cell " *
        "(N = $N): bond N has $(size(Wds[N], 3)) right levels vs bond 1 has $(size(Wds[1], 1)) left levels " *
        "—— ket 单胞长度须为算符单胞长度的整数倍"))
    ml = [size(W, 1) for W in Wds]               # lefts[ℓ] 的 w 维（键 ℓ-1 的层数）
    nr = [size(W, 3) for W in Wds]               # rights[ℓ] 的 w 维（键 ℓ 的层数）
    lefts = [zeros(T, Dl[ℓ], ml[ℓ], Dl[ℓ]) for ℓ in 1:N]
    rights = [zeros(T, Dr[ℓ], nr[ℓ], Dr[ℓ]) for ℓ in 1:N]
    IdL = [Matrix{T}(I, Dl[ℓ], Dl[ℓ]) for ℓ in 1:N]
    IdR = [Matrix{T}(I, Dr[ℓ], Dr[ℓ]) for ℓ in 1:N]

    # 单位层：level 1（左真空）与 level n_ℓ（右闭合，逐站）= ρ = I（AL/AR 规范固定点）
    for ℓ in 1:N
        lefts[ℓ][:, 1, :] .= IdL[ℓ]
        rights[ℓ][:, nr[ℓ], :] .= IdR[ℓ]
    end

    # 对标 MPSKit environment_alg：krylovdim 截断到环境向量空间维数 D·D
    max_krylovdim = Dl[1] * Dl[1]
    linalg = KrylovKit.GMRES(; tol = tol, maxiter = maxiter,
                            krylovdim = min(max_krylovdim, krylovdim))

    # ---- 左环境 ----
    # 对标 MPSKit compute_leftenvs!：每 level 先 cyclethrough 全胞扫描（通道跳转
    # 可发生在任意中间 site），在 site 1 解 (1 − T)·x = RHS 后再扫描一次展开。
    # 矩形链的轮次结构：内层标号 i = 2..jmax 递增处理——
    # · i ≤ m₁−1：键 N 的内层，需要定点求解（恒等内层走 regularize 路径）；
    # · i > m₁−1：瞬态层（只在更宽的中间键上存在，无环返回通道），单次前向
    #   扫描即定值；最后是闭合通道轮（键 N 的闭合层，角块恒为 I ⇒ 转移 =
    #   纯 MPS 转移，须 regularize——均匀链上即原 i = nl 的恒等层轮）。
    jmax = maximum(nr) - 1
    for i in 2:jmax
        if i <= ml[1] - 1
            D = Dl[1]        # 左环境定义在键 N 上
            # 热启动初值（对标 MPSKit：复用上一轮环境的第 i 层）
            prev = if init_lefts !== nothing && size(init_lefts[1]) == size(lefts[1])
                vec(copy(init_lefts[1][:, i, :]))
            else
                vec(copy(lefts[1][:, i, :]))
            end
            # 第一次全胞扫描：RHS 落在 lefts[1][i]（写 site+1，读 site，顺序覆盖）
            _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Dl, T)
            RHS = copy(lefts[1][:, i, :])
            if isidentitylevel(H, i)
                # MPSKit：T=regularize(Tm, l_LL=I, r_LL=C[N]C[N]')，linsolve 用 flip(T)，
                # 而 flip(RegTM) 交换 l/r 参数（transfermatrix.jl:40），
                # 故实际作用为 T(v) − tr(r_LL·v)·l_LL = T(v) − tr(ρr·v)·I。
                I1 = IdL[1]
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
                    _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Dl, T)
                end
                # 恒等层：逐 site 投影掉固定点分量
                for ℓ in 1:N
                    ρr_ℓ = ψ.C[ℓ - 1] * ψ.C[ℓ - 1]'
                    regularize!(@view(lefts[ℓ][:, i, :]), ρr_ℓ, IdL[ℓ])
                end
            else
                if !isemptylevel(H, i)
                    op = function (v::AbstractVector)
                        X = reshape(v, D, D)
                        for ℓ in 1:N
                            X = _push_slice_left(X, _diag_channel(Wds[ℓ], i), ψ.AL[ℓ])
                        end
                        return vec(X)
                    end
                    x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
                    lefts[1][:, i, :] .= reshape(x, D, D)
                end
                if N > 1
                    _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Dl, T)
                end
            end
        else
            # 瞬态内层：无环返回通道（T = 0），单次前向扫描即定值
            _left_cyclethrough!(lefts, Wds, ψ.AL, i, N, Dl, T)
        end
    end

    # ---- 左环境的闭合通道轮（键 N 的闭合层；在所有内层/瞬态层定值之后） ----
    begin
        i = ml[1]             # = n_N：键 N 的闭合层标号
        D = Dl[1]
        prev = if init_lefts !== nothing && size(init_lefts[1]) == size(lefts[1])
            vec(copy(init_lefts[1][:, i, :]))
        else
            vec(copy(lefts[1][:, i, :]))
        end
        _left_closing_sweep!(lefts, Wds, ψ.AL, N, T)
        RHS = copy(lefts[1][:, i, :])
        # corner 流的环转移 = 纯 MPS 转移（同上 isidentitylevel 路径的 regularization）
        I1 = IdL[1]
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
        if N > 1
            _left_closing_sweep!(lefts, Wds, ψ.AL, N, T)
        end
        # 恒等层：逐 site 投影掉固定点分量（闭合层逐键换标号 = 各键的最后一层）
        for ℓ in 1:N
            ρr_ℓ = ψ.C[ℓ - 1] * ψ.C[ℓ - 1]'
            regularize!(@view(lefts[ℓ][:, ml[ℓ], :]), ρr_ℓ, IdL[ℓ])
        end
    end

    # ---- 右环境 ----
    # 矩形链的轮次结构：内层标号 i = jmax..1 递减处理（高标签先定值）：
    # · i ≤ n_N−1：键 N 的内层/真空层，定点求解（i = 1 真空走 regularize 路径）；
    # · i > n_N−1：瞬态层，单次反向扫描即定值。闭合右环境恒为 I 播种、不解。
    for i in jmax:-1:1
        if i <= nr[N] - 1
            D = Dr[N]        # 右环境定义在键 N 上
            # 热启动初值（对标 MPSKit：复用上一轮环境的第 i 层）
            prev = if init_rights !== nothing && size(init_rights[N]) == size(rights[N])
                vec(copy(init_rights[N][:, i, :]))
            else
                vec(copy(rights[N][:, i, :]))
            end
            # 第一次反向全胞扫描：RHS 落在 rights[N][i]
            _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Dr, T)
            RHS = copy(rights[N][:, i, :])
            if isidentitylevel(H, i)
                # l_RR(ψ, 1) = C[N]'·C[N]（默认 loc=1，site 0 即 site N），r_RR = I
                IN = IdR[N]
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
                    _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Dr, T)
                end
                # 恒等层：逐 site 投影
                for ℓ in 1:N
                    ρl_ℓ = ψ.C[ℓ]' * ψ.C[ℓ]
                    regularize!(@view(rights[ℓ][:, i, :]), ρl_ℓ, IdR[ℓ])
                end
            else
                if !isemptylevel(H, i)
                    op = function (v::AbstractVector)
                        X = reshape(v, D, D)
                        for ℓ in N:-1:1
                            X = _push_slice_right(X, _diag_channel(Wds[ℓ], i), ψ.AR[ℓ])
                        end
                        return vec(X)
                    end
                    x, info = linsolve(op, vec(RHS), prev, linalg; a₀ = 1, a₁ = -1)
                    rights[N][:, i, :] .= reshape(x, D, D)
                end
                if N > 1
                    _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Dr, T)
                end
            end
        else
            # 瞬态内层：单次反向扫描即定值
            _right_cyclethrough!(rights, Wds, ψ.AR, i, N, Dr, T)
        end
    end

    return DMRGCache(H, ψ, lefts, rights)
end

"""
    recalculate!(envs::DMRGCache, ψ, operator = envs.operator; kwargs...) -> envs

Recompute the fixed points from scratch for the (possibly updated) `ψ` and
re-expand the local environments (warm start: the old environments serve as
the initial values of the per-level linsolves, mirroring MPSKit's in-place
recalculate!). `kwargs` (e.g. `tol`, `maxiter`, `krylovdim`) are forwarded to
the `DMRGCache` constructor, mirroring the dynamic environment tolerance of
MPSKit's `recalculate!(...; alg_environments.tol)`.
"""
function recalculate!(envs::DMRGCache, ψ::CanonicalIMPS,
                      operator::Union{SparseIMPO,DenseIMPO} = envs.operator;
                      kwargs...)
    # 热启动：把旧环境作为逐 level linsolve 的初值（对标 MPSKit 原地 recalculate!）
    old_lefts = envs.lefts
    old_rights = envs.rights
    if operator isa SparseIMPO
        envs2 = DMRGCache(ψ, operator;
                          init_lefts = old_lefts, init_rights = old_rights, kwargs...)
    else
        envs2 = DMRGCache(ψ, operator; kwargs...)
    end
    copy!(envs.lefts, envs2.lefts)
    copy!(envs.rights, envs2.rights)
    envs.ket ≡ ψ || (envs = envs2)
    return envs
end

# ---- environment normalization and incremental pushes ----

"""
    normalize_envs!(envs, below, operator, above) -> envs

Mirrors MPSKit's `normalize!`:
- right environments are normalized to unit norm;
- left environments are scaled such that `dot(C, C_hamiltonian * C) = 1`.

Prevents environment-norm blowup during IDMRG sweeps.
"""
function normalize_envs!(envs::DMRGCache, below::CanonicalIMPS,
                         operator::Union{DenseIMPO,SparseIMPO},
                         above::CanonicalIMPS)
    N = length(below)
    for i in 1:N
        # normalize the right environment to unit norm
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

Push the left environment from `site − 1` to `site` (the incremental update of
the IDMRG sweep, mirroring MPSKit's `transfer_leftenv!`).
"""
function transfer_leftenv!(envs::DMRGCache, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓm = _mod1(site - 1, length(ψ))
    W = operator isa SparseIMPO ?
        tompotensor(operator[ℓm]) : operator[ℓm]   # Schur 张量稠密化进统一 kernel
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], W, ψ.AL[ℓm])
    return envs
end

"Push the right environment from `site + 1` to `site`."
function transfer_rightenv!(envs::DMRGCache, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓp = _mod1(site + 1, length(ψ))
    W = operator isa SparseIMPO ?
        tompotensor(operator[ℓp]) : operator[ℓp]   # Schur 张量稠密化进统一 kernel
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], W, ψ.AR[ℓp])
    return envs
end
