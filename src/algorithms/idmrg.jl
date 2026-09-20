# ---------------- IDMRG（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，single-site） ----------------
#
# 算法定义（IDMRG 参数对象）见 algdefs.jl。

# ---------------- 哈密顿量环境缓存（DMRGCache） ----------------

"""
    DMRGCache(operator, ket, lefts, rights)
    DMRGCache(ψ, operator) -> DMRGCache

哈密顿量通道环境缓存（基态搜索 VUMPS / IDMRG / TDVP 与能量计算）：
`operator::Union{MPOHamiltonian,InfiniteMPO}`——`MPOHamiltonian` 的 w 维 =
Jordan 层数（逐 level 求解）；`InfiniteMPO` 为稠密 MPO 哈密顿量
（转移矩阵主本征向量）。

- `lefts[ℓ]`：site ℓ 左环境 `(ket键, w, ket键)`；
- `rights[ℓ]`：site ℓ 右环境 `(ket键, w, ket键)`。
"""
struct DMRGCache{H<:Union{MPOHamiltonian,InfiniteMPO},K<:InfiniteCanonicalMPS,T} <: Environments
    operator::H
    ket::K
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

"""
    DMRGCache(ψ, W::InfiniteMPO; kwargs...) -> DMRGCache

稠密 MPO 哈密顿量通道：转移矩阵主本征向量（Krylov Arnoldi）。`InfiniteMPO`
可直接作为基态搜索的哈密顿量，也用于 `expectationvalue(ψ, W)`。
"""
function DMRGCache(ψ::InfiniteCanonicalMPS, operator::InfiniteMPO; kwargs...)
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
    DMRGCache(ψ, H::MPOHamiltonian; tol, maxiter, krylovdim, init_lefts, init_rights)
        -> DMRGCache

Jordan 哈密顿量环境：逐 level 线性求解（对标 MPSKit 的
`compute_leftenvs!/compute_rightenvs!(::InfiniteMPOHamiltonian)`）。
`init_lefts`/`init_rights` 提供热启动初值（见 [`recalculate!`](@ref)）。
"""
function DMRGCache(ψ::InfiniteCanonicalMPS, H::MPOHamiltonian;
                   tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                   krylovdim::Int = Defaults.krylovdim,
                   init_lefts::Union{Nothing,Vector{<:AbstractArray}} = nothing,
                   init_rights::Union{Nothing,Vector{<:AbstractArray}} = nothing)
    N = length(ψ)
    nl = bonddim(H)
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

"""
    recalculate!(envs::DMRGCache, ψ, operator = envs.operator; kwargs...) -> envs

按（可能已更新的）`ψ` 从头重算固定点并重新展开局部环境（热启动：旧环境作为
逐 level linsolve 的初值，对标 MPSKit 原地 recalculate!）。
`kwargs`（如 `tol`、`maxiter`、`krylovdim`）透传给 `DMRGCache` 构造器，
对标 MPSKit `recalculate!(...; alg_environments.tol)` 的动态环境容差。
"""
function recalculate!(envs::DMRGCache, ψ::InfiniteCanonicalMPS,
                      operator::Union{MPOHamiltonian,InfiniteMPO} = envs.operator;
                      kwargs...)
    # 热启动：把旧环境作为逐 level linsolve 的初值（对标 MPSKit 原地 recalculate!）
    old_lefts = envs.lefts
    old_rights = envs.rights
    if operator isa MPOHamiltonian
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

"_transpose_tail(A) / _transpose_front(A)：前后端指标交换 `(Dl, d, Dr) ↔ (Dr, d, Dl)`。"
_transpose_tail(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))
_transpose_front(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))

"_left_orth3(AC; alg)：`(Dl·d, Dr)` QR 分裂 → `(AL, C)`（`positive = true` 即 `QRpos`）。"
function _left_orth3(AC::AbstractArray{T,3}; alg = Defaults.alg_orth()) where {T}
    Dl, d, Dr = size(AC)
    Q, R = leftorth(reshape(AC, Dl * d, Dr); alg = alg)
    return reshape(Q, Dl, d, size(Q, 2)), R
end

"_right_orth3(AC; alg)：`(Dl, d·Dr)` LQ 分裂 → `(C, AR)`。"
function _right_orth3(AC::AbstractArray{T,3}; alg = LQpos()) where {T}
    Dl, d, Dr = size(AC)
    L, Q = rightorth(reshape(AC, Dl, d * Dr); alg = alg)
    return L, reshape(Q, size(L, 2), d, Dr)
end

"MPSKit 的 `_localupdate_sweep_idmrg!`：前向 + 后向扫描，返回 `(ψ, envs, C_old, E)`。"
function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve)
    N = length(ψ)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:N
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = _left_orth3(ψ.AC[pos])
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end
    # right to left sweep
    for pos in N:-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.C[pos - 1], ψ.AR[pos] = _right_orth3(ψ.AC[pos])
        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end

function find_groundstate(ψ₀::InfiniteCanonicalMPS, operator, alg::IDMRG,
                          envs::Environments = DMRGCache(ψ₀, operator))
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    E = expectationvalue(ψ, operator, envs)
    alg.verbosity > 0 && _logiter(stdout, "IDMRG", 0, ϵ, "f" => E)
    iter = 0
    for outer iter in 1:alg.maxiter
        # MPSKit 第 iter 次扫描时 state.iter = iter-1
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter - 1, ϵ)
        ψ, envs, C_old, E_new = _localupdate_sweep_idmrg!(ψ, operator, envs, alg_eigsolve)
        # error criterion（bond 0 中心矩阵之差）
        ϵ = norm(ψ.C[0] - C_old)
        # new energy
        ΔE = (E_new - E) / 2
        E = E_new
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ, "f" => E, "ΔE" => ΔE)
        ϵ ≤ alg.tol && break
    end
    # 规范恢复：从 AR 重建（对标 MPSKit 的 `InfiniteMPS(mps.AR)`）
    N = length(ψ)
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = InfiniteCanonicalMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, ϵ
end
