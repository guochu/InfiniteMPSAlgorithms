# ---------------- IDMRG（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，single-site） ----------------
#
# 算法定义（IDMRG 参数对象）见 algdefs.jl。

# ---------------- 哈密顿量环境缓存（DMRGCache） ----------------

"""
    DMRGCache(operator, ket, lefts, rights)
    DMRGCache(ψ, operator) -> DMRGCache

Environment cache of the Hamiltonian channel (ground-state VUMPS / IDMRG /
TDVP and energy evaluation): `operator::Union{SparseIMPO,DenseIMPO}` —
the `w` dimension of an `SparseIMPO` equals the number of Jordan levels
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

"""
    DMRGCache(ψ, W::DenseIMPO; kwargs...) -> DMRGCache

Dense MPO Hamiltonian channel: transfer-matrix dominant eigenvector
(Krylov Arnoldi). An `DenseIMPO` can be used directly as the Hamiltonian of
a ground-state search, as well as in `expectationvalue(ψ, W)`.
"""
function DMRGCache(ψ::CanonicalIMPS, operator::DenseIMPO; kwargs...)
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
    # 对标 MPSKit normalize!(::DenseIMPO)：先把每个 GR 做 Frobenius 归一，
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
    DMRGCache(ψ, H::SparseIMPO; tol, maxiter, krylovdim, init_lefts, init_rights)
        -> DMRGCache

Jordan Hamiltonian environments: per-level linear solves (mirroring MPSKit's
`compute_leftenvs!/compute_rightenvs!(::SparseIMPO)`).
`init_lefts`/`init_rights` provide warm-started initial values (see
[`recalculate!`](@ref)).
"""
function DMRGCache(ψ::CanonicalIMPS, H::SparseIMPO;
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

"_transpose_tail(A) / _transpose_front(A): swap the front and back indices
`(Dl, d, Dr) ↔ (Dr, d, Dl)`."
_transpose_tail(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))
_transpose_front(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))

"MPSKit's `_localupdate_sweep_idmrg!`: forward + backward sweep; returns
`(ψ, envs, C_old, E)`."
function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve)
    N = length(ψ)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:N
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = leftorth(ψ.AC[pos], (1, 2), (3,))
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end
    # right to left sweep
    for pos in N:-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.C[pos - 1], ψ.AR[pos] = rightorth(ψ.AC[pos], (1,), (2, 3))
        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end

function find_groundstate(ψ₀::CanonicalIMPS, operator, alg::IDMRG,
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
    ψ′ = CanonicalIMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, ϵ
end
