# ---------------- IDMRG（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，single-site） ----------------
#
# 算法定义（IDMRG 参数对象）见 algdefs.jl；哈密顿量环境缓存（DMRGCache）的
# 定义与构造器见 groundstates/envs.jl——本文件只剩 IDMRG 算法本体。

"MPSKit's `_localupdate_sweep_idmrg!`: forward + backward sweep; returns
`(ψ, envs, C_old, E)`."
function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve, alg_orth)
    N = length(ψ)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:N
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = _leftsplit(ψ.AC[pos], alg_orth)
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end
    # right to left sweep
    for pos in N:-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.C[pos - 1], ψ.AR[pos] = _rightsplit(ψ.AC[pos], alg_orth)
        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end

"""
    find_groundstate(ψ₀::CanonicalIMPS, operator::SparseIMPO, alg::IDMRG, [envs])
        -> (ψ, envs, info)

IDMRG ground-state search. `operator` **必须是 `SparseIMPO`**（同
[`find_groundstate`](@ref) 的 VUMPS 版说明：`DenseIMPO` 的周期 trace 期望不是
能量，传入会报 `ArgumentError`）。The third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 迭代轮数、`losses` = [初始 Galerkin
残差, 逐轮 bond-0 中心矩阵漂移...]、`converged` 收敛标志）。
"""
function find_groundstate(ψ₀::CanonicalIMPS, operator::SparseIMPO, alg::IDMRG,
                          envs::Environments = DMRGCache(ψ₀, operator))
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    E = expectationvalue(ψ, operator, envs)
    alg.verbosity > 0 && _logiter(stdout, "IDMRG", 0, ϵ, "f" => E)
    iter = 0
    losses = Float64[ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # MPSKit 第 iter 次扫描时 state.iter = iter-1
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter - 1, ϵ)
        ψ, envs, C_old, E_new = _localupdate_sweep_idmrg!(ψ, operator, envs,
                                                          alg_eigsolve, alg.alg_orth)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        ψ, envs = alg.finalize(iter, ψ, operator, envs)
        # error criterion（bond 0 中心矩阵之差）
        ϵ = norm(ψ.C[0] - C_old)
        push!(losses, ϵ)
        # new energy
        ΔE = (E_new - E) / 2
        E = E_new
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ, "f" => E, "ΔE" => ΔE)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建（对标 MPSKit 的 `InfiniteMPS(mps.AR)`，容差取
    # alg_gauge 的动态适配）
    N = length(ψ)
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = CanonicalIMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, IterativeConvergenceInfo(iter, losses, converged)
end
