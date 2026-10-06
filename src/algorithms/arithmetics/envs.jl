# ---------------- 压缩/代数通道的环境缓存抽象 ----------------
#
# VOMPS/IDMRG 压缩引擎（mult / hadamard / compress 共用）的环境缓存层次：
# OverlapCache（恒等通道）、MultCache（mpo·mps 施加 / mpo·mpo 组合通道）、
# HadamardCache（zip 通道）——与哈密顿量通道的 DMRGCache
# （groundstates/envs.jl）相区分。

abstract type CompressionEnvironments <: Environments end

"压缩通道缓存的站数（`Environments` 的 `length` 契约）：输入单胞长度的最小
公倍数——构造时已保证 bra 与 ket 同长、operator（若存在）长度为其因子，故
= `length(envs.bra)`。"
Base.length(envs::CompressionEnvironments) = length(envs.bra)

"压缩通道缓存标量类型（环境张量的实际 eltype——实输入下复提升后的通道算术
类型；构造器已把各槽位提升到该类型）。"
scalartype(envs::CompressionEnvironments) = scalartype(envs.lefts[1])

# ---------------- 统一压缩扫掠引擎（compress / mult / hadamard 共用） ----------------

"""
    compression_sweeps!(envs::CompressionEnvironments, alg::VOMPS) -> (envs, info)

压缩通道的统一变分扫掠引擎（`compress`/`mult`/`hadamard` 及其 in-place 版本的
共享底层；MPSKit `approximate` 的 `IterativeSolver` 管道）：被优化的态即
`envs.bra`（构造缓存的初态），目标由缓存持有——OverlapCache = ket（variational
compression）、MultCache = operator·ket（mpo·mps / mpo·mpo）、HadamardCache =
zip(ket1, ket2)。`alg::VOMPS` 跑 Jacobi 式 ALS 轮：`localupdate`
（`_local_AC`/`_local_C` → `regauge!`）→ `gauge_step!` → `recalculate!`
环境热启动重解 → 扫掠后检查 Galerkin 残差。全部原地：态写回 `envs.bra`、
环境写回 `lefts`/`rights`，返回 `(envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = [初始残差,
逐轮 Galerkin 残差...]、`converged` 收敛标志）。返回时 `envs.bra` 已按包约定
归一化且处于混合规范——调用方无需再做任何收尾。
"""
function compression_sweeps!(envs::CompressionEnvironments, alg::VOMPS)
    x = envs.bra                       # 原地演化的态（缓存 bra 本体）
    N = length(envs)
    T = scalartype(envs)
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = calc_galerkin(envs, x)
    iter = 0
    losses = [ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # similar 仅提供 eltype 模板，形状由 regauge! 的输出决定）
        ALs = [similar(x.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            ALs[ℓ] = regauge!(_local_AC(envs, ℓ), _local_C(envs, ℓ);
                              alg = alg.alg_orth)
        end
        # gauge: restore the global right gauge（动态容差）
        alg_g = updatetol(alg.alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（bra 更新 + 热启动 + 动态环境容差）
        alg_envs = updatetol(alg.alg_environments, iter - 1, ϵ)
        recalculate!(envs, x, alg_envs)
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        ϵ = calc_galerkin(envs, x)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    _global_normalize!(x)
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG) -> (envs, info)

统一扫掠引擎的 IDMRG 模板（MPSKit `approximate(ψ₀, ..., IDMRG())`）：sequential
Gauss–Seidel double sweep with on-the-fly environment transfer
([`transfer_leftenv!`](@ref)/[`transfer_rightenv!`](@ref)), `leftorth`/
`rightorth` splits of the normalized local projections, per-double-sweep
environment rescaling ([`normalize_envs!`](@ref)) and center-matrix-drift
convergence `ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is
rebuilt from the `AR` string（[`_rebuild`](@ref)，容差取 `alg_gauge` 的动态
适配）and the environments are re-solved for the final state
([`recalculate!`](@ref))。全部原地：态写回 `envs.bra`、环境写回
`lefts`/`rights`，返回 `(envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = 逐轮中心
矩阵漂移、`converged` 收敛标志）。返回时 `envs.bra` 已按包约定归一化且处于
混合规范——调用方无需再做任何收尾。
"""
function compression_sweeps!(envs::CompressionEnvironments, alg::IDMRG)
    x = envs.bra                       # 原地演化的态（缓存 bra 本体）
    N = length(envs)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _local_AC(envs, ℓ)
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            transfer_leftenv!(envs, x, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _local_AC(envs, ℓ)
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            transfer_rightenv!(envs, x, ℓ - 1)
        end
        # 环境重标定
        normalize_envs!(envs, x)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义；原地契约——态写回缓存 bra）
        x, envs = alg.finalize(iter, x, _finalize_target(envs), envs)
        envs.bra ≡ x || _copyinto!(envs.bra, x)
        x = envs.bra
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建混合规范（_rebuild，容差取 alg_gauge 的动态适配），
    # 环境对终态重解
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = _rebuild([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol,
                 maxiter = alg_g.maxiter)
    _global_normalize!(x)
    recalculate!(envs, x, alg.alg_environments)
    return envs, IterativeConvergenceInfo(iter, losses, converged)
end
