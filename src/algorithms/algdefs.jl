# ---------------- 算法定义（VUMPS / IDMRG / VOMPS / TDVP） ----------------
#
# 变分基态与时间演化算法的参数对象集中定义于此。

"""
    VUMPS(; tol, maxiter, verbosity, alg_gauge, alg_eigsolve, alg_environments, finalize)

均匀 MPS 变分基态算法（Zaletel–Pollmann / Vanderstraeten 等，对标 MPSKit 的 `VUMPS`）。

每轮迭代（MPSKit 模板）：
1. `localupdate_step!`：逐 site 解 `AC_hamiltonian` 与 `C_hamiltonian` 最小本征对
   （`fixedpoint`，热启动），`regauge!` 得到候选 `AL`；
2. `gauge_step!`：`gaugefix!(ψ, ALs, ψ.C[end]; order = :R)` 恢复整体右规范，
   随后 `AC = AL·C`；
3. `envs_step!`：`recalculate!` 重算环境；
4. `finalize` 回调；收敛判据 `calc_galerkin ≤ tol`。
"""
@kwdef struct VUMPS{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve = Defaults.alg_eigsolve()
    alg_environments = Defaults.alg_environments()
    finalize::F = Defaults._finalize
end

"""
    IDMRG(; tol, maxiter, verbosity, alg_gauge, alg_eigsolve)

single-site 无限 DMRG（对标 MPSKit 的 `IDMRG`）。

每轮迭代（MPSKit 模板）：
1. 前向扫描：逐 site 解 `AC_hamiltonian` 最小本征对，`left_orth` 分裂为 `AL/C`，
   `transfer_leftenv!` 增量推进环境；
2. 后向扫描：逐 site 再解 AC，`right_orth` 分裂为 `C/AR`，
   `transfer_rightenv!` 增量推进环境；
3. 收敛判据 `ϵ = ‖C − C_old‖`（取 bond 0 的中心矩阵），能量增量 `ΔE = ΔE_iter/2`。

结束后从 `AR` 重建混合规范态（对标 `InfiniteMPS(mps.AR)`）并重算环境。
"""
@kwdef struct IDMRG{A} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve::A = Defaults.alg_eigsolve()
end

"""
    VOMPS(; tol, maxiter, verbosity)

MPO·MPS / MPO·MPO 迭代乘法的重叠最大化算法参数（命名对标 MPSKit 的 `VOMPS` 家族）。
键维由初态决定（变分流形上最大化重叠，与 MPSKit 一致）。
"""
@kwdef struct VOMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
end

"""
    TDVP(; integrator, tolgauge, gaugemaxiter, finalize)

single-site TDVP 时间演化（Haegeman et al.，对标 MPSKit 的 `TDVP`）。
无限系统版本：每步将所有 `AC` 与 `C` 用同一个 `dt` 独立演化，随后
`regauge!` 成对重新规范并整体 `gaugefix!`（右规范）重建状态。
"""
@kwdef struct TDVP{I,F} <: Algorithm
    integrator::I = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
end
