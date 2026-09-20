# ---------------- 算法定义（VUMPS / IDMRG / VOMPS） ----------------
#
# 三个变分算法的参数对象集中定义于此，均带 `trunc` 字段：
# - `trunc`：截断方案，**必须带 `D` 字段**（即 `TruncateDim` / `TruncateDimCutoff`，
#   构造时校验），键维通过 [`bondD`](@ref) 提取；
# - `mult` / `add` / `hadamard` 的压缩路径直接取 `D = bondD(alg.trunc)`；
# - `find_groundstate`（VUMPS / IDMRG）为 single-site 算法，不增长键维（键维由
#   初态决定），trunc 字段在该路径不参与截断。

"截断方案是否带键维 `D` 字段（`TruncateDim` / `TruncateDimCutoff` 为真）。"
hasbondD(trunc::TruncationScheme) = hasfield(typeof(trunc), :D)

"""
    bondD(trunc::TruncationScheme) -> Int

截断方案的键维 `D`。要求 `trunc` 带 `D` 字段（即 `TruncateDim` /
`TruncateDimCutoff`；`NoTruncation` / `TruncateRelError` 不满足），
否则抛出 `ArgumentError`。
"""
function bondD(trunc::TruncationScheme)
    hasbondD(trunc) || throw(ArgumentError(
        "trunc 必须是带 D 字段的 TruncationScheme（TruncateDim / TruncateDimCutoff），收到 $(typeof(trunc))"))
    return trunc.D
end

"""
    VUMPS(; tol, maxiter, verbosity, alg_gauge, alg_eigsolve, alg_environments, finalize, trunc)

均匀 MPS 变分基态算法（Zaletel–Pollmann / Vanderstraeten 等，对标 MPSKit 的 `VUMPS`）。

每轮迭代（MPSKit 模板）：
1. `localupdate_step!`：逐 site 解 `AC_hamiltonian` 与 `C_hamiltonian` 最小本征对
   （`fixedpoint`，热启动），`regauge!` 得到候选 `AL`；
2. `gauge_step!`：`gaugefix!(ψ, ALs, ψ.C[end]; order = :R)` 恢复整体右规范，
   随后 `AC = AL·C`；
3. `envs_step!`：`recalculate!` 重算环境；
4. `finalize` 回调；收敛判据 `calc_galerkin ≤ tol`。

`trunc`：截断方案，必须带 `D` 字段（`TruncateDim` / `TruncateDimCutoff`）。
基态搜索为 single-site 算法、不增长键维（键维由初态决定），trunc 在该路径
不参与截断；`trunc.D` 供 `mult` 等压缩路径取键维。
"""
@kwdef struct VUMPS{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve = Defaults.alg_eigsolve()
    alg_environments = Defaults.alg_environments()
    finalize::F = Defaults._finalize
    trunc::TruncationScheme = truncdim(Defaults.truncD)
    function VUMPS(tol, maxiter, verbosity, alg_gauge, alg_eigsolve, alg_environments,
                   finalize, trunc)
        hasbondD(trunc) || throw(ArgumentError(
            "VUMPS 的 trunc 必须是带 D 字段的 TruncationScheme（TruncateDim / TruncateDimCutoff），收到 $(typeof(trunc))"))
        return new{typeof(finalize)}(tol, maxiter, verbosity, alg_gauge, alg_eigsolve,
                                     alg_environments, finalize, trunc)
    end
end

"""
    IDMRG(; tol, maxiter, verbosity, alg_gauge, alg_eigsolve, trunc)

single-site 无限 DMRG（对标 MPSKit 的 `IDMRG`）。

每轮迭代（MPSKit 模板）：
1. 前向扫描：逐 site 解 `AC_hamiltonian` 最小本征对，`left_orth` 分裂为 `AL/C`，
   `transfer_leftenv!` 增量推进环境；
2. 后向扫描：逐 site 再解 AC，`right_orth` 分裂为 `C/AR`，
   `transfer_rightenv!` 增量推进环境；
3. 收敛判据 `ϵ = ‖C − C_old‖`（取 bond 0 的中心矩阵），能量增量 `ΔE = ΔE_iter/2`。

结束后从 `AR` 重建混合规范态（对标 `InfiniteMPS(mps.AR)`）并重算环境。

`trunc`：截断方案，必须带 `D` 字段（`TruncateDim` / `TruncateDimCutoff`）。
基态搜索为 single-site 算法、不增长键维（键维由初态决定），trunc 在该路径
不参与截断；`trunc.D` 供 `mult` 等压缩路径取键维。
"""
@kwdef struct IDMRG{A} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve::A = Defaults.alg_eigsolve()
    trunc::TruncationScheme = truncdim(Defaults.truncD)
    function IDMRG(tol, maxiter, verbosity, alg_gauge, alg_eigsolve, trunc)
        hasbondD(trunc) || throw(ArgumentError(
            "IDMRG 的 trunc 必须是带 D 字段的 TruncationScheme（TruncateDim / TruncateDimCutoff），收到 $(typeof(trunc))"))
        return new{typeof(alg_eigsolve)}(tol, maxiter, verbosity, alg_gauge, alg_eigsolve, trunc)
    end
end

"""
    VOMPS(; tol, maxiter, verbosity, trunc)

MPO·MPS / MPO·MPO 迭代乘法的重叠最大化算法参数（命名对标 MPSKit 的 `VOMPS` 家族）。
`trunc`：截断方案，必须带 `D` 字段（`TruncateDim` / `TruncateDimCutoff`）——
`mult` / `add` / `hadamard` 的输出键维即 `trunc.D`（朴素构造键维 ≤ D 时
短路返回精确结果）。
"""
@kwdef struct VOMPS <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    trunc::TruncationScheme = truncdim(Defaults.truncD)
    function VOMPS(tol, maxiter, verbosity, trunc)
        hasbondD(trunc) || throw(ArgumentError(
            "VOMPS 的 trunc 必须是带 D 字段的 TruncationScheme（TruncateDim / TruncateDimCutoff），收到 $(typeof(trunc))"))
        return new(tol, maxiter, verbosity, trunc)
    end
end
