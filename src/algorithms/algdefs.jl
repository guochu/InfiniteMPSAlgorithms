# ---------------- algorithm definitions (VUMPS / IDMRG / VOMPS / TDVP) ----------------
#
# Parameter objects of the variational ground-state and time-evolution
# algorithms are collected here.

"""
    VUMPS(; D = Defaults.D, tol, maxiter, verbosity, alg_gauge, alg_eigsolve, alg_environments, finalize)

Uniform-MPS variational ground-state algorithm (Zaletel–Pollmann /
Vanderstraeten et al.; mirrors MPSKit's `VUMPS`). `D` (Int, default
`Defaults.D`) is the bond dimension: `find_groundstate(operator, alg)`
generates a random initial state with `bonddim = D`; when an initial state
`ψ₀` is passed explicitly, `alg.D` is ignored and `ψ₀`'s bond profile is used.
`operator` 必须是 `SparseIMPO`（哈密顿量 Schur 形式）——`DenseIMPO` 的周期 trace
期望不是能量，会被拒绝。

Each iteration (MPSKit template):
1. `localupdate_step!`: solve the smallest-eigenpair problems of
   `AC_hamiltonian` and `C_hamiltonian` site by site (`fixedpoint`, warm
   started), then `regauge!` to obtain candidate `AL`s;
2. `gauge_step!`: `gaugefix!(ψ, ALs, ψ.C[end]; order = :R)` restores the global
   right gauge, followed by `AC = AL·C`;
3. `envs_step!`: `recalculate!` recomputes the environments;
4. the `finalize` callback; convergence criterion `calc_galerkin ≤ tol`.
"""
@kwdef struct VUMPS{F,G,E,N} <: Algorithm
    D::Int = Defaults.D
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge::G = Defaults.alg_gauge()
    alg_eigsolve::E = Defaults.alg_eigsolve()
    alg_environments::N = Defaults.alg_environments()
    finalize::F = Defaults._finalize
end

"""
    IDMRG(; D = Defaults.D, tol, maxiter, verbosity, alg_gauge, alg_environments,
          alg_eigsolve, alg_orth, finalize)

Single-site infinite DMRG (mirrors MPSKit's `IDMRG`). `D` (Int, default
`Defaults.D`) is the bond dimension: `find_groundstate(operator, alg)`
generates a random initial state with `bonddim = D`; when an initial state
`ψ₀` is passed explicitly, `alg.D` is ignored and `ψ₀`'s bond profile is used
(the compression path of `mult` / `compress` / `hadamard` takes `D` from
`alg.D`). `operator` 必须是 `SparseIMPO`（同 `VUMPS`）。

Each iteration (MPSKit template):
1. forward sweep: solve the `AC_hamiltonian` smallest-eigenpair problem site by
   site, split via `left_orth` into `AL/C`, and push the environments
   incrementally with `transfer_leftenv!`;
2. backward sweep: solve the AC problem again site by site, split via
   `right_orth` into `C/AR`, and push the environments incrementally with
   `transfer_rightenv!`;
3. convergence criterion `ϵ = ‖C − C_old‖` (the center matrix on bond 0), with
   energy increment `ΔE = ΔE_iter/2`.

Afterwards the mixed-canonical state is rebuilt from `AR` (mirroring
`InfiniteMPS(mps.AR)`) and the environments are recomputed.

`alg_orth`（正交分解 `leftorth`/`rightorth` 的因式化算法）随 `alg` 传入各
IDMRG 引擎；`alg_environments`（环境求解算法，`fixedpoint` 分派，同
`VUMPS`/`VOMPS` 的同名字段）提供扫掠的初始/收尾环境求解容差；`alg_gauge`
专用于收尾的 AR 混合规范重建（动态容差经 `updatetol(alg_gauge, iter, ϵ)`
适配，对标 MPSKit `InfiniteMPS(mps.AR)`）——IDMRG 扫掠内无规范固定步，故
`alg_gauge` 仅在收尾重建使用；
`finalize(iter, ψ, operator, envs) -> (ψ, envs)` 为逐迭代回调
（默认恒等，`Defaults._finalize`，同 `VUMPS`/`TDVP`）。
"""
@kwdef struct IDMRG{G,N,A,O,F} <: Algorithm
    D::Int = Defaults.D
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge::G = Defaults.alg_gauge()
    alg_environments::N = Defaults.alg_environments()
    alg_eigsolve::A = Defaults.alg_eigsolve()
    alg_orth::O = Defaults.alg_orth()
    finalize::F = Defaults._finalize
end

"""
    VOMPS(; D = Defaults.D, tol, maxiter, verbosity, alg_gauge, alg_environments,
          alg_orth, finalize)

Overlap-maximization algorithm parameters for the iterative
MPO·MPS / MPO·MPO multiplication (named after MPSKit's `VOMPS` family).
`D` (Int, default `Defaults.D`) fixes the bond dimension of the variational
compression (overlap maximization over the variational manifold, consistent
with MPSKit). The driver functions (`mult`, `compress`,
`hadamard`) take the bond dimension from `alg.D`; the in-place drivers
(`mult!`, `compress!`, `hadamard!`) take it from the provided initial guess
`out` and ignore `alg.D`. The strict compression-free constructions live as
the typed operators `Base.:*(::DenseIMPO, ::DenseIMPO)` /
`Base.:*(::DenseIMPO, ::DenseIMPS)` and
`⊙(::DenseIMPS, ::DenseIMPS)` (see `states/linalg.jl` and
`operators/linalg.jl`).

`alg_gauge`/`alg_environments`（动态容差规范固定 / 环境重解，MPSKit
`adapt_solver` 语义）与 `alg_orth`（候选 `AL` 的 `regauge!` 正交化算法）随
`alg` 传入各变分引擎；`finalize(iter, x, operator, envs) -> (x, envs)` 为
逐迭代回调（默认恒等，`Defaults._finalize`）。
"""
@kwdef struct VOMPS{G,N,O,F} <: Algorithm
    D::Int = Defaults.D
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge::G = Defaults.alg_gauge()
    alg_environments::N = Defaults.alg_environments()
    alg_orth::O = Defaults.alg_orth()
    finalize::F = Defaults._finalize
end

"""
    TDVP(; integrator, alg_gauge, alg_orth, verbosity, finalize)

Single-site TDVP time evolution (Haegeman et al.; mirrors MPSKit's `TDVP`).
Infinite-system version: each step evolves all `AC`s and `C`s independently
with the same `dt`, then re-canonicalizes pairwise via `regauge!` and rebuilds
the state with a global `gaugefix!` (right-canonical). `alg_gauge` follows the
`VUMPS`/`VOMPS` convention（`Defaults.alg_gauge()`：`(; tol, maxiter)` 或其
`DynamicTol` 包装），其 `tol`/`maxiter` 以 keyword 形式喂给收尾的 `gaugefix!`；
`alg_orth` is the QR/LQ factorization algorithm of the pairwise `regauge!`;
`verbosity` 控制迭代日志（`time_evolve` 的逐轮 `_logiter`，`> 0` 打印）。
"""
@kwdef struct TDVP{I,G,O,F} <: Algorithm
    integrator::I = Defaults.alg_expsolve()
    alg_gauge::G = Defaults.alg_gauge()
    alg_orth::O = Defaults.alg_orth()
    verbosity::Int = Defaults.verbosity
    finalize::F = Defaults._finalize
end

struct IterativeConvergenceInfo
	niter::Int
	converged::Bool
	losses::Vector{Float64}
	itererr::Float64
	residual::Union{Float64, Nothing}
end

function IterativeConvergenceInfo(niter::Int, losses::Vector{Float64}, converged::Bool; residual::Union{Float64, Nothing} = nothing)
	# IDMRG 模板的 losses 逐轮记录（maxiter = 0 时为空）
	itererr = isempty(losses) ? NaN : losses[end]
	return IterativeConvergenceInfo(niter, converged, losses, itererr, residual)
end
# since we may have considered initial loss
IterativeConvergenceInfo(losses::Vector{Float64}, converged::Bool; kwargs...) = IterativeConvergenceInfo(length(losses) - 1, losses, converged, kwargs...)

