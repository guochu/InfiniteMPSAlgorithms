# ---------------- algorithm definitions (VUMPS / IDMRG / VOMPS / TDVP) ----------------
#
# Parameter objects of the variational ground-state and time-evolution
# algorithms are collected here.

"""
    VUMPS(; D, tol, maxiter, verbosity, alg_gauge, alg_eigsolve, alg_environments, finalize)

Uniform-MPS variational ground-state algorithm (Zaletel–Pollmann /
Vanderstraeten et al.; mirrors MPSKit's `VUMPS`).

`D` is the (reserved) target bond dimension: `find_groundstate` takes the
bond dimension from the provided initial state `ψ₀`, so `alg.D` is currently
ignored by the ground-state drivers.

Each iteration (MPSKit template):
1. `localupdate_step!`: solve the smallest-eigenpair problems of
   `AC_hamiltonian` and `C_hamiltonian` site by site (`fixedpoint`, warm
   started), then `regauge!` to obtain candidate `AL`s;
2. `gauge_step!`: `gaugefix!(ψ, ALs, ψ.C[end]; order = :R)` restores the global
   right gauge, followed by `AC = AL·C`;
3. `envs_step!`: `recalculate!` recomputes the environments;
4. the `finalize` callback; convergence criterion `calc_galerkin ≤ tol`.
"""
@kwdef struct VUMPS{F} <: Algorithm
    D::Union{Nothing,Int} = nothing
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve = Defaults.alg_eigsolve()
    alg_environments = Defaults.alg_environments()
    finalize::F = Defaults._finalize
end

"""
    IDMRG(; D, tol, maxiter, verbosity, alg_gauge, alg_eigsolve)

Single-site infinite DMRG (mirrors MPSKit's `IDMRG`).

`D` is the (reserved) target bond dimension: `find_groundstate` takes the
bond dimension from the provided initial state `ψ₀`, so `alg.D` is currently
ignored by the ground-state drivers (the compression path of `mult` /
`compress` / `hadamard` takes `D` from `alg.D`).

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
"""
@kwdef struct IDMRG{A} <: Algorithm
    D::Union{Nothing,Int} = nothing
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve::A = Defaults.alg_eigsolve()
end

"""
    VOMPS(; D, tol, maxiter, verbosity)

Overlap-maximization algorithm parameters for the iterative
MPO·MPS / MPO·MPO multiplication (named after MPSKit's `VOMPS` family).
The bond dimension is fixed by `D`: `D = nothing` (the default) returns the
exact naive construction without compression; `D::Int` variationally
compresses to bond dimension `D` (overlap maximization over the variational
manifold, consistent with MPSKit). The driver functions (`mult`, `compress`,
`hadamard`) take the bond dimension from `alg.D`; the in-place drivers
(`mult!`, `compress!`, `hadamard!`) take it from the provided initial guess
`out` and ignore `alg.D`.
"""
@kwdef struct VOMPS <: Algorithm
    D::Union{Nothing,Int} = nothing
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
end

"""
    TDVP(; integrator, tolgauge, gaugemaxiter, finalize)

Single-site TDVP time evolution (Haegeman et al.; mirrors MPSKit's `TDVP`).
Infinite-system version: each step evolves all `AC`s and `C`s independently
with the same `dt`, then re-canonicalizes pairwise via `regauge!` and rebuilds
the state with a global `gaugefix!` (right-canonical).
"""
@kwdef struct TDVP{I,F} <: Algorithm
    integrator::I = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
end
