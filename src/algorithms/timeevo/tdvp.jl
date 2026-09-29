# ---------------- TDVP (mirrors MPSKit src/algorithms/timestep/tdvp.jl + time_evolve.jl) ----------------
#
# The algorithm definition (the TDVP parameter object) lives in algdefs.jl.

"""
    integrate(H, x, t, dt, alg; imaginary_evolution = false) -> x′

Local time evolution under the effective Hamiltonian `H`:
`x′ = exp(δ·H)·x` with `δ = -im·dt` (real time) or `δ = -dt` (imaginary time).
`alg` is a KrylovKit solver.
"""
function integrate(H, x, t::Number, dt::Number, alg::KrylovKit.KrylovAlgorithm;
                   imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    return exponentiate(H, δ, x; ishermitian = alg isa KrylovKit.Lanczos,
                        tol = alg.tol, krylovdim = alg.krylovdim, maxiter = alg.maxiter)[1]
end

"""
    timestep(ψ, H, t, dt, [alg], [envs]; imaginary_evolution = false) -> (ψ, envs)

Evolve one step of size `dt` at time `t` (solving `i∂ψ/∂t = Hψ`).

**The bond dimension is preserved**: `TDVP` is a single-site integrator, so this
step never changes `bonddim(ψ)`. `ψ` must already carry the bond dimension
required by the evolved state — see [`time_evolve`](@ref) for how to raise it
(`changebond!`, or a few two-site `apply!` steps); a too-small initial bond
dimension makes the result systematically inaccurate for any `dt`.
"""
function timestep(ψ::CanonicalIMPS, H, t::Number, dt::Number, alg::TDVP = TDVP(),
                  envs::Environments = DMRGCache(ψ, H);
                  imaginary_evolution::Bool = false)
    N = length(ψ)
    temp_ACs = Vector{eltype(ψ.AC)}(undef, N)
    temp_Cs = Vector{eltype(ψ.C)}(undef, N)
    for loc in 1:N
        Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
        temp_ACs[loc] = integrate(Hac, ψ.AC[loc], t, dt, alg.integrator; imaginary_evolution)
        Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
        temp_Cs[loc] = integrate(Hc, ψ.C[loc], t, dt, alg.integrator; imaginary_evolution)
    end
    ALs = regauge!(temp_ACs, temp_Cs; alg = Defaults.alg_orth())
    ψ′ = CanonicalIMPS(ALs, ψ.C[end]; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    recalculate!(envs, ψ′, H)
    return ψ′, envs
end

"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; verbosity = 0, imaginary_evolution = false, observer = nothing)
        -> (ψ, envs)

Step through the evolution over the time points `t_span` (mirrors MPSKit's
`time_evolve`). With `imaginary_evolution = true` this is imaginary-time
evolution `exp(-H·dt)`. The `observer(ψ, iter, t)` callback collects data at
each step.

**The bond dimension of `ψ₀` is preserved** (`TDVP` is a single-site
integrator), and the evolution is confined to the MPS manifold of that bond
dimension. Raise the bond dimension *before* calling `time_evolve` whenever the
evolved state needs more than `ψ₀` provides — for example

- `changebond!(ψ₀; D = D₀)`: resizes every bond to `D₀` and re-canonicalizes
  (the grown directions are filled with `noise`, default `1e-10`, which avoids a
  rank-deficient gauge), or
- a few two-site steps `apply!(UnitaryGate(g), ψ₀)` / `apply!(GeneralGate(g), ψ₀)`,
  which grow the bond dimension along the gates' bonds,

then evolve the resulting state. In particular, for imaginary-time cooling to a
thermal state do not start from the bond-dimension-1 infinite-temperature
purification `|I⟩` (nor from a hand-padded copy of it): a single-site integrator
cannot build up correlations, so such a run stays in the product-state manifold
and returns the unphysical mean-field energy regardless of `dt`.

When padding a state to a larger bond dimension by hand, the padded spectrum
must be the rectangular padding `C = Diagonal([1, 0, 0, …])` — the added bond
indices carry zero weight, so the state is unchanged — **not** `C = I`.
`C = I` does not represent the padded state, and its degenerate (replicated)
singular values silently corrupt the truncated two-site updates
([`apply!`](@ref)) that follow.
"""
function time_evolve(ψ₀::CanonicalIMPS, H, t_span::AbstractVector{<:Number},
                     alg::TDVP = TDVP(), envs::Environments = DMRGCache(ψ₀, H);
                     verbosity::Int = 0, imaginary_evolution::Bool = false, observer = nothing)
    ψ = copy(ψ₀)
    if scalartype(ψ) <: Real && (!imaginary_evolution || !isreal(dt_span_diff(t_span)))
        ψ = CanonicalIMPS(PeriodicVector(complex.(parent(ψ.AL))),
                              PeriodicVector(complex.(parent(ψ.AR))),
                              PeriodicVector(complex.(parent(ψ.C))),
                              PeriodicVector(complex.(parent(ψ.AC))))
    end
    history = Any[]
    push_history!(h, obs, ψ, iter, t) =
        push!(h, isnothing(obs) ? expectationvalue(ψ, H, envs) : obs(ψ, iter, t))
    push_history!(history, observer, ψ, 0, t_span[1])
    for iter in 1:(length(t_span) - 1)
        t = t_span[iter]
        dt = t_span[iter+1] - t
        ψ, envs = timestep(ψ, H, t, dt, alg, envs; imaginary_evolution)
        ψ, envs = alg.finalize(t, ψ, H, envs)
        push_history!(history, observer, ψ, iter, t_span[iter+1])
        verbosity > 0 && _logiter(stdout, "TDVP", iter, abs(dt), "t" => t_span[iter+1])
    end
    return ψ, envs, history
end

function dt_span_diff(t_span::AbstractVector{<:Number})
    return length(t_span) > 1 ? t_span[2] - t_span[1] : first(t_span)
end
