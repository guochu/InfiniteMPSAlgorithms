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
