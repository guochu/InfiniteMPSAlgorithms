# ---------------- TDVP（对标 MPSKit src/algorithms/timestep/tdvp.jl + time_evolve.jl） ----------------

"""
    integrate(H, x, t, dt, alg; imaginary_evolution = false) -> x′

对有效哈密顿量 `H` 做局部时间演化：`x′ = exp(δ·H)·x`，
`δ = -im·dt`（实时间）或 `δ = -dt`（虚时间）。`alg` 为 KrylovKit 求解器。
"""
function integrate(H, x, t::Number, dt::Number, alg::KrylovKit.KrylovAlgorithm;
                   imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    return exponentiate(H, δ, x; ishermitian = alg isa KrylovKit.Lanczos,
                        tol = alg.tol, krylovdim = alg.krylovdim, maxiter = alg.maxiter)[1]
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

"""
    timestep(ψ, H, t, dt, [alg], [envs]; imaginary_evolution = false) -> (ψ, envs)

以步长 `dt` 在时刻 `t` 演化一步（解 `i∂ψ/∂t = Hψ`）。
"""
function timestep(ψ::InfiniteCanonicalMPS, H, t::Number, dt::Number, alg::TDVP = TDVP(),
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
    ψ′ = InfiniteCanonicalMPS(ALs, ψ.C[end]; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    recalculate!(envs, ψ′, H)
    return ψ′, envs
end

"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; verbosity = 0, imaginary_evolution = false, observer = nothing)
        -> (ψ, envs)

在时间点序列 `t_span` 上逐步演化（对标 MPSKit 的 `time_evolve`）。
`imaginary_evolution = true` 时为虚时间演化 `exp(-H·dt)`。
`observer(ψ, iter, t)` 回调逐步收集数据。
"""
function time_evolve(ψ₀::InfiniteCanonicalMPS, H, t_span::AbstractVector{<:Number},
                     alg::TDVP = TDVP(), envs::Environments = DMRGCache(ψ₀, H);
                     verbosity::Int = 0, imaginary_evolution::Bool = false, observer = nothing)
    ψ = copy(ψ₀)
    if scalartype(ψ) <: Real && (!imaginary_evolution || !isreal(dt_span_diff(t_span)))
        ψ = InfiniteCanonicalMPS(PeriodicVector(complex.(parent(ψ.AL))),
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
