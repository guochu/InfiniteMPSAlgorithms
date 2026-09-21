# ---------------- VUMPS (strictly mirrors MPSKit src/algorithms/groundstate/vumps.jl) ----------------
#
# The algorithm definition (the VUMPS parameter object) lives in algdefs.jl.

"localupdate_step!: solve the AC/C subproblems site by site and `regauge!`;
returns the candidate `AL` string."
function localupdate_step!(ψ::CanonicalIMPS, operator, envs::Environments,
                           which::Symbol, alg_eigsolve)
    N = length(ψ)
    ALs = Vector{eltype(ψ.AL)}(undef, N)
    for site in 1:N
        Hac = AC_hamiltonian(site, ψ, operator, ψ, envs)
        _, AC = fixedpoint(Hac, ψ.AC[site], which, alg_eigsolve)
        Hc = C_hamiltonian(site, ψ, operator, ψ, envs)
        _, C = fixedpoint(Hc, ψ.C[site], which, alg_eigsolve)
        ALs[site] = regauge!(AC, C; alg = Defaults.alg_orth())
    end
    return ALs
end

"gauge_step!: write the candidate `AL` string into `ψ.AL`, restore the global
right gauge via `gaugefix!(; order = :R)`, then `AC = AL·C` (MPSKit template)."
function gauge_step!(ψ::CanonicalIMPS, ALs::Vector, C₀; tol::Real, maxiter::Int)
    for ℓ in eachindex(ALs)
        ψ.AL[ℓ] = ALs[ℓ]
    end
    gaugefix!(ψ, ψ.AL, C₀; order = :R, tol = tol, maxiter = maxiter)
    for ℓ in 1:length(ψ)
        ψ.AC[ℓ] = _mulAL(ψ.AL[ℓ], ψ.C[ℓ])
    end
    return ψ
end

function find_groundstate(ψ₀::CanonicalIMPS, operator, alg::VUMPS,
                          envs::Environments = DMRGCache(ψ₀, operator);
                          which::Symbol = :SR)
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    alg_envs = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, operator; tol = alg_envs.tol)
    for iter in 1:alg.maxiter
        # at MPSKit's iteration `iter`, state.iter = iter - 1
        siter = iter - 1
        alg_eigsolve = updatetol(alg.alg_eigsolve, siter, ϵ)
        # localupdate: solve the AC and C subproblems site by site
        ALs = localupdate_step!(ψ, operator, envs, which, alg_eigsolve)
        # gauge: gauge restoration (mirrors MPSKit's dynamic gauge tolerance)
        alg_gauge = updatetol(alg.alg_gauge, siter, ϵ)
        ψ = gauge_step!(ψ, ALs, ψ.C[end]; tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
        # envs (mirrors MPSKit's dynamic environment tolerance)
        alg_envs = updatetol(alg.alg_environments, siter, ϵ)
        recalculate!(envs, ψ, operator; tol = alg_envs.tol)
        # finalize
        ψ, envs = alg.finalize(iter, ψ, operator, envs)
        ϵ = calc_galerkin(ψ, operator, envs)
        f = expectationvalue(ψ, operator, envs)
        alg.verbosity > 0 && _logiter(stdout, "VUMPS", iter, ϵ, "f" => f)
        ϵ ≤ alg.tol && break
    end
    return ψ, envs, ϵ
end

find_groundstate(ψ₀::CanonicalIMPS, operator; kwargs...) =
    find_groundstate(ψ₀, operator, VUMPS(; kwargs...))

"""
    calc_galerkin(ψ, operator, envs) -> Float64

Mirrors MPSKit's `calc_galerkin`: normalize the gradient
`x = H_AC(AC)/‖H_AC(AC)‖`, project out the `AL` gauge direction, and take
`ϵ = max_site ‖x − AL·(AL†·x)‖`.
"""
function calc_galerkin(ψ::CanonicalIMPS, operator, envs::Environments)
    N = length(ψ)
    ϵ = 0.0
    for site in 1:N
        Hac = AC_hamiltonian(site, ψ, operator, ψ, envs)
        x = Hac(ψ.AC[site])
        x ./= norm(x)
        AL = ψ.AL[site]
        @tensor q[c, b] := conj(AL[a, s, c]) * x[a, s, b]
        @tensor p[a, s, b] := AL[a, s, c] * q[c, b]
        ϵ = max(ϵ, norm(x .- p))
    end
    return ϵ
end
