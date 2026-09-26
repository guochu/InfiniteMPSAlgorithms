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

"""
    find_groundstate(ψ₀::CanonicalIMPS, operator::SparseIMPO, alg::VUMPS, [envs]; which = :SR)
        -> (ψ, envs, ϵ)

VUMPS ground-state search. `operator` **必须是 `SparseIMPO`**（哈密顿量的 Schur
形式）：只有它才给出能量泛函的正确收缩（闭列公式 + 非齐次线性解的环境）。
`DenseIMPO` 的周期 trace 期望不是能量（见 [`DMRGCache`](@ref) 的 `DenseIMPO`
版说明），传入会直接报 `ArgumentError`。
"""
function find_groundstate(ψ₀::CanonicalIMPS, operator::SparseIMPO, alg::VUMPS,
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

"""
    find_groundstate(operator::SparseIMPO, alg::Union{VUMPS,IDMRG}, [envs]) -> (ψ, envs, ϵ)

Convenience method without an explicit initial state: `ψ₀` is generated
randomly (`randomimps`) with bond dimension `alg.D`, taking the physical
dimensions and scalar type from `operator`.
"""
function find_groundstate(operator::SparseIMPO, alg::Union{VUMPS,IDMRG},
                          envs::Union{Nothing,Environments} = nothing)
    ψ₀ = randomimps(scalartype(operator), phydims(operator), alg.D)
    envs0 = envs === nothing ? DMRGCache(ψ₀, operator) : envs
    return find_groundstate(ψ₀, operator, alg, envs0)
end

"基态搜索只接受 `SparseIMPO`：`DenseIMPO` 通道的周期 trace 期望不是能量（见
[`DMRGCache`](@ref) 的 `DenseIMPO` 版说明，MPSKit 亦另设专用哈密顿量方法）。
请改用 `SparseIMPO`（如 `tfim_hamiltonian` / `heisenberg_hamiltonian` /
`mpohamiltonian`，或 `tfim()/heisenberg_xxz()` 返回的 `hamiltonian` 字段）。"
_dense_groundstate_error() = throw(ArgumentError(
    "find_groundstate 只支持 SparseIMPO：DenseIMPO 的周期 trace 期望不是能量" *
    "（详见 DMRGCache 的 DenseIMPO 版 docstring）。请改用 SparseIMPO，" *
    "例如 tfim_hamiltonian / heisenberg_hamiltonian / mpohamiltonian 或 " *
    "tfim()/heisenberg_xxz() 的 hamiltonian 字段。"))

### NOTE: 这两个方法只用于给出清晰的报错（比缺方法的 MethodError 好读）
find_groundstate(ψ₀::CanonicalIMPS, operator::DenseIMPO, alg::Union{VUMPS,IDMRG},
                 envs = nothing) = _dense_groundstate_error()
find_groundstate(operator::DenseIMPO, alg::Union{VUMPS,IDMRG},
                 envs = nothing) = _dense_groundstate_error()

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
