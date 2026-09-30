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
    find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                     alg::VUMPS, [envs]; which = :SR) -> (ψ, envs, info)

VUMPS ground-state search. `operator` 两种形式皆可（对标 MPSKit 的
VUMPS 同时接受 `InfiniteMPOHamiltonian`/`InfiniteMPO`）：

- `SparseIMPO`（哈密顿量的 Schur 形式）：闭列公式 + 非齐次线性解的环境——
  **能量泛函的正确收缩，求能量用它**；
- `DenseIMPO`：周期 trace 完整收缩（转移矩阵主本征向量环境，MPSKit
  `InfiniteMPO` 通道的对标实现）——Hamiltonian 型 MPO 的该期望含恒等层
  bookkeeping，不是真实能量（见 [`DMRGCache`](@ref) 的 `DenseIMPO` 版说明；
  与 MPSKit `expectation_value(ψ, ::InfiniteMPO)` 逐位同约定）。

The third return is the [`IterativeConvergenceInfo`](@ref)（`niter` 迭代轮数、
`losses` = [初始 Galerkin 残差, 逐轮 Galerkin 残差...]、`converged` 收敛标志）。
"""
function find_groundstate(ψ₀::CanonicalIMPS, operator::Union{SparseIMPO,DenseIMPO},
                          alg::VUMPS,
                          envs::Environments = DMRGCache(ψ₀, operator);
                          which::Symbol = :SR)
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    alg_envs = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, operator; tol = alg_envs.tol)
    losses = Float64[ϵ]
    converged = false
    iter = 0
    for outer iter in 1:alg.maxiter
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
        push!(losses, ϵ)
        f = expectationvalue(ψ, operator, envs)
        alg.verbosity > 0 && _logiter(stdout, "VUMPS", iter, ϵ, "f" => f)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    return ψ, envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    find_groundstate(operator::Union{SparseIMPO,DenseIMPO},
                     alg::Union{VUMPS,IDMRG}, [envs]) -> (ψ, envs, info)

Convenience method without an explicit initial state: `ψ₀` is generated
randomly (`randomimps`) with bond dimension `alg.D`, taking the physical
dimensions and scalar type from `operator`.
"""
function find_groundstate(operator::Union{SparseIMPO,DenseIMPO},
                          alg::Union{VUMPS,IDMRG},
                          envs::Union{Nothing,Environments} = nothing)
    ψ₀ = randomimps(scalartype(operator), phydims(operator); D = alg.D)
    envs0 = envs === nothing ? DMRGCache(ψ₀, operator) : envs
    return find_groundstate(ψ₀, operator, alg, envs0)
end

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
