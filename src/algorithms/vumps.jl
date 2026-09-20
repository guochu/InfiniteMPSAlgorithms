# ---------------- VUMPS（严格对标 MPSKit src/algorithms/groundstate/vumps.jl） ----------------

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

"localupdate_step!：逐 site 解 AC/C 子问题并 `regauge!`，返回候选 `AL` 串。"
function localupdate_step!(ψ::MixedCanonicalMPS, operator, envs::Environments,
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

"gauge_step!：候选 `AL` 写入 `ψ.AL` 后 `gaugefix!(; order = :R)`，再 `AC = AL·C`（MPSKit 模板）。"
function gauge_step!(ψ::MixedCanonicalMPS, ALs::Vector, C₀; tol::Real, maxiter::Int)
    for ℓ in eachindex(ALs)
        ψ.AL[ℓ] = ALs[ℓ]
    end
    gaugefix!(ψ, ψ.AL, C₀; order = :R, tol = tol, maxiter = maxiter)
    for ℓ in 1:length(ψ)
        ψ.AC[ℓ] = _mulAL(ψ.AL[ℓ], ψ.C[ℓ])
    end
    return ψ
end

function find_groundstate(ψ₀::MixedCanonicalMPS, operator, alg::VUMPS,
                          envs::Environments = environments(ψ₀, operator);
                          which::Symbol = :SR)
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    alg_envs = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, operator; tol = alg_envs.tol)
    for iter in 1:alg.maxiter
        # MPSKit 第 iter 步的 state.iter = iter-1
        siter = iter - 1
        alg_eigsolve = updatetol(alg.alg_eigsolve, siter, ϵ)
        # localupdate：逐 site 解 AC 与 C 子问题
        ALs = localupdate_step!(ψ, operator, envs, which, alg_eigsolve)
        # gauge：规范恢复（对标 MPSKit 的动态 gauge 容差）
        alg_gauge = updatetol(alg.alg_gauge, siter, ϵ)
        ψ = gauge_step!(ψ, ALs, ψ.C[end]; tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
        # envs（对标 MPSKit 的动态环境容差）
        alg_envs = updatetol(alg.alg_environments, siter, ϵ)
        recalculate!(envs, ψ, operator; tol = alg_envs.tol)
        # finalize
        ψ, envs = alg.finalize(iter, ψ, operator, envs)
        ϵ = calc_galerkin(ψ, operator, envs)
        f = expectation_value(ψ, operator, envs)
        alg.verbosity > 0 && _logiter(stdout, "VUMPS", iter, ϵ, "f" => f)
        ϵ ≤ alg.tol && break
    end
    return ψ, envs, ϵ
end

find_groundstate(ψ₀::MixedCanonicalMPS, operator; kwargs...) =
    find_groundstate(ψ₀, operator, VUMPS(; kwargs...))

"""
    calc_galerkin(ψ, operator, envs) -> Float64

对标 MPSKit `calc_galerkin`：归一化梯度 `x = H_AC(AC)/‖H_AC(AC)‖`，再投影掉
`AL` 规范方向，`ϵ = max_site ‖x − AL·(AL†·x)‖`。
"""
function calc_galerkin(ψ::MixedCanonicalMPS, operator, envs::Environments)
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
