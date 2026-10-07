# ---------------- TDVP (mirrors MPSKit src/algorithms/timestep/tdvp.jl + time_evolve.jl) ----------------
#
# The algorithm definition (the TDVP parameter object) lives in algdefs.jl.

"""
    integrate(H, x, dt, alg) -> x′

Local time evolution under the effective Hamiltonian `H`: `x′ = exp(dt·H)·x`
——**`dt` 即指数的系数本身**（不做任何 `-im` 换算）：实时演化直接输入
`dt = -im·t`（`exp(-i·H·t)`），虚时演化（imaginary-time cooling）输入
`dt = -τ`（τ > 0，`exp(-H·τ)`）。`alg` is a KrylovKit solver; exponentiate
未收敛时 `@warn`。
"""
function integrate(H, x, dt::Number, alg::KrylovKit.KrylovAlgorithm)
    x′, info = exponentiate(H, dt, x; ishermitian = alg isa KrylovKit.Lanczos,
                            tol = alg.tol, krylovdim = alg.krylovdim, maxiter = alg.maxiter)
    info.converged == 0 &&
        @warn "TDVP integrate: exponentiate not converged" normres = info.normres dt
    return x′
end

"""
    timestep(ψ, H, dt, [alg], [envs]) -> (ψ, envs)

Evolve one step with coefficient `dt`（见 [`integrate`](@ref)）：实时演化
`dt = -im·t`，虚时演化 `dt = -τ`（τ > 0）。

**The bond dimension is preserved**: `TDVP` is a single-site integrator, so this
step never changes `bonddim(ψ)`. `ψ` must already carry the bond dimension
required by the evolved state — see [`time_evolve`](@ref) for how to raise it
(`changebond!`, or a few two-site `apply!` steps); a too-small initial bond
dimension makes the result systematically inaccurate for any `dt`.
"""
function timestep(ψ::CanonicalIMPS, H, dt::Number, alg::TDVP = TDVP(),
                  envs::Environments = DMRGCache(ψ, H))
    N = length(ψ)
    temp_ACs = Vector{eltype(ψ.AC)}(undef, N)
    temp_Cs = Vector{eltype(ψ.C)}(undef, N)
    for loc in 1:N
        Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
        temp_ACs[loc] = integrate(Hac, ψ.AC[loc], dt, alg.integrator)
        Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
        temp_Cs[loc] = integrate(Hc, ψ.C[loc], dt, alg.integrator)
    end
    ALs = regauge!(temp_ACs, temp_Cs; alg = alg.alg_orth)
    # gauge 参数按 VOMPS 惯例存于 alg_gauge（(; tol, maxiter) 或 DynamicTol 包装），
    # 以 keyword 形式喂给收尾的右规范化 gaugefix!（经 2 参构造器透传，
    # `order = :R` 与 MPSKit 的 InfiniteMPS(AL, C₀) 重建一致）
    g = alg.alg_gauge isa DynamicTol ? alg.alg_gauge.alg : alg.alg_gauge
    ψ′ = CanonicalIMPS(ALs, ψ.C[end]; tol = g.tol, maxiter = g.maxiter)
    recalculate!(envs, ψ′, H)
    return ψ′, envs
end

"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; observer = nothing) -> (ψ, envs, history)

Step through the evolution over the points `t_span`（mirrors MPSKit's
`time_evolve`）：每步系数 `dt = t_span[iter+1] − t_span[iter]` **即指数的系数
本身**（见 [`integrate`](@ref)）——实时演化输入纯虚步长（如
`t_span = (-im) .* (0:0.01:1)`，`dt = -im·0.01` ⇒ `exp(-i·H·0.01)`），虚时
演化输入负实步长（如 `t_span = -(0:0.05:20)`，`dt = -0.05` ⇒
`exp(-H·0.05)`，cooling）。实数链在实时演化（非实 `dt`）时自动升复，虚时
演化保持实数域。迭代日志由 `alg.verbosity` 控制；`observer(ψ, iter, t)`
callback collects data at each step.

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
                     observer = nothing)
    ψ = copy(ψ₀)
    # 实数链 + 实时演化（dt 非实，即 t_span 为纯虚步长）→ 升复；虚时（dt 负实）
    # 保持实数域
    if scalartype(ψ) <: Real && !isreal(dt_span_diff(t_span))
        ψ = complex(ψ)
    end
    history = Any[]
    push_history!(h, obs, ψ, iter, t) =
        push!(h, isnothing(obs) ? expectationvalue(ψ, H, envs) : obs(ψ, iter, t))
    push_history!(history, observer, ψ, 0, t_span[1])
    for iter in 1:(length(t_span) - 1)
        dt = t_span[iter+1] - t_span[iter]
        ψ, envs = timestep(ψ, H, dt, alg, envs)
        ψ, envs = alg.finalize(t_span[iter], ψ, H, envs)
        push_history!(history, observer, ψ, iter, t_span[iter+1])
        alg.verbosity > 0 &&
            _logiter(stdout, "TDVP", iter, abs(dt), "t" => t_span[iter+1])
    end
    return ψ, envs, history
end

function dt_span_diff(t_span::AbstractVector{<:Number})
    return length(t_span) > 1 ? t_span[2] - t_span[1] : first(t_span)
end
