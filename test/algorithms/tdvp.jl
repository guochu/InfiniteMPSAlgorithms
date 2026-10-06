@testset "TDVP 时间演化（single-site）" begin
    T = ComplexF64
    Hm = heisenberg_hamiltonian(T = T)

    # alg_gauge field：按 VOMPS 惯例取 Defaults.alg_gauge()（(; tol, maxiter)
    # 的 NamedTuple，动态容差下为 DynamicTol 包装）；显式传 NamedTuple 验证
    # keyword 透传（与默认路径轨迹一致到 ~1e-10）
    g0 = TDVP().alg_gauge
    g0 = g0 isa DynamicTol ? g0.alg : g0
    @test g0.tol == Defaults.tolgauge && g0.maxiter == Defaults.maxiter
    alg_g = TDVP(integrator = Defaults.alg_expsolve(),
                 alg_gauge = (; tol = Defaults.tolgauge, maxiter = Defaults.maxiter))

    # 实时间：从基态出发能量守恒
    ψg, envsg, _ = find_groundstate(randomimps(T, [2, 2]; D = 8), Hm,
                                    VUMPS(D = 8, maxiter = 200, tol = 1e-9))
    e0 = real(expectationvalue(ψg, Hm) / 2)
    tspan = 0:0.01:0.1
    ψt, envst, history = time_evolve(ψg, Hm, tspan, TDVP(); observer = (ψ, k, t) -> real(expectationvalue(ψ, Hm) / 2))
    @test length(history) == 11
    @test abs(history[end] - e0) < 1e-6
    @test abs(norm(ψt) - 1) < 1e-8
    ψt2, _, history2 = time_evolve(ψg, Hm, tspan, alg_g;
                                   observer = (ψ, k, t) -> real(expectationvalue(ψ, Hm) / 2))
    @test history2 ≈ history atol = 1e-10

    # 虚时间：收敛到基态能量（阈值 2e-3：无限链 e_exact 与 D = 8 变分基态的
    # 有限键差 ~1e-3 量级，未播种初态的收敛盆地带来波动）
    ψr = randomimps(T, [2, 2]; D = 8)
    tspanβ = 0:0.05:20
    ψβ, _, historyβ = time_evolve(ψr, Hm, tspanβ, TDVP(); imaginary_evolution = true)
    e_exact = 0.25 - log(2)
    @test abs(real(expectationvalue(ψβ, Hm) / 2) - e_exact) < 2e-3

    # 与 WII 演化的一致性（短时间）：能量守恒
    _, bulk = heisenberg_xxz(T = T)
    W2 = make_time_mpo(bulk, 0.01, WII())
    # 严格施加（W * ψ 的规范代表；二参数 mult 已删除，等价表达）
    apply_exact(W, ψ) = CanonicalIMPS((W * DenseIMPS(collect(ψ.AL))).As)
    outψ = apply_exact(W2, ψg)
    @test abs(real(expectationvalue(outψ, Hm) / 2) - e0) < 1e-6
end
