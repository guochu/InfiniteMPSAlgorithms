@testset "TDVP 时间演化（single-site）" begin
    T = ComplexF64
    Hm = heisenberg_hamiltonian(T = T)

    # 实时间：从基态出发能量守恒
    ψg, envsg, _ = find_groundstate(random_mps(T, [2, 2], 8), Hm,
                                    VUMPS(maxiter = 200, tol = 1e-9))
    e0 = real(expectation_value(ψg, Hm) / 2)
    tspan = 0:0.01:0.1
    ψt, envst, history = time_evolve(ψg, Hm, tspan, TDVP(); observer = (ψ, k, t) -> real(expectation_value(ψ, Hm) / 2))
    @test length(history) == 11
    @test abs(history[end] - e0) < 1e-6
    @test abs(norm(ψt) - 1) < 1e-8

    # 虚时间：收敛到基态能量
    ψr = random_mps(T, [2, 2], 8)
    tspanβ = 0:0.05:20
    ψβ, _, historyβ = time_evolve(ψr, Hm, tspanβ, TDVP(); imaginary_evolution = true)
    e_exact = 0.25 - log(2)
    @test abs(real(expectation_value(ψβ, Hm) / 2) - e_exact) < 1e-3

    # 与 WII 演化的一致性（短时间）：能量守恒
    _, bulk = heisenberg_xxz(T = T)
    W2 = make_time_mpo(bulk, 0.01, WII())
    outψ, _ = apply(ψg, W2)
    @test abs(real(expectation_value(outψ, Hm) / 2) - e0) < 1e-6
end
