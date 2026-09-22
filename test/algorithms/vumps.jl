@testset "VUMPS 基态" begin
    T = ComplexF64

    # XXZ Δ=1（Heisenberg，Schur 哈密顿量）：e₀ = 1/4 − ln2
    # 阈值覆盖随机初态的亚稳态收敛（与 MPSKit 同 iter/同样的逐位变分极限）
    Hxxz = heisenberg_hamiltonian(T = T)
    ψ0 = randomimps(T, [2, 2], 12)
    ψ, envs, ϵ = find_groundstate(ψ0, Hxxz, VUMPS(maxiter = 300, tol = 1e-10, verbosity = 0))
    e = real(expectationvalue(ψ, Hxxz, envs) / 2)
    e_exact = 0.25 - log(2)
    @test ϵ < 1e-9
    @test abs(e - e_exact) < 2e-4

    # TFIM 临界点：e₀ = −4/π
    Htfim = tfim_hamiltonian(T = T)
    ψ2, envs2, ϵ2 = find_groundstate(randomimps(T, [2, 2], 12), Htfim,
                                     VUMPS(maxiter = 300, tol = 1e-10))
    @test abs(real(expectationvalue(ψ2, Htfim, envs2) / 2) + 4 / pi) < 1e-6

    # 熵的量级：临界 Heisenberg
    S = entropy(ψ)
    @test 0.5 < S < 2.0
end
