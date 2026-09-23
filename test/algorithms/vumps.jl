@testset "VUMPS 基态" begin
    T = ComplexF64

    # XXZ Δ=1（Heisenberg，Schur 哈密顿量）：e₀ = 1/4 − ln2
    # 阈值覆盖随机初态的亚稳态收敛（与 MPSKit 同 iter/同样的逐位变分极限）
    Hxxz = heisenberg_hamiltonian(T = T)
    ψ0 = randomimps(T, [2, 2], 12)
    ψ, envs, ϵ = find_groundstate(ψ0, Hxxz, VUMPS(D = 12, maxiter = 300, tol = 1e-10, verbosity = 0))
    e = real(expectationvalue(ψ, Hxxz, envs) / 2)
    e_exact = 0.25 - log(2)
    @test ϵ < 1e-9
    @test abs(e - e_exact) < 2e-4

    # TFIM 临界点：e₀ = −4/π
    Htfim = tfim_hamiltonian(T = T)
    ψ2, envs2, ϵ2 = find_groundstate(randomimps(T, [2, 2], 12), Htfim,
                                     VUMPS(D = 12, maxiter = 300, tol = 1e-10))
    @test abs(real(expectationvalue(ψ2, Htfim, envs2) / 2) + 4 / pi) < 1e-6

    # 熵的量级：临界 Heisenberg
    S = entropy(ψ)
    @test 0.5 < S < 2.0

    # 便捷方法：不传 ψ₀，随机生成 bonddim = alg.D 的初态（cell/物理维度取自 H）
    # 配置与 twosite 的 seed-71 已验证路径一致（1-site 单胞的基态驱动存在
    # 独立的收敛问题，见 test/algorithms/idmrg.jl 同注）
    Wd(J) = mpohamiltonian(zeros(T, 2, 2),
                           [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J, Sz(T), Sz(T))])
    H2 = SparseIMPO([Wd(1.0), Wd(1.0)])
    Random.seed!(71)
    ψr, envsr, ϵr = find_groundstate(H2, VUMPS(D = 12, maxiter = 300, tol = 1e-10))
    @test max_bonddim(ψr) <= 12
    @test abs(real(expectationvalue(ψr, H2, envsr) / 2) - e_exact) < 2e-4
end
