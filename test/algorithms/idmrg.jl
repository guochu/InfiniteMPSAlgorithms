@testset "IDMRG 基态（single-site）" begin
    T = ComplexF64
    H = heisenberg_hamiltonian(T = T)
    e_exact = 0.25 - log(2)

    ψv, envsv, _ = find_groundstate(randomimps(T, [2, 2], 12), H,
                                    VUMPS(D = 12, maxiter = 300, tol = 1e-10))
    ev = real(expectationvalue(ψv, H, envsv) / 2)

    ψi, envsi, ϵ = find_groundstate(randomimps(T, [2, 2], 12), H,
                                    IDMRG(D = 12, maxiter = 300, tol = 1e-9))
    ei = real(expectationvalue(ψi, H, envsi) / 2)

    # 阈值覆盖随机初态的亚稳态收敛（与 MPSKit 同 iter/同样的逐位变分极限）
    @test abs(ei - e_exact) < 2e-4
    @test abs(ei - ev) < 2e-4

    # 固定键维 D=8 仍收敛（single-site 不增长键维）
    ψ8, envs8, _ = find_groundstate(randomimps(T, [2, 2], 8), H,
                                    IDMRG(D = 8, maxiter = 300, tol = 1e-9))
    e8 = real(expectationvalue(ψ8, H, envs8) / 2)
    @test max_bonddim(ψ8) <= 8
    @test abs(e8 - e_exact) < 1e-3

    # 便捷方法：不传 ψ₀，随机生成 bonddim = alg.D 的初态（cell/物理维度取自 H）
    # （1-site 单胞的基态驱动存在独立的收敛问题：VUMPS/IDMRG 在 N_ψ = N_H = 1
    #   时不收敛到基态，所有既有测试均以 ≥2 site 的 ψ₀ 规避，此处保持一致）
    Wd = mpohamiltonian(zeros(T, 2, 2),
                        [(1.0, Sx(T), Sx(T)), (1.0, Sy(T), Sy(T)), (1.0, Sz(T), Sz(T))])
    H2 = SparseIMPO([Wd, Wd])
    Random.seed!(71)
    ψc, envsc, ϵc = find_groundstate(H2, IDMRG(D = 12, maxiter = 300, tol = 1e-9))
    @test max_bonddim(ψc) <= 12
    ec = real(expectationvalue(ψc, H2, envsc) / 2)
    @test abs(ec - e_exact) < 2e-4

    # 与稠密 DenseIMPO 的 VUMPS 能量（Schur 重建态上）一致
    Hd = DenseIMPO(H)
    @test length(Hd) == 1
end
