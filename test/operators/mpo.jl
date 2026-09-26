@testset "SparseIMPO 与 infinite_mpo" begin
    T = ComplexF64

    # bulk → DenseIMPO 构造
    Hm, bulk = heisenberg_xxz(T = T)
    @test Hm isa DenseIMPO
    @test length(Hm) == 1
    @test bonddim(Hm, 1) == 4   # identity + 3 个通道（SxSx/SySy/SzSz 共 3 个 NN 项）

    # `find_groundstate` 只支持 SparseIMPO（DenseIMPO 的周期 trace 期望不是能量，
    # 见 DMRGCache 的 DenseIMPO 版说明）⇒ 传 DenseIMPO 显式报 ArgumentError
    @test_throws ArgumentError find_groundstate(randomimps(T, [2, 2], 10), Hm,
                                               VUMPS(D = 10, maxiter = 10, tol = 1e-9,
                                                     verbosity = 0))
    @test_throws ArgumentError find_groundstate(randomimps(T, [2, 2], 10),
                                               DenseIMPO(tfim_hamiltonian(T = T)),
                                               IDMRG(D = 10, maxiter = 10, tol = 1e-9))
    @test_throws ArgumentError find_groundstate(DenseIMPO(tfim_hamiltonian(T = T)),
                                               VUMPS(D = 10, maxiter = 10, tol = 1e-9))
    # 同一模型的 SparseIMPO 形式可正常求基态
    Hs = heisenberg_hamiltonian(T = T)
    ψs, envss, _ = find_groundstate(randomimps(T, [2, 2], 10), Hs,
                                    VUMPS(D = 10, maxiter = 100, tol = 1e-9, verbosity = 0))
    @test isfinite(real(expectationvalue(ψs, Hs, envss)))

    # tompotensors（有限稠密 MPO）的形状（全部虚拟层，无端点收缩）
    toms = tompotensors(SparseIMPO([bulk, bulk]))
    @test size(toms[1]) == (5, 2, 5, 2)
    @test size(toms[1], 2) == 2

    # Hubbard 模型可构造且 MPO 期望为实数
    Hf, _ = fermi_hubbard(T = T)
    ψf = prodimps(T, [4, 4], [1, 1])
    @test abs(imag(expectationvalue(ψf, Hf))) < 1e-12
end

@testset "Schur SparseIMPO" begin
    T = ComplexF64
    e_exact = 0.25 - log(2)

    # 构造与稠密化
    H = heisenberg_hamiltonian(T = T)
    @test H isa SparseIMPO
    @test bonddim(H) == 5        # 2 单位层 + 3 通道
    @test isidentitylevel(H, 1) && isidentitylevel(H, 5)
    @test !isemptylevel(H, 2)
    Wd = tompotensor(H[1])
    @test size(Wd) == (5, 2, 5, 2)
    @test Wd[1, :, 1, :] ≈ Matrix{T}(I, 2, 2)
    @test Wd[5, :, 5, :] ≈ Matrix{T}(I, 2, 2)

    # 块矩阵访问
    @test H[1][1, 1] ≈ Matrix{T}(I, 2, 2)
    @test H[1][1, 5] ≈ H[1].D
    @test H[1][1, 2] ≈ H[1].C[:, 1, :]

    # 有限链逐项能量对照：Schur 收缩 vs 显式算符平均（随机态、短链近似）
    ψ0 = randomimps(T, [2, 2], 10)
    envs = DMRGCache(ψ0, H)
    eH = real(expectationvalue(ψ0, H, envs))
    @test isfinite(eH) && abs(imag(eH)) < 1e-10

    # VUMPS 在 Schur 哈密顿量上收敛到精确能量密度
    # 阈值覆盖随机初态的亚稳态收敛（变分极限与 MPSKit 逐位一致，见 debug/suite_full.log）
    ψr, envsr, ϵ = find_groundstate(ψ0, H, VUMPS(D = 10, maxiter = 300, tol = 1e-10))
    er = real(expectationvalue(ψr, H, envsr) / 2)
    @test abs(er - e_exact) < 5e-4

    # H + λs（能量平移）：对标 MPSKit 语义——逐 site 加 λ（i => scale!(id, λ)），
    # 2-site 单胞的 cell 能量和增加 N·λ
    H2 = H + [0.1]
    @test bonddim(H2) == bonddim(H)
    @test real(expectationvalue(ψr, H2)) ≈ real(expectationvalue(ψr, H)) + 0.2 atol = 1e-8

    # TFIM Hamiltonian：乘积态期望 = -h·N·⟨σz⟩
    Ht = tfim_hamiltonian(T = T)
    ψp = prodimps(T, [2], [1])   # 全 |0⟩（σz = +1）
    @test abs(real(expectationvalue(ψp, Ht)) - (-1.0)) < 1e-12

    # make_time_mpo 路径（SparseIMPO 的逐 site Schur tensor 演化）
    U = make_time_mpo(H, 0.01, WII(); imaginary_evolution = true)
    @test U isa DenseIMPO
end
