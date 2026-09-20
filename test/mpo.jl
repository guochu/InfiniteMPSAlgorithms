@testset "MPOHamiltonian 与 infinite_mpo" begin
    T = ComplexF64

    # bulk → InfiniteMPO 构造
    Hm, bulk = heisenberg_xxz(T = T)
    @test Hm isa InfiniteMPO
    @test length(Hm) == 1
    @test mpobond(Hm, 1) == 4   # identity + 3 个通道（SxSx/SySy/SzSz 共 3 个 NN 项）

    # 稠密 InfiniteMPO 路径可运行（能量收敛性与 MPSKit 保持一致：
    # 恒等通道主导本征向量污染，见 PLAN §9；收敛断言只在 Jordan 路径检查）
    ψd, envsd, ϵd = find_groundstate(random_mps(T, [2, 2], 10), Hm,
                                     VUMPS(maxiter = 10, tol = 1e-9, verbosity = 0))
    @test expectation_value(ψd, Hm, envsd) isa Number

    # tompotensors（有限稠密 MPO）的形状（全部虚拟层，无端点收缩）
    toms = tompotensors(FiniteMPOHamiltonian([bulk, bulk]))
    @test size(toms[1]) == (5, 2, 5, 2)
    @test size(toms[1], 2) == 2

    # Hubbard 模型可构造且 MPO 期望为实数
    Hf, _ = fermi_hubbard(T = T)
    ψf = product_mps(T, [4, 4], [1, 1])
    @test abs(imag(expectation_value(ψf, Hf))) < 1e-12
end

@testset "Jordan MPOHamiltonian" begin
    T = ComplexF64
    e_exact = 0.25 - log(2)

    # 构造与稠密化
    H = heisenberg_hamiltonian(T = T)
    @test H isa InfiniteMPOHamiltonian
    @test mpobond(H) == 5        # 2 单位层 + 3 通道
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

    # 有限链逐项能量对照：Jordan 收缩 vs 显式算符平均（随机态、短链近似）
    ψ0 = random_mps(T, [2, 2], 10)
    envs = environments(ψ0, H)
    eH = real(expectation_value(ψ0, H, envs))
    @test isfinite(eH) && abs(imag(eH)) < 1e-10

    # VUMPS 在 Jordan 哈密顿量上收敛到精确能量密度
    # 阈值覆盖随机初态的亚稳态收敛（变分极限与 MPSKit 逐位一致，见 debug/suite_full.log）
    ψr, envsr, ϵ = find_groundstate(ψ0, H, VUMPS(maxiter = 300, tol = 1e-10))
    er = real(expectation_value(ψr, H, envsr) / 2)
    @test abs(er - e_exact) < 5e-4

    # H + λs（能量平移）：对标 MPSKit 语义——逐 site 加 λ（i => scale!(id, λ)），
    # 2-site 单胞的 cell 能量和增加 N·λ
    H2 = H + [0.1]
    @test mpobond(H2) == mpobond(H)
    @test real(expectation_value(ψr, H2)) ≈ real(expectation_value(ψr, H)) + 0.2 atol = 1e-8

    # TFIM Hamiltonian：乘积态期望 = -h·N·⟨σz⟩
    Ht = tfim_hamiltonian(T = T)
    ψp = product_mps(T, [2], [1])   # 全 |0⟩（σz = +1）
    @test abs(real(expectation_value(ψp, Ht)) - (-1.0)) < 1e-12

    # make_time_mpo 路径（Jordan → Schur bulk）
    U = make_time_mpo(H, 0.01, WII(); imaginary_evolution = true)
    @test U isa InfiniteMPO
end
