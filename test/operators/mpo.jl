@testset "SparseIMPO 与模型构造" begin
    T = ComplexF64

    # bulk → DenseIMPO 构造
    Hm, bulk = heisenberg_xxz(T = T)
    @test Hm isa DenseIMPO
    @test length(Hm) == 1
    @test bonddim(Hm, 1) == 5   # 2 单位层 + 3 个通道（SxSx/SySy/SzSz 共 3 个 NN 项）

    # DenseIMPO 通道（MPSKit InfiniteMPO 通道的对标实现）：VUMPS/IDMRG 可直接
    # 求基态——该通道的「能量」为周期 trace 收缩（含恒等层 bookkeeping，见
    # DMRGCache 的 DenseIMPO 版说明）；恒等层结构的收敛较慢，这里只断言可跑且
    # 期望有限，与 MPSKit 的一致性由 concordance 测试固化（同参数收敛能量对比）。
    # 投影有效哈密顿量因环境简并非严格厄米，局部求解显式用 Arnoldi
    Hd = DenseIMPO(tfim_hamiltonian(T = T))
    alg_dense = Defaults.alg_eigsolve(; ishermitian = false)
    ψd, envsd, _ = find_groundstate(randomimps(T, [2, 2]; D = 10), Hd,
                                    VUMPS(D = 10, maxiter = 300, tol = 1e-9,
                                          verbosity = 0, alg_eigsolve = alg_dense))
    @test isfinite(real(expectationvalue(ψd, Hd, envsd)))
    ψd2, envsd2, _ = find_groundstate(Hd, IDMRG(D = 10, maxiter = 300, tol = 1e-9,
                                                alg_eigsolve = alg_dense))
    @test isfinite(real(expectationvalue(ψd2, Hd, envsd2)))
    # 同一模型的 SparseIMPO 形式可正常求基态
    Hs = heisenberg_hamiltonian(T = T)
    ψs, envss, _ = find_groundstate(randomimps(T, [2, 2]; D = 10), Hs,
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
    ψ0 = randomimps(T, [2, 2]; D = 10)
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

@testset "矩形 Schur 链（非方阵 SchurMPOTensor，unitcell = 3）" begin
    T = ComplexF64
    # 二聚化 TFIM：NN 耦合只在键 (1,2) 上（通道只跨越键 1），
    # 各站逻辑形状 [2×3, 3×2, 2×2]、键层数 [3, 2, 2]
    h, J = T(0.3), T(1.0)
    W1 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 2, 3)
    W1[1, 1] = one(T); W1[2, 3] = one(T)
    W1[1, 3] = Matrix{T}(-h * σz(T)); W1[1, 2] = Matrix{T}(-J * σx(T))
    W2 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 3, 2)
    W2[1, 1] = one(T); W2[3, 2] = one(T)
    W2[1, 2] = Matrix{T}(-h * σz(T)); W2[2, 2] = Matrix{T}(σx(T))
    W3 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 2, 2)
    W3[1, 1] = one(T); W3[2, 2] = one(T)
    W3[1, 2] = Matrix{T}(-h * σz(T))

    # 构造与链式闭合检查
    H = SparseIMPO([W1, W2, W3])
    @test [size(H[i]) for i in 1:3] == [(2, 3), (3, 2), (2, 2)]
    @test [bonddim(H, ℓ) for ℓ in 1:3] == [2, 3, 2]   # 键 ℓ-1 的层数
    @test max_bonddim(H) == 3
    @test_throws ArgumentError bonddim(H)              # 非均匀层数：0 参版报错
    @test_throws DimensionMismatch SparseIMPO([W1, W1, W3])  # 链式闭合破坏
    @test size(tompotensor(H[1])) == (2, 2, 3, 2)
    Hd = DenseIMPO(H)
    @test [bonddim(Hd, ℓ) for ℓ in 1:3] == [2, 3, 2]   # 稠密化保持键 profile

    # 逐层环境：lefts 的 w 维 = 键 ℓ-1 层数、rights 的 = 键 ℓ 层数
    Random.seed!(2)
    ψ0 = randomimps(T, 3; d = 2, D = 8)
    envs0 = DMRGCache(ψ0, H)
    @test [size(leftenv(envs0, ℓ), 2) for ℓ in 1:3] == [2, 3, 2]
    @test [size(rightenv(envs0, ℓ), 2) for ℓ in 1:3] == [3, 2, 2]
    @test isfinite(real(expectationvalue(ψ0, H, envs0)))

    # VUMPS 收敛到二聚体 + 自由自旋的解析基态能量：
    # 偶宇称块 diag(-2h, 2h) ⊕ -J ⇒ E_dimer = -sqrt(4h² + J²)，自由 site 3 = -|h|
    E_exact = -sqrt(4 * abs2(h) + abs2(J)) - abs(h)
    Random.seed!(3)
    ψg, eg, _ = find_groundstate(randomimps(T, 3; d = 2, D = 8), H,
                                 VUMPS(D = 8, maxiter = 300, tol = 1e-12, verbosity = 0))
    @test abs(real(expectationvalue(ψg, H, eg)) - E_exact) < 1e-8

    # 与同一算符的均匀方形（3×3）表示逐位一致（通道层处处保留、部分空置）
    V1 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 3, 3)
    V1[1, 1] = one(T); V1[3, 3] = one(T)
    V1[1, 3] = Matrix{T}(-h * σz(T)); V1[1, 2] = Matrix{T}(-J * σx(T))
    V2 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 3, 3)
    V2[1, 1] = one(T); V2[3, 3] = one(T)
    V2[1, 3] = Matrix{T}(-h * σz(T)); V2[2, 3] = Matrix{T}(σx(T))
    V3 = Matrix{Union{Missing,T,Matrix{T}}}(missing, 3, 3)
    V3[1, 1] = one(T); V3[3, 3] = one(T)
    V3[1, 3] = Matrix{T}(-h * σz(T))
    Hsq = SparseIMPO([V1, V2, V3])
    @test bonddim(Hsq) == 3
    ψr = randomimps(T, 3; d = 2, D = 6)
    @test expectationvalue(ψr, H) ≈ expectationvalue(ψr, Hsq) atol = 1e-10

    # 直和加法（内层通道拼接、单位层共享）与能量平移
    H2 = H + H
    @test [size(H2[i]) for i in 1:3] == [(2, 4), (4, 2), (2, 2)]
    @test [bonddim(H2, ℓ) for ℓ in 1:3] == [2, 4, 2]
    @test abs(real(expectationvalue(ψg, H2)) - 2 * E_exact) < 1e-8
    Hλ = H + [0.1, 0.2, 0.3]
    @test abs(real(expectationvalue(ψg, Hλ)) - (E_exact + 0.6)) < 1e-8

    # 逐站 Schur 演化（W^II）在矩形链上可装配
    U = make_time_mpo(H, 0.01, WII(); imaginary_evolution = true)
    @test U isa DenseIMPO
end

@testset "DenseIMPO / CanonicalIMPO 非均匀键（unitcell 内 bonddim 不同）" begin
    T = ComplexF64
    Random.seed!(4)
    Wd = DenseIMPO([randn(T, 2, 2, 3, 2), randn(T, 3, 2, 2, 2), randn(T, 2, 2, 2, 2)])
    @test [bonddim(Wd, ℓ) for ℓ in 1:3] == [2, 3, 2]
    @test max_bonddim(Wd) == 3

    # CanonicalIMPO：混合规范化保留非均匀键 profile（规范变换不改变射线）。
    # bonddim 约定与 MPS 侧一致：C[ℓ] 在键 ℓ（site ℓ 右侧）
    Wc = CanonicalIMPO([copy(w) for w in Wd.Ws])
    @test Wc isa CanonicalIMPO
    @test [bonddim(Wc, ℓ) for ℓ in 1:3] == [3, 2, 2]
    @test ismixedcanonical(Wc)
    @test fidelity(DenseIMPO(Wc), Wd) ≈ 1 atol = 1e-8

    # vectorize / devectorize 往返保持键 profile
    Wrt = devectorize(vectorize(Wd))
    @test [bonddim(Wrt, ℓ) for ℓ in 1:3] == [2, 3, 2]
    @test all(ℓ -> size(Wrt[ℓ]) == size(Wd[ℓ]), 1:3)

    # 非均匀键 DenseIMPO 的期望值通道（主本征向量环境）
    ψ = randomimps(T, 3; d = 2, D = 4)
    @test isfinite(real(expectationvalue(ψ, Wd)))
end
