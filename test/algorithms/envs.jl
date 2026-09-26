@testset "转移矩阵与环境" begin
    T = ComplexF64
    ψ = randomimps(T, [2, 3], 5)

    # 恒等通道环境 = 恒等矩阵（AL 严格规范，对应 MPSKit 的 l_LL/r_RR）
    envs = OverlapCache(ψ)
    @test leftenv(envs, 1) == reshape(Matrix{T}(I, 5, 5), 5, 1, 5)
    @test rightenv(envs, 1) == reshape(Matrix{T}(I, 5, 5), 5, 1, 5)

    # 恒等 MPO（作为 Σᵢ I 的和式哈密顿量）期望 = N
    I2 = identityimpo(T, [2, 3])
    @test abs(expectationvalue(ψ, I2) - 2) < 1e-8

    # TransferMatrix：与 push_env_left 序列一致
    tm = TransferMatrix(ψ)
    L = Matrix{T}(I, 5, 5)
    w = tm * vec(L)
    Lfull = reshape(w, 5, 5)
    Lseq = push_env_left(L, ψ.AL[1])
    Lseq = push_env_left(Lseq, ψ.AL[2])
    @test norm(Lfull - Lseq) / norm(Lseq) < 1e-10
    @test size(tm) == (25, 25)

    # fixedpoint：AL 转移矩阵主本征值 ≈ 1（左正则 → L = I 固定点）
    λ, v = fixedpoint(TransferMatrix(ψ.AL, ψ.AL), vec(I(5)), :LM,
                      Defaults.alg_eigsolve(; ishermitian = true))
    @test abs(λ - 1) < 1e-8
end

@testset "非均匀键：环境与期望值（unit cell > 1）" begin
    T = ComplexF64
    Random.seed!(7)
    Hs = _tfim3(J = 1.0, h = 1.3, T = T)
    nl = bonddim(Hs)

    # ---- 最小非均匀键态 χ = [2, 4, 2]（unit cell = 3）----
    ψnu = _nonuniform_mps([2, 4, 2], 2)
    @test [bonddim(ψnu, ℓ) for ℓ in 1:3] == [2, 4, 2]
    @test ismixedcanonical(ψnu)

    envs = DMRGCache(ψnu, Hs)
    # 左环境在键 ℓ-1 上、右环境在键 ℓ 上（非均匀键下二者维数不同）
    @test [size(leftenv(envs, ℓ), 1) for ℓ in 1:3] == [2, 2, 4]
    @test [size(rightenv(envs, ℓ), 1) for ℓ in 1:3] == [2, 4, 2]
    @test all(size(leftenv(envs, ℓ), 2) == nl for ℓ in 1:3)
    @test all(size(rightenv(envs, ℓ), 2) == nl for ℓ in 1:3)
    Enu = real(expectationvalue(ψnu, Hs, envs))
    @test isfinite(Enu)

    # 非均匀键态上的基态搜索：两种算法收敛到同一能量，且保留非均匀 profile
    Random.seed!(11)
    ψ1, ε1, _ = find_groundstate(copy(ψnu), Hs, VUMPS(D = 4, maxiter = 100, tol = 1e-10))
    Random.seed!(11)
    ψ2, ε2, _ = find_groundstate(copy(ψnu), Hs, IDMRG(D = 4, maxiter = 100, tol = 1e-10))
    @test abs(real(expectationvalue(ψ1, Hs, ε1)) -
              real(expectationvalue(ψ2, Hs, ε2))) < 1e-8
    @test [bonddim(ψ1, ℓ) for ℓ in 1:3] == [2, 4, 2]
    @test [bonddim(ψ2, ℓ) for ℓ in 1:3] == [2, 4, 2]

    # ---- 零填充扩键：同一物理态，观测量必须逐位不变 ----
    ψu = randomimps(T, fill(2, 3), 4)
    E0 = real(expectationvalue(ψu, Hs))
    ψp = _padbond!(copy(ψu), 2, 1)
    @test [bonddim(ψp, ℓ) for ℓ in 1:3] == [4, 5, 4]
    @test abs(dot(ψp, ψu) / (norm(ψp) * norm(ψu)) - 1) < 1e-12
    @test abs(real(expectationvalue(ψp, Hs)) - E0) < 1e-10

    # 非均匀 MPO（同一算符）：DenseIMPO 通道也必须给出同一期望值
    Hd = DenseIMPO([randn(T, 3, 2, 3, 2) for _ in 1:3])
    Ed0 = real(expectationvalue(ψu, Hd))
    @test abs(real(expectationvalue(ψu, _padbond(Hd, 2, 1))) - Ed0) < 1e-10
end

