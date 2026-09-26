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

@testset "异构物理维（unit cell 内各站 d 不同）" begin
    T = ComplexF64
    Random.seed!(3)
    dims = [2, 3, 2]                       # 单胞内物理维不一致
    ψ = randomimps(T, dims, 4)
    @test phydims(ψ) == dims
    @test ismixedcanonical(ψ)
    @test abs(norm(ψ) - 1) < 1e-10

    # 观测量：逐站尺寸不同
    @test all(abs(real(expectationvalue(ψ, (ℓ,) => Matrix{T}(I, dims[ℓ], dims[ℓ]))) - 1) < 1e-10
              for ℓ in 1:3)
    O12 = Matrix{T}(I, dims[1] * dims[2], dims[1] * dims[2])
    @test abs(real(expectationvalue(ψ, (1, 2) => O12)) - 1) < 1e-10
    @test isfinite(entropy(ψ, 2))

    # MPO 层：DenseIMPO / CanonicalIMPO 逐站物理维不同
    H1 = DenseIMPO([randn(T, 1, dims[ℓ], 1, dims[ℓ]) for ℓ in 1:3])
    @test phydims(H1) == dims
    W = CanonicalIMPO([randn(T, 2, dims[ℓ], 2, dims[ℓ]) for ℓ in 1:3])
    @test ismixedcanonical(W)

    # SparseIMPO：层数一致、各站物理维不同
    Hs = SparseIMPO([mpohamiltonian(randn(T, dims[ℓ], dims[ℓ]),
                                    Tuple{Float64,Matrix{T},Matrix{T}}[]) for ℓ in 1:3])
    @test phydims(Hs) == dims
    @test bonddim(Hs) == 2
    e0 = real(expectationvalue(ψ, Hs))
    @test isfinite(e0)
    # H + λs 逐站取物理维（回归：曾用 site 1 的 d）
    @test phydims(Hs + [0.1, 0.2, 0.3]) == dims
    @test abs(real(expectationvalue(ψ, Hs + ones(3))) - e0 - 3) < 1e-10

    # 算法：可解析的异构模型（逐站对角 on-site 场 ⇒ 基态是乘积态，E = Σ min λ）
    λs = [T[-2, 1], T[-3, 0, 2], T[-2, 1]]
    hd = [Matrix(Diagonal(λs[ℓ])) for ℓ in 1:3]
    Hdiag = SparseIMPO([mpohamiltonian(hd[ℓ], Tuple{Float64,Matrix{T},Matrix{T}}[])
                        for ℓ in 1:3])
    E_exact = sum(minimum(real.(λs[ℓ])) for ℓ in 1:3)
    ψ0 = prodimps(T, dims, [argmin(real.(λs[ℓ])) for ℓ in 1:3])   # D = 1 精确基态
    for alg in (VUMPS(maxiter = 200, tol = 1e-10, verbosity = 0),
                IDMRG(maxiter = 200, tol = 1e-10, verbosity = 0))
        ψg, eg, _ = find_groundstate(copy(ψ0), Hdiag, alg)
        @test abs(real(expectationvalue(ψg, Hdiag, eg)) - E_exact) < 1e-8
        @test phydims(ψg) == dims
    end
    # 随机异构模型上 VUMPS/IDMRG/虚时演化 均保持异构物理维
    Random.seed!(5)
    ψv, ev, _ = find_groundstate(copy(ψ), Hs, VUMPS(D = 4, maxiter = 30, tol = 1e-8, verbosity = 0))
    @test phydims(ψv) == dims
    Random.seed!(5)
    ψi, ei, _ = find_groundstate(copy(ψ), Hs, IDMRG(D = 4, maxiter = 30, tol = 1e-8, verbosity = 0))
    @test phydims(ψi) == dims
    @test isfinite(real(expectationvalue(ψv, Hs, ev)))
    @test isfinite(real(expectationvalue(ψi, Hs, ei)))
    ψt, _, _ = time_evolve(ψ, Hs, 0:0.05:1, TDVP(); imaginary_evolution = true)
    @test phydims(ψt) == dims

    # changebond! / compress / mult 保持 phydims
    q = copy(ψ)
    changebond!(q; D = 6)
    @test phydims(q) == dims && all(bonddim(q, ℓ) == 6 for ℓ in 1:3)
    y, = compress(ψ, VOMPS(D = 6, maxiter = 10))
    @test phydims(y) == dims
    ym, _ = mult(H1, ψ)
    @test phydims(ym) == dims
    @test phydims(superoperator(H1)) == [dims[ℓ]^2 for ℓ in 1:3]
    @test isfinite(real(correlator(ψ, Matrix{T}(I, dims[1], dims[1]),
                                   Matrix{T}(I, dims[2], dims[2]), 1, 2)))

    # TEBD 两体门（异构 d 用 rank-4 张量形式；含周期 wrap）
    t12 = zeros(T, dims[1], dims[2], dims[1], dims[2])
    for i in 1:dims[1], j in 1:dims[2]
        t12[i, j, i, j] = 1
    end
    @test phydims(apply!(UnitaryGate((1, 2), t12), copy(ψ))) == dims
    t34 = zeros(T, dims[3], dims[1], dims[3], dims[1])
    for i in 1:dims[3], j in 1:dims[1]
        t34[i, j, i, j] = 1
    end
    @test phydims(apply!(UnitaryGate((3, 4), t34), copy(ψ))) == dims
end

