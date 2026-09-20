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
