# =====================================================================
# longrangeop：指数衰减长程算符（ExpDecayOpTerm / ExpDecayOpSum）
# =====================================================================

@testset "ExpDecayOpTerm / ExpDecayOpSum" begin
    T = ComplexF64
    Random.seed!(11)
    a = randn(T, 2, 2)
    m = randn(T, 2, 2)
    b = randn(T, 2, 2)

    # 单项：α·λ^d·a m^{d-1} b，Jordan 通道结构
    t = ExpDecayOpTerm(a, m, b, 0.8, 0.5)
    @test scalartype(t) == ComplexF64
    J = JordanMPOTensor(t)
    @test J isa JordanMPOTensor && nlvls(J) == 3
    @test J[1, 2] ≈ 0.8 * a atol = 1e-12          # 通道开启（α·a）
    @test J[2, 2] ≈ 0.5 * m atol = 1e-12          # 通道自传播（λ·m）
    @test J[2, 3] ≈ 0.5 * b atol = 1e-12          # 通道关闭（λ·b）
    I2 = Matrix{T}(I, 2, 2)
    @test J[1, 1] == I2 && J[3, 3] == I2 && J[1, 3] == zeros(T, 2, 2)

    # 求和：每项一个通道
    αs = [1.0, 0.5]
    λs = [0.6, 0.3]
    s = ExpDecayOpSum(a, m, b, αs, λs)
    @test scalartype(s) == ComplexF64
    J2 = JordanMPOTensor(s)
    @test nlvls(J2) == 4
    @test J2[1, 2] ≈ αs[1] * a atol = 1e-12
    @test J2[1, 3] ≈ αs[2] * a atol = 1e-12
    @test J2[2, 2] ≈ λs[1] * m atol = 1e-12
    @test J2[3, 3] ≈ λs[2] * m atol = 1e-12
    @test J2[2, 4] ≈ λs[1] * b atol = 1e-12
    @test J2[3, 4] ≈ λs[2] * b atol = 1e-12
end

@testset "ExpDecayOpSum 周期平铺的最近邻算符" begin
    T = ComplexF64
    a = [0.5 0.1; -0.2 0.3]
    m = [0.2 0.0; 0.1 -0.1]
    b = [1.0 0.2; 0.0 0.5]
    αs = [1.0, -0.4]
    λs = [0.5, 0.25]
    W = tompotensor(JordanMPOTensor(ExpDecayOpSum(a, m, b, αs, λs)))

    # 周期平铺：site i → i+1 的最近邻算符（Jordan 矩阵的平方 (1, end) 块，
    # 步数恰好为 2、无回绕混入）= Σ_p α_p λ_p · a b
    n, dphys = size(W, 1), size(W, 2)
    K = zeros(T, n * dphys, n * dphys)
    for i in 1:n, j in 1:n
        K[(i-1)*dphys+1:i*dphys, (j-1)*dphys+1:j*dphys] .= W[i, :, j, :]
    end
    tgt = sum(αs[p] * λs[p] * a * b for p in eachindex(αs))
    got = (K^2)[1:dphys, (n - 1)*dphys+1:n*dphys]
    @test got ≈ tgt atol = 1e-12
    # 单 site 的开+关同 site 块为零（无 h1）：即 cell[1, end] = 0 的稠密化
    @test W[1, :, n, :] == zeros(T, 2, 2)
end
