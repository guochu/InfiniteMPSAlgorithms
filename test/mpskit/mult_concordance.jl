# =====================================================================
# mult / exact_mult 与 MPSKit 对比（固化自 debug/mult_concordance.jl）
#
# MPSKit 0.13 相关实现：
# - `Base.:*(mpo1, mpo2)`：朴素 fuse_mul_mpo 乘法 ↔ 本包 exact_mult(W1, W2)；
# - `Base.:*(mpo, mps)`：朴素 MPO·MPS 施加 ↔ 本包 exact_mult(W, ψ)；
# - `MPSKit.approximate(ψ₀, (O, ψ), VOMPS()/IDMRG())`：变分施加
#   ↔ 本包 mult(W, ψ, alg)（D 由 alg.D 提供）。
# 对比一律在稠密周期 trace 表示 / 射线保真度意义下进行（规范与尺度不变）。
# =====================================================================

Random.seed!(2024)
T = ComplexF64
N = 2                    # 单胞长度
d = 2                    # 物理维
Dψ = 4                   # ψ 键维
Dw = 3                   # W 键维

ψ = randomimps(T, fill(d, N), Dψ)
W1 = randomimpo(T, fill(d, N), Dw)
W2 = randomimpo(T, fill(d, N), 2)

@testset "exact_mult mpo*mpo ≡ MPSKit *(DenseIMPO, DenseIMPO)" begin
    P = exact_mult(W1.Ws, W2.Ws)
    PO = to_mpskit(W1) * to_mpskit(W2)
    @test mpo_ray_residual(DenseIMPO(P), from_mpskit(PO)) < 1e-10
end

@testset "exact_mult mpo*mps ≡ MPSKit *(DenseIMPO, InfiniteMPS)" begin
    K = exact_mult(W1.Ws, ψ.AL)
    # 本包：朴素 fuse 后规范化成规范存储取稠密波形
    Kψ = vec(_dense_mps_repr(CanonicalIMPS(collect(K))))
    # MPSKit：朴素施加结果（自身规范化的 AL）
    ϕ = to_mpskit(W1) * to_mpskit(ψ)
    Kmk = vec(_dense_mps_repr(from_mpskit(ϕ)))
    ls = dot(Kmk, Kψ) / dot(Kmk, Kmk)
    @test norm(Kmk .- ls .* Kψ) / norm(Kmk) < 1e-10
end

@testset "mult VOMPS ≡ MPSKit approximate VOMPS" begin
    Dtar = 8                                    # 精确键维 Dw·Dψ = 12 → 变分压到 8
    y, ov = mult(W1, ψ, VOMPS(D = Dtar, maxiter = 200, tol = 1e-11))
    ϕk, _, δ = MPSKit.approximate(to_mpskit(randomimps(T, fill(d, N), Dtar)),
                                  (to_mpskit(W1), to_mpskit(ψ)),
                                  MPSKit.VOMPS(; tol = 1e-11, maxiter = 200))
    # 有损压缩（D < 精确键维）的变分最优点对实现细节敏感：实测本包解对精确
    # 施加态的保真度为 1（全局最优），MPSKit ≈ 0.992（次优盆地）。这里只要求
    # 两包解同物理（|dot| > 0.99）；本包内部 VOMPS ≡ IDMRG 不动点一致性
    # （|dot| = 1）在下一 testset 单独验证。
    yk = from_mpskit(ϕk)
    @test abs(dot(y, yk)) > 0.99
end

@testset "mult IDMRG ≡ MPSKit approximate IDMRG" begin
    Dtar = 8
    y, ov = mult(W1, ψ, IDMRG(D = Dtar, maxiter = 200, tol = 1e-11))
    ϕk, _, δ = MPSKit.approximate(to_mpskit(randomimps(T, fill(d, N), Dtar)),
                                  (to_mpskit(W1), to_mpskit(ψ)),
                                  MPSKit.IDMRG(; tol = 1e-11, maxiter = 200))
    yk = from_mpskit(ϕk)
    @test abs(dot(y, yk)) > 0.99
end

@testset "mult VOMPS ≡ IDMRG（本包双算法不动点一致）" begin
    yv, _ = mult(W1, ψ, VOMPS(D = 8, maxiter = 200, tol = 1e-11))
    yi, _ = mult(W1, ψ, IDMRG(D = 8, maxiter = 200, tol = 1e-11))
    @test abs(dot(yv, yi)) > 1 - 1e-6
end

@testset "mult mpo*mpo ≡ exact_mult（VOMPS / IDMRG 不动点一致）" begin
    # 恒等 MPO 复合：mult(W2, I2) 精确恢复 W2 的射线（两算法）
    I2 = identityimpo(T, [2, 2])
    for alg in (VOMPS(D = 2, maxiter = 200, tol = 1e-11), IDMRG(D = 2, maxiter = 200, tol = 1e-11))
        y, ov = mult(W2, I2, alg)
        @test y isa CanonicalIMPO && ismixedcanonical(y)
        @test mpo_ray_residual(DenseIMPO(y), W2) < 1e-6
    end
    # 变分复合 vs 朴素精确：满键维时 overlap = N 且稠密表示平行
    y, ov = mult(W1, W2, VOMPS(D = 6, maxiter = 300, tol = 1e-12))
    @test real(ov) > N - 1e-6
    @test mpo_ray_residual(DenseIMPO(y), DenseIMPO(exact_mult(W1.Ws, W2.Ws))) < 1e-6
end
