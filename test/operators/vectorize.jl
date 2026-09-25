# =====================================================================
# 算符代数变换：vectorize / devectorize / superoperator
#
# 约定：MPO 物理腿融合 f = u + du·(d - 1)（bra/行腿 u 为快指标，与
# asmps_view 一致；Julia 列主序下就是 reshape(X, :)）；
#   superoperator(:left)  = W ⊗ I（W 在慢/bra 通道）：𝓦·vec(X) = vec(W·X)
#   superoperator(:right) = I ⊗ Wᵀ（Wᵀ 在快/ket 通道）：𝓦·vec(X) = vec(X·W)
# 组合（mult 精确路径）：
#   mult(superoperator(W1, :left),  vectorize(W2)) == vectorize(W1·W2)
#   mult(superoperator(W2, :right), vectorize(W1)) == vectorize(W1·W2)
# =====================================================================

@testset "vectorize / devectorize / superoperator" begin
    T = ComplexF64
    Random.seed!(2026)

    @testset "vectorize ⇄ devectorize 精确往返" begin
        Ws = [randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)]      # L=2, d=2, 键 3→3→3
        Wc = CanonicalIMPO(Ws)
        ψ = vectorize(Wc)
        @test ψ isa CanonicalIMPS
        @test ismixedcanonical(ψ)                               # 规范族逐位携带
        @test phydims(ψ) == [4, 4]                              # 物理腿融合 d²
        @test bonddim(ψ, 1) == bonddim(Wc, 1)                   # 键维不变
        # 纯 reshape：融合张量与规范族自身的 asmps_view 逐位相等
        @test collect(parent(ψ.AL)) == asmps_view(collect(Wc.AL))
        @test collect(parent(ψ.AC)) == asmps_view(collect(Wc.AC))
        # 精确往返
        W2 = devectorize(ψ)
        @test W2 isa CanonicalIMPO
        @test collect(W2.AL) == collect(Wc.AL)
        @test collect(W2.AR) == collect(Wc.AR)
        @test collect(W2.C) == collect(Wc.C)
        @test collect(W2.AC) == collect(Wc.AC)
    end

    @testset "superoperator 约定（单点矩阵级）" begin
        d = 2
        W4 = randn(T, 1, d, 1, d)
        Wm = W4[1, :, 1, :]                          # d×d 算符矩阵
        X = randn(T, d, d)
        vX = reshape(X, :)                           # f = u + d(d-1)（列主序）
        𝓦L = superoperator(DenseIMPO([W4]); side = :left)[1]
        vout = reshape(𝓦L, d * d, d * d) * vX
        @test reshape(vout, d, d) ≈ Wm * X           # :left ⇒ vec(W·X)
        𝓦R = superoperator(DenseIMPO([W4]); side = :right)[1]
        vout2 = reshape(𝓦R, d * d, d * d) * vX
        @test reshape(vout2, d, d) ≈ X * Wm          # :right ⇒ vec(X·W)
        @test_throws ArgumentError superoperator(DenseIMPO([W4]); side = :bogus)
        # 非方算符拒绝（DenseIMPO 构造即报物理维不匹配）
        @test_throws DimensionMismatch DenseIMPO([randn(T, 1, 2, 1, 3)])
    end

    @testset "mpo1·mpo2 的两条 mpo·mps 路径" begin
        W1 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
        W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
        P = collect(exact_mult(W1.Ws, W2.Ws))        # 精确 MPO 乘积（参考）
        ψref = vectorize(CanonicalIMPO(P))
        # a) 𝓦_L(W1) 作用在 vec(W2)
        ψa, ova = mult(superoperator(W1; side = :left), vectorize(CanonicalIMPO(W2)))
        # b) 𝓦_R(W2) 作用在 vec(W1)
        ψb, ovb = mult(superoperator(W2; side = :right), vectorize(CanonicalIMPO(W1)))
        @test ova ≈ 2 atol = 1e-10                   # 精确路径 overlap = N
        @test ovb ≈ 2 atol = 1e-10
        @test fidelity(ψa, ψref) ≈ 1 atol = 1e-9          # Hilbert–Schmidt 内积下同态
        @test fidelity(ψb, ψref) ≈ 1 atol = 1e-9
        @test fidelity(ψa, ψb) ≈ 1 atol = 1e-9
        # MPO 版 fidelity：devectorize 回算符后的 HS 保真度
        @test fidelity(devectorize(ψa), CanonicalIMPO(P)) ≈ 1 atol = 1e-9
    end

    @testset "kron / transpose（超算符的 kron 表达）" begin
        W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
        I2m = identityimpo(T, [2, 2])
        # 超算符的 kron 表达（与逐元素约定逐位一致）
        @test superoperator(W; side = :left).Ws == kron(W, I2m).Ws
        @test superoperator(W; side = :right).Ws == kron(I2m, transpose(W)).Ws
        # transpose：逐 site 矩阵级 + 往返
        X4 = randn(T, 1, 2, 1, 2)
        @test collect(transpose(DenseIMPO([X4]))[1])[1, :, 1, :] == transpose(X4[1, :, 1, :])
        @test transpose(transpose(W)).Ws == W.Ws
        # kron：单点矩阵级（a 腿快 ⇒ 矩阵级 = Base.kron(B, A)）
        Y4 = randn(T, 1, 2, 1, 2)
        K = kron(DenseIMPO([X4]), DenseIMPO([Y4]))
        @test reshape(collect(K[1]), 4, 4) == kron(Y4[1, :, 1, :], X4[1, :, 1, :])
        # 多 site：物理维相乘、键维相乘
        K2 = kron(W, W)
        @test phydims(K2) == [4, 4]
        @test [bonddim(K2, ℓ) for ℓ in 1:2] == [9, 9]
        # 单胞长度不一致报错
        @test_throws DimensionMismatch kron(W, DenseIMPO([randn(T, 1, 2, 1, 2)]))
    end
end
