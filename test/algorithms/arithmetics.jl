# =====================================================================
# 算术运算测试：exact_*（朴素精确，debug 基准）+ 迭代 add / hadamard
#
# - exact_*：输入/输出均为 PeriodicVector{Array} 张量串，周期 trace 表示下
#   验证精确可加 / 可乘 / 外积结构；
# - add / hadamard：朴素精确构造（D = nothing）对标 MPSKit 语义
#   （键维直和 / fuse_mul_mpo）；
# - 变分压缩（D::Int）验证 VOMPS 与 IDMRG 两条算法路径收敛到同一结果；
# - 数值断言基于周期 trace 表示（_dense_mps_repr / _dense_mpo_repr，定义于
#   testhelpers.jl；规范变换下望远相消，是纯规范不变的波形/算符表示）。
# =====================================================================

@testset "exact_*：朴素精确构造（debug 基准）" begin
    T = ComplexF64
    Random.seed!(47)
    W1 = randomimpo(T, [2, 2], 2)
    W2 = randomimpo(T, [2, 2], 3)
    ψ1 = randomimps(T, [2, 2], 3)

    # mpo*mpo：键维 = 两键维乘积，稠密算符 = 矩阵乘积
    P = exact_mult(W1.Ws, W2.Ws)
    @test P isa PeriodicVector && length(P) == 2
    @test size(P[1]) == (6, 2, 6, 2)
    @test reshape(_dense_mpo_repr(DenseIMPO(P)), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10

    # mpo*mps：键维 = W 键维 × ψ 键维，波形 ∝ W·ψ（稠密算符作用在波形上）
    Kψ = exact_mult(W1.Ws, ψ1.AL)
    @test Kψ isa PeriodicVector && length(Kψ) == 2
    @test size(Kψ[1]) == (6, 2, 6)
    cψf = vec(_dense_mps_repr(CanonicalIMPS(collect(Kψ))))
    M = reshape(_dense_mpo_repr(W1), 4, 4)
    dψ = vec(_dense_mps_repr(ψ1))
    ls = dot(M * dψ, cψf) / dot(cψf, cψf)
    @test norm(M * dψ .- ls .* cψf) / norm(M * dψ) < 1e-10
end

@testset "exact_add：周期 trace 精确可加（debug 基准）" begin
    T = ComplexF64
    Random.seed!(48)
    ψ1 = randomimps(T, [2, 2], 3)
    ψ2 = randomimps(T, [2, 2], 4)
    W1 = randomimpo(T, [2, 2], 2)
    W2 = randomimpo(T, [2, 2], 3)

    K = exact_add(ψ1.AL, ψ2.AL)
    @test length(K) == 2 && size(K[1]) == (7, 2, 7)
    @test _dense_mps_repr(CanonicalIMPS(collect(K))) ≈
          _dense_mps_repr(ψ1) + _dense_mps_repr(ψ2) atol = 1e-10

    K4 = exact_add(W1.Ws, W2.Ws)
    @test size(K4[1]) == (5, 2, 5, 2)
    @test _dense_mpo_repr(DenseIMPO(K4)) ≈
          _dense_mpo_repr(W1) + _dense_mpo_repr(W2) atol = 1e-10

    # 长度不匹配抛错
    @test_throws DimensionMismatch exact_add(ψ1.AL, randomimps(T, [2, 2, 2], 3).AL)
end

@testset "exact_hadamard：波形逐点乘积（debug 基准）" begin
    T = ComplexF64
    Random.seed!(49)
    ψ1 = randomimps(T, [2, 2], 3)
    ψ2 = randomimps(T, [2, 2], 4)
    K = exact_hadamard(ψ1.AL, ψ2.AL)
    @test size(K[1]) == (12, 2, 12)   # 物理维不变，键维 = 3·4
    # 原始 zip 张量串的周期 trace 严格逐点：两条虚拟链独立 → trace 因子化
    c1 = _dense_trace(collect(ψ1.AL))
    c2 = _dense_trace(collect(ψ2.AL))
    @test _dense_trace(collect(K)) ≈ c1 .* c2 atol = 1e-10
end

@testset "mult：MPO 乘法（朴素精确对照 + 压缩）" begin
    T = ComplexF64
    Random.seed!(44)
    W1 = randomimpo(T, [2, 2], 2)
    W2 = randomimpo(T, [2, 2], 3)
    I2 = identityimpo(T, [2, 2])
    # 朴素乘法 = 稠密算符矩阵乘法，键维 = 两键维乘积
    P = exact_mult(W1.Ws, W2.Ws)
    @test bonddim(DenseIMPO(P), 1) == 6
    @test reshape(_dense_mpo_repr(DenseIMPO(P)), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10
    # 迭代乘法：W1·I = W1（D = 2 = 目标键维）。与目标平行即可（整体相位/尺度是
    # 输出射线规范的自由度）
    P1, ov1 = mult(W1, I2, VOMPS(D = 2))
    @test P1 isa CanonicalIMPO && bonddim(P1, 1) == 2
    @test real(ov1) > 2 - 1e-6   # overlap = N 即方向一致
    dP1 = vec(_dense_mpo_repr(DenseIMPO(P1)))
    dW1v = vec(_dense_mpo_repr(W1))
    ls1 = dot(dP1, dW1v) / dot(dP1, dP1)
    @test norm(dW1v .- ls1 .* dP1) / norm(dW1v) < 1e-8
    # 压缩路径：2·Wa·I（键 2）压到 D=1 精确（秩-1 目标），两算法。
    wa = randomimpo(T, [2, 2], 1)
    dwa = _dense_mpo_repr(wa)
    tgt = 2 .* dwa
    s2 = DenseIMPO(collect(exact_add(wa.Ws, wa.Ws)))
    for alg in (VOMPS(D = 1, maxiter = 200), IDMRG(D = 1, maxiter = 200))
        Random.seed!(1)
        Pc, ov = mult(s2, I2, alg)
        @test bonddim(Pc, 1) == 1
        d = _dense_mpo_repr(DenseIMPO(Pc))
        ls = dot(vec(d), vec(tgt)) / dot(vec(d), vec(d))
        @test norm(vec(tgt) .- ls .* vec(d)) / norm(vec(tgt)) < 1e-6
    end
end

@testset "hadamard：element-wise 乘积（物理维不变）" begin
    T = ComplexF64
    Random.seed!(45)
    ψ1 = randomimps(T, [2, 2], 3)
    ψ2 = randomimps(T, [2, 2], 3)
    c1 = _dense_mps_repr(ψ1)
    c2 = _dense_mps_repr(ψ2)

    # 精确路径（D = nothing）：波形 ∝ 逐点乘积 c1 .* c2，物理维不变、键维 = 3·3。
    # 注：朴素 zip 一般非规范，构造器规范化后与 c1.*c2 相差一个正实标量（射线代表）
    H12 = hadamard(ψ1, ψ2)
    @test H12 isa CanonicalIMPS
    @test phydims(H12) == [2, 2]
    @test max_bonddim(H12) == 9
    dH = vec(_dense_mps_repr(H12))
    tgt = vec(c1 .* c2)
    ls = real(dot(dH, tgt)) / real(dot(dH, dH))
    @test norm(tgt .- ls .* dH) / norm(tgt) < 1e-10

    # 压缩路径（两算法）：D = 9 = 精确键维 → 无损，与 exact 平行
    for alg in (VOMPS(D = 9, maxiter = 200), IDMRG(D = 9, maxiter = 200))
        Hc, _ = hadamard(ψ1, ψ2, alg)
        @test max_bonddim(Hc) == 9
        @test abs(dot(Hc, H12)) > 1 - 1e-8
    end

    # 长度 / 逐 site 物理维不匹配
    @test_throws DimensionMismatch hadamard(ψ1, randomimps(T, [2, 2, 2], 2))
    @test_throws DimensionMismatch hadamard(ψ1, randomimps(T, [3, 2], 3))
end
