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
    @test reshape(_dense_mpo_repr(InfiniteMPO(P)), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10

    # mpo*mps：键维 = W 键维 × ψ 键维，波形 ∝ W·ψ（稠密算符作用在波形上）
    Kψ = exact_mult(W1.Ws, ψ1.AL)
    @test Kψ isa PeriodicVector && length(Kψ) == 2
    @test size(Kψ[1]) == (6, 2, 6)
    cψf = vec(_dense_mps_repr(InfiniteCanonicalMPS(collect(Kψ))))
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
    @test _dense_mps_repr(InfiniteCanonicalMPS(collect(K))) ≈
          _dense_mps_repr(ψ1) + _dense_mps_repr(ψ2) atol = 1e-10

    K4 = exact_add(W1.Ws, W2.Ws)
    @test size(K4[1]) == (5, 2, 5, 2)
    @test _dense_mpo_repr(InfiniteMPO(K4)) ≈
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

@testset "add：MPS 加法（朴素精确，对标 MPSKit）" begin
    T = ComplexF64
    Random.seed!(42)
    ψu = prodimps(T, [2, 2], [1, 1])   # |00⟩
    ψd = prodimps(T, [2, 2], [2, 2])   # |11⟩
    s = add(ψu, ψd)
    @test s isa InfiniteCanonicalMPS && max_bonddim(s) == 2
    # 波形 = |00⟩ + |11⟩（trace 表示的振幅矩阵 = 单位阵）
    @test _dense_mps_repr(s) ≈ Matrix{T}(I, 2, 2) atol = 1e-12
    # 长度 / 逐 site 物理维不匹配
    @test_throws DimensionMismatch add(ψu, randomimps(T, [2, 2, 2], 2))
    @test_throws DimensionMismatch add(ψu, randomimps(T, [3, 2], 2))
end

@testset "add：MPS 加法压缩（VOMPS 与 IDMRG）" begin
    T = ComplexF64
    Random.seed!(42)
    ψ1 = randomimps(T, [2, 2], 3)
    d1 = _dense_mps_repr(ψ1)
    for alg in (VOMPS(maxiter = 200), IDMRG(maxiter = 200))
        Random.seed!(1)
        s2 = add(ψ1, ψ1; D = 3, alg = alg)      # ψ1 + ψ1 = 2ψ1（键 6 → 3 无损）
        @test max_bonddim(s2) == 3
        # 同射线：|dot| = 1（Cauchy–Schwarz 饱和）。相位是规范自由度：
        # IDMRG 的 C 链本征解相位不钉定，cross-dot 可带 twist 相位（VOMPS 继承
        # ket 相位故恰好为 1）；环迹输出公式中 twist 自动抵消，不影响幅值。
        @test abs(dot(s2, ψ1)) ≈ 1 atol = 1e-6
        # 波形与 ψ1 平行（twist 相位鲁棒）：因子 2 被射线规范化吸收
        ds2 = _dense_mps_repr(s2)
        @test abs(dot(vec(ds2), vec(d1))) / (norm(ds2) * norm(d1)) ≈ 1 atol = 1e-6
    end
end

@testset "add：MPO 加法（朴素精确 + 压缩）" begin
    T = ComplexF64
    Random.seed!(43)
    W1 = randomimpo(T, [2, 2], 2)
    W2 = randomimpo(T, [2, 2], 3)
    dW1 = _dense_mpo_repr(W1)
    dW2 = _dense_mpo_repr(W2)
    # 朴素：键维直和、稠密算符可加
    s = add(W1, W2)
    @test s isa InfiniteMPO && bonddim(s, 1) == 5 && bonddim(s, 2) == 5
    @test _dense_mpo_repr(s) ≈ dW1 + dW2 atol = 1e-10
    # 与 -W1 相加 = 零（符号经首张量缩放折入，MPSKit 标量乘约定）
    z = add(W1, -W1)
    @test _dense_mpo_repr(z) ≈ zeros(T, 2, 2, 2, 2) atol = 1e-10
    # 压缩：I + I = 2I（键 2 → 1 精确，两算法）。与目标平行即可——整体相位是
    # 规范自由度不作要求（幅值要求保留在复比例残差中）
    I2 = identityimpo(T, [2, 2])
    dI = _dense_mpo_repr(I2)
    tgt = 2 .* dI
    for alg in (VOMPS(maxiter = 200), IDMRG(maxiter = 200))
        Random.seed!(1)
        s2 = add(I2, I2; D = 1, alg = alg)
        @test bonddim(s2, 1) == 1
        d = _dense_mpo_repr(s2)
        ls = dot(vec(d), vec(tgt)) / dot(vec(d), vec(d))
        @test norm(vec(tgt) .- ls .* vec(d)) / norm(vec(tgt)) < 1e-6
    end
end

@testset "mult：MPO 乘法（朴素精确对照 + 压缩）" begin
    T = ComplexF64
    Random.seed!(44)
    W1 = randomimpo(T, [2, 2], 2)
    W2 = randomimpo(T, [2, 2], 3)
    I2 = identityimpo(T, [2, 2])
    # 朴素乘法 = 稠密算符矩阵乘法，键维 = 两键维乘积
    P = exact_mult(W1.Ws, W2.Ws)
    @test bonddim(InfiniteMPO(P), 1) == 6
    @test reshape(_dense_mpo_repr(InfiniteMPO(P)), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10
    # 迭代乘法：W1·I = W1（D = 2 = 目标键维）。与目标平行即可（整体相位/尺度是
    # 输出射线规范的自由度）
    P1, ov1 = mult(W1, I2; D = 2)
    @test P1 isa InfiniteCanonicalMPO && bonddim(P1, 1) == 2
    @test real(ov1) > 2 - 1e-6   # overlap = N 即方向一致
    dP1 = vec(_dense_mpo_repr(InfiniteMPO(P1)))
    dW1v = vec(_dense_mpo_repr(W1))
    ls1 = dot(dP1, dW1v) / dot(dP1, dP1)
    @test norm(dW1v .- ls1 .* dP1) / norm(dW1v) < 1e-8
    # 压缩路径：2·Wa·I（键 2）压到 D=1 精确（秩-1 目标），两算法。
    wa = randomimpo(T, [2, 2], 1)
    dwa = _dense_mpo_repr(wa)
    tgt = 2 .* dwa
    s2 = add(wa, wa)
    for alg in (VOMPS(maxiter = 200), IDMRG(maxiter = 200))
        Random.seed!(1)
        Pc, ov = mult(s2, I2; D = 1, alg = alg)
        @test bonddim(Pc, 1) == 1
        d = _dense_mpo_repr(InfiniteMPO(Pc))
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
    @test H12 isa InfiniteCanonicalMPS
    @test phydims(H12) == [2, 2]
    @test max_bonddim(H12) == 9
    dH = vec(_dense_mps_repr(H12))
    tgt = vec(c1 .* c2)
    ls = real(dot(dH, tgt)) / real(dot(dH, dH))
    @test norm(tgt .- ls .* dH) / norm(tgt) < 1e-10

    # 压缩路径（两算法）：D = 9 = 精确键维 → 无损，与 exact 平行
    for alg in (VOMPS(maxiter = 200), IDMRG(maxiter = 200))
        Hc = hadamard(ψ1, ψ2; D = 9, alg = alg)
        @test max_bonddim(Hc) == 9
        @test abs(dot(Hc, H12)) > 1 - 1e-8
    end

    # 长度 / 逐 site 物理维不匹配
    @test_throws DimensionMismatch hadamard(ψ1, randomimps(T, [2, 2, 2], 2))
    @test_throws DimensionMismatch hadamard(ψ1, randomimps(T, [3, 2], 3))
end
