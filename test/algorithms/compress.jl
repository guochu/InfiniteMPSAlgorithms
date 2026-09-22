# =====================================================================
# changebond! / compress / inplace (mult!/hadamard!/compress!) /
# svdguess_* 初始猜
# =====================================================================

@testset "changebond!" begin
    T = ComplexF64
    Random.seed!(77)

    # 扩键（零填充，态不变）：prodimps 键 1 → 均匀 profile min(D, ∏d) = 4
    ψp = prodimps(T, [2, 3])
    changebond!(ψp; D = 4)
    @test bonddim(ψp, 1) == 4 && bonddim(ψp, 2) == 4
    @test ismixedcanonical(ψp)
    @test abs(dot(ψp, ψp)) ≈ 1 atol = 1e-10            # 零填充不改变态

    # 缩键（截取前导奇异值子空间）：键 profile 达标、规范保持、保真度 = 逐 bond
    # 截断水平（块对角直和的逐 bond 截断不能精确回到分量——那需要变分压缩）
    ψ1 = randomimps(T, [2, 3], 4)
    ψsum = CanonicalIMPS(collect(exact_add(ψ1.AL, ψ1.AL)))
    @test max_bonddim(ψsum) == 8
    changebond!(ψsum; D = 4)
    @test max_bonddim(ψsum) == 4
    @test ismixedcanonical(ψsum)
    @test abs(dot(ψsum, ψ1)) / (norm(ψsum) * norm(ψ1)) > 0.8
end

@testset "compress" begin
    T = ComplexF64
    Random.seed!(78)
    # 可精确表示的目标：ψ1 的自直和（键 2D，波形 = 2ψ1）
    ψ1 = randomimps(T, [2, 3], 4)
    ψsum = CanonicalIMPS(collect(exact_add(ψ1.AL, ψ1.AL)))
    @test max_bonddim(ψsum) == 8

    # D ≥ 输入键：短路精确返回（无压缩，overlap = N）
    y0, ov0 = compress(ψsum; D = 8)
    @test max_bonddim(y0) == 8
    @test real(ov0) ≈ length(ψsum) atol = 1e-10

    # 有损压缩：overlap ∈ (0, N]，且不劣于随机初猜的同键压缩
    ψbig = randomimps(T, [2, 2], 8)
    y4, ov4 = compress(ψbig; D = 4)
    @test max_bonddim(y4) == 4 && 0 < real(ov4) ≤ length(ψbig) + 1e-12
    yr, ovr = compress(ψbig; D = 4, x0 = randomimps(T, [2, 2], 4))
    @test real(ov4) ≥ real(ovr) - 1e-9

    # IDMRG 路径同一不动点（有损压缩）
    ψbig = randomimps(T, [2, 2], 8)
    y4, ov4 = compress(ψbig; D = 4)
    y4b, ov4b = compress(ψbig; D = 4, alg = IDMRG(maxiter = 200))
    @test abs(real(ov4b) - real(ov4)) < 1e-4

    # MPO 版：CanonicalIMPO 与 DenseIMPO（随机谱平缓，两者均为重叠最大化
    # 收敛停点，断言一致到收敛差异内）
    W = randomimpo(T, [2, 2], 6)
    Wc = CanonicalIMPO(collect(W.Ws))
    Vc, ovc = compress(Wc; D = 3)
    @test max_bonddim(Vc) == 3 && ismixedcanonical(Vc)
    Vd, ovd = compress(W; D = 3)
    @test max_bonddim(Vd) == 3
    @test abs(real(ovc) - real(ovd)) < 0.1
end

@testset "inplace: mult! / hadamard! / compress!" begin
    T = ComplexF64
    Random.seed!(79)
    ψ1 = randomimps(T, [2, 2], 4)
    ψ2 = randomimps(T, [2, 2], 4)

    # hadamard!
    out = randomimps(T, [2, 2], 2)
    hadamard!(out, ψ1, ψ2; D = 4)
    yh, ovh = hadamard(ψ1, ψ2; D = 4)
    @test abs(dot(out, yh)) / sqrt(abs(dot(out, out)) * abs(dot(yh, yh))) ≈ 1 atol = 1e-8

    # mult!（mpo·mps）
    W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    out = randomimps(T, [2, 2], 2)
    mult!(out, W, ψ1; D = 4)
    ym, ovm = mult(W, ψ1; D = 4)
    @test abs(dot(out, ym)) / sqrt(abs(dot(out, out)) * abs(dot(ym, ym))) ≈ 1 atol = 1e-8

    # mult!（mpo·mpo）
    W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    outo = CanonicalIMPO([randn(T, 2, 2, 2, 2), randn(T, 2, 2, 2, 2)])
    mult!(outo, W, W2; D = 3)
    yo, ovo = mult(W, W2; D = 3)
    @test max_bonddim(outo) == 3 && ismixedcanonical(outo)
    # 幅值无关的射线比较（转移半径自身归一）
    vo, vy = vectorize(outo), vectorize(yo)
    @test abs(dot(vo, vy)) /
          sqrt(abs(dot(vo, vo)) * abs(dot(vy, vy))) ≈ 1 atol = 1e-8

    # compress!
    ψbig = randomimps(T, [2, 2], 8)
    out = randomimps(T, [2, 2], 4)
    compress!(out, ψbig; D = 4)
    yc, ovc = compress(ψbig; D = 4)
    @test abs(dot(out, yc)) / sqrt(abs(dot(out, out)) * abs(dot(yc, yc))) ≈ 1 atol = 1e-8
end

@testset "svdguess_*" begin
    T = ComplexF64
    Random.seed!(80)
    ψ1 = randomimps(T, [2, 2], 3)
    ψ2 = randomimps(T, [2, 2], 3)

    gh = svdguess_hadamard(ψ1, ψ2, 5)
    @test max_bonddim(gh) ≤ 5 && ismixedcanonical(gh)

    W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    gm = svdguess_mult(W, ψ1, 4)
    @test max_bonddim(gm) ≤ 4 && ismixedcanonical(gm)
    W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    gmm = svdguess_mult(W, W2, 5)
    @test max_bonddim(gmm) ≤ 5 && ismixedcanonical(gmm)

    ψbig = randomimps(T, [2, 2], 8)
    gc = svdguess_compress(ψbig, 4)
    @test max_bonddim(gc) == 4 && ismixedcanonical(gc)

    # svdguess 初猜不劣于随机初猜（overlap 最大化扫描单调）
    y_svd, ov_svd = mult(W, ψ1; D = 3)
    y_rnd, ov_rnd = mult(W, ψ1; D = 3, ψ₀ = randomimps(T, [2, 2], 3))
    @test real(ov_svd) ≥ real(ov_rnd) - 1e-9
end
