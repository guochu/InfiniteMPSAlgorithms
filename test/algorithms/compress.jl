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
    y0, ov0 = compress(ψsum, VOMPS(D = 8))
    @test max_bonddim(y0) == 8
    @test real(ov0) ≈ length(ψsum) atol = 1e-10

    # 有损压缩：overlap ∈ (0, N]，且不劣于随机初猜的同键压缩（随机初猜经
    # in-place 版本提供）
    ψbig = randomimps(T, [2, 2], 8)
    y4, ov4 = compress(ψbig, VOMPS(D = 4))
    @test max_bonddim(y4) == 4 && 0 < real(ov4) ≤ length(ψbig) + 1e-12
    yr = randomimps(T, [2, 2], 4)
    compress!(yr, ψbig, VOMPS(D = 4))
    @test abs(real(dot(yr, y4))) / sqrt(abs(dot(yr, yr)) * abs(dot(y4, y4))) ≈ 1 atol = 1e-6

    # IDMRG 路径同一不动点（有损压缩）
    ψbig = randomimps(T, [2, 2], 8)
    y4, ov4 = compress(ψbig, VOMPS(D = 4))
    y4b, ov4b = compress(ψbig, IDMRG(D = 4, maxiter = 200))
    @test abs(real(ov4b) - real(ov4)) < 1e-4

    # MPO 版：CanonicalIMPO 与 DenseIMPO（随机谱平缓，两者均为重叠最大化
    # 收敛停点，断言一致到收敛差异内）；DenseIMPO 输入同样返回 CanonicalIMPO
    W = randomimpo(T, [2, 2], 6)
    Wc = CanonicalIMPO(collect(W.Ws))
    Vc, ovc = compress(Wc, VOMPS(D = 3))
    @test max_bonddim(Vc) == 3 && ismixedcanonical(Vc)
    Vd, ovd = compress(W, VOMPS(D = 3))
    @test Vd isa CanonicalIMPO && max_bonddim(Vd) == 3 && ismixedcanonical(Vd)
    @test abs(real(ovc) - real(ovd)) < 0.1
end

@testset "compress：MPO 输入一律返回 CanonicalIMPO 且 ismixedcanonical" begin
    # 确认点 1：naive 兜底 / 短路不得直接透出 DenseIMPO——MPO 结果一律混合正则
    T = ComplexF64
    Random.seed!(81)
    Wr = randomimpo(T, [2, 2], 3)
    # 压缩路径：DenseIMPO 输入 → CanonicalIMPO
    y2, ov2 = compress(Wr, VOMPS(D = 2))
    @test y2 isa CanonicalIMPO && max_bonddim(y2) == 2 && ismixedcanonical(y2)
    # IDMRG 路径同
    y2b, _ = compress(Wr, IDMRG(D = 2, maxiter = 200))
    @test y2b isa CanonicalIMPO && ismixedcanonical(y2b)
    # D ≥ max_bonddim：短路——输入先转规范存储再返回（强制归一化非纯规范，
    # 算符值整体缩放但射线严格不变），仍为 CanonicalIMPO
    yr0, ov0 = compress(Wr, VOMPS(D = 3))
    @test yr0 isa CanonicalIMPO && ismixedcanonical(yr0)
    @test real(ov0) ≈ 2 atol = 1e-10
    dr = vec(_dense_mpo_repr(DenseIMPO(yr0)))
    dw = vec(_dense_mpo_repr(Wr))
    ls = dot(dr, dw) / dot(dr, dr)
    @test norm(dw .- ls .* dr) / norm(dw) < 1e-9
    # 实数输入：正则性成立；标量类型随通道转移的实际 eltype（ALS 期间融合
    # 转移可为复主导，通道按 MPSKit 对齐提升为复，故不断言实性）
    Wf = randomimpo(Float64, [2, 2], 3)
    yf, _ = compress(Wf, VOMPS(D = 2))
    @test yf isa CanonicalIMPO && ismixedcanonical(yf)
end

@testset "inplace: mult! / hadamard! / compress!" begin
    T = ComplexF64
    Random.seed!(79)
    ψ1 = randomimps(T, [2, 2], 4)
    ψ2 = randomimps(T, [2, 2], 4)

    # hadamard!（初猜 out 提供 D）
    out = randomimps(T, [2, 2], 4)
    hadamard!(out, ψ1, ψ2, VOMPS(D = 4))
    yh, ovh = hadamard(ψ1, ψ2, VOMPS(D = 4))
    @test abs(dot(out, yh)) / sqrt(abs(dot(out, out)) * abs(dot(yh, yh))) ≈ 1 atol = 1e-8

    # mult!（mpo·mps，初猜 out 提供 D）
    W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    out = randomimps(T, [2, 2], 4)
    mult!(out, W, ψ1, VOMPS(D = 4))
    ym, ovm = mult(W, ψ1, VOMPS(D = 4))
    @test abs(dot(out, ym)) / sqrt(abs(dot(out, out)) * abs(dot(ym, ym))) ≈ 1 atol = 1e-8

    # mult!（mpo·mpo）
    W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    outo = CanonicalIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    mult!(outo, W, W2, VOMPS(D = 3))
    yo, ovo = mult(W, W2, VOMPS(D = 3))
    @test max_bonddim(outo) == 3 && ismixedcanonical(outo)
    # 幅值无关的射线比较（转移半径自身归一）
    vo, vy = vectorize(outo), vectorize(yo)
    @test abs(dot(vo, vy)) /
          sqrt(abs(dot(vo, vo)) * abs(dot(vy, vy))) ≈ 1 atol = 1e-8

    # compress!（初猜 out 提供 D）
    ψbig = randomimps(T, [2, 2], 8)
    out = randomimps(T, [2, 2], 4)
    compress!(out, ψbig, VOMPS(D = 4))
    yc, ovc = compress(ψbig, VOMPS(D = 4))
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

    # svdguess 初猜（默认）与随机初猜（in-place 的 out 提供）都应收敛到
    # 高保真度的压缩结果（与精确构造射线的保真度 ≥ 0.9·N）
    y_exact, _ = mult(W, ψ1)                       # 精确朴素构造（二参数版本）
    y_svd, ov_svd = mult(W, ψ1, VOMPS(D = 3))      # svdguess 初猜
    @test real(ov_svd) > 0.9 * length(ψ1)
    out = randomimps(T, [2, 2], 3)
    mult!(out, W, ψ1, VOMPS(D = 3))                # 随机初猜（in-place）
    @test fidelity(out, y_exact) > 0.9
end
