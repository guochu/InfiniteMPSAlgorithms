# =====================================================================
# changebond! / compress / inplace (mult!/hadamard!/compress!) /
# svdguess_* 初始猜
# =====================================================================

@testset "_resize_dim" begin
    T = ComplexF64
    Random.seed!(76)
    A = randn(T, 2, 3, 2)

    # 扩容：保留首部子块，新增块零填充 / noise 填充
    B = InfiniteMPSAlgorithms._resize_dim(A, 1, 4; noise = 0)
    @test size(B) == (4, 3, 2) && B[1:2, :, :] == A && all(iszero, B[3:4, :, :])
    Bn = InfiniteMPSAlgorithms._resize_dim(A, 1, 4; noise = 1e-3)
    @test size(Bn) == (4, 3, 2) && Bn[1:2, :, :] == A && !all(iszero, Bn[3:4, :, :])

    # 缩键：只保留前导块
    C_ = InfiniteMPSAlgorithms._resize_dim(A, 3, 1)
    @test size(C_) == (2, 3, 1) && C_ == A[:, :, 1:1]

    # 尺寸不变时原样返回
    @test InfiniteMPSAlgorithms._resize_dim(A, 1, 2) === A
end

@testset "changebond!" begin
    T = ComplexF64
    Random.seed!(77)

    # 扩键（零填充，态不变）：prodimps 键 1 → 均匀 profile min(D, ∏d) = 4
    ψp = prodimps(T, [2, 3])
    ψref = deepcopy(ψp)
    changebond!(ψp; D = 4, noise = 0)
    @test bonddim(ψp, 1) == 4 && bonddim(ψp, 2) == 4
    @test ismixedcanonical(ψp)
    @test abs(dot(ψp, ψref)) / (norm(ψp) * norm(ψref)) ≈ 1 atol = 1e-10   # 零填充不改变态
    # noise 关键字（默认 1e-10）：扩容块被 noise·randn 填充、态只被 O(noise) 扰动
    ψpn = prodimps(T, [2, 3])
    ψpn_ref = deepcopy(ψpn)
    changebond!(ψpn; D = 4)
    @test bonddim(ψpn, 1) == 4 && bonddim(ψpn, 2) == 4
    @test ismixedcanonical(ψpn)
    @test abs(dot(ψpn, ψpn_ref)) / (norm(ψpn) * norm(ψpn_ref)) ≈ 1 atol = 1e-6

    # 缩键（截取前导奇异值子空间）：键 profile 达标、规范保持、保真度 = 逐 bond
    # 截断水平（块对角直和的逐 bond 截断不能精确回到分量——那需要变分压缩）
    ψ1 = randomimps(T, [2, 3], 4)
    ψsum = CanonicalIMPS(collect(exact_add(ψ1.AL, ψ1.AL)))
    @test max_bonddim(ψsum) == 8
    changebond!(ψsum; D = 4)
    @test max_bonddim(ψsum) == 4
    @test ismixedcanonical(ψsum)
    @test abs(dot(ψsum, ψ1)) / (norm(ψsum) * norm(ψ1)) > 0.8

    # 键 profile 已达标 ⇒ 提前返回，四个分量张量都不被改动
    # （ψ1 键 = 4 = min(4, ∏d = 6)，故 D = 4 命中提前返回）
    ψe = randomimps(T, [2, 3], 4)
    refe = deepcopy(ψe)
    changebond!(ψe; D = 4)
    @test ψe.AL[1] == refe.AL[1] && ψe.AR[1] == refe.AR[1] &&
          ψe.C[1] == refe.C[1] && ψe.AC[1] == refe.AC[1]
end

@testset "changebond!（MPO 版）" begin
    T = ComplexF64
    Random.seed!(79)
    ov(A, B) = abs(dot(vectorize(A), vectorize(B))) / (norm(vectorize(A)) * norm(vectorize(B)))

    # 强制键 profile：缩键与扩容两向都改到 min(D, feasible)
    W = CanonicalIMPO([randn(T, 6, 2, 6, 2), randn(T, 6, 2, 6, 2)])
    for D in (2, 4, 6, 16)
        W2 = deepcopy(W)
        changebond!(W2; D = D)
        @test all(bonddim(W2, ℓ) == D for ℓ in 1:2)
        @test ismixedcanonical(W2)
    end

    # 扩容（noise = 0）：态不变
    W0 = CanonicalIMPO([randn(T, 2, 2, 2, 2), randn(T, 2, 2, 2, 2)])
    W0ref = deepcopy(W0)
    changebond!(W0; D = 8, noise = 0)
    @test all(bonddim(W0, ℓ) == 8 for ℓ in 1:2)
    @test ismixedcanonical(W0)
    @test ov(W0, W0ref) ≈ 1 atol = 1e-10

    # noise 关键字（默认 1e-10）：态只被 O(noise) 扰动
    Wn = CanonicalIMPO([randn(T, 2, 2, 2, 2), randn(T, 2, 2, 2, 2)])
    Wnref = deepcopy(Wn)
    changebond!(Wn; D = 8)
    @test all(bonddim(Wn, ℓ) == 8 for ℓ in 1:2)
    @test ismixedcanonical(Wn)
    @test ov(Wn, Wnref) ≈ 1 atol = 1e-6

    # 键 profile 已达标 ⇒ 提前返回，四个分量张量都不被改动
    W6 = CanonicalIMPO([randn(T, 6, 2, 6, 2), randn(T, 6, 2, 6, 2)])
    ref6 = deepcopy(W6)
    changebond!(W6; D = 6)
    @test W6.AL[1] == ref6.AL[1] && W6.AR[1] == ref6.AR[1] &&
          W6.C[1] == ref6.C[1] && W6.AC[1] == ref6.AC[1]
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
