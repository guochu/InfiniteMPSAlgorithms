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

    # 扩键（零填充，态不变）：prodimps 键 1 → 均匀 profile D = 4（∏d = 6 ≥ 4）
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
    ψ1 = randomimps(T, [2, 3]; D = 4)
    ψsum = CanonicalIMPS([cat(ψ1.AL[ℓ], ψ1.AL[ℓ]; dims = (1, 3)) for ℓ in 1:length(ψ1)])
    @test max_bonddim(ψsum) == 8
    changebond!(ψsum; D = 4)
    @test max_bonddim(ψsum) == 4
    @test ismixedcanonical(ψsum)
    @test abs(dot(ψsum, ψ1)) / (norm(ψsum) * norm(ψ1)) > 0.8

    # 键 profile 已达标 ⇒ 提前返回，四个分量张量都不被改动
    # （ψ1 键 = 4 = D，故 D = 4 命中提前返回）
    ψe = randomimps(T, [2, 3]; D = 4)
    refe = deepcopy(ψe)
    changebond!(ψe; D = 4)
    @test ψe.AL[1] == refe.AL[1] && ψe.AR[1] == refe.AR[1] &&
          ψe.C[1] == refe.C[1] && ψe.AC[1] == refe.AC[1]

    # D 不受物理维乘积限制：infinite MPS 忠实按用户给的 D
    # （回归：曾被截到 min(D, ∏d)）
    ψcap = prodimps(T, [2, 3])                      # ∏d = 6
    changebond!(ψcap; D = 32, noise = 0)
    @test all(bonddim(ψcap, ℓ) == 32 for ℓ in 1:2)
    @test ismixedcanonical(ψcap)
end

@testset "changebond!（MPO 版）" begin
    T = ComplexF64
    Random.seed!(79)
    ov(A, B) = abs(dot(vectorize(A), vectorize(B))) / (norm(vectorize(A)) * norm(vectorize(B)))

    # 强制键 profile：缩键与扩容两向都改到 D
    W = CanonicalIMPO([randn(T, 6, 2, 6, 2), randn(T, 6, 2, 6, 2)])
    for D in (2, 4, 6, 16)
        W2 = deepcopy(W)
        changebond!(W2; D = D)
        @test all(bonddim(W2, ℓ) == D for ℓ in 1:2)
        @test ismixedcanonical(W2)
    end
    # D 不受物理维乘积限制（MPO 的 MPS 视图物理维 = du·dd = 4，∏ = 16）
    Wcap = CanonicalIMPO([randn(T, 2, 2, 2, 2), randn(T, 2, 2, 2, 2)])
    changebond!(Wcap; D = 32)
    @test all(bonddim(Wcap, ℓ) == 32 for ℓ in 1:2)
    @test ismixedcanonical(Wcap)

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
    ψ1 = randomimps(T, [2, 3]; D = 4)
    ψsum = CanonicalIMPS([cat(ψ1.AL[ℓ], ψ1.AL[ℓ]; dims = (1, 3)) for ℓ in 1:length(ψ1)])
    @test max_bonddim(ψsum) == 8

    # D ≥ 输入键：短路精确返回（无压缩）
    y0 = compress(ψsum, VOMPS(D = 8))
    @test max_bonddim(y0) == 8
    @test fidelity(y0, ψsum) ≈ 1 atol = 1e-10

    # 有损压缩：不劣于随机初猜的同键压缩（随机初猜经 in-place 版本提供）
    ψbig = randomimps(T, [2, 2]; D = 8)
    y4 = compress(ψbig, VOMPS(D = 4))
    @test max_bonddim(y4) == 4
    yr = randomimps(T, [2, 2]; D = 4)
    compress!(yr, ψbig, VOMPS(D = 4))
    @test abs(real(dot(yr, y4))) / sqrt(abs(dot(yr, yr)) * abs(dot(y4, y4))) ≈ 1 atol = 1e-6

    # IDMRG 路径同一不动点（有损压缩，射线方向一致）
    ψbig = randomimps(T, [2, 2]; D = 8)
    y4 = compress(ψbig, VOMPS(D = 4))
    y4b = compress(ψbig, IDMRG(D = 4, maxiter = 200))
    @test abs(real(dot(y4, y4b))) /
          sqrt(abs(dot(y4, y4)) * abs(dot(y4b, y4b))) ≈ 1 atol = 1e-3

    # MPO 版：CanonicalIMPO 与 DenseIMPO（随机谱平缓，两者均为重叠最大化
    # 收敛停点，断言一致到收敛差异内）；DenseIMPO 输入同样返回 CanonicalIMPO
    W = randomimpo(T, [2, 2]; D = 6)
    Wc = CanonicalIMPO(collect(W.Ws))
    Vc = compress(Wc, VOMPS(D = 3))
    @test max_bonddim(Vc) == 3 && ismixedcanonical(Vc)
    Vd = compress(W, VOMPS(D = 3))
    @test Vd isa CanonicalIMPO && max_bonddim(Vd) == 3 && ismixedcanonical(Vd)
    @test fidelity(Vc, Vd) > 0.9
end

@testset "compress：MPO 输入一律返回 CanonicalIMPO 且 ismixedcanonical" begin
    # 确认点 1：naive 兜底 / 短路不得直接透出 DenseIMPO——MPO 结果一律混合正则
    T = ComplexF64
    Random.seed!(81)
    Wr = randomimpo(T, [2, 2]; D = 3)
    # 压缩路径：DenseIMPO 输入 → CanonicalIMPO
    y2 = compress(Wr, VOMPS(D = 2))
    @test y2 isa CanonicalIMPO && max_bonddim(y2) == 2 && ismixedcanonical(y2)
    # IDMRG 路径同
    y2b = compress(Wr, IDMRG(D = 2, maxiter = 200))
    @test y2b isa CanonicalIMPO && ismixedcanonical(y2b)
    # D ≥ max_bonddim：短路——输入先转规范存储再返回（强制归一化非纯规范，
    # 算符值整体缩放但射线严格不变），仍为 CanonicalIMPO
    yr0 = compress(Wr, VOMPS(D = 3))
    @test yr0 isa CanonicalIMPO && ismixedcanonical(yr0)
    dr = vec(_dense_mpo_repr(DenseIMPO(yr0)))
    dw = vec(_dense_mpo_repr(Wr))
    ls = dot(dr, dw) / dot(dr, dr)
    @test norm(dw .- ls .* dr) / norm(dw) < 1e-9
    # 实数输入：正则性成立；标量类型随通道转移的实际 eltype（ALS 期间融合
    # 转移可为复主导，通道按 MPSKit 对齐提升为复，故不断言实性）
    Wf = randomimpo(Float64, [2, 2]; D = 3)
    yf = compress(Wf, VOMPS(D = 2))
    @test yf isa CanonicalIMPO && ismixedcanonical(yf)
end

@testset "inplace: mult! / hadamard! / compress!" begin
    T = ComplexF64
    Random.seed!(79)
    ψ1 = randomimps(T, [2, 2]; D = 4)
    ψ2 = randomimps(T, [2, 2]; D = 4)

    # hadamard!（初猜 out 提供 D）
    out = randomimps(T, [2, 2]; D = 4)
    hadamard!(out, ψ1, ψ2, VOMPS(D = 4))
    yh = hadamard(ψ1, ψ2, VOMPS(D = 4))
    @test abs(dot(out, yh)) / sqrt(abs(dot(out, out)) * abs(dot(yh, yh))) ≈ 1 atol = 1e-8

    # mult!（mpo·mps，初猜 out 提供 D）
    W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    out = randomimps(T, [2, 2]; D = 4)
    mult!(out, W, ψ1, VOMPS(D = 4))
    ym = mult(W, ψ1, VOMPS(D = 4))
    @test abs(dot(out, ym)) / sqrt(abs(dot(out, out)) * abs(dot(ym, ym))) ≈ 1 atol = 1e-8

    # mult!（mpo·mpo）
    W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    outo = CanonicalIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    mult!(outo, W, W2, VOMPS(D = 3))
    yo = mult(W, W2, VOMPS(D = 3))
    @test max_bonddim(outo) == 3 && ismixedcanonical(outo)
    # 幅值无关的射线比较（转移半径自身归一）
    vo, vy = vectorize(outo), vectorize(yo)
    @test abs(dot(vo, vy)) /
          sqrt(abs(dot(vo, vo)) * abs(dot(vy, vy))) > 0.9  # 同一吸引域内的方向一致

    # compress!（初猜 out 提供 D）
    ψbig = randomimps(T, [2, 2]; D = 8)
    out = randomimps(T, [2, 2]; D = 4)
    compress!(out, ψbig, VOMPS(D = 4))
    yc = compress(ψbig, VOMPS(D = 4))
    @test abs(dot(out, yc)) / sqrt(abs(dot(out, out)) * abs(dot(yc, yc))) ≈ 1 atol = 1e-8
end

@testset "svdguess_*" begin
    T = ComplexF64
    Random.seed!(80)
    ψ1 = randomimps(T, [2, 2]; D = 3)
    ψ2 = randomimps(T, [2, 2]; D = 3)

    gh = svdguess_hadamard(ψ1, ψ2, 5)
    @test max_bonddim(gh) ≤ 5 && ismixedcanonical(gh)

    W = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    gm = svdguess_mult(W, ψ1, 4)
    @test max_bonddim(gm) ≤ 4 && ismixedcanonical(gm)
    W2 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    gmm = svdguess_mult(W, W2, 5)
    @test max_bonddim(gmm) ≤ 5 && ismixedcanonical(gmm)

    ψbig = randomimps(T, [2, 2]; D = 8)
    gc = svdguess_compress(ψbig, 4)
    @test max_bonddim(gc) == 4 && ismixedcanonical(gc)

    # lazy（流式）构造：naive 乘积的 site tensor 现算、自右向左 SVD 截断，
    # 整条 naive 串从不 materialize（回归：曾先建整条串再规范化+截断）
    ψa = randomimps(T, [2, 2]; D = 4)
    ψb = randomimps(T, [2, 2]; D = 4)
    nv = CanonicalIMPS([InfiniteMPSAlgorithms._naive_hadamard_tensor(ψa.AL[ℓ], ψb.AL[ℓ])
                        for ℓ in 1:2])                       # 键 rank ≤ 16
    ge = svdguess_hadamard(ψa, ψb, 16)                        # D ≥ rank ⇒ 无截断 ⇒ 精确
    @test abs(dot(ge, nv)) / (norm(ge) * norm(nv)) ≈ 1 atol = 1e-10
    @test max_bonddim(ge) ≤ 16 && ismixedcanonical(ge)
    gd = svdguess_hadamard(ψa, ψb, 4)                         # D < rank ⇒ 截断初猜
    @test max_bonddim(gd) ≤ 4 && ismixedcanonical(gd)
    @test abs(dot(gd, nv)) / (norm(gd) * norm(nv)) > 0.5
    # _lazy_svd_guess：carry 在构造时被 site 吸收（这里用朴素吸收作驱动层测试），
    # wrap 键在 site 1 上 Schmidt 截断 ⇒ 输出的每个键都 ≤ D；site 2:L 右规范
    ψc = randomimps(T, [2, 2, 2]; D = 4)
    site = (ℓ, carry) -> begin
        B = InfiniteMPSAlgorithms._naive_hadamard_tensor(ψa.AL[ℓ], ψc.AL[ℓ])
        carry === nothing && return B
        @tensor B2[p, s2, f] := B[p, s2, q] * carry[q, f]
        return B2
    end
    out3 = InfiniteMPSAlgorithms._lazy_svd_guess(site, 3, 3)
    @test all(size(A, 1) ≤ 3 && size(A, 3) ≤ 3 for A in out3)
    @test size(out3[1], 1) == size(out3[3], 3)              # 两侧看到同一个 wrap 键
    for A in out3[2:end-1]
        m = reshape(A, size(A, 1), :)
        @test m * m' ≈ I atol = 1e-10                       # 中间站右等距（右规范）
    end
    # site L = v∘uᵀ（wrap 键 Schmidt 因子）：u 非方阵时不严格等距，只查键维
    # （态的正确性由上面的 fid 断言与 CanonicalIMPS 重新规范保证）
    # 裸张量串（PeriodicVector）版本：下游底层入口；Canonical 方法 = 其输出
    # 重新规范化（逐位套壳 ⇒ 与 Canonical 版本保真度 1）
    gv = svdguess_hadamard(ψa.AL, ψb.AL, 4)
    @test gv isa Vector{<:Array{T,3}}
    @test all(size(A, 1) ≤ 4 && size(A, 3) ≤ 4 for A in gv)
    @test fidelity(CanonicalIMPS(gv), gd) ≈ 1 atol = 1e-10
    gmv = svdguess_mult(PeriodicVector(W.Ws), ψ1.AL, 4)
    @test fidelity(CanonicalIMPS(gmv), gm) ≈ 1 atol = 1e-10
    W3 = DenseIMPO([randn(T, 3, 2, 3, 2), randn(T, 3, 2, 3, 2)])
    nvm = CanonicalIMPS([fuse(W3[ℓ], ψa.AL[ℓ]) for ℓ in 1:2])
    gme = svdguess_mult(W3, ψa, 9)                            # 键 rank ≤ 12 ⇒ D = 9 有截断
    @test max_bonddim(gme) ≤ 9 && ismixedcanonical(gme)
    @test abs(dot(gme, nvm)) / (norm(gme) * norm(nvm)) > 0.5

    # svdguess 初猜（默认）与随机初猜（in-place 的 out 提供）都应收敛到
    # 高保真度的压缩结果（与精确构造射线的保真度 ≥ 0.9·N）
    y_exact = mult(W, ψ1)                          # 精确朴素构造（二参数版本）
    y_svd = mult(W, ψ1, VOMPS(D = 3))              # svdguess 初猜
    @test fidelity(y_svd, y_exact) > 0.9           # 方向一致（幅值无意义）
    out = randomimps(T, [2, 2]; D = 3)
    mult!(out, W, ψ1, VOMPS(D = 3))                # 随机初猜（in-place）
    @test fidelity(out, y_exact) > 0.9
end
