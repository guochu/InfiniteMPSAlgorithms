# =====================================================================
# 算术运算测试：严格运算（DenseIMPO * / ⊙(DenseIMPS)）+ 迭代压缩 +
# mult!/compress!/hadamard! 的完全对齐
#
# - DenseIMPO * DenseIMPO / DenseIMPO * DenseIMPS / ⊙(DenseIMPS)：
#   严格运算（不压缩、不正则化），周期 trace 表示下验证算符乘积 / 波形作用 /
#   逐点乘积结构；
# - copyphyims：DenseIMPS→DenseIMPO 的物理腿复制（对齐 FiniteMPSAlgorithms
#   的 copyphydims），⊙ ≡ mult(copyphyims(ψ1), ψ2)；
# - mult! ≡ compress!（同初态+同参数：同迭代数、终态一致）；
# - 变分压缩（D::Int）验证 VOMPS 与 IDMRG 两条算法路径收敛到同一结果；
# - 数值断言基于周期 trace 表示（_dense_mps_repr / _dense_mpo_repr，定义于
#   testhelpers.jl；规范变换下望远相消，是纯规范不变的波形/算符表示）。
# =====================================================================

@testset "严格运算：DenseIMPO * / ⊙(DenseIMPS)（debug 基准）" begin
    T = ComplexF64
    Random.seed!(47)
    W1 = rand_denseimpo(T, [2, 2]; D = 2)
    W2 = rand_denseimpo(T, [2, 2]; D = 3)
    ψ1 = randomimps(T, [2, 2]; D = 3)

    # mpo*mpo：键维 = 两键维乘积，稠密算符 = 矩阵乘积，返回原始 DenseIMPO
    P = W1 * W2
    @test P isa DenseIMPO && length(P) == 2
    @test size(P[1]) == (6, 2, 6, 2)
    @test reshape(_dense_mpo_repr(P), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10

    # mpo*mps：键维 = W 键维 × ψ 键维，波形 ∝ W·ψ（稠密算符作用在波形上），
    # 返回原始 DenseIMPS
    Kψ = W1 * DenseIMPS(collect(ψ1.AL))
    @test Kψ isa DenseIMPS && length(Kψ) == 2
    @test size(Kψ[1]) == (6, 2, 6)
    cψf = vec(_dense_mps_repr(Kψ))
    M = reshape(_dense_mpo_repr(W1), 4, 4)
    dψ = vec(_dense_mps_repr(ψ1))
    ls = dot(M * dψ, cψf) / dot(cψf, cψf)
    @test norm(M * dψ .- ls .* cψf) / norm(M * dψ) < 1e-10
end

@testset "⊙(DenseIMPS)：波形逐点乘积（debug 基准）" begin
    T = ComplexF64
    Random.seed!(49)
    ψ1 = randomimps(T, [2, 2]; D = 3)
    ψ2 = randomimps(T, [2, 2]; D = 4)
    K = DenseIMPS(collect(ψ1.AL)) ⊙ DenseIMPS(collect(ψ2.AL))
    @test K isa DenseIMPS
    @test size(K[1]) == (12, 2, 12)   # 物理维不变，键维 = 3·4
    # 原始 zip 张量串的周期 trace 严格逐点：两条虚拟链独立 → trace 因子化
    c1 = _dense_trace(collect(ψ1.AL))
    c2 = _dense_trace(collect(ψ2.AL))
    @test _dense_trace(collect(K)) ≈ c1 .* c2 atol = 1e-10
end

@testset "mult：MPO 乘法（朴素精确对照 + 压缩）" begin
    T = ComplexF64
    Random.seed!(44)
    W1 = rand_denseimpo(T, [2, 2]; D = 2)
    W2 = rand_denseimpo(T, [2, 2]; D = 3)
    I2 = identityimpo(T, [2, 2])
    # 严格乘法 = 稠密算符矩阵乘法，键维 = 两键维乘积
    P = W1 * W2
    @test P isa DenseIMPO && bonddim(P, 1) == 6
    @test reshape(_dense_mpo_repr(P), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10
    # 迭代乘法：W1·I = W1（D = 2 = 目标键维）。与目标平行即可（整体相位/尺度是
    # 输出射线规范的自由度）
    P1, _, _ = mult(W1, I2, VOMPS(D = 2))
    @test P1 isa CanonicalIMPO && bonddim(P1, 1) == 2
    dP1 = vec(_dense_mpo_repr(DenseIMPO(P1)))
    dW1v = vec(_dense_mpo_repr(W1))
    ls1 = dot(dP1, dW1v) / dot(dP1, dP1)
    @test norm(dW1v .- ls1 .* dP1) / norm(dW1v) < 1e-8
    # 压缩路径：2·Wa·I（键 2）压到 D=1 精确（秩-1 目标），两算法。
    wa = rand_denseimpo(T, [2, 2]; D = 1)
    dwa = _dense_mpo_repr(wa)
    tgt = 2 .* dwa
    s2 = DenseIMPO([cat(wa[ℓ], wa[ℓ]; dims = (1, 3)) for ℓ in 1:length(wa)])   # blockdiag(wa, wa)
    for alg in (VOMPS(D = 1, maxiter = 200), IDMRG(D = 1, maxiter = 200))
        Random.seed!(1)
        Pc, _, _ = mult(s2, I2, alg)
        @test bonddim(Pc, 1) == 1
        d = _dense_mpo_repr(DenseIMPO(Pc))
        ls = dot(vec(d), vec(tgt)) / dot(vec(d), vec(d))
        @test norm(vec(tgt) .- ls .* vec(d)) / norm(vec(tgt)) < 1e-6
    end
end

"严格施加（W * ψ 的规范代表；二参数 mult 已删除，等价表达）。"
apply_exact(W, ψ) = InfiniteMPSAlgorithms._global_normalize!(
    CanonicalIMPS((W * DenseIMPS(collect(ψ.AL))).As))

@testset "mult：结果类型与正则性（含 naive 兜底）" begin
    # 确认点 1：mult 的所有 MPO 输出（精确 / lazy / naive 兜底）一律
    # CanonicalIMPO 且 ismixedcanonical，不得透出 DenseIMPO
    T = ComplexF64
    Random.seed!(46)
    W1 = rand_denseimpo(T, [2, 2]; D = 2)
    W2 = rand_denseimpo(T, [2, 2]; D = 3)
    I2 = identityimpo(T, [2, 2])
    # 精确（无 alg）：严格 `*` + 规范化（mult 的两参数 mpo·mpo 版本已删除）
    Pe = InfiniteMPSAlgorithms._global_normalize!(CanonicalIMPO(W1 * W2))
    @test Pe isa CanonicalIMPO && ismixedcanonical(Pe) && bonddim(Pe, 1) == 6
    # mpo·mps 精确路径
    ψ = randomimps(T, [2, 2]; D = 3)
    y = apply_exact(W1, ψ)
    @test y isa CanonicalIMPS && ismixedcanonical(y)
    # lazy 路径
    Pl, _, _ = mult(W1, I2, VOMPS(D = 2))
    @test Pl isa CanonicalIMPO && ismixedcanonical(Pl)
    # naive 兜底：maxiter = 0 使 lazy 引擎不收敛（overlap 不达 0.9N）而触发兜底
    Pf, _, _ = mult(W1, W2, VOMPS(D = 4, maxiter = 0))
    @test Pf isa CanonicalIMPO && ismixedcanonical(Pf)
    Pf2, _, _ = mult(W1, W2, IDMRG(D = 4, maxiter = 0))
    @test Pf2 isa CanonicalIMPO && ismixedcanonical(Pf2)
end

@testset "mult：实输入 × 复 leading vector（MPSKit 对齐）" begin
    # 确认点 2：MPSKit 的环境张量在复域分配——实输入下 ⟨bra|W|ket⟩ 融合转移
    # 的 leading vector 可为复，环境按 eigsolve 的实际 eltype（复）存放、通道
    # 升为复算术（不取实部、不报错）。本测试构造 bond-1 链（d = 2，A_p = 1/√2）
    # 与 w = 2 的实 MPO，使融合转移 T = 2·[[0,1],[-1,0]]（本征值 ±2i，严格
    # 无实主导本征对）。
    T = Float64
    d = 2
    a = fill(1 / sqrt(2), d)
    ψ = CanonicalIMPS([reshape(a, 1, d, 1)])                # N = 1，键维 1
    M = [0.0 1.0; -1.0 0.0]
    W4 = reshape([M[w2, w1] / 4 for w1 in 1:2, u in 1:d, w2 in 1:2, dd in 1:d], 2, d, 2, d)
    Wo = DenseIMPO([W4])   # Σ_{u,d} W[w,u,w′,d] = M[w′,w] ⇒ 融合转移 = 2M
    # 前提固化：融合转移主本征值严格非实，eigsolve 返回复向量
    λs, _ = eigsolve(v -> vec(push_env_left(reshape(v, 1, 2, 1), ψ.AL[1], Wo[1], ψ.AL[1])),
                     ones(2), 1, :LM; ishermitian = false)
    @test abs(imag(λs[1])) > 0.9 * abs(λs[1])

    # 精确 mult（naive fuse + gaugefix，实通道）：正则且保持实
    ye = apply_exact(Wo, ψ)
    @test ye isa CanonicalIMPS && ismixedcanonical(ye) && scalartype(ye) == Float64
    # VOMPS / IDMRG：复环境通道下不崩溃；结果提升为复（MPSKit 对齐）、
    # 混合正则恒等式严格成立、范数 1
    for alg in (VOMPS(D = 2, maxiter = 50), IDMRG(D = 2, maxiter = 50))
        yv, _, _ = mult(Wo, ψ, alg)
        @test yv isa CanonicalIMPS && ismixedcanonical(yv)
        @test scalartype(yv) <: Complex
        @test norm(yv) ≈ 1 atol = 1e-10
    end
    # mpo·mpo 兜底（identity 通道，实）：同样正则
    Pf, _, _ = mult(Wo, Wo, VOMPS(D = 2, maxiter = 0))
    @test Pf isa CanonicalIMPO && ismixedcanonical(Pf)
end

@testset "hadamard：element-wise 乘积（物理维不变）" begin
    T = ComplexF64
    Random.seed!(45)
    ψ1 = randomimps(T, [2, 2]; D = 3)
    ψ2 = randomimps(T, [2, 2]; D = 3)
    c1 = _dense_mps_repr(ψ1)
    c2 = _dense_mps_repr(ψ2)

    # 严格路径（DenseIMPS）：原始 zip 张量串的周期 trace 严格逐点，物理维不变、
    # 键维 = 3·3（严格运算不支持 CanonicalIMPS——需要规范代表时显式转换）
    H12 = DenseIMPS(collect(ψ1.AL)) ⊙ DenseIMPS(collect(ψ2.AL))
    @test H12 isa DenseIMPS
    @test phydims(H12) == [2, 2]
    @test max_bonddim(H12) == 9
    dH = vec(_dense_mps_repr(H12))
    tgt = vec(c1 .* c2)
    ls = real(dot(dH, tgt)) / real(dot(dH, dH))
    @test norm(tgt .- ls .* dH) / norm(tgt) < 1e-10

    # 压缩路径（两算法）：D = 9 = 精确键维 → 无损，与严格乘积的规范代表平行
    H12c = CanonicalIMPS(collect(H12.As))
    for alg in (VOMPS(D = 9, maxiter = 200), IDMRG(D = 9, maxiter = 200))
        Hc, _, _ = hadamard(ψ1, ψ2, alg)
        @test max_bonddim(Hc) == 9
        @test abs(dot(Hc, H12c)) > 1 - 1e-8
    end

    # 长度不同 ⇒ lcm 单胞（逐周期平铺 zip）；逐 site 物理维不匹配仍抛错
    h23 = DenseIMPS(collect(ψ1.AL)) ⊙
          DenseIMPS(collect(randomimps(T, [2, 2, 2]; D = 2).AL))
    @test h23 isa DenseIMPS && length(h23) == 6        # lcm(2, 3) = 6
    @test_throws DimensionMismatch (DenseIMPS(collect(ψ1.AL)) ⊙
                                    DenseIMPS(collect(randomimps(T, [3, 2]; D = 3).AL)))

    # 语义注记：ψ2 = dag(ψ1) 时 c12 ∝ |c1|² 为逐点模方（非负实波形），
    # 而不是密度矩阵；密度矩阵需 u/d 双腿的 MPO 表示（另一类构造）
    Habs = DenseIMPS(collect(ψ1.AL)) ⊙ DenseIMPS(collect(dag(ψ1).AL))
    dHabs = vec(_dense_mps_repr(Habs))
    tgtabs = vec(abs2.(c1))
    lsabs = real(dot(dHabs, tgtabs)) / real(dot(dHabs, dHabs))
    @test norm(tgtabs .- lsabs .* dHabs) / norm(tgtabs) < 1e-10
end

@testset "hadamard zip 核：非均匀键（矩形张量）" begin
    # _zip_push_left / _mapAC_zip 的输出腿是因子的**右键**——旧实现误用左键维
    # 分配/reshape，均匀键下恰好恒等（测试抓不到），矩形键下直接形状错。
    # 此处用左右键维不同的张量对照逐物理片 staging 的直接收缩参照固化
    # （@tensor 不支持同一未收缩指标跨两个操作数的 batched 形式，须切片）。
    T = ComplexF64
    Random.seed!(46)
    A1 = randn(T, 3, 2, 4)      # A1[a, s, b]：左键 3、右键 4
    A2 = randn(T, 5, 2, 2)      # A2[c, s, e]：左键 5、右键 2
    below = randn(T, 6, 2, 7)   # below[bl, s, bl′]
    L = randn(T, 6, 3, 5)       # L[bl, a, c]（site 左键侧环境）
    GR = randn(T, 4, 2, 9)      # GR[b, e, xR]（site 右键侧环境）
    GL = randn(T, 8, 3, 5)      # GL[xL, a, c]

    Lp = InfiniteMPSAlgorithms._zip_push_left(L, below, A2, A1)
    Lref = zeros(T, 7, 4, 2)
    for s in 1:2
        bs, A1s, A2s = below[:, s, :], A1[:, s, :], A2[:, s, :]
        Lref .+= @tensor tmp[bl′, b, e] :=
            conj(bs[bl, bl′]) * L[bl, a, c] * A2s[c, e] * A1s[a, b]
    end
    @test size(Lp) == (7, 4, 2)
    @test Lp ≈ Lref atol = 1e-12

    k = InfiniteMPSAlgorithms._mapAC_zip(GL, GR, A2, A1)
    kref = zeros(T, 8, 2, 9)
    for p in 1:2
        A1p, A2p = A1[:, p, :], A2[:, p, :]
        kref[:, p, :] .+= @tensor tmp[xL, xR] :=
            GL[xL, a, c] * A2p[c, e] * A1p[a, b] * GR[b, e, xR]
    end
    @test size(k) == (8, 2, 9)
    @test k ≈ kref atol = 1e-12
end

@testset "非均匀键：compress / mult / hadamard 端到端" begin
    # 键 profile 逐站不同（周期闭合）的链/算符走全部三条变分通道——
    # zip 核的矩形键修复（上一 testset）由此在引擎级固化：旧实现下
    # hadamard 通道直接形状错；其余通道核逐站取维（审计无均匀键假设），
    # 端到端覆盖 VOMPS（环境 fixedpoint 重解路径）与 IDMRG（增量推进路径）。
    T = ComplexF64
    Random.seed!(48)

    # 键 profile [4,2,2]（phys [2,3,2]）；bond ℓ = site ℓ 右键
    ψnu = CanonicalIMPS([randn(T, 2, 2, 4), randn(T, 4, 3, 2), randn(T, 2, 2, 2)])
    # MPO 键 profile [4,2,3]
    Ws = [randn(T, 3, 2, 4, 2), randn(T, 4, 3, 2, 3), randn(T, 2, 2, 3, 2)]
    W = CanonicalIMPO(Ws)
    # mult 的 ket：键 [3,2,2]；乘积键 = (wl·bl, wr·br) = [12,4,6] → D = 12 无损
    As2 = [randn(T, 2, 2, 3), randn(T, 3, 3, 2), randn(T, 2, 2, 2)]
    ψ2nu = CanonicalIMPS(As2)
    # mpo·mpo 因子：键 [3,2,3] / [3,4,4]；乘积键 [12,8,12] → D = 12 无损
    W1s = [randn(T, 3, 2, 3, 2), randn(T, 3, 3, 2, 3), randn(T, 2, 2, 3, 2)]
    W1 = CanonicalIMPO(W1s)
    W2s = [randn(T, 4, 2, 3, 2), randn(T, 3, 3, 4, 3), randn(T, 4, 2, 4, 2)]
    W2 = CanonicalIMPO(W2s)
    # hadamard 因子：键 [3,2,2] / [2,3,2]；zip 键 [6,6,4] → D = 6 无损
    ψ1h = CanonicalIMPS([randn(T, 2, 2, 3), randn(T, 3, 3, 2), randn(T, 2, 2, 2)])
    ψ2h = CanonicalIMPS([randn(T, 2, 2, 2), randn(T, 2, 3, 3), randn(T, 3, 2, 2)])

    # ---- compress（MPS）：非均匀键 bra（D=4 ≥ 全部键 ⇒ 初猜即原链）跑通两引擎
    for alg in (VOMPS(D = 4, maxiter = 300), IDMRG(D = 4, maxiter = 300))
        y, _, _ = compress(ψnu, alg)
        @test ismixedcanonical(y) && max_bonddim(y) == 4
        @test abs(dot(y, ψnu)) > 1 - 1e-10
    end
    # 真截断（D=2）：变分不低于逐键谱截断参照
    y2, _, _ = compress(ψnu, VOMPS(D = 2, maxiter = 300))
    ref2, _ = truncate!(copy(ψnu); trunc = truncdim(2))
    @test fidelity(y2, ψnu) ≥ fidelity(ref2, ψnu) - 1e-9
    @test fidelity(y2, ψnu) < 1 - 1e-6

    # ---- compress（MPO）：vectorize 的 MPS 视图通道（逐站融合物理腿）
    Wc, _, _ = compress(W, VOMPS(D = 3, maxiter = 300))
    refW, _ = truncate!(copy(W); trunc = truncdim(3))
    @test Wc isa CanonicalIMPO && ismixedcanonical(Wc) && max_bonddim(Wc) == 3
    @test fidelity(Wc, W) ≥ fidelity(refW, W) - 1e-9

    # ---- mult（mpo·mps）：无损对照 = 朴素 fuse 乘积串
    ref_mult = CanonicalIMPS([InfiniteMPSAlgorithms.fuse(Ws[ℓ], As2[ℓ]) for ℓ in 1:3])
    for alg in (VOMPS(D = 12, maxiter = 300), IDMRG(D = 12, maxiter = 300))
        ym, _, _ = mult(W, ψ2nu, alg)
        @test ismixedcanonical(ym) && max_bonddim(ym) == 12
        @test abs(dot(ym, ref_mult)) > 1 - 1e-8
    end

    # ---- mult（mpo·mpo）：无损对照 = 朴素乘积算符
    ref_mm = CanonicalIMPO([InfiniteMPSAlgorithms._naive_mul_tensor(W1s[ℓ], W2s[ℓ])
                            for ℓ in 1:3])
    for alg in (VOMPS(D = 12, maxiter = 300), IDMRG(D = 12, maxiter = 300))
        P, _, _ = mult(W1, W2, alg)
        @test P isa CanonicalIMPO && ismixedcanonical(P) && max_bonddim(P) == 12
        @test fidelity(P, ref_mm) > 1 - 1e-8
    end

    # ---- hadamard：无损对照 = 朴素 zip 串
    ref_h = CanonicalIMPS([InfiniteMPSAlgorithms._naive_hadamard_tensor(ψ1h.AL[ℓ],
                                                                        ψ2h.AL[ℓ])
                           for ℓ in 1:3])
    for alg in (VOMPS(D = 6, maxiter = 300), IDMRG(D = 6, maxiter = 300))
        Hc, _, _ = hadamard(ψ1h, ψ2h, alg)
        @test ismixedcanonical(Hc) && max_bonddim(Hc) == 6
        @test abs(dot(Hc, ref_h)) > 1 - 1e-8
    end
end

# ---------------- mult!/compress!/hadamard! 完全对齐 ----------------
#
# 收敛信息经由内部函数取得：`_mult`/`_compress`/`_hadamard` 返回
# `(y, envs, info)`（info::IterativeConvergenceInfo，niter = 扫掠轮数）；
# 导出层（mult/compress/hadamard）与 in-place 版本均返回
# `(result, envs, info)`。

@testset "mult! ≡ compress!（完全对齐：同初态+同参数 → 同迭代数、终态一致）" begin
    # 语义对齐：mult!(out, W, ψ, alg)（compute-on-the-fly 通道）与
    # compress!(out, W*DenseIMPS / W*W2, alg)（naive 乘积压缩通道）优化的是
    # 同一目标射线——同初态+同算法参数下迭代数与解（ray 意义）逐一相等。
    # （实现注记：compress 的 mpo 目标视图取 CanonicalIMPO 的 **AL 家族**——
    #   其周期 trace 才是算符串 wavefunction；AC 家族携带 C 加权，会定义另
    #   一个变分问题。）
    T = ComplexF64
    Random.seed!(50)
    W = rand_denseimpo(T, [2, 2]; D = 3)
    ψ = randomimps(T, [2, 2]; D = 4)
    W2 = rand_denseimpo(T, [2, 2]; D = 2)
    Wraw = W * W2
    D0 = 3
    ψraw = W * DenseIMPS(collect(ψ.AL))
    for alg in (VOMPS(D = D0, tol = 1e-12, maxiter = 300),
                IDMRG(D = D0, tol = 1e-12, maxiter = 300))
        # mpo·mps（同初态：mult 通道 out1 ≡ compress 通道 envs.bra = out2）
        out1 = randomimps(T, [2, 2]; D = D0)
        out2 = copy(out1)
        y1, _, i1 = mult!(out1, W, ψ, alg)
        ket = CanonicalIMPS(collect(ψraw.As))
        envs2 = OverlapCache(out2, ket, alg.alg_environments)
        _, i2 = InfiniteMPSAlgorithms.compression_sweeps!(envs2, alg)
        y2 = envs2.bra                                                     # 引擎已归一化
        @test i1.niter == i2.niter
        v1 = vec(_dense_mps_repr(y1))
        v2 = vec(_dense_mps_repr(y2))
        ls = dot(v2, v1) / dot(v2, v2)
        @test norm(v1 .- ls .* v2) / norm(v1) < 1e-10

        # mpo·mpo
        m1 = randomimpo(T, [2, 2]; D = D0)
        m2 = copy(m1)
        p0 = m1
        z1, _, j1 = mult!(p0, W, W2, alg)
        ketw = CanonicalIMPS(InfiniteMPSAlgorithms.vectorize(Wraw).As)
        envs3 = OverlapCache(vectorize(m2), ketw, alg.alg_environments)
        _, j2 = InfiniteMPSAlgorithms.compression_sweeps!(envs3, alg)
        z2 = InfiniteMPSAlgorithms.devectorize(envs3.bra)
        @test j1.niter == j2.niter
        w1 = vectorize(z1)
        w2 = vectorize(z2)
        # 同迭代数 + fidelity → 1（两路径环境重解的 round-off ~1e-10）
        @test fidelity(w1, w2) > 1 - 1e-8
    end
end

@testset "copyphyims：hadamard ≡ mult（完全对齐）" begin
    # copyphyims(ψ1) 把 ψ1 变成物理对角的算符链（对齐 FiniteMPSAlgorithms 的
    # copyphydims）；把它作用到 ψ2 上（mult 通道）与 ψ1⊙ψ2（hadamard 通道）
    # 是同一目标——严格波形相等，变分路径同初态+同参数下同迭代数、终态一致
    # （ray 意义）。
    T = ComplexF64
    Random.seed!(51)
    ψ1 = randomimps(T, [2, 2]; D = 3)
    ψ2 = randomimps(T, [2, 2]; D = 4)
    W1 = copyphyims(DenseIMPS(collect(ψ1.AL)))

    # 严格层面：copyphyims(ψ1) * ψ2 与 ⊙(DenseIMPS) 波形一致
    hd = DenseIMPS(collect(ψ1.AL)) ⊙ DenseIMPS(collect(ψ2.AL))
    mp = W1 * DenseIMPS(collect(ψ2.AL))
    @test _dense_trace(collect(hd)) ≈ _dense_trace(collect(mp)) atol = 1e-10

    # 变分层面（mult! ≡ hadamard!，同初态+同参数 → 同迭代数、终态一致）：
    # mult!(out, copyphyims(ψ1), ψ2, alg) ≡ hadamard!(out, ψ1, ψ2, alg)
    D0 = 6
    for alg in (VOMPS(D = D0, tol = 1e-12, maxiter = 300),
                IDMRG(D = D0, tol = 1e-12, maxiter = 300))
        out1 = randomimps(T, [2, 2]; D = D0)
        out2 = copy(out1)
        y1, _, i1 = mult!(out1, W1, ψ2, alg)
        y2, _, i2 = hadamard!(out2, ψ1, ψ2, alg)
        @test i1.niter == i2.niter
        v1 = vec(_dense_mps_repr(y1))
        v2 = vec(_dense_mps_repr(y2))
        ls = dot(v2, v1) / dot(v2, v2)
        @test norm(v1 .- ls .* v2) / norm(v1) < 1e-8
    end
end

@testset "dense ≡ canonical 输入（compress/mult/hadamard 直接吃 AbstractInfinite*）" begin
    # dense 类型经 getproperty 家族视图（AL/AR/AC 周期视图 + C 单位矩阵视图）
    # 直接进入环境与局部映射（不做任何规范转换）：同一射线的 dense/canonical
    # 输入定义同一变分问题，输出类型恒为 CanonicalIMPS/CanonicalIMPO，终态在
    # 收敛精度内一致（不同初猜的轨迹差只余 round-off）。
    T = ComplexF64
    Random.seed!(55)
    ψraw = [randn(T, 4, 2, 4), randn(T, 4, 2, 4)]
    ψc = CanonicalIMPS(ψraw)
    ψd = DenseIMPS(collect(ψc.AR))             # 同射线的 dense 表示
    ψ2raw = [randn(T, 3, 2, 3), randn(T, 3, 2, 3)]
    ψ2c = CanonicalIMPS(ψ2raw)
    ψ2d = DenseIMPS(collect(ψ2c.AR))
    Wd = rand_denseimpo(T, [2, 2]; D = 3)
    Wc = CanonicalIMPO(collect(Wd.Ws))
    W2d = rand_denseimpo(T, [2, 2]; D = 2)
    W2c = CanonicalIMPO(collect(W2d.Ws))

    "两结果的 ray 残差（dense 周期 trace 表示，规范与尺度不变）。"
    function _rayres(x, y)
        vx = vec(_dense_mps_repr(x))
        vy = vec(_dense_mps_repr(y))
        ls = dot(vy, vx) / dot(vy, vy)
        return norm(vx .- ls .* vy) / norm(vx)
    end

    for alg in (VOMPS(D = 3, tol = 1e-12, maxiter = 300),
                IDMRG(D = 3, tol = 1e-12, maxiter = 300))
        # compress（MPS 目标）
        yc, _, ic = compress(ψc, alg)
        yd, _, id = compress(ψd, alg)
        @test yc isa CanonicalIMPS && yd isa CanonicalIMPS
        @test ic.converged && id.converged
        @test _rayres(yc, yd) < 1e-8
        # compress（MPO 目标：dense 为纯融合视图目标，canonical 携带规范）
        zc, _, _ = compress(Wc, alg)
        zd, _, _ = compress(Wd, alg)
        @test zc isa CanonicalIMPO && zd isa CanonicalIMPO
        @test _rayres(vectorize(zc), vectorize(zd)) < 1e-6
        # mult（mpo·mps）：输出恒 CanonicalIMPS
        m1c, _, _ = mult(Wc, ψc, alg)
        m1d, _, _ = mult(Wd, ψd, alg)
        @test m1c isa CanonicalIMPS && m1d isa CanonicalIMPS
        @test _rayres(m1c, m1d) < 1e-6
        # mult（mpo·mpo）：输出恒 CanonicalIMPO
        m2c, _, _ = mult(Wc, W2c, alg)
        m2d, _, _ = mult(Wd, W2d, alg)
        @test m2c isa CanonicalIMPO && m2d isa CanonicalIMPO
        @test _rayres(vectorize(m2c), vectorize(m2d)) < 1e-6
        # hadamard：输出恒 CanonicalIMPS
        hc, _, _ = hadamard(ψc, ψ2c, alg)
        hd, _, _ = hadamard(ψd, ψ2d, alg)
        @test hc isa CanonicalIMPS && hd isa CanonicalIMPS
        @test _rayres(hc, hd) < 1e-6
    end
end

@testset "inplace 类型提升兜底：实 out 无法表示复通道解 → 首个返回值为更新 bra" begin
    # 通道算术类型 = 环境 eltype（缓存构造器把 bra 槽提升到该类型）：复输入使
    # 通道为复，实 out 无法原地表示通道解——mult!/compress!/hadamard! 的第一个
    # 返回值应为提升后的更新 bra（envs.bra）本身，out 不写回（仅 changebond!
    # 预处理生效）。纯实输入下 leading vector 复化的场景走同一代码路径。
    T = ComplexF64
    Random.seed!(57)
    W = rand_denseimpo(Float64, [2, 2]; D = 2)
    W2 = rand_denseimpo(Float64, [2, 2]; D = 2)
    ψc = randomimps(T, [2, 2]; D = 2)
    ψr = randomimps(Float64, [2, 2]; D = 2)

    # mult!（mpo·mps）：严格目标键维 4，D = 4 精确（VOMPS/IDMRG 双引擎）；
    # 严格代数不接受混合 eltype，参考侧把 W 提升为复
    Wc = DenseIMPO([ComplexF64.(w) for w in W.Ws])
    yexact = apply_exact(Wc, ψc)
    for alg in (VOMPS(D = 4, tol = 1e-12, maxiter = 200),
                IDMRG(D = 4, tol = 1e-12, maxiter = 200))
        out = randomimps(Float64, [2, 2]; D = 4)
        y1, envs1, _ = mult!(out, W, ψc, alg)
        @test y1 === envs1.bra && scalartype(y1) == T
        @test eltype(out.AL[1]) == Float64
        @test fidelity(y1, yexact) > 1 - 1e-8
    end

    # mult!（mpo·mpo）
    m0 = randomimpo(Float64, [2, 2]; D = 4)
    z1, envsz, _ = mult!(m0, W, W2, VOMPS(D = 4, tol = 1e-12, maxiter = 200))
    @test z1 === envsz.bra && z1 isa CanonicalIMPO && scalartype(z1) == T
    @test eltype(m0.AL[1]) == Float64
    dz = vec(_dense_mpo_repr(DenseIMPO(z1)))
    W2c = DenseIMPO([ComplexF64.(w) for w in W2.Ws])
    dref = vec(_dense_mpo_repr(Wc * W2c))
    ls = dot(dref, dz) / dot(dref, dref)
    @test norm(dz .- ls .* dref) / norm(dref) < 1e-8

    # compress!（复 ket 目标）
    out2 = randomimps(Float64, [2, 2]; D = 2)
    y2, envs2, _ = compress!(out2, ψc, VOMPS(D = 2, tol = 1e-12, maxiter = 200))
    @test y2 === envs2.bra && scalartype(y2) == T
    @test eltype(out2.AL[1]) == Float64
    @test fidelity(y2, ψc) > 1 - 1e-8

    # hadamard!（复因子）：精确参考 = ψr ⊙ ψc（严格 zip，键维 4）
    out3 = randomimps(Float64, [2, 2]; D = 4)
    y3, envs3, _ = hadamard!(out3, ψr, ψc, VOMPS(D = 4, tol = 1e-12, maxiter = 200))
    @test y3 === envs3.bra && scalartype(y3) == T
    @test eltype(out3.AL[1]) == Float64
    hexact = DenseIMPS([ComplexF64.(a) for a in collect(ψr.AL)]) ⊙
             DenseIMPS(collect(ψc.AL))
    @test fidelity(DenseIMPS(y3), hexact) > 1 - 1e-8
end
