# =====================================================================
# 算术运算测试：严格运算（DenseIMPO * / hadamard(DenseIMPS)）+ 迭代压缩 +
# mult!/compress!/hadamard! 的完全对齐
#
# - DenseIMPO * DenseIMPO / DenseIMPO * DenseIMPS / hadamard(DenseIMPS)：
#   严格运算（不压缩、不正则化），周期 trace 表示下验证算符乘积 / 波形作用 /
#   逐点乘积结构；
# - copyphyims：DenseIMPS→DenseIMPO 的物理腿复制（对齐 FiniteMPSAlgorithms
#   的 copyphydims），hadamard ≡ mult(copyphyims(ψ1), ψ2)；
# - mult! ≡ compress!（同初态+同参数：同迭代数、终态一致）；
# - 变分压缩（D::Int）验证 VOMPS 与 IDMRG 两条算法路径收敛到同一结果；
# - 数值断言基于周期 trace 表示（_dense_mps_repr / _dense_mpo_repr，定义于
#   testhelpers.jl；规范变换下望远相消，是纯规范不变的波形/算符表示）。
# =====================================================================

@testset "严格运算：DenseIMPO * / hadamard(DenseIMPS)（debug 基准）" begin
    T = ComplexF64
    Random.seed!(47)
    W1 = randomimpo(T, [2, 2]; D = 2)
    W2 = randomimpo(T, [2, 2]; D = 3)
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

@testset "hadamard(DenseIMPS)：波形逐点乘积（debug 基准）" begin
    T = ComplexF64
    Random.seed!(49)
    ψ1 = randomimps(T, [2, 2]; D = 3)
    ψ2 = randomimps(T, [2, 2]; D = 4)
    K = hadamard(DenseIMPS(collect(ψ1.AL)), DenseIMPS(collect(ψ2.AL)))
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
    W1 = randomimpo(T, [2, 2]; D = 2)
    W2 = randomimpo(T, [2, 2]; D = 3)
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
    wa = randomimpo(T, [2, 2]; D = 1)
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

@testset "mult：结果类型与正则性（含 naive 兜底）" begin
    # 确认点 1：mult 的所有 MPO 输出（精确 / lazy / naive 兜底）一律
    # CanonicalIMPO 且 ismixedcanonical，不得透出 DenseIMPO
    T = ComplexF64
    Random.seed!(46)
    W1 = randomimpo(T, [2, 2]; D = 2)
    W2 = randomimpo(T, [2, 2]; D = 3)
    I2 = identityimpo(T, [2, 2])
    # 精确（无 alg）：严格 `*` + 规范化（mult 的两参数 mpo·mpo 版本已删除）
    Pe = InfiniteMPSAlgorithms._global_normalize!(CanonicalIMPO(W1 * W2))
    @test Pe isa CanonicalIMPO && ismixedcanonical(Pe) && bonddim(Pe, 1) == 6
    # mpo·mps 精确路径
    ψ = randomimps(T, [2, 2]; D = 3)
    y = mult(W1, ψ)
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
    ye = mult(Wo, ψ)
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
    H12 = hadamard(DenseIMPS(collect(ψ1.AL)), DenseIMPS(collect(ψ2.AL)))
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

    # 长度 / 逐 site 物理维不匹配
    @test_throws DimensionMismatch hadamard(DenseIMPS(collect(ψ1.AL)),
                                            DenseIMPS(collect(randomimps(T, [2, 2, 2]; D = 2).AL)))
    @test_throws DimensionMismatch hadamard(DenseIMPS(collect(ψ1.AL)),
                                            DenseIMPS(collect(randomimps(T, [3, 2]; D = 3).AL)))

    # 语义注记：ψ2 = dag(ψ1) 时 c12 ∝ |c1|² 为逐点模方（非负实波形），
    # 而不是密度矩阵；密度矩阵需 u/d 双腿的 MPO 表示（另一类构造）
    Habs = hadamard(DenseIMPS(collect(ψ1.AL)), DenseIMPS(collect(dag(ψ1).AL)))
    dHabs = vec(_dense_mps_repr(Habs))
    tgtabs = vec(abs2.(c1))
    lsabs = real(dot(dHabs, tgtabs)) / real(dot(dHabs, dHabs))
    @test norm(tgtabs .- lsabs .* dHabs) / norm(tgtabs) < 1e-10
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
    W = randomimpo(T, [2, 2]; D = 3)
    ψ = randomimps(T, [2, 2]; D = 4)
    W2 = randomimpo(T, [2, 2]; D = 2)
    Wraw = W * W2
    D0 = 3
    ψraw = W * DenseIMPS(collect(ψ.AL))
    for alg in (VOMPS(D = D0, tol = 1e-12, maxiter = 300),
                IDMRG(D = D0, tol = 1e-12, maxiter = 300))
        # mpo·mps（同初态：mult 通道 out1 ≡ compress 通道 envs.bra = out2）
        out1 = randomimps(T, [2, 2]; D = D0)
        out2 = copy(out1)
        y1, _, i1 = InfiniteMPSAlgorithms._mult(W, ψ, alg, out1; D = D0)
        ket = CanonicalIMPS(collect(ψraw.As))
        envs2 = OverlapCache(out2, ket, alg.alg_environments)
        _, i2 = alg isa VOMPS ?
                InfiniteMPSAlgorithms._overlap_vomps_sweeps!(envs2, alg) :
                InfiniteMPSAlgorithms._overlap_idmrg_sweeps!(envs2, alg)
        y2 = InfiniteMPSAlgorithms._global_normalize!(envs2.bra)
        @test i1.niter == i2.niter
        v1 = vec(_dense_mps_repr(y1))
        v2 = vec(_dense_mps_repr(y2))
        ls = dot(v2, v1) / dot(v2, v2)
        @test norm(v1 .- ls .* v2) / norm(v1) < 1e-10

        # mpo·mpo
        m1 = CanonicalIMPO(collect(randomimpo(T, [2, 2]; D = D0).Ws))
        m2 = copy(m1)
        p0 = m1
        z1, _, j1 = InfiniteMPSAlgorithms._mult(W, W2, alg, p0; D = D0)
        ketw = CanonicalIMPS(InfiniteMPSAlgorithms.vectorize(Wraw).As)
        envs3 = OverlapCache(vectorize(m2), ketw, alg.alg_environments)
        _, j2 = alg isa VOMPS ?
                InfiniteMPSAlgorithms._overlap_vomps_sweeps!(envs3, alg) :
                InfiniteMPSAlgorithms._overlap_idmrg_sweeps!(envs3, alg)
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

    # 严格层面：copyphyims(ψ1) * ψ2 与 hadamard(DenseIMPS) 波形一致
    hd = hadamard(DenseIMPS(collect(ψ1.AL)), DenseIMPS(collect(ψ2.AL)))
    mp = W1 * DenseIMPS(collect(ψ2.AL))
    @test _dense_trace(collect(hd)) ≈ _dense_trace(collect(mp)) atol = 1e-10

    # 变分层面（内部函数取得迭代数）：
    # _mult(out, copyphyims(ψ1), ψ2, alg) ≡ _hadamard(out, ψ1, ψ2, alg)
    D0 = 6
    for alg in (VOMPS(D = D0, tol = 1e-12, maxiter = 300),
                IDMRG(D = D0, tol = 1e-12, maxiter = 300))
        out1 = randomimps(T, [2, 2]; D = D0)
        out2 = copy(out1)
        y1, _, i1 = InfiniteMPSAlgorithms._mult(W1, ψ2, alg, out1; D = D0)
        y2, _, i2 = InfiniteMPSAlgorithms._hadamard(ψ1, ψ2, alg, out2; D = D0)
        @test i1.niter == i2.niter
        v1 = vec(_dense_mps_repr(y1))
        v2 = vec(_dense_mps_repr(y2))
        ls = dot(v2, v1) / dot(v2, v2)
        @test norm(v1 .- ls .* v2) / norm(v1) < 1e-8
    end
end
