# =====================================================================
# 迭代代数运算测试：MPS/MPO 加法、MPO 乘法、MPS hadamard（密度矩阵）
#
# - 朴素精确构造（D = nothing）对标 MPSKit 语义（键维直和 / fuse_mul_mpo）；
# - 变分压缩（D::Int）验证 VOMPS（DMRG 型 ALS）与 VUMPS（本征求解模板）
#   两条算法路径收敛到同一结果；
# - 数值断言基于周期 trace 表示（_dense_mps_repr / _dense_mpo_repr，定义于
#   runtests.jl；规范变换下望远相消，是纯规范不变的波形/算符表示）。
# =====================================================================

@testset "MPS 加法（朴素精确，对标 MPSKit）" begin
    T = ComplexF64
    Random.seed!(42)
    ψu = product_mps(T, [2, 2], [1, 1])   # |00⟩
    ψd = product_mps(T, [2, 2], [2, 2])   # |11⟩
    s = ψu + ψd
    @test s isa MixedCanonicalMPS && maxbond(s) == 2
    # 波形 = |00⟩ + |11⟩（trace 表示的振幅矩阵 = 单位阵）
    @test _dense_mps_repr(s) ≈ Matrix{T}(I, 2, 2) atol = 1e-12
    # 长度 / 逐 site 物理维不匹配
    @test_throws DimensionMismatch ψu + random_mps(T, [2, 2, 2], 2)
    @test_throws DimensionMismatch ψu + random_mps(T, [3, 2], 2)
end

@testset "MPS 加法压缩（VOMPS 与 VUMPS）" begin
    T = ComplexF64
    Random.seed!(42)
    ψ1 = random_mps(T, [2, 2], 3)
    d1 = _dense_mps_repr(ψ1)
    for alg in (VOMPS(maxiter = 200), VUMPS(maxiter = 200))
        Random.seed!(1)
        s2 = +(ψ1, ψ1; D = 3, alg = alg)      # ψ1 + ψ1 = 2ψ1（键 6 → 3 无损）
        @test maxbond(s2) == 3
        # 同射线：|dot| = 1（Cauchy–Schwarz 饱和）。相位是规范自由度：
        # VUMPS 的 C 链本征解相位不钉定，cross-dot 可带 twist 相位（VOMPS 继承
        # ket 相位故恰好为 1）；环迹输出公式中 twist 自动抵消，不影响幅值。
        @test abs(dot(s2, ψ1)) ≈ 1 atol = 1e-6
        # 波形与 ψ1 平行（twist 相位鲁棒）：因子 2 被射线规范化吸收
        ds2 = _dense_mps_repr(s2)
        @test abs(dot(vec(ds2), vec(d1))) / (norm(ds2) * norm(d1)) ≈ 1 atol = 1e-6
    end
end

@testset "MPO 加法 / 减法" begin
    T = ComplexF64
    Random.seed!(43)
    W1 = random_mpo(T, [2, 2], 2)
    W2 = random_mpo(T, [2, 2], 3)
    dW1 = _dense_mpo_repr(W1)
    dW2 = _dense_mpo_repr(W2)
    # 朴素：键维直和、稠密算符可加
    s = W1 + W2
    @test s isa InfiniteMPO && mpobond(s, 1) == 5 && mpobond(s, 2) == 5
    @test _dense_mpo_repr(s) ≈ dW1 + dW2 atol = 1e-10
    # 减法：符号折入首张量（MPSKit 标量乘约定；N=2 时逐张量缩放的旧 bug 会使此处非零）
    z = W1 - W1
    @test _dense_mpo_repr(z) ≈ zeros(T, 2, 2, 2, 2) atol = 1e-10
    # 压缩：I + I = 2I（键 2 → 1 精确，两算法）。与目标平行即可——整体相位是
    # 规范自由度不作要求（幅值要求保留在复比例残差中）
    I2 = identity_mpo(T, [2, 2])
    dI = _dense_mpo_repr(I2)
    tgt = 2 .* dI
    for alg in (VOMPS(maxiter = 200), VUMPS(maxiter = 200))
        Random.seed!(1)
        s2 = +(I2, I2; D = 1, alg = alg)
        @test mpobond(s2, 1) == 1
        d = _dense_mpo_repr(s2)
        ls = dot(vec(d), vec(tgt)) / dot(vec(d), vec(d))
        @test norm(vec(tgt) .- ls .* vec(d)) / norm(vec(tgt)) < 1e-6
    end
    # MixedCanonicalMPO 包装器的代数委托：InfiniteMPO(M) 收集 AL，与构造输入
    # 平行即可——整体相位/比例是规范自由度（λ 被 normalize!(C) 吸收，射线表示）
    M1 = MixedCanonicalMPO(W1)
    M2 = MixedCanonicalMPO(W2)
    dM1 = _dense_mpo_repr(InfiniteMPO(M1))
    dM2 = _dense_mpo_repr(InfiniteMPO(M2))
    for (dM, dW) in ((dM1, dW1), (dM2, dW2))
        ls = dot(vec(dM), vec(dW)) / dot(vec(dM), vec(dM))
        @test norm(vec(dW) .- ls .* vec(dM)) / norm(vec(dW)) < 1e-8
    end
    # 委托代数在转换后幅值层面精确可加 / 可消（M1±M2 即 InfiniteMPO 层朴素和）
    @test _dense_mpo_repr(M1 + M2) ≈ dM1 + dM2 atol = 1e-8
    @test _dense_mpo_repr(M1 - M1) ≈ zeros(T, 2, 2, 2, 2) atol = 1e-8
end

@testset "MPO 乘法（算符复合，对标 fuse_mul_mpo）" begin
    T = ComplexF64
    Random.seed!(44)
    W1 = random_mpo(T, [2, 2], 2)
    W2 = random_mpo(T, [2, 2], 3)
    I2 = identity_mpo(T, [2, 2])
    # 朴素乘法 = 稠密算符矩阵乘法，键维 = 两键维乘积
    P = W1 * W2
    @test mpobond(P, 1) == 6
    @test reshape(_dense_mpo_repr(P), 4, 4) ≈
          reshape(_dense_mpo_repr(W1), 4, 4) * reshape(_dense_mpo_repr(W2), 4, 4) atol = 1e-10
    # 恒等算符：W1·I = W1；D = naive 键维 → 短路返回精确结果
    P1 = *(W1, I2; D = 2)
    @test mpobond(P1, 1) == 2
    @test _dense_mpo_repr(P1) ≈ _dense_mpo_repr(W1) atol = 1e-10
    # 压缩路径：2·Wa·I（键 2）压到 D=1 精确（秩-1 目标），两算法。
    # 与目标平行即可（整体相位是规范自由度）
    wa = random_mpo(T, [2, 2], 1)
    dwa = _dense_mpo_repr(wa)
    tgt = 2 .* dwa
    s2 = wa + wa
    for alg in (VOMPS(maxiter = 200), VUMPS(maxiter = 200))
        Random.seed!(1)
        Pc = *(s2, I2; D = 1, alg = alg)
        @test mpobond(Pc, 1) == 1
        d = _dense_mpo_repr(Pc)
        ls = dot(vec(d), vec(tgt)) / dot(vec(d), vec(d))
        @test norm(vec(tgt) .- ls .* vec(d)) / norm(vec(tgt)) < 1e-6
    end
end

@testset "hadamard 与密度矩阵" begin
    T = ComplexF64
    Random.seed!(45)
    ψu = product_mps(T, [2, 2], [1, 1])
    ψd = product_mps(T, [2, 2], [2, 2])
    cu = vec(_dense_mps_repr(ψu))
    cd = vec(_dense_mps_repr(ψd))
    # hadamard 的算符表示 = 波幅外积：O[(u…),(d…)] = c1[u…]·c2[d…]
    Hm = hadamard(ψu, ψd)
    @test Hm isa InfiniteMPO && mpobond(Hm, 1) == 1
    @test reshape(_dense_mpo_repr(Hm), 4, 4) ≈ cu * cd' atol = 1e-12
    # 纯态密度矩阵 ρ = hadamard(ψ, dag(ψ))：Hermitian、迹 1、纯度 1
    ρ = hadamard(ψu, dag(ψu))
    Oρ = reshape(_dense_mpo_repr(ρ), 4, 4)
    @test Oρ ≈ Oρ' atol = 1e-12
    @test tr(Oρ) ≈ 1 atol = 1e-12
    @test real(tr(Oρ * Oρ)) ≈ 1 atol = 1e-12
    # 混合态 ρu + ρd：迹 2、正交支撑（ρ² = ρ）、Hermitian
    ρsum = +(hadamard(ψu, dag(ψu)), hadamard(ψd, dag(ψd)))
    Oρs = reshape(_dense_mpo_repr(ρsum), 4, 4)
    @test tr(Oρs) ≈ 2 atol = 1e-12
    @test real(tr(Oρs * Oρs)) ≈ 2 atol = 1e-12
    @test Oρs ≈ Oρs' atol = 1e-12
    # 压缩（无损，秩-1 目标）：hadamard(ψa, ψa; D = 1) 的算符表示 = 波幅外积
    # ca ⊗ ca。与目标平行即可（整体相位是规范自由度）
    ψa = random_mps(T, [2], 1)
    ca = vec(_dense_mps_repr(ψa))
    tgt = vec(ca * permutedims(ca))
    for alg in (VOMPS(maxiter = 200), VUMPS(maxiter = 200))
        Random.seed!(1)
        Hc = hadamard(ψa, ψa; D = 1, alg = alg)
        @test mpobond(Hc, 1) == 1
        d = vec(_dense_mpo_repr(Hc))
        ls = dot(d, tgt) / dot(d, d)
        @test norm(tgt .- ls .* d) / norm(tgt) < 1e-6
    end
end

@testset "InfiniteMPO 标量乘（MPSKit 首张量约定）" begin
    T = ComplexF64
    Random.seed!(46)
    W = random_mpo(T, [2, 2], 2)
    dW = _dense_mpo_repr(W)
    # 算符值 α·W（逐张量缩放的旧实现会给出 α^N·W）
    @test _dense_mpo_repr(2.5 * W) ≈ 2.5 .* dW atol = 1e-10
    # 偶数单胞长度 N=2 时负号不失效
    @test _dense_mpo_repr(-W) ≈ -dW atol = 1e-10
    @test _dense_mpo_repr(W / 2) ≈ 0.5 .* dW atol = 1e-10
    # 仅首张量被缩放
    @test (2.5 * W)[2] == W[2]
    @test (2.5 * W)[1] ≈ 2.5 * W[1]
end
