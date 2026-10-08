@testset "CanonicalIMPS 数据结构与规范" begin
    T = ComplexF64
    ψ = randomimps(T, [2, 2]; D = 6)

    @test length(ψ) == 2
    @test phydims(ψ) == [2, 2]
    @test scalartype(ψ) == T

    # 左正交性：Σ AL† AL = 1
    for ℓ in 1:2
        @tensor g[a, b] := conj(ψ.AL[ℓ][x, s, a]) * ψ.AL[ℓ][x, s, b]
        @test norm(g - I) < 1e-10
    end
    # 右正交性：Σ AR AR† = 1
    for ℓ in 1:2
        @tensor g[a, b] := ψ.AR[ℓ][a, s, x] * conj(ψ.AR[ℓ][b, s, x])
        @test norm(g - I) < 1e-10
    end
    # MPSKit 约定：AC[i] = AL[i]·C[i] = C[i-1]·AR[i]
    for ℓ in 1:2
        @tensor AC1[a, s, c] := ψ.AL[ℓ][a, s, b] * ψ.C[ℓ][b, c]
        @tensor AC2[a, s, c] := ψ.C[ℓ - 1][a, b] * ψ.AR[ℓ][b, s, c]
        @test norm(ψ.AC[ℓ] - AC1) < 1e-10
        @test norm(ψ.AC[ℓ] - AC2) < 1e-10
    end
    # 周期下标：ψ.AL[0] == ψ.AL[N]
    @test ψ.AL[0] == ψ.AL[2]
    @test ψ.C[3] == ψ.C[1]

    # 范数与归一化（MPSKit: norm(ψ) = norm(ψ.AC[1])）
    @test abs(norm(ψ) - 1) < 1e-10
    normalize!(ψ)
    @test abs(norm(ψ.C[1]) - 1) < 1e-10

    # 乘积态：熵为 0
    ρ = prodimps(T, [2, 2], [1, 2])
    @test abs(norm(ρ) - 1) < 1e-12
    @test entropy(ρ) < 1e-12
    @test expectationvalue(ρ) ≈ 1 atol = 1e-12

    # 乘积态的 ⟨Z⟩（site 1 处于 |0⟩ = |↑⟩）
    @test real(expectationvalue(ρ, (1,) => σz(T))) ≈ 1 atol = 1e-12
end

@testset "fidelity / infidelity" begin
    T = ComplexF64
    Random.seed!(7)
    ψ1 = randomimps(T, [2, 2]; D = 6)
    ψ2 = randomimps(T, [2, 2]; D = 6)

    f11 = fidelity(ψ1, ψ1)
    @test f11 ≈ 1 atol = 1e-12
    @test infidelity(ψ1, ψ1) ≈ 0 atol = 1e-12
    f12 = fidelity(ψ1, ψ2)
    @test 0 ≤ f12 ≤ 1
    @test infidelity(ψ1, ψ2) ≈ 1 - f12 atol = 1e-12

    # 相位不变：位点 1 的三族张量同乘 e^{iθ}（保持正交性与恒等式网络的规范旋转）
    ψ1p = copy(ψ1)
    ph = exp(0.9im)
    ψ1p.AL[1] .*= ph
    ψ1p.AR[1] .*= ph
    ψ1p.AC[1] .*= ph
    @test fidelity(ψ1, ψ1p) ≈ 1 atol = 1e-11
    @test infidelity(ψ1, ψ1p) ≈ 0 atol = 1e-11
end

@testset "ismixedcanonical 负检查" begin
    T = ComplexF64
    Random.seed!(9)
    ψ = randomimps(T, [2, 2]; D = 4)
    @test ismixedcanonical(ψ)
    ϵs = mixedcanonical_errors(ψ)
    @test all(ϵs .< 1e-12)

    # 故意破坏规范：缩放单 site AL 破坏左正交性
    ψbad = copy(ψ)
    ψbad.AL[1] .*= 2.0
    @test !ismixedcanonical(ψbad)
    # 破坏混合一致性：打乱 C
    ψbad2 = copy(ψ)
    ψbad2.C[1] .*= 3.0
    @test !ismixedcanonical(ψbad2)

    # verbosity 路径不报错
    @test ismixedcanonical(ψ; verbosity = 1) === true

    # 三参数形式（AL, C, AR；kwargs 同样生效）
    @test ismixedcanonical(collect(ψ.AL), collect(ψ.C), collect(ψ.AR))
    @test !ismixedcanonical(collect(ψbad.AL), collect(ψbad.C), collect(ψbad.AR))
    @test ismixedcanonical(collect(ψ.AL), collect(ψ.C), collect(ψ.AR);
                           verbosity = 1) === true

    # rank-4（MPOTensor）家族直接输入：kernel 在融合物理腿视图上检查，
    # 与 vectorize 路径逐位一致
    W = randomimpo(T, [2, 2]; D = 3)
    Wg = gaugefix!(copy(W), collect(W.AR))
    ϵ_v = mixedcanonical_errors(vectorize(Wg))
    ϵ_r = mixedcanonical_errors(Wg.AL, Wg.C, Wg.AR)
    @test collect(ϵ_r) ≈ collect(ϵ_v) atol = 1e-13
    @test ismixedcanonical(Wg.AL, Wg.C, Wg.AR)
end

@testset "非均匀键 profile（unit cell > 1）" begin
    T = ComplexF64
    Random.seed!(7)

    # ---- MPS：键 profile [4, 2, 2]，phys [2, 3, 2] ----
    As = [randn(T, 2, 2, 4), randn(T, 4, 3, 2), randn(T, 2, 2, 2)]
    ψ = CanonicalIMPS(As)
    @test [bonddim(ψ, ℓ) for ℓ in 1:3] == [4, 2, 2]
    @test max_bonddim(ψ) == 4
    @test ismixedcanonical(ψ)
    @test all(mixedcanonical_errors(ψ) .< 1e-12)
    @test abs(norm(ψ) - 1) < 1e-8
    @test abs(dot(ψ, ψ) - 1) < 1e-8

    # 环境无关的局部观测量（site 1 的物理维为 2）
    @test isfinite(real(expectationvalue(ψ, (1,) => σz(T))))

    # 周期闭合那一条键也要校验（旧版只看相邻 1:N-1）
    @test_throws DimensionMismatch CanonicalIMPS([randn(T, 2, 2, 3), randn(T, 4, 2, 2)])
    @test_throws DimensionMismatch CanonicalIMPS([randn(T, 4, 2, 4), randn(T, 4, 2, 3)])

    # ALs + C₀ 构造路径：C₀ 作用在键 N 上，必须是 (χ_N, χ_N) = (2, 2)
    @test_throws DimensionMismatch CanonicalIMPS([copy(a) for a in As], Matrix{T}(I, 3, 3))
    ψa = CanonicalIMPS([copy(a) for a in As], Matrix{T}(I, 2, 2))
    @test [bonddim(ψa, ℓ) for ℓ in 1:3] == [4, 2, 2]

    # ---- 等距 AL 串 + C₀ → 混合规范 ----
    ψiso = _nonuniform_mps([4, 2, 2], 2)
    @test [bonddim(ψiso, ℓ) for ℓ in 1:3] == [4, 2, 2]
    @test ismixedcanonical(ψiso)

    # ---- 不可行键 profile 的清理（对齐 MPSKit `InfiniteMPS(A)` 的 makefullrank!）----
    # bond1 的 χ = 5 > Dl·d = 2·2 = 4 ⇒ 冗余，应被删到 4（周期 trace 即态不变）
    Ar = [randn(T, 2, 2, 5), randn(T, 5, 2, 2), randn(T, 2, 2, 2)]
    t0 = _dense_trace([copy(a) for a in Ar])
    ψr = CanonicalIMPS(Ar)
    @test [bonddim(ψr, ℓ) for ℓ in 1:3] == [4, 2, 2]
    @test ismixedcanonical(ψr)
    t1 = _dense_trace(collect(ψr.AL))
    @test abs(dot(vec(t1), vec(t0))) / (norm(vec(t1)) * norm(vec(t0))) > 1 - 1e-10
    # 反向：左键不可行（Dl > Dr·d）走另一分支清理
    Al = [randn(T, 5, 2, 2), randn(T, 2, 2, 2), randn(T, 2, 2, 5)]
    t2 = _dense_trace([copy(a) for a in Al])
    ψl = CanonicalIMPS(Al)
    @test [bonddim(ψl, ℓ) for ℓ in 1:3] == [2, 2, 4]
    t3 = _dense_trace(collect(ψl.AL))
    @test abs(dot(vec(t3), vec(t2))) / (norm(vec(t3)) * norm(vec(t2))) > 1 - 1e-10

    # ---- MPO：键 profile [4, 2, 2] ----
    Ws = [randn(T, 2, 2, 4, 2), randn(T, 4, 2, 2, 2), randn(T, 2, 2, 2, 2)]
    W = CanonicalIMPO(Ws)
    @test [bonddim(W, ℓ) for ℓ in 1:3] == [4, 2, 2]
    @test ismixedcanonical(W)
    @test all(mixedcanonical_errors(W) .< 1e-12)

    # ---- changebond! 的既有语义：强制拉成均匀 profile ----
    q = copy(ψ)
    changebond!(q; D = 4)
    @test all(bonddim(q, ℓ) == 4 for ℓ in 1:3)
    @test ismixedcanonical(q)
    q2 = copy(ψ)
    changebond!(q2; D = 2)
    @test all(bonddim(q2, ℓ) == 2 for ℓ in 1:3)
    @test ismixedcanonical(q2)

    # ---- 零填充扩键：同一物理态，所有键 profile 按逐键赋值 ----
    ψu = randomimps(T, [2, 2, 2]; D = 4)
    ψp = _padbond!(copy(ψu), 2, 1)
    @test [bonddim(ψp, ℓ) for ℓ in 1:3] == [4, 5, 4]
    @test ismixedcanonical(ψp)
    @test all(mixedcanonical_errors(ψp) .< 1e-12)
    @test abs(dot(ψp, ψu) / (norm(ψp) * norm(ψu)) - 1) < 1e-12
end

@testset "phydim / randomimps / randomimpo 接口" begin
    T = ComplexF64
    Random.seed!(17)

    # phydim：unit cell 内逐站物理维
    ψ = randomimps(T, [2, 3, 4]; D = 4)
    @test phydim(ψ, 1) == 2 && phydim(ψ, 2) == 3 && phydim(ψ, 3) == 4
    @test phydim(ψ, 4) == 2                       # 周期下标
    @test phydims(ψ) == [2, 3, 4]
    W = randomimpo(T, [2, 3]; D = 2)              # CanonicalIMPO：构造即混合规范
    @test W isa CanonicalIMPO && ismixedcanonical(W)
    @test phydim(W, 1) == 2 && phydim(W, 2) == 3
    @test phydims(W) == [2, 3]
    Wd = rand_denseimpo(T, [2, 3]; D = 2)
    M = CanonicalIMPO(collect(Wd.Ws))             # DenseIMPO → CanonicalIMPO（方算符）
    @test phydim(M, 1) == 2 && phydim(M, 2) == 3
    @test phydims(M) == [2, 3]
    # 非方算符（du ≠ dd）：融合物理维非完全平方数，转换 kernel 即拒绝
    @test_throws ArgumentError CanonicalIMPO([randn(T, 1, 2, 1, 3), randn(T, 1, 3, 1, 2)])

    # 无 T 时默认 Float64；d 默认 2；D 必须显式给出
    ψf = randomimps(3; D = 4)
    @test ψf isa CanonicalIMPS{Float64} && size(ψf.AL[1]) == (4, 2, 4)
    ψf2 = randomimps(3; d = 3, D = 4)
    @test ψf2 isa CanonicalIMPS{Float64} && size(ψf2.AL[1], 2) == 3
    Wf = randomimpo(2; D = 3)
    @test Wf isa CanonicalIMPO{Float64} && size(Wf.AL[1]) == (3, 2, 3, 2)
    Wf2 = randomimpo(2; d = 3, D = 3)
    @test Wf2 isa CanonicalIMPO{Float64} && size(Wf2.AL[1], 2) == 3
    Wkw = randomimpo(T, [2, 2]; D = 3, tol = 1e-12)   # kwargs 透传 CanonicalIMPO（gaugefix）
    @test Wkw isa CanonicalIMPO && ismixedcanonical(Wkw)

    # rng 可复现
    g = MersenneTwister(42)
    ψa = randomimps(Float64, 2; D = 3, rng = g)
    g = MersenneTwister(42)
    ψb = randomimps(Float64, 2; D = 3, rng = g)
    @test ψa.AL[1] == ψb.AL[1]

    # 缺 D 报错（D 无默认值，必须用户输入）
    @test_throws UndefKeywordError randomimps(2)
    @test_throws UndefKeywordError randomimpo(2)
end

@testset "DenseIMPS：scalar / hadamard / dot / norm / fidelity / distance" begin
    T = ComplexF64
    Random.seed!(21)
    ψ1 = randomimps(T, [2, 3]; D = 4)
    ψ2 = randomimps(T, [2, 3]; D = 4)
    d1 = DenseIMPS(collect(ψ1.AL))
    d2 = DenseIMPS(collect(ψ2.AL))
    n1 = norm(ψ1)

    # scalar 乘法：只缩放第一个张量；态 = α·ψ（非射线）
    d1α = 3.5 * d1
    @test d1α isa DenseIMPS
    @test norm(d1α) ≈ 3.5 * n1 atol = 1e-10
    @test norm(d1 / 2.5) ≈ n1 / 2.5 atol = 1e-10
    @test norm(-d1) ≈ n1 atol = 1e-10
    # 原始张量串的波形 = α × 原波形（_dense_trace 对 raw 串逐点缩放检查：
    # α 缩放在第一个张量上 → 波形整体乘 α）
    w0 = _dense_trace(collect(d1.As))
    wα = _dense_trace(collect(d1α.As))
    @test wα ≈ 3.5 .* w0 atol = 1e-10

    # dot / norm：转移主导本征值
    @test real(dot(d1, d1)) ≈ n1^2 atol = 1e-10 * max(n1^2, 1)
    @test abs(dot(d1, d2)) ≤ n1 * norm(d2) + 1e-10

    # fidelity / infidelity：标度与相位不变
    @test fidelity(d1, d1) ≈ 1 atol = 1e-12
    @test infidelity(d1, d1) ≈ 0 atol = 1e-12
    @test fidelity(d1, 2.0 * d1) ≈ 1 atol = 1e-12
    f12 = fidelity(d1, d2)
    @test 0 ≤ f12 ≤ 1
    @test infidelity(d1, d2) ≈ 1 - f12 atol = 1e-12
    # 与 CanonicalIMPS 侧的 fidelity 交叉验证（同一态的两种表示）
    @test abs(fidelity(d1, d2) - fidelity(ψ1, ψ2)) < 1e-10

    # distance / distance2：‖ψ1 − ψ2‖² = sA + sB − 2Re⟨A,B⟩
    @test distance(d1, d1) ≈ 0 atol = 1e-10
    d2sc = distance2(d1, d2)
    @test d2sc ≈ real(dot(d1, d1)) + real(dot(d2, d2)) - 2 * real(dot(d1, d2)) atol =
          1e-9 * max(d2sc, 1)
    @test distance(d1, d2) ≈ sqrt(d2sc) atol = 1e-10 * max(distance(d1, d2), 1)

    # ⊙（严格 hadamard）：物理维不变、键维 = 4·4，波形逐点乘积（原始串严格成立）
    h = d1 ⊙ d2
    @test h isa DenseIMPS && phydims(h) == [2, 3] && max_bonddim(h) == 16
    w1 = _dense_trace(collect(d1.As))
    w2 = _dense_trace(collect(d2.As))
    @test _dense_trace(collect(h.As)) ≈ w1 .* w2 atol = 1e-10 * max(abs(w1[1] * w2[1]), 1)
    # 不支持 Canonical 输入
    @test_throws MethodError ψ1 ⊙ ψ2
end

@testset "truncate!：逐键 C 截断（toiadt!/toipt! 语义）" begin
    T = ComplexF64
    Random.seed!(23)

    # ---- CanonicalIMPS：默认方案（D 封顶 + 相对阈值）在良态链上近无操作 ----
    ψ = randomimps(T, [2, 2]; D = 4)
    ψ0 = copy(ψ)
    y, err = truncate!(ψ)
    @test y === ψ && err < 1e-12
    @test ismixedcanonical(ψ)
    @test bonddim(ψ, 1) == 4 && bonddim(ψ, 2) == 4   # 键 profile 不变
    @test fidelity(ψ, ψ0) > 1 - 1e-10                # 射线不变
    @test norm(ψ) ≈ norm(ψ0) atol = 1e-9             # 权重保留

    # ---- CanonicalIMPS：相对阈值的秩亏清理（toiadt! 的 XTRG 用例）----
    # 零块 (ψ2, 0) 链：键 4 但第二通道范数为零（不进入环形 winding），键谱
    # {s1, s2, 0, 0}——相对阈值截掉零方向 → 键 profile 降到 [2, 2]，射线不变
    A2 = [randn(T, 2, 2, 2), randn(T, 2, 2, 2)]
    ψblk = CanonicalIMPS([cat(A2[ℓ], zeros(T, 2, 2, 2); dims = (1, 3)) for ℓ in 1:2])
    ψblk0 = copy(ψblk)
    @test max_bonddim(ψblk) == 4                     # 构造不做降键（键可行即保留）
    _, errb = truncate!(ψblk; trunc = truncrelerr(ϵ = 1e-10))
    @test bonddim(ψblk, 1) == 2 && bonddim(ψblk, 2) == 2
    @test ismixedcanonical(ψblk) && errb < 1e-12
    @test fidelity(ψblk, ψblk0) > 1 - 1e-10

    # ---- CanonicalIMPS：truncdim(2) 强截断 ----
    # 真截断（丢有限权重）：截断 + 完全重正则化（Û 投影 AL + gaugefix!(; order=:LR)
    # 重解 C）——三项正则误差均在 alg_gauge.tol（~1e-13）级，与丢弃权重无关：
    # 截断不自洽只进射线精度（fidelity），不进正则性
    Random.seed!(25)
    ψ = randomimps(T, [2, 2]; D = 4)
    ψ0 = copy(ψ)
    _, err2 = truncate!(ψ; trunc = truncdim(2))
    @test bonddim(ψ, 1) == 2 && bonddim(ψ, 2) == 2
    @test ismixedcanonical(ψ; tol = 1e-10)
    @test maximum(mixedcanonical_errors(ψ)) < 1e-10
    @test 0 < fidelity(ψ, ψ0) < 1                    # 真截断：射线改变
    @test err2 > 0

    # ---- CanonicalIMPO：相对阈值的秩亏清理 ----
    W2raw = [randn(T, 2, 2, 2, 2), randn(T, 2, 2, 2, 2)]
    Wblk = CanonicalIMPO([cat(W2raw[ℓ], zeros(T, 2, 2, 2, 2); dims = (1, 3)) for ℓ in 1:2])
    Wblk0 = copy(Wblk)
    @test max_bonddim(Wblk) == 4
    _, errw = truncate!(Wblk; trunc = truncrelerr(ϵ = 1e-10))
    @test bonddim(Wblk, 1) == 2 && bonddim(Wblk, 2) == 2
    @test ismixedcanonical(Wblk) && errw < 1e-12
    @test fidelity(Wblk, Wblk0) > 1 - 1e-10          # Hilbert–Schmidt 保真度

    # ---- CanonicalIMPO：truncdim(2) 强截断（正则性约定同 MPS 版）----
    Random.seed!(26)
    W = randomimpo(T, [2, 2]; D = 4)
    W0 = copy(W)
    _, errw2 = truncate!(W; trunc = truncdim(2))
    @test max_bonddim(W) == 2
    @test ismixedcanonical(W; tol = 1e-10)
    @test maximum(mixedcanonical_errors(W)) < 1e-10
    @test 0 < fidelity(W, W0) < 1
    @test errw2 > 0
end

@testset "gaugefix!(InfiniteOrthogonalize)：精确混合正则化（mixedcanonicalize2!）" begin
    Random.seed!(31)

    # ---- 良态链 + 随机规范变换：射线不变、严格正则（~1e-14）、normalize 语义 ----
    # 规范变换 A[i] → g_{i-1}⁻¹·A[i]·g[i] 在周期 trace 下严格保持射线
    for T in (Float64, ComplexF64), N in (1, 2, 3)
        As = [randn(T, 4, 2, 4) for _ in 1:N]
        gs = [randn(T, 4, 4) .+ 4 .* Matrix{T}(I, 4, 4) for _ in 1:N]
        ginvs = [inv(g) for g in gs]
        Asg = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            @tensor Ag[a, s, b] := ginvs[mod1(ℓ - 1, N)][a, c] * As[ℓ][c, s, d] * gs[ℓ][d, b]
            Asg[ℓ] = Ag
        end
        ψ0 = CanonicalIMPS([copy(a) for a in As])
        # 输出恒归一：射线不变、标量类型保持（实链不升复）
        ψ = CanonicalIMPS([copy(a) for a in As])
        gaugefix!(ψ, Asg, InfiniteOrthogonalize())
        @test ismixedcanonical(ψ; tol = 1e-10)
        @test scalartype(ψ) == T
        @test fidelity(ψ, ψ0) > 1 - 1e-10
        @test norm(ψ) ≈ 1 atol = 1e-10
    end

    # ---- 相对阈值的秩亏清理：零块 (ψ2 ⊕ 0) 链（随机规范扰动后）----
    T = ComplexF64
    A2 = [randn(T, 2, 2, 2) for _ in 1:2]
    Asblk = [cat(A2[ℓ], zeros(T, 2, 2, 2); dims = (1, 3)) for ℓ in 1:2]
    gs = [randn(T, 4, 4) .+ 4 .* Matrix{T}(I, 4, 4) for _ in 1:2]
    ginvs = [inv(g) for g in gs]
    Asblkg = Vector{Array{T,3}}(undef, 2)
    for ℓ in 1:2
        @tensor Ag[a, s, b] := ginvs[mod1(ℓ - 1, 2)][a, c] * Asblk[ℓ][c, s, d] * gs[ℓ][d, b]
        Asblkg[ℓ] = Ag
    end
    ψref = CanonicalIMPS([copy(a) for a in A2])
    ψ = CanonicalIMPS([copy(a) for a in Asblk])
    gaugefix!(ψ, Asblkg, InfiniteOrthogonalize(trunc = truncrelerr(ϵ = 1e-10)))
    @test bonddim(ψ, 1) == 2 && bonddim(ψ, 2) == 2
    @test ismixedcanonical(ψ; tol = 1e-10)
    @test fidelity(ψ, ψref) > 1 - 1e-10

    # ---- 强截断（truncdim）下的严格正则性 ----
    # mixedcanonicalize2! 的 site-1 收尾 Diag(S)⁻¹·x[1] 只在无截断时保持右正交，
    # 强截断下破缺 O(err)——:LR 装配的 R 趟（LQ 重建 AR + C 重解）必须把它修复：
    # 三个正则误差与截断强度无关，恒 ~alg_gauge.tol（截断不自洽只进射线精度）
    ψs0 = CanonicalIMPS([randn(ComplexF64, 4, 2, 4) for _ in 1:2])
    Asg = collect(ψs0.AL)
    ψs = copy(ψs0)
    gaugefix!(ψs, Asg, InfiniteOrthogonalize(trunc = truncdim(2)))
    @test bonddim(ψs, 1) == 2 && bonddim(ψs, 2) == 2
    @test maximum(mixedcanonical_errors(ψs)) < 1e-10
    @test 0 < fidelity(ψs, ψs0) < 1                   # 真截断：射线改变

    # ---- CanonicalIMPO：rank-4 串 + 随机规范（键腿 g·W·g⁻¹），MPS 视图正则化 ----
    for T in (Float64, ComplexF64)
        Ws = [randn(T, 4, 2, 4, 2) for _ in 1:2]
        gs = [randn(T, 4, 4) .+ 4 .* Matrix{T}(I, 4, 4) for _ in 1:2]
        ginvs = [inv(g) for g in gs]
        Wsg = Vector{Array{T,4}}(undef, 2)
        for ℓ in 1:2
            @tensor Wg[a, u, b, d] := gs[mod1(ℓ - 1, 2)][a, c] * Ws[ℓ][c, u, e, d] *
                                      ginvs[ℓ][e, b]
            Wsg[ℓ] = Wg
        end
        W0 = CanonicalIMPO([copy(W) for W in Ws])
        W = CanonicalIMPO([copy(W) for W in Ws])
        gaugefix!(W, Wsg, InfiniteOrthogonalize())
        @test ismixedcanonical(W; tol = 1e-10)
        @test scalartype(W) == T
        @test fidelity(W, W0) > 1 - 1e-10        # Hilbert–Schmidt 保真度
    end
end

@testset "混合规范构造器 CanonicalIMPS(AL, C, AR[, AC]) / CanonicalIMPO(AL, C, AR[, AC])" begin
    T = ComplexF64
    Random.seed!(27)

    # 从已正则链提取三族重建：四族逐位一致（AC = AL·C 闭式装配）、正则性保持
    ψ = randomimps(T, [2, 2]; D = 3)
    ψ2 = CanonicalIMPS(ψ.AL, ψ.C, ψ.AR)
    @test collect(ψ2.AL) == collect(ψ.AL)
    @test collect(ψ2.C) == collect(ψ.C)
    @test collect(ψ2.AR) == collect(ψ.AR)
    @test collect(ψ2.AC) == collect(ψ.AC)
    @test ismixedcanonical(ψ2)
    # Vector 输入（MPSKit 的 InfiniteMPS(AL, C, AR) 亦接受）
    ψ3 = CanonicalIMPS(collect(ψ.AL), collect(ψ.C), collect(ψ.AR))
    @test collect(ψ3.AC) == collect(ψ.AC)
    # 显式四族
    ψ4 = CanonicalIMPS(ψ.AL, ψ.C, ψ.AR, ψ.AC)
    @test collect(ψ4.AL) == collect(ψ.AL)

    W = randomimpo(T, [2, 2]; D = 3)
    W2 = CanonicalIMPO(W.AL, W.C, W.AR)
    @test collect(W2.AL) == collect(W.AL)
    @test collect(W2.C) == collect(W.C)
    @test collect(W2.AR) == collect(W.AR)
    @test collect(W2.AC) == collect(W.AC)
    @test ismixedcanonical(W2)
    W3 = CanonicalIMPO(collect(W.AL), collect(W.C), collect(W.AR))
    @test collect(W3.AC) == collect(W.AC)
    W4 = CanonicalIMPO(W.AL, W.C, W.AR, W.AC)
    @test collect(W4.AC) == collect(W.AC)
end

