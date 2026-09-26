@testset "CanonicalIMPS 数据结构与规范" begin
    T = ComplexF64
    ψ = randomimps(T, [2, 2], 6)

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
    ψ1 = randomimps(T, [2, 2], 6)
    ψ2 = randomimps(T, [2, 2], 6)

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
    ψ = randomimps(T, [2, 2], 4)
    @test ismixedcanonical(ψ)
    ϵs = mixedcanonical_error(ψ)
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
    @test all(mixedcanonical_error(ψ) .< 1e-12)
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
    @test all(mixedcanonical_error(W) .< 1e-12)

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
    ψu = randomimps(T, [2, 2, 2], 4)
    ψp = _padbond!(copy(ψu), 2, 1)
    @test [bonddim(ψp, ℓ) for ℓ in 1:3] == [4, 5, 4]
    @test ismixedcanonical(ψp)
    @test all(mixedcanonical_error(ψp) .< 1e-12)
    @test abs(dot(ψp, ψu) / (norm(ψp) * norm(ψu)) - 1) < 1e-12
end

