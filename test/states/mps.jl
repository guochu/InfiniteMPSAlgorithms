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
