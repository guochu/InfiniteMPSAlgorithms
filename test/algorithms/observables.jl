@testset "观测量" begin
    T = ComplexF64
    Z = σz(T)

    # 乘积态 |↑↑⟩
    ρ = prodimps(T, [2, 2], [1, 1])
    @test real(expectationvalue(ρ, (1,) => Z)) ≈ 1 atol = 1e-12
    @test real(expectationvalue(ρ, (1, 2) => kron(Z, Z))) ≈ 1 atol = 1e-12
    C = correlator(ρ, Z, Z, 1, 2:5)
    # correlator 为 raw 约定（MPSKit 对标）：⟨Z₁Zⱼ⟩ = 1（|↑↑⟩ 直积态非零关联）
    @test all(abs.(C .- 1) .< 1e-12)
    Cxy = correlator(ρ, σx(T), σx(T), 1, 2:4)
    @test all(abs.(Cxy) .< 1e-12)

    # 反铁磁乘积态（Néel，单胞 2 site）
    ρn = prodimps(T, [2, 2], [1, 2])   # |↑↓⟩
    @test real(expectationvalue(ρn, (1,) => Z)) ≈ 1 atol = 1e-12
    @test real(expectationvalue(ρn, (2,) => Z)) ≈ -1 atol = 1e-12
    @test real(expectationvalue(ρn, (1, 2) => kron(Z, Z))) ≈ -1 atol = 1e-12

    # Heisenberg 基态（Schur 哈密顿量）：SzSz 关联头值 = 1/4
    # （MPSKit 无 ⟨Sz⟩≈0 测试：随机初态下 VUMPS 可能停在 SU(2) 破缺亚稳态，
    #   该断言不稳定，故不设；固定种子保证其余断言可复现）
    Hm = heisenberg_hamiltonian(T = T)
    Random.seed!(1)
    ψ, envs, _ = find_groundstate(randomimps(T, [2, 2], 12), Hm,
                                  VUMPS(maxiter = 200, tol = 1e-9))
    @test abs(expectationvalue(ψ, (1,) => Sz(T)^2) - 0.25) < 1e-12
    Czz = correlator(ψ, Sz(T), Sz(T), 1, 2:6)
    @test real(Czz[1]) < 0   # 近邻反铁磁关联为负
    @test all(real.(Czz) .< 0.26)

    # 熵：Heisenberg 临界态为正，谱归一
    S = entropy(ψ)
    @test 0.5 < S < 2.0
    spec = entanglement_spectrum(ψ)
    @test abs(sum(spec) - 1) < 1e-10
    @test spec[1] >= spec[end]

    # von Neumann ≥ Rényi-2
    S2 = entropy(ψ; α = 2)
    @test S >= S2 - 1e-12
end
