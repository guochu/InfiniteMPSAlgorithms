# =====================================================================
# VUMPS / IDMRG 基态能量与 MPSKit 对比（固化自 debug/vumps_idmrg_alignment.jl）
#
# 同一初始态、同一 Jordan 哈密顿量（TFIM / Heisenberg），对比基态能量收敛值。
# 注：Heisenberg 基态破缺平移对称（动量 π），需用 2-site 单胞——单-site 单胞
#     的 Heisenberg 在双方包中都会停滞（非算法问题，与 MPSKit 行为一致）；
#     TFIM（顺磁基态，动量 0）1-site 单胞两包均正常收敛。
# =====================================================================

function compare_groundstate(H_our, H_mk, ψ0, D; e_exact = nothing,
                             tol = 1e-8, e_tol = 1e-4)
    L = length(ψ0)
    # ---- VUMPS ----
    ψ1, envs1, ϵ1 = find_groundstate(ψ0, H_our,
                VUMPS(D = D, maxiter = 300, tol = 1e-11, verbosity = 0))
    ψ2, envs2, ϵ2 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                MPSKit.VUMPS(maxiter = 300, tol = 1e-11, verbosity = 0))
    e1 = real(expectationvalue(ψ1, H_our, envs1) / L)
    e2 = real(MPSKit.expectation_value(ψ2, H_mk) / L)
    @test abs(e1 - e2) < tol
    isnothing(e_exact) || @test abs(e1 - e_exact) < e_tol
    # ---- IDMRG ----
    ψ3, envs3, ϵ3 = find_groundstate(ψ0, H_our,
                IDMRG(D = D, maxiter = 300, tol = 1e-10, verbosity = 0))
    ψ4, envs4, ϵ4 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                MPSKit.IDMRG(maxiter = 300, tol = 1e-10, verbosity = 0))
    e3 = real(expectationvalue(ψ3, H_our, envs3) / L)
    e4 = real(MPSKit.expectation_value(ψ4, H_mk) / L)
    @test abs(e3 - e4) < tol
    isnothing(e_exact) || @test abs(e3 - e_exact) < e_tol
    return nothing
end

@testset "VUMPS / IDMRG 基态能量 ≡ MPSKit（同初态同参数）" begin
    T = ComplexF64
    d = 2

    # ---- TFIM（1-site 单胞，有解析解 e₀ = -4/π） ----
    J, h = 1.0, 1.0
    lattice1 = fill(ℂ^d, 1)
    Random.seed!(1234)
    ψ_tfim = CanonicalIMPS([randn(T, 8, d, 8)])
    @testset "TFIM" begin
        compare_groundstate(tfim_hamiltonian(J = J, h = h, T = T),
                            MPSKit.InfiniteMPOHamiltonian(lattice1,
                                1 => -h * σz_tk(T),
                                (1, 2) => -J * (σx_tk(T) ⊗ σx_tk(T))),
                            ψ_tfim, 8; e_exact = -4 / π)
    end

    # ---- Heisenberg XXX（2-site 单胞，e₀ = 1/4 - ln2） ----
    lattice2 = fill(ℂ^d, 2)
    hh = (1 / 4) * (σx_tk(T) ⊗ σx_tk(T) + σy_tk(T) ⊗ σy_tk(T) + σz_tk(T) ⊗ σz_tk(T))
    Random.seed!(1234)
    ψ_heis = CanonicalIMPS([randn(T, 12, d, 12), randn(T, 12, d, 12)])
    @testset "Heisenberg" begin
        compare_groundstate(heisenberg_hamiltonian(T = T),
                            MPSKit.InfiniteMPOHamiltonian(lattice2, (1, 2) => hh, (2, 3) => hh),
                            ψ_heis, 12; e_exact = 0.25 - log(2))
    end
end
