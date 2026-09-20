# ================= debug：VUMPS / IDMRG 与 MPSKit 完全对齐 =================
#
# 运行：julia --project=debug debug/vumps_idmrg_alignment.jl
#
# 同一初始态、同一 Jordan 哈密顿量（TFIM / Heisenberg），对比基态能量收敛值。
# 注意：Heisenberg 基态破缺平移对称（动量 π），需用 2-site 单胞，
#       单-site 单胞在双方包中都会停滞（非算法问题）。

using Test
using Random
using LinearAlgebra
using TensorKit
using MPSKit
using InfiniteMPSAlgorithms
import InfiniteMPSAlgorithms: scalartype
include(joinpath(@__DIR__, "mpsconvert.jl"))

T = ComplexF64
d = 2

function compare(algname::String, H_our, H_mk, ψ0; e_exact = nothing,
                 tol = 1e-8, e_tol = 1e-4, L = length(ψ0))
    ψ_mk = mkinfinitemps(ψ0)

    ψ1, envs1, ϵ1 = InfiniteMPSAlgorithms.find_groundstate(ψ0, H_our,
                InfiniteMPSAlgorithms.VUMPS(maxiter = 300, tol = 1e-11, verbosity = 0))
    ψ2, envs2, ϵ2 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                MPSKit.VUMPS(maxiter = 300, tol = 1e-11, verbosity = 0))
    e1 = real(InfiniteMPSAlgorithms.expectation_value(ψ1, H_our, envs1) / L)
    e2 = real(MPSKit.expectation_value(ψ2, H_mk) / L)
    @info "VUMPS $algname" our = e1 mpskit = e2 Δ = abs(e1 - e2) eps_our = ϵ1 eps_mk = ϵ2
    @test abs(e1 - e2) < tol
    isnothing(e_exact) || @test abs(e1 - e_exact) < e_tol

    ψ3, envs3, ϵ3 = InfiniteMPSAlgorithms.find_groundstate(ψ0, H_our,
                InfiniteMPSAlgorithms.IDMRG(maxiter = 300, tol = 1e-10, verbosity = 0))
    ψ4, envs4, ϵ4 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                MPSKit.IDMRG(maxiter = 300, tol = 1e-10, verbosity = 0))
    e3 = real(InfiniteMPSAlgorithms.expectation_value(ψ3, H_our, envs3) / L)
    e4 = real(MPSKit.expectation_value(ψ4, H_mk) / L)
    @info "IDMRG $algname" our = e3 mpskit = e4 Δ = abs(e3 - e4) eps_our = ϵ3 eps_mk = ϵ4
    @test abs(e3 - e4) < tol
    isnothing(e_exact) || @test abs(e3 - e_exact) < e_tol
    return nothing
end

# ---- TFIM（1-site 单胞，有解析解 e₀ = -4/π） ----
J, h = 1.0, 1.0
lattice1 = fill(ℂ^d, 1)
Random.seed!(1234)
ψ_tfim = MixedCanonicalMPS([randn(T, 8, d, 8)])
compare("TFIM",
        tfim_hamiltonian(J = J, h = h, T = T),
        MPSKit.InfiniteMPOHamiltonian(lattice1,
                                      (1 => -h * σz(T), (1, 2) => -J * (σx(T) ⊗ σx(T)))),
        ψ_tfim; e_exact = -4 / π, L = 1)

# ---- Heisenberg XXX（2-site 单胞，e₀ = 1/4 - ln2） ----
lattice2 = fill(ℂ^d, 2)
hh = (1 / 4) * (σx(T) ⊗ σx(T) + σy(T) ⊗ σy(T) + σz(T) ⊗ σz(T))
Random.seed!(1234)
ψ_heis = MixedCanonicalMPS([randn(T, 12, d, 12), randn(T, 12, d, 12)])
compare("Heisenberg",
        heisenberg_hamiltonian(T = T),
        MPSKit.InfiniteMPOHamiltonian(lattice2, ((1, 2) => hh, (2, 3) => hh)),
        ψ_heis; e_exact = 0.25 - log(2), L = 2)

println("\nVUMPS / IDMRG 与 MPSKit 对齐测试通过。")
