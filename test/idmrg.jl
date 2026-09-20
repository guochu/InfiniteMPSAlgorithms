@testset "IDMRG 基态（single-site）" begin
    T = ComplexF64
    H = heisenberg_hamiltonian(T = T)
    e_exact = 0.25 - log(2)

    ψv, envsv, _ = find_groundstate(random_mps(T, [2, 2], 12), H,
                                    VUMPS(maxiter = 300, tol = 1e-10))
    ev = real(expectation_value(ψv, H, envsv) / 2)

    ψi, envsi, ϵ = find_groundstate(random_mps(T, [2, 2], 12), H,
                                    IDMRG(maxiter = 300, tol = 1e-9))
    ei = real(expectation_value(ψi, H, envsi) / 2)

    # 阈值覆盖随机初态的亚稳态收敛（与 MPSKit 同 iter/同样的逐位变分极限）
    @test abs(ei - e_exact) < 2e-4
    @test abs(ei - ev) < 2e-4

    # 固定键维 D=8 仍收敛（single-site 不增长键维）
    ψ8, envs8, _ = find_groundstate(random_mps(T, [2, 2], 8), H,
                                    IDMRG(maxiter = 300, tol = 1e-9))
    e8 = real(expectation_value(ψ8, H, envs8) / 2)
    @test maxbond(ψ8) <= 8
    @test abs(e8 - e_exact) < 1e-3

    # 与稠密 InfiniteMPO 的 VUMPS 能量（Jordan 重建态上）一致
    Hd = InfiniteMPO(H)
    @test length(Hd) == 1
end
