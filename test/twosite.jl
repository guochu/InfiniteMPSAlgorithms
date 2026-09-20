# =====================================================================
# unit cell size = 2 的哈密顿量全链路测试
#
# H = Σᵢ [J₁ S_{2i-1}·S_{2i} + J₂ S_{2i}·S_{2i+1}]（2-site 单胞 Jordan 形式）：
# - J₁ = J₂ = 1：均匀 Heisenberg（2-site 单胞表示），e₀ = 1/4 − ln2 精确锚点；
# - J₁ = 1, J₂ = 1/2：交变耦合（dimerized），VUMPS 与 IDMRG 交叉验证。
# 覆盖：结构 / vumps / idmrg / mult / w1w2 / tdvp / observables。
# =====================================================================

@testset "unit cell size = 2" begin
    T = ComplexF64
    e_exact = 0.25 - log(2)          # 均匀 Heisenberg 能量密度（精确）

    # ---- 2-site 单胞 Jordan 哈密顿量 ----
    Wd(J) = mpohamiltonian(zeros(T, 2, 2),
                           [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J, Sz(T), Sz(T))])
    H = InfiniteMPOHamiltonian([Wd(1.0), Wd(1.0)])          # 均匀
    Hd = InfiniteMPOHamiltonian([Wd(1.0), Wd(0.5)])         # dimerized（J₁=1, J₂=1/2）

    @testset "结构" begin
        @test length(H) == 2
        @test bonddim(H) == 5
        @test isidentitylevel(H, 1) && isidentitylevel(H, 5)
        @test !isemptylevel(H, 2)
    end

    # ---- vumps ----
    Random.seed!(71)
    ψg, envsg, ϵg = find_groundstate(randomimps(T, [2, 2], 12), H,
                                     VUMPS(maxiter = 300, tol = 1e-10, verbosity = 0))
    eg = real(expectationvalue(ψg, H, envsg) / 2)
    @test ϵg < 1e-9
    @test abs(eg - e_exact) < 2e-4

    # TFIM 2-site 单胞：e₀ = −4/π（临界点解析解）
    Wt = mpohamiltonian(-σz(T), [(-1.0, σx(T), σx(T))])
    Ht = InfiniteMPOHamiltonian([Wt, Wt])
    ψt2, _, _ = find_groundstate(randomimps(T, [2, 2], 12), Ht,
                                 VUMPS(maxiter = 300, tol = 1e-10, verbosity = 0))
    @test abs(real(expectationvalue(ψt2, Ht) / 2) + 4 / π) < 1e-6

    # dimerized：能量介于两个极限之间（J₂=0 孤立二聚体 −0.375 / J₂=1 均匀 −0.4431；
    # S·S 非半正定，两极限间无普遍变分序），VUMPS 与 IDMRG 交叉验证见下
    ψd, envsd, _ = find_groundstate(randomimps(T, [2, 2], 12), Hd,
                                    VUMPS(maxiter = 300, tol = 1e-10, verbosity = 0))
    ed = real(expectationvalue(ψd, Hd, envsd) / 2)
    @test -0.4431 - 1e-9 < ed < -0.375 + 1e-9

    # ---- idmrg ----
    ψi, envsi, ϵi = find_groundstate(randomimps(T, [2, 2], 12), H,
                                     IDMRG(maxiter = 300, tol = 1e-9))
    ei = real(expectationvalue(ψi, H, envsi) / 2)
    @test abs(ei - e_exact) < 2e-4
    @test abs(ei - eg) < 2e-4
    @test max_bonddim(ψi) <= 12

    # dimerized：VUMPS 与 IDMRG 交叉验证（有能隙，收敛更快）
    ψdi, envsdi, _ = find_groundstate(randomimps(T, [2, 2], 12), Hd,
                                      IDMRG(maxiter = 300, tol = 1e-9))
    edi = real(expectationvalue(ψdi, Hd, envsdi) / 2)
    @test abs(edi - ed) < 2e-4
    @test -0.4431 - 1e-9 < edi < -0.375 + 1e-9

    # ---- observables ----
    Z = Sz(T)
    @test real(expectationvalue(ψg, H)) ≈ 2eg atol = 1e-8      # 默认 DMRGCache 路径
    @test abs(expectationvalue(ψg, (1,) => Z)) < 0.15         # SU(2) 单态：⟨Sz⟩ ≈ 0
    # （能量收敛到 1e-9 后仍允许 ~0.08 的对称破缺残差，阈值从宽）
    # SU(2) 对称：⟨SzSz⟩ = e₀/3
    G = correlator(ψg, Z, Z, 1, 2:3)
    @test abs(G[1] - e_exact / 3) < 5e-3
    S = entropy(ψg)
    @test 0.5 < S < 2.0                                        # 临界 Heisenberg 熵为正
    spec = entanglement_spectrum(ψg)
    @test abs(sum(spec) - 1) < 1e-10 && spec[1] >= spec[end]

    # ---- mult / w1w2 ----
    I2 = identityimpo(T, [2, 2])
    ψa, ova = mult(I2, ψg)
    @test abs(dot(ψa, ψg)) ≈ 1 atol = 1e-8
    @test real(ova) ≈ 2 atol = 1e-8                            # 恒等 MPO 和式期望 = N

    # WII 时间演化（make_time_mpo 直接接受 2-site Jordan 哈密顿量）：能量守恒
    # （WII·GS 的朴素键维超过默认 trunc.D=64，用 GS 本身作初态）
    W2t = make_time_mpo(H, 0.01, WII())
    ψw, ovw = mult(W2t, ψg; ψ₀ = ψg)
    @test abs(real(expectationvalue(ψw, H) / 2) - eg) < 1e-4
    @test real(ovw) > 0.999

    # WI 路径（阈值覆盖 WI MPO 自身 O(dt) 能量不守恒，同 1-site 测试）
    W1t = make_time_mpo(H, 0.01, WI())
    ψw1, _ = mult(W1t, ψg; ψ₀ = ψg)
    @test abs(real(expectationvalue(ψw1, H) / 2) - eg) < 1e-4

    # MPO 压缩：2-site 恒等 MPO 压到 D=1
    comp = mpo_compress(I2, 1)
    @test max_bonddim(comp.W) == 1
    @test abs(comp.overlap - 2) < 1e-8

    # ---- tdvp ----
    tspan = 0:0.01:0.1
    ψtv, _, history = time_evolve(ψg, H, tspan, TDVP();
                                  observer = (ψ, k, t) -> real(expectationvalue(ψ, H) / 2))
    @test length(history) == 11
    @test abs(history[end] - eg) < 1e-6                        # 实时间能量守恒
    @test abs(norm(ψtv) - 1) < 1e-8

    # 虚时间：随机初态收敛到基态
    ψβ, _, _ = time_evolve(randomimps(T, [2, 2], 12), H, 0:0.05:20, TDVP();
                           imaginary_evolution = true)
    @test abs(real(expectationvalue(ψβ, H) / 2) - e_exact) < 1e-3
end
