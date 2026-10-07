# =====================================================================
# unit cell size = 2 的哈密顿量全链路测试
#
# H = Σᵢ [J₁ S_{2i-1}·S_{2i} + J₂ S_{2i}·S_{2i+1}]（2-site 单胞 Schur 形式）：
# - J₁ = J₂ = 1：均匀 Heisenberg（2-site 单胞表示），e₀ = 1/4 − ln2 精确锚点；
# - J₁ = 1, J₂ = 1/2：交变耦合（dimerized），VUMPS 与 IDMRG 交叉验证。
# 覆盖：结构 / vumps / idmrg / mult / w1w2 / tdvp / observables。
# =====================================================================

@testset "unit cell size = 2" begin
    T = ComplexF64
    e_exact = 0.25 - log(2)          # 均匀 Heisenberg 能量密度（精确）

    # ---- 2-site 单胞 Schur 哈密顿量 ----
    Wd(J) = mpohamiltonian(zeros(T, 2, 2),
                           [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J, Sz(T), Sz(T))])
    H = SparseIMPO([Wd(1.0), Wd(1.0)])          # 均匀
    Hd = SparseIMPO([Wd(1.0), Wd(0.5)])         # dimerized（J₁=1, J₂=1/2）

    @testset "结构" begin
        @test length(H) == 2
        @test bonddim(H) == 5
        @test isidentitylevel(H, 1) && isidentitylevel(H, 5)
        @test !isemptylevel(H, 2)
    end

    # ---- vumps ----
    Random.seed!(71)
    ψg, envsg, infog = find_groundstate(randomimps(T, [2, 2]; D = 12), H,
                                        VUMPS(D = 12, maxiter = 300, tol = 1e-10, verbosity = 0))
    eg = real(expectationvalue(ψg, H, envsg) / 2)
    @test infog.itererr < 1e-9
    @test abs(eg - e_exact) < 2e-4

    # TFIM 2-site 单胞：e₀ = −4/π（临界点解析解）
    Wt = mpohamiltonian(-σz(T), [(-1.0, σx(T), σx(T))])
    Ht = SparseIMPO([Wt, Wt])
    ψt2, _, _ = find_groundstate(randomimps(T, [2, 2]; D = 12), Ht,
                                 VUMPS(D = 12, maxiter = 300, tol = 1e-10, verbosity = 0))
    @test abs(real(expectationvalue(ψt2, Ht) / 2) + 4 / π) < 1e-6

    # dimerized：能量介于两个极限之间（J₂=0 孤立二聚体 −0.375 / J₂=1 均匀 −0.4431；
    # S·S 非半正定，两极限间无普遍变分序），VUMPS 与 IDMRG 交叉验证见下
    ψd, envsd, _ = find_groundstate(randomimps(T, [2, 2]; D = 12), Hd,
                                    VUMPS(D = 12, maxiter = 300, tol = 1e-10, verbosity = 0))
    ed = real(expectationvalue(ψd, Hd, envsd) / 2)
    @test -0.4431 - 1e-9 < ed < -0.375 + 1e-9

    # ---- idmrg ----
    ψi, envsi, ϵi = find_groundstate(randomimps(T, [2, 2]; D = 12), H,
                                     IDMRG(D = 12, maxiter = 300, tol = 1e-9))
    ei = real(expectationvalue(ψi, H, envsi) / 2)
    @test abs(ei - e_exact) < 2e-4
    @test abs(ei - eg) < 2e-4
    @test max_bonddim(ψi) <= 12

    # dimerized：VUMPS 与 IDMRG 交叉验证（有能隙，收敛更快）
    ψdi, envsdi, _ = find_groundstate(randomimps(T, [2, 2]; D = 12), Hd,
                                      IDMRG(D = 12, maxiter = 300, tol = 1e-9))
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
    # 严格施加（W * ψ 的规范代表；二参数 mult 已删除，等价表达）
    apply_exact(W, ψ) = InfiniteMPSAlgorithms._global_normalize!(
        CanonicalIMPS((W * DenseIMPS(collect(ψ.AL))).As))
    ψa = apply_exact(I2, ψg)
    @test abs(dot(ψa, ψg)) ≈ 1 atol = 1e-8

    # WII 时间演化（timeevompo 直接接受 2-site Schur 哈密顿量）：能量守恒
    # （D = nothing → 精确的朴素构造 + 规范存储，不压缩）
    W2t = timeevompo(H, -im * 0.01, WII())
    ψw = apply_exact(W2t, ψg)
    @test abs(real(expectationvalue(ψw, H) / 2) - eg) < 1e-4

    # WI 路径（阈值覆盖 WI MPO 自身 O(dt) 能量不守恒，同 1-site 测试）
    W1t = timeevompo(H, -im * 0.01, WI())
    ψw1 = apply_exact(W1t, ψg)
    @test abs(real(expectationvalue(ψw1, H) / 2) - eg) < 1e-4

    # MPO 压缩：2-site 恒等 MPO 压到 D=1
    comp, _, _ = compress(I2, VOMPS(D = 1))
    @test comp isa CanonicalIMPO && max_bonddim(comp) == 1
    @test fidelity(comp, CanonicalIMPO(I2)) ≈ 1 atol = 1e-8

    # ---- tdvp ----（dt = -im·0.01：纯虚步长即实时演化）
    tspan = (-im) .* (0:0.01:0.1)
    ψtv, _, history = time_evolve(ψg, H, tspan, TDVP();
                                  observer = (ψ, k, t) -> real(expectationvalue(ψ, H) / 2))
    @test length(history) == 11
    @test abs(history[end] - eg) < 1e-6                        # 实时间能量守恒
    @test abs(norm(ψtv) - 1) < 1e-8

    # 虚时间：随机初态收敛到基态（dt = -0.05 负实步长即虚时演化）
    ψβ, _, _ = time_evolve(randomimps(T, [2, 2]; D = 12), H, -(0:0.05:20), TDVP())
    @test abs(real(expectationvalue(ψβ, H) / 2) - e_exact) < 1e-3
end
