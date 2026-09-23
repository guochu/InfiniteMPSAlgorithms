# =====================================================================
# 有限温热平衡态（纯化 + 虚时 TDVP）与 MPSKit 对比
# （固化自 debug/finite_t_tdvp.jl，去掉日志文件，保留断言）
#
# 协议（两包严格同参）：
#   模型：TFIM H = −JΣσˣσˣ − hΣσᶻ（J = h = 1），d = 2，L = 2 单胞；
#   加倍空间：每个物理位点升为 d² = 4（融合约定同 vectorize：
#     f = u + d·(d−1)，u = system 快指标，d = ancilla 慢指标）；
#   纯化初态：|I⟩ 的 χ=D 提升（积态）：A[a,f,b] = δ[a,b]·φ[f]，
#     φ = |I⟩/√d（系统 RDM = I/d，无穷温，E(β=0) = 0）；
#   生成元：H_gen = H⊗I + I⊗Hᵀ（系统/辅助两条 σx 串 + 两条点场）；
#   演化：虚时 1-site TDVP，dβ = 0.05，β ∈ (0, 1]，χ = D = 8；
#   测量：E(β) = ⟨H⊗I⟩，单胞总能量 = 2×能量密度。
# 对齐判据：两包 E(β) 曲线最大偏差 < 1e-6（同一 dβ、同一 Krylov 默认容差）。
# =====================================================================

@testset "有限温纯化虚时 TDVP ≡ MPSKit" begin
    d = 2                  # 物理维度
    D = 8                  # 纯化键维
    dβ = 0.05
    βmax = 1.0
    L = 2                  # 单胞长度
    J = 1.0
    h = 1.0

    # ---- 融合空间的局部算符（f = u + d·(d−1)：u = system 快指标） ----
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Sxm = Matrix{ComplexF64}([0 1; 1 0])
    Szm = Matrix{ComplexF64}([1 0; 0 -1])
    SxA = kron(Sxm, I2)                      # system 通道：A⊗I
    SzA = kron(Szm, I2)
    Sxanc = kron(I2, transpose(Sxm))         # ancilla 通道：I⊗Bᵀ
    Szanc = kron(I2, transpose(Szm))

    # ---- 生成元/测量（Jordan 稀疏，mpohamiltonian 构造器） ----
    # H_gen = H⊗I + I⊗Hᵀ：系统/辅助两条 NN σxσx 串 + 两条点场；
    # H_sys = H⊗I（测量系统通道能量）。
    Hgen = SparseIMPO([mpohamiltonian(-h * SzA - h * Szanc,
                                      [(-J, SxA, SxA), (-J, Sxanc, Sxanc)]),
                       mpohamiltonian(-h * SzA - h * Szanc,
                                      [(-J, SxA, SxA), (-J, Sxanc, Sxanc)])])
    Hsys = SparseIMPO([mpohamiltonian(-h * SzA, [(-J, SxA, SxA)]),
                       mpohamiltonian(-h * SzA, [(-J, SxA, SxA)])])

    # ---- 纯化初态：χ=D 提升 |I⟩（积态，E(0)=0） ----
    T = ComplexF64
    AL0 = zeros(T, D, d * d, D)
    for s in 1:d
        f = s + d * (s - 1)                  # |s⟩⊗|s⟩ 的对角融合指标
        for a in 1:D
            AL0[a, f, a] = 1 / √d
        end
    end
    ψ0 = CanonicalIMPS(PeriodicVector([AL0, copy(AL0)]),
                       PeriodicVector([copy(AL0), copy(AL0)]),
                       PeriodicVector([Matrix{T}(I, D, D), Matrix{T}(I, D, D)]),
                       PeriodicVector([copy(AL0), copy(AL0)]))

    # ---- 本包：虚时冷却 ----
    env = DMRGCache(ψ0, Hgen)
    ψ = copy(ψ0)
    βs = collect(0:dβ:βmax)
    Es_ours = Float64[real(expectationvalue(ψ, Hsys))]
    for β in dβ:dβ:βmax
        ψ, env = timestep(ψ, Hgen, β, dβ, TDVP(), env; imaginary_evolution = true)
        push!(Es_ours, real(expectationvalue(ψ, Hsys)))
    end

    # ---- MPSKit：同一生成元/测量算符（两位点 Pair 构造） ----
    op2(OM::AbstractMatrix) =
        TensorMap(reshape(Matrix{ComplexF64}(OM), 16, 16), ℂ^4 * ℂ^4, ℂ^4 * ℂ^4)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    # 键算符：NN(系统) + NN(辅助) + 两侧点场的一半（平移等价分配，每位点一份点场）
    GA = kron(SxA, SxA) + kron(Sxanc, Sxanc) -
         h / 2 * (kron(SzA, I4) + kron(I4, SzA) + kron(Szanc, I4) + kron(I4, Szanc))
    Hgen_k = MPSKit.InfiniteMPOHamiltonian(fill(ℂ^4, 2),
                                           (1, 2) => op2(GA), (2, 1) => op2(GA))
    GAs = kron(SxA, SxA) - h / 2 * (kron(SzA, I4) + kron(I4, SzA))
    Hsys_k = MPSKit.InfiniteMPOHamiltonian(fill(ℂ^4, 2),
                                           (1, 2) => op2(GAs), (2, 1) => op2(GAs))

    ψk = mkinfinitemps(ψ0)
    envk = MPSKit.environments(ψk, Hgen_k)
    Es_mps = Float64[]
    for β in vcat([0.0], dβ:dβ:βmax)
        if β > 0
            ψk, envk = MPSKit.timestep(ψk, Hgen_k, β, dβ, MPSKit.TDVP(), envk;
                                       imaginary_evolution = true)
        end
        push!(Es_mps, real(MPSKit.expectation_value(ψk, Hsys_k)) / real(MPSKit.dot(ψk, ψk)))
    end

    # ---- 对齐断言 ----
    # 注：生成元双通道各携带 β ⇒ 纯化态为 e^{-2βH} 热态（精确自由费米子曲线
    # 作锚点仅在 debug 版打印核对）；对齐判据 = 两包 E(β) 曲线最大偏差
    #（debug 版实测 < 1e-6）。
    devmax = 0.0
    for i in eachindex(βs)
        dev = abs(Es_ours[i] - Es_mps[i])
        devmax = max(devmax, dev)
    end
    @test abs(Es_ours[1]) < 1e-10            # 无穷温 E(0) = 0
    @test Es_ours[end] < 0                   # 虚时冷却降低能量
    @test devmax < 1e-6                      # 两包 E(β) 曲线逐步对齐
end
