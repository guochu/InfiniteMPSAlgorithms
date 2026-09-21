@testset "W^I / W^II 时间演化" begin
    T = ComplexF64
    Hxxz = heisenberg_xxz(T = T)   # (; mpo, bulk, hamiltonian)
    HJ = Hxxz.hamiltonian   # Jordan 哈密顿量（能量断言用）
    bulk = Hxxz.bulk
    dt = 0.01
    ψ0 = randomimps(T, [2, 2], 8)
    e0 = real(expectationvalue(ψ0, HJ) / 2)

    # 恒等哈密顿量：WI 演化 MPO = 恒等，mult 后状态不变
    empty_bulk = bulk_mpo(zeros(T, 2, 2), Tuple{Float64,Matrix{T},Matrix{T}}[])
    Ievo = make_time_mpo(empty_bulk, dt, WI())
    out = mult(Ievo, ψ0)
    @test abs(real(expectationvalue(out[1], HJ) / 2) - e0) < 1e-10
    @test abs(out[2] - 2) < 1e-8   # 恒等 MPO 的和式期望 = N = 2

    # WII：能量守恒（阈值覆盖 WII MPO 自身的 O(dt) 能量不守恒：精确施加实测
    # ~1.9e-5 @ dt=0.01，随 dt 线性缩放；VOMPS 施加额外偏差同量级）
    W2 = make_time_mpo(bulk, dt, WII())
    out2ψ, out2ov = mult(W2, ψ0; ψ₀ = ψ0)
    @test abs(real(expectationvalue(out2ψ, HJ) / 2) - e0) < 1e-4
    @test real(out2ov) > 0.999

    # WI 精度：误差量级 = WI MPO 自身的 O(dt) 能量不守恒（精确施加实测 ~1.6e-5，
    # 与 WII 同量级，故不做 err1 > err2 排序断言）
    W1 = make_time_mpo(bulk, dt, WI())
    out1ψ, _ = mult(W1, ψ0; ψ₀ = ψ0)
    err1 = abs(real(expectationvalue(out1ψ, HJ) / 2) - e0)
    @test err1 < 1e-4

    # make_time_mpo 直接接受 Jordan 哈密顿量；阈值同上（MPO 自身误差）
    W2h = make_time_mpo(HJ, dt, WII())
    out3ψ, _ = mult(W2h, ψ0; ψ₀ = ψ0)
    @test abs(real(expectationvalue(out3ψ, HJ) / 2) - e0) < 1e-4

    # MPO 压缩：恒等 MPO 压到 D=1（本包扩展，MPSKit 无对标）。结构 + 压缩保真度
    # （归一化保真度 × N）是规范不变的；输出的逐 site 相位是压缩本征解的规范
    # 自由度（MPSKit 同样不钉定本征解相位），期望值等规范依赖量不作断言。
    I2 = identityimpo(T, [2, 2])
    comp = mpo_compress(I2, 1)
    @test max_bonddim(comp.W) == 1
    @test abs(comp.overlap - 2) < 1e-8
end
