# =====================================================================
# make_time_mpo（W^I / W^II）与 MPSKit 的行为对齐测试
#
# 对齐标准（双轨）：算子级严格一致——(i) 包内 `distance`/`fidelity
# (::DenseIMPO, ::DenseIMPO)`（vectorize 转移矩阵主导本征值的 infinite 语义，
# 无相位/尺度自由 / 射线语义）；(ii) periodic-repr 直接差（稠密周期 trace，
# 实测 ~1e-16）作交叉验证——periodic trace 把 Infinite MPO 当成有限环 MPO
# 处理，概念上不完备（只能作辅助证据，不能单独作为一致判据）。不做张量
# 逐位比较——同一算符的 MPO 表示有键空间规范自由（实测逐张量相对差
# ~0.2-0.3）。两侧实现的都是 Zaletel 块指数方案，Jordan/Schur 分解同构 ⇒
# 输出 MPO 表示同一算符且同一整体尺度（本包 LinearAlgebra.exp 块指数 vs
# MPSKit Arnoldi exponentiate 解同一块方程）。注意 Gram 消去 + 转移矩阵
# 本征值各自求解的浮点消去给 distance 留下 √ε ~ 1e-8 量级地板（dt = 0.1
# 实测 0 ~ 6e-8），断言阈值相应放宽到 1e-7 而非 1e-10；fidelity 的地板是
# eigsolve 相对误差 ~1e-12 量级。
#
# 注：MPSKit 只实现了 WII（无 WI），WI 无对标对象。
# =====================================================================

const _make_time_mpo = InfiniteMPSAlgorithms.make_time_mpo

@testset "make_time_mpo（WII）≡ MPSKit：算子级一致" begin
    T = ComplexF64
    # 强步长 dt = 0.1：两侧相对严格解 exp(δH) 都不精确（Trotter 误差 O(dt²)），
    # 但两者实现同一 Zaletel 块指数近似 ⇒ 仍应完全一致——对齐 ≠ 精确，
    # 对齐不受 dt 大小影响。实时（δ = -im·dt，两侧默认约定一致）与虚时
    # （δ = -dt）全覆盖。
    dt = 0.1

    # ---- TFIM（N = 1：ZZ 链 + 横场）----
    J, h = 1.0, 1.0
    H = tfim_hamiltonian(J = J, h = h, T = T)
    H_k = MPSKit.InfiniteMPOHamiltonian(fill(ℂ^2, 1),
                                        1 => -h * σz_tk(T),
                                        (1, 2) => -J * (σx_tk(T) ⊗ σx_tk(T)))
    for kwargs in (NamedTuple(), (; imaginary_evolution = true))
        W = _make_time_mpo(H, dt, WII(); kwargs...)
        W_k = from_mpskit(MPSKit.make_time_mpo(H_k, dt, MPSKit.WII(); kwargs...))
        # 双轨：包内 infinite 语义 + periodic-repr（稠密周期 trace 直接差
        # ~1e-16，辅助交叉验证）
        @test distance(W, W_k) < 1.0e-7
        @test fidelity(W, W_k) > 1 - 1.0e-8
        dW = _dense_mpo_repr(W)
        @test norm(dW - _dense_mpo_repr(W_k)) / norm(dW) < 1.0e-10
    end

    # ---- Heisenberg XXX（N = 1，多通道：XX+YY+ZZ 的 Schur 层族）----
    hh = (1 / 4) * (σx_tk(T) ⊗ σx_tk(T) + σy_tk(T) ⊗ σy_tk(T) + σz_tk(T) ⊗ σz_tk(T))
    H2 = heisenberg_hamiltonian(T = T)
    H2_k = MPSKit.InfiniteMPOHamiltonian(fill(ℂ^2, 1), (1, 2) => hh)
    W2_k0 = nothing
    for kwargs in (NamedTuple(), (; imaginary_evolution = true))
        W2 = _make_time_mpo(H2, dt, WII(); kwargs...)
        W2_k = from_mpskit(MPSKit.make_time_mpo(H2_k, dt, MPSKit.WII(); kwargs...))
        @test distance(W2, W2_k) < 1.0e-7
        @test fidelity(W2, W2_k) > 1 - 1.0e-8
        dW2 = _dense_mpo_repr(W2)
        @test norm(dW2 - _dense_mpo_repr(W2_k)) / norm(dW2) < 1.0e-10
        isempty(kwargs) && (W2_k0 = W2_k)
    end

    # 键维约定：两侧输出 MPO 的键维一致（Schur 通道数，去掉首尾恒等层）
    @test max_bonddim(_make_time_mpo(H2, dt, WII())) == max_bonddim(W2_k0)
end
