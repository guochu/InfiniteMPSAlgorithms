# =====================================================================
# VUMPS / IDMRG 与 MPSKit 的行为一致性（固化自 debug/vumps_idmrg_alignment.jl）
#
# 一致性判据（**固定小迭代数，不跑收敛**）：同一初态、同一参数（D、动态容差），
# `tol = 0` 强制两包恰好跑 k 轮（k = 1, 2, 5）。此时两包的态都离基态很远——
# 若在前几轮迭代上行为就一致，才是「行为一致」的最强证据：任何算法语义差异
# （局部映射、环境约定、规范恢复的可分派差异）都会在第一轮就体现为远超
# round-off 的偏差。实测（TFIM/Heisenberg/DenseIMPO 通道，k = 1, 2, 5）：
# - VUMPS：能量差 ~1e-14、态 ray 残差 ~1e-12（严格一致，round-off 量级）；
#   态一致判据双轨：periodic-repr 射线残差（辅助）+ 包内 fidelity
#   （转移矩阵主导本征值的 infinite 语义，主判）；
# - IDMRG：能量差 ≤ 8e-5、态残差 ≤ 4e-3（Gauss–Seidel 顺序扫描 + 中间规范
#   约定的实现差异，随 k 收敛：两者收敛到同一不动点）；
# - DenseIMPO（InfiniteMPO 通道）：能量差 ≤ 2e-6；态**不**比对——恒等层
#   bookkeeping 使转移矩阵主导本征空间（近似）简并，环境取向不由归一化唯一
#   确定（DMRGCache 的 DenseIMPO 版说明），两包的态路径不可复现，但能量对
#   环境的物理等价类不敏感。
# =====================================================================

"两包态的 ray 残差（dense 周期 trace 表示，规范与尺度不变）——periodic repr
把 Infinite MPS 当成有限环 MPS 处理，概念上不完备，只作辅助证据。"
function _state_ray_residual(ψ_our::CanonicalIMPS, ψ_mk)
    a = vec(_dense_mps_repr(ψ_our))
    b = vec(_dense_mps_repr(from_mpskit(ψ_mk)))
    ls = dot(b, a) / dot(b, b)
    return norm(a .- ls .* b) / norm(a)
end

"两包态的包内 fidelity（转移矩阵主导本征值的 infinite 射线语义，主判；
地板为 eigsolve 相对误差 ~1e-12 量级）。"
_state_fidelity(ψ_our::CanonicalIMPS, ψ_mk) =
    fidelity(DenseIMPS(ψ_our), DenseIMPS(from_mpskit(ψ_mk)))

"同初态、固定迭代数（`tol = 0` ⇒ 恰好 k 轮）的 VUMPS/IDMRG 行为一致性：
k 轮后的能量（与态）一致。容差按通道与算法的实测偏差量级给定。
`alg_eigsolve`（可空）转发给本包侧 VUMPS/IDMRG——DenseIMPO 通道的投影有效
哈密顿量因恒等层环境简并非严格厄米，传 Arnoldi 以免 Lanczos 逐 Krylov 步
刷「might not be hermitian」告警。"
function compare_groundstate_k(H_our, H_mk, ψ0, D; ks = (1, 2, 5),
                               e_tol_v = 1e-10, ψ_tol_v = 1e-8, ψ_fid_v = 1e-8,
                               e_tol_i = 1e-3, ψ_tol_i = 1e-2, ψ_fid_i = 1e-4,
                               compare_state = true, alg_eigsolve = nothing)
    L = length(ψ0)
    ours = alg_eigsolve === nothing ? NamedTuple() : (alg_eigsolve = alg_eigsolve,)
    for k in ks
        # ---- VUMPS：至多 k 轮（`tol = 0` 保底；恰到达精确不动点时允许提前停） ----
        ψ1, envs1, i1 = find_groundstate(ψ0, H_our,
                    VUMPS(; D = D, maxiter = k, tol = 0.0, verbosity = 0, ours...))
        ψ2, envs2, i2 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                    MPSKit.VUMPS(maxiter = k, tol = 0.0, verbosity = 0))
        @test i1.niter ≤ k
        @test abs(real(expectationvalue(ψ1, H_our, envs1) / L) -
                  real(MPSKit.expectation_value(ψ2, H_mk) / L)) < e_tol_v
        compare_state && @test _state_ray_residual(ψ1, ψ2) < ψ_tol_v
        compare_state && @test _state_fidelity(ψ1, ψ2) > 1 - ψ_fid_v
        # ---- IDMRG：至多 k 轮（Dense 通道的恒等层结构会在少数几轮内到达
        # 精确不动点——C 漂移恰为 0 而提前收敛，属合法行为） ----
        ψ3, envs3, i3 = find_groundstate(ψ0, H_our,
                    IDMRG(; D = D, maxiter = k, tol = 0.0, verbosity = 0, ours...))
        ψ4, envs4, i4 = MPSKit.find_groundstate(mkinfinitemps(ψ0), H_mk,
                    MPSKit.IDMRG(maxiter = k, tol = 0.0, verbosity = 0))
        @test i3.niter ≤ k
        @test abs(real(expectationvalue(ψ3, H_our, envs3) / L) -
                  real(MPSKit.expectation_value(ψ4, H_mk) / L)) < e_tol_i
        compare_state && @test _state_ray_residual(ψ3, ψ4) < ψ_tol_i
        compare_state && @test _state_fidelity(ψ3, ψ4) > 1 - ψ_fid_i
    end
    return nothing
end

@testset "VUMPS / IDMRG 行为一致性 ≡ MPSKit（同初态固定小迭代）" begin
    T = ComplexF64
    d = 2

    # ---- TFIM（1-site 单胞） ----
    J, h = 1.0, 1.0
    lattice1 = fill(ℂ^d, 1)
    Random.seed!(1234)
    ψ_tfim = CanonicalIMPS([randn(T, 8, d, 8)])
    @testset "TFIM" begin
        compare_groundstate_k(tfim_hamiltonian(J = J, h = h, T = T),
                              MPSKit.InfiniteMPOHamiltonian(lattice1,
                                  1 => -h * σz_tk(T),
                                  (1, 2) => -J * (σx_tk(T) ⊗ σx_tk(T))),
                              ψ_tfim, 8)
    end

    # ---- Heisenberg XXX（2-site 单胞） ----
    lattice2 = fill(ℂ^d, 2)
    hh = (1 / 4) * (σx_tk(T) ⊗ σx_tk(T) + σy_tk(T) ⊗ σy_tk(T) + σz_tk(T) ⊗ σz_tk(T))
    Random.seed!(1234)
    ψ_heis = CanonicalIMPS([randn(T, 12, d, 12), randn(T, 12, d, 12)])
    @testset "Heisenberg" begin
        compare_groundstate_k(heisenberg_hamiltonian(T = T),
                              MPSKit.InfiniteMPOHamiltonian(lattice2, (1, 2) => hh, (2, 3) => hh),
                              ψ_heis, 12)
    end

    # ---- DenseIMPO 哈密顿量通道（MPSKit 的 InfiniteMPO 通道） ----
    # 同一 TFIM 哈密顿量的稠密形式（Schur 稠密化，含恒等层 bookkeeping）：能量
    # 一致；态因环境简并不比对（见文件头说明与 DMRGCache 的 DenseIMPO 版说明）
    Random.seed!(1234)
    ψ_dense = CanonicalIMPS([randn(T, 8, d, 8)])
    @testset "DenseIMPO（InfiniteMPO 通道）" begin
        # 投影有效哈密顿量非严格厄米（恒等层 bookkeeping）⇒ 本包侧局部求解显式
        # 用 Arnoldi；MPSKit 侧保持其默认（行为不比对，只比能量）
        compare_groundstate_k(DenseIMPO(tfim_hamiltonian(J = 1.0, h = 1.0, T = T)),
                              to_mpskit(DenseIMPO(tfim_hamiltonian(J = 1.0, h = 1.0, T = T))),
                              ψ_dense, 8; e_tol_v = 3e-6, e_tol_i = 3e-6,
                              compare_state = false,
                              alg_eigsolve = Defaults.alg_eigsolve(; ishermitian = false))
    end
end
