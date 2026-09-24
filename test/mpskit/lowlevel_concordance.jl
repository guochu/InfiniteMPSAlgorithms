# =====================================================================
# 底层函数与 MPSKit 逐步对齐（含默认参数）
#
# 覆盖：Defaults 常量、DynamicTol/updatetol、Left/Right/MixedCanonical 默认值、
# gaugefix!/uniform_leftorth!/uniform_rightorth!/regauge!、calc_galerkin、
# regularize!、TransferMatrix/fixedpoint、DMRGCache/recalculate!、
# transfer_leftenv!/transfer_rightenv!、push_env_left/right（对应 MPSKit 内部
# transfer_left/transfer_right）、dominant_env（对应 MPSKit compute_leftenvs!
# 的 fixedpoint 步骤，MPSKit 无公开同名接口）、dot。
#
# 哈密顿量构造语义差异（重要）：
# - 本包模型构造（tfim_hamiltonian 等）返回 1-单胞平移不变 MPO，平铺到 N-site
#   状态时自动枚举全部周期键与全部 site 的单体项；
# - MPSKit 的 InfiniteMPOHamiltonian pair 接口只在**显式给出的位置**放项，
#   不做平移镜像。N=2 的 TFIM 须写
#   `1 => -hσz, 2 => -hσz, (1,2) => -Jσxσx, (2,3) => -Jσxσx`：
#   wrap 键的正确语法是 (2,3)（从 site 2 出发向前跨胞到 site 3≡1），
#   不能写 (2,1)——后者会被布局成与 (1,2) 同路由的重复项，其闭合列
#   记账不自洽（对算符仍等价，VUMPS 仍收敛到同一解析值，但期望值诊断失真）。
# - 该配置下两包在 N=2 随机态上 ⟨H⟩ 与 ϵ 逐位一致（1e-13），见下方测试。
#
# 已知的（可调和）差异，测试注释中标明：
# - Defaults.verbosity：本包默认 0（静默），MPSKit 为 VERBOSE_ITER(3)——纯日志行为；
# - gaugefix! 后本包同步刷新 ψ.AC；MPSKit 的 InfiniteMPS.AC 字段留在旧值
#   （其算法均经 AL/AR/C 读取，无实际影响）；本包维护 AC ≡ AL·C 不变量；
# - regauge! 向量版本包返回新数组，MPSKit 原地改写返回（数值一致）；
# - TransferMatrix 的翻转方向：本包 side = :left/:right 关键字，MPSKit 用
#   isflipped::Bool（:left ≡ flipped，:right ≡ unflipped，见逐元测试）；
# - push_env_left/right 与 dominant_env 在 MPSKit 中为内部机制（无公开同名
#   接口），通过与 MPSKit 的 TransferMatrix / environments 输出逐元对比验证。
#
# 已知问题（@test_broken 固化）：无。此前记录的 N ≥ 2 闭合列差异经查为
# 测试侧 wrap 键语法错误（(2,1) 应为 (2,3)），修正后两包在随机态上亦逐位一致。
# =====================================================================

using InfiniteMPSAlgorithms:  # 显式消歧（MPSKit 亦导出其中部分名称）
    DynamicTol, updatetol, LeftCanonical, RightCanonical, MixedCanonical,
    gaugefix!, regauge!, regularize!, TransferMatrix, fixedpoint,
    DMRGCache, recalculate!, transfer_leftenv!, transfer_rightenv!,
    calc_galerkin, push_env_left, push_env_right, Defaults

"Run the entire low-level comparison under the given scalar type `T` (all
fixtures and testsets parametrized): both real (`Float64`, with the additional
assertion that real inputs stay real — no spurious channel promotion) and
complex (`ComplexF64`) must align with MPSKit."
function run_lowlevel_concordance(::Type{T}) where {T<:Number}
    # ---- 共享固定装置：同一随机（未规范）张量、同一 TFIM 哈密顿量 ----
    Random.seed!(77)
    N = 2
    d = 2
    D = 5

    A_ours = [randn(T, D, d, D) for _ in 1:N]
    ψA = CanonicalIMPS([copy(a) for a in A_ours])           # 本包规范存储
    ψ_mk = mkinfinitemps(ψA)                                # 同一规范 MPSKit 存储
    J, h = 1.0, 1.3
    H_our = tfim_hamiltonian(J = J, h = h, T = T)           # 1-单胞平移不变
    H_mk = MPSKit.InfiniteMPOHamiltonian(fill(fld(T)^d, N),
                                         1 => -h * σz_tk(T),
                                         2 => -h * σz_tk(T),
                                         (1, 2) => -J * (σx_tk(T) ⊗ σx_tk(T)),
                                         (2, 3) => -J * (σx_tk(T) ⊗ σx_tk(T)))

    # N = 1 固定装置（哈密顿量敏感的跨包对比在此进行）
    ψA1 = CanonicalIMPS([copy(A_ours[1])])
    ψ_mk1 = mkinfinitemps(ψA1)
    H_mk1 = MPSKit.InfiniteMPOHamiltonian(fill(fld(T)^d, 1),
                                          1 => -h * σz_tk(T),
                                          (1, 2) => -J * (σx_tk(T) ⊗ σx_tk(T)))

    "把键矩阵包装成 MPSKit TensorMap(D ← D)。"
    bond_tm(M::AbstractMatrix) = TensorMap(copy(M), fld(T)^size(M, 1), fld(T)^size(M, 2))

    "复比例对齐残差：‖X − e^{iφ}·Y‖/‖X‖（e^{iφ} 取最小相位差）。"
    _phase_aligned_relerr(X::AbstractArray, Y::AbstractArray) = begin
        z = dot(vec(Y), vec(X))
        s = abs(z) == 0 ? one(T) : z / abs(z)
        return norm(X .- s .* Y) / norm(X)
    end

    @testset "底层对比 [$T]" begin

@testset "Defaults 常量 ≡ MPSKit" begin
    @test Defaults.eltype == MPSKit.Defaults.eltype
    @test Defaults.maxiter == MPSKit.Defaults.maxiter
    @test Defaults.tolgauge == MPSKit.Defaults.tolgauge
    @test Defaults.tol == MPSKit.Defaults.tol
    @test Defaults.krylovdim == MPSKit.Defaults.krylovdim
    @test Defaults.dynamic_tols == MPSKit.Defaults.dynamic_tols
    @test Defaults.tol_min == MPSKit.Defaults.tol_min
    @test Defaults.tol_max == MPSKit.Defaults.tol_max
    @test Defaults.eigs_tolfactor == MPSKit.Defaults.eigs_tolfactor
    @test Defaults.gauge_tolfactor == MPSKit.Defaults.gauge_tolfactor
    @test Defaults.envs_tolfactor == MPSKit.Defaults.envs_tolfactor
    @test Defaults.VERBOSE_NONE == MPSKit.Defaults.VERBOSE_NONE == 0
    @test Defaults.VERBOSE_WARN == MPSKit.Defaults.VERBOSE_WARN == 1
    @test Defaults.VERBOSE_CONV == MPSKit.Defaults.VERBOSE_CONV == 2
    @test Defaults.VERBOSE_ITER == MPSKit.Defaults.VERBOSE_ITER == 3
    @test Defaults.VERBOSE_ALL == MPSKit.Defaults.VERBOSE_ALL == 4
    # 注：Defaults.verbosity 本包默认 0，MPSKit 为 VERBOSE_ITER —— 有意的差异
    #（本包默认静默），其余算法级默认（alg_eigsolve 的 verbosity = 0 等）一致。
end

@testset "DynamicTol / updatetol ≡ MPSKit（含默认参数）" begin
    dt_our = DynamicTol((tol = 1.0, maxiter = 5))
    dt_mk = MPSKit.DynamicTol((tol = 1.0, maxiter = 5))
    @test dt_our.tol_min == dt_mk.tol_min == 1.0e-6      # DynamicTol 默认参数
    @test dt_our.tol_max == dt_mk.tol_max == 1.0e-2
    @test dt_our.tol_factor == dt_mk.tol_factor == 0.1
    for (iter, ϵ) in ((1, 1.0), (3, 1.0e-3), (10, 1.0e-8), (0, 1.0e-9))
        a = updatetol(dt_our, iter, ϵ)
        b = MPSKit.updatetol(dt_mk, iter, ϵ)
        @test a.tol == b.tol                              # 同一 clamp 公式
        @test a.tol == clamp(ϵ * 0.1 / sqrt(max(iter, 1)), 1.0e-6, 1.0e-2)
    end
    # 非 DynamicTol 直通（恒等）
    plain = (tol = 1.0, maxiter = 5)
    @test updatetol(plain, 3, 1.0e-4) === plain
    @test MPSKit.updatetol(plain, 3, 1.0e-4) === plain
end

@testset "Left/Right/MixedCanonical 默认参数 ≡ MPSKit" begin
    for (ours, mk) in ((LeftCanonical(), MPSKit.LeftCanonical()),
                       (RightCanonical(), MPSKit.RightCanonical()))
        @test ours.tol == mk.tol == Defaults.tolgauge
        @test ours.maxiter == mk.maxiter == Defaults.maxiter
        @test ours.verbosity == mk.verbosity == Defaults.VERBOSE_WARN
        @test ours.eig_miniter == mk.eig_miniter == 10
    end
    @test MixedCanonical().order == :LR
    @test MPSKit.MixedCanonical().order == :LR
end

@testset "uniform_leftorth! / uniform_rightorth! 直接调用 ≡ MPSKit（默认参数）" begin
    # 左正交化（写入 AL, C）
    AL1 = [copy(a) for a in ψA.AL]
    C1 = InfiniteMPSAlgorithms.PeriodicVector([copy(c) for c in ψA.C])
    InfiniteMPSAlgorithms.uniform_leftorth!((AL1, C1), [copy(a) for a in A_ours],
                                            copy(ψA.C[end]), LeftCanonical())
    ALm = MPSKit.PeriodicArray([mkmpstensor(a) for a in ψA.AL])
    Cm = MPSKit.PeriodicVector([bond_tm(c) for c in ψA.C])
    MPSKit.uniform_leftorth!((ALm, Cm), [mkmpstensor(a) for a in A_ours],
                             bond_tm(ψA.C[end]), MPSKit.LeftCanonical())
    for ℓ in 1:N
        @test _phase_aligned_relerr(AL1[ℓ], tensor_to_array(ALm[ℓ])) < 1e-7
        @test _phase_aligned_relerr(C1[ℓ], tensor_to_array(Cm[ℓ])) < 1e-7
        @test eltype(AL1[ℓ]) == T && eltype(C1[ℓ]) == T   # 实输入不产生虚假提升
    end
    # 右正交化（写入 AR, C）
    AR1 = [copy(a) for a in ψA.AR]
    C2 = InfiniteMPSAlgorithms.PeriodicVector([copy(c) for c in ψA.C])
    InfiniteMPSAlgorithms.uniform_rightorth!((AR1, C2), [copy(a) for a in A_ours],
                                             copy(ψA.C[end]), RightCanonical())
    ARm = MPSKit.PeriodicArray([mkmpstensor(a) for a in ψA.AR])
    Cm2 = MPSKit.PeriodicVector([bond_tm(c) for c in ψA.C])
    MPSKit.uniform_rightorth!((ARm, Cm2), [mkmpstensor(a) for a in A_ours],
                              bond_tm(ψA.C[end]), MPSKit.RightCanonical())
    for ℓ in 1:N
        @test _phase_aligned_relerr(AR1[ℓ], tensor_to_array(ARm[ℓ])) < 1e-7
        @test _phase_aligned_relerr(C2[ℓ], tensor_to_array(Cm2[ℓ])) < 1e-7
    end
end

@testset "gaugefix! ≡ MPSKit（同 A 同 C₀，order = :LR/:RL/:L/:R 默认参数）" begin
    for order in (:LR, :RL, :L, :R)
        ψo = CanonicalIMPS([copy(a) for a in A_ours])
        gaugefix!(ψo, [copy(a) for a in A_ours]; order = order)
        ψm = mkinfinitemps(ψA)
        MPSKit.gaugefix!(ψm, [mkmpstensor(a) for a in A_ours]; order = order)
        # 直接逐元对比 MPSKit 张量（规范只差全局相位，逐 site 相位对齐）。
        # 注意：无限 MPS 的态由 (AL, C) 共同决定，不能用 from_mpskit 之类的
        # 单场重建做态比较。
        for ℓ in 1:N
            @test _phase_aligned_relerr(ψo.C[ℓ], tensor_to_array(ψm.C[ℓ])) < 1e-7
            order === :R ||
                @test _phase_aligned_relerr(ψo.AL[ℓ], tensor_to_array(ψm.AL[ℓ])) < 1e-7
            order === :L ||
                @test _phase_aligned_relerr(ψo.AR[ℓ], tensor_to_array(ψm.AR[ℓ])) < 1e-7
        end
        # 本包附加行为：gaugefix! 后维护 AC ≡ AL·C（MPSKit 的 AC 字段留在旧值）
        order in (:LR, :RL) && @test ismixedcanonical(ψo)
        # 实输入保持实（无虚假通道提升）；输出规范与 MPSKit 一致
        @test InfiniteMPSAlgorithms.scalartype(ψo) == T
    end
end

@testset "regauge! ≡ MPSKit（默认 alg_orth）" begin
    AC0 = [ψA.AC[ℓ] .+ 1.0e-2 .* randn(T, D, d, D) for ℓ in 1:N]
    AL_new = regauge!([copy(ac) for ac in AC0], [copy(c) for c in ψA.C])
    ACm = [mkmpstensor(ac) for ac in AC0]
    Cm = [bond_tm(c) for c in ψA.C]
    AL_mk = MPSKit.regauge!(ACm, Cm)                     # MPSKit 原地改写 ACm 并返回 AL
    for ℓ in 1:N
        @test _phase_aligned_relerr(AL_new[ℓ], tensor_to_array(AL_mk[ℓ])) < 1e-7
    end
    # 标量形式
    AL1 = regauge!(copy(AC0[1]), copy(ψA.C[1]))
    ALm1 = MPSKit.regauge!(mkmpstensor(AC0[1]), bond_tm(ψA.C[1]))
    @test _phase_aligned_relerr(AL1, tensor_to_array(ALm1)) < 1e-7
end

@testset "calc_galerkin ≡ MPSKit（N = 1）" begin
    envs_our = DMRGCache(ψA1, H_our)
    envs_mk = MPSKit.environments(ψ_mk1, H_mk1)
    @test all(mixedcanonical_error(ψA1) .< 1e-10)     # 共享固定装置未被污染
    ϵ_our = calc_galerkin(ψA1, H_our, envs_our)
    ϵ_mk = MPSKit.calc_galerkin(ψ_mk1, H_mk1, ψ_mk1, envs_mk)
    @test abs(ϵ_our - ϵ_mk) < 1e-6 * max(ϵ_our, ϵ_mk)
    # 逐 site 版本：MPSKit 的实现为 ‖x − AL·(AL†·x)‖（x = H·AC 归一化后），
    # 此处内联展开同一公式对照。
    Hac = AC_hamiltonian(1, ψA1, H_our, ψA1, envs_our)
    x = Hac(ψA1.AC[1])
    x ./= norm(x)
    AL = ψA1.AL[1]
    q = similar(x, D, D)
    @tensor q[c, b] := conj(AL[a, s, c]) * x[a, s, b]
    p = similar(x)
    @tensor p[a, s, b] := AL[a, s, c] * q[c, b]
    ϵ_site = norm(x .- p)
    ϵ_mk_site = MPSKit.calc_galerkin(1, ψ_mk1, H_mk1, ψ_mk1, envs_mk)
    @test abs(ϵ_site - ϵ_mk_site) < 1e-6 * max(ϵ_site, 1.0)
end

@testset "regularize! ≡ MPSKit（非单位 lvec/rvec）" begin
    v0 = randn(T, D, D)
    lv = randn(T, D, D)
    rv = randn(T, D, D)
    v_our = copy(v0)
    regularize!(v_our, lv, rv)
    v_mk = bond_tm(v0)
    MPSKit.regularize!(v_mk, bond_tm(lv), bond_tm(rv))
    @test v_our ≈ reshape(_tkdata(v_mk), D, D) atol = 1e-12
end

@testset "TransferMatrix 方向与 fixedpoint ≡ MPSKit" begin
    # 混合转移矩阵主导特征值（规范固定点）：|λ| = 1，两包一致
    alg_our = KrylovKit.Arnoldi(; tol = 1.0e-12, krylovdim = 20, maxiter = 200, eager = true)
    alg_mk = MPSKit.KrylovKit.Arnoldi(; tol = 1.0e-12, krylovdim = 20, maxiter = 200, eager = true)
    λ1, v1 = fixedpoint(TransferMatrix([copy(a) for a in ψA.AL], [copy(a) for a in ψA.AL];
                                       side = :left),
                        vec(copy(ψA.C[end])), :LM, alg_our)
    @test abs(abs(λ1) - 1) < 1e-8
    # 与 MPSKit flip(TransferMatrix(A, AL)) 逐元对齐（side = :left ≡ flipped）
    Tmk = TensorKit.flip(MPSKit.TransferMatrix([mkmpstensor(a) for a in ψA.AL],
                                               nothing,
                                               [mkmpstensor(a) for a in ψA.AL]))
    λ2, v2 = MPSKit.fixedpoint(Tmk, bond_tm(ψA.C[end]), :LM, alg_mk)
    @test abs(λ1 - λ2) < 1e-8
    C1m = reshape(v1, D, D)
    C2m = tensor_to_array(v2)
    @test _phase_aligned_relerr(C1m, C2m) < 1e-6
    # 非翻转方向（side = :right ≡ unflipped）：主导特征值同为单位模
    λ3, _ = fixedpoint(TransferMatrix([copy(a) for a in ψA.AR], [copy(a) for a in ψA.AR];
                                      side = :right),
                       vec(copy(ψA.C[1])), :LM, alg_our)
    @test abs(abs(λ3) - 1) < 1e-8
end

@testset "DMRGCache 默认参数 / recalculate! / transfer_*（本包内部一致性）" begin
    # 默认 kwargs 与显式 Defaults 值一致
    e1 = DMRGCache(ψA, H_our)
    e2 = DMRGCache(ψA, H_our; tol = Defaults.tol, maxiter = Defaults.maxiter,
                   krylovdim = Defaults.krylovdim)
    for ℓ in 1:N
        @test e1.lefts[ℓ] ≈ e2.lefts[ℓ] rtol = 1e-6 atol = 1e-10
        @test e1.rights[ℓ] ≈ e2.rights[ℓ] rtol = 1e-6 atol = 1e-10
    end
    # recalculate! 重算 ≡ 新建
    recalculate!(e1, ψA, H_our)
    fresh = DMRGCache(ψA, H_our)
    for ℓ in 1:N
        @test e1.lefts[ℓ] ≈ fresh.lefts[ℓ] rtol = 1e-6 atol = 1e-10
        @test e1.rights[ℓ] ≈ fresh.rights[ℓ] rtol = 1e-6 atol = 1e-10
    end
    # transfer_* 增量 ≡ 直接 push（本包内部一致性）
    transfer_leftenv!(e1, ψA, H_our, ψA, 2)
    L2_direct = push_env_left(leftenv(e1, 1), tompotensor(H_our[1]), ψA.AL[1])
    @test leftenv(e1, 2) ≈ L2_direct rtol = 1e-10
    transfer_rightenv!(e1, ψA, H_our, ψA, 2)
    R2_direct = push_env_right(rightenv(e1, 1), tompotensor(H_our[1]), ψA.AR[1])
    @test rightenv(e1, 2) ≈ R2_direct rtol = 1e-10
end

@testset "DMRGCache 环境逐元 ≡ MPSKit（N = 1）" begin
    envs_our = DMRGCache(ψA1, H_our)
    envs_mk = MPSKit.environments(ψ_mk1, H_mk1)
    GL_our = leftenv(envs_our, 1)
    GR_our = rightenv(envs_our, 1)
    GL_mk = envarray(convert(TensorMap, MPSKit.leftenv(envs_mk, 1, ψ_mk1)))
    GR_mk = envarray(convert(TensorMap, MPSKit.rightenv(envs_mk, 1, ψ_mk1)))
    # 逐 level 允许整体缩放（两包环境归一化约定不同，物理等价）
    for l in 1:size(GL_our, 2)
        our_l = GL_our[:, l, :]
        mk_l = GL_mk[:, l, :]
        if norm(our_l) > 1e-15 && norm(mk_l) > 1e-15
            s = dot(vec(mk_l), vec(our_l)) / dot(vec(our_l), vec(our_l))
            @test norm(our_l .* s .- mk_l) / norm(mk_l) < 1e-8
        end
    end
    for l in 1:size(GR_our, 2)
        our_l = GR_our[:, l, :]
        mk_l = GR_mk[:, l, :]
        if norm(our_l) > 1e-15 && norm(mk_l) > 1e-15
            s = dot(vec(mk_l), vec(our_l)) / dot(vec(our_l), vec(our_l))
            @test norm(our_l .* s .- mk_l) / norm(mk_l) < 1e-8
        end
    end
    # 期望值与有效哈密顿量作用逐元一致
    e_our = real(expectationvalue(ψA1, H_our, envs_our))
    e_mk = real(MPSKit.expectation_value(ψ_mk1, H_mk1))
    @test abs(e_our - e_mk) < 1e-9 * abs(e_mk)
    x = randn(T, D, d, D)
    hac_our = AC_hamiltonian(1, ψA1, H_our, ψA1, envs_our)(x)
    hac_mk = mpsarray(MPSKit.AC_hamiltonian(1, ψ_mk1, H_mk1, ψ_mk1, envs_mk)(
        mkmpstensor(x)))
    @test norm(hac_our - hac_mk) / norm(hac_our) < 1e-9
end

@testset "push_env_* 跨包逐元 ≡ MPSKit TransferMatrix 应用" begin
    # 双包用同一份 W 数据（本包 tompotensor 生成），检验纯 push 语义逐元一致；
    # TensorKit 包装沿用 MPSKit 真实对象的 space（含对偶标记），避免空间簿记差异。
    W4 = tompotensor(H_our[1])
    nl = size(W4, 1)
    menvs1 = MPSKit.environments(ψ_mk1, H_mk1)
    env_sp_L = space(convert(TensorMap, MPSKit.leftenv(menvs1, 1, ψ_mk1)))
    env_sp_R = space(convert(TensorMap, MPSKit.rightenv(menvs1, 1, ψ_mk1)))
    W_sp = space(convert(TensorMap, H_mk1[1]))
    Wm = TensorMap(copy(vec(_tkdata(mkmpotensor(W4)))), W_sp)
    ALm1 = mkmpstensor(ψA.AL[1])
    ARm1 = mkmpstensor(ψA.AR[1])
    # MPO 通道左推：本包 push_env_left(L, below, W, above) ≡ Lm * T(above, W, below)
    L3 = randn(T, D, nl, D)
    Lm = TensorMap(copy(vec(L3)), env_sp_L)
    out_our = push_env_left(L3, ψA.AL[1], W4, ψA.AL[1])
    out_mk = Lm * MPSKit.TransferMatrix(ALm1, Wm, ALm1)
    @test norm(reshape(_tkdata(out_mk), D, nl, D) .- out_our) / norm(out_our) < 1e-10
    # MPO 通道右推：push_env_right(R, above, W, below) ≡ T(above, W, below) * Rm
    R3 = randn(T, D, nl, D)
    Rm = TensorMap(copy(vec(R3)), env_sp_R)
    out_our = push_env_right(R3, ψA.AR[1], W4, ψA.AR[1])
    out_mk = MPSKit.TransferMatrix(ARm1, Wm, ARm1) * Rm
    @test norm(reshape(_tkdata(out_mk), D, nl, D) .- out_our) / norm(out_our) < 1e-10
    # 恒等通道（rank-2 环境）
    L2 = randn(T, D, D)
    L2m = bond_tm(L2)
    out_our = push_env_left(L2, ψA.AL[1], ψA.AL[1])
    out_mk = L2m * MPSKit.TransferMatrix(ALm1, ALm1)
    @test norm(reshape(_tkdata(out_mk), D, D) .- out_our) / norm(out_our) < 1e-10
end

@testset "dominant_env（≡ MPSKit compute_leftenvs 的 fixedpoint 步骤）" begin
    λL, L1 = InfiniteMPSAlgorithms.dominant_env(ψA)      # 默认 side = :left, which = :LM
    @test abs(λL - 1) < 1e-8                             # AL 转移矩阵恒等通道 λ = 1
    @test norm(L1) ≈ 1 atol = 1e-12                      # 输出 Frobenius 归一
    λR, R1 = InfiniteMPSAlgorithms.dominant_env(ψA; side = :right)
    @test abs(λR - 1) < 1e-8
    @test norm(R1) ≈ 1 atol = 1e-12
    # 与 MPSKit 环境恒等 level 的固定点对齐（N = 1，无构造歧义）
    menvs1 = MPSKit.environments(ψ_mk1, H_mk1)
    GL1 = envarray(convert(TensorMap, MPSKit.leftenv(menvs1, 1, ψ_mk1)))[:, 1, :]
    sL = dot(vec(GL1), vec(L1)) / dot(vec(GL1), vec(GL1))
    @test norm(L1 .- sL .* GL1) < 1e-6
end

@testset "dot ≡ MPSKit（默认 krylovdim = 30）" begin
    @test abs(dot(ψA, ψA) - 1) < 1e-8
    @test abs(MPSKit.dot(ψ_mk, ψ_mk) - 1) < 1e-6          # 随机初猜的 Krylov 收敛
    @test abs(dot(ψA, ψA; krylovdim = 10) - 1) < 1e-8     # krylovdim 关字生效
    ψB = CanonicalIMPS([randn(T, D, d, D) for _ in 1:N])
    ψB_mk = mkinfinitemps(ψB)
    # 不同态之间的主导混合转移特征值：两包一致
    @test abs(dot(ψA, ψB) - MPSKit.dot(ψ_mk, ψB_mk)) < 1e-6 * max(1.0, abs(dot(ψA, ψB)))
end

@testset "随机态 ⟨H⟩ / ϵ ≡ MPSKit（N = 2）" begin
    # 正确 wrap 语法（(2,3)）下，随机非收敛态上 ⟨H⟩ 与 ϵ 亦逐位一致：
    # 本包平铺 Schur 结构 ≡ MPSKit 显式 pair 结构（数学等价的直接验证）。
    envs_our = DMRGCache(ψA, H_our)
    envs_mk = MPSKit.environments(ψ_mk, H_mk)
    e_our = real(expectationvalue(ψA, H_our, envs_our))
    e_mk = real(MPSKit.expectation_value(ψ_mk, H_mk))
    @test abs(e_our - e_mk) < 1e-9 * abs(e_mk)
    ϵ_our = calc_galerkin(ψA, H_our, envs_our)
    ϵ_mk = MPSKit.calc_galerkin(ψ_mk, H_mk, ψ_mk, envs_mk)
    @test abs(ϵ_our - ϵ_mk) < 1e-9 * max(ϵ_our, ϵ_mk)
end
    end   # outer @testset "底层对比 [$T]"
    return nothing
end

# 实数与复数夹具都必须与 MPSKit 对齐
run_lowlevel_concordance(Float64)
run_lowlevel_concordance(ComplexF64)
