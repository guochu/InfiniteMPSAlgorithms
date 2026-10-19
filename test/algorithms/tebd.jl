# =====================================================================
# TEBD 量子门（Hastings 更新）测试
#
# Hastings 语义（对齐 TEMPO/GTEMPO）：
# - 窗口 AC[i]·G·AR[i+1] 一次 SVD；AL[i]←U（严格列正交）、AR[i+1]←V†
#   （严格行正交）、C[i]←S（谱直写，从不除奇异值）；AC 按定义重算；
#   接缝解相反侧恒等式（恒等门时精确，一般门 O(门) 正交误差）。
# - 无截断时窗口精确重建：任意门保态精确，swap 为无损重规范。
# - Hastings 技巧在 trunc err → 0 时无损；测试 (1) 验证该收敛性。
# - 大截断下正则形式（AR 侧 + C 谱 + 恒等式网络）仍严格保持；
#   测试 (2) 验证这一点。
# - 一般混合规范态（未做任何谱对齐初始化）上直接 apply!/swap!：正则形式
#   机器精度保持（接缝严格正交、AC = C·AR 严格），门序列（含 wrap 键 (4,5)
#   与非相邻 (1,4)）正则误差不增长，swap²/恒等门/g·g† 无损；
#   测试 (3) 固化该行为。
# =====================================================================

@testset "TEBD gates" begin
    T = ComplexF64
    Random.seed!(2026)
    L, d, chi = 4, 2, 4
    ψ = randomimps(T, fill(d, L); D = chi)
    gaugefix!(ψ, parent(ψ.AR))
    @test ismixedcanonical(ψ)

    @testset "gate 类型（Gate，FMA 统一门接口）" begin
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = pair_gate(Pair(2, 3), G)
        @test g isa Gate
        @test positions(g) == (2, 3)
        @test size(g.op) == (d, d, d, d)
        @test scalartype(typeof(g)) == T
        # rank-4 张量构造与 Pair 矩阵构造给出同一算符
        G4 = permutedims(reshape(G, d, d, d, d), (2, 1, 4, 3))
        g4 = Gate((2, 3), G4)
        @test g.op == g4.op
        # 矩阵构造 Gate(positions, op, phydims)（FMA 新增形式）
        gm = Gate((2, 3), G, d)
        @test positions(gm) == (2, 3)
        @test gm.op ≈ g.op
        # isunitary：构造器按 atol 自动检测（QR 因子酉、随机矩阵不酉）
        @test isunitary(g4)
        @test isunitary(gm)
        @test !isunitary(Gate((1, 2), randn(T, d, d, d, d)))
        @test !isunitary(Gate((1, 2), randn(T, d * d, d * d), d))
        # 单比特门：rank-2 算子张量 / d×d 矩阵构造
        U = qr(randn(T, d, d)).Q |> Matrix
        g1 = Gate((2,), U)
        @test positions(g1) == (2,)
        @test size(g1.op) == (d, d)
        @test isunitary(g1)
        @test Gate((2,), U, d).op ≈ g1.op
        # 秩错误
        @test_throws ArgumentError Gate((1, 2), randn(T, d, d, d))
        # positions 乱序：Gate 内部排序（support 升序 + 算符腿跟随置换，
        # 不再抛错）——(3, 2) 输入与 (2, 3) 输入给出同一门
        g32 = Gate((3, 2), permutedims(G4, (2, 1, 4, 3)))
        @test positions(g32) == (2, 3)
        @test g32.op ≈ G4
        # positions 重复拒绝
        @test_throws ArgumentError Gate((2, 2), G4)
        # adjoint：共轭转置且支持不变
        ga = adjoint(g)
        @test positions(ga) == (2, 3)
        @test adjoint(ga).op ≈ g.op
        # shift
        @test positions(shift(g, 1)) == (3, 4)
    end

    @testset "swap! 内容交换（SWAP 门）" begin
        ψ1 = copy(ψ)
        swap!(ψ1, 2)                       # 交换 sites 2, 3（腿交叉 SWAP 窗口）
        # AR 侧规范严格：AR[3]（SVD 右因子）行正交、AC = C·AR
        @tensor XR[a, b] := ψ1.AR[3][a, p, c] * conj(ψ1.AR[3][b, p, c])
        @test norm(XR - I(size(XR, 1))) ≈ 0 atol = 1e-11
        @tensor ACc[x, p, y] := ψ1.C[1][x, a] * ψ1.AR[2][a, p, y]
        @test norm(ACc - ψ1.AC[2]) ≈ 0 atol = 1e-10
        # SWAP² = I：截断压回键维后态还原（截断/接缝误差级偏差）
        swap!(ψ1, 2)
        gaugefix!(ψ1, parent(ψ1.AR))
        @test fidelity(ψ, ψ1) > 0.95
    end

    @testset "Gate 作用（幺正门）" begin
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = pair_gate(Pair(2, 3), G)
        # 无截断：门改变态
        ψ1 = copy(ψ)
        ψ1 = apply!(g, ψ1)
        @test fidelity(ψ, ψ1) < 1 - 1e-3
        # g† 回到原态（无截断时 Hastings 更新保态精确）
        ψ1 = apply!(adjoint(g), ψ1)
        @test ismixedcanonical(ψ1)
        @test fidelity(ψ, ψ1) ≈ 1 atol = 1e-9
        # 非相邻门：swap 移动 + 门 + swap 移回，g† 后精确还原
        ψ2 = copy(ψ)
        gn = pair_gate(Pair(1, 4), G)
        ψ2 = apply!(gn, ψ2)
        ψ2 = apply!(adjoint(gn), ψ2)
        @test ismixedcanonical(ψ2)
        @test fidelity(ψ, ψ2) ≈ 1 atol = 1e-9
        # 截断：gate bond 维数受 trunc 控制
        ψ3 = copy(ψ)
        ψ3 = apply!(g, ψ3; trunc = truncdim(chi))
        @test size(ψ3.C[2], 1) <= chi
    end

    @testset "单比特门作用（AbstractGate{1}）" begin
        U = qr(randn(T, d, d)).Q |> Matrix
        gu = Gate((2,), U)
        @test isunitary(gu)
        # 规范保持：C 不变，AL/AR 由 regauge! 重算
        ψ1 = copy(ψ)
        Cprev = copy(ψ1.C[1]); Cnext = copy(ψ1.C[2])
        apply!(gu, ψ1)
        @test ismixedcanonical(ψ1)
        @test maximum(mixedcanonical_errors(ψ1)) < 1e-12
        @test ψ1.C[1] ≈ Cprev atol = 1e-12
        @test ψ1.C[2] ≈ Cnext atol = 1e-12
        # 与两站 kron(U, I)（site 2 为慢指标）作用等价
        ψ2 = copy(ψ)
        apply!(pair_gate(Pair(2, 3), kron(U, Matrix{T}(I, d, d))), ψ2)
        @test fidelity(ψ1, ψ2) ≈ 1 atol = 1e-10
        # g·g† 往返无损
        apply!(adjoint(gu), ψ1)
        @test fidelity(ψ, ψ1) ≈ 1 atol = 1e-12
        @test ismixedcanonical(ψ1)
        # wrap 站点（i = 1 与 i = L）
        for i in (1, L)
            ψw = apply!(Gate((i,), U), copy(ψ))
            @test ismixedcanonical(ψw)
            apply!(Gate((i,), adjoint(U)), ψw)
            @test fidelity(ψ, ψw) ≈ 1 atol = 1e-12
        end
        # 非幺正单比特门：缺省 canonicalize = !isunitary(g)
        gn = Gate((3,), randn(T, d, d))
        @test !isunitary(gn)
        ψn = apply!(gn, copy(ψ))
        @test ismixedcanonical(ψn)
        # 显式 canonicalize = false 可跳过重正则化
        ψn2 = apply!(gn, copy(ψ); canonicalize = false)
        @test ψn2 isa typeof(ψ)
        # support bounds（1 <= i <= L）
        @test_throws BoundsError apply!(Gate((0,), U), copy(ψ))
        @test_throws BoundsError apply!(Gate((L + 1,), U), copy(ψ))
    end

    @testset "非幺正门（canonicalize = !isunitary(g)）" begin
        # FMA 统一门接口语义：Gate 不区分幺正/非幺正类型，非幺正门破坏
        # Hastings 更新的严格正交性（gauge 性质）——缺省 canonicalize =
        # !isunitary(g) 自动重正则化；显式 canonicalize = false 可跳过。
        gg = pair_gate(Pair(2, 3), randn(T, d * d, d * d))
        @test !isunitary(gg)
        ψ1 = copy(ψ)
        ψ1 = apply!(gg, ψ1)                 # 缺省：非幺正门自动 canonicalize
        @test ismixedcanonical(ψ1)
        ψ1c = copy(ψ)
        ψ1c = apply!(gg, ψ1c; canonicalize = true)
        @test ismixedcanonical(ψ1c)
        @test ψ1.AC[2] ≈ ψ1c.AC[2] atol = 1e-10
        # wrap 键上的非相邻非幺正门（gather swaps + 门 + canonicalize + 回移）
        gn = pair_gate(Pair(1, 4), randn(T, d * d, d * d))
        ψ2 = apply!(gn, copy(ψ))
        @test ismixedcanonical(ψ2)
    end

    @testset "support bounds（1 <= i <= L，j - i < L）" begin
        @test_throws BoundsError apply!(pair_gate(Pair(0, 1), randn(T, d * d, d * d)), copy(ψ))
        @test_throws BoundsError apply!(pair_gate(Pair(5, 6), randn(T, d * d, d * d)), copy(ψ))
        # j - i < L：在 L = 4 的链上 (1, 5) 会绕过整圈
        @test_throws BoundsError apply!(pair_gate(Pair(1, 5), randn(T, d * d, d * d)), copy(ψ))
        # 允许的 wrap 键 (L, L+1) = (4, 5)
        gw = pair_gate(Pair(4, 5), Matrix{T}(I, d * d, d * d))
        @test apply!(gw, copy(ψ)) isa typeof(ψ)
    end

    @testset "Hastings 无损性随 trunc err 收敛" begin
        # 确认点 1：Hastings 技巧在 trunc err → 0 时无损；
        # g·g† 的保真度随截断键维放松单调趋于 1
        G = qr(randn(ComplexF64, d * d, d * d)).Q |> Matrix
        g = pair_gate(Pair(2, 3), G)
        prev = 0.0
        for D in (2, 3, 4, 5, 6)
            ψD = copy(ψ)
            apply!(g, ψD; trunc = truncdim(D))
            apply!(adjoint(g), ψD; trunc = truncdim(D))
            f = fidelity(ψ, ψD)
            @test f >= prev - 1e-12          # 单调不降
            D >= 3 && @test f > 0.9          # D ≥ 3 时接近无损
            prev = f
        end
        ψD = copy(ψ)
        apply!(g, ψD)                        # NoTruncation
        apply!(adjoint(g), ψD)
        @test fidelity(ψ, ψD) ≈ 1 atol = 1e-9     # 无截断极限：精确无损
    end

    @testset "大截断下正则形式保持" begin
        # 确认点 2：激进截断（D=2）后，正则形式仍严格保持：
        # AR[3]（SVD 右因子）严格行正交；C[2] 对角谱；
        # 恒等式网络 AC = AL·C（窗口键）严格；接缝 AR[2] 严格正交
        # （regauge! 规范转换语义）；接缝键的 AC = C·AR 机器精度保持。
        gI = pair_gate(Pair(2, 3), Matrix{T}(I, d * d, d * d))
        ψT = copy(ψ)
        ψT = apply!(gI, ψT; trunc = truncdim(2))
        @tensor XR[a, b] := ψT.AR[3][a, p, c] * conj(ψT.AR[3][b, p, c])
        @test norm(XR - I(size(XR, 1))) ≈ 0 atol = 1e-11
        @tensor XL[a, a0] := conj(ψT.AL[2][x, p, a]) * ψT.AL[2][x, p, a0]
        @test norm(XL - I(size(XL, 1))) ≈ 0 atol = 1e-11
        @test norm(ψT.C[2] - Diagonal(diag(ψT.C[2]))) ≈ 0 atol = 1e-12
        @tensor ACl[x, p, y] := ψT.AL[2][x, p, a] * ψT.C[2][a, y]
        @test norm(ACl - ψT.AC[2]) ≈ 0 atol = 1e-10
        @tensor XRs[a, b] := ψT.AR[2][a, p, c] * conj(ψT.AR[2][b, p, c])
        @test norm(XRs - I(size(XRs, 1))) ≈ 0 atol = 1e-11   # 接缝严格正交
    end

    @testset "一般混合规范态直接使用（无谱对齐初始化）" begin
        # 确认点 3：apply!/swap! 在未做任何谱对齐初始化的一般混合规范态上：
        # 正则形式机器精度保持（接缝严格正交、AC = C·AR 严格），门序列
        # （含 wrap 键 (4,5) 与非相邻 (1,4)）正则误差不增长，恒等门 /
        # swap² / g·g† 无损。
        mc(ψx) = maximum(mixedcanonical_errors(ψx))
        @test mc(ψ) < 1e-12

        # 单次 swap：正则形式机器精度保持；swap² 无损
        ψ1 = copy(ψ)
        swap!(ψ1, 2)
        @test mc(ψ1) < 1e-12
        @test ismixedcanonical(ψ1)
        swap!(ψ1, 2)
        @test fidelity(ψ, ψ1) ≈ 1 atol = 1e-12
        @test mc(ψ1) < 1e-12
        @test ismixedcanonical(ψ1)

        # 恒等门无损
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = pair_gate(Pair(2, 3), G)
        gI = pair_gate(Pair(2, 3), Matrix{T}(I, d * d, d * d))
        ψi = apply!(gI, copy(ψ))
        @test fidelity(ψ, ψi) ≈ 1 atol = 1e-12
        @test mc(ψi) < 1e-12
        @test ismixedcanonical(ψi)

        # 随机门 + g·g† 无损
        ψg = apply!(g, copy(ψ))
        @test mc(ψg) < 1e-12
        @test ismixedcanonical(ψg)
        apply!(adjoint(g), ψg)
        @test fidelity(ψ, ψg) ≈ 1 atol = 1e-12
        @test mc(ψg) < 1e-12
        @test ismixedcanonical(ψg)

        # 非相邻门 g·g† 无损
        ψn = copy(ψ)
        gn = pair_gate(Pair(1, 4), G)
        apply!(gn, ψn)
        @test ismixedcanonical(ψn)
        apply!(adjoint(gn), ψn)
        @test fidelity(ψ, ψn) ≈ 1 atol = 1e-12
        @test mc(ψn) < 1e-12
        @test ismixedcanonical(ψn)

        # 门序列（相邻 / 非相邻 / wrap 键）：正则误差不增长，范数守恒
        ψseq = copy(ψ)
        for p in ((2, 3), (3, 4), (1, 2), (4, 5), (2, 3), (1, 4), (3, 4), (4, 5))
            apply!(pair_gate(Pair(p...), qr(randn(T, d * d, d * d)).Q |> Matrix), ψseq)
            @test mc(ψseq) < 1e-12
            @test ismixedcanonical(ψseq)
        end
        @test norm(ψseq) ≈ 1 atol = 1e-12
        @test ismixedcanonical(ψseq)
    end
end
