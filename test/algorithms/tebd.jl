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
# =====================================================================

@testset "TEBD gates" begin
    T = ComplexF64
    Random.seed!(2026)
    L, d, chi = 4, 2, 4
    ψ = randomimps(T, fill(d, L), chi)
    gaugefix!(ψ, parent(ψ.AR))
    @test ismixedcanonical(ψ)
    fid(ψ1, ψ2) = abs(dot(ψ1, ψ2)) / (norm(ψ1) * norm(ψ2))

    @testset "gate 类型" begin
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = UnitaryGate(Pair(2, 3), G)
        @test positions(g) == (2, 3)
        @test size(operator(g)) == (d, d, d, d)
        @test scalartype(typeof(g)) == T
        # rank-4 张量构造与 Pair 矩阵构造给出同一算符
        G4 = permutedims(reshape(G, d, d, d, d), (2, 1, 4, 3))
        g4 = UnitaryGate((2, 3), G4)
        @test operator(g) == operator(g4)
        # 非 unitary 拒绝
        @test_throws ArgumentError UnitaryGate((1, 2), randn(T, d, d, d, d))
        # 秩错误
        @test_throws ArgumentError UnitaryGate((1, 2), randn(T, d, d, d))
        # positions 乱序拒绝
        @test_throws ArgumentError UnitaryGate((3, 2), G4)
        # GeneralGate 不校验 unitarity
        @test GeneralGate((1, 2), randn(T, d, d, d, d)) isa GeneralGate
        # adjoint：共轭转置且支持不变
        ga = adjoint(g)
        @test positions(ga) == (2, 3)
        @test operator(adjoint(ga)) ≈ operator(g)
        # shift
        @test positions(shift(g, 1)) == (3, 4)
    end

    @testset "swap! 无损重规范" begin
        # 恒等门窗口自身行正交 × 行正交：接缝精确，规范完全保持
        ψ1 = copy(ψ)
        swap!(ψ1, 2)
        @test ismixedcanonical(ψ1)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-11
        swap!(ψ1, 2)
        @test ismixedcanonical(ψ1)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-11
    end

    @testset "UnitaryGate 作用" begin
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = UnitaryGate(Pair(2, 3), G)
        # 无截断：门改变态
        ψ1 = copy(ψ)
        ψ1 = apply!(g, ψ1)
        @test fid(ψ, ψ1) < 1 - 1e-3
        # g† 回到原态（无截断时 Hastings 更新保态精确）
        ψ1 = apply!(adjoint(g), ψ1)
        @test ismixedcanonical(ψ1)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-9
        # 非相邻门：swap 移动 + 门 + swap 移回，g† 后精确还原
        ψ2 = copy(ψ)
        gn = UnitaryGate(Pair(1, 4), G)
        ψ2 = apply!(gn, ψ2)
        ψ2 = apply!(adjoint(gn), ψ2)
        @test ismixedcanonical(ψ2)
        @test fid(ψ, ψ2) ≈ 1 atol = 1e-9
        # 截断：gate bond 维数受 trunc 控制
        ψ3 = copy(ψ)
        ψ3 = apply!(g, ψ3; trunc = truncdim(chi))
        @test size(ψ3.C[2], 1) <= chi
    end

    @testset "GeneralGate" begin
        # 非 unitary 门作用后 apply! 内部 gaugefix 恢复全规范
        gg = GeneralGate((2, 3), randn(T, d, d, d, d))
        ψ1 = copy(ψ)
        ψ1 = apply!(gg, ψ1)
        @test ismixedcanonical(ψ1)
    end

    @testset "Hastings 无损性随 trunc err 收敛" begin
        # 确认点 1：Hastings 技巧在 trunc err → 0 时无损；
        # g·g† 的保真度随截断键维放松单调趋于 1
        G = qr(randn(ComplexF64, d * d, d * d)).Q |> Matrix
        g = UnitaryGate(Pair(2, 3), G)
        prev = 0.0
        for D in (2, 3, 4, 5, 6)
            ψD = copy(ψ)
            apply!(g, ψD; trunc = truncdim(D))
            apply!(adjoint(g), ψD; trunc = truncdim(D))
            f = fid(ψ, ψD)
            @test f >= prev - 1e-12          # 单调不降
            D >= 3 && @test f > 0.9          # D ≥ 3 时接近无损
            prev = f
        end
        ψD = copy(ψ)
        apply!(g, ψD)                        # NoTruncation
        apply!(adjoint(g), ψD)
        @test fid(ψ, ψD) ≈ 1 atol = 1e-9     # 无截断极限：精确无损
    end

    @testset "大截断下正则形式保持" begin
        # 确认点 2：激进截断（D=2）后，正则形式仍严格保持：
        # AR[3]（SVD 右因子）严格行正交；C[2] 对角谱；
        # 恒等式网络 AC = AL·C = C·AR 严格；接缝 AR[2] 的正交性
        # 带 O(trunc err) 误差（Hastings 语义），恒等式仍精确。
        gI = UnitaryGate(Pair(2, 3), Matrix{T}(I, d * d, d * d))
        ψT = copy(ψ)
        ψT = apply!(gI, ψT; trunc = truncdim(2))
        @tensor XR[a, b] := ψT.AR[3][a, p, c] * conj(ψT.AR[3][b, p, c])
        @test norm(XR - I(size(XR, 1))) ≈ 0 atol = 1e-11
        @tensor XL[a, a0] := conj(ψT.AL[2][x, p, a]) * ψT.AL[2][x, p, a0]
        @test norm(XL - I(size(XL, 1))) ≈ 0 atol = 1e-11
        @test norm(ψT.C[2] - Diagonal(diag(ψT.C[2]))) ≈ 0 atol = 1e-12
        @tensor ACl[x, p, y] := ψT.AL[2][x, p, a] * ψT.C[2][a, y]
        @test norm(ACl - ψT.AC[2]) ≈ 0 atol = 1e-10
        @tensor ACr[x, p, y] := ψT.C[1][x, a] * ψT.AR[2][a, p, y]
        @test norm(ACr - ψT.AC[2]) ≈ 0 atol = 1e-10
    end
end
