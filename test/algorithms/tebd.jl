# =====================================================================
# TEBD 量子门（Hastings 更新的混合规范实现）测试
#
# 覆盖：gate 类型构造与约定 / swap! 无损重规范 / UnitaryGate 作用
# （无截断保态、截断行为、非相邻门移动）/ GeneralGate 重规范。
# 无限态 fidelity 用 AL 转移矩阵主特征值（dot）度量。
# =====================================================================

@testset "TEBD gates" begin
    T = ComplexF64
    Random.seed!(2026)
    L, d, chi = 4, 2, 4
    ψ = randomimps(T, fill(d, L), chi)
    gaugefix!(ψ, parent(ψ.AR))
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
        ψ1 = copy(ψ)
        swap!(ψ1, 2)
        @test ismixedcanonical(ψ1)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-11
        swap!(ψ1, 2)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-11
        # 非相邻内容交换（Hastings SWAP 序列），两次回到原态
        ψ2 = copy(ψ)
        swap!(ψ2, 1, 4)
        @test ismixedcanonical(ψ2)
        swap!(ψ2, 1, 4)
        gaugefix!(ψ2, parent(ψ2.AR))
        @test fid(ψ, ψ2) ≈ 1 atol = 1e-9
    end

    @testset "UnitaryGate 作用" begin
        G = qr(randn(T, d * d, d * d)).Q |> Matrix
        g = UnitaryGate(Pair(2, 3), G)
        # 无截断：门改变态
        ψ1 = copy(ψ)
        ψ1 = apply!(g, ψ1)
        @test fid(ψ, ψ1) < 1 - 1e-3
        # g† 回到原态（接缝正交性 O(门) 退化不改变物理态，gaugefix 后检查）
        ψ1 = apply!(adjoint(g), ψ1)
        gaugefix!(ψ1, parent(ψ1.AR))
        @test ismixedcanonical(ψ1)
        @test fid(ψ, ψ1) ≈ 1 atol = 1e-9
        # 非相邻门：swap 移动 + 门 + swap 移回，g† 后精确还原
        ψ2 = copy(ψ)
        gn = UnitaryGate(Pair(1, 4), G)
        ψ2 = apply!(gn, ψ2)
        ψ2 = apply!(adjoint(gn), ψ2)
        gaugefix!(ψ2, parent(ψ2.AR))
        @test ismixedcanonical(ψ2)
        @test fid(ψ, ψ2) ≈ 1 atol = 1e-9
        # 截断：gate bond 维数受 trunc 控制
        ψ3 = copy(ψ)
        ψ3 = apply!(g, ψ3; trunc = truncdim(chi))
        @test size(ψ3.C[2], 1) <= chi
    end

    @testset "GeneralGate" begin
        # 非 unitary 门作用后 apply! 内部重规范
        gg = GeneralGate((2, 3), randn(T, d, d, d, d))
        ψ1 = copy(ψ)
        ψ1 = apply!(gg, ψ1)
        @test ismixedcanonical(ψ1)
    end
end
