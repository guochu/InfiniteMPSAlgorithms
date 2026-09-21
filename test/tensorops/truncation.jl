# =====================================================================
# tensorops：截断方案与张量分解（tsvd / leftorth / rightorth）
# =====================================================================

@testset "truncdim / tsvd 截断分解" begin
    T = ComplexF64
    Random.seed!(7)
    A = randn(T, 6, 4) .+ 1im * randn(T, 6, 4)
    U, s, V, err = tsvd(A; trunc = truncdim(3))
    @test size(U) == (6, 3) && length(s) == 3 && size(V) == (3, 4)
    # 保留的是最大的 3 个奇异值
    sfull = svdvals(A)
    @test s ≈ sfull[1:3]
    @test err ≈ norm(sfull[4:end])
    # 正交性
    @test U' * U ≈ Matrix{T}(I, 3, 3) atol = 1e-12
    @test V * V' ≈ Matrix{T}(I, 3, 3) atol = 1e-12
    # NoTruncation 完整分解
    U0, s0, V0, err0 = tsvd(A)
    @test err0 == 0 && length(s0) == 4
end

@testset "leftorth / rightorth（矩阵与张量分区）" begin
    T = ComplexF64
    Random.seed!(8)
    AC = randn(T, 6, 2, 5) .+ 1im * randn(T, 6, 2, 5)
    # (Dl·d, Dr) QR 分裂 → (AL, C)：AL 左正交
    AL, C = leftorth(AC, (1, 2), (3,))
    @tensor iso[a, c] := conj(AL[x, s, a]) * AL[x, s, c]
    @test iso ≈ Matrix{T}(I, size(iso)...) atol = 1e-12
    @tensor ac1[a, s, b] := AL[a, s, m] * C[m, b]
    @test ac1 ≈ AC atol = 1e-12
    # (Dl, d·Dr) LQ 分裂 → (C, AR)：AR 右正交
    C2, AR = rightorth(AC, (1,), (2, 3))
    @tensor isor[a, c] := AR[a, s, x] * conj(AR[c, s, x])
    @test isor ≈ Matrix{T}(I, size(isor)...) atol = 1e-12
    @tensor ac2[a, s, b] := C2[a, m] * AR[m, s, b]
    @test ac2 ≈ AC atol = 1e-12
end
