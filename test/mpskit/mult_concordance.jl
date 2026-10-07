# =====================================================================
# mult / 严格乘法与 MPSKit 对比（固化自 debug/mult_concordance.jl）
#
# MPSKit 0.13 相关实现：
# - `Base.:*(mpo1, mpo2)`：朴素 fuse_mul_mpo 乘法 ↔ 本包 `W1 * W2`；
# - `Base.:*(mpo, mps)`：朴素 MPO·MPS 施加 ↔ 本包 `W * DenseIMPS(...)`；
# - `MPSKit.approximate(ψ₀, (O, ψ), VOMPS()/IDMRG())`：变分施加
#   ↔ 本包 mult(W, ψ, alg)（D 由 alg.D 提供）。
# 对比基准按语义分档，且一律"periodic-repr 辅助 + 包内 infinite 语义主判"：
# - 同尺度语义（两侧构造确定性同尺度，如严格乘法）：包内 `distance` +
#   `fidelity(::DenseIMPO, ::DenseIMPO)`（vectorize 转移矩阵语义，无相位/
#   尺度自由 / 射线语义）；
# - 射线语义（变分不动点输出带归一化自由、迭代中间态）：包内 `fidelity`；
# - `_dense_mpo_repr`（稠密周期 trace）把 Infinite MPO 当成有限环 MPO 处理，
#   概念上不完备，只作辅助证据保留（`mpo_ray_residual` / 直接差）。
# 阈值：Gram 消去 + eigsolve 容差给 distance 留 ~√ε 量级地板（实测 2e-6），
# fidelity 的地板是 eigsolve 相对误差 ~1e-12 量级。
# =====================================================================

Random.seed!(2024)
T = ComplexF64
N = 2                    # 单胞长度
d = 2                    # 物理维
Dψ = 4                   # ψ 键维
Dw = 3                   # W 键维

ψ = randomimps(T, fill(d, N); D = Dψ)
W1 = DenseIMPO(randomimpo(T, fill(d, N); D = Dw))
W2 = DenseIMPO(randomimpo(T, fill(d, N); D = 2))

@testset "* (DenseIMPO, DenseIMPO) ≡ MPSKit *(DenseIMPO, DenseIMPO)" begin
    P = W1 * W2
    PO = to_mpskit(W1) * to_mpskit(W2)
    # 两侧都是朴素精确乘法 ⇒ 同一算符同一尺度：包内 infinite 语义主判
    # （distance 的 eigsolve 地板实测 ~2e-6，阈值相应放宽）+ periodic-repr
    # 射线残差（~2e-16）辅助
    @test distance(P, from_mpskit(PO)) < 1.0e-5
    @test fidelity(P, from_mpskit(PO)) > 1 - 1.0e-10
    @test mpo_ray_residual(P, from_mpskit(PO)) < 1e-10
end

@testset "* (DenseIMPO, DenseIMPS) ≡ MPSKit *(DenseIMPO, InfiniteMPS)" begin
    Kψd = W1 * DenseIMPS(collect(ψ.AL))
    # 本包：朴素 fuse 后的原始 DenseIMPS 取稠密波形
    Kψ = vec(_dense_mps_repr(Kψd))
    # MPSKit：朴素施加结果（自身规范化的 AL）
    ϕ = to_mpskit(W1) * to_mpskit(ψ)
    Kmk = vec(_dense_mps_repr(from_mpskit(ϕ)))
    # 射线比较：Kψ 与 Kmk 平行（差一个构造归一标量）——periodic-repr 投影
    # （辅助）+ 包内 fidelity（infinite 射线语义主判）；distance 不适用
    # （两侧构造归一化不同，尺度自由是本场景的真实自由度）
    ls = dot(Kψ, Kmk) / dot(Kψ, Kψ)
    @test norm(Kmk .- ls .* Kψ) / norm(Kmk) < 1e-10
    @test fidelity(Kψd, DenseIMPS(from_mpskit(ϕ))) > 1 - 1.0e-10
end

@testset "mult VOMPS ≡ MPSKit approximate VOMPS" begin
    Dtar = 8                                    # 精确键维 Dw·Dψ = 12 → 变分压到 8
    y, _, _ = mult(W1, ψ, VOMPS(D = Dtar, maxiter = 200, tol = 1e-11))
    ϕk, _, δ = MPSKit.approximate(to_mpskit(randomimps(T, fill(d, N); D = Dtar)),
                                  (to_mpskit(W1), to_mpskit(ψ)),
                                  MPSKit.VOMPS(; tol = 1e-11, maxiter = 200))
    # 有损压缩（D < 精确键维）的变分最优点对实现细节敏感：实测本包解对精确
    # 施加态的保真度为 1（全局最优），MPSKit ≈ 0.992（次优盆地）。这里只要求
    # 两包解同物理（|dot| > 0.99）；本包内部 VOMPS ≡ IDMRG 不动点一致性
    # （|dot| = 1）在下一 testset 单独验证。
    yk = from_mpskit(ϕk)
    @test abs(dot(y, yk)) > 0.99
end

@testset "mult IDMRG ≡ MPSKit approximate IDMRG" begin
    Dtar = 8
    y, _, _ = mult(W1, ψ, IDMRG(D = Dtar, maxiter = 200, tol = 1e-11))
    ϕk, _, δ = MPSKit.approximate(to_mpskit(randomimps(T, fill(d, N); D = Dtar)),
                                  (to_mpskit(W1), to_mpskit(ψ)),
                                  MPSKit.IDMRG(; tol = 1e-11, maxiter = 200))
    yk = from_mpskit(ϕk)
    @test abs(dot(y, yk)) > 0.99
end

@testset "mult VOMPS ≡ IDMRG（本包双算法不动点一致）" begin
    yv, _, _ = mult(W1, ψ, VOMPS(D = 8, maxiter = 200, tol = 1e-11))
    yi, _, _ = mult(W1, ψ, IDMRG(D = 8, maxiter = 200, tol = 1e-11))
    @test abs(dot(yv, yi)) > 1 - 1e-6
end

@testset "mult mpo*mpo ≡ 严格乘法（VOMPS / IDMRG 不动点一致）" begin
    # 恒等 MPO 复合：mult(W2, I2) 精确恢复 W2 的射线（两算法）
    I2 = identityimpo(T, [2, 2])
    for alg in (VOMPS(D = 2, maxiter = 200, tol = 1e-11), IDMRG(D = 2, maxiter = 200, tol = 1e-11))
        y, _, _ = mult(W2, I2, alg)
        @test y isa CanonicalIMPO && ismixedcanonical(y)
        # 变分输出带归一化自由 ⇒ 射线语义：包内 fidelity 主判 + periodic-repr
        # 射线残差辅助
        @test fidelity(DenseIMPO(y), W2) > 1 - 1.0e-8
        @test mpo_ray_residual(DenseIMPO(y), W2) < 1e-6
    end
    # 变分复合 vs 朴素精确：满键维时 overlap = N 且稠密表示平行。变分输出是
    # 归一化代表元（转移谱 |λ(y)| = 1，而 |λ(W1*W2)| = 0.258 ⇒ 周期 trace
    # 尺度比 ‖D_y‖/‖D_e‖ = 1/√0.258 ≈ 1.97；补上全局标量 s 后
    # distance(s·y, W1*W2) ≈ 2e-6，即求解地板，无任何结构误差）——整体尺度
    # 是变分输出的真实自由度，这正是必须用射线语义而非严格距离的实例
    y, _, _ = mult(W1, W2, VOMPS(D = 6, maxiter = 300, tol = 1e-12))
    @test fidelity(DenseIMPO(y), W1 * W2) > 1 - 1.0e-8
    @test mpo_ray_residual(DenseIMPO(y), W1 * W2) < 1e-6
end
