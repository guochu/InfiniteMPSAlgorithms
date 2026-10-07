# =====================================================================
# compress ↔ MPSKit.approximate 严格对齐测试
#
# MPSKit 0.13 的 approximate（VOMPS / IDMRG，src/algorithms/approximate/）
# 与本包 compress（经 compression_sweeps!）的逐步对齐：
# - 相同的随机目标态与随机初态（同一组张量，两包各持独立副本）；
# - 完全相同的算法参数（tol / maxiter）；
# - VOMPS / IDMRG：逐迭代对比（maxiter = k、tol = 0 强制两包都恰好跑 k 轮）；
# - 收敛迭代数一致（info.niter ↔ MPSKit 的最小收敛轮数扫描）；
# - 收敛终态在数值精度下一致：包内 `fidelity`（转移矩阵主导本征值的
#   infinite 射线语义，主判）+ dense 周期 trace 表示的射线残差（periodic
#   repr，辅助证据）。
# MPSKit 的 approximate 无 MPO 目标版本 ⇒ MPO 压缩按 `vectorize` 转成 MPS
# 视图（恒等 MPO 通道）对比。
#
# 对齐语义备注（MPSKit 源码，2025 主线）：
# - VOMPS：Jacobi 式 localupdate（AC/C 投影 + regauge!，全部站点对同一批
#   环境）→ gaugefix!(:R) → 环境重解（热启动）→ calc_galerkin ≤ tol（扫掠
#   之后判定，至少跑一轮）；
# - IDMRG：Gauss–Seidel 顺序双扫（投影 → normalize! → left_orth!/right_orth!
#   → transfer_leftenv!/transfer_rightenv! 即时推进环境）→ normalize!(envs)
#   → ϵ = ‖C₀_new − C₀_old‖。
# =====================================================================

Random.seed!(20260928)
T = ComplexF64
N = 2                    # 单胞长度
d = 2                    # 物理维
Dψ = 6                   # 目标态键维
D0 = 3                   # 压缩键维

ψ = randomimps(T, fill(d, N); D = Dψ)
x0 = randomimps(T, fill(d, N); D = D0)

ψt_mk = to_mpskit(ψ)
# 恒等 MPO（键维 1）：MPSKit 的 VOMPS/IDMRG 要求显式 (O, ψ) 元组
Imk = MPSKit.InfiniteMPO([mkmpotensor(identityimpo(T, fill(d, N))[ℓ]) for ℓ in 1:N])

"两包收敛态的射线残差（dense 周期 trace 表示，规范与尺度不变）——periodic
repr 把 Infinite MPS 当成有限环 MPS 处理，概念上不完备，只作辅助证据。"
function _compress_ray_residual(ya::CanonicalIMPS, yb::CanonicalIMPS)
    a = vec(_dense_mps_repr(ya))
    b = vec(_dense_mps_repr(yb))
    ls = dot(b, a) / dot(b, b)
    return norm(a .- ls .* b) / norm(a)
end

"两包收敛态的包内 fidelity（转移矩阵主导本征值的 infinite 射线语义主判；
地板为 eigsolve 相对误差 ~1e-12 量级）。"
_compress_fidelity(ya, yb) = fidelity(DenseIMPS(ya), DenseIMPS(yb))

"MPSKit `approximate` 的最小收敛轮数（单调谓词 ϵ(k) ≤ tol 的指数括号 + 二分；
MPSKit 在第 k 轮扫掠后 ϵ ≤ tol 即提前返回，故 ϵ(k) 随 k 单调下降且 k ≥ iter*
时 ϵ 恒为 ϵ(iter*)）。返回 (iter*, ϕ)。"
function _mpskit_converged_iter(algmk, tol)
    evalk = k -> MPSKit.approximate(mkinfinitemps(x0), (Imk, ψt_mk),
                                    algmk(; tol = tol, maxiter = k, verbosity = 0))
    ϕ, _, ϵ = evalk(1)
    ϵ ≤ tol && return 1, ϕ
    hi = 2
    while true
        ϕ, _, ϵ = evalk(hi)
        ϵ ≤ tol && break
        hi *= 2
        hi ≤ 2^12 || error("MPSKit approximate 不收敛")
    end
    lo = hi ÷ 2                       # ϵ(lo) > tol（倍增路径上已验证）
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        ϕm, _, ϵm = evalk(mid)
        if ϵm ≤ tol
            hi = mid
            ϕ = ϕm
        else
            lo = mid
        end
    end
    return hi, ϕ
end

"本包 `compression_sweeps!`（VOMPS）恰好跑 k 轮的压缩结果。"
function _ours_vomps(ψ, x0, k)
    alg = VOMPS(D = D0, tol = 0.0, maxiter = k)
    envs = OverlapCache(copy(x0), ψ, alg.alg_environments)
    InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
    return envs.bra
end

"本包 `compression_sweeps!`（IDMRG）恰好跑 k 轮的压缩结果。"
function _ours_idmrg(ψ, x0, k)
    alg = IDMRG(D = D0, tol = 0.0, maxiter = k)
    envs = OverlapCache(copy(x0), ψ, alg.alg_environments)
    InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
    return envs.bra
end

"MPSKit `approximate` 恰好跑 k 轮（tol = 0）的压缩结果（只取态）。"
_mpskit_approx(algmk, k) = MPSKit.approximate(
    mkinfinitemps(x0), (Imk, ψt_mk),
    algmk(; tol = 0.0, maxiter = k, verbosity = 0))[1]

@testset "compress VOMPS ≡ MPSKit approximate VOMPS（逐迭代对齐）" begin
    for k in 1:5
        y = _ours_vomps(ψ, x0, k)
        ϕ = _mpskit_approx(MPSKit.VOMPS, k)
        @test _compress_ray_residual(y, from_mpskit(ϕ)) < 1e-8
        @test _compress_fidelity(y, from_mpskit(ϕ)) > 1 - 1.0e-8
    end
end

@testset "compress IDMRG ≡ MPSKit approximate IDMRG（逐迭代对齐）" begin
    for k in 1:5
        y = _ours_idmrg(ψ, x0, k)
        ϕ = _mpskit_approx(MPSKit.IDMRG, k)
        @test _compress_ray_residual(y, from_mpskit(ϕ)) < 1e-8
        @test _compress_fidelity(y, from_mpskit(ϕ)) > 1 - 1.0e-8
    end
end

@testset "compress VOMPS 收敛迭代数与终态 ≡ MPSKit" begin
    tol = 1.0e-10
    alg = VOMPS(D = D0, tol = tol, maxiter = 500)
    envs = OverlapCache(copy(x0), ψ, alg.alg_environments)
    _, info = InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
    y = envs.bra
    mk_iter, ϕ = _mpskit_converged_iter(MPSKit.VOMPS, tol)
    @test info.niter == mk_iter
    @test _compress_ray_residual(y, from_mpskit(ϕ)) < 1e-8
    @test _compress_fidelity(y, from_mpskit(ϕ)) > 1 - 1.0e-8
end

@testset "compress IDMRG 收敛迭代数与终态 ≡ MPSKit" begin
    tol = 1.0e-10
    alg = IDMRG(D = D0, tol = tol, maxiter = 500)
    envs = OverlapCache(copy(x0), ψ, alg.alg_environments)
    _, info = InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
    y = envs.bra
    mk_iter, ϕ = _mpskit_converged_iter(MPSKit.IDMRG, tol)
    @test info.niter == mk_iter
    @test _compress_ray_residual(y, from_mpskit(ϕ)) < 1e-8
    @test _compress_fidelity(y, from_mpskit(ϕ)) > 1 - 1.0e-8
end

# ---- MPO 压缩（MPSKit 无 MPO 目标 approximate ⇒ 走 vectorize 的 MPS 视图）----

@testset "compress(MPO) ≡ MPSKit approximate（MPS 视图，VOMPS/IDMRG）" begin
    W = rand_denseimpo(T, fill(d, N); D = 4)
    Wview = vectorize(W).As
    x0w = randomimps(T, fill(size(Wview[1], 2), N); D = D0)
    Wmk = MPSKit.InfiniteMPS([mkmpstensor(a) for a in Wview])
    # 恒等 MPO 的物理维必须与 MPS 视图的融合物理维 (u·d) 一致
    Imkw = MPSKit.InfiniteMPO([mkmpotensor(identityimpo(T, fill(size(Wview[1], 2), N))[ℓ])
                               for ℓ in 1:N])
    for (alg, algmk) in ((VOMPS(D = D0, tol = 1.0e-12, maxiter = 300), MPSKit.VOMPS),
                         (IDMRG(D = D0, tol = 1.0e-12, maxiter = 300), MPSKit.IDMRG))
        ket = CanonicalIMPS(vectorize(W).As)
        envs = OverlapCache(x0w, ket, alg.alg_environments)
        if alg isa VOMPS
            InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
        else
            InfiniteMPSAlgorithms.compression_sweeps!(envs, alg)
        end
        y = devectorize(envs.bra)                                           # CanonicalIMPO
        ϕ = MPSKit.approximate(mkinfinitemps(x0w), (Imkw, Wmk),
                               algmk(; tol = 1.0e-12, maxiter = 300,
                                     verbosity = 0))[1]
        yview = vectorize(y)
        @test _compress_ray_residual(yview, from_mpskit(ϕ)) < 1e-8
        @test _compress_fidelity(yview, from_mpskit(ϕ)) > 1 - 1.0e-8
    end
end
