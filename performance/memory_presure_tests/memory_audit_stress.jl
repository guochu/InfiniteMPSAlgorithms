# 全通道内存压力审计（加强版）：输入键 32（乘积键 1024 >> 目标 D），
# D = 25/50/100 全截断——真实压缩压力（前版 D=100 因乘积键 64 < D 退化为
# 无损恒等问题、iters=1 未构成压力）。
# 组合：{compress, mult mpo·mps, mult mpo·mpo, hadamard} × {IDMRG1, IDMRG2, VOMPS}。
using Random, LinearAlgebra
using InfiniteMPSAlgorithms
include(joinpath(@__DIR__, "..", "..", "test", "testhelpers.jl"))
const IMA = InfiniteMPSAlgorithms

rss_now() = parse(Int, match(r"VmRSS:\s+(\d+)", read("/proc/self/status", String)).captures[1]) * 1024
hwm_now() = parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String)).captures[1]) * 1024
gb(x) = round(x / 1024^3, digits = 3)
mb(x) = round(x / 1024^2, digits = 1)

function run_case(name, f)
    GC.gc(); GC.gc()
    rss0, hwm0 = rss_now(), hwm_now()
    alloc0 = Base.gc_bytes()[1]
    t = @elapsed r = f()
    alloc1 = Base.gc_bytes()[1]
    GC.gc(); GC.gc()
    rss1, hwm1 = rss_now(), hwm_now()
    y, envs, info = r
    println(name, ":  t=", round(t, digits = 1), "s",
            "  alloc=", gb(alloc1 - alloc0), "GiB",
            "  RSS +", mb(rss1 - rss0), "MB (HWM +", mb(hwm1 - hwm0), "MB)",
            "  驻留=", mb(Base.summarysize(y) + Base.summarysize(envs)), "MB",
            "  iters=", info.niter, info.converged ? " conv" : " unc")
    flush(stdout)
    return y
end

T = ComplexF64
Din = 32                                 # 输入键维（乘积键 1024 >> 目标 D）
alg(D) = (IDMRG(D = D, tol = 1e-6, maxiter = 30),
          IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30),
          VOMPS(D = D, tol = 1e-6, maxiter = 30))
anames = ("IDMRG1", "IDMRG2", "VOMPS")

println("======== compress（ψ 键 32 → 目标 D，全截断）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        ψ = randomimps(T, [2, 2]; D = Din)
        run_case("  compress $an D=$D", () -> compress(ψ, alg(D)[i]))
    end
end

println("======== mult mpo·mps（W 键 32 × ψ 键 32 → D，乘积键 1024 全截断）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        W = rand_denseimpo(T, [2, 2]; D = Din)
        ψ = randomimps(T, [2, 2]; D = Din)
        run_case("  mult·mps $an D=$D", () -> mult(W, ψ, alg(D)[i]))
    end
end

println("======== mult mpo·mpo（W1 键 32 × W2 键 32 → D）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        W1 = rand_denseimpo(T, [2, 2]; D = Din)
        W2 = rand_denseimpo(T, [2, 2]; D = Din)
        run_case("  mult·mpo $an D=$D", () -> mult(W1, W2, alg(D)[i]))
    end
end

println("======== hadamard（ψ1 键 32 ⊙ ψ2 键 32 → D）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        ψ1 = randomimps(T, [2, 2]; D = Din)
        ψ2 = randomimps(T, [2, 2]; D = Din)
        run_case("  hadamard $an D=$D", () -> hadamard(ψ1, ψ2, alg(D)[i]))
    end
end

println("======== 完成 ========")
