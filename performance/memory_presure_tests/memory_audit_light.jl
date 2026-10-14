# 全通道内存压力审计：{compress, mult mpo·mps, mult mpo·mpo, hadamard}
# × {IDMRG1, IDMRG2, VOMPS} × D = {25, 50, 100}，unit cell = 2，d = 2。
# 监控：耗时、GC 吞吐分配、RSS 增量 / 峰值增量（VmHWM）、终态+环境驻留。
# 暴涨判据：HWM 增量 >> 驻留（泄漏/外积巨块），或同规模下 alloc 异常量级。
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
Dw = 8                                   # 输入键维（算符/因子）
algs(D) = (IDMRG(D = D, tol = 1e-6, maxiter = 30),
           IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30),
           VOMPS(D = D, tol = 1e-6, maxiter = 30))
anames = ("IDMRG1", "IDMRG2", "VOMPS")

println("======== compress（ψ 键 8 → 目标 D）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        ψ = randomimps(T, [2, 2]; D = Dw)
        alg = algs(D)[i]
        run_case("  compress $an D=$D",
                 () -> (an == "IDMRG2" ? compress(ψ, IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30)) :
                        an == "IDMRG1" ? compress(ψ, IDMRG(D = D, tol = 1e-6, maxiter = 30)) :
                         compress(ψ, VOMPS(D = D, tol = 1e-6, maxiter = 30))))
    end
end

println("======== mult mpo·mps（W 键 8 × ψ 键 8）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        W = rand_denseimpo(T, [2, 2]; D = Dw)
        ψ = randomimps(T, [2, 2]; D = Dw)
        run_case("  mult·mps $an D=$D",
                 () -> (an == "IDMRG2" ? mult(W, ψ, IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30)) :
                        an == "IDMRG1" ? mult(W, ψ, IDMRG(D = D, tol = 1e-6, maxiter = 30)) :
                         mult(W, ψ, VOMPS(D = D, tol = 1e-6, maxiter = 30))))
    end
end

println("======== mult mpo·mpo（W1 键 8 × W2 键 8）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        W1 = rand_denseimpo(T, [2, 2]; D = Dw)
        W2 = rand_denseimpo(T, [2, 2]; D = Dw)
        run_case("  mult·mpo $an D=$D",
                 () -> (an == "IDMRG2" ? mult(W1, W2, IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30)) :
                        an == "IDMRG1" ? mult(W1, W2, IDMRG(D = D, tol = 1e-6, maxiter = 30)) :
                         mult(W1, W2, VOMPS(D = D, tol = 1e-6, maxiter = 30))))
    end
end

println("======== hadamard（ψ1 键 8 ⊙ ψ2 键 8）========")
for D in (25, 50, 100)
    for (i, an) in enumerate(anames)
        Random.seed!(42)
        ψ1 = randomimps(T, [2, 2]; D = Dw)
        ψ2 = randomimps(T, [2, 2]; D = Dw)
        run_case("  hadamard $an D=$D",
                 () -> (an == "IDMRG2" ? hadamard(ψ1, ψ2, IDMRG2(trunc = truncdim(D), tol = 1e-6, maxiter = 30)) :
                        an == "IDMRG1" ? hadamard(ψ1, ψ2, IDMRG(D = D, tol = 1e-6, maxiter = 30)) :
                         hadamard(ψ1, ψ2, VOMPS(D = D, tol = 1e-6, maxiter = 30))))
    end
end

println("======== 完成 ========")
