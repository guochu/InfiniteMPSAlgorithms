using Test
using LinearAlgebra
using Random
using TensorOperations
using KrylovKit
using InfiniteMPSAlgorithms
import InfiniteMPSAlgorithms: scalartype, asmps_view

include("testhelpers.jl")

@testset "states" begin
    t = @elapsed begin
        include("states/mps.jl")
    end
    println("states: ", round(t; digits = 2), " s")
end

@testset "operators" begin
    t = @elapsed begin
        include("operators/mpo.jl")
        include("operators/longrangeop.jl")
        include("operators/w1w2.jl")
        include("operators/vectorize.jl")
    end
    println("operators: ", round(t; digits = 2), " s")
end

@testset "algorithms" begin
    t = @elapsed begin
        include("algorithms/envs.jl")
        include("algorithms/vumps.jl")
        include("algorithms/idmrg.jl")
        include("algorithms/twosite.jl")
        include("algorithms/tdvp.jl")
        include("algorithms/observables.jl")
        include("algorithms/arithmetics.jl")
        include("algorithms/compress.jl")
        include("algorithms/tebd.jl")
        include("algorithms/api.jl")
    end
    println("algorithms: ", round(t; digits = 2), " s")
end

# ---- MPSKit 对齐测试（concordance，固化自 debug/ 的逐步对齐行为）----
# 依赖 MPSKit / TensorKit（Project.toml 的 test target 依赖）；
# 与 MPSKit 的导出名冲突在 mpskit/testhelpers_mpskit.jl 中统一消歧。
@testset "MPSKit concordance" begin
    t = @elapsed begin
        include("mpskit/testhelpers_mpskit.jl")
        include("mpskit/lowlevel_concordance.jl")
        include("mpskit/mult_concordance.jl")
        include("mpskit/groundstate_concordance.jl")
        include("mpskit/envs_concordance.jl")
        include("mpskit/finite_t_concordance.jl")
    end
    println("mpskit: ", round(t; digits = 2), " s")
end
