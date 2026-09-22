using Test
using LinearAlgebra
using Random
using TensorOperations
using InfiniteMPSAlgorithms
import InfiniteMPSAlgorithms: scalartype, asmps_view

include("testhelpers.jl")

@testset "tensorops" begin
    t = @elapsed include("tensorops/truncation.jl")
    println("tensorops: ", round(t; digits = 2), " s")
end

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
        include("algorithms/tebd.jl")
        include("algorithms/api.jl")
    end
    println("algorithms: ", round(t; digits = 2), " s")
end
