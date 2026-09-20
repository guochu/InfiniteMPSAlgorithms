using Test
using LinearAlgebra
using Random
using TensorOperations
using InfiniteMPSAlgorithms
import InfiniteMPSAlgorithms: scalartype, asmps_view

include("testhelpers.jl")

@testset "InfiniteMPSAlgorithms" begin
    include("mps.jl")
    include("envs.jl")
    include("vumps.jl")
    include("idmrg.jl")
    include("twosite.jl")
    include("mpo.jl")
    include("w1w2.jl")
    include("tdvp.jl")
    include("observables.jl")
    include("arithmetics.jl")
    include("api.jl")
end
