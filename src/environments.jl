# ---------------- environment machinery (mirroring MPSKit src/environments/infinite_envs.jl) ----------------
#
# Concrete cache types are defined together with their algorithms (the
# constructors in the algorithm files directly produce the caches):
# - `DMRGCache`: Hamiltonian channel ⟨ψ|H|ψ⟩ (ground-state VUMPS / IDMRG / TDVP
#   and energy evaluation), see algorithms/groundstates/envs.jl + idmrg.jl;
# - `MultCache`: MPO-application channel ⟨bra|W|ket⟩ (iterative MPO
#   multiplication mult), see algorithms/arithmetics/mult.jl;
# - `OverlapCache`: pure overlap channel ⟨bra|ket⟩ (variational compression of
#   the algebra operations), see algorithms/arithmetics/overlap.jl;
# - `HadamardCache`: zip channel ⟨below|zip(ψ1, ψ2)⟩ (iterative Hadamard
#   product), see algorithms/arithmetics/hadamard.jl.
#
# This file only keeps the abstract supertype `Environments` and the
# environment accessors; the environment solve/increment machinery lives with
# its consumers (algorithms/groundstates/envs.jl and the arithmetics engines).

"""
    Environments

Abstract supertype of the left/right fixed-point environments of
⟨below|operator|ket⟩-type double-layer networks (mirroring MPSKit's
InfiniteEnvironments); concrete types: [`DMRGCache`](@ref) (Hamiltonian
channel), [`MultCache`](@ref) (MPO application), [`OverlapCache`](@ref)
(pure overlap) and [`HadamardCache`](@ref) (zip product).
"""
abstract type Environments end

"leftenv(envs, ℓ): the left environment of site ℓ."
leftenv(envs::Environments, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs.ket))]
"rightenv(envs, ℓ): the right environment of site ℓ."
rightenv(envs::Environments, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs.ket))]
