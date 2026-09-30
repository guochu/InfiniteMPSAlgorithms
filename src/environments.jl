# ---------------- environment machinery (mirroring MPSKit src/environments/infinite_envs.jl) ----------------
#
# Concrete cache types are defined together with their algorithms (the
# constructors in the algorithm files directly produce the caches):
# - `DMRGCache`: Hamiltonian channel ⟨ψ|H|ψ⟩ (ground-state VUMPS / IDMRG / TDVP
#   and energy evaluation), see algorithms/groundstates/envs.jl + idmrg.jl;
# - `MultCache`: MPO-application channel ⟨bra|W|ket⟩ (iterative MPO
#   multiplication mult), see algorithms/arithmetics/mult.jl;
# - `OverlapCache`: pure overlap channel ⟨bra|ket⟩ (variational compression of
#   the algebra operations), see algorithms/arithmetics/compress.jl;
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

**`length` 契约**：每个子类必须定义 `Base.length(envs)` = 环境的单胞长度 =
输入态/算符单胞长度的最小公倍数（环境数组 `lefts`/`rights` 的元素个数、以及
该通道输出的量 `out` 的长度都等于这个数）；`leftenv`/`rightenv` 以它为周期
取模循环站点指标。
"""
abstract type Environments end

"leftenv(envs, ℓ): the left environment of site ℓ（以 `length(envs)` 为周期
取模循环）。"
leftenv(envs::Environments, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs))]
"rightenv(envs, ℓ): the right environment of site ℓ（以 `length(envs)` 为
周期取模循环）。"
rightenv(envs::Environments, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs))]
