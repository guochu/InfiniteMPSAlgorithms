# InfiniteMPSAlgorithms.jl

Symmetry-free infinite matrix-product-state (MPS) / matrix-product-operator
(MPO) tensor-network algorithms: a clean, dense implementation of the core
[MPSKit](https://github.com/QuantumKitHub/MPSKit.jl) algorithms.

Site tensors are plain `Array`s and all contractions are written with
`TensorOperations` (`@tensor`), so the package is easy to read, hack, and
extend.

## Features

- **Data structures** — `CanonicalIMPS` / `CanonicalIMPO` in the mixed
  canonical form (`AL` / `AR` / `C` / `AC` families with periodic indexing),
  mirroring MPSKit's `InfiniteMPS` layout, plus `DenseIMPO` and the
  Schur-structured `SparseIMPO`.
- **Ground state** — single-site variational uniform MPS (`VUMPS`) and
  infinite DMRG (`IDMRG`) with Galerkin-residual convergence criteria,
  dynamic tolerances, and warm-started environments.
- **Time evolution** — single-site TDVP (`timestep` / `time_evolve`),
  two-site TEBD gates with the Hastings update (`apply!` / `swap!`), and
  W^I / W^II time-evolution MPOs (`make_time_mpo`).
- **Iterative MPO algebra** — `mult` (MPO·MPS application and MPO·MPO
  composition), `compress` (bond-dimension reduction), and `hadamard`
  (elementwise product), all with *compute-on-the-fly* variational
  compression: the naive target family is never materialized and intermediate
  memory stays at the single-site level.
- **Observables** — MPO and local expectation values, two-point correlators,
  entanglement entropies and spectra.
- **Models** — transverse-field Ising, Heisenberg XXZ, Fermi-Hubbard.

## Installation

```julia
julia> using Pkg
julia> Pkg.add(path = "path/to/InfiniteMPSAlgorithms")
```

The package depends on `KrylovKit`, `MatrixAlgebraKit`, `TensorOperations`,
`LinearAlgebra`, `Random`, and `Printf`.

## Quick start

### Ground state (VUMPS / IDMRG)

```julia
using InfiniteMPSAlgorithms

H = tfim_hamiltonian()                     # transverse-field Ising MPOHamiltonian

ψ, envs, ϵ = find_groundstate(H, VUMPS(D = 16))   # random ψ₀ with bond dimension 16
e = real(expectationvalue(ψ, H) / length(ψ))      # energy density
```

An explicit initial state works too (`alg.D` is ignored in that case):
`find_groundstate(ψ, H, VUMPS(D = 16))`.

Exact reference for J = h = 1: `e₀ = -4/π ≈ -1.2732395`.

### Time evolution (TDVP)

```julia
dt = 0.1
ψt, envs, history = time_evolve(ψ, H, 0.0:dt:1.0, TDVP(); verbosity = 1)
```

### Time-evolution MPOs (W^I / W^II)

```julia
W = make_time_mpo(H, dt, WII())            # exp(-i·H·dt) as an InfiniteMPO
ψw, fidelity = mult(W, ψ)                  # apply it (exact, compute-on-the-fly)
```

### Iterative MPO algebra

Every driver function takes its algorithm object as a **positional argument**
(see [Algorithms](@ref)), mirroring the MPSKit API. The target bond dimension
is carried by the algorithm object (`alg.D`); the in-place twins take it from
the provided initial guess `out` instead:

```julia
ψ′, ov = mult(W, ψ, VOMPS(D = 32))         # apply an MPO, compress to bond 32
W2,  ov = mult(W1, W2, IDMRG(D = 12))      # compose two MPOs
h,   ov = hadamard(ψ1, ψ2, VOMPS(D = 16))  # elementwise product c1 .* c2
y,   ov = compress(ψ, VOMPS(D = 8))        # variational bond-dimension reduction
mult!(out, W, ψ, VOMPS(D = 4))             # in-place: D taken from `out` (alg.D ignored)
```

The two-argument `mult(W, ψ)` / `mult(W, W2)` / `hadamard(ψ1, ψ2)` forms give
the exact naive construction (no compression). The `naive_*` twins of
`mult` / `hadamard` materialize the full target family first and are intended
for debugging.

### Observables

```julia
expectationvalue(ψ, H)                     # total energy over the unit cell
expectationvalue(ψ, (1,) => Sz())          # local ⟨Sz⟩
correlator(ψ, Sz(), Sz(), 1, 2:10)         # two-point correlators
entropy(ψ, 1)                              # entanglement entropy on bond 1
entanglement_spectrum(ψ, 1)
```

## Where to go next

- [Conventions](@ref): index orders, the mixed canonical form, and the
  package-wide **forced normalization convention** (implementation details).
- [Algorithms](@ref): the algorithm parameter objects and every driver
  function.
- [Library](@ref): the full API reference generated from the docstrings.
