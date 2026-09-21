# InfiniteMPSAlgorithms.jl

Symmetry-free infinite matrix-product-state (MPS) / matrix-product-operator
(MPO) tensor-network algorithms — a clean, dense implementation of the core
[MPSKit](https://github.com/QuantumKitHub/MPSKit.jl) algorithms.

Site tensors are plain `Array`s and all contractions are written with
`TensorOperations` (`@tensor`), so the package is easy to read, hack, and
extend.

## Features

- **Data structures** — `InfiniteCanonicalMPS` / `InfiniteCanonicalMPO` in the
  mixed canonical form (`AL` / `AR` / `C` / `AC` families with periodic
  indexing), mirroring MPSKit's `InfiniteMPS` layout.
- **Ground state** — single-site variational uniform MPS (`VUMPS`) and
  infinite DMRG (`IDMRG`) with Galerkin-residual convergence criteria,
  dynamic tolerances, and warm-started environments.
- **Time evolution** — single-site TDVP (`timestep` / `time_evolve`) and
  W^I / W^II time-evolution MPOs (`make_time_mpo`, after
  [arXiv:1407.1832](https://arxiv.org/abs/1407.1832)).
- **Iterative MPO algebra** — `mult` (MPO·MPS application and MPO·MPO
  composition), `add` (mps+mps / mpo+mpo), and `hadamard` (elementwise
  product), all with *compute-on-the-fly* variational compression: the naive
  target family is never materialized and intermediate memory stays at the
  single-site level.
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
ψ = randomimps(ComplexF64, [2, 2], 16)     # random initial state, bond dimension 16

ψ, envs, ϵ = find_groundstate(ψ, H, VUMPS())   # IDMRG() works as well
e = real(expectationvalue(ψ, H) / length(ψ))   # energy density
```

Exact reference for J = h = 1: `e₀ = -4/π ≈ -1.2732395`.

### Time evolution (TDVP)

```julia
dt = 0.1
ψt, envs, history = time_evolve(ψ, H, 0.0:dt:1.0, TDVP(); verbosity = 1)
```

### Time-evolution MPOs (W^I / W^II)

```julia
W = make_time_mpo(H, dt, WII())            # exp(-i·H·dt) as an InfiniteMPO
ψw, fidelity = mult(W, ψ; ψ₀ = ψ)          # apply it (compute-on-the-fly)
```

### Iterative MPO algebra

```julia
ψ′, ov = mult(W, ψ; D = 32)                # apply an MPO, compress to bond 32
W2,  ov = mult(W1, W2; D = 12)             # compose two MPOs
s,   ov = add(ψ1, ψ2; D = 16)              # mps + mps
h,   ov = hadamard(ψ1, ψ2; D = 16)         # elementwise product c1 .* c2
```

With `D = nothing` (the default) the naive exact construction is returned
instead (no compression). The `naive_*` twins of `mult` / `add` / `hadamard`
materialize the full target family first and are intended for debugging.

### Observables

```julia
expectationvalue(ψ, H)                     # total energy over the unit cell
expectationvalue(ψ, (1,) => Sz())          # local ⟨Sz⟩
correlator(ψ, Sz(), Sz(), 1, 2:10)         # two-point correlators
entropy(ψ, 1)                              # entanglement entropy on bond 1
entanglement_spectrum(ψ, 1)
```

## Conventions

- **MPS tensors** are `(Dl, s, Dr)`; **MPO tensors** are `(wl, u, wr, d)`
  (bond indices in slots 1 and 3, aligned with TEMPO), where `u`/`d` are the
  operator row/column indices.
- **Mixed canonical form**: `AL[i]·C[i] = AC[i] = C[i-1]·AR[i]` with
  `Σ AL†·AL = Σ AR·AR† = 1`, all indices taken periodically (unit cell of
  length `N = length(ψ)`).
- **Operator values** live in the periodic trace representation:
  gauge transformations (including per-site phases) telescope away in
  `tr(∏ W[ℓ])`, so amplitudes are preserved exactly by the canonical storage.
- The compression routines report `overlap`, the ring-trace fidelity in
  `[0, N]`: `overlap ≈ N` means the result points in the same direction as the
  target.

## Relationship to MPSKit and TEMPO

Function names follow MPSKit wherever possible (`find_groundstate`,
`timestep`, `expectationvalue`, `correlator`, `gaugefix!`, ...), and the
algorithms are documented against their MPSKit counterparts. The low-level
tensor factorizations (`tsvd`, `leftorth`, `rightorth`) and truncation schemes
are ported from TEMPO's `tensorops`. Compared to MPSKit, this package:

- targets **plain dense arrays** instead of symmetry tensors;
- provides **iterative MPO arithmetic** (`mult` / `add` / `hadamard`) with
  compute-on-the-fly compression;
- keeps the implementation deliberately small and self-contained.

## Testing

```julia
julia> Pkg.test()
```

The `debug/` directory contains concordance scripts that compare results
against MPSKit (exact multiplication, `mult` ≈ `approximate`) and verify the
internal consistency of the iterative algebra.

## References

- Vanderstraeten, Verstraete, Pollmann, *Fortran MPSKit / MPSKit.jl*,
  [SciPost Phys. Codebases 12 (2022)](https://scipost.org/10.21468/SciPostPhysCodeb.12)
- Zaletel, Pollmann, *Isometric Tensor Network States in Two Dimensions*,
  [PRB 104, 075125 (2021)](https://arxiv.org/abs/2102.13027)
- Haegeman et al., *Unifying time evolution and optimization with MPS*,
  [PRL 115, 180404 (2015)](https://arxiv.org/abs/1407.1832)
- Zaletel, Kłys, Chen, Shen, Czisch, *Time-evolving a matrix product state
  with long-ranged interactions*, [PRB 107, 035121 (2023)](https://arxiv.org/abs/1407.1832)
