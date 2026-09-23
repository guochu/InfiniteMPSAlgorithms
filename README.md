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
  composition), `hadamard` (elementwise product), and `compress` (bond
  reduction), all with *compute-on-the-fly* variational compression: the naive
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

The target bond dimension is carried by the algorithm object (`alg.D`); the
in-place twins take it from the provided initial guess `out` instead:

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

## Function-name correspondence with MPSKit

Function names follow MPSKit 0.13 wherever possible; the concordance tests in
`test/mpskit/` compare results against MPSKit step by step (same initial
states, same parameters).

**Renamed**

| MPSKit | InfiniteMPSAlgorithms | remark |
|---|---|---|
| `InfiniteMPS` | `CanonicalIMPS` | mixed-canonical storage (`AL`/`AR`/`C`/`AC`), same layout |
| `InfiniteMPO` | `DenseIMPO` | dense MPO; bond indices in slots 1/3 (TEMPO order) |
| `MPOHamiltonian` | `SparseIMPO` | Jordan/Schur block-form Hamiltonian MPO |
| `JordanMPOTensor` | `SchurMPOTensor` | renamed (upper-triangular block tensor) |
| `expectation_value` | `expectationvalue` | |
| `approximate` | `mult` / `compress` | variational application/compression with compute-on-the-fly targets |
| `changebonds` / `changebonds!` | `changebond!` | uniform bond profile (zero-pad / truncate) |

**Same names** (semantics mirror MPSKit unless noted): `PeriodicVector`,
`PeriodicArray`, `find_groundstate`, `VUMPS`, `IDMRG`, `VOMPS`, `TDVP`,
`timestep`, `time_evolve`, `make_time_mpo`, `WI`, `WII`, `environments`,
`leftenv`, `rightenv`, `AC_hamiltonian`, `C_hamiltonian`, `calc_galerkin`,
`gaugefix!`, `regauge!`, `LeftCanonical`, `RightCanonical`, `MixedCanonical`,
`TransferMatrix`, `regularize!`, `correlator`, `entropy`,
`entanglement_spectrum`, `DynamicTol`, `updatetol`, `Defaults`.

**Implemented here but not in MPSKit**

- **Iterative MPO algebra** (compute-on-the-fly, naive family never stored):
  `mult` / `mult!` (variational MPO·MPO composition and MPO·MPS application),
  `compress` / `compress!` (standalone bond reduction), `hadamard` /
  `hadamard!` (elementwise product), and `mpo_compress`.
- **Naive exact constructors**: `exact_mult` (corresponds to MPSKit's naive
  `*`), `exact_add`, `exact_hadamard`; deterministic initial guesses
  `svdguess_mult` / `svdguess_hadamard` / `svdguess_compress`.
- **TEBD quantum gates** on infinite MPS: `apply!`, `swap!`,
  `UnitaryGate`, `GeneralGate` (Hastings update, aligned with TEMPO/GTEMPO).
- **Superoperator layer**: `vectorize`, `devectorize`, `superoperator`.
- **Schur/Jordan helpers and models**: `tompotensor`, `tompotensors`,
  `infinite_mpo`, `bulk_mpo`, `mpohamiltonian`, `isidentitylevel`,
  `isemptylevel`, `nlvls`, `tfim_hamiltonian`, `heisenberg_hamiltonian`,
  `heisenberg_xxz`, `fermi_hubbard` (MPSKit keeps models in MPSKitModels.jl).
- **Constructors and misc**: `randomimps`, `prodimps`, `identityimpo`,
  `randomimpo`, `fidelity`, `infidelity`, `renyi_entropy`,
  `contract_mpo_expval`, `push_env_left`, `push_env_right`.
- **Plain-`Array` tensor factorizations and truncation** (`tsvd`, `leftorth`,
  `rightorth`, `TruncateDim`, `truncrelerr`, ...), ported from TEMPO's
  `tensorops`; MPSKit relies on TensorKit/MatrixAlgebraKit for these.

MPSKit features deliberately out of scope here: finite/window MPS,
quasiparticle excitations, two-site variants (`DMRG2`/`IDMRG2`/`TDVP2`),
`GradientGrassmann`, `TaylorCluster`, and dynamical DMRG.

## Relationship to MPSKit and TEMPO

Function names follow MPSKit wherever possible (`find_groundstate`,
`timestep`, `expectationvalue`, `correlator`, `gaugefix!`, ...), and the
algorithms are documented against their MPSKit counterparts. The low-level
tensor factorizations (`tsvd`, `leftorth`, `rightorth`) and truncation schemes
are ported from TEMPO's `tensorops`. Compared to MPSKit, this package:

- targets **plain dense arrays** instead of symmetry tensors;
- provides **iterative MPO arithmetic** (`mult` / `hadamard` / `compress`) with
  compute-on-the-fly compression;
- keeps the implementation deliberately small and self-contained.

## Testing

```julia
julia> Pkg.test()
```

The `test/mpskit/` concordance suite compares results against MPSKit step by
step (exact multiplication, ground states, environments and effective
Hamiltonians, finite-temperature purification). The `debug/` directory keeps
exploratory scratch scripts.

## Documentation

Documentation lives in `docs/` (Documenter.jl):

```julia
julia> cd("docs")
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
julia> include("make.jl")
```
