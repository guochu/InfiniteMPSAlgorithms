# Algorithms

All driver functions follow the MPSKit API shape: the **algorithm object is a
positional argument** (with a default), while data options (`envs`,
`verbosity`, `observer`, ...) are keyword arguments. The parameter objects are
lightweight `@kwdef` structs, so every field can be set by keyword at
construction:

```julia
VUMPS(tol = 1e-12, maxiter = 500, verbosity = 1)
```

Wrapping an inner solver in [`DynamicTol`](@ref) (the default for the
Krylov sub-solvers) activates the MPSKit-style dynamic tolerance update
`new_tol = clamp(ϵ·factor/√iter, min, max)`.

## Ground state

```julia
find_groundstate(operator, alg::Union{VUMPS,IDMRG}, [envs]) -> (ψ, envs, ϵ)
find_groundstate(ψ₀, operator, alg::Union{VUMPS,IDMRG},
                 envs = DMRGCache(ψ₀, operator); which = :SR) -> (ψ, envs, ϵ)
```

`alg.D` (required, `Int`) is the bond dimension. In the convenience form
(without `ψ₀`) a random initial state with `bonddim = alg.D` is generated via
`randomimps`, taking the physical dimensions and scalar type from
`operator`; when an explicit `ψ₀` is passed, `alg.D` is ignored and `ψ₀`'s
bond profile is used.

- **`VUMPS`** — variational uniform MPS (Zaletel–Pollmann /
  Vanderstraeten et al.). Each iteration runs the MPSKit template:
  `localupdate_step!` (solve the `AC`/`C` smallest-eigenpair problems site by
  site, `regauge!` to candidate `AL`s), `gauge_step!` (`gaugefix!` restores
  the global right gauge), `envs_step!` (recompute the environments), then
  the `finalize` callback. The convergence criterion is the Galerkin
  residual `calc_galerkin(ψ, H, envs) ≤ alg.tol`.
- **`IDMRG`** — single-site infinite DMRG. Each iteration is a forward
  sweep (solve `AC_hamiltonian` per site, split with `leftorth`, push the
  left environments incrementally) followed by a backward sweep (split with
  `rightorth`, push the right environments). Convergence is
  `ϵ = ‖C[0] − C_old‖`; the state is rebuilt from `AR` afterwards.

Both return `(ψ, envs, ϵ)` with `ϵ` the final convergence measure.

## Time evolution

### TDVP

```julia
timestep(ψ, H, t, dt, alg::TDVP = TDVP(),
         envs = DMRGCache(ψ, H); imaginary_evolution = false) -> (ψ, envs)

time_evolve(ψ₀, H, t_span, alg::TDVP = TDVP(),
            envs = DMRGCache(ψ₀, H);
            verbosity = 0, imaginary_evolution = false, observer = nothing)
    -> (ψ, envs, history)
```

Single-site TDVP (Haegeman et al.): every `AC` and `C` is evolved
independently with the local exponentiate (`alg.integrator`, a KrylovKit
solver), then re-canonicalized pairwise and rebuilt with a global
`gaugefix!`. `imaginary_evolution = true` gives `exp(−H·dt)` (imaginary
time). `observer(ψ, iter, t)` collects data at each step; `history` stores
the collected values (or the energy when no observer is given).

### TEBD gates

```julia
apply!(g::UnitaryGate{2}, ψ; trunc = NoTruncation()) -> ψ
apply!(g::GeneralGate{2}, ψ; trunc = NoTruncation(), kwargs...) -> ψ
swap!(ψ, i; trunc = NoTruncation()) -> ψ
```

Two-site gates with the Hastings update (aligned with TEMPO/GTEMPO): a
single SVD of the post-gate window distributes the factors onto
`AL[i]`, `AR[i+1]`, `C[i]` without ever dividing the spectrum, so the
physical state is preserved exactly when no truncation is applied.
Non-adjacent gate sites are moved together with exact unitary swaps.
General (non-unitary) gates re-canonicalize with `gaugefix!` afterwards.
No initialization is needed: on any mixed-canonical input the canonical
identity network is preserved at machine precision, for single gates and
for gate sequences alike.

## Iterative MPO algebra

The algebra drivers compute `y ≈ W·x` where `x` is an MPS (operator
application) or an MPO (operator composition), with **compute-on-the-fly**
compression: the naive target family is never materialized, and intermediate
memory stays at the single-site level. The algorithm object is positional:
[`VOMPS`](@ref) (overlap-maximizing ALS sweeps) or [`IDMRG`](@ref)
(rank-1 effective-Hamiltonian eigen-solves); both share the same fixed point.

```julia
mult(W, ψ, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPS, overlap)
mult(W, W2, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPO, overlap)
hadamard(ψ1, ψ2, alg::Union{VOMPS,IDMRG}) -> (y, overlap)
compress(x, alg::Union{VOMPS,IDMRG}) -> (y, overlap)
```

- The target bond dimension is carried by the algorithm object: `alg.D`
  (required, `Int`) variationally compresses to bond dimension `D`.
- The initial guess is the deterministic `svdguess_*` (bond-wise SVD
  truncation of the naive target); the in-place twins below replace it with
  the user-provided guess.
- In-place twins `mult!(out, W, x, alg)`, `hadamard!(out, ψ1, ψ2, alg)`,
  `compress!(out, x, alg)` write the result back into a user-provided `out`:
  the target bond dimension is taken from the bond profile of `out`
  (`D = max_bonddim(out)`, uniformized with `changebond!`), and `alg.D` is
  ignored.
- The two-argument forms `mult(W, ψ)` / `mult(W, W2)` / `hadamard(ψ1, ψ2)`
  perform the exact naive construction + canonical storage (no compression).
- The debug twins `naive_mult` / `naive_hadamard` materialize the complete
  naive family before compressing (reference implementations).
- `overlap` is the ring-trace fidelity in `[0, N]` (`= N` means same
  direction; see [Conventions](@ref)).

Applying a time-evolution MPO is `ψ′, _ = mult(make_time_mpo(H, dt, WII()), ψ)`.

## Time-evolution MPOs (W^I / W^II)

```julia
make_time_mpo(H, dt, alg::Union{WI,WII}) -> W ≈ exp(-im·H·dt)
```

Long-range time-evolution MPOs after Zaletel et al. (PRB 107, 035121
(2023)), built on the Schur-structured `SparseIMPO` layer.

## Sparse MPO layer

`SparseIMPO` stores MPO tensors in Schur (upper-triangular block) form
(level structure
`isidentitylevel` / `isemptylevel` / `nlvls`), used by the Hamiltonian
environments (per-level solves) and the W^I/W^II construction. Densify with
`DenseIMPO(W)`; inspect per-level dense tensors with `tompotensor`.

## Observables

```julia
expectationvalue(ψ, H)              # total over the unit cell (N = length(ψ))
expectationvalue(ψ, (1,) => O)      # local one-site value
expectationvalue(ψ, (1, 2) => O12)  # local two-site value
correlator(ψ, O1, O2, i, js)        # ⟨O1(i) O2(j)⟩ for j ∈ js
entropy(ψ, loc; α = 1)              # von Neumann (α = 1) / Rényi entropy
entanglement_spectrum(ψ, loc)
```

`expectationvalue(ψ, H)` / `length(ψ)` is the energy density.

## Models

```julia
tfim_hamiltonian(; J, h, T)     # SparseIMPO Hamiltonian
heisenberg_xxz(; J, Δ, h, T)
fermi_hubbard(; t, U, μ, T)
```

Each model builder returns a ready-to-use Schur `SparseIMPO`; the `tfim`
family also returns the dense `DenseIMPO`, the `bulk` SchurMPOTensor, and
the `hamiltonian` in one named tuple.

## Environment caches

- `DMRGCache(ψ, H)` — Hamiltonian-channel environments for the ground-state
  and TDVP drivers (Schur per-level solves for `SparseIMPO`, transfer-matrix
  dominant eigenvectors for `DenseIMPO`).
- `MultCache(below, operator, above)` / `OverlapCache(below, above)` — the
  ternary application/overlap channels of the algebra engines.
- `recalculate!(envs, ψ, H; kwargs...)` recomputes environments with warm
  starts; the drivers call it automatically with dynamic tolerances.
