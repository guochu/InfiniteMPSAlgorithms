# Conventions

This page collects the package-wide conventions: index orders, the mixed
canonical form, and the **forced normalization convention** together with its
implementation details.

## Index orders

- **MPS site tensors** are rank-3 `(Dl, s, Dr)` arrays: left bond, physical,
  right bond.
- **MPO site tensors** are rank-4 `(wl, u, wr, d)` arrays: the bond indices
  sit in slots 1 and 3 (aligned with TEMPO), and `u` / `d` are the operator
  row (output) / column (input) physical indices. The MPO·MPO multiplication
  kernel is
  `W12[(wl1,wl2), u, (wr1,wr2), d] = Σ_m W1[wl1,u,wr1,m] · W2[wl2,m,wr2,d]`
  with the contracted middle physical index `m = W1.d = W2.u`.
- **Gates** ([`UnitaryGate`](@ref) / [`GeneralGate`](@ref)) store the
  rank-`2N` tensor `(i1', …, iN', i1, …, iN)`: all bra (output) physical
  indices first, all ket (input) physical indices second, each block ordered
  with site 1 the slowest index.
- **Unit cells** are periodic. All tensor strings are stored in
  [`PeriodicVector`](@ref)s of length `N = length(ψ)`, and every index wraps
  around with period `N` (`ψ.C[0] == ψ.C[N]`, and so on).

## Mixed canonical form

The state storage types [`CanonicalIMPS`](@ref) and [`CanonicalIMPO`](@ref)
mirror MPSKit's `InfiniteMPS` layout: four families of tensors,

- `AL[ℓ]`: left-canonical site tensors, `Σ_s AL[ℓ][:,s,:]†·AL[ℓ][:,s,:] = I`;
- `AR[ℓ]`: right-canonical site tensors, `Σ_s AR[ℓ][:,s,:]·AR[ℓ][:,s,:]† = I`;
- `C[ℓ]`: center matrices on the bond to the right of site `ℓ`;
- `AC[ℓ]`: center site tensors,

satisfying the mixed-canonical identities

```text
AL[ℓ]·C[ℓ] = AC[ℓ] = C[ℓ-1]·AR[ℓ]        (all indices periodic, ℓ = 1..N)
```

The package norm is `norm(ψ) = norm(ψ.AC[1])` (Frobenius, consistent with
MPSKit). The diagnostics `mixedcanonical_error` and `ismixedcanonical` report
the three residuals `‖ΣAL†AL − I‖`, `‖ΣAR·AR† − I‖`, and
`‖AL·C − C·AR‖` (maximized over sites, `C` closed periodically).

## The forced normalization convention

The package maintains a strict, *forced* normalization discipline: **no
procedure ever returns an unnormalized or arbitrarily scaled representative
unless its docstring says so explicitly**. This section documents what the
convention is and how it is implemented.

### 1. The norm convention

```text
norm(ψ) = ‖AC[1]‖ = 1        (the package-wide norm convention)
```

[`LinearAlgebra.norm(ψ::CanonicalIMPS)`](@ref) is the Frobenius norm of the
first center tensor, `norm(ψ.AC[1])`; for a mixed-canonical state this is the
ring norm of the state. `CanonicalIMPO` uses the identical convention on its
MPS view `(wl, u·d, wr)`: `norm(W) = norm(W.AC[1])`, i.e. the Hilbert–Schmidt
norm of the vectorized operator.

### 2. `normalize!` distributes the norm over the whole cell

[`LinearAlgebra.normalize!(ψ::CanonicalIMPS)`](@ref) mirrors MPSKit:

```julia
normalize!.(parent(ψ.C))    # every bond matrix to unit Frobenius norm
normalize!.(parent(ψ.AC))   # every center tensor to unit Frobenius norm
```

For a mixed-canonical state the two consistency relations imply that *all*
bond matrices share one common Frobenius norm,
`‖C[ℓ]‖ = ‖AC[ℓ+1]‖ = norm(ψ)` for every `ℓ`. Normalizing every `C` and every
`AC` to unit norm therefore fixes this single common scale — the scale is not
parked in one tensor but forced onto the whole cell uniformly. This is the
ring normalization `⟨ψ, ψ⟩ = 1` with `norm(ψ) = norm(ψ.AC[1]) = 1`.

The same method is defined for `CanonicalIMPO` on the MPS view
(`⟨W, W⟩ = 1`).

### 3. `norm`-preserving global normalization of algebra results

The variational algebra engines (`mult`, `hadamard`, `compress`,
`mpo_compress`, ...) converge to a **ray**: the result is only defined up to a
global phase and scale. Internally every engine terminates with
`_global_normalize!(x)` ([`mult.jl`](../../src/algorithms/mult.jl)):

```julia
n = norm(x)                    # = ‖x.AC[1]‖
x.AC[ℓ] ./= n   and   x.C[ℓ] ./= n   for all ℓ in 1:N
```

i.e. **one common scalar** is divided out of the `AC` and `C` families
simultaneously. The common scalar (rather than a per-site normalization) is
essential:

- it commutes with all gauge structures and preserves the pointwise identity
  `AC = AL·C` exactly — no re-canonicalization is needed afterwards;
- it forces `‖AC[1]‖ = 1`, the package-wide norm convention.

Per-site rescaling of a partially canonical state would instead break
`AC = AL·C`, which is why the algebra engines never use per-site
normalization on their intermediate results.

### 4. Ray semantics: amplitudes are deliberately dropped

Because a variational approximation of `W·ψ` is defined up to a global phase
and scale, the algebra results are **ray representatives**: the absolute
operator amplitude is not restored after compression. Consequently:

- compare results with [`fidelity`](@ref) / `infidelity` (invariant under
  independent phases and scalings) or with the reported `overlap`, never by
  raw amplitudes;
- `overlap` is the **ring-trace fidelity** in `[0, N]`:
  `N·|⟨x|target⟩| / √(⟨x|x⟩⟨target|target⟩)`; `overlap ≈ N` means the result
  points in the same direction as the target. For the `D = nothing` exact
  paths the output is the target ray itself, so `overlap = N` identically;
- gauge transformations of the underlying tensors (including per-site
  phases) telescope away in the periodic trace `tr(∏ W[ℓ])`, so amplitudes
  of the *naive* constructions are preserved exactly by the canonical
  storage.

### 5. Canonical storage conversions

[`DenseIMPO(W::CanonicalIMPO)`](@ref) collects the **left-canonical** family
`W.AL` (not `W.AC`): `tr(∏ AL[ℓ])` is the gauge-invariant operator amplitude
(equal to the construction input amplitude up to a positive real scalar),
whereas `tr(∏ AC[ℓ])` is `C`-weighted and gauge dependent. Using `AL` makes
the conversion unique.

### 6. Environment normalization

Environment caches implement the same discipline, mirroring MPSKit's
`normalize!(::InfiniteEnvironments)`:

- **`DMRGCache` (dense channel)** — each right environment `GR[ℓ]` is
  Frobenius-normalized first; then, per site,
  `λᵢ = ⟨ACᵢ | GL[i+1]·Wᵢ·GR[i] | ACᵢ⟩` scales `GL[i+1]` so that the local
  contraction of every site is exactly `1` (identity-MPO expectation = `N`).
- **`MultCache` / `OverlapCache` (ternary channels)** — identical scheme with
  the local overlap `λℓ = ⟨x.C[ℓ], C_map(ℓ)⟩` scaling `GLs[ℓ+1]`; this keeps
  every local channel contraction at exactly `1` across sweeps, so reported
  overlaps and Galerkin residuals are scale-free.
- **`DMRGCache` (Schur channel)** — the identity levels of the left/right
  environments are pinned to the identity matrix (the `AL`/`AR` gauge fixed
  points), and the dominant ring fixed-point component is projected out of
  every identity-level slice with `regularize!`; the remaining levels are
  obtained from per-level regularized linear solves.
- **`normalize_envs!`** — the incremental (IDMRG) variant: right environments
  are kept at unit norm and left environments are scaled such that
  `dot(C, C_hamiltonian·C) = 1`, preventing environment-norm blowup during
  sweeps.

### 7. The AR + s form (TEBD)

[`spectralize!`](@ref) produces the **AR + s form** used by the TEBD gate
discipline: `C[ℓ]` diagonal positive (the bond Schmidt spectra) with the
mixed-canonical identity network exact by construction. In this gauge the
Hastings seam solves are exact and `apply!` / `swap!` keep the canonical form
at machine precision. Note that `spectralize!` is *not* a pure gauge
transform — it replaces the bond matrices by the compatible Hermitian
positive chain and generally changes the state (see its docstring); the norm
is fixed afterwards with `normalize!`.

## Relationship to MPSKit and TEMPO

Function names follow MPSKit wherever possible (`find_groundstate`,
`timestep`, `expectationvalue`, `correlator`, `gaugefix!`, ...), and the
algorithms are documented against their MPSKit counterparts. The low-level
tensor factorizations (`tsvd`, `leftorth`, `rightorth`) and truncation
schemes are ported from TEMPO's `tensorops`. Compared to MPSKit, this package:

- targets **plain dense arrays** instead of symmetry tensors;
- provides **iterative MPO arithmetic** (`mult` / `hadamard`) with
  compute-on-the-fly compression;
- keeps the implementation deliberately small and self-contained.
