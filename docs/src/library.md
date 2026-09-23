# Library

Documentation generated from the docstrings.

## States and containers

```@docs
CanonicalIMPS
CanonicalIMPO
PeriodicVector
PeriodicArray
scalartype
phydims
bonddim
max_bonddim
eachsite
dag
norm
normalize!
dot
fidelity
infidelity
ismixedcanonical
mixedcanonical_error
randomimps
prodimps
```

## Operators

```@docs
DenseIMPO
SparseIMPO
SchurMPOTensor
identityimpo
randomimpo
vectorize
devectorize
superoperator
asmps_view
mps_view_to_mpo
fuse
```

## Gauges and transfer matrices

```@docs
gaugefix!
regauge!
LeftCanonical
RightCanonical
MixedCanonical
TransferMatrix
push_env_left
push_env_right
fixedpoint
linsolve
regularize!
transfer_leftenv!
transfer_rightenv!
```

## Environments

```@docs
Environments
DMRGCache
MultCache
OverlapCache
recalculate!
leftenv
rightenv
AC_hamiltonian
C_hamiltonian
calc_galerkin
normalize_envs!
```

## Algorithms and drivers

```@docs
Algorithm
VUMPS
IDMRG
VOMPS
TDVP
DynamicTol
updatetol
find_groundstate
timestep
time_evolve
integrate
```

## Iterative MPO algebra

```@docs
mult
mult!
naive_mult
compress
compress!
hadamard
hadamard!
naive_hadamard
svdguess_mult
svdguess_hadamard
svdguess_compress
changebond!
mpo_compress
exact_mult
exact_add
exact_hadamard
```

## TEBD gates

```@docs
AbstractGate
UnitaryGate
GeneralGate
apply!
swap!
positions
operator
shift
```

## Tensor operations and truncation

```@docs
tsvd
tsvd!
leftorth
leftorth!
rightorth
rightorth!
OrthogonalFactorizationAlgorithm
QR
QRpos
LQ
LQpos
SVD
SDD
Polar
TruncationScheme
NoTruncation
TruncateDim
truncdim
TruncateRelError
truncrelerr
TruncateDimCutoff
truncdimcutoff
permute
distance
distance2
isometry
renyi_entropy
```

## Sparse MPO and time evolution MPOs

```@docs
ExpDecayOpTerm
ExpDecayOpSum
isidentitylevel
isemptylevel
nlvls
tompotensors
tompotensor
infinite_mpo
WI
WII
make_time_mpo
```

## Observables

```@docs
expectationvalue
correlator
entropy
entanglement_spectrum
contract_mpo_expval
```

## Models

```@docs
bulk_mpo
mpohamiltonian
heisenberg_xxz
heisenberg_hamiltonian
tfim
tfim_hamiltonian
fermi_hubbard
σx
σy
σz
Sx
Sy
Sz
```

## Defaults

```@docs
Defaults
```
