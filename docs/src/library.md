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
```

## Iterative MPO algebra

```@docs
mult
mult!
compress
compress!
hadamard
hadamard!
svdguess_mult
svdguess_hadamard
svdguess_compress
changebond!
DenseIMPS
copyphyims
```

## Strict algebra operators

```@docs
Base.:*(::DenseIMPO, ::DenseIMPO)
Base.:*(::DenseIMPO, ::DenseIMPS)
hadamard(::DenseIMPS, ::DenseIMPS)
```

## TEBD gates

```@docs
AbstractGate
InfiniteMPSAlgorithms.GateOp
InfiniteMPSAlgorithms.isunitary
apply!
swap!
positions
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
tompotensors
tompotensor
WI
WII
TaylorCluster
timeevompo
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

模型构造器与公共算符收在 `InfiniteMPSAlgorithms.Models` 子模块：
`using InfiniteMPSAlgorithms.Models` 后直接可用。

```@docs
InfiniteMPSAlgorithms.Models.bulk_mpo
InfiniteMPSAlgorithms.Models.mpohamiltonian
InfiniteMPSAlgorithms.Models.heisenberg_xxz
InfiniteMPSAlgorithms.Models.heisenberg_hamiltonian
InfiniteMPSAlgorithms.Models.tfim
InfiniteMPSAlgorithms.Models.tfim_hamiltonian
InfiniteMPSAlgorithms.Models.fermi_hubbard
InfiniteMPSAlgorithms.Models.σx
InfiniteMPSAlgorithms.Models.σy
InfiniteMPSAlgorithms.Models.σz
InfiniteMPSAlgorithms.Models.Sx
InfiniteMPSAlgorithms.Models.Sy
InfiniteMPSAlgorithms.Models.Sz
```

## Defaults

默认参数与默认算法构造器收在 `InfiniteMPSAlgorithms.Defaults` 子模块：
`using InfiniteMPSAlgorithms.Defaults` 后直接可用。

```@docs
InfiniteMPSAlgorithms.Defaults
```
