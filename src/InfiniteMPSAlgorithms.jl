"""
    InfiniteMPSAlgorithms

Symmetry-free infinite MPS/MPO tensor-network algorithms (a clean dense
implementation of the core MPSKit algorithms).

- Site tensors are plain `Array`s; contractions use `TensorOperations` (`@tensor`);
- Low-level tensor operations (`tsvd`, `leftorth`, `rightorth`, truncation
  schemes) come from FiniteMPSAlgorithms and are re-exported;
- The `CanonicalIMPS` data layout mirrors MPSKit's `InfiniteMPS`
  (`AL`/`AR`/`C`/`AC` families with periodic indexing); function names follow
  MPSKit wherever possible;
- MPO site tensors follow TEMPO's convention: `W[wl, u, wr, d]` (bond indices
  in slots 1 and 3);
- Algorithms: single-site VUMPS / IDMRG, TDVP (`timestep`/`time_evolve`),
  W^I/W^II time-evolution MPOs (`make_time_mpo`) + iterative MPO
  multiplication `mult` and MPO compression.
"""
module InfiniteMPSAlgorithms

using LinearAlgebra
using Random
using Printf
using TensorOperations
using KrylovKit
using FiniteMPSAlgorithms

# tensorops 层由 FiniteMPSAlgorithms 提供：显式 import 为本模块自有绑定，
# 使 `InfiniteMPSAlgorithms.tsvd` 等限定访问可用，并与下方 export 组成再导出
import FiniteMPSAlgorithms: TruncationScheme, NoTruncation, TruncateDim,
       TruncateRelError, TruncateDimCutoff, truncdim, truncrelerr, truncdimcutoff,
       truncate!, OrthogonalFactorizationAlgorithm, QR, QRpos, LQ, LQpos, SVD, SDD,
       Polar, tsvd, tsvd!, leftorth, leftorth!, rightorth, rightorth!, tie, permute,
       scalar, isometry, renyi_entropy, distance, distance2

include("utility.jl")

# ---- data structures (states) ----
include("states/canonicalmps.jl")
include("states/ortho.jl")
include("states/constructors.jl")

# ---- operators ----
include("operators/infinitempo.jl")
include("operators/sparsempotensor.jl")
include("operators/mpohamiltonian.jl")
include("operators/longrangeop.jl")
# CanonicalIMPO (a "states" data structure; contains the MPS view
# transforms and mpo_compress; depends on DenseIMPO, hence included after
# infinitempo.jl)
include("states/canonicalmpo.jl")
include("operators/w1w2.jl")

# ---- transfer matrices ----
include("transfermatrix.jl")

# ---- environment machinery (abstract type and shared kernels; concrete
#      caches are defined inside their algorithm files) ----
include("environments.jl")
include("effective.jl")

# ---- algorithms ----
# algdefs: VUMPS / IDMRG / VOMPS parameter objects;
# mult: iterative MPO multiplication (compression engine);
# add / hadamard: iterative arithmetic (sharing the same engine);
# arithmetics: naive exact constructors exact_* (debug only)
include("algorithms/algdefs.jl")
include("algorithms/vumps.jl")
include("algorithms/idmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/mult.jl")
include("algorithms/hadamard.jl")
include("algorithms/compress.jl")
include("algorithms/arithmetics.jl")
include("algorithms/tebd.jl")

# ---- observables ----
include("observables/expval.jl")
include("observables/correlators.jl")
include("observables/toolbox.jl")

# ---- models ----
include("models.jl")

# ---- exports (names aligned with MPSKit; the sparse MPO layer keeps TEMPO names) ----
export
    # periodic containers
    PeriodicVector, PeriodicArray,
    # truncation and factorizations (tensorops)
    TruncationScheme, NoTruncation, TruncateDim, truncdim,
    TruncateRelError, truncrelerr, TruncateDimCutoff, truncdimcutoff,
    DefaultTruncation, truncate!,
    tsvd, tsvd!, leftorth, leftorth!, rightorth, rightorth!,
    OrthogonalFactorizationAlgorithm, QR, QRpos, LQ, LQpos, SVD, SDD, Polar, tie,
    # data structures
    CanonicalIMPS, DenseIMPO, CanonicalIMPO,
    scalartype, phydims, max_bonddim, bonddim, dag, eachsite,
    ismixedcanonical, mixedcanonical_error,
    norm, normalize!, dot,
    # constructors
    randomimps, prodimps, identityimpo, randomimpo,
    # gauges
    gaugefix!, regauge!,
    LeftCanonical, RightCanonical, MixedCanonical,
    # transfer matrices and fixed points
    TransferMatrix, push_env_left, push_env_right, fixedpoint, linsolve, regularize!,
    transfer_leftenv!, transfer_rightenv!,
    # environment caches (concrete types live in their algorithm files)
    Environments, OverlapCache, MultCache, DMRGCache,
    recalculate!, leftenv, rightenv,
    AC_hamiltonian, C_hamiltonian, calc_galerkin,
    # ground-state and time-evolution algorithms
    Algorithm, VUMPS, IDMRG, TDVP, VOMPS,
    find_groundstate, timestep, time_evolve, integrate,
    mult, naive_mult, hadamard, naive_hadamard,
    exact_mult, exact_add, exact_hadamard, fuse, mpo_compress,
    compress, mult!, hadamard!, compress!,
    changebond!, svdguess_mult, svdguess_hadamard, svdguess_compress,
    vectorize, devectorize, superoperator,
    fidelity, infidelity,
    DynamicTol, updatetol,
    # TEBD quantum gates
    AbstractGate, UnitaryGate, GeneralGate, apply!, swap!, positions, shift,
    # SparseIMPO (Schur structure) and time-evolution MPOs
    SchurMPOTensor, SparseIMPO,
    ExpDecayOpTerm, ExpDecayOpSum,
    isidentitylevel, isemptylevel, nlvls,
    tompotensors, tompotensor, infinite_mpo,
    WI, WII, make_time_mpo,
    # observables
    expectationvalue, correlator, entropy, entanglement_spectrum,
    contract_mpo_expval,
    # models
    bulk_mpo, mpohamiltonian, heisenberg_xxz, heisenberg_hamiltonian,
    tfim, tfim_hamiltonian, fermi_hubbard,
    σx, σy, σz, Sx, Sy, Sz,
    # misc
    Defaults, renyi_entropy, isometry, permute, distance, distance2, scalar

end # module
