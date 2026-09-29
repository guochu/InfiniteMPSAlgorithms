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
# tensorops 层由 FiniteMPSAlgorithms 提供（本包仅调用、无方法扩展）；
# 下方 export 将其再导出，对外 API 不变
using FiniteMPSAlgorithms

include("utility.jl")

# ---- data structures (states) ----
include("states/canonicalmps.jl")
include("states/infinitemps.jl")
include("states/linalg.jl")
include("states/ortho.jl")
include("states/constructors.jl")

# ---- operators ----
include("operators/infinitempo.jl")
include("operators/linalg.jl")
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
# groundstates/: VUMPS / IDMRG ground-state searches;
# timeevo/: TDVP and TEBD time evolution;
# arithmetics/: iterative MPO algebra (mult / hadamard / compress) sharing the
#               VOMPS / IDMRG compression engines, and the exact_* debug
#               constructors;
include("algorithms/algdefs.jl")
include("algorithms/groundstates/vumps.jl")
include("algorithms/groundstates/idmrg.jl")
include("algorithms/timeevo/tdvp.jl")
include("algorithms/arithmetics/mult.jl")
include("algorithms/arithmetics/hadamard.jl")
include("algorithms/arithmetics/compress.jl")
include("algorithms/timeevo/tebd.jl")

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
    CanonicalIMPS, DenseIMPS, DenseIMPO, CanonicalIMPO,
    scalartype, phydim, phydims, max_bonddim, bonddim, dag, eachsite,
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
    mult, hadamard,
    fuse, mpo_compress, copyphyims,
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
