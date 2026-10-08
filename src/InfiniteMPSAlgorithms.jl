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
  W^I/W^II time-evolution MPOs (`timeevompo`) + iterative MPO
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
# FMA Defaults 的默认正则化截断方案（truncate!/TEBD gates 的 trunc 缺省）；
# `using ...Defaults: f` 只引入函数名，不引入 Defaults 模块名（避免与本包
# 的同名 Defaults 模块冲突）
using FiniteMPSAlgorithms.Defaults: alg_orth_trunc, alg_trunc
# 方法扩展（集中在此声明，各子文件不再出现 using/import；scalartype 扩展
# TensorOperations 的同名函数）。
#
# import 约定（便于用户同时使用两包——同名函数经 import 合并为同一函数
# 对象、按类型自动分派；**仅 using 不 import 会静默创建同名本地函数、
# 切裂方法表**，phydim 委托 bug 的教训）：
# - tompotensor：FMA 未导出的 Schur 稠密化（tompotensors 的底层）；
# - 含义类似、类型分派不重叠的 FMA 导出函数族（作用于 FMA 有限链类型，
#   本包方法作用于 infinite 类型）：distance/distance2/⊙/truncate!/
#   timeevompo/phydim(phydims)/bonddim/vectorize/devectorize/superoperator/
#   fidelity/infidelity/expectationvalue/entanglement_spectrum/changebond!/
#   compress(compress!)/mult(mult!)/hadamard(hadamard!)/tompotensors/
#   svdguess_{mult,hadamard,compress} 与 TEBD 门族——门类型本身
#   （AbstractGate/UnitaryGate/GeneralGate）与基础接口（positions/shift/
#   adjoint）同为 FMA 提供，tebd.jl 只保留 Pair 矩阵便捷构造与
#   CanonicalIMPS 上的 apply!/swap!；
# - **有意遮蔽（不 import，经模块名限定访问 FMA 版）**：
#   · σx/σy/σz/Sx/Sy/Sz：签名相同但缺省类型不同（FMA 全 ComplexF64，
#     本包实矩阵缺省 Float64）；
#   · heisenberg_hamiltonian/tfim_hamiltonian/fermi_hubbard：返回类型不同
#     （FMA 带 L 位置参数的有限 MPOHamiltonian vs 本包无限 SparseIMPO）；
#   · linsolve：语义分层不同（FMA 的 ALS 变分求解 vs 本包的 KrylovKit
#     稠密包装）；
#   · 类型撞名（struct 无法 import 合并）——**门类型已改为直接用 FMA 的**
#   （tebd.jl 删除本地定义）：剩 DMRGCache/MultCache/OverlapCache/
#     HadamardCache（算法 cache，构造器语义已对齐 chain-first）；
#   · Defaults：模块撞名，仅显式 `using ...Defaults: f` 引入个别函数名。
import FiniteMPSAlgorithms: distance, distance2, ⊙, truncate!, tompotensor,
                             timeevompo, phydim, phydims, bonddim,
                             vectorize, devectorize, superoperator,
                             fidelity, infidelity, expectationvalue,
                             entanglement_spectrum, changebond!,
                             compress, compress!, mult, mult!, hadamard, hadamard!,
                             tompotensors, svdguess_mult, svdguess_hadamard,
                             svdguess_compress,
                             apply!, swap!, positions, shift,
                             AbstractGate, UnitaryGate, GeneralGate
import TensorOperations: scalartype

include("utility.jl")

# ---- data structures (states) ----
include("states/canonicalmps.jl")
include("states/infinitemps.jl")
include("states/canonicalmpo.jl")
include("states/linalg.jl")
include("states/ortho.jl")
include("states/ortho_exact.jl")
include("states/constructors.jl")

# ---- operators ----
include("operators/infinitempo.jl")
include("operators/mpohamiltonian.jl")
include("operators/linalg.jl")
# SchurMPOTensor 类型本体与全部块级操作（含 ::ExpDecayOpSum/ExpDecayOpTerm
# 长程算符构造）由 FiniteMPSAlgorithms 直接提供并经上方 using 再导出
# （原 sparsempotensor.jl 薄适配层——nlvls 与 *(λ, W)——无消费者，已删除）

# ---- transfer matrices ----
include("transfermatrix.jl")

# ---- environment machinery (abstract type; the DMRGCache concrete cache and
#      the effective Hamiltonians live in algorithms/groundstates/) ----
include("environments.jl")

# ---- algorithms ----
# algdefs: VUMPS / IDMRG / VOMPS parameter objects;
# groundstates/: environment solvers + DMRGCache + effective Hamiltonians and
#                the VUMPS / IDMRG ground-state searches;
# timeevo/: TDVP and TEBD time evolution, and the W^I/W^II time-evolution
#            MPOs (the `timeevompo(::SparseIMPO)` method);
# arithmetics/: iterative MPO algebra (mult / hadamard / compress) sharing the
#               VOMPS / IDMRG compression engines, and the exact_* debug
#               constructors;
include("algorithms/algdefs.jl")
include("algorithms/groundstates/envs.jl")
include("algorithms/groundstates/effective.jl")
include("algorithms/groundstates/vumps.jl")
include("algorithms/groundstates/idmrg.jl")
include("algorithms/timeevo/tdvp.jl")
include("algorithms/arithmetics/envs.jl")
include("algorithms/arithmetics/mult.jl")
include("algorithms/arithmetics/hadamard.jl")
include("algorithms/arithmetics/compress.jl")
include("algorithms/timeevo/tebd.jl")
include("algorithms/timeevo/w1w2.jl")

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
    truncate!,
    tsvd, tsvd!, leftorth, leftorth!, rightorth, rightorth!,
    OrthogonalFactorizationAlgorithm, QR, QRpos, LQ, LQpos, SVD, SDD, Polar, tie,
    # data structures
    CanonicalIMPS, DenseIMPS, DenseIMPO, CanonicalIMPO,
    scalartype, phydim, phydims, max_bonddim, bonddim, dag, eachsite,
    ismixedcanonical, mixedcanonical_errors,
    norm, normalize!, dot,
    # constructors
    randomimps, prodimps, identityimpo, randomimpo,
    # gauges
    gaugefix!, regauge!,
    LeftCanonical, RightCanonical, MixedCanonical, InfiniteOrthogonalize,
    # transfer matrices and fixed points
    TransferMatrix, push_env_left, push_env_right, fixedpoint, linsolve, regularize!,
    transfer_leftenv!, transfer_rightenv!,
    # environment caches (concrete types live in their algorithm files)
    Environments, CompressionEnvironments, OverlapCache, MultCache, DMRGCache,
    recalculate!, leftenv, rightenv,
    # infinite chain/operator abstract types and family views
    AbstractInfiniteMPS, AbstractInfiniteMPO, BondView,
    AC_hamiltonian, C_hamiltonian, calc_galerkin,
    # ground-state and time-evolution algorithms
    Algorithm, VUMPS, IDMRG, TDVP, VOMPS, IterativeConvergenceInfo,
    find_groundstate, timestep, time_evolve, integrate,
    mult, hadamard, ⊙,
    fuse, copyphyims,
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
    isidentitylevel, isemptylevel,
    tompotensors, tompotensor,
    WI, WII, timeevompo,
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
