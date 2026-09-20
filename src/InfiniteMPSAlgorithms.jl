"""
    InfiniteMPSAlgorithms

无对称性的无限 MPS/MPO 张量网络算法包（MPSKit 核心算法的干净稠密实现）。

- 张量为普通 `Array`，收缩用 `TensorOperations`（`@tensor`）；
- 低层张量操作复用 TEMPO 的 `tensorops`（`tsvd`/`leftorth`/`rightorth`/截断方案）；
- `MixedCanonicalMPS` 数据布局与 MPSKit 的 `InfiniteMPS` 一致
  （`AL`/`AR`/`C`/`AC` + 周期下标），函数名尽可能与 MPSKit 对齐；
- MPO site 张量指标与 TEMPO 对齐：`W[wl, u, wr, d]`（键指标在 1、3 位）；
- 算法：single-site VUMPS / IDMRG、TDVP（`timestep`/`time_evolve`）、
  W^I/W^II（`make_time_mpo`）+ 变分 `apply`、MPO 压缩。
"""
module InfiniteMPSAlgorithms

using LinearAlgebra
using Random
using Printf
using TensorOperations
using KrylovKit
using MatrixAlgebraKit

# ---- 低层张量操作（复制自 TEMPO/src/tensorops）----
include("tensorops/tensorops.jl")

include("utility.jl")

# ---- 数据结构（states）----
include("states/mixedcanonicalmps.jl")
include("states/ortho.jl")
include("states/constructors.jl")

# ---- 算子（operators）----
include("operators/infinitempo.jl")
include("operators/jordanmpotensor.jl")
include("operators/mpohamiltonian.jl")
# MixedCanonicalMPO（数据结构上属于 states，含 MPS 视图变换与 mpo_compress；
# 依赖 InfiniteMPO，故在 infinitempo.jl 之后 include）
include("states/mixedcanonicalmpo.jl")
include("operators/w1w2.jl")

# ---- 转移矩阵 ----
include("transfermatrix.jl")

# ---- 环境（OverlapCache / DMRGCache 与有效哈密顿量）----
include("environments.jl")
include("effective.jl")

# ---- 算法 ----
include("algorithms/vumps.jl")
include("algorithms/idmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/apply.jl")
include("algorithms/algebra.jl")

# ---- 观测量 ----
include("observables/expval.jl")
include("observables/correlators.jl")
include("observables/toolbox.jl")

# ---- 模型 ----
include("models.jl")

# ---- 导出（命名与 MPSKit 对齐；MPO 稀疏层保留 TEMPO 命名）----
export
    # 周期容器
    PeriodicVector, PeriodicArray,
    # 截断与分解（tensorops）
    TruncationScheme, NoTruncation, TruncateDim, truncdim,
    TruncateRelError, truncrelerr, TruncateDimCutoff, truncdimcutoff,
    tsvd, tsvd!, leftorth, leftorth!, rightorth, rightorth!,
    OrthogonalFactorizationAlgorithm, QR, QRpos, LQ, LQpos, SVD, SDD, Polar,
    # 数据结构
    MixedCanonicalMPS, InfiniteMPO, MixedCanonicalMPO,
    scalartype, physicaldims, maxbond, bond, mpobond, dag, eachsite,
    norm, normalize!, dot,
    # 构造器
    random_mps, product_mps, identity_mpo, random_mpo,
    # 规范
    gaugefix!, regauge!,
    LeftCanonical, RightCanonical, MixedCanonical,
    # 转移矩阵与固定点
    TransferMatrix, push_env_left, push_env_right, fixedpoint, linsolve, regularize!,
    transfer_leftenv!, transfer_rightenv!,
    # 环境
    environments, Environments, OverlapCache, MultCache, DMRGCache,
    recalculate!, leftenv, rightenv,
    AC_hamiltonian, C_hamiltonian, calc_galerkin,
    # 基态与演化算法
    Algorithm, VUMPS, IDMRG, TDVP, VOMPS,
    find_groundstate, timestep, time_evolve, integrate,
    apply, approximate, fuse, mpo_compress,
    hadamard,
    DynamicTol, updatetol,
    # MPOHamiltonian（Jordan 结构）与时间演化 MPO
    JordanMPOTensor, MPOHamiltonian, FiniteMPOHamiltonian, InfiniteMPOHamiltonian,
    isidentitylevel, isemptylevel, nlvls,
    tompotensors, tompotensor, infinite_mpo,
    WI, WII, make_time_mpo,
    # 观测量
    expectation_value, correlator, entropy, entanglement_spectrum,
    contract_mpo_expval,
    # 模型
    bulk_mpo, mpohamiltonian, heisenberg_xxz, heisenberg_hamiltonian,
    tfim, tfim_hamiltonian, fermi_hubbard,
    σx, σy, σz, Sx, Sy, Sz,
    # 其他
    Defaults, renyi_entropy, isometry, permute, distance, distance2

end # module
