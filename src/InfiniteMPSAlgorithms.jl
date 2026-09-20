"""
    InfiniteMPSAlgorithms

无对称性的无限 MPS/MPO 张量网络算法包（MPSKit 核心算法的干净稠密实现）。

- 张量为普通 `Array`，收缩用 `TensorOperations`（`@tensor`）；
- 低层张量操作复用 TEMPO 的 `tensorops`（`tsvd`/`leftorth`/`rightorth`/截断方案）；
- `InfiniteCanonicalMPS` 数据布局与 MPSKit 的 `InfiniteMPS` 一致
  （`AL`/`AR`/`C`/`AC` + 周期下标），函数名尽可能与 MPSKit 对齐；
- MPO site 张量指标与 TEMPO 对齐：`W[wl, u, wr, d]`（键指标在 1、3 位）；
- 算法：single-site VUMPS / IDMRG、TDVP（`timestep`/`time_evolve`）、
  W^I/W^II（`make_time_mpo`）+ 迭代 MPO 乘法 `mult`、MPO 压缩。
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
# InfiniteCanonicalMPO（数据结构上属于 states，含 MPS 视图变换与 mpo_compress；
# 依赖 InfiniteMPO，故在 infinitempo.jl 之后 include）
include("states/mixedcanonicalmpo.jl")
include("operators/w1w2.jl")

# ---- 转移矩阵 ----
include("transfermatrix.jl")

# ---- 环境机制（抽象类型与共享 kernel；具体 Cache 定义在 algorithms/ 各算法内）----
include("environments.jl")
include("effective.jl")

# ---- 算法 ----
# algdefs：VUMPS / IDMRG / VOMPS 参数对象（含 trunc 字段）；
# mult：迭代 MPO 乘法（压缩引擎）；add / hadamard：迭代算术（与 mult 共用引擎）；
# arithmetics：朴素精确构造 exact_*（仅供 debug）
include("algorithms/algdefs.jl")
include("algorithms/vumps.jl")
include("algorithms/idmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/mult.jl")
include("algorithms/add.jl")
include("algorithms/hadamard.jl")
include("algorithms/arithmetics.jl")

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
    InfiniteCanonicalMPS, InfiniteMPO, InfiniteCanonicalMPO,
    scalartype, phydims, max_bonddim, bonddim, dag, eachsite,
    ismixedcanonical, mixedcanonical_error,
    norm, normalize!, dot,
    # 构造器
    randomimps, prodimps, identityimpo, randomimpo,
    # 规范
    gaugefix!, regauge!,
    LeftCanonical, RightCanonical, MixedCanonical,
    # 转移矩阵与固定点
    TransferMatrix, push_env_left, push_env_right, fixedpoint, linsolve, regularize!,
    transfer_leftenv!, transfer_rightenv!,
    # 环境缓存（具体类型定义在所属算法文件内）
    Environments, OverlapCache, MultCache, DMRGCache,
    recalculate!, leftenv, rightenv,
    AC_hamiltonian, C_hamiltonian, calc_galerkin,
    # 基态与演化算法
    Algorithm, VUMPS, IDMRG, TDVP, VOMPS,
    find_groundstate, timestep, time_evolve, integrate,
    mult, exact_mult, exact_add, exact_hadamard, add, hadamard, fuse, mpo_compress,
    DynamicTol, updatetol,
    # MPOHamiltonian（Jordan 结构）与时间演化 MPO
    JordanMPOTensor, MPOHamiltonian, FiniteMPOHamiltonian, InfiniteMPOHamiltonian,
    isidentitylevel, isemptylevel, nlvls,
    tompotensors, tompotensor, infinite_mpo,
    WI, WII, make_time_mpo,
    # 观测量
    expectationvalue, correlator, entropy, entanglement_spectrum,
    contract_mpo_expval,
    # 模型
    bulk_mpo, mpohamiltonian, heisenberg_xxz, heisenberg_hamiltonian,
    tfim, tfim_hamiltonian, fermi_hubbard,
    σx, σy, σz, Sx, Sy, Sz,
    # 其他
    Defaults, renyi_entropy, isometry, permute, distance, distance2

end # module
