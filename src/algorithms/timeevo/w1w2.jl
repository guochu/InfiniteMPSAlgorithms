# W^I / W^II time-evolution MPOs — interface aligned with MPSKit (`WI`/`WII`/
# `make_time_mpo`) and FMA (`timeevompo`). Reference: arXiv:1407.1832
# "Time-evolving a matrix product state with long-ranged interactions".
#
# 步进器类型（`WI`/`WII`）与张量级演化内核（`timeevompo` 的块指数方案：
# WII 的 4d×4d 块传播矩阵 `LinearAlgebra.exp`、WI 的一阶展开、`_sqrt2` 的
# 实/复 √δ 分解）由 FiniteMPSAlgorithms 提供
#（`src/algorithms/timeevo/w1w2.jl`，两侧移植自同一 TEMPO/MPSKit 源）；
# 本文件只保留 infinite 侧的方法扩展：`SparseIMPO` 的逐站演化 + `DenseIMPO`
# 装配（δ 即指数系数本身——实时/虚时由用户经 δ 指定，同 TDVP `integrate`
# 的约定）。

"""
    timeevompo(H::SparseIMPO, δ, alg::Union{WI,WII}) -> DenseIMPO

Build the periodic time-evolution MPO approximating `exp(δ·H)` — the
`SparseIMPO` method of FiniteMPSAlgorithms' `timeevompo` (mirrors MPSKit's
`make_time_mpo`). `δ` is the exponent coefficient itself——实时演化输入
`δ = -im·t`（`exp(-i·H·t)`）、虚时冷却输入 `δ = -τ`（`exp(-H·τ)`），同本包
TDVP `integrate` 的约定. Internally implements the W^I/W^II block-exponential
schemes: the Schur tensor of **every site** in the unit cell is evolved
separately (mirroring MPSKit's `tmap(H.Ws) do W ... end` +
`DenseIMPO(PeriodicArray(O))`), supporting arbitrary unit-cell lengths.
"""
function timeevompo(H::SparseIMPO, δ::Number, alg::Union{WI,WII})
    # mirroring MPSKit: evolve the Schur tensor of every site in the unit cell
    # (supports arbitrary unit-cell lengths)
    O = [timeevompo(W, δ, alg) for W in H.Ws]
    return DenseIMPO(O)
end
