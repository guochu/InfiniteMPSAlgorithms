# W^I / W^II time-evolution MPOs — interface aligned with MPSKit (`WI`/`WII`/
# `make_time_mpo`, acting on SchurMPOTensor). Reference: arXiv:1407.1832
# "Time-evolving a matrix product state with long-ranged interactions".
#
# 步进器类型（`WI`/`WII`）与张量级演化内核（`timeevompo` 的块指数方案：
# WII 的 4d×4d 块传播矩阵 `LinearAlgebra.exp`、WI 的一阶展开、`_sqrt2` 的
# 实/复 √δ 分解）由 FiniteMPSAlgorithms 提供
#（`src/algorithms/timeevo/w1w2.jl`，两侧移植自同一 TEMPO/MPSKit 源）；
# 本文件只保留 infinite 侧的包装：δ 换算（`exp(-i·H·dt)` / `exp(-H·dt)`）
# 与 `DenseIMPO` 装配（平移不变单胞的逐站演化）。

"""
    make_time_mpo(bulk::SchurMPOTensor, dt, alg::Union{WI,WII};
                  imaginary_evolution = false) -> DenseIMPO
    make_time_mpo(H::SparseIMPO, dt, alg; kwargs...) -> DenseIMPO

Build the periodic time-evolution MPO approximating `exp(-i·H·dt)` (mirrors
MPSKit's `make_time_mpo`; with `imaginary_evolution = true` it is
`exp(-H·dt)`). Internally implements the W^I/W^II block-exponential schemes
(via FiniteMPSAlgorithms' `timeevompo`); for an `SparseIMPO`, the Schur tensor
of **every site** in the unit cell is evolved separately (mirroring MPSKit's
`tmap(H.Ws) do W ... end` + `DenseIMPO(PeriodicArray(O))`), supporting
arbitrary unit-cell lengths.
"""
function make_time_mpo(bulk::SchurMPOTensor, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    return DenseIMPO([timeevompo(bulk, δ, alg)])
end

function make_time_mpo(H::SparseIMPO, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    # mirroring MPSKit: evolve the Schur tensor of every site in the unit cell
    # (supports arbitrary unit-cell lengths)
    O = [timeevompo(W, δ, alg) for W in H.Ws]
    return DenseIMPO(O)
end
