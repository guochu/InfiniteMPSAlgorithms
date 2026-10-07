# ---------------- exponentially decaying long-range operators ----------------
#
# 类型与 W-form 构造由 FiniteMPSAlgorithms 提供（`src/operators/longrangeop.jl`）：
# exponentially decaying long-range interaction terms
# `α·λ^d·(â ⊗ m̂^⊗(d-1) ⊗ b̂)`（distance d ≥ 1），每个 (αₚ, λₚ) 参数对一个
# Schur 通道——通道开启 `C = α·λ·a`（携带首步衰减）、自传播 `A = λ·m`、关闭
# `B = b`，距离 d 的路径恰积累 `α·λ^d`（arXiv:1407.1832 的 Schur 构造）。
# 本文件只保留无 on-site 项的单参便捷构造（infinite 平铺用法）。

"""
    SchurMPOTensor(s::ExpDecayOpSum) -> SchurMPOTensor
    SchurMPOTensor(t::ExpDecayOpTerm) -> SchurMPOTensor

无 on-site 项（`hloc = 0`）的便捷构造：FMA 后端
`SchurMPOTensor(s, hloc)` 的 `hloc = zeros` 特例，每项一个 Schur 通道。
"""
SchurMPOTensor(s::ExpDecayOpSum) =
    SchurMPOTensor(s, zeros(scalartype(s), size(s.a, 1), size(s.a, 1)))
SchurMPOTensor(t::ExpDecayOpTerm) = SchurMPOTensor(ExpDecayOpSum(t))
