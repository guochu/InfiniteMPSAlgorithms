# ---------------- SchurMPOTensor：FiniteMPSAlgorithms 后端 + 本包语义扩展 ----------------
#
# Upper-triangular block-matrix representation of an MPO (symmetry-free, plain
# Array version). The physical index order matches the rest of the package
# (TEMPO convention `W[wl, u, wr, d]`, bond indices in slots 1 and 3):
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# 类型本体（struct、逻辑矩阵构造器、块访问 getindex/setindex!、稠密化
# tompotensor、直和加法 `+`（通道拼接、D 角相加）、copy/scalartype/complex）
# 由 FiniteMPSAlgorithms 提供（`src/operators/schurmpotensor.jl`，由本包移植
# 并扩展了矩形通道空间 `space_l ≠ space_r` 的支持——本包的平移不变单胞张量
# 恒为方形）。本文件只保留本包侧的命名与语义扩展。

"""
    nlvls(W::SchurMPOTensor)

The number of virtual levels of the Schur tensor (= bond channels + 2 unit
levels)——本包命名。For the square (translation-invariant unit cell) tensors
of this package `nlvls(W) == space_l(W) == space_r(W)`; FMA 后端的矩形张量
请直接用 `space_l`/`space_r`。
"""
nlvls(W::SchurMPOTensor) = space_l(W)

"""
    Base.:*(λ::Number, W::SchurMPOTensor) -> SchurMPOTensor

物理块缩放（本包语义扩展，FMA 后端无对应）：只缩放物理项 `B`/`C`/`D`，
传播块 `A` 与隐式恒等角保持不动——非恒等路径的端点各携带缩放因子时传播
块不参与（`A` 缩放会使路径值依赖路径长度地多乘 λ，破坏谱语义）。
"""
Base.:*(λ::Number, W::SchurMPOTensor) =
    SchurMPOTensor(copy(W.A), λ .* W.B, λ .* W.C, λ .* W.D)
