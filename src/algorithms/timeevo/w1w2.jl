# W^I / W^II time-evolution MPOs — interface aligned with MPSKit (`WI`/`WII`/
# `make_time_mpo`) and FMA (`timeevompo`). Reference: arXiv:1407.1832
# "Time-evolving a matrix product state with long-ranged interactions".
#
# 步进器类型（`WI`/`WII`/`TaylorCluster`）与张量级演化内核（`timeevompo` 的
# 块指数方案：WII 的 4d×4d 块传播矩阵 `LinearAlgebra.exp`、WI 的一阶展开、
# `_sqrt2` 的实/复 √δ 分解；TaylorCluster 的 N 折通道簇代数——`_site_pow`/
# `_taylor_*` 全套）由 FiniteMPSAlgorithms 提供
#（`src/algorithms/timeevo/{w1w2,taylorcluster}.jl`，两侧移植自同一
# TEMPO/MPSKit 源）；本文件只保留 infinite 侧的方法扩展：`SparseIMPO` 的
# 逐站演化 + `DenseIMPO` 装配（δ 即指数系数本身——实时/虚时由用户经 δ
# 指定，同 TDVP `integrate` 的约定）。

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

"""
    timeevompo(H::SparseIMPO, δ, alg::TaylorCluster) -> DenseIMPO

Build the periodic time-evolution MPO approximating `exp(δ·H)` with the
`N`-th order Taylor cluster expansion ([`TaylorCluster`](@ref)；MPSKit
`make_time_mpo` 家族的 `SparseIMPO` 方法). The **per-site cluster algebra is
FiniteMPSAlgorithms' `timeevompo(::SchurMPOTensor, dt, ::TaylorCluster)`
verbatim**（`H^N` 的 N 折通道字典 + cluster loop-back / equivalents 合并 /
可选修正压缩——两侧移植自同一 MPSKit 源）；本方法只补无限侧装配：环上无
vacuum/closing 边界，每个键的存活通道 = 左站非零列 ∩ 右站非零行（MPSKit
整链 `remove_orphans!` 的逐键等价物），`sort!` 保持原 column-major 空间序
——键空间通道顺序与 MPSKit 一致（输出可与 MPSKit 逐张量比较）。`δ` 约定
同 [`timeevompo`](@ref) 的 `WI`/`WII` 方法（指数系数本身）。
"""
function timeevompo(H::SparseIMPO, δ::Number, alg::TaylorCluster)
    L = length(H)
    T = promote_type(scalartype(H), typeof(δ))
    slices = [timeevompo(W, δ, alg) for W in H.Ws]     # FMA 单站块字典
    # 每键存活通道：左站非零列 ∩ 右站非零行（环闭合；死通道两侧必有一侧无块）
    nzr = [Set(al for (al, br) in keys(s) if !iszero(s[(al, br)])) for s in slices]
    nzc = [Set(br for (al, br) in keys(s) if !iszero(s[(al, br)])) for s in slices]
    I = Vector{Vector{Int}}(undef, L + 1)
    I[1] = sort!(collect(intersect(nzc[L], nzr[1])))
    for s in 2:L
        I[s] = sort!(collect(intersect(nzc[s-1], nzr[s])))
    end
    I[L+1] = I[1]                                      # 环闭合：bond L = bond 0
    O = Vector{Array{T,4}}(undef, L)
    for i in 1:L
        d = phydim(H.Ws[i])
        O[i] = zeros(T, length(I[i]), d, length(I[i+1]), d)
        for (key, blk) in slices[i]
            iszero(blk) && continue
            r = findfirst(==(key[1]), I[i])
            c = findfirst(==(key[2]), I[i+1])
            (r === nothing || c === nothing) && continue
            O[i][r, :, c, :] .= blk
        end
    end
    return DenseIMPO(O)
end
