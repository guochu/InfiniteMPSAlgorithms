# ---------------- 朴素精确构造 exact_*（仅供 debug 的基准对照） ----------------
#
# MPSKit 语义基准（src/states/finitemps.jl、src/operators/mpo.jl、abstractmpo.jl）：
# - `FiniteMPS + FiniteMPS`、`FiniteMPO + FiniteMPO`、`InfiniteMPO * InfiniteMPO`：
#   朴素精确构造（键维直和 / fuse），**不做变分压缩**，且要求长度相等（check_length）；
# - `fuse_mul_mpo`：`(O1*O2)` 物理出 = O1 的 u、物理入 = O2 的 d，
#   中间物理指标 m = O1 的 d = O2 的 u，键维 = 两键维乘积；
# - 标量乘只缩放第一个张量（见 infinitempo.jl）。
#
# 本文件只放朴素精确构造：输入/输出均为 `PeriodicVector{Array}` 张量串
# （兼容普通 Vector 输入），不做任何规范化或压缩，**仅供 debug**
# （作为迭代版 mult / add / hadamard 的基准对照）。
# 迭代版本：mult 见 mult.jl，add 见 add.jl，hadamard 见 hadamard.jl。

# ---- 朴素精确构造核（键维直和 / fuse / zip）----

"rank-3 键直和（MPS 加法核）：`A12[(al,bl), s, (ar,br)] = blockdiag(A1, A2)`。"
function _naive_sum_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    size(A1, 2) == size(A2, 2) || throw(DimensionMismatch("MPS 加法要求逐 site 物理维相等"))
    out = zeros(T, size(A1, 1) + size(A2, 1), size(A1, 2), size(A1, 3) + size(A2, 3))
    out[1:size(A1, 1), :, 1:size(A1, 3)] .= A1
    out[size(A1, 1)+1:end, :, size(A1, 3)+1:end] .= A2
    return out
end

"rank-4 键直和（MPO 加法核）：`W12[(wl1,wl2), u, (wr1,wr2), d] = blockdiag(W1, W2)`。"
function _naive_sum_tensor(W1::AbstractArray{T,4}, W2::AbstractArray{T,4}) where {T}
    (size(W1, 2) == size(W2, 2) && size(W1, 4) == size(W2, 4)) ||
        throw(DimensionMismatch("MPO 加法要求逐 site 物理 (u,d) 相等"))
    out = zeros(T, size(W1, 1) + size(W2, 1), size(W1, 2),
                size(W1, 3) + size(W2, 3), size(W1, 4))
    out[1:size(W1, 1), :, 1:size(W1, 3), :] .= W1
    out[size(W1, 1)+1:end, :, size(W1, 3)+1:end, :] .= W2
    return out
end

"""
    _naive_hadamard_tensor(A1, A2) -> A12

rank-3 zip（Hadamard/Schur 积核）：两条虚拟腿 zip 融合、**物理腿共享**，
`A12[(a,c), s, (b,e)] = A1[a,s,b]·A2[c,s,e]`。两条虚拟链独立，周期 trace
因子化：`tr(∏A12) = tr(∏A1)·tr(∏A2)`，即波形逐点乘积 `c12 = c1 .* c2` 的
MPS 表示（物理维不变，键维 = 两键维乘积）。
"""
function _naive_hadamard_tensor(A1::AbstractArray{T,3}, A2::AbstractArray{T,3}) where {T}
    a, s, b = size(A1)
    c, s2, e = size(A2)
    (s2 == s) || throw(DimensionMismatch("hadamard 要求逐 site 物理维相等"))
    # (a,c)/(b,e) 融合顺序：a、b 为主序；逐物理切片做 Kronecker 积
    # （TensorOperations @tensor 不支持 s 在两 operand 同时出现且不收缩的 batched 写法）
    out = similar(A1, a * c, s, b * e)
    for k in 1:s
        out[:, k, :] = kron(A2[:, k, :], A1[:, k, :])
    end
    return out
end

# ---------------- exact_*：朴素精确构造（仅供 debug） ----------------

"""
    exact_mult(W1, W2) -> PeriodicVector{Array{T,4}}
    exact_mult(W, ψ) -> PeriodicVector{Array{T,3}}

朴素精确 MPO 乘法（mpo*mpo：键维 = 两键维乘积，对标 MPSKit `fuse_mul_mpo`；
mpo*mps：键维 = W 键维 × ψ 键维）。输入/输出均为 `PeriodicVector{Array}` 周期
张量串，不做任何规范化或压缩。**仅供 debug**（作为变分 `mult` 的基准对照）；
周期 trace 表示下块对角望远相消，`tr(∏exact_mult(W1,W2)) = tr(∏W1)·tr(∏W2)`
不成立——乘法的算符幅值对照请用稠密表示（见 debug/ 测试）。
"""
function exact_mult(W1::AbstractVector{<:Array{T,4}}, W2::AbstractVector{<:Array{T,4}}) where {T}
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO 乘法要求长度相等（对标 MPSKit check_length）"))
    return PeriodicVector([_naive_mul_tensor(W1[ℓ], W2[ℓ]) for ℓ in 1:length(W1)])
end

function exact_mult(W::AbstractVector{<:Array{T,4}}, ψ::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ) % length(W) == 0) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    return PeriodicVector([fuse(W[_mod1(ℓ, length(W))], ψ[ℓ]) for ℓ in 1:length(ψ)])
end

"""
    exact_add(a1, a2) -> PeriodicVector

朴素精确加法（mps+mps / mpo+mpo）：逐 site 键维直和 `blockdiag(a1[ℓ], a2[ℓ])`，
要求长度与逐 site 物理维相等（对标 MPSKit check_length）。周期 trace 表示下
**精确可加**：`tr(∏exact_add(a1,a2)) = tr(∏a1) + tr(∏a2)`。
输入/输出均为 `PeriodicVector{Array}` 张量串，不做任何规范化。**仅供 debug**。
"""
function exact_add(ψ1::AbstractVector{<:Array{T,3}}, ψ2::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("MPS 加法要求长度相等（对标 MPSKit check_length）"))
    return PeriodicVector([_naive_sum_tensor(ψ1[ℓ], ψ2[ℓ]) for ℓ in 1:length(ψ1)])
end

function exact_add(W1::AbstractVector{<:Array{T,4}}, W2::AbstractVector{<:Array{T,4}}) where {T}
    (length(W1) == length(W2)) ||
        throw(DimensionMismatch("MPO 加法要求长度相等（对标 MPSKit check_length）"))
    return PeriodicVector([_naive_sum_tensor(W1[ℓ], W2[ℓ]) for ℓ in 1:length(W1)])
end

"""
    exact_hadamard(a1, a2) -> PeriodicVector{Array{T,3}}

朴素精确 Hadamard 积（element-wise 乘积的无限 MPS 推广）：逐 site 虚拟腿 zip、
物理腿共享，`A12[(a,c), s, (b,e)] = a1[a,s,b]·a2[c,s,e]`。要求长度与逐 site
物理维相等（结果物理维不变，键维 = 两键维乘积）。周期 trace 因子化给出
**波形逐点乘积**：`dense(exact_hadamard(a1,a2)) = dense(a1) .* dense(a2)`。
输入/输出均为 `PeriodicVector{Array}` 张量串，不做任何规范化。**仅供 debug**。
"""
function exact_hadamard(ψ1::AbstractVector{<:Array{T,3}}, ψ2::AbstractVector{<:Array{T,3}}) where {T}
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard 要求长度相等"))
    all(size(ψ1[ℓ], 2) == size(ψ2[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard 要求逐 site 物理维相等"))
    return PeriodicVector([_naive_hadamard_tensor(ψ1[ℓ], ψ2[ℓ]) for ℓ in 1:length(ψ1)])
end
