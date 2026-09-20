# ---------------- 迭代 hadamard（Hadamard/Schur 积的无限 MPS 推广） ----------------
#
# 波形逐点乘积 `c₁₂ = c₁ .* c₂`：逐 site 虚拟腿 zip、物理腿共享
# （核 `_naive_hadamard_tensor` 见 arithmetics.jl），物理维不变、
# 键维 = 两键维乘积。压缩装配与 add 共用（add.jl 的
# _compress_ket / _algebra_result）。

"""
    hadamard(ψ₁, ψ₂; alg = VOMPS()) -> InfiniteCanonicalMPS

Hadamard/Schur 积的无限 MPS 推广（element-wise 乘积）：波形逐点乘积
`c₁₂[(s…)] = c₁[(s…)] · c₂[(s…)]`，**物理维不变**，键维变为两键维乘积。
朴素构造：逐 site 虚拟腿 zip、物理腿共享（理由同 `add` 的 AL 选择），
两条虚拟链独立使周期 trace 因子化，`tr(∏A12) = tr(∏AL₁)·tr(∏AL₂)` 精确。

输出键维 `D = bondD(alg.trunc)`；朴素构造键维 `≤ D` 时短路返回精确结果，
否则变分压缩（`alg` 支持 `VOMPS()` 与 `IDMRG()`）。

注意：朴素 zip 一般**不是**规范形式（两条虚拟链共享物理指标使正交性和式耦合），
构造器规范化后输出与 `c₁.*c₂` 相差一个构造器确定的正实标量（射线代表）；
需要严格逐点幅值的原始张量串时用 [`exact_hadamard`](@ref)。
"""
function hadamard(ψ1::InfiniteCanonicalMPS, ψ2::InfiniteCanonicalMPS;
                  alg::Union{VOMPS,IDMRG} = VOMPS())
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard 要求长度相等"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard 要求逐 site 物理维相等"))
    K = [_naive_hadamard_tensor(ψ1.AL[ℓ], ψ2.AL[ℓ]) for ℓ in 1:length(ψ1)]
    return _algebra_result(K, bondD(alg.trunc), alg)
end
