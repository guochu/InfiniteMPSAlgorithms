# ---------------- exact mixed canonicalization（移植自 InfiniteTEMPO） ----------------
#
# InfiniteTEMPO 的 `InfiniteOrthogonalize`（adt/util.jl）与 `mixedcanonicalize2!`
# （adt/orth/orth.jl，Reference: PHYSICAL REVIEW B 78, 155117）的 plain-array 移植：
# 先解出转移矩阵的左右主导边界本征对（η, Vl, Vr），谱截断 Cholesky 白化
# （chol_split），`tsvd(Y·X)` 给出闭合键的键谱并施加截断方案，随后一次前向
# QR（QRpos）扫掠 + 一次反向 SVD 扫掠完成正则化——两趟有限深度扫掠即收敛，
# 与 `states/ortho.jl` 的迭代 power-sweep（uniform_leftorth!/uniform_rightorth!）
# 互补。入口为 `gaugefix!(ψ, As, alg::InfiniteOrthogonalize)`。

"""
    InfiniteOrthogonalize(; trunc = alg_trunc(), alg_gauge = Defaults.alg_gauge(),
                          alg_environments = Defaults.alg_environments(),
                          alg_orth = Defaults.alg_orth(), verbosity = 0)

Exact mixed-canonicalization algorithm（移植自 InfiniteTEMPO 的同名结构；
Reference: PHYSICAL REVIEW B 78, 155117）：通过左右主导边界本征对的白化 +
单趟前向 QR / 反向 SVD 扫掠，把任意键可行的无限张量串精确正则化（非迭代
power sweep）。输出态恒归一（`‖ψ‖ = 1`；不设 `normalize` 开关，对齐本包
其他 gauge 函数）。

- `trunc`：键谱截断方案（[`TruncationScheme`](@ref)，作用于闭合键的
  `tsvd(Y·X)` 与反向扫掠的逐站 SVD）；
- `alg_environments`：边界本征对求解（Arnoldi）的容差与迭代上限
  （`Defaults.alg_environments()` 惯例：`(; tol, maxiter)` NamedTuple 或
  `DynamicTol` 包装）；
- `alg_gauge`：`(AR, C) → AL` 装配扫掠（`gaugefix!(; order = :L)` 的
  `uniform_leftorth!`）的迭代参数；
- `alg_orth`：前向 QR 扫掠与装配扫掠的正交化算法（默认 `QRpos()`）；
- `verbosity`：`≥ 2` 时打印左右主导本征值。
"""
@kwdef struct InfiniteOrthogonalize{T<:TruncationScheme, G, E, O} <: Algorithm
    trunc::T = alg_trunc()
    alg_gauge::G = Defaults.alg_gauge()
    alg_environments::E = Defaults.alg_environments()
    alg_orth::O = Defaults.alg_orth()
    verbosity::Int = 0
end

# ---- 辅助（InfiniteTEMPO adt/util.jl 的移植） ----

"`lmul!(1/tr(x), x)`：迹归一（边界本征矩阵的归一约定）。"
_normalize_trace!(x::AbstractMatrix) = lmul!(1 / tr(x), x)

_safesign(v) = iszero(v) ? oneunit(v) : v / abs(v)

"`lmul!(conj(safesign(argmax(abs, x))), x)`：按最大分量固定整体相位（确定性 +
厄米边界矩阵的相位代表元）。"
function _normalize_angle!(x::AbstractArray)
    v = argmax(abs, x)
    return lmul!(conj(_safesign(v)), x)
end

const _CHOL_SPLIT_TOL = 1.0e-12

"""
    _chol_split_tol(trunc) -> Real

Eigenvector-direction cutoff for [`_chol_split`](@ref), read from the **user
truncation scheme** (`InfiniteOrthogonalize.trunc`). The boundary matrices
`Vl`/`Vr` carry the chain's bond-space Gram structure: a small eigenvalue there
does **not** imply the corresponding bond direction is physically empty — for a
near-identity target `h ≈ I + ε·V` the perturbation direction sits at weight
~ε²·(overlap) (measured 9.7e-10 on the Rabi PT stepper at dt=1/512, whose bond
tensor weight is 0.0884), and dropping it silently projects the chain tensor
component on that bond away: the canonicalized output loses the bond entirely
(the old `10 * alg_environments.tol` cut — the environment *solver* tolerance —
collapsed the same stepper to bond 1 vs bond 2 at dt=0.01, with the trim
independent of `trunc.ϵ`). Schemes without a spectrum cutoff (dimension-only or
none) fall through to `0.0`; the caller clamps the result at
`_CHOL_SPLIT_TOL`, the numerical floor below which an eigen-direction is
genuinely unresolvable.
"""
_chol_split_tol(trunc::TruncateDimCutoff) = trunc.ϵ
_chol_split_tol(trunc::TruncateRelError) = trunc.ϵ
_chol_split_tol(::TruncationScheme) = 0.0

"""
    _chol_split(m, tol) -> Matrix

谱截断 Cholesky 因子（InfiniteTEMPO 的 `chol_split`）：`eigen(Hermitian(m))`
后丢弃 `≤ tol` 的本征值，返回 `Y = Diagonal(√evals)·V̂'`（`Y'Y ≈ m`，PSD 部分）。
非正定（负本征值超出 `_CHOL_SPLIT_TOL`）时告警。
"""
function _chol_split(m::AbstractMatrix{<:Number}, tol::Real)
    evals, evecs = eigen(Hermitian(m))
    k = length(evals) + 1
    for i in 1:length(evals)
        evals[i] > tol && (k = i; break)
    end
    # positivity check
    (maximum(-, view(evals, 1:k-1); init = 0.0) < _CHOL_SPLIT_TOL) ||
        @warn "input matrix is not positive (with negative eigenvalue $(argmax(-, view(evals, 1:k-1))))"
    return Diagonal(sqrt.(evals[k:end])) * evecs[:, k:end]'
end

"""
    _overlap_leading_boundaries(As; tol, maxiter, verbosity) -> (η, Vl, Vr)

周期转移矩阵的左右主导边界本征对（InfiniteTEMPO 的
`overlap_leading_boundaries`）：随机秩 1 PSD 初值出发，左边界为
`L ↦ push_env_left` 一周期的主导本征矩阵 `Vl`（迹归一），右边界为
`R ↦ push_env_right` 的 `Vr`（迹归一）；两者本征值一致（η，每周期），
不一致时告警。
"""
function _overlap_leading_boundaries(As::AbstractVector{<:Array{T,3}};
                                     tol::Real, maxiter::Int, verbosity::Int = 0) where {T}
    D = size(As[1], 1)
    tml = TransferMatrix(As, As; side = :left)
    tmr = TransferMatrix(As, As; side = :right)
    vl = randn(T, D, D)
    vr = randn(T, D, D)
    vl = vl * vl'
    vr = vr * vr'
    alg = KrylovKit.Arnoldi(; tol = tol, maxiter = maxiter, krylovdim = Defaults.krylovdim)
    λl, Vl = gauge_fixedpoint(tml, vec(vl), :LM, alg)
    λr, Vr = gauge_fixedpoint(tmr, vec(vr), :LM, alg)
    verbosity >= 2 && println("left/right leading eigenvalues: $λl, $λr")
    _normalize_angle!(Vl)
    _normalize_angle!(Vr)
    λl ≈ λr ||
        @warn "left and right dominate eigenvalues $λl and $λr mismatch"
    return λl, _normalize_trace!(reshape(Vl, D, D)), _normalize_trace!(reshape(Vr, D, D))
end

"""
    mixedcanonicalize2!(x::AbstractVector{<:Array{T,3}}, alg::InfiniteOrthogonalize)
        -> sv

InfiniteTEMPO `mixedcanonicalize2!`（Reference: PHYSICAL REVIEW B 78, 155117）
的 plain-array 移植：**原地**把张量串 `x`（`(wl, s, wr)`）正则化为右正则串，
返回逐站谱 `sv`（`sv[i]` 作用在 site `i` 的**左**键上，InfiniteTEMPO 的
`x.s` 约定；我们 `CanonicalIMPS.C` 的「键在 site ℓ 右侧」约定对应
`C[ℓ] = sv[ℓ+1]`）。

流程（对齐源实现）：
1. 主导边界本征对 `(η, Vl, Vr)`；`Y = chol_split(Vl)`（`Y'Y ≈ Vl`）、
   `X = chol_split(Vr)'`（`XX' ≈ Vr`）；
2. `U, S, V = tsvd(Y·X; trunc)`：闭合键键谱 + 截断方案；
3. `m = U'Y` 前向 QR（QRpos）扫掠 sites `1..N-1`，`m2 = XV'` 吸收进 site N；
4. 反向 SVD 扫掠 sites `N..2`：右正则化 + 谱归一（`normalize!(ss)`），
   `m = v·Diagonal(ss)` 逐站吸收进左邻；
5. site 1 左除 `Diagonal(S)`（把 `AL·C` 形式化为 `C_left·AR`）。输出恒归一：
   N = 1 时 `x[1]` 自带 `√η` 总尺度（η = 转移矩阵每周期主导本征值），先除掉；
   N > 1 时尺度已在反向扫掠的逐站谱归一中吸收。

键维上界为 `min(D, ⌊D·d⌋)` 类的可行性约束与
[`_makefullrank!`](@ref) 无关：不可行的键 profile（`Dr > Dl·d`）下 QR 产物
`Q` 为方阵、键维增长，不做删键处理。
"""
function mixedcanonicalize2!(x::AbstractVector{<:Array{T,3}},
                             alg::InfiniteOrthogonalize) where {T}
    N = length(x)
    # 白化的方向截断阈值取用户 trunc 方案（clamp 在数值下限）；**不得**用
    # alg_environments 的求解容差——那会把边界矩阵里 ~1e-9 的真实键方向当
    # 数值零丢掉（见 _chol_split_tol 的 docstring）
    tolchol = max(_chol_split_tol(alg.trunc), _CHOL_SPLIT_TOL)
    g = alg.alg_environments isa DynamicTol ? alg.alg_environments.alg : alg.alg_environments
    η, Vl, Vr = _overlap_leading_boundaries(x; tol = g.tol, maxiter = g.maxiter,
                                            verbosity = alg.verbosity)

    Y = _chol_split(Vl, tolchol)      # (r×D),  Y'Y ≈ Vl
    X = _chol_split(Vr, tolchol)'     # (D×r'), XX' ≈ Vr
    U, S, V, _ = tsvd(Y * X; trunc = alg.trunc)

    # 前向 QR 扫掠：m = U'Y 白化 + 旋转，逐站 leftorth（QRpos）
    m = U' * Y
    for i in 1:(N-1)
        @tensor xj[α, s, b] := m[α, a] * x[i][a, s, b]
        Q, m = leftorth(xj, (1, 2), (3,); alg = alg.alg_orth)
        x[i] = Q
    end
    m2 = X * V'
    @tensor xN[α, s, β] := m[α, a] * x[N][a, s, b] * m2[b, β]
    x[N] = xN

    # 反向 SVD 扫掠：右正则化 + 谱归一，m = v·Diagonal(ss) 吸收进左邻
    sv = Vector{Vector{Float64}}(undef, N)
    for i in N:-1:2
        v, ss, xj2, _ = tsvd(x[i], (1,), (2, 3); trunc = alg.trunc)
        x[i] = xj2
        normalize!(ss)
        sv[i] = ss
        m = v * Diagonal(ss)
        @tensor xm[a, s, β] := x[i-1][a, s, b] * m[b, β]
        x[i-1] = xm
    end

    # site 1：左除 Diagonal(S)（源实现的 tie(x[1], (1,2)) = reshape(x[1], wl, :)）；
    # 输出恒归一——N = 1 时 x[1] 自带 √η 总尺度（边界本征值 η），先除掉
    ηs = sqrt(real(η))
    if N == 1
        lmul!(1 / ηs, x[1])
        x[1] = reshape(Diagonal(S) \ reshape(x[1], size(x[1], 1), :), size(x[1]))
        normalize!(S)
    else
        normalize!(S)
        x[1] = reshape(Diagonal(S) \ reshape(x[1], size(x[1], 1), :), size(x[1]))
    end
    sv[1] = S
    return sv
end

"""
    gaugefix!(ψ::CanonicalIMPS, As, alg::InfiniteOrthogonalize) -> ψ

[`InfiniteOrthogonalize`](@ref) 的 `CanonicalIMPS` 入口：对张量串 `As`
（`(wl, s, wr)`，键维逐站可不同、闭合键须方形）执行精确混合正则化并写回
`ψ` 的四族。`mixedcanonicalize2!` 产出右正则串 `AR = x` 与逐站谱 `sv`
（`sv[1]` = 闭合键 N 的谱）后，以 `C₀ = Diagonal(sv[1])` 经
`gaugefix!(; order = :LR)` 装配四族——**全程无除法**（不采用
`AL = C₋·AR/C′` 的右除装配，谱有接近截断阈值的奇异值时条件数失控）：

- **L 趟**：QR 扫掠从 `(x, C₀)` 装配 `AL`/`C`（构造性左正交）；
- **R 趟**：从左正交的 `AL` 串经 LQ 扫掠重建 `AR`、重解 `C`——必要的收尾：
  `mixedcanonicalize2!` 的 site-1 收尾 `Diag(S)⁻¹·x[1]`（谱 bookkeeping）只
  在无截断时保持右正交，强截断下 `x[1]` 的右正交性破缺 O(err)，且该非正交
  使串的每周期转移尺度 λ ≠ 1——`:L` 单独装配的收敛判据（逐轮 C 的方向差）
  对该尺度盲，会留下 `ϵ_mixed ~ |√λ − 1|` 的缺口；R 趟的 LQ 重建使 `AR`
  构造性右正交、`C` 重解到固定点，三个正则误差全部回到 `alg_gauge.tol`
  （默认 1e-13）量级——**与截断强度无关**（截断不自洽只进射线精度）。

`alg_gauge` 的 `tol`/`maxiter` 与 `alg_orth` 为两趟扫掠的迭代参数。
"""
function gaugefix!(ψ::CanonicalIMPS, As, alg::InfiniteOrthogonalize)
    x = [copy(a) for a in As]
    sv = mixedcanonicalize2!(x, alg)
    copy!(ψ.AR, x)
    # 闭合键谱 C₀ = Diag(sv[1])（sv[i] 作用在 site i 的左键上，键 N = site 1 左键）；
    # :LR 两趟装配：L 趟 QR 装配 AL/C（输入 = mixedcanonicalize2! 的串 x），
    # R 趟 LQ 重建 AR/重解 C（修复强截断下 x[1] 的右正交性破缺，见 docstring）
    T = scalartype(ψ)
    g = alg.alg_gauge isa DynamicTol ? alg.alg_gauge.alg : alg.alg_gauge
    gaugefix!(ψ, ψ.AR, Matrix{T}(Diagonal(sv[1])); order = :LR,
              tol = g.tol, maxiter = g.maxiter, alg_orth = alg.alg_orth)
    return ψ
end

"""
    gaugefix!(W::CanonicalIMPO, As, alg::InfiniteOrthogonalize) -> W

[`InfiniteOrthogonalize`](@ref) 的 `CanonicalIMPO` 入口：`As` 为 rank-4
`(wl, u, wr, d)` MPO 张量串（自动取 [`vectorize`](@ref) 的 MPS 视图）或
rank-3 融合视图；在 MPS 视图 `(wl, u·d, wr)` 上正则化后经
[`devectorize`](@ref) 写回 `W` 的四族（同 `truncate!` 的 MPO 通道）。
"""
function gaugefix!(W::CanonicalIMPO, As, alg::InfiniteOrthogonalize)
    ψ = vectorize(W)
    Av = As isa AbstractVector{<:AbstractArray{<:Number,4}} ?
         vectorize(collect(As)) : collect(As)
    gaugefix!(ψ, Av, alg)
    W4 = devectorize(ψ)
    for ℓ in 1:length(W)
        W.AL[ℓ] = W4.AL[ℓ]
        W.AR[ℓ] = W4.AR[ℓ]
        W.C[ℓ] = ψ.C[ℓ]
        W.AC[ℓ] = W4.AC[ℓ]
    end
    return W
end
