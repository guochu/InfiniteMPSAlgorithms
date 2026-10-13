# ---------------- iterative hadamard (infinite-MPS generalization of the Hadamard/Schur product) ----------------
#
# Elementwise waveform product `c₁₂ = c₁ .* c2`: virtual legs zipped per site,
# physical leg shared (kernel `_naive_hadamard_tensor` in states/linalg.jl);
# the physical dimension is unchanged and the bond dimension becomes the product
# of the two. The iterative version shares the unified compression sweep engine
# （[`compression_sweeps!`](@ref)，arithmetics/envs.jl）
# ——乘积因子 (ψ1, ψ2) 直接进入环境/局部映射收缩，融合 zip 张量
# 从不物化）。The strict compression-free Hadamard
# product lives in states/linalg.jl (`⊙(::DenseIMPS, ::DenseIMPS)`);
# there is no `CanonicalIMPS` method — convert explicitly first.

"""
    hadamard(ψ₁, ψ₂, alg::CompressionAlgorithm) -> (y::CanonicalIMPS, envs, info)

Compute-on-the-fly variational compression of the Hadamard/Schur product to
the bond dimension `alg.D`: the zip target is consumed in **factorized form**
(the (ψ1, ψ2) tensor pairs go straight into the environment/局部映射收缩 — the
fused zip tensors are never materialized) and variationally compressed with
the positional algorithm object `alg` (VOMPS/IDMRG/IDMRG2), starting from the
deterministic `svdguess_hadamard` initial state（扫掠经
[`compression_sweeps!`](@ref)）. Convergence is judged by the Galerkin
residual alone (no overlap is computed — same contract as `mult`/`compress`).

The second return is the engine's final environments（zip 通道的
[`HadamardCache`](@ref)：bra = 被优化的乘积链）; the third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 扫掠轮数、`losses` 逐轮残差/漂移、
`converged` 收敛标志）。
"""
function hadamard(ψ1::AbstractInfiniteMPS, ψ2::AbstractInfiniteMPS, alg::CompressionAlgorithm)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    # 因子化 zip target（compute-on-the-fly）：融合 zip 张量从不物化，环境与
    # 局部映射直接消费 (ψ1, ψ2) 因子对（先与环境收缩——键优先的显式分步 GEMM，
    # 见 `_zip_push_left`/`_mapAC_zip`）；C 的融合序 = `kron(C2, C1)` 与 zip
    # kernel 的 (ψ2 主, ψ1 次) 键序逐位对齐。dense 输入经 getproperty 家族视图
    # 直接参与（不做规范转换），输出类型恒为 CanonicalIMPS。
    # IDMRG2 的初猜键维从 trunc 提取（无维数语义回退无损上界，同 compress）
    D = something(_alg_guess_dim(alg),
                  maximum(size(ψ1.AL[ℓ], 1) * size(ψ2.AL[ℓ], 1) for ℓ in 1:length(ψ1)))
    envs = HadamardCache(svdguess_hadamard(ψ1, ψ2, D), ψ1, ψ2,
                         alg.alg_environments)
    envs, info = compression_sweeps!(envs, alg)
    return envs.bra, envs, info
end

# ---------------- svdguess_hadamard (deterministic initial guess) & hadamard! (in-place) ----------------

"""
    svdguess_hadamard(ψ₁, ψ₂, D) -> CanonicalIMPS
    svdguess_hadamard(A1s::PeriodicVector{<:Array{T,3}},
                      A2s::PeriodicVector{<:Array{T,3}}, D) -> Vector{Array{T,3}}

Deterministic initial guess of the iterative [`hadamard`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_hadamard`): the pointwise product's site tensors
are generated one at a time, **with the streaming carry absorbed during
construction** (contracting into the zip inputs before fusing — the naive
product tensor is never materialized), and streamed right→left through a
truncating right-orthogonalization with bond cap `D`
([`_lazy_svd_guess`](@ref)); the ring's wrap bond is Schmidt-truncated at
site 1. Every output bond is ≤ `D`.

The bare-tensor method takes the site-tensor strings directly (e.g. `ψ.AL` /
`ψ.AR` of a [`CanonicalIMPS`](@ref)) and returns the right-gauge tensor
string — the low-level entry point for downstream packages; the
`CanonicalIMPS` method is a thin wrapper that re-canonicalizes its output
（`kwargs` 透传末端 `CanonicalIMPS` 构造器 → `gaugefix!`）.
"""
function svdguess_hadamard(ψ1::AbstractInfiniteMPS, ψ2::AbstractInfiniteMPS, D::Int;
                           kwargs...)
    return CanonicalIMPS(svdguess_hadamard(ψ1.AL, ψ2.AL, D); kwargs...)
end

function svdguess_hadamard(A1s::PeriodicVector{<:Array{T,3}},
                           A2s::PeriodicVector{<:Array{U,3}}, D::Int) where {T,U}
    (length(A1s) == length(A2s)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    N = length(A1s)
    all(size(A1s[ℓ], 2) == size(A2s[ℓ], 2) for ℓ in 1:N) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    # carry 在构造时吸收（kron(A2, A1) 的融合序：左腿 (c 慢, a 快)、右腿 (e 慢, b 快)）：
    # B'[(c,a), s, f] = Σ_{e,b} A2[c,s,e]·A1[a,s,b]·carry[(e,b), f]
    # 先用 A1 的 b 腿吸收 carry、再用 A2 的 e 腿收缩 —— 两个矩阵乘，不落地大张量
    site = (ℓ, carry) -> begin
        A1 = A1s[ℓ]; A2 = A2s[ℓ]
        carry === nothing && return _naive_hadamard_tensor(A1, A2)
        a, s, b = size(A1); c, _, e = size(A2)
        f = size(carry, 2)
        l4 = reshape(carry, b, e, f)                     # carry 行 = (e-1)·b + b：b 最快
        B3 = Array{promote_type(T, U, scalartype(carry)),4}(undef, c, a, s, f)
        @inbounds for k in 1:s
            # Y[a,e,f] = Σ_b A1[a,s,b]·l4[b,e,f]
            Y = reshape(view(A1, :, k, :) * reshape(l4, b, e * f), a, e, f)
            # B'[c,a,f] = Σ_e A2[c,s,e]·Y[a,e,f]
            Bk = view(A2, :, k, :) * reshape(permutedims(Y, (2, 1, 3)), e, a * f)
            B3[:, :, k, :] = reshape(Bk, c, a, f)
        end
        # 左腿 flatten 与 naive 一致：行 = (c-1)·a + a（c 慢 a 快）
        return reshape(permutedims(B3, (2, 1, 3, 4)), a * c, s, f)
    end
    return _lazy_svd_guess(site, N, D)
end

"""
    hadamard!(out, ψ₁, ψ₂, alg::CompressionAlgorithm) -> (out, envs, info)

In-place [`hadamard`](@ref): `out` is the user-provided state to be optimized
as the initial guess; its bond profile is first brought to uniform
`D = alg.D` with [`changebond!`](@ref). The optimized result is written back
into `out`——类型守卫：通道算术类型宽于 `out` 的标量类型时（如实 `out` 遇复
通道），结果无法原地表示，此时 `out` 不写回（仅 `changebond!` 预处理生效），
第一个返回值为提升后的更新 bra（`envs.bra`）本身。Returns `(out, envs, info)`
——`envs` 为引擎的最终环境（bra = 乘积链）、`info` 为
[`IterativeConvergenceInfo`](@ref)。
"""
function hadamard!(out::CanonicalIMPS, ψ1::AbstractInfiniteMPS, ψ2::AbstractInfiniteMPS,
                   alg::CompressionAlgorithm)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    D = _alg_guess_dim(alg)
    D === nothing || changebond!(out; D)
    envs = HadamardCache(out, ψ1, ψ2, alg.alg_environments)
    envs, info = compression_sweeps!(envs, alg)
    return _copyinto!(out, envs.bra), envs, info
end

# ---------------- zip（Hadamard）通道：HadamardCache 与统一引擎方法 ----------------
#
# hadamard 的惰性 target：`Ket[(c,a), s, (e,b)] = A2[c,s,e]·A1[a,s,b]`（ψ2 键为
# 主指标、ψ1 为次指标，与 `kron(A2, A1)` 融合序一致；物理腿 s 是 below/A1/A2
# 三方共享的收缩边）。与 MultCache 的接口统一：HadamardCache 持 (bra, ket1, ket2)
# 三槽、环境为 rank-3 `(below, a, c)`/`(b, e, below)`（与 MultCache 组合通道的
# `(below, wl1, wl2)`/`(wr1, wr2, below)` 同格式）；环境/局部映射直接消费
# (ψ1, ψ2) 因子对——共享物理腿按站点分片（per-s 切片 GEMM，同
# `_naive_hadamard_tensor` 的 kron 方案）、键指标优先，最大中间张量
# O(Dx·D1·D2·d)，融合 zip 张量从不落地。

"`_zip_push_left(L, below, A2, A1) -> L′`（rank-3 环境）：zip 通道 identity 左推
`L′[bl′, b, e] = Σ conj(below[bl, s, bl′])·L[bl, a, c]·A2[c,s,e]·A1[a,s,b]`。
物理腿 s 为三方共享边 ⇒ 按物理片收缩（per-s 切片 GEMM），最大中间张量
O(Dx·D1·D2·d)，融合 zip 张量从不物化。"
function _zip_push_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,3},
                        A2::AbstractArray{Ta,3}, A1::AbstractArray{T1,3}) where {TL,Tb,Ta,T1}
    bl′ = size(below, 3)
    d = size(A2, 2)                        # A2[c, s, e]：物理腿共享
    D1r = size(A1, 3)                      # A1 右键（输出 b 腿）
    D2r = size(A2, 3)                      # A2 右键（输出 e 腿）
    T = promote_type(scalartype(L), scalartype(below), scalartype(A2), scalartype(A1))
    belowC = conj(below)                   # (bl, s, bl′)
    W = zeros(T, D2r, D1r, bl′)            # 各 s 片累加：(e, b, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl, bl′)
        A1s = @view A1[:, s, :]            # (a, b)
        A2s = @view A2[:, s, :]            # (c, e)
        @tensor W[e, b, bl′] += L[bl, a, c] * A2s[c, e] * A1s[a, b] * below_s[bl, bl′]
    end
    return permutedims(W, (3, 2, 1))       # [bl′, b, e] = (below, ψ1 键, ψ2 键)：b 快
end

"`_zip_push_right(R, A2, A1, below) -> R′`（rank-3 环境）：zip 通道 identity 右推
`R′[a, c, bl′] = Σ R[b, e, bl]·A2[c,s,e]·A1[a,s,b]·conj(below[bl′, s, bl])`
（below = x.AR：第一维为新键、第三维为旧键）。物理腿按物理片收缩，最大中间
张量 O(D2·D1·Dx·d)，融合 zip 张量从不物化。"
function _zip_push_right(R::AbstractArray{TR,3}, A2::AbstractArray{Ta,3},
                         A1::AbstractArray{T1,3}, below::AbstractArray{Tb,3}) where {TR,Ta,T1,Tb}
    bl′ = size(below, 1)                   # 新 below 键（输出末维）
    bl = size(below, 3)                    # 旧 below 键
    D2, d = size(A2, 1), size(A2, 2)       # A2[c, s, e]
    D1 = size(A1, 1)                       # A1[a, s, b]
    T = promote_type(scalartype(R), scalartype(below), scalartype(A2), scalartype(A1))
    belowC = conj(below)                   # (bl′, s, bl)
    W = zeros(T, D1, D2, bl′)              # 各 s 片累加：(a, c, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl′, bl)
        A1s = @view A1[:, s, :]            # (a, b)
        A2s = @view A2[:, s, :]            # (c, e)
        @tensor W[a, c, bl′] += A2s[c, e] * A1s[a, b] * R[b, e, bl] * below_s[bl′, bl]
    end
    return W                               # [a, c, bl′] = (ψ1 键, ψ2 键, below)：a 快
end

"`_mapAC_zip(GL, GR, A2ac, A1ac) -> k`（rank-3 环境）：zip 通道局部 AC 投影
`k[xL, p, xR] = Σ GL[xL, a, c]·A2ac[c,p,e]·A1ac[a,p,b]·GR[b, e, xR]`。
物理腿 p 为两因子共享的开放指标（per-p 片 batched GEMM），键 c → 键 a →
右键 (b, e) 分步收缩，最大中间张量 O(Dx·D1·d·D2)，融合 zip AC 从不物化。"
function _mapAC_zip(GL::AbstractArray{Tg,3}, GR::AbstractArray{Tgr,3},
                    A2ac::AbstractArray{Ta,3}, A1ac::AbstractArray{T1,3}) where {Tg,Tgr,Ta,T1}
    Dx = size(GL, 1)
    D2, d = size(A2ac, 1), size(A2ac, 2)   # A2ac[c, p, e]：左键 + 物理
    D1 = size(A1ac, 1)                     # A1ac[a, p, b]：左键
    D2r = size(A2ac, 3)                    # A2ac 右键（e 腿；非均匀键下 ≠ D2）
    D1r = size(A1ac, 3)                    # A1ac 右键（b 腿；非均匀键下 ≠ D1）
    T = promote_type(scalartype(GL), scalartype(GR), scalartype(A2ac), scalartype(A1ac))
    # 步1（键 c）：Y[a, xL, p, e] = Σ_c GL[xL, a, c]·A2ac[c, p, e]
    @tensor Y[a, xL, p, e] := GL[xL, a, c] * A2ac[c, p, e]      # 中间 Dx·D1·d·D2r
    # 步2a（键 a；p 逐片——p 为两因子共享的开放指标，按物理片 batched GEMM）：
    # Z[p, b, e, xL] = Σ_a Y[a, xL, p, e]·A1ac[a, p, b]
    Y3 = reshape(Y, D1, Dx, d, D2r)                              # (a, xL, p, e)
    Z = zeros(T, d, D1r, D2r, Dx)
    for p in 1:d
        A1p = view(A1ac, :, p, :)                                # (a, b)
        Yp = @view Y3[:, :, p, :]                                # (a, xL, e)
        # GEMM → (b, (xL, e))，重排为 (b, e, xL) 后写入 Z[p, b, e, xL]
        Z[p, :, :, :] .= reshape(permutedims(
            reshape(transpose(A1p) * reshape(Yp, D1, Dx * D2r), D1r, Dx, D2r), (1, 3, 2)), D1r, D2r, Dx)
    end
    # 步2b（键 b, e）：k[xL, p, xR] = Σ Z·GR
    xR = size(GR, 3)
    Zp = reshape(permutedims(Z, (1, 4, 2, 3)), d * Dx, D1r * D2r)  # (p, xL, b, e)：列 = b + (e-1)·D1r
    kR = Zp * reshape(GR, D1r * D2r, xR)                           # (d·D1, xR)
    return reshape(permutedims(reshape(kR, d, Dx, xR), (2, 1, 3)), Dx, d, xR)
end

"`_mapC_zip(GL, C2, C1, GR) -> Cnew`（rank-3 环境）：zip 通道局部 C 投影
`Cnew[xL, xR] = Σ GL[xL, a, c]·C2[c, e]·C1[a, b]·GR[b, e, xR]`（kron(C2, C1)
从不物化）。"
function _mapC_zip(GL::AbstractArray{Tg,3}, C2::AbstractMatrix,
                   C1::AbstractMatrix, GR::AbstractArray{Tgr,3}) where {Tg,Tgr}
    @tensor Y[a, xL, e] := GL[xL, a, c] * C2[c, e]              # 中间 Dx·D1·D2
    @tensor S[a, e, xR] := C1[a, b] * GR[b, e, xR]              # 中间 D1·D2·Dx
    @tensor Cnew[xL, xR] := Y[a, xL, e] * S[a, e, xR]
    return Cnew
end

# ---------------- HadamardCache ----------------

"""
    HadamardCache(bra, ket1, ket2, lefts, rights)
    HadamardCache(below, ψ1, ψ2, alg; GL0, GR0) -> HadamardCache

zip（Hadamard/Schur 乘积）通道 `⟨below|zip(ψ1, ψ2)⟩` 的环境缓存——接口与
[`MultCache`](@ref) 统一（rank-3 环境 `(below, a, c)`/`(b, e, below)`，两个
因子的键腿分开存放）。固定点由 [`hadamard_fixedpoints`](@ref) 的 :LM 主本征对
解出（`alg` 提供 `tol`/`maxiter`），归一化同 MPSKit（GR Frobenius 归一、GL 按
局部 C 通道 overlap λ 缩放；`kron(C2, C1)` 从不物化）。
"""
struct HadamardCache{B<:CanonicalIMPS,K1<:AbstractInfiniteMPS,K2<:AbstractInfiniteMPS,T} <: CompressionEnvironments
    bra::B
    ket1::K1   # ψ1（次指标因子）
    ket2::K2   # ψ2（主指标因子）
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

function HadamardCache(below::CanonicalIMPS, ψ1::AbstractInfiniteMPS, ψ2::AbstractInfiniteMPS,
                       alg = Defaults.alg_environments();
                       GL0::Union{Nothing,AbstractArray} = nothing,
                       GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = hadamard_fixedpoints(below, ψ1, ψ2, alg; GL0, GR0)
    # 槽位提升到通道标量类型（环境 eltype）：[`compression_sweeps!`](@ref) 对缓存
    # bra 的原地演化恒在同型算术上进行
    T = scalartype(GLs[1])
    return HadamardCache(_promote_scalar(T, below), _promote_scalar(T, ψ1),
                         _promote_scalar(T, ψ2), GLs, GRs)
end

"""
    recalculate!(envs::HadamardCache, newbra, [alg_environments]) -> envs

Recompute the `⟨bra|zip(ψ1, ψ2)⟩` fixed points for the updated bra `newbra`
（因子不变，[`hadamard_fixedpoints`](@ref) 重解），边界 eigsolve 以当前存储的
`lefts[1]`/`rights[end]` 热启动（对标 MPSKit 的原地 `recalculate!`）。
`newbra` 与新不动点就地写回 `envs`（原地更新，返回 `envs` 本身）。**类型
守卫**：新环境 eltype 宽于当前存储时（实 cache 遇复不动点）返回全槽位
提升后的新 envs——调用方须以 `envs = recalculate!(...)` 接住返回值。
"""
function recalculate!(envs::HadamardCache, newbra::CanonicalIMPS,
                      alg_environments = Defaults.alg_environments())
    GLs, GRs = hadamard_fixedpoints(newbra, envs.ket1, envs.ket2,
                                    alg_environments;
                                    GL0 = envs.lefts[1], GR0 = envs.rights[end])
    TL, TR = eltype(GLs[1]), eltype(GRs[end])
    if !(TL <: eltype(envs.lefts[1]) && TR <: eltype(envs.rights[end]))
        T = promote_type(TL, TR)
        new = HadamardCache(_promote_scalar(T, envs.bra), _promote_scalar(T, envs.ket1),
                            _promote_scalar(T, envs.ket2), GLs, GRs)
        new.bra ≡ newbra || _copyinto!(new.bra, newbra)
        return new
    end
    copy!(envs.lefts, GLs)
    copy!(envs.rights, GRs)
    envs.bra ≡ newbra || _copyinto!(envs.bra, newbra)
    return envs
end

"zip 通道的 identity 通道固定点（rank-3 环境；环境核与 mult.jl 的
[`mixed_fixedpoints`](@ref) 统一——左右不动点由 :LM 主本征对经 [`fixedpoint`](@ref)
解出（`alg` 分派 `tol`/`maxiter`：NamedTuple / DynamicTol / KrylovKit 算法皆可），
复环境按解的实际 eltype 存放，`kron(C2, C1)` 从不物化）。"
function hadamard_fixedpoints(below::CanonicalIMPS, ψ1::AbstractInfiniteMPS, ψ2::AbstractInfiniteMPS,
                              alg = Defaults.alg_environments();
                              GL0::Union{Nothing,AbstractArray} = nothing,
                              GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(below)
    T = promote_type(scalartype(below), scalartype(ψ1), scalartype(ψ2))
    Dl = size(below.AL[1], 1)
    D1 = size(ψ1.AL[1], 1)
    D2 = size(ψ2.AL[1], 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, D1, D2)
        for ℓ in 1:N
            GL = _zip_push_left(GL, below.AL[ℓ], ψ2.AL[ℓ], ψ1.AL[ℓ])
        end
        return vec(GL)
    end
    # 冷启动随机初猜（MPSKit approximate 的 randomize! 对齐；可被 GL0/GR0 热启动覆盖）
    v0L = GL0 === nothing ? randn(T, Dl * D1 * D2) : vec(copy(GL0))
    # forcereal：实输入下解出的复边界强投实（InfiniteTEMPO 对齐）；复输入不起作用
    fc = _forcereal(alg) && T <: Real
    _, vL = fixedpoint(Tleft, v0L, :LM, alg)
    fc && (vL = _realify!(vL))
    # 复环境提升（MPSKit 对齐：环境按 eigsolve 返回的实际 eltype 存放）
    TCL = promote_type(T, scalartype(vL))
    GLs = Vector{Array{TCL,3}}(undef, N)
    GLs[1] = GL = reshape(vL, Dl, D1, D2)
    for ℓ in 2:N
        GLs[ℓ] = GL = _zip_push_left(GL, below.AL[ℓ-1], ψ2.AL[ℓ-1], ψ1.AL[ℓ-1])
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, D1, D2, Dl)
        for ℓ in N:-1:1
            GR = _zip_push_right(GR, ψ2.AR[ℓ], ψ1.AR[ℓ], below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? randn(T, D1 * D2 * Dl) : vec(copy(GR0))
    _, vR = fixedpoint(Tright, v0R, :LM, alg)
    fc && (vR = _realify!(vR))
    TCR = promote_type(T, scalartype(vR))
    GRs = Vector{Array{TCR,3}}(undef, N)
    GRs[N] = GR = reshape(vR, D1, D2, Dl)
    for ℓ in N-1:-1:1
        GRs[ℓ] = GR = _zip_push_right(GR, ψ2.AR[ℓ+1], ψ1.AR[ℓ+1], below.AR[ℓ+1])
    end

    # 归一化（MPSKit 约定：GR Frobenius、GL 乘局部 overlap λ；kron(C2, C1) 不物化）
    for ℓ in 1:N
        GRs[ℓ] ./= norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC_zip(GLs[inext], ψ2.C[ℓ], ψ1.C[ℓ], GRs[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("zip environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] ./= λ
    end
    return GLs, GRs
end

# zip 通道的增量环境推进（rank-3 环境）
function transfer_leftenv!(envs::HadamardCache, x::CanonicalIMPS, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = _zip_push_left(envs.lefts[ℓm], x.AL[ℓm], envs.ket2.AL[ℓm],
                                   envs.ket1.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::HadamardCache, x::CanonicalIMPS, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = _zip_push_right(envs.rights[ℓp], envs.ket2.AR[ℓp],
                                     envs.ket1.AR[ℓp], x.AR[ℓp])
    return envs
end

# ---------------- zip_sweeps! 融合核（InfiniteTEMPO get_left_xy 机制） ----------------
#
# 每站一次融合 (GL·A2.AL·A1.AL)——共享物理腿 s 收缩进融合张量（zip 的物理腿
# 三方共享，map 中开放、push 中随 bra 收缩）——同一融合张量双用：map 侧折
# C1/C2 权重后与 GR 收缩出局部 AC，push 侧与 conj(新 bra) 单 GEMM 出下一站
# 环境。与 `_mapAC_zip`/`_zip_push_left` 同一收缩图的分解重排。

"左融合：`K0[xL, s, n, m] = Σ GL[xL, a, c]·A1.AL[a, s, n]·A2.AL[c, s, m]`
（Ket[(c,a), s, (e,b)] 的因子化部分，(n, m) = 两因子右键开放）。物理腿 s 为
A1/A2 共享的开放指标——不存在合法的二元收缩树 ⇒ 键 a 先单步收缩，再按物理
片 batched GEMM（同 `_mapAC_zip` 步 2a 方案），最大中间张量 O(Dx·D1·D2·d)。"
function _fuse_left(envs::HadamardCache, ℓ::Int)
    GL = envs.lefts[ℓ]                    # (xL, a, c)
    A2 = envs.ket2.AL[ℓ]                  # (c, s, m)
    A1 = envs.ket1.AL[ℓ]                  # (a, s, n)
    xL, d, n1, m2 = size(GL, 1), size(A1, 2), size(A1, 3), size(A2, 3)
    D2 = size(A2, 1)
    T = promote_type(scalartype(GL), scalartype(A2), scalartype(A1))
    # 步1（键 a，合法二元收缩）：Y[xL, c, s, n] = Σ_a GL[xL, a, c]·A1[a, s, n]
    @tensor Y[xL, c, s, n] := GL[xL, a, c] * A1[a, s, n]
    # 步2（键 c；s 逐片）：K0[xL, s, n, m] = Σ_c Y[xL, c, s, n]·A2[c, s, m]
    K0 = zeros(T, xL, d, n1, m2)
    for s in 1:d
        Ym = reshape(permutedims((@view Y[:, :, s, :]), (1, 3, 2)), xL * n1, D2)  # ((xL n), c)
        K0[:, s, :, :] .= reshape(Ym * (@view A2[:, s, :]), xL, n1, m2)
    end
    return K0
end

"map（L→R）：`AC[xL, p, xR] = Σ K0·C1[n, b]·C2[m, e]·GR[b, e, xR]`（`_mapAC_zip`
的 `GL·A2.AC·A1.AC·GR` 经 AC = AL·C 分解重排）。"
function _local_AC_fused(K0, envs::HadamardCache, ℓ::Int)
    GR = envs.rights[ℓ]                   # (b, e, xR)
    K = _fold_bond(K0, Val(3), _bondC(envs.ket1, ℓ))    # n ← C1[n, b]
    K = _fold_bond(K, Val(4), _bondC(envs.ket2, ℓ))     # m ← C2[m, e]
    return @tensor AC[xL, p, xR] := K[xL, p, b, e] * GR[b, e, xR]
end

"push（L→R）：`lefts[ℓ+1][bl′, n, m] = Σ conj(x.AL[ℓ][bl, s, bl′])·K0[bl, s, n, m]`
（共享 (bl, s) 单 GEMM）。"
function _push_left_fused!(envs::HadamardCache, K0, x, ℓ::Int)
    below = x.AL[ℓ]                       # (bl, s, bl′)
    Bmat = adjoint(reshape(below, :, size(below, 3)))   # (bl′, bl·s)
    new = reshape(Bmat * reshape(K0, size(below, 1) * size(below, 2), :),
                  size(below, 3), size(K0, 3), size(K0, 4))             # (bl′, n, m)
    envs.lefts[_mod1(ℓ + 1, length(envs))] = new
    return envs
end

"右融合：`S0[m, n, s, xR] = Σ GR[b, e, xR]·A2.AR[m, s, e]·A1.AR[n, s, b]`。
物理腿 s 为 A1/A2 共享的开放指标 ⇒ 键 b 先单步收缩，再按物理片 batched
GEMM（同 `_mapAC_zip` 步 2a 方案），最大中间张量 O(D1·D2·Dx·d)。"
function _fuse_right(envs::HadamardCache, ℓ::Int)
    GR = envs.rights[ℓ]                   # (b, e, xR)
    A2 = envs.ket2.AR[ℓ]                  # (m, s, e)
    A1 = envs.ket1.AR[ℓ]                  # (n, s, b)
    m2, d, e2, n1 = size(A2, 1), size(A2, 2), size(A2, 3), size(A1, 1)
    xR = size(GR, 3)
    T = promote_type(scalartype(GR), scalartype(A2), scalartype(A1))
    # 步1（键 b，合法二元收缩）：X[n, s, e, bR] = Σ_b A1[n, s, b]·GR[b, e, bR]
    @tensor X[n, s, e, bR] := A1[n, s, b] * GR[b, e, bR]
    # 步2（键 e；s 逐片）：S0[m, n, s, bR] = Σ_e A2[m, s, e]·X[n, s, e, bR]
    S0 = zeros(T, m2, n1, d, xR)
    for s in 1:d
        Xm = reshape(permutedims((@view X[:, s, :, :]), (2, 1, 3)), e2, n1 * xR)  # (e, (n bR))
        S0[:, :, s, :] .= reshape((@view A2[:, s, :]) * Xm, m2, n1, xR)
    end
    return S0
end

"map（R→L）：`AC[xL, p, xR] = Σ GL·C1[a, n]·C2[c, m]·S0[m, n, p, xR]`（AC = C·AR
的左键 C 折入 GL）。"
function _local_AC_fusedR(S0, envs::HadamardCache, ℓ::Int)
    N = length(envs)
    GL = envs.lefts[ℓ]                    # (xL, a, c)
    G = _fold_bond(GL, Val(2), _bondC(envs.ket1, _mod1(ℓ - 1, N)))  # a ← C1[a, n]
    G = _fold_bond(G, Val(3), _bondC(envs.ket2, _mod1(ℓ - 1, N)))   # c ← C2[c, m]
    return @tensor AC[xL, p, xR] := G[xL, n, m] * S0[m, n, p, xR]
end

"两站 map（IDMRG2 zip 融合）：`Θ[xL, s1, s2, xR] = Σ K0·C1·C2·A1.AR[next]·
A2.AR[next]·GR`（`_local_AC2` 的 `GL·t2·t1·GR`（t = AC·AR 两站因子）经
K0 = GL·A1.AL·A2.AL 融合 + AC = AL·C 分解重排；K0 由 `_fuse_left(envs, pos)`
双用——map 与左推共享）。next 站两因子的物理腿 s2 共享开放（同 `_local_AC2`
的 batched 限制）⇒ 键 (n′, m′) 先单步折 C，再按 s2 片 GEMM，最大中间张量
O(Dx·D1·D2·Dx·d)。"
function _local_AC2_fused(K0, envs::HadamardCache, pos::Int)
    N = length(envs)
    next = _mod1(pos + 1, N)
    A1 = envs.ket1.AR[next]               # (n′, s2, b)
    A2 = envs.ket2.AR[next]               # (m′, s2, e)
    GR = envs.rights[next]                # (b, e, xR)
    K = _fold_bond(K0, Val(3), _bondC(envs.ket1, pos))   # n ← C1[n, n′]
    K = _fold_bond(K, Val(4), _bondC(envs.ket2, pos))    # m ← C2[m, m′]
    # 步1（键 b，合法二元收缩）：R1[n′, s2, e, xR] = Σ_b A1[n′, s2, b]·GR[b, e, xR]
    @tensor R1[n′, s2, e, xR] := A1[n′, s2, b] * GR[b, e, xR]
    # 步2（键 e；s2 逐片）：R2[n′, m′, s2, xR] = Σ_e R1[n′, s2, e, xR]·A2[m′, s2, e]
    # 步3：Θ[:, :, s2, :] = Σ_{n′, m′} K·R2（K 行序 (xL·s1)、R2 行序 (n′·m′)）
    xL, d = size(K, 1), size(K, 2)
    n2, m2, e2, xR = size(A1, 1), size(A2, 1), size(A2, 3), size(GR, 3)
    T = promote_type(scalartype(K), scalartype(R1), scalartype(A2))
    Θ = zeros(T, xL, d, d, xR)
    Km = reshape(K, xL * d, n2 * m2)
    for s2 in 1:d
        R1m = reshape(permutedims((@view R1[:, s2, :, :]), (1, 3, 2)), n2 * xR, e2)  # ((n′ xR), e)
        R2 = reshape(R1m * transpose((@view A2[:, s2, :])), n2, xR, m2)             # (n′, xR, m′)
        R2m = reshape(permutedims(R2, (1, 3, 2)), n2 * m2, xR)                      # ((n′ m′), xR)
        Θ[:, :, s2, :] .= reshape(Km * R2m, xL, d, xR)
    end
    return Θ
end

"两站 map（IDMRG2 zip 融合，R→L）：`Θ = GL·A1.AL[pos]·A2.AL[pos]·C1·C2·S0`
（`_local_AC2_fused` 的镜像：融合在 next 站——S0 由 `_fuse_right(envs, next)`
双用，map 与右推共享；窗口 C 权重折入 S0 的 (m, n) 键腿）。pos 站两因子物理腿
s1 共享开放 ⇒ 键 (n′, m′) 先两步收缩，再按 s1 片 GEMM。"
function _local_AC2_fusedR(S0, envs::HadamardCache, pos::Int)
    A1a = envs.ket1.AL[pos]               # (a, s1, n′)
    A2a = envs.ket2.AL[pos]               # (c, s1, m′)
    GL = envs.lefts[pos]                  # (xL, a, c)
    S = _fold_bond(S0, Val(1), _bondCT(_bondC(envs.ket2, pos)))   # m ← C2ᵀ[m′, m]
    S = _fold_bond(S, Val(2), _bondCT(_bondC(envs.ket1, pos)))    # n ← C1ᵀ[n′, n]
    # 步1（键 a，合法二元收缩，4 腿，s1 批量）：Z[xL, c, s1, n′] = Σ_a GL·A1a
    @tensor Z[xL, c, s1, n′] := GL[xL, a, c] * A1a[a, s1, n′]
    # 步2（键 (c, n′, m′)；s1 逐片——s1 为两因子共享开放指标，与
    # `_local_AC2_fused` 的 per-s 方案镜像）：
    #   Θs[xL, s2, xR] = Σ_{c, n′, m′} Z[:, s1, :]·A2a[:, s1, :]·S
    xL, d, n2 = size(Z, 1), size(Z, 3), size(Z, 4)
    m2, s2d, xR = size(S, 1), size(S, 3), size(S, 4)
    T = promote_type(scalartype(Z), scalartype(S), scalartype(A2a))
    Θ = zeros(T, xL, d, s2d, xR)
    for s1 in 1:d
        Zs = @view Z[:, :, s1, :]                       # (xL, c, n′)
        A2s1 = @view A2a[:, s1, :]                      # (c, m′)
        @tensor Θs[xL, s2, xR] := Zs[xL, c, n′] * A2s1[c, m′] * S[m′, n′, s2, xR]
        Θ[:, s1, :, :] = Θs
    end
    return Θ
end

"push（R→L）：`rights[ℓ-1][n, m, bl′] = Σ conj(x.AR[ℓ][bl′, s, bl])·
S0[n, m, s, bl]`（共享 (s, bl) 单 GEMM）。"
function _push_right_fused!(envs::HadamardCache, S0, x, ℓ::Int)
    below = x.AR[ℓ]                       # (bl′, s, bl)
    Bmat = reshape(conj(below), size(below, 1), :)       # (bl′, s·bl)：bra 共轭（同 general push）
    Smat = reshape(permutedims(S0, (3, 4, 1, 2)), :, size(S0, 1) * size(S0, 2))  # (s·bl, n·m)
    new = reshape(Bmat * Smat, size(below, 1), size(S0, 1), size(S0, 2))         # (bl′, m, n)
    envs.rights[_mod1(ℓ - 1, length(envs))] = permutedims(new, (3, 2, 1))        # (n, m, bl′)
    return envs
end

"zip 通道的环境重标定（MPSKit `normalize!` 语义：GR Frobenius 归一、GL[ℓ+1]
按局部 C 通道 overlap λ 缩放；kron(C2, C1) 从不物化）。"
function normalize_envs!(envs::HadamardCache, x::CanonicalIMPS)
    N = length(envs)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _local_C(envs, ℓ)
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("zip idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"压缩通道的统一局部映射入口（[`compression_sweeps!`](@ref) 用；全部输入由缓存
持有）：`_local_AC(envs, ℓ)` 为 site ℓ 的 AC 局部投影、`_local_C(envs, ℓ)` 为
bond ℓ 的 C 局部投影（zip 通道 = `_mapAC_zip`/`_mapC_zip`）。"
function _local_AC(envs::HadamardCache, ℓ::Int)
    return _mapAC_zip(leftenv(envs, ℓ), rightenv(envs, ℓ),
                      envs.ket2.AC[ℓ], envs.ket1.AC[ℓ])
end

"`_local_AC2(envs, pos)`：`(pos, pos+1)` 窗口的两站局部映射（[`IDMRG2`](@ref)
用；rank-3 环境 `(below, a, c)`/`(b, e, below)`）。两因子的两站张量
（`:ACAR` 形态，物理腿保留）先各自融合，再经 GL/GR 夹闭——共享物理腿
`s1`/`s2` 是 bra 网络的开放指标（同 `_mapAC_zip` 的单站约定），窗口精确
变分最优，输出 rank-4 `(xL, s1, s2, xR)` 供 `_svd2!` 分割。"
function _local_AC2(envs::HadamardCache, pos::Int)
    N = length(envs)
    next = _mod1(pos + 1, N)
    @tensor t1[a, s1, s2, b] := envs.ket1.AC[pos][a, s1, m] * envs.ket1.AR[next][m, s2, b]
    @tensor t2[c, s1, s2, e] := envs.ket2.AC[pos][c, s1, m] * envs.ket2.AR[next][m, s2, e]
    GL, GR = leftenv(envs, pos), rightenv(envs, next)
    @tensor X[xL, a, s1, s2, cr] := GL[xL, a, c] * t2[c, s1, s2, cr]
    # 因子 1 的 a 腿收缩：共享物理腿 (s1, s2) 是双方重复指标，@tensor 不支持
    # batched 形式——按物理片切片 GEMM（同 `_mapAC_zip` 的单站方案）
    Dx, Da, Dbr, Dcr = size(X, 1), size(t1, 1), size(t1, 4), size(t2, 4)
    d = size(t1, 2)
    T = promote_type(scalartype(X), scalartype(t1))
    Y = zeros(T, Dx, d, d, Dbr, Dcr)
    @inbounds for s1 in 1:d, s2 in 1:d
        Xp = @view X[:, :, s1, s2, :]                    # (xL, a, cr)
        t1p = @view t1[:, s1, s2, :]                     # (a, b)
        Y[:, s1, s2, :, :] .= @tensor Ys[xL, b, cr] := Xp[xL, a, cr] * t1p[a, b]
    end
    return @tensor Θ[xL, s1, s2, xR] := Y[xL, s1, s2, b, cr] * GR[b, cr, xR]
end

function _local_C(envs::HadamardCache, ℓ::Int)
    return _mapC_zip(leftenv(envs, _mod1(ℓ + 1, length(envs))),
                     envs.ket2.C[ℓ], envs.ket1.C[ℓ], rightenv(envs, ℓ))
end

"扫掠 finalize 回调的通道目标（[`compression_sweeps!`](@ref) 用）：zip 通道 =
ψ2（主指标因子）。"
_finalize_target(envs::HadamardCache) = envs.ket2

"zip 通道的最大逐站 Galerkin 残差（语义同 `_galerkin(AL, AC_map)`：
local-map 输出在当前态 `x.AL` 切空间上的正交分量范数）。"
function calc_galerkin(envs::HadamardCache, x::CanonicalIMPS)
    N = length(envs)
    ϵ = 0.0
    for ℓ in 1:N
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], _local_AC(envs, ℓ)))
    end
    return ϵ
end

# zip 通道的统一扫掠引擎（`compression_sweeps!` 的 HadamardCache 方法）与
# hadamard/hadamard! 的装配见文件前部；OverlapsCache 版本见 compress.jl。
