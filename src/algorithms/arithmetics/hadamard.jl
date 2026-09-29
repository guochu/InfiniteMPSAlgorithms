# ---------------- iterative hadamard (infinite-MPS generalization of the Hadamard/Schur product) ----------------
#
# Elementwise waveform product `c₁₂ = c₁ .* c2`: virtual legs zipped per site,
# physical leg shared (kernel `_naive_hadamard_tensor` in states/linalg.jl);
# the physical dimension is unchanged and the bond dimension becomes the product
# of the two. The iterative version runs the zip channel's own variational
# engines（`_zip_vomps_sweeps`/`_zip_idmrg_sweeps`，环境缓存为
# `HadamardCache`——乘积因子 (ψ1, ψ2) 直接进入环境/局部映射收缩，融合 zip 张量
# 从不物化）。The strict compression-free Hadamard
# product lives in states/linalg.jl (`hadamard(::DenseIMPS, ::DenseIMPS)`);
# there is no `CanonicalIMPS` method — convert explicitly first.

"""
    hadamard(ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> (y::CanonicalIMPS, envs, info)

Compute-on-the-fly variational compression of the Hadamard/Schur product to
the bond dimension `alg.D`: the zip target is consumed in **factorized form**
(the (ψ1, ψ2) tensor pairs go straight into the environment/局部映射收缩 — the
fused zip tensors are never materialized) and
variationally compressed with the positional algorithm
object `alg` (VOMPS/IDMRG), starting from the deterministic
`svdguess_hadamard` initial state. Convergence is judged by the Galerkin
residual alone (no overlap is computed — same contract as `mult`/`compress`).

The second return is the engine's final environments（zip 通道的
[`HadamardCache`](@ref)，对返回态的射线解出）; the third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 扫掠轮数、`losses` 逐轮残差/漂移、
`converged` 收敛标志）。
"""
hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    _hadamard(ψ1, ψ2, alg, nothing; D = alg.D)

function _hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    # 因子化 zip target（compute-on-the-fly）：融合 zip 张量从不物化，环境与
    # 局部映射直接消费 (ψ1, ψ2) 因子对（先与环境收缩——键优先的显式分步 GEMM，
    # 见 `_zip_push_left`/`_mapAC_zip`）；C 的融合序 = `kron(C2, C1)` 与 zip
    # kernel 的 (ψ2 主, ψ1 次) 键序逐位对齐。
    x0 = x0 === nothing ? svdguess_hadamard(ψ1, ψ2, D) : x0
    y, envs, info = if alg isa VOMPS
        _zip_vomps_sweeps(ψ2, ψ1, x0, alg)
    else
        _zip_idmrg_sweeps(ψ2, ψ1, x0, alg)
    end
    return y, envs, info
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
`CanonicalIMPS` method is a thin wrapper that re-canonicalizes its output.
"""
function svdguess_hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, D::Int)
    return CanonicalIMPS(svdguess_hadamard(ψ1.AL, ψ2.AL, D))
end

function svdguess_hadamard(A1s::PeriodicVector{<:Array{T,3}},
                           A2s::PeriodicVector{<:Array{T,3}}, D::Int) where {T}
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
        B3 = Array{promote_type(eltype(A1), eltype(carry)),4}(undef, c, a, s, f)
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
    hadamard!(out, ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)

In-place [`hadamard`](@ref): `out` is the user-provided state to be optimized
as the initial guess. The target bond dimension is taken from the bond profile
of `out` (its bond profile is first brought to uniform `D = max_bonddim(out)`
with [`changebond!`](@ref)); `alg.D` is ignored. The optimized result is
written back into `out`. Returns `(out, envs, info)`——`envs` 为引擎的最终环境、
`info` 为 [`IterativeConvergenceInfo`](@ref)。
"""
function hadamard!(out::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, envs, info = _hadamard(ψ1, ψ2, alg, out; D = D)
    return _copyinto!(out, y), envs, info
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
    D2, d = size(A2, 1), size(A2, 2)       # A2[c, s, e]
    D1 = size(A1, 1)                       # A1[a, s, b]
    T = promote_type(eltype(L), eltype(below), eltype(A2), eltype(A1))
    belowC = conj(below)                   # (bl, s, bl′)
    W = zeros(T, D2, D1, bl′)              # 各 s 片累加：(e, b, bl′)
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
    T = promote_type(eltype(R), eltype(below), eltype(A2), eltype(A1))
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
    D2, d = size(A2ac, 1), size(A2ac, 2)   # A2ac[c, p, e]
    D1 = size(A1ac, 1)                     # A1ac[a, p, b]
    T = promote_type(eltype(GL), eltype(GR), eltype(A2ac), eltype(A1ac))
    # 步1（键 c）：Y[a, xL, p, e] = Σ_c GL[xL, a, c]·A2ac[c, p, e]
    @tensor Y[a, xL, p, e] := GL[xL, a, c] * A2ac[c, p, e]      # 中间 Dx·D1·d·D2
    # 步2a（键 a；p 逐片——p 为两因子共享的开放指标，按物理片 batched GEMM）：
    # Z[p, b, e, xL] = Σ_a Y[a, xL, p, e]·A1ac[a, p, b]
    Y3 = reshape(Y, D1, Dx, d, D2)                               # (a, xL, p, e)
    Z = zeros(T, d, D1, D2, Dx)
    for p in 1:d
        A1p = view(A1ac, :, p, :)                                # (a, b)
        Yp = @view Y3[:, :, p, :]                                # (a, xL, e)
        # GEMM → (b, (xL, e))，重排为 (b, e, xL) 后写入 Z[p, b, e, xL]
        Z[p, :, :, :] .= reshape(permutedims(
            reshape(transpose(A1p) * reshape(Yp, D1, Dx * D2), D1, Dx, D2), (1, 3, 2)), D1, D2, Dx)
    end
    # 步2b（键 b, e）：k[xL, p, xR] = Σ Z·GR
    xR = size(GR, 3)
    Zp = reshape(permutedims(Z, (1, 4, 2, 3)), d * Dx, D1 * D2)  # (p, xL, b, e)：列 = b + (e-1)·D1
    kR = Zp * reshape(GR, D1 * D2, xR)                            # (d·D1, xR)
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
struct HadamardCache{B<:CanonicalIMPS,K1<:CanonicalIMPS,K2<:CanonicalIMPS,T} <: Environments
    bra::B
    ket1::K1   # ψ1（次指标因子）
    ket2::K2   # ψ2（主指标因子）
    lefts::Vector{Array{T,3}}
    rights::Vector{Array{T,3}}
end

function HadamardCache(below::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
                       alg = Defaults.alg_environments();
                       GL0::Union{Nothing,AbstractArray} = nothing,
                       GR0::Union{Nothing,AbstractArray} = nothing)
    GLs, GRs = hadamard_fixedpoints(below, ψ1, ψ2, alg; GL0, GR0)
    return HadamardCache(below, ψ1, ψ2, GLs, GRs)
end

"`leftenv(envs, ℓ)` / `rightenv(envs, ℓ)`：zip 通道环境访问（HadamardCache 的
字段为 ket1/ket2，无 ket 槽）。"
leftenv(envs::HadamardCache, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs.ket1))]
rightenv(envs::HadamardCache, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs.ket1))]

"zip 通道的 identity 通道固定点（rank-3 环境；环境核与 mult.jl 的
[`mixed_fixedpoints`](@ref) 统一——左右不动点由 :LM 主本征对经 [`fixedpoint`](@ref)
解出（`alg` 分派 `tol`/`maxiter`：NamedTuple / DynamicTol / KrylovKit 算法皆可），
复环境按解的实际 eltype 存放，`kron(C2, C1)` 从不物化）。"
function hadamard_fixedpoints(below::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
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
    v0L = GL0 === nothing ? ones(T, Dl * D1 * D2) : vec(copy(GL0))
    _, vL = fixedpoint(Tleft, v0L, :LM, alg)
    # 复环境提升（MPSKit 对齐：环境按 eigsolve 返回的实际 eltype 存放）
    TCL = promote_type(T, eltype(vL))
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
    v0R = GR0 === nothing ? ones(T, D1 * D2 * Dl) : vec(copy(GR0))
    _, vR = fixedpoint(Tright, v0R, :LM, alg)
    TCR = promote_type(T, eltype(vR))
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
function transfer_leftenv!(envs::HadamardCache, x::CanonicalIMPS,
                           ket2::CanonicalIMPS, ket1::CanonicalIMPS, site::Int)
    N = length(ket1)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = _zip_push_left(envs.lefts[ℓm], x.AL[ℓm], ket2.AL[ℓm], ket1.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::HadamardCache, x::CanonicalIMPS,
                            ket2::CanonicalIMPS, ket1::CanonicalIMPS, site::Int)
    N = length(ket1)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = _zip_push_right(envs.rights[ℓp], ket2.AR[ℓp], ket1.AR[ℓp],
                                     x.AR[ℓp])
    return envs
end

"zip 通道的环境重标定（MPSKit `normalize!` 语义：GR Frobenius 归一、GL[ℓ+1]
按局部 C 通道 overlap λ 缩放；kron(C2, C1) 从不物化）。"
function _normalize_ternary_envs!(envs::HadamardCache, x::CanonicalIMPS,
                                  ket2::CanonicalIMPS, ket1::CanonicalIMPS)
    N = length(ket1)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _mapC_zip(leftenv(envs, _mod1(ℓ + 1, N)), ket2.C[ℓ], ket1.C[ℓ],
                         rightenv(envs, ℓ))
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("zip idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"zip 通道的最大逐站 Galerkin 残差（语义同 `_galerkin_err(operator::DenseIMPO, ...)`）。"
function _galerkin_err(ket2::CanonicalIMPS, ket1::CanonicalIMPS,
                       x::CanonicalIMPS, envs::HadamardCache)
    N = length(ket1)
    ϵ = 0.0
    for ℓ in 1:N
        k = _mapAC_zip(leftenv(envs, ℓ), rightenv(envs, ℓ), ket2.AC[ℓ], ket1.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k))
    end
    return ϵ
end

"""
    _zip_vomps_sweeps(ket2, ket1, x0, alg::VOMPS) -> (x, envs, info)

zip（Hadamard）通道的 VOMPS 模板（`hadamard(ψ1, ψ2, alg)` 的引擎；管道与
mpo·mps/mpo·mpo 版 [`_vomps_sweeps`](@ref) 完全一致）：localupdate
（`k = _mapAC_zip(...)`、`ĉ = _mapC_zip(...)` → `regauge!`）→ `gauge_step!` →
热启动环境重解 → 扫掠后检查 Galerkin 残差。乘积因子各自保持规范 ⇒ 无需
gauge twist，融合 zip 张量从不物化。返回 `(x, envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = [初始残差,
逐轮 Galerkin 残差...]、`converged` 收敛标志）。
"""
function _zip_vomps_sweeps(ket2::CanonicalIMPS, ket1::CanonicalIMPS,
                           x0::CanonicalIMPS, alg::VOMPS)
    N = length(ket1)
    x = copy(x0)
    envs = HadamardCache(x, ket1, ket2, alg.alg_environments)
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释）
    T = promote_type(scalartype(ket1), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = _galerkin_err(ket2, ket1, x, envs)
    iter = 0
    losses = [ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # 候选 AL 与 ket1.AC 同形，eltype 提升到通道标量类型 T）
        ALs = [similar(ket1.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            k = _mapAC_zip(leftenv(envs, ℓ), rightenv(envs, ℓ), ket2.AC[ℓ], ket1.AC[ℓ])
            ĉ = _mapC_zip(leftenv(envs, _mod1(ℓ + 1, N)), ket2.C[ℓ], ket1.C[ℓ],
                          rightenv(envs, ℓ))
            ALs[ℓ] = regauge!(k, ĉ; alg = alg.alg_orth)
        end
        # gauge: restore the global right gauge（动态容差）
        alg_g = updatetol(alg.alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（热启动 + 动态环境容差）
        alg_envs = updatetol(alg.alg_environments, iter - 1, ϵ)
        envs = HadamardCache(x, ket1, ket2, alg_envs;
                             GL0 = envs.lefts[1], GR0 = envs.rights[N])
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket2, envs)
        ϵ = _galerkin_err(ket2, ket1, x, envs)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        if ϵ ≤ alg.tol
            converged = true
            break
        end
    end
    _global_normalize!(x)
    return x, envs, IterativeConvergenceInfo(iter, losses, converged)
end

"""
    _zip_idmrg_sweeps(ket2, ket1, x0, alg::IDMRG) -> (x, envs, info)

zip（Hadamard）通道的 IDMRG 模板：Gauss–Seidel 双扫掠 + 即时环境推进 +
重标定（[`_normalize_ternary_envs!`](@ref) 的 zip 方法）+ C 漂移收敛；随后从
AR 串重建混合规范、环境对终态重解。返回 `(x, envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = 逐轮中心
矩阵漂移、`converged` 收敛标志）。
"""
function _zip_idmrg_sweeps(ket2::CanonicalIMPS, ket1::CanonicalIMPS,
                           x0::CanonicalIMPS, alg::IDMRG)
    N = length(ket1)
    x = copy(x0)
    # 初始环境：由初态解一次左右不动点，扫掠中只做增量 transfer 与重标定
    envs = HadamardCache(x, ket1, ket2, Defaults.alg_environments())
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释）
    T = promote_type(scalartype(ket1), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _mapAC_zip(leftenv(envs, ℓ), rightenv(envs, ℓ),
                                 ket2.AC[ℓ], ket1.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            transfer_leftenv!(envs, x, ket2, ket1, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _mapAC_zip(leftenv(envs, ℓ), rightenv(envs, ℓ),
                                 ket2.AC[ℓ], ket1.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            transfer_rightenv!(envs, x, ket2, ket1, ℓ - 1)
        end
        # 环境重标定
        _normalize_ternary_envs!(envs, x, ket2, ket1)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket2, envs)
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建混合规范，环境对终态重解
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = _rebuild([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
    envs = HadamardCache(x, ket1, ket2, Defaults.alg_environments())
    _global_normalize!(x)
    return x, envs, IterativeConvergenceInfo(iter, losses, converged)
end
