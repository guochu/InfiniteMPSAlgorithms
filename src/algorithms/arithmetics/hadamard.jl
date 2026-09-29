# ---------------- iterative hadamard (infinite-MPS generalization of the Hadamard/Schur product) ----------------
#
# Elementwise waveform product `c₁₂ = c₁ .* c2`: virtual legs zipped per site,
# physical leg shared (kernel `_naive_hadamard_tensor` in states/linalg.jl);
# the physical dimension is unchanged and the bond dimension becomes the product
# of the two. The iterative version runs the factorized zip engine
# (`_vomps_zip_sweeps`/`_idmrg_zip_sweeps` on the `ZipKet` target — the fused
# zip tensors are never materialized, the environment/本地映射直接消费
# (ψ1, ψ2) 因子对). The strict compression-free Hadamard
# product lives in states/linalg.jl (`hadamard(::DenseIMPS, ::DenseIMPS)`);
# there is no `CanonicalIMPS` method — convert explicitly first.

"""
    hadamard(ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> y::CanonicalIMPS

Compute-on-the-fly variational compression of the Hadamard/Schur product to
the bond dimension `alg.D`: the zip target is consumed in **factorized form**
(the (ψ1, ψ2) tensor pairs go straight into the environment/局部映射收缩 — the
fused zip tensors are never materialized) and
variationally compressed with the positional algorithm
object `alg` (VOMPS/IDMRG), starting from the deterministic
`svdguess_hadamard` initial state. Convergence is judged by the Galerkin
residual alone (no overlap is computed — same contract as `mult`/`compress`).

The internal [`_hadamard`](@ref) additionally returns the sweep count
(`(y, iters)`); the exported wrapper discards it.
"""
hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    first(_hadamard(ψ1, ψ2, alg, nothing; D = alg.D))

function _hadamard(ψ1::CanonicalIMPS, ψ2::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    (length(ψ1) == length(ψ2)) ||
        throw(DimensionMismatch("hadamard requires equal lengths"))
    all(size(ψ1.AL[ℓ], 2) == size(ψ2.AL[ℓ], 2) for ℓ in 1:length(ψ1)) ||
        throw(DimensionMismatch("hadamard requires equal per-site physical dimensions"))
    N = length(ψ1)
    # 因子化 zip target（compute-on-the-fly）：融合 zip 张量从不物化，环境与
    # 局部映射直接消费 (ψ1, ψ2) 因子对（先与环境收缩——键优先的显式分步 GEMM，
    # 见 `_zip_push_left`/`_mapAC_zip`）；C 的融合序 = `kron(C2, C1)` 与 zip
    # kernel 的 (ψ2 主, ψ1 次) 键序逐位对齐。
    ket = ZipKet(
        ℓ -> ψ1.AL[ℓ], ℓ -> ψ2.AL[ℓ],
        ℓ -> ψ1.AR[ℓ], ℓ -> ψ2.AR[ℓ],
        ℓ -> ψ1.AC[ℓ], ℓ -> ψ2.AC[ℓ],
        ℓ -> ψ1.C[ℓ],  ℓ -> ψ2.C[ℓ],
    )
    x0 = x0 === nothing ? svdguess_hadamard(ψ1, ψ2, D) : x0
    iters = Ref(0)
    y = if alg isa VOMPS
        _vomps_zip_sweeps(ket, x0, N; tol = alg.tol, maxiter = alg.maxiter,
                          verbosity = alg.verbosity, iters = iters)
    else
        _idmrg_zip_sweeps(ket, x0, N; tol = alg.tol, maxiter = alg.maxiter,
                          verbosity = alg.verbosity, iters = iters)
    end
    return y, iters[]
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
    hadamard!(out, ψ₁, ψ₂, alg::Union{VOMPS,IDMRG}) -> out

In-place [`hadamard`](@ref): `out` is the user-provided state to be optimized
as the initial guess. The target bond dimension is taken from the bond profile
of `out` (its bond profile is first brought to uniform `D = max_bonddim(out)`
with [`changebond!`](@ref)); `alg.D` is ignored. The optimized result is
written back into `out`.
"""
function hadamard!(out::CanonicalIMPS, ψ1::CanonicalIMPS, ψ2::CanonicalIMPS,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, _ = _hadamard(ψ1, ψ2, alg, out; D = D)
    return _copyinto!(out, y)
end
# ---------------- factorized zip engine (hadamard channel; bond-first contractions) ----------------
#
# hadamard 的惰性 target：`Ket[(c,a), s, (e,b)] = A2[c,s,e]·A1[a,s,b]`（ψ2 键为
# 主指标、ψ1 为次指标，与 `kron(A2, A1)` 融合序一致；物理腿 s 是 below/A1/A2
# 三方共享的收缩边）。经由 LazyKet 闭包的做法每次调用都先物化 (D1·D2, d, D1·D2)
# 的融合张量、再与环境收缩——等价于「先做乘法再压缩」的最差收缩路径（中间张量
# O((D1·D2)²·d)）。因子化引擎把 (A1, A2) 对一路保留到环境/局部映射的收缩里：
# 共享物理腿按站点分片（per-s 片 GEMM，同 `_naive_hadamard_tensor` 的 kron 方
# 案）、键指标优先——最大中间张量 O(Dx·D1·D2·d)，融合 zip 张量从不落地。

"""
    ZipKet(ALf1, ALf2, ARf1, ARf2, ACf1, ACf2, Cf1, Cf2)

因子化 zip（Hadamard/Schur 乘积）惰性 target：`Ket[(c,a), s, (e,b)] =
A2[c,s,e]·A1[a,s,b]`（ψ2 键为主指标、ψ1 为次指标；`AL·C = C·AR = AC` 逐点
成立，C 的融合序 = `kron(C2, C1)`）。环境与局部映射直接消费 `(A1, A2)` 因子
对——融合 zip 张量从不物化。
"""
struct ZipKet{A1,A2,B1,B2,C1,C2,D1,D2}
    ALf1::A1; ALf2::A2
    ARf1::B1; ARf2::B2
    ACf1::C1; ACf2::C2
    Cf1::D1;  Cf2::D2
end

"""
    _zip_push_left(L, below, A2, A1) -> Matrix

因子化 zip 的 identity 通道左推（环境 `L` 为 `(x 键, 融合键 (c·a))` 矩阵，
a 为快指标）：
`L′[bl′, (e·b)] = Σ conj(below[bl, s, bl′])·L[bl, (c·a)]·A2[c,s,e]·A1[a,s,b]`.
物理腿 s 为 below/A2/A1 三方共享边 ⇒ 按物理片收缩（per-s 切片 @tensor，同
`_naive_hadamard_tensor` 的 kron 方案）：最大中间张量 O(Dx·D1·D2·d)，融合
zip 张量从不物化。
"""
function _zip_push_left(L::AbstractMatrix, below::AbstractArray{Tb,3},
                        A2::AbstractArray{Ta,3}, A1::AbstractArray{T1,3}) where {Tb,Ta,T1}
    Dx = size(L, 1)
    bl′ = size(below, 3)
    D2, d = size(A2, 1), size(A2, 2)       # A2[c, s, e]
    D1 = size(A1, 1)                       # A1[a, s, b]
    T = promote_type(eltype(L), eltype(below), eltype(A2), eltype(A1))
    L3 = reshape(L, Dx, D1, D2)            # (bl, a, c)：融合 (c·a) 的 a 为快指标
    belowC = conj(below)                   # (bl, s, bl′)
    W = zeros(T, D2, D1, bl′)              # 各 s 片累加：(e, b, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl, bl′)
        A1s = @view A1[:, s, :]            # (a, b)
        A2s = @view A2[:, s, :]            # (c, e)
        @tensor W[e, b, bl′] += L3[bl, a, c] * A2s[c, e] * A1s[a, b] * below_s[bl, bl′]
    end
    return reshape(permutedims(W, (3, 2, 1)), bl′, D1 * D2)      # (bl′, (e·b))：b 为快指标
end

"""
    _zip_push_right(R, A2, A1, below) -> Matrix

因子化 zip 的 identity 通道右推（环境 `R` 为 `(融合键 (e·b), x 键)` 矩阵，
b 为快指标）：
`R′[(c·a), bl′] = Σ R[(e·b), bl]·A2[c,s,e]·A1[a,s,b]·conj(below[bl′, s, bl])`
（below = x.AR：第一维为新键、第三维为旧键）。物理腿按物理片收缩（per-s
切片 @tensor），最大中间张量 O(D2·D1·Dx·d)，融合 zip 张量从不物化。
"""
function _zip_push_right(R::AbstractMatrix, A2::AbstractArray{Ta,3},
                         A1::AbstractArray{T1,3}, below::AbstractArray{Tb,3}) where {Ta,T1,Tb}
    bl′ = size(below, 1)                   # 新 below 键（输出列）
    bl = size(below, 3)                    # 旧 below 键（R 的列）
    D2, d = size(A2, 1), size(A2, 2)       # A2ar[e, s, c]
    D1 = size(A1, 1)                       # A1ar[b, s, a]
    T = promote_type(eltype(R), eltype(below), eltype(A2), eltype(A1))
    R3 = reshape(R, D1, D2, bl)            # (b, e, bl)：GR 行 = b + (e-1)·D1，b 为快指标
    belowC = conj(below)                   # (bl′, s, bl)
    W = zeros(T, D1, D2, bl′)              # 各 s 片累加：(a, c, bl′)
    for s in 1:d
        below_s = @view belowC[:, s, :]    # (bl′, bl)
        A1s = @view A1[:, s, :]            # (a, b)：A1 的左键 a 为行
        A2s = @view A2[:, s, :]            # (c, e)：A2 的左键 c 为行
        @tensor W[a, c, bl′] += A2s[c, e] * A1s[a, b] * R3[b, e, bl] * below_s[bl′, bl]
    end
    return reshape(W, D1 * D2, bl′)                             # ((c, a), bl′)：a 为快指标
end

"""
    _mapAC_zip(GL, GR, A2ac, A1ac) -> Array{T,3}

因子化 zip 的 identity 通道局部 AC 映射：
`k[xL, p, xR] = Σ GL[xL, (c·a)]·A2ac[c,p,e]·A1ac[a,p,b]·GR[(e·b), xR]`.
物理腿 p 为两因子共享的开放指标（非收缩边），分两步 @tensor：键 c → 键 a →
融合右键 (e·b)，最大中间张量 O(Dx·D1·d·D2)，融合 zip AC 张量从不物化。
"""
function _mapAC_zip(GL::AbstractMatrix, GR::AbstractMatrix,
                    A2ac::AbstractArray{Ta,3}, A1ac::AbstractArray{T1,3}) where {Ta,T1}
    Dx = size(GL, 1)
    D2, d = size(A2ac, 1), size(A2ac, 2)   # A2ac[c, p, e]
    D1 = size(A1ac, 1)                     # A1ac[a, p, b]
    T = promote_type(eltype(GL), eltype(GR), eltype(A2ac), eltype(A1ac))
    GL3 = reshape(GL, Dx, D1, D2)          # (xL, a, c)：融合 (c·a) 的 a 为快指标
    GR3 = reshape(GR, D1, D2, size(GR, 2)) # (b, e, xR)：融合 (e·b) 的 b 为快指标
    # 步1（键 c）：Y[a, xL, p, e] = Σ_c GL3[xL, a, c]·A2ac[c, p, e]
    @tensor Y[a, xL, p, e] := GL3[xL, a, c] * A2ac[c, p, e]      # 中间 Dx·D1·d·D2
    # 步2a（键 a；p 逐片——p 为两因子共享的开放指标，@tensor 不支持批量共享，
    #      按物理片 batched GEMM）：Z[p, b, e, xL] = Σ_a Y[a, xL, p, e]·A1ac[a, p, b]
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
    xR = size(GR, 2)
    Zp = reshape(permutedims(Z, (1, 4, 2, 3)), d * Dx, D1 * D2)  # (p, xL, b, e)：列 = b + (e-1)·D1
    kR = Zp * reshape(GR3, D1 * D2, xR)                           # (d·D1, xR)
    return reshape(permutedims(reshape(kR, d, Dx, xR), (2, 1, 3)), Dx, d, xR)
end

"""
    _mapC_zip(GL, C2, C1, GR) -> Matrix

因子化 zip 的 identity 通道局部 C 映射：
`Cnew[xL, xR] = Σ GL[xL, (c·a)]·C2[c, e]·C1[a, b]·GR[(e·b), xR]`.
分两步 @tensor（键 c 与键 b），最大中间张量 O(Dx·D1·D2)，`kron(C2, C1)`
从不物化。
"""
function _mapC_zip(GL::AbstractMatrix, C2::AbstractMatrix,
                   C1::AbstractMatrix, GR::AbstractMatrix)
    D2, D1 = size(C2, 1), size(C1, 1)
    Dx = size(GL, 1)
    GL3 = reshape(GL, Dx, D1, D2)          # (xL, a, c)：融合 (c·a) 的 a 为快指标
    GR3 = reshape(GR, D1, D2, size(GR, 2)) # (b, e, xR)：融合 (e·b) 的 b 为快指标
    @tensor Y[a, xL, e] := GL3[xL, a, c] * C2[c, e]              # 中间 Dx·D1·D2
    @tensor S[a, e, xR] := C1[a, b] * GR3[b, e, xR]              # 中间 D1·D2·Dx
    @tensor Cnew[xL, xR] := Y[a, xL, e] * S[a, e, xR]
    return Cnew
end

"""
    _lazy_ternary_fixedpoints(x, ket::ZipKet; tol, krylovdim, maxiter, GL0, GR0) -> (GLs, GRs)

因子化 zip target 的 identity 通道固定点：环境推直接消费 `(A1, A2)` 因子对
（[`_zip_push_left`](@ref)/[`_zip_push_right`](@ref)）。`GL`/`GR` 为
`(x 键, 融合键 (c·a)/(e·b))` 矩阵。`GL0`/`GR0` 可选：用上一轮环境热启动
eigsolve（保持不动点在逐轮之间的连续性）。
"""
function _lazy_ternary_fixedpoints(x::CanonicalIMPS, ket::ZipKet;
                                   tol::Real = Defaults.tol,
                                   krylovdim::Int = Defaults.krylovdim,
                                   maxiter::Int = Defaults.maxiter,
                                   GL0::Union{Nothing,AbstractArray} = nothing,
                                   GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(x)
    T = scalartype(x)
    Dl = size(x.AL[1], 1)
    Dr = size(x.AR[1], 3)
    Da1 = size(ket.ALf1(1), 1) * size(ket.ALf2(1), 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Da1)
        for ℓ in 1:N
            GL = _zip_push_left(GL, x.AL[ℓ], ket.ALf2(ℓ), ket.ALf1(ℓ))
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Da1) : vec(copy(GL0))
    _, GL1 = _eigsolve(Tleft, v0L, 1, :LM; ishermitian = false, tol = tol,
                       krylovdim = krylovdim, maxiter = maxiter)
    TCL = promote_type(T, eltype(GL1[1]))
    GLs = Vector{Matrix{TCL}}(undef, N)
    GLs[1] = GL = reshape(GL1[1], Dl, Da1)
    for ℓ in 2:N
        GL = _zip_push_left(GL, x.AL[ℓ-1], ket.ALf2(ℓ-1), ket.ALf1(ℓ-1))
        GLs[ℓ] = GL
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da1, Dr)
        for ℓ in N:-1:1
            GR = _zip_push_right(GR, ket.ARf2(ℓ), ket.ARf1(ℓ), x.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da1 * Dr) : vec(copy(GR0))
    _, GRN = _eigsolve(Tright, v0R, 1, :LM; ishermitian = false, tol = tol,
                       krylovdim = krylovdim, maxiter = maxiter)
    TCR = promote_type(T, eltype(GRN[1]))
    GRs = Vector{Matrix{TCR}}(undef, N)
    GRs[N] = GR = reshape(GRN[1], Da1, Dr)
    for ℓ in N-1:-1:1
        GR = _zip_push_right(GR, ket.ARf2(ℓ+1), ket.ARf1(ℓ+1), x.AR[ℓ+1])
        GRs[ℓ] = GR
    end

    # 归一化（MPSKit 约定：GR Frobenius、GL 乘局部 overlap λ）
    for ℓ in 1:N
        GRs[ℓ] = GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC_zip(GLs[inext], ket.Cf2(ℓ), ket.Cf1(ℓ), GRs[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("factorized zip environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] = GLs[inext] ./ λ
    end
    return GLs, GRs
end

"因子化 zip target 的 per-site Galerkin 残差（语义同 `_galerkin_err`）。"
function _lazy_galerkin_err(x::CanonicalIMPS, ket::ZipKet, GLs, GRs, N)
    ϵ = 0.0
    for ℓ in 1:N
        k = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], k))
    end
    return ϵ
end

"因子化 zip 通道的环境重标定（MPSKit `normalize!` 语义：GR Frobenius 归一、
GL[ℓ+1] 按局部 C 通道 overlap λ 缩放，同 mult 组合通道的
`_normalize_ternary_envs!`）。"
function _normalize_lazy_zip_envs!(GLs, GRs, x::CanonicalIMPS, ket::ZipKet)
    N = length(x)
    for ℓ in 1:N
        GRs[ℓ] = GRs[ℓ] ./ norm(GRs[ℓ])
        Cnew = _mapC_zip(GLs[_mod1(ℓ + 1, N)], ket.Cf2(ℓ), ket.Cf1(ℓ), GRs[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("factorized zip idmrg sweep: local overlap λ = 0 at site $ℓ")
        GLs[_mod1(ℓ + 1, N)] = GLs[_mod1(ℓ + 1, N)] ./ λ
    end
    return GLs, GRs
end

"""
    _vomps_zip_sweeps(ket::ZipKet, x0, N; tol, maxiter, verbosity, iters) -> x

VOMPS template on the factorized zip (Hadamard) channel (mirroring
[`_vomps_sweeps`](@ref)): Jacobi-style rounds — all sites updated against the
same environments via `_mapAC_zip`/`_mapC_zip` → `regauge!` → `gauge_step!`
(dynamically adapted gauge tolerance) → environments re-solved warm-started
(`_lazy_ternary_fixedpoints`, dynamically adapted tolerance) → Galerkin
residual checked **after** the sweep. The fused zip tensors are never
materialized. `iters::Ref{Int}` optionally receives the sweep count.
"""
function _vomps_zip_sweeps(ket::ZipKet, x0::CanonicalIMPS, N::Int;
                           tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                           verbosity::Int = Defaults.verbosity,
                           iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    T0 = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T0, eltype(GLs[1])), x)
    T = eltype(x.AL[1])
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
    iter = 0
    for outer iter in 1:maxiter
        ALs = Vector{Array{T,3}}(undef, N)
        for ℓ in 1:N
            k = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
            ĉ = _mapC_zip(GLs[_mod1(ℓ + 1, N)], ket.Cf2(ℓ), ket.Cf1(ℓ), GRs[ℓ])
            ALs[ℓ] = regauge!(k, ĉ; alg = Defaults.alg_orth())
        end
        alg_gauge = updatetol(Defaults.alg_gauge(), iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
        alg_envs = updatetol(Defaults.alg_environments(), iter - 1, ϵ)
        GLs, GRs = _lazy_ternary_fixedpoints(x, ket; GL0 = GLs[1], GR0 = GRs[N],
                                             tol = alg_envs.tol)
        ϵ = _lazy_galerkin_err(x, ket, GLs, GRs, N)
        verbosity > 0 && _logiter(stdout, "VOMPS", iter, ϵ)
        ϵ ≤ tol && break
    end
    iters === nothing || (iters[] = iter)
    _global_normalize!(x)
    return x
end

"""
    _idmrg_zip_sweeps(ket::ZipKet, x0, N; tol, maxiter, verbosity, iters) -> x

IDMRG template on the factorized zip (Hadamard) channel (mirroring
[`_idmrg_sweeps`](@ref)): sequential Gauss–Seidel double sweep with on-the-fly
environment transfer (`_zip_push_left`/`_zip_push_right` through the fresh
`AL`/`AR` and the `(A1, A2)` factor pairs), `leftorth`/`rightorth` splits of
the normalized local projections, per-double-sweep environment rescaling
([`_normalize_lazy_zip_envs!`](@ref)), and center-matrix-drift convergence
`ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is rebuilt from
the `AR` string (MPSKit `MultilineMPS(ψ.AR)`). `iters::Ref{Int}` optionally
receives the sweep count.
"""
function _idmrg_zip_sweeps(ket::ZipKet, x0::CanonicalIMPS, N::Int;
                           tol::Real = Defaults.tol, maxiter::Int = Defaults.maxiter,
                           verbosity::Int = Defaults.verbosity,
                           iters::Union{Nothing,Base.RefValue{Int}} = nothing)
    T0 = promote_type(scalartype(x0), eltype(ket.ACf1(1)))
    x = copy(x0)
    GLs, GRs = _lazy_ternary_fixedpoints(x, ket)
    x = _promote_scalar(promote_type(T0, eltype(GLs[1])), x)
    ϵ = 2 * tol
    iter = 0
    for outer iter in 1:maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = leftorth(x.AC[ℓ], (1, 2), (3,))
            GLs[_mod1(ℓ + 1, N)] = _zip_push_left(GLs[ℓ], x.AL[ℓ],
                                                  ket.ALf2(ℓ), ket.ALf1(ℓ))
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = _mapAC_zip(GLs[ℓ], GRs[ℓ], ket.ACf2(ℓ), ket.ACf1(ℓ))
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = rightorth(x.AC[ℓ], (1,), (2, 3))
            GRs[_mod1(ℓ - 1, N)] = _zip_push_right(GRs[ℓ], ket.ARf2(ℓ),
                                                   ket.ARf1(ℓ), x.AR[ℓ])
        end
        # 环境重标定（MPSKit normalize!(envs, below, operator, above) 语义）
        _normalize_lazy_zip_envs!(GLs, GRs, x, ket)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        ϵ < tol && break
    end
    iters === nothing || (iters[] = iter)
    # 规范恢复：从 AR 重建混合规范（MPSKit MultilineMPS(ψ.AR; alg_gauge...)）
    alg_gauge = updatetol(Defaults.alg_gauge(), iter, ϵ)
    x = CanonicalIMPS([x.AR[ℓ] for ℓ in 1:N]; tol = alg_gauge.tol,
                      maxiter = alg_gauge.maxiter)
    _global_normalize!(x)
    return x
end
