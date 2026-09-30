# ---------------- compress (bond-dimension reduction of a single chain, mirroring MPSKit's approximate) ----------------
#
# Find `out ≈ x` with `max_bonddim(out) = D`: the overlap-maximizing
# variational approximation of a *given* chain (the target is the chain itself,
# unlike add/mult/hadamard whose targets are naive algebraic constructions).
# The identity-channel engine (`OverlapCache` + `compression_sweeps!`) is
# defined in arithmetics/envs.jl；MPO 施加 / zip 通道的引擎（mult.jl /
# hadamard.jl）共用同一管道。

"""
    svdguess_compress(x, D) -> CanonicalIMPS
    svdguess_compress(ALs::PeriodicVector{<:Array{T,3}}, D) -> Vector{Array{T,3}}

Deterministic initial guess of the iterative [`compress`](@ref) (reference:
FiniteMPSAlgorithms' `svdguess_compress`): the bond-wise SVD truncation of the
input to `D` (accurate but costly compared to [`changebond!`](@ref) zero
padding). `CanonicalIMPO` input returns the MPS-view guess (`CanonicalIMPS`
on the doubled space).

The bare-tensor method accepts the site-tensor string directly (e.g.
`ψ.AL` / `ψ.AR` of a [`CanonicalIMPS`](@ref)) and returns a plain right-gauge
tensor string truncated to `D` (wrap bond Schmidt-truncated at site 1,
[`_lazy_svd_guess`](@ref)) — the low-level entry point for downstream
packages; the `CanonicalIMPS`/`CanonicalIMPO` methods are thin wrappers that
re-canonicalize its output（`kwargs` 透传末端 `CanonicalIMPS` 构造器 →
`gaugefix!`，如 `tol`/`maxiter`）.
"""
function svdguess_compress(x::CanonicalIMPS, D::Int; kwargs...)
    max_bonddim(x) ≤ D && return copy(x)
    return CanonicalIMPS(svdguess_compress(x.AL, D); kwargs...)
end

function svdguess_compress(x::CanonicalIMPO, D::Int; kwargs...)
    return svdguess_compress(vectorize(x), D; kwargs...)
end

"`svdguess_compress` 的 [`AbstractInfiniteMPS`](@ref) 泛型入口（`DenseIMPS` 等
非规范链：对 `.AL` 家族视图做流式 SVD 截断后混合规范化——初态恒为
`CanonicalIMPS`；`kwargs` 透传其构造器）。"
svdguess_compress(x::AbstractInfiniteMPS, D::Int; kwargs...) =
    CanonicalIMPS(svdguess_compress(x.AL, D); kwargs...)

"Low-level tensor-string entry: streaming SVD truncation of a bare `AL`/`AR`
string to bond cap `D` (wrap bond Schmidt-truncated at site 1). The carry
(the previous site's truncated left basis) is absorbed into the next site
tensor on the fly — the carry-ignoring identity closure would break bond
consistency between the SVD outputs."
function svdguess_compress(ALs::PeriodicVector{<:Array{T,3}}, D::Int) where {T}
    site = (ℓ, carry) -> begin
        carry === nothing && return ALs[ℓ]
        A = ALs[ℓ]
        # B[a, s, f] = Σ_bb A[a, s, bb]·carry[bb, f]（carry 的行 = 上一站 SVD 截断
        # 出的右键基，与本站右键收缩；reshape 矩阵乘，@tensor 不接受开索引双侧出现）
        return reshape(reshape(A, :, size(A, 3)) * carry,
                       size(A, 1), size(A, 2), size(carry, 2))
    end
    return _lazy_svd_guess(site, length(ALs), D)
end

"""
    compress(x::AbstractInfiniteMPS, alg::Union{VOMPS,IDMRG}) -> (x′::CanonicalIMPS, envs, info)
    compress(W::AbstractInfiniteMPO, alg::Union{VOMPS,IDMRG}) -> (x′::CanonicalIMPO, envs, info)

Bond-dimension-`alg.D` variational approximation of a single chain (mirroring
MPSKit's `approximate`: maximize the overlap between the compressed chain and
the input chain, fixed point = the best rank-`D` approximation in the
ring-trace fidelity sense). `DenseIMPS`/`CanonicalIMPS`（及 `DenseIMPO`/
`CanonicalIMPO`）皆可直接输入——dense 类型经 `getproperty` 家族视图直接作为
压缩目标（不做规范转换），输出类型恒为 `CanonicalIMPS`/`CanonicalIMPO`.
The default initial guess is
[`svdguess_compress`](@ref) (the input's own SVD truncation to `alg.D`). The
positional `alg` dispatches [`VOMPS`](@ref) (ALS sweeps) or [`IDMRG`](@ref)
(MPSKit sequential Gauss–Seidel sweeps with on-the-fly environment transfer),
which share the same fixed point.

The second return is the engine's final environments（纯重叠通道的
[`OverlapCache`](@ref)：bra = 被优化的压缩链、ket = 目标链；MPO 压缩在
`vectorize` 的 MPS 视图上演化，envs 持该 MPS 视图）; the third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 扫掠轮数、`losses` 逐轮残差/漂移、
`converged` 收敛标志）。
"""
function compress(ψ::AbstractInfiniteMPS, alg::Union{VOMPS,IDMRG})
    envs = OverlapCache(svdguess_compress(ψ, alg.D), ψ, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return envs.bra, envs, info
end

function compress(W::AbstractInfiniteMPO, alg::Union{VOMPS,IDMRG})
    # 目标 MPS 的 ray 必须取 **AL 家族**（左正则串）：其周期 trace 才是算符串的
    # wavefunction（相似变换不变）；AC 家族在张量间携带 C 加权
    # （AC[ℓ] = AL[ℓ]·C[ℓ]），周期 trace 是规范依赖的加权量，作为压缩目标会
    # 定义另一个变分问题（与 `mult!` 的因子化目标不等价）。
    # `vectorize`：CanonicalIMPO 逐家族携带规范数据（其 AL 家族正是该 ray 的
    # 左正则串）；DenseIMPO 为纯融合视图（不做规范转换）。
    ket = vectorize(W)
    envs = OverlapCache(svdguess_compress(ket, alg.D), ket, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return devectorize(envs.bra), envs, info
end

# ---------------- compress! (in-place) ----------------

"""
    compress!(out, ψ::AbstractInfiniteMPS, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)
    compress!(out, W::AbstractInfiniteMPO, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)

In-place [`compress`](@ref): `out` is the user-provided chain to be optimized
as the initial guess; its bond profile is first brought to uniform
`D = alg.D` with [`changebond!`](@ref). The target accepts `DenseIMPS`/
`CanonicalIMPS`（及 `DenseIMPO`/`CanonicalIMPO`，dense 类型经 `getproperty`
家族视图直接参与，不做规范转换）. The optimized result is written back
into `out`. Returns `(out, envs, info)`——`envs` 为引擎的最终环境（bra = 压缩
链）、`info` 为 [`IterativeConvergenceInfo`](@ref)。
"""
function compress!(out::CanonicalIMPS, ψ::AbstractInfiniteMPS,
                   alg::Union{VOMPS,IDMRG})
    changebond!(out; D = alg.D)
    envs = OverlapCache(out, ψ, alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return _copyinto!(out, envs.bra), envs, info
end

function compress!(out::CanonicalIMPO, W::AbstractInfiniteMPO,
                   alg::Union{VOMPS,IDMRG})
    changebond!(out; D = alg.D)
    envs = OverlapCache(vectorize(out), vectorize(W), alg.alg_environments)
    _, info = compression_sweeps!(envs, alg)
    return _copyinto!(out, devectorize(envs.bra)), envs, info
end

# ---- lazy (on-the-fly) naive-SVD initial guess (shared by mult / hadamard) ----
#
# 参考 FiniteMPSAlgorithms 的 `_naive_svd_guess`，但收缩路径更省：naive 乘积的
# site tensor 由 `site(i, carry)` **在构造时就把 carry 吸收进收缩**（优先与 carry
# 收缩），而不是先 materialize 完整的 naive 大张量再乘 carry —— 后者会多出一个
# 键维 = 输入键维乘积 的大中间张量。自右向左流式做带截断的右正交化（键 ≤ `D`）：
# 右因子作为输出 site tensor，`U·diag(s)` 作为 carry 传给左侧下一站。整条 naive
# 乘积族从不落地，峰值中间内存只有「首个（无 carry 的）大张量 + 键 ≤ D 的输出」。
# 收尾在 site 1 上对 wrap 键（bond N）做 Schmidt 截断：扫掠是一条线扫，环形闭合
# 的键在扫掠中截不到；但此时 site 2:L 右规范 ⇒ site 1 张量的 (1,)|(2,3) SVD 恰是
# 该键的谱（截断精确最优），`uᵀ` 吸收回 site L（右规范保持：(uᵀA)(uᵀA)† = uᵀu）。
# 因此**输出的每个键都 ≤ D**，调用方无需再做收尾截断。

"""
    _lazy_svd_guess(site, L, D) -> Vector{Array{T,3}}

`site(i, carry)`（`i = 1:L`，`carry::Union{Nothing,AbstractMatrix}`）现算第 `i`
站张量并在构造中吸收 carry（见上方注释）；自右向左流式 SVD 截断到键 ≤ `D`，
收尾做 wrap 键截断。返回右规范串（site 2:L）＋携带余量的 site 1，
**所有键 ≤ `D`**。
"""
function _lazy_svd_guess(site::F, L::Int, D::Int) where {F}
    B = site(L, nothing)
    out = Vector{typeof(B)}(undef, L)
    carry = nothing
    for i in L:-1:2
        u, s, v, _ = tsvd(B, (1,), (2, 3); trunc = truncdim(D))
        out[i] = v
        carry = u * Diagonal(s)
        B = site(i - 1, carry)      # 为下一轮准备；i = 2 时即 site(1, carry)
    end
    # B = site(1, carry)：wrap 键（bond N）的 Schmidt 截断（bond N = site 1 的左键
    # = site L 的右键）
    u, s, v, _ = tsvd(B, (1,), (2, 3); trunc = truncdim(D))
    v3 = Diagonal(s) * reshape(v, length(s), :)             # v 是秩-3 (r, s, D)
    out[1] = reshape(v3, length(s), size(B, 2), size(B, 3))
    if L >= 2
        NL = out[L]
        out[L] = @tensor A[a, s2, b] := NL[a, s2, bb] * u[bb, b]    # uᵀ 吸收回 site L 右腿
    else
        # L = 1：没有独立的 wrap 键，对第二个键再做一次 SVD（初猜用途的近似）
        r1, s1, _ = size(out[1])
        u2, svals, _, _ = tsvd(out[1], (1, 2), (3,); trunc = truncdim(D))
        u2m = @tensor uu[p, q, f2] := u2[p, q, k] * Diagonal(svals)[k, f2]
        out[1] = reshape(u2m, r1, s1, length(svals))
    end
    return out
end

"Copy the tensor families of `y` into `out` (both mixed-canonical)."
function _copyinto!(out::CanonicalIMPS, y::CanonicalIMPS)
    copy!(out.AL, y.AL)
    copy!(out.AR, y.AR)
    copy!(out.C, y.C)
    copy!(out.AC, y.AC)
    return out
end

function _copyinto!(out::CanonicalIMPO, y::CanonicalIMPO)
    copy!(out.AL, y.AL)
    copy!(out.AR, y.AR)
    copy!(out.C, y.C)
    copy!(out.AC, y.AC)
    return out
end

# ---------------- 纯重叠通道：OverlapCache 与 compress 的 VOMPS / IDMRG 引擎 ----------------
#
# ⟨below|above⟩（identity 通道，无算符）的环境缓存与变分扫掠——本文件的压缩
# 引擎（MPSKit `approximate(ψ₀, ϕ, ...)` 的无算符分支；mult 的 MPO 施加通道见
# mult.jl 的 MultCache 版本）。环境张量为 rank-2 矩阵：`lefts[ℓ]` = `(below
# bond, above bond)`、`rights[ℓ]` = `(above bond, below bond)`，相比 rank-3 的
# w=1 占位维省一份冗余维。
#
# 注意：rank-2 双层 push（push_env_left/right(::AbstractMatrix, above, below)）
# 的参数序是 `(above, below)`——conj 落在 below 上；与 rank-3 三元方法的
# `(below, above)` 相反。下面的调用点均已按此对齐（conj 恒落在演动态 x 上）。

# ---- 环境缓存（OverlapCache；先于 Hamiltonian 层定义——
# AC_Hamiltonian/C_Hamiltonian 的签名引用本类型，须先绑定本包的名字，
# 避免解析到 FiniteMPSAlgorithms 导出的同名类型） ----

"""
    OverlapCache(bra, ket, lefts, rights)
    OverlapCache(ψ) -> OverlapCache
    OverlapCache(below, above; tol, krylovdim, maxiter) -> OverlapCache

Environments of the pure overlap channel `⟨bra|ket⟩` (identity channel): the
left/right fixed points of the double-layer transfer, used by the VOMPS / IDMRG
compression sweeps of `compress` ([`compression_sweeps!`](@ref)).

- `lefts[ℓ]`: rank-2 `(below bond, above bond)` left environment of site ℓ;
- `rights[ℓ]`: rank-2 `(above bond, below bond)` right environment of site ℓ;
- `OverlapCache(ψ)`: the AL/AR gauges make the fixed points identity matrices;
- `OverlapCache(below, above)`: the fixed points are obtained from the :LM
  eigenpairs of the fused transfer `T(above.AL, below.AL)`
  ([`overlap_fixedpoints`](@ref)); normalization
  mirrors MPSKit (unit-Frobenius GRs, GLs scaled by the local C-channel
  overlap λ).
"""
struct OverlapCache{B<:CanonicalIMPS,K<:AbstractInfiniteMPS,T} <: CompressionEnvironments
    bra::B
    ket::K
    lefts::Vector{Array{T,2}}
    rights::Vector{Array{T,2}}
end

# ---- 纯重叠通道的局部映射（rank-2 `_mapAC`/`_mapC`，与三元/zip 通道的
# `_mapAC`/`_mapC` 同名分派：环境为 rank-2 矩阵，被作用张量经参数传入） ----

"纯重叠通道 site 站的 AC 局部投影：
`ACnew[aL, p, aR] = GL[aL, bL]·ketac[bL, p, bR]·GR[bR, aR]`。"
function _mapAC(GL::AbstractMatrix{Tg}, GR::AbstractMatrix{Tgr},
                ketac::AbstractArray{Tk,3}) where {Tg,Tgr,Tk}
    @tensor ACnew[aL, p, aR] := GL[aL, bL] * ketac[bL, p, bR] * GR[bR, aR]
    return ACnew
end

"纯重叠通道 bond site 的 C 局部投影：
`Cnew[aL, aR] = GL[aL, bL]·ketc[bL, bR]·GR[bR, aR]`。"
function _mapC(GL::AbstractMatrix{Tg}, GR::AbstractMatrix{Tgr},
               ketc::AbstractMatrix{Tk}) where {Tg,Tgr,Tk}
    @tensor Cnew[aL, aR] := GL[aL, bL] * ketc[bL, bR] * GR[bR, aR]
    return Cnew
end

"Identity channel: in the AL/AR gauges the fixed points are identity matrices."
function OverlapCache(ψ::AbstractInfiniteMPS; kwargs...)
    N = length(ψ)
    T = scalartype(ψ)
    lefts = Vector{Matrix{T}}(undef, N)
    rights = Vector{Matrix{T}}(undef, N)
    for ℓ in 1:N
        Dl, Dr = size(ψ.AL[ℓ], 1), size(ψ.AL[ℓ], 3)
        lefts[ℓ] = Matrix{T}(I, Dl, Dl)
        rights[ℓ] = Matrix{T}(I, Dr, Dr)
    end
    return OverlapCache(ψ, ψ, lefts, rights)
end

"rank-2 恒等通道固定点（环境核与 mult.jl 的 [`mixed_fixedpoints`](@ref) 统一——
左右不动点由 :LM 主本征对经 [`fixedpoint`](@ref) 解出（`alg` 分派 `tol`/`maxiter`：
NamedTuple / DynamicTol / KrylovKit 算法皆可），复环境按解的实际 eltype 存放；
本函数是三元通道的无算符分支，环境为 rank-2 矩阵）。"
function overlap_fixedpoints(below::CanonicalIMPS, above::AbstractInfiniteMPS,
                             alg = Defaults.alg_environments();
                             GL0::Union{Nothing,AbstractArray} = nothing,
                             GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(below)
    T = promote_type(scalartype(below), scalartype(above))
    # 键 profile 逐站可变：环境定义在周期闭合的键 N 上（below/above 同长）
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)

    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Da)
        for ℓ in 1:N
            GL = push_env_left(GL, above.AL[ℓ], below.AL[ℓ])
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Da) : vec(copy(GL0))
    _, vL = fixedpoint(Tleft, v0L, :LM, alg)
    # 复环境提升（MPSKit 对齐：环境按 eigsolve 返回的实际 eltype 存放）
    TCL = promote_type(T, eltype(vL))
    GLs = Vector{Matrix{TCL}}(undef, N)
    GLs[1] = GL = reshape(vL, Dl, Da)
    for ℓ in 2:N
        GLs[ℓ] = GL = push_env_left(GL, above.AL[ℓ-1], below.AL[ℓ-1])
    end

    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dl)
        for ℓ in N:-1:1
            GR = push_env_right(GR, above.AR[ℓ], below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da * Dl) : vec(copy(GR0))
    _, vR = fixedpoint(Tright, v0R, :LM, alg)
    TCR = promote_type(T, eltype(vR))
    GRs = Vector{Matrix{TCR}}(undef, N)
    GRs[N] = GR = reshape(vR, Da, Dl)
    for ℓ in N-1:-1:1
        GRs[ℓ] = GR = push_env_right(GR, above.AR[ℓ+1], below.AR[ℓ+1])
    end

    # 归一化（MPSKit：GR Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）
    for ℓ in 1:N
        GRs[ℓ] ./= norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        Cnew = _mapC(GLs[inext], GRs[ℓ], above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("overlap environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] ./= λ
    end
    return GLs, GRs
end

"Ternary overlap channel: left/right fixed points of ⟨below|above⟩（
[`overlap_fixedpoints`](@ref) 的 `alg` 分派 `tol`/`maxiter`）。"
function OverlapCache(below::CanonicalIMPS, above::AbstractInfiniteMPS,
                      alg = Defaults.alg_environments();
                      GL0 = nothing, GR0 = nothing)
    GLs, GRs = overlap_fixedpoints(below, above, alg; GL0, GR0)
    # bra/ket 提升到环境标量类型：缓存内所有 fields 同一浮点类型（dense/canonical
    # 槽各自原类型提升，不做规范转换）
    T = eltype(GLs[1])
    return OverlapCache(_promote_scalar(T, below), _promote_scalar(T, above),
                        GLs, GRs)
end

"""
    recalculate!(envs::OverlapCache, newbra, [alg_environments]) -> envs

Recompute the ⟨bra|ket⟩ fixed points for the updated bra `newbra`（ket 不变，
[`overlap_fixedpoints`](@ref) 重解），边界 eigsolve 以当前存储的
`lefts[1]`/`rights[end]` 热启动（对标 MPSKit 的原地 `recalculate!`）。
`newbra` 与新不动点就地写回 `envs`（原地更新，返回 `envs` 本身）。
"""
function recalculate!(envs::OverlapCache, newbra::CanonicalIMPS,
                      alg_environments = Defaults.alg_environments())
    GLs, GRs = overlap_fixedpoints(newbra, envs.ket, alg_environments;
                                   GL0 = envs.lefts[1], GR0 = envs.rights[end])
    copy!(envs.lefts, GLs)
    copy!(envs.rights, GRs)
    envs.bra ≡ newbra || _copyinto!(envs.bra, newbra)
    return envs
end

# ---- 纯重叠通道的增量环境推进（IDMRG 扫掠用；rank-2 参数序 (above, below)）----

function transfer_leftenv!(envs::OverlapCache, x::CanonicalIMPS, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], envs.ket.AL[ℓm], x.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::OverlapCache, x::CanonicalIMPS, site::Int)
    N = length(envs)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], envs.ket.AR[ℓp], x.AR[ℓp])
    return envs
end

"纯重叠通道的环境重标定（MPSKit `normalize!(envs, below, above)` 语义：GR
Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）。"
function normalize_envs!(envs::OverlapCache, x::CanonicalIMPS)
    N = length(envs)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = _local_C(envs, ℓ)
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("overlap idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"压缩通道的统一局部映射入口（[`compression_sweeps!`](@ref) 用；全部输入由缓存
持有）：`_local_AC(envs, ℓ)` 为 site ℓ 的 AC 局部投影、`_local_C(envs, ℓ)` 为
bond ℓ 的 C 局部投影（无算符通道 = rank-2 `_mapAC`/`_mapC`）。"
_local_AC(envs::OverlapCache, ℓ::Int) =
    _mapAC(leftenv(envs, ℓ), rightenv(envs, ℓ), envs.ket.AC[ℓ])

_local_C(envs::OverlapCache, ℓ::Int) =
    _mapC(leftenv(envs, _mod1(ℓ + 1, length(envs))), rightenv(envs, ℓ),
          envs.ket.C[ℓ])

"扫掠 finalize 回调的通道目标（[`compression_sweeps!`](@ref) 用）：compress
通道 = ket。"
_finalize_target(envs::OverlapCache) = envs.ket

"纯重叠通道的最大逐站 Galerkin 残差（语义同 mult.jl 的 `calc_galerkin`，
无算符插入）。"
function calc_galerkin(envs::OverlapCache, x::CanonicalIMPS)
    N = length(envs)
    ϵ = 0.0
    for ℓ in 1:N
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], _local_AC(envs, ℓ)))
    end
    return ϵ
end

# 统一扫掠引擎 `compression_sweeps!`（三个通道共享的 VOMPS/IDMRG 模板）定义在
# arithmetics/envs.jl（CompressionEnvironments 层次所在处）。
