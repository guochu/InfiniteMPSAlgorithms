# ---------------- compress (bond-dimension reduction of a single chain, mirroring MPSKit's approximate) ----------------
#
# Find `out ≈ x` with `max_bonddim(out) = D`: the overlap-maximizing
# variational approximation of a *given* chain (the target is the chain itself,
# unlike add/mult/hadamard whose targets are naive algebraic constructions).
# The identity-channel engines (`OverlapCache` +
# `_overlap_vomps_sweeps`/`_overlap_idmrg_sweeps`) live at the end of this
# file; the MPO 施加 / zip 通道的引擎（mult.jl / hadamard.jl）共用同一管道。

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
re-canonicalize its output.
"""
function svdguess_compress(x::CanonicalIMPS, D::Int)
    max_bonddim(x) ≤ D && return copy(x)
    return CanonicalIMPS(svdguess_compress(x.AL, D))
end

function svdguess_compress(x::CanonicalIMPO, D::Int)
    return svdguess_compress(vectorize(x), D)
end

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
    compress(x::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) -> (x′::CanonicalIMPS, envs, info)
    compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) -> (x′::CanonicalIMPO, envs, info)
    compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG}) -> (x′::CanonicalIMPO, envs, info)

Bond-dimension-`alg.D` variational approximation of a single chain (mirroring
MPSKit's `approximate`: maximize the overlap between the compressed chain and
the input chain, fixed point = the best rank-`D` approximation in the
ring-trace fidelity sense). `alg.D ≥ max_bonddim` of the
input short-circuits to the exact input. MPO results are always returned in
mixed-canonical storage (`CanonicalIMPO`, satisfying `ismixedcanonical`) — a
`DenseIMPO` input is canonicalized exactly (the operator value is preserved in
the periodic-trace representation) instead of being passed through. The
default initial guess is [`svdguess_compress`](@ref) (the input's own SVD
truncation). The positional `alg` dispatches [`VOMPS`](@ref) (ALS sweeps) or
[`IDMRG`](@ref) (MPSKit sequential Gauss–Seidel sweeps with on-the-fly
environment transfer), which share the same fixed point.

The second return is the engine's final environments（纯重叠通道的
[`OverlapCache`](@ref)，对返回态的射线解出；MPO 压缩在 `vectorize` 的 MPS 视图
上演化，envs 持该 MPS 视图）; the third return is the
[`IterativeConvergenceInfo`](@ref)（`niter` 扫掠轮数、`losses` 逐轮残差/漂移、
`converged` 收敛标志；短路的精确路径为 0 轮、已收敛、恒等矩阵环境）。
"""
compress(ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) =
    _compress(ψ, alg, nothing; D = alg.D)

compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) =
    _compress(W, alg, nothing; D = alg.D)

compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG}) =
    _compress(W, alg, nothing; D = alg.D)

function _compress(ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    if D >= max_bonddim(ψ)
        out = copy(ψ)
        return out, OverlapCache(out), IterativeConvergenceInfo(0, Float64[0.0], true)
    end
    x0 = x0 === nothing ? svdguess_compress(ψ, D) : x0
    x, envs, info = if alg isa VOMPS
        _overlap_vomps_sweeps(ψ, x0, alg)
    else
        _overlap_idmrg_sweeps(ψ, x0, alg)
    end
    return _global_normalize!(x), envs, info
end

function _compress(W::CanonicalIMPO, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    if D >= max_bonddim(W)
        out = copy(W)
        return out, OverlapCache(vectorize(out)),
               IterativeConvergenceInfo(0, Float64[0.0], true)
    end
    # 目标 MPS 的 ray 必须取 **AL 家族**（左正则串）：其周期 trace 才是算符串的
    # wavefunction（相似变换不变）；AC 家族在张量间携带 C 加权
    # （AC[ℓ] = AL[ℓ]·C[ℓ]），周期 trace 是规范依赖的加权量，作为压缩目标会
    # 定义另一个变分问题（与 `mult!` 的因子化目标不等价）。
    # `vectorize` 逐家族携带规范数据，其 AL 家族正是该 ray 的左正则串。
    ket = vectorize(W)
    x0 = x0 === nothing ? svdguess_compress(W, D) : x0
    x, envs, info = if alg isa VOMPS
        _overlap_vomps_sweeps(ket, x0, alg)
    else
        _overlap_idmrg_sweeps(ket, x0, alg)
    end
    x = _global_normalize!(x)
    return devectorize(x), envs, info
end

function _compress(W::DenseIMPO, alg::Union{VOMPS,IDMRG},
                   x0::Union{Nothing,CanonicalIMPS}; D::Int)
    if D >= max_bonddim(W)
        out = CanonicalIMPO(collect(W.Ws))
        return out, OverlapCache(vectorize(out)),
               IterativeConvergenceInfo(0, Float64[0.0], true)
    end
    ket = CanonicalIMPS(vectorize(W).As)
    x0 = x0 === nothing ? svdguess_compress(ket, D) : x0
    x, envs, info = if alg isa VOMPS
        _overlap_vomps_sweeps(ket, x0, alg)
    else
        _overlap_idmrg_sweeps(ket, x0, alg)
    end
    x = _global_normalize!(x)
    return devectorize(x), envs, info
end

# ---------------- compress! (in-place) ----------------

"""
    compress!(out, ψ::CanonicalIMPS, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)
    compress!(out, W::CanonicalIMPO, alg::Union{VOMPS,IDMRG}) -> (out, envs, info)

In-place [`compress`](@ref): `out` is the user-provided chain to be optimized
as the initial guess. The target bond dimension is taken from the bond profile
of `out` (its bond profile is first brought to uniform `D = max_bonddim(out)`
with [`changebond!`](@ref)); `alg.D` is ignored. The optimized result is
written back into `out`. Returns `(out, envs, info)`——`envs` 为引擎的最终环境、
`info` 为 [`IterativeConvergenceInfo`](@ref)。
"""
function compress!(out::CanonicalIMPS, ψ::CanonicalIMPS,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, envs, info = _compress(ψ, alg, out; D = D)
    return _copyinto!(out, y), envs, info
end

function compress!(out::CanonicalIMPO, W::CanonicalIMPO,
                   alg::Union{VOMPS,IDMRG})
    D = max_bonddim(out)
    changebond!(out; D = D)
    y, envs, info = _compress(W, alg, vectorize(out); D = D)
    return _copyinto!(out, y), envs, info
end

"Raw-target variants: `compress!` of a strict-algebra result
(`DenseIMPO * DenseIMPS` / `DenseIMPO * DenseIMPO`, not yet canonicalized) —
the input is canonicalized and the standard `compress!` pipeline runs."
function compress!(out::CanonicalIMPS, ψ::DenseIMPS,
                   alg::Union{VOMPS,IDMRG})
    return compress!(out, CanonicalIMPS(collect(ψ.As)), alg)
end

function compress!(out::CanonicalIMPO, W::DenseIMPO,
                   alg::Union{VOMPS,IDMRG})
    return compress!(out, CanonicalIMPO(collect(W.Ws)), alg)
end

"Variational compression of a raw strict-algebra result
(`DenseIMPO * DenseIMPS`, not yet canonicalized)."
compress(ψ::DenseIMPS, alg::Union{VOMPS,IDMRG}) =
    _compress(CanonicalIMPS(collect(ψ.As)), alg, nothing; D = alg.D)

# ---------------- 共享的代数压缩装配（原 add.jl；add 已删除） ----------------

"""
    _compress_ket(K, physdims, D, alg; x0 = nothing) -> CanonicalIMPS

Variationally compress the naively constructed target tensor string `K` (MPS
view, rank-3) to bond dimension `D`: `alg::VOMPS` runs the ALS sweeps
(`_overlap_vomps_sweeps`), `alg::IDMRG` the MPSKit sequential-sweep template
(`_overlap_idmrg_sweeps`); finally the result is globally normalized (norm
convention `‖AC[1]‖ = 1`). `x0` optionally provides the initial state (defaults
to `svdguess`: the target's own bond-wise SVD truncation, deterministic and
inside the correct basin).
"""
function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::VOMPS; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? _truncate_bonddim(copy(ket), D) : x0
    x, _, _ = _overlap_vomps_sweeps(ket, x0, alg)
    return _global_normalize!(x)
end

function _compress_ket(K::Vector{<:Array{T,3}}, physdims::AbstractVector{Int}, D::Int,
                       alg::IDMRG; x0::Union{Nothing,CanonicalIMPS} = nothing) where {T}
    ket = CanonicalIMPS(K)
    x0 = x0 === nothing ? _truncate_bonddim(copy(ket), D) : x0
    x, _, _ = _overlap_idmrg_sweeps(ket, x0, alg)
    return _global_normalize!(x)
end

_compress_ket(::Vector{<:Array{T,3}}, ::AbstractVector{Int}, ::Int,
              alg::Algorithm) where {T} =
    throw(ArgumentError("algebra compression only supports VOMPS() (DMRG-type) or IDMRG() algorithms; got $(typeof(alg))"))

"""
    _truncate_bonddim(ψ, D) -> CanonicalIMPS

Deterministic initial state for the compression of a block-degenerate target
(such as the direct sums of `add`): the SVD truncation of the naive target
itself to bond dimension `D` — always inside the correct ALS basin, unlike a
random initial state (the orthogonalities of `AL`/`AR` are preserved: the
truncation factors `U`/`V` act between orthogonality-protected bonds).
"""
function _truncate_bonddim(ψ::CanonicalIMPS{T}, D::Int) where {T}
    N = length(ψ)
    # 逐 bond 在原态上独立 SVD（不同 bond 的正交投影互易，可统一应用；
    # 串行"截一个再截下一个"会在已投影的态上用旧基因子，破坏混合规范）
    svds = Dict{Int,Any}()
    for ℓ in 1:N
        size(ψ.C[ℓ], 1) > D || continue
        svds[ℓ] = tsvd(ψ.C[ℓ]; trunc = truncdim(D))
    end
    isempty(svds) && return ψ
    # 应用：AL/AR[ℓ] 右键乘 U_ℓ、左键乘 V_{ℓ-1}；C[ℓ] = diag(s_ℓ)。
    # U/V 正交 ⇒ AR 串仍是右规范串，从它重建混合规范。
    for ℓ in 1:N
        ℓm = _mod1(ℓ - 1, N)
        if haskey(svds, ℓ)
            U, s, V, _ = svds[ℓ]
            ψ.AL[ℓ] = @tensor A[a, s2, c] := ψ.AL[ℓ][a, s2, bb] * U[bb, c]
            ψ.AR[ℓ] = @tensor A[a, s2, c] := ψ.AR[ℓ][a, s2, bb] * U[bb, c]
            ψ.C[ℓ] = Matrix{T}(Diagonal(s))
        end
        if haskey(svds, ℓm)
            _, _, Vm, _ = svds[ℓm]
            ψ.AL[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AL[ℓ][bb, s2, b]
            ψ.AR[ℓ] = @tensor A[a, s2, b] := Vm[a, bb] * ψ.AR[ℓ][bb, s2, b]
        end
    end
    y = CanonicalIMPS(collect(ψ.AR))
    copy!(ψ.AL, y.AL)
    copy!(ψ.AR, y.AR)
    copy!(ψ.C, y.C)
    copy!(ψ.AC, y.AC)
    return ψ
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

"DenseIMPO 结果写回 `CanonicalIMPO` 缓存：先转规范形式再逐家族复制。"
_copyinto!(out::CanonicalIMPO, y::DenseIMPO) = _copyinto!(out, CanonicalIMPO(collect(y.Ws)))

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
compression sweeps of `compress` ([`_overlap_vomps_sweeps`](@ref) /
[`_overlap_idmrg_sweeps`](@ref)).

- `lefts[ℓ]`: rank-2 `(below bond, above bond)` left environment of site ℓ;
- `rights[ℓ]`: rank-2 `(above bond, below bond)` right environment of site ℓ;
- `OverlapCache(ψ)`: the AL/AR gauges make the fixed points identity matrices;
- `OverlapCache(below, above)`: the fixed points are obtained from the :LM
  eigenpairs of the fused transfer `T(above.AL, below.AL)`
  ([`overlap_fixedpoints`](@ref)); normalization
  mirrors MPSKit (unit-Frobenius GRs, GLs scaled by the local C-channel
  overlap λ).
"""
struct OverlapCache{B<:CanonicalIMPS,K<:CanonicalIMPS,T} <: CompressionEnvironments
    bra::B
    ket::K
    lefts::Vector{Array{T,2}}
    rights::Vector{Array{T,2}}
end

# ---- 纯重叠通道的局部映射（Overlap_AC/C_Hamiltonian） ----
#
# 接口与 effective.jl 的 MPO_AC/C_Hamiltonian 对齐：只存 leftenv/rightenv，
# 被作用的局部张量经调用传入——AC_Hamiltonian(site, envs)(ket.AC[site])、
# C_Hamiltonian(site, envs)(ket.C[site])，线性映射 `h(x) = GL·x·GR`。

"""
    Overlap_AC_Hamiltonian(leftenv, rightenv)
    AC_Hamiltonian(site, envs::OverlapCache) -> callable

纯重叠通道 site 站的 AC 局部映射：`h(ketac) = GL·ketac·GR`（被作用对象 =
`envs.ket.AC[site]`，即对 `ket.AC` 的 Jacobi 投影）。
"""
struct Overlap_AC_Hamiltonian{L<:AbstractMatrix,R<:AbstractMatrix}
    leftenv::L
    rightenv::R
end

"""
    Overlap_C_Hamiltonian(leftenv, rightenv)
    C_Hamiltonian(site, envs::OverlapCache) -> callable

纯重叠通道 bond site 的 C 局部映射：`h(ketc) = GL·ketc·GR`（被作用对象 =
`envs.ket.C[site]`，即对 `ket.C` 的 Jacobi 投影）。
"""
struct Overlap_C_Hamiltonian{L<:AbstractMatrix,R<:AbstractMatrix}
    leftenv::L
    rightenv::R
end

"AC_Hamiltonian(site, envs) 的装配：`leftenv(envs, site)` 与 `rightenv(envs, site)`。"
function AC_Hamiltonian(site::Int, envs::OverlapCache)
    return Overlap_AC_Hamiltonian(leftenv(envs, site), rightenv(envs, site))
end

"C_Hamiltonian(site, envs) 的装配：`leftenv(envs, site + 1)` 与 `rightenv(envs, site)`。"
function C_Hamiltonian(site::Int, envs::OverlapCache)
    return Overlap_C_Hamiltonian(leftenv(envs, site + 1), rightenv(envs, site))
end

function (h::Overlap_AC_Hamiltonian)(ketac::AbstractArray{T,3}) where {T}
    @tensor y[aL, p, aR] := h.leftenv[aL, bL] * ketac[bL, p, bR] * h.rightenv[bR, aR]
    return y
end

function (h::Overlap_C_Hamiltonian)(ketc::AbstractMatrix{T}) where {T}
    @tensor y[a, a′] := h.leftenv[a, b] * ketc[b, b′] * h.rightenv[b′, a′]
    return y
end

"Identity channel: in the AL/AR gauges the fixed points are identity matrices."
function OverlapCache(ψ::CanonicalIMPS; kwargs...)
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
function overlap_fixedpoints(below::CanonicalIMPS, above::CanonicalIMPS,
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
        Cnew = Overlap_C_Hamiltonian(GLs[inext], GRs[ℓ])(above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("overlap environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] ./= λ
    end
    return GLs, GRs
end

"Ternary overlap channel: left/right fixed points of ⟨below|above⟩（
[`overlap_fixedpoints`](@ref) 的 `alg` 分派 `tol`/`maxiter`）。"
function OverlapCache(below::CanonicalIMPS, above::CanonicalIMPS,
                      alg = Defaults.alg_environments();
                      GL0 = nothing, GR0 = nothing)
    GLs, GRs = overlap_fixedpoints(below, above, alg; GL0, GR0)
    # bra/ket 提升到环境标量类型：缓存内所有 fields 同一浮点类型
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

function transfer_leftenv!(envs::OverlapCache, x::CanonicalIMPS,
                           ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓm = _mod1(site - 1, N)
    envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], ket.AL[ℓm], x.AL[ℓm])
    return envs
end

function transfer_rightenv!(envs::OverlapCache, x::CanonicalIMPS,
                            ket::CanonicalIMPS, site::Int)
    N = length(ket)
    ℓ = _mod1(site, N)
    ℓp = _mod1(site + 1, N)
    envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ket.AR[ℓp], x.AR[ℓp])
    return envs
end

"纯重叠通道的环境重标定（MPSKit `normalize!(envs, below, above)` 语义：GR
Frobenius 归一、GL[ℓ+1] 按局部 C 通道 overlap λ 缩放）。"
function normalize_envs!(envs::OverlapCache, x::CanonicalIMPS,
                         ket::CanonicalIMPS)
    N = length(ket)
    for ℓ in 1:N
        GR = envs.rights[ℓ]
        nr = norm(GR)
        nr > 0 && (GR ./= nr)
        Cnew = C_Hamiltonian(ℓ, envs)(ket.C[ℓ])
        λ = dot(x.C[ℓ], Cnew)
        λ == 0 && error("overlap idmrg sweep: local overlap λ = 0 at site $ℓ")
        envs.lefts[_mod1(ℓ + 1, N)] ./= λ
    end
    return envs
end

"纯重叠通道的最大逐站 Galerkin 残差（语义同 mult.jl 的
`_galerkin_err(operator, ket, x, envs)`，无算符插入）。"
function _galerkin_err(ket::CanonicalIMPS, x::CanonicalIMPS, envs::OverlapCache)
    N = length(ket)
    ϵ = 0.0
    for ℓ in 1:N
        ACmap = AC_Hamiltonian(ℓ, envs)(ket.AC[ℓ])
        ϵ = max(ϵ, _galerkin(x.AL[ℓ], ACmap))
    end
    return ϵ
end

"""
    _overlap_vomps_sweeps(ket, x0, alg::VOMPS) -> (x, envs, info)

Overlap-maximizing VOMPS sweeps on the pure overlap channel (the operator-free
branch of MPSKit's `approximate(ψ₀, ϕ, VOMPS())`): find `x` approximating the
target chain `ket` itself — variational compression. Jacobi-style rounds, the
same `IterativeSolver` pipeline as the mult channel ([`_vomps_sweeps`](@ref)):
`localupdate`（`AC_new = GL·ket.AC·GR`、`C_new = GL₊·ket.C·GR` → `regauge!`）→
`gauge_step!` → warm-started environment re-solve → Galerkin residual checked
after the sweep. Returns `(x, envs, info)`，`info` 为
[`IterativeConvergenceInfo`](@ref)（`niter` = 扫掠轮数、`losses` = [初始残差,
逐轮 Galerkin 残差...]、`converged` 收敛标志）。
"""
function _overlap_vomps_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS, alg::VOMPS)
    N = length(ket)
    x = copy(x0)
    envs = OverlapCache(x, ket, alg.alg_environments)
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释；缓存的 bra/ket 已由
    # 构造器提升到环境标量类型）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    # 初始残差（收敛判定在扫掠之后，MPSKit IterativeSolver 语义）
    ϵ = _galerkin_err(ket, x, envs)
    iter = 0
    losses = [ϵ]
    converged = false
    for outer iter in 1:alg.maxiter
        # localupdate: per-site local maps + regauge（全部站点对同一批环境；
        # 候选 AL 与 ket.AC 同形，eltype 提升到通道标量类型 T）
        ALs = [similar(ket.AC[ℓ], T) for ℓ in 1:N]
        for ℓ in 1:N
            AC_new = AC_Hamiltonian(ℓ, envs)(ket.AC[ℓ])
            C_new = C_Hamiltonian(ℓ, envs)(ket.C[ℓ])
            ALs[ℓ] = regauge!(AC_new, C_new; alg = alg.alg_orth)
        end
        # gauge: restore the global right gauge（动态容差）
        alg_g = updatetol(alg.alg_gauge, iter - 1, ϵ)
        gauge_step!(x, ALs, x.C[N]; tol = alg_g.tol, maxiter = alg_g.maxiter)
        # envs_step!（bra 更新 + 热启动 + 动态环境容差）
        alg_envs = updatetol(alg.alg_environments, iter - 1, ϵ)
        recalculate!(envs, x, alg_envs)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket, envs)
        ϵ = _galerkin_err(ket, x, envs)
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
    _overlap_idmrg_sweeps(ket, x0, alg::IDMRG) -> (x, envs, info)

IDMRG template on the pure overlap channel (the operator-free branch of
MPSKit's `approximate(ψ₀, ϕ, IDMRG())`): sequential Gauss–Seidel double sweep
with on-the-fly environment transfer, `leftorth`/`rightorth` splits of the
normalized local projections, per-double-sweep environment rescaling
([`normalize_envs!`](@ref)) and center-matrix-drift convergence
`ϵ = ‖C₀_new − C₀_old‖`; afterwards the mixed-canonical state is rebuilt from
the `AR` string and the environments are re-solved for the final state.
Returns `(x, envs, info)`，`info` 为 [`IterativeConvergenceInfo`](@ref)
（`niter` = 扫掠轮数、`losses` = 逐轮中心矩阵漂移、`converged` 收敛标志）。
"""
function _overlap_idmrg_sweeps(ket::CanonicalIMPS, x0::CanonicalIMPS, alg::IDMRG)
    N = length(ket)
    x = copy(x0)
    # 初始环境：由初态解一次左右不动点，扫掠中只做增量 transfer 与重标定
    envs = OverlapCache(x, ket, alg.alg_environments)
    # 通道标量类型提升（见 mult.jl `_vomps_sweeps` 注释；缓存的 bra/ket 已由
    # 构造器提升到环境标量类型）
    T = promote_type(scalartype(ket), eltype(leftenv(envs, 1)))
    x = _promote_scalar(T, x)
    ϵ = 2 * alg.tol
    iter = 0
    losses = Float64[]
    converged = false
    for outer iter in 1:alg.maxiter
        C_old = copy(x.C[0])
        # left to right sweep（Gauss–Seidel：环境随扫掠即时推进）
        for ℓ in 1:N
            x.AC[ℓ] = AC_Hamiltonian(ℓ, envs)(ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.AL[ℓ], x.C[ℓ] = _leftsplit(x.AC[ℓ], alg.alg_orth)
            transfer_leftenv!(envs, x, ket, ℓ + 1)
        end
        # right to left sweep
        for ℓ in N:-1:1
            x.AC[ℓ] = AC_Hamiltonian(ℓ, envs)(ket.AC[ℓ])
            normalize!(x.AC[ℓ])
            x.C[ℓ - 1], x.AR[ℓ] = _rightsplit(x.AC[ℓ], alg.alg_orth)
            transfer_rightenv!(envs, x, ket, ℓ - 1)
        end
        # 环境重标定
        normalize_envs!(envs, x, ket)
        # 收敛判据：bond 0 中心矩阵漂移
        ϵ = norm(x.C[0] - C_old)
        push!(losses, ϵ)
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ)
        # finalize（逐迭代回调，MPSKit finalize! 语义）
        x, envs = alg.finalize(iter, x, ket, envs)
        if ϵ < alg.tol
            converged = true
            break
        end
    end
    # 规范恢复：从 AR 重建混合规范（容差取 alg_gauge 的动态适配），环境对终态重解
    alg_g = updatetol(alg.alg_gauge, iter, ϵ)
    x = CanonicalIMPS([x.AR[ℓ] for ℓ in 1:N]; tol = alg_g.tol,
                      maxiter = alg_g.maxiter)
    recalculate!(envs, x, alg.alg_environments)
    _global_normalize!(x)
    return x, envs, IterativeConvergenceInfo(iter, losses, converged)
end
