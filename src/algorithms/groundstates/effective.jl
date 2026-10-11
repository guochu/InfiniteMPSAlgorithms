# ---------------- effective local Hamiltonians (mirroring MPSKit AC_hamiltonian / C_hamiltonian) ----------------
#
# Environment index conventions (consistent with MPSKit):
#   GL = leftenv[ℓ]  = (bra bond, w, ket bond)
#   GR = rightenv[ℓ] = (ket bond, w, bra bond)
# The effective Hamiltonians are strictly linear operators (no conjugation of
# x, MPSKit form; correct for complex data as well).

# 内部类型约束别名（不导出，故意遮蔽 FMA 的同名 Array 别名——此处取更宽的
# AbstractArray 以接受 view 输入；用户侧无冲突）
const MPOTensor{T} = AbstractArray{T, 4}
const MPSTensor{T} = AbstractArray{T, 3}

"""
    MPO_AC_Hamiltonian(leftenv, operator, rightenv)

Effective operator obtained by differentiating an MPS-MPO-MPS sandwich with
respect to the local AC tensor (mirrors MPSKit's `AC_hamiltonian` structure).
`operator` is a single rank-4 MPO tensor, or the `SparseIMPO` Schur tensor
(applied block-wise per level pair — no densification).
"""
struct MPO_AC_Hamiltonian{L<:MPSTensor,O<:Union{MPOTensor,SchurMPOTensor},R<:MPSTensor}
    leftenv::L
    operators::O
    rightenv::R
end

"""
    MPO_C_Hamiltonian(leftenv, rightenv)

Effective operator on the center bond (the C problem): a pure left/right
environment sandwich without operator insertion (mirrors MPSKit's
`C_hamiltonian` structure).
"""
struct MPO_C_Hamiltonian{L<:MPSTensor,R<:MPSTensor}
    leftenv::L
    rightenv::R
end

"""
    C_hamiltonian(site, below, operator, above, envs) -> callable

Effective Hamiltonian on bond `site` (mirrors MPSKit): uses
`leftenv(envs, site + 1)` and `rightenv(envs, site)`, with action

```julia
C′[α, β] = Σ GL[α, w, α′] · C[α′, β′] · GR[β′, w, β]
```

`operator` is accepted for call-site parity with [`AC_hamiltonian`](@ref) and
ignored (the C problem carries no operator insertion).
"""
function C_hamiltonian(site::Int, below, operator, above, envs::DMRGCache)
    return MPO_C_Hamiltonian(leftenv(envs, site + 1), rightenv(envs, site))
end

"""
    AC_hamiltonian(site, below, operator, above, envs) -> callable

Effective Hamiltonian on site `site` (mirrors MPSKit): uses
`leftenv(envs, site)`, `operator[site]`, and `rightenv(envs, site)`, with action

```julia
AC′[b, u_out, b′] = Σ GL[b, wl, ā] · x[ā, u_in, b̄] · W[wl, u_out, wr, u_in] · GR[b̄, wr, b′]
```
"""
function AC_hamiltonian(site::Int, below, operator, above, envs::DMRGCache)
    return MPO_AC_Hamiltonian(leftenv(envs, site), operator[site], rightenv(envs, site))
end

# ---- action (MPSKit form, linear operators) ----

function (h::MPO_C_Hamiltonian)(x::AbstractMatrix{T}) where {T}
    GL, GR = h.leftenv, h.rightenv
    @tensor y[α, β] := GL[α, w, α′] * x[α′, β′] * GR[β′, w, β]
    return y
end

# 稠密 MPO 张量的 AC 作用（约束 Op<:MPOTensor 与下方 Schur 特化方法签名互斥）
function (h::MPO_AC_Hamiltonian{L,Op,R})(x::AbstractArray{T,3}) where {L,Op<:MPOTensor,R,T}
    GL, GR = h.leftenv, h.rightenv
    W4 = h.operators
    @tensor y[b, u_out, b′] := GL[b, wl, ā] * x[ā, u_in, b̄] *
                               W4[wl, u_out, wr, u_in] * GR[b̄, wr, b′]
    return y
end

# Schur 稀疏算符的 AC 作用：按 level 对 (i, j) 逐块收缩（iszero 零块跳过——
# Schur 上三角结构的零块从不落地），语义与稠密张量收缩逐位一致。
# 矩形张量：行/列各自取 space_l/space_r（GL 的 w 维 = space_l、GR 的 = space_r）。
function (h::MPO_AC_Hamiltonian{L,Op,R})(x::AbstractArray{T,3}) where {L,Op<:SchurMPOTensor,R,T}
    GL, GR, W = h.leftenv, h.rightenv, h.operators
    y = zeros(T, size(GL, 1), size(x, 2), size(GR, 3))
    for i in 1:space_l(W), j in 1:space_r(W)
        Wij = W[i, j]                 # (u_out, u_in) 块
        iszero(Wij) && continue
        GLi = @view GL[:, i, :]       # (bra bond, ket bond)
        GRj = @view GR[:, j, :]
        @tensor y[b, u_out, c] += GLi[b, a] * x[a, u_in, bb] *
                                  Wij[u_out, u_in] * GRj[bb, c]
    end
    return y
end

"""
    MPO_AC2_Hamiltonian(leftenv, operator1, operator2, rightenv)

Two-site effective operator（[`AC2_hamiltonian`](@ref) 的载体；MPSKit
`AC2_hamiltonian` 结构的两站推广）：把左环境、算符两站张量对、右环境
预先折叠，作用在 rank-4 两站张量 `x[kl, s1, s2, kr]`（物理腿逐站保留）
上。`operator1`/`operator2` 各为单个 rank-4 MPO 张量或 `SparseIMPO` 的
Schur 张量（链式 level 相容：`space_r(O1) = space_l(O2)`）。
"""
struct MPO_AC2_Hamiltonian{L,O1<:Union{MPOTensor,SchurMPOTensor},
                           O2<:Union{MPOTensor,SchurMPOTensor},R<:MPSTensor}
    leftenv::L
    operator1::O1
    operator2::O2
    rightenv::R
end

"""
    AC2(pos, ψ; kind = :ACAR) -> rank-4 两站张量
    AC2_hamiltonian(pos, below, operator, above, envs) -> callable

Two-site counterparts of [`AC_hamiltonian`](@ref)（MPSKit `AC2`/`AC2_hamiltonian`
的移植，`IDMRG2` 基态引擎用）。`kind = :ACAR` 融合 `AC[pos]·AR[pos+1]`、
`:ALAC` 融合 `AL[pos]·AC[pos+1]`（MPSKit 同款两站规范形态；`pos = 0` 即
wrap 对 `(N, 1)`，`PeriodicVector` 家族下自然闭合）。有效算符作用：

```julia
y[b, u1, u2, c] = Σ GL[b, w1, a] · x[a, s1, s2, bb] ·
                  W₁[w1, u1, w2, s1] · W₂[w2, u2, wr, s2] · GR[bb, wr, c]
```

其中 `GL = leftenv(envs, pos)`、`GR = rightenv(envs, pos + 1)`、
`Wᵢ = operator[对应站]`；`x` 与 `y` 的物理腿逐站保留（不融合）。
"""
function AC2(ψ::CanonicalIMPS, pos::Int; kind::Symbol = :ACAR)
    N = length(ψ)
    if kind === :ACAR
        next = _mod1(pos + 1, N)
        A₁, A₂ = ψ.AC[pos], ψ.AR[next]
    elseif kind === :ALAC
        next = _mod1(pos + 1, N)
        A₁, A₂ = ψ.AL[pos], ψ.AC[next]
    else
        throw(ArgumentError("unknown AC2 kind $kind (expected :ACAR or :ALAC)"))
    end
    ac2 = similar(A₁, size(A₁, 1), size(A₁, 2), size(A₂, 2), size(A₂, 3))
    @tensor ac2[a, s1, s2, b] := A₁[a, s1, m] * A₂[m, s2, b]
    return ac2
end

function AC2_hamiltonian(pos::Int, below, operator, above, envs::DMRGCache)
    L = length(operator)
    return MPO_AC2_Hamiltonian(leftenv(envs, pos), operator[_mod1(pos, L)],
                               operator[_mod1(pos + 1, L)], rightenv(envs, pos + 1))
end

# 稠密 MPO 张量对的 AC2 作用
function (h::MPO_AC2_Hamiltonian{L,O1,O2,R})(x::AbstractArray{T,4}) where {L,O1<:MPOTensor,O2<:MPOTensor,R,T}
    GL, W₁, W₂, GR = h.leftenv, h.operator1, h.operator2, h.rightenv
    @tensor y[b, u1, u2, c] := GL[b, w1, a] * x[a, s1, s2, bb] *
                               W₁[w1, u1, w2, s1] * W₂[w2, u2, wr, s2] *
                               GR[bb, wr, c]
    return y
end

# Schur 稀疏算符对的 AC2 作用：level 三元组 (i₁, j₁, j₂) 逐块收缩（链式
# 相容 space_r(W₁) = space_l(W₂)；iszero 零块跳过），语义与稠密收缩逐位一致。
# 分步收缩（每步 ≤2 操作数）：物理腿 s1/s2 分别与两站算符块收缩、键腿与
# GL/GR 收缩，避免多操作数规划的维度歧义。
function (h::MPO_AC2_Hamiltonian{L,O1,O2,R})(x::AbstractArray{T,4}) where {L,O1<:SchurMPOTensor,O2<:SchurMPOTensor,R,T}
    GL, W₁, W₂, GR = h.leftenv, h.operator1, h.operator2, h.rightenv
    (space_r(W₁) == space_l(W₂)) || throw(DimensionMismatch(
        "operator level chain does not match between the two window sites"))
    y = zeros(T, size(GL, 1), size(x, 2), size(x, 3), size(GR, 3))
    for i1 in 1:space_l(W₁), j1 in 1:space_r(W₁), j2 in 1:space_r(W₂)
        iszero(W₁[i1, j1]) && continue
        iszero(W₂[j1, j2]) && continue
        GLi = @view GL[:, i1, :]              # (bra b, ket a)
        W1ij = W₁[i1, j1]                     # (u1, s1)
        W2jk = W₂[j1, j2]                     # (u2, s2)
        GRj = @view GR[:, j2, :]              # (ket bb, bra c)
        # 分步（各 2 操作数）：x → W1ij(s1) → GLi(a) → W2jk(s2) → GRj(bb)
        t0 = zeros(T, size(x, 1), size(W1ij, 1), size(x, 3), size(x, 4))
        @tensor t0[a, u1, s2, bb] := x[a, s1, s2, bb] * W1ij[u1, s1]
        t1 = zeros(T, size(GLi, 1), size(t0, 2), size(t0, 3), size(t0, 4))
        @tensor t1[b, u1, s2, bb] := GLi[b, a] * t0[a, u1, s2, bb]
        t2 = zeros(T, size(t1, 1), size(W2jk, 1), size(t1, 3), size(t1, 4))
        @tensor t2[b, u1, u2, bb] := t1[b, u1, s2, bb] * W2jk[u2, s2]
        t3 = zeros(T, size(t2, 1), size(t2, 2), size(t2, 3), size(GRj, 2))
        @tensor t3[b, u1, u2, c] := t2[b, u1, u2, bb] * GRj[bb, c]
        y += t3
    end
    return y
end
