# ---------------- effective local Hamiltonians (mirroring MPSKit AC_hamiltonian / C_hamiltonian) ----------------
#
# Environment index conventions (consistent with MPSKit):
#   GL = leftenv[ℓ]  = (bra bond, w, ket bond)
#   GR = rightenv[ℓ] = (ket bond, w, bra bond)
# The effective Hamiltonians are strictly linear operators (no conjugation of
# x, MPSKit form; correct for complex data as well).

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
# Schur 上三角结构的零块从不落地），语义与稠密张量收缩逐位一致
function (h::MPO_AC_Hamiltonian{L,Op,R})(x::AbstractArray{T,3}) where {L,Op<:SchurMPOTensor,R,T}
    GL, GR, W = h.leftenv, h.rightenv, h.operators
    nl = nlvls(W)
    y = zeros(T, size(GL, 1), size(x, 2), size(GR, 3))
    for i in 1:nl, j in 1:nl
        Wij = W[i, j]                 # (u_out, u_in) 块
        iszero(Wij) && continue
        GLi = @view GL[:, i, :]       # (bra bond, ket bond)
        GRj = @view GR[:, j, :]
        @tensor y[b, u_out, c] += GLi[b, a] * x[a, u_in, bb] *
                                  Wij[u_out, u_in] * GRj[bb, c]
    end
    return y
end
