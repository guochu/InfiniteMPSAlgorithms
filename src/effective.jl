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
`operator === nothing` → the operator-free (identity-channel) action; a single
rank-4 MPO tensor (an `SparseIMPO` Schur tensor is densified on construction)
→ the AC action.
"""
struct MPO_AC_Hamiltonian{L<:MPSTensor,O<:Union{Nothing,MPOTensor,SchurMPOTensor},R<:MPSTensor}
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
function C_hamiltonian(site::Int, below, operator, above, envs::Environments)
    # 恒等通道（OverlapCache）的环境为 rank-2 矩阵，经 _to3 升为 (b, 1, b') 后进统一 kernel
    return MPO_C_Hamiltonian(_to3(leftenv(envs, site + 1)), _to3(rightenv(envs, site)))
end

"""
    AC_hamiltonian(site, below, operator, above, envs) -> callable

Effective Hamiltonian on site `site` (mirrors MPSKit): uses
`leftenv(envs, site)`, `operator[site]`, and `rightenv(envs, site)`, with action

```julia
AC′[b, u_out, b′] = Σ GL[b, wl, ā] · x[ā, u_in, b̄] · W[wl, u_out, wr, u_in] · GR[b̄, wr, b′]
```
"""
function AC_hamiltonian(site::Int, below, operator, above, envs::Environments)
    O = if isnothing(operator)
        nothing
    elseif operator isa SparseIMPO
        tompotensor(operator[site])   # densify the Schur tensor into the unified kernel
    else
        operator[site]
    end
    return MPO_AC_Hamiltonian(_to3(leftenv(envs, site)), O, _to3(rightenv(envs, site)))
end

# ---- action (MPSKit form, linear operators) ----

function (h::MPO_C_Hamiltonian)(x::AbstractMatrix{T}) where {T}
    GL, GR = h.leftenv, h.rightenv
    @tensor y[α, β] := GL[α, w, α′] * x[α′, β′] * GR[β′, w, β]
    return y
end

function (h::MPO_AC_Hamiltonian{L,Nothing,R})(x::AbstractArray{T,3}) where {L,R,T}
    GL, GR = h.leftenv, h.rightenv
    @tensor y[b, d, b′] := GL[b, 1, ā] * x[ā, d, b̄] * GR[b̄, 1, b′]
    return y
end

# 泛化方法约束 W <: MPOTensor（与 Nothing 特化方法互斥，杜绝分派歧义）
function (h::MPO_AC_Hamiltonian{L,W,R})(x::AbstractArray{T,3}) where {L,W<:MPOTensor,R,T}
    GL, GR = h.leftenv, h.rightenv
    W4 = h.operators
    @tensor y[b, u_out, b′] := GL[b, wl, ā] * x[ā, u_in, b̄] *
                               W4[wl, u_out, wr, u_in] * GR[b̄, wr, b′]
    return y
end
