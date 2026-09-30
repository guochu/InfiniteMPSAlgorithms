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
`operator` is a single rank-4 MPO tensor (an `SparseIMPO` Schur tensor is
densified on construction).
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
    O = operator isa SparseIMPO ?
        tompotensor(operator[site]) : operator[site]   # Schur 张量稠密化进统一 kernel
    return MPO_AC_Hamiltonian(leftenv(envs, site), O, rightenv(envs, site))
end

# ---- action (MPSKit form, linear operators) ----

function (h::MPO_C_Hamiltonian)(x::AbstractMatrix{T}) where {T}
    GL, GR = h.leftenv, h.rightenv
    @tensor y[α, β] := GL[α, w, α′] * x[α′, β′] * GR[β′, w, β]
    return y
end

function (h::MPO_AC_Hamiltonian)(x::AbstractArray{T,3}) where {T}
    GL, GR = h.leftenv, h.rightenv
    W = h.operators
    @tensor y[b, u_out, b′] := GL[b, wl, ā] * x[ā, u_in, b̄] *
                               W[wl, u_out, wr, u_in] * GR[b̄, wr, b′]
    return y
end
