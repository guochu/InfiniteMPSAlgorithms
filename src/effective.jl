# ---------------- effective local Hamiltonians (mirroring MPSKit AC_hamiltonian / C_hamiltonian) ----------------
#
# Environment index conventions (consistent with MPSKit):
#   GL = leftenv[ℓ]  = (bra bond, w, ket bond)
#   GR = rightenv[ℓ] = (ket bond, w, bra bond)
# The effective Hamiltonians are strictly linear operators (no conjugation of
# x, MPSKit form; correct for complex data as well).

"""
    MPODerivativeOperator(leftenv, operators::Tuple, rightenv)

Effective operator obtained by differentiating an MPS-MPO-MPS sandwich with
respect to the local tensor (mirrors MPSKit's structure of the same name).
Empty `operators` → the C problem; one operator → the AC problem; two
operators → the AC2 problem.
"""
struct MPODerivativeOperator{L,O<:Tuple,R}
    leftenv::L
    operators::O
    rightenv::R
end

const MPO_C_Hamiltonian{L,R} = MPODerivativeOperator{L,Tuple{},R}
const MPO_AC_Hamiltonian{L,O,R} = MPODerivativeOperator{L,Tuple{O},R}
MPO_C_Hamiltonian(GL, GR) = MPODerivativeOperator(GL, (), GR)
MPO_AC_Hamiltonian(GL, O, GR) = MPODerivativeOperator(GL, (O,), GR)

"""
    C_hamiltonian(site, below, operator, above, envs) -> callable

Effective Hamiltonian on bond `site` (mirrors MPSKit): uses
`leftenv(envs, site + 1)` and `rightenv(envs, site)`, with action

```julia
C′[α, β] = Σ GL[α, w, α′] · C[α′, β′] · GR[β′, w, β]
```
"""
function C_hamiltonian(site::Int, below, operator, above, envs::Environments)
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
function AC_hamiltonian(site::Int, below, operator, above, envs::Environments)
    O = if isnothing(operator)
        nothing
    elseif operator isa SparseIMPO
        tompotensor(operator[site])   # densify the Schur tensor into the unified kernel
    else
        operator[site]
    end
    return MPO_AC_Hamiltonian(leftenv(envs, site), O, rightenv(envs, site))
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

function (h::MPO_AC_Hamiltonian)(x::AbstractArray{T,3}) where {T}
    GL, GR = h.leftenv, h.rightenv
    W = only(h.operators)
    @tensor y[b, u_out, b′] := GL[b, wl, ā] * x[ā, u_in, b̄] *
                               W[wl, u_out, wr, u_in] * GR[b̄, wr, b′]
    return y
end
