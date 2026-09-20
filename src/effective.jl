# ---------------- 有效局域哈密顿量（对标 MPSKit AC_hamiltonian / C_hamiltonian） ----------------
#
# 环境指标约定（与 MPSKit 一致）：
#   GL = leftenv[ℓ]  = (bra键, w, ket键)
#   GR = rightenv[ℓ] = (ket键, w, bra键)
# 有效哈密顿量为严格的线性算子（不对 x 取共轭，MPSKit 形式；复数下亦正确）。

"""
    MPODerivativeOperator(leftenv, operators::Tuple, rightenv)

MPS-MPO-MPS 三明治对局域张量求导得到的有效算子（对标 MPSKit 同名结构）。
`operators` 为空 → C 问题；单算子 → AC 问题；双算子 → AC2 问题。
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

bond `site` 的有效哈密顿量（对标 MPSKit）：使用 `leftenv(envs, site + 1)` 与
`rightenv(envs, site)`，作用为

```julia
C′[α, β] = Σ GL[α, w, α′] · C[α′, β′] · GR[β′, w, β]
```
"""
function C_hamiltonian(site::Int, below, operator, above, envs::Environments)
    return MPO_C_Hamiltonian(leftenv(envs, site + 1), rightenv(envs, site))
end

"""
    AC_hamiltonian(site, below, operator, above, envs) -> callable

site `site` 的有效哈密顿量（对标 MPSKit）：使用 `leftenv(envs, site)`、
`operator[site]` 与 `rightenv(envs, site)`，作用为

```julia
AC′[b, u_out, b′] = Σ GL[b, wl, ā] · x[ā, u_in, b̄] · W[wl, u_out, wr, u_in] · GR[b̄, wr, b′]
```
"""
function AC_hamiltonian(site::Int, below, operator, above, envs::Environments)
    O = if isnothing(operator)
        nothing
    elseif operator isa MPOHamiltonian
        tompotensor(operator[site])   # Jordan 张量稠密化后进入统一 kernel
    else
        operator[site]
    end
    return MPO_AC_Hamiltonian(leftenv(envs, site), O, rightenv(envs, site))
end

# ---- 作用（MPSKit 形式，线性算子） ----

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
