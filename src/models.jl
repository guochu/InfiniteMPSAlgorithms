# ---------------- physical models (Schur bulk → DenseIMPO; Schur level matrices → SparseIMPO) ----------------
#
# H = Σᵢ h1ᵢ + Σᵢ Σₚ coeffₚ · aₚ,ᵢ ⊗ bₚ,ᵢ₊₁ (on-site + nearest neighbor;
# periodic boundary conditions)

"""
    bulk_mpo(h1, pairs) -> SchurMPOTensor

Build the Schur bulk tensor of an on-site + nearest-neighbor Hamiltonian:

- `h1`: the on-site operator (may be 0);
- `pairs`: a list of `(coeff, a, b)` tuples, corresponding to the
  nearest-neighbor terms `coeff·aᵢ⊗bᵢ₊₁`.
"""
function bulk_mpo(h1, pairs::Vector{<:Tuple})
    N = length(pairs)
    T = scalartype(h1 isa AbstractMatrix ? h1 : Float64)
    for (coeff, a, b) in pairs
        T = promote_type(T, scalartype(a), scalartype(b), typeof(coeff))
    end
    if h1 isa AbstractMatrix
        d = size(h1, 1)
    elseif !isempty(pairs)
        d = size(pairs[1][2], 1)
    else
        d = 2
    end
    cell = Matrix{Any}(undef, N + 2, N + 2)
    cell .= zero(T)
    cell[1, 1] = one(T)
    cell[end, end] = one(T)
    cell[1, end] = h1 isa AbstractMatrix ? h1 : zero(T)
    for (i, (coeff, a, b)) in enumerate(pairs)
        cell[1, i+1] = coeff * a
        cell[i+1, i+1] = zero(T)
        cell[i+1, end] = b
    end
    return SchurMPOTensor(cell)
end

# ---- common operators (Pauli matrices, spin 1/2) ----

σx(::Type{T} = ComplexF64) where {T} = Matrix{T}([0 1; 1 0])
σy(::Type{T} = ComplexF64) where {T} = Matrix{T}([0 -im; im 0])
σz(::Type{T} = ComplexF64) where {T} = Matrix{T}([1 0; 0 -1])
Sx(::Type{T} = ComplexF64) where {T} = σx(T) ./ 2
Sy(::Type{T} = ComplexF64) where {T} = σy(T) ./ 2
Sz(::Type{T} = ComplexF64) where {T} = σz(T) ./ 2

"""
    mpohamiltonian(h1, pairs) -> Matrix

Schur level matrix of an on-site + nearest-neighbor Hamiltonian (for
`SparseIMPO`): levels `1`/`n` are the unit levels, `[1, n] = h1`,
and channel `k`: `[1, k+1] = coeffₖ·aₖ`, `[k+1, n] = bₖ`.
"""
function mpohamiltonian(h1::AbstractMatrix, pairs::Vector{<:Tuple})
    N = length(pairs)
    d = size(h1, 1)
    T = promote_type(eltype(h1), (eltype(a) for (_, a, _) in pairs)...)
    n = N + 2
    W = Matrix{Union{Missing,T,Matrix{T}}}(missing, n, n)
    W[1, 1] = one(T)
    W[n, n] = one(T)
    # wrap in Matrix{T}: operators in pairs may be Adjoints (e.g. the c† of
    # fermi_hubbard); the storage layer requires Matrix{T} rather than lazy
    # wrapper types
    W[1, n] = Matrix{T}(h1)
    for (k, (coeff, a, b)) in enumerate(pairs)
        W[1, k+1] = Matrix{T}(coeff * a)
        W[k+1, n] = Matrix{T}(b)
    end
    return W
end

"""
    heisenberg_hamiltonian(; J=1.0, Δ=1.0, h=0.0, T=ComplexF64) -> SparseIMPO

Schur form of `H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ`.
"""
function heisenberg_hamiltonian(; J::Real = 1.0, Δ::Real = 1.0, h::Real = 0.0,
                                T::Type = ComplexF64)
    W = mpohamiltonian(-h * Sz(T), [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J * Δ, Sz(T), Sz(T))])
    return SparseIMPO([W])
end

"""
    tfim_hamiltonian(; J=1.0, h=1.0, T=ComplexF64) -> SparseIMPO

Schur form of the transverse-field Ising model
`H = −J Σ σˣσˣ − h Σ σᶻ`.
"""
function tfim_hamiltonian(; J::Real = 1.0, h::Real = 1.0, T::Type = ComplexF64)
    W = mpohamiltonian(-h * σz(T), [(-J, σx(T), σx(T))])
    return SparseIMPO([W])
end

"""
    heisenberg_xxz(; J=1.0, Δ=1.0, h=0.0, T=ComplexF64) -> (; mpo, bulk, hamiltonian)

`H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ`. For J=Δ=1 the ground-state energy
density is `1/4 − ln2`. Returns `(; mpo, bulk, hamiltonian)`: `mpo` is the
periodic `DenseIMPO` (的周期 trace 完整收缩适合做**时间演化生成元**，见
`make_time_mpo`)、`bulk` the Schur bulk，`hamiltonian` 是同一模型的
`SparseIMPO`（**求能量用这个**：闭列公式，见 [`DMRGCache`](@ref) 的 `DenseIMPO`
版说明）。
"""
function heisenberg_xxz(; J::Real = 1.0, Δ::Real = 1.0, h::Real = 0.0, T::Type = ComplexF64)
    bulk = bulk_mpo(-h * Sz(T), [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J * Δ, Sz(T), Sz(T))])
    return (; mpo = infinite_mpo(bulk), bulk = bulk,
            hamiltonian = heisenberg_hamiltonian(; J = J, Δ = Δ, h = h, T = T))
end

"""
    tfim(; J=1.0, h=1.0, T=ComplexF64) -> (; mpo, bulk, hamiltonian)

Transverse-field Ising model `H = −J Σ σˣσˣ − h Σ σᶻ`. For J=h=1 the
ground-state energy density is `−4/π`. `mpo` 是周期 `DenseIMPO`（适合做时间演化
生成元，见 `make_time_mpo`）、`bulk` 是 Schur bulk、`hamiltonian` 是同一模型的
`SparseIMPO`（**求能量用这个**：闭列公式，见 [`DMRGCache`](@ref) 的 `DenseIMPO`
版说明）。
"""
function tfim(; J::Real = 1.0, h::Real = 1.0, T::Type = ComplexF64)
    bulk = bulk_mpo(-h * σz(T), [(-J, σx(T), σx(T))])
    return (; mpo = infinite_mpo(bulk), bulk = bulk,
            hamiltonian = tfim_hamiltonian(; J = J, h = h, T = T))
end

"""
    fermi_hubbard(; t=1.0, U=0.0, μ=0.0, T=ComplexF64) -> (; mpo, bulk)

Fermi-Hubbard model with local space `(|0⟩,|↑⟩,|↓⟩,|↑↓⟩)` (JW signs ordered
with ↑<↓):

`H = −t Σσ (c†σ,ᵢ cσ,ᵢ₊₁ + h.c.) + U Σᵢ n↑,ᵢn↓,ᵢ − μ Σᵢ nᵢ`.
"""
function fermi_hubbard(; t::Real = 1.0, U::Real = 0.0, μ::Real = 0.0, T::Type = ComplexF64)
    z, o, m = zero(T), one(T), -one(T)
    # annihilation operators (row = output/bra, column = input/ket)
    a_up = T[z o z z; z z z m; z z z z; z z z z]      # c↑ : |↑⟩→|0⟩, |↑↓⟩→−|↓⟩
    a_dn = T[z z o z; z z z o; z z z z; z z z z]      # c↓ : |↓⟩→|0⟩, |↑↓⟩→+|↑⟩
    c_up = a_up'                                      # c†↑
    c_dn = a_dn'                                      # c†↓
    n = T[0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 2]
    n_updn = T[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
    h1 = U * n_updn - μ * n
    pairs = [(-t, a_up, c_up), (-t, a_dn, c_dn)]
    bulk = bulk_mpo(h1, pairs)
    return (; mpo = infinite_mpo(bulk), bulk = bulk,
            hamiltonian = SparseIMPO([mpohamiltonian(h1, pairs)]))
end
