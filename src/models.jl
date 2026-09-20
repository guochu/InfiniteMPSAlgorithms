# ---------------- 物理模型（Jordan bulk → InfiniteMPO；Jordan 层矩阵 → MPOHamiltonian） ----------------
#
# H = Σᵢ h1ᵢ + Σᵢ Σₚ coeffₚ · aₚ,ᵢ ⊗ bₚ,ᵢ₊₁ （on-site + 最近邻；周期边界）

"""
    bulk_mpo(h1, pairs) -> JordanMPOTensor

构造 on-site + 最近邻 Hamiltonian 的 Jordan bulk 张量：

- `h1`：on-site 算符（可为 0）；
- `pairs`：`(coeff, a, b)` 元组列表，对应 `coeff·aᵢ⊗bᵢ₊₁` 最近邻项。
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
    return JordanMPOTensor(cell)
end

# ---- 常用算符（Pauli，自旋 1/2） ----

σx(::Type{T} = ComplexF64) where {T} = Matrix{T}([0 1; 1 0])
σy(::Type{T} = ComplexF64) where {T} = Matrix{T}([0 -im; im 0])
σz(::Type{T} = ComplexF64) where {T} = Matrix{T}([1 0; 0 -1])
Sx(::Type{T} = ComplexF64) where {T} = σx(T) ./ 2
Sy(::Type{T} = ComplexF64) where {T} = σy(T) ./ 2
Sz(::Type{T} = ComplexF64) where {T} = σz(T) ./ 2

"""
    mpohamiltonian(h1, pairs) -> Matrix

on-site + 最近邻 Hamiltonian 的 Jordan 层矩阵（供 `InfiniteMPOHamiltonian`）：
层 `1`/`n` 为单位层，`[1, n] = h1`，通道 `k`：`[1, k+1] = coeffₖ·aₖ`、
`[k+1, n] = bₖ`。
"""
function mpohamiltonian(h1::AbstractMatrix, pairs::Vector{<:Tuple})
    N = length(pairs)
    d = size(h1, 1)
    T = promote_type(eltype(h1), (eltype(a) for (_, a, _) in pairs)...)
    n = N + 2
    W = Matrix{Union{Missing,T,Matrix{T}}}(missing, n, n)
    W[1, 1] = one(T)
    W[n, n] = one(T)
    # 包 Matrix{T}：pairs 中的算符可为 Adjoint（如 fermi_hubbard 的 c†），
    # 存储层要求 Matrix{T} 而非惰性包装类型
    W[1, n] = Matrix{T}(h1)
    for (k, (coeff, a, b)) in enumerate(pairs)
        W[1, k+1] = Matrix{T}(coeff * a)
        W[k+1, n] = Matrix{T}(b)
    end
    return W
end

"""
    heisenberg_hamiltonian(; J=1.0, Δ=1.0, h=0.0, T=ComplexF64) -> InfiniteMPOHamiltonian

`H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ` 的 Jordan 形式。
"""
function heisenberg_hamiltonian(; J::Real = 1.0, Δ::Real = 1.0, h::Real = 0.0,
                                T::Type = ComplexF64)
    W = mpohamiltonian(-h * Sz(T), [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J * Δ, Sz(T), Sz(T))])
    return InfiniteMPOHamiltonian([W])
end

"""
    tfim_hamiltonian(; J=1.0, h=1.0, T=ComplexF64) -> InfiniteMPOHamiltonian

横场 Ising 模型 `H = −J Σ σˣσˣ − h Σ σᶻ` 的 Jordan 形式。
"""
function tfim_hamiltonian(; J::Real = 1.0, h::Real = 1.0, T::Type = ComplexF64)
    W = mpohamiltonian(-h * σz(T), [(-J, σx(T), σx(T))])
    return InfiniteMPOHamiltonian([W])
end

"""
    heisenberg_xxz(; J=1.0, Δ=1.0, h=0.0, T=ComplexF64) -> (; mpo, bulk, hamiltonian)

`H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ`。J=Δ=1 时基态能量密度 `1/4 − ln2`。
返回 `(; mpo, bulk)`：`mpo` 为周期 `InfiniteMPO`，`bulk` 为 Jordan bulk
（供 `make_time_mpo` 做时间演化）。
"""
function heisenberg_xxz(; J::Real = 1.0, Δ::Real = 1.0, h::Real = 0.0, T::Type = ComplexF64)
    bulk = bulk_mpo(-h * Sz(T), [(J, Sx(T), Sx(T)), (J, Sy(T), Sy(T)), (J * Δ, Sz(T), Sz(T))])
    return (; mpo = infinite_mpo(bulk), bulk = bulk,
            hamiltonian = heisenberg_hamiltonian(; J = J, Δ = Δ, h = h, T = T))
end

"""
    tfim(; J=1.0, h=1.0, T=ComplexF64) -> (; mpo, bulk)

横场 Ising 模型 `H = −J Σ σˣσˣ − h Σ σᶻ`。J=h=1 时基态能量密度 `−4/π`。
"""
function tfim(; J::Real = 1.0, h::Real = 1.0, T::Type = ComplexF64)
    bulk = bulk_mpo(-h * σz(T), [(-J, σx(T), σx(T))])
    return (; mpo = infinite_mpo(bulk), bulk = bulk,
            hamiltonian = tfim_hamiltonian(; J = J, h = h, T = T))
end

"""
    fermi_hubbard(; t=1.0, U=0.0, μ=0.0, T=ComplexF64) -> (; mpo, bulk)

Fermi-Hubbard 模型，局域空间 `(|0⟩,|↑⟩,|↓⟩,|↑↓⟩)`（JW 符号按 ↑<↓ 排序）：

`H = −t Σσ (c†σ,ᵢ cσ,ᵢ₊₁ + h.c.) + U Σᵢ n↑,ᵢn↓,ᵢ − μ Σᵢ nᵢ`。
"""
function fermi_hubbard(; t::Real = 1.0, U::Real = 0.0, μ::Real = 0.0, T::Type = ComplexF64)
    z, o, m = zero(T), one(T), -one(T)
    # 湮灭算符（行 = 输出/bra，列 = 输入/ket）
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
            hamiltonian = InfiniteMPOHamiltonian([mpohamiltonian(h1, pairs)]))
end
