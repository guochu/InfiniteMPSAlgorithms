# ---------------- expectation values (mirrors MPSKit src/algorithms/expval.jl) ----------------

"""
    expectationvalue(ψ, O, [envs])
    expectationvalue(ψ, inds => O)

Operator expectation values:

- `O::DenseIMPO`: full MPO contraction, returning the total expectation over
  the unit cell (e.g. `expectationvalue(ψ, H)` = total energy)。
  **注意**：这是周期 trace 的完整收缩（MPSKit `expectation_value(ψ, ::InfiniteMPO)`
  语义，恒等 MPO 给 N）；对 Hamiltonian 型 MPO 它不等于能量（多出恒等层
  bookkeeping），能量请用 `O::SparseIMPO`（闭列公式，对齐 MPSKit 的
  `InfiniteMPOHamiltonian`），详见 [`DMRGCache`](@ref) 的 `DenseIMPO` 版说明；
- `O = (i,) => A`: single-site local operator;
- `O = ((i, i+1),) => A12`: nearest-neighbor two-site operator (`A12` is a
  `d²×d²` matrix with index order `(u1, u2; d1, d2)`);
- `expectationvalue(ψ)`: `⟨ψ|ψ⟩`.
"""
function expectationvalue end

expectationvalue(ψ::CanonicalIMPS) = dot(ψ, ψ)

expectationvalue(ψ::CanonicalIMPS, operator::Nothing, envs...) = dot(ψ, ψ)

function expectationvalue(ψ::CanonicalIMPS, (inds, O)::Pair)
    sites = Tuple(inds)
    (length(sites) == 1) && return _local_expectation1(ψ, sites[1], O)
    (length(sites) == 2) && return _local_expectation2(ψ, sites[1], sites[2], O)
    throw(ArgumentError("only single-site or nearest-neighbor two-site local operators are supported"))
end

function _local_expectation1(ψ::CanonicalIMPS, site::Int, O::AbstractMatrix)
    AC = ψ.AC[site]
    @tensor E[] := conj(AC[a, u, b]) * O[u, s] * AC[a, s, b]
    return E[] / dot(ψ, ψ)
end

function _local_expectation2(ψ::CanonicalIMPS, i::Int, j::Int, O12::AbstractMatrix)
    N = length(ψ)
    (j == _mod1(i + 1, N)) || throw(ArgumentError("two-site operators only support adjacent sites"))
    d1 = size(ψ.AC[i], 2)
    d2 = size(ψ.AR[j], 2)
    (size(O12, 1) == size(O12, 2) == d1 * d2) ||
        throw(DimensionMismatch("O12 must be a $(d1*d2)×$(d1*d2) matrix"))
    T = reshape(O12, d1, d2, d1, d2)   # (u1, u2; d1, d2)
    AC = ψ.AC[i]
    AR = ψ.AR[j]
    # in the mixed canonical form the left environment of AC[i] and the right
    # environment of AR[j] are both I: on the bra/ket chains the AC right bond
    # connects directly to the AR left bond (mirrors MPSKit expectation_value
    # (i,j)=>O)
    @tensor E[] := conj(AC[a, u1, x]) * conj(AR[x, u2, b̄]) * T[u1, u2, d1, d2] *
                   AC[a, d1, y] * AR[y, d2, b̄]
    return E[] / dot(ψ, ψ)
end

"MPS-MPO-MPS single-site sandwich contraction (mirrors MPSKit's
contract_mpo_expval)."
function contract_mpo_expval(AC, GL, O, GR, ACbar = AC)
    @tensor E[] := GL[ā, w, b] * AC[b, d, b′] * GR[b′, w′, b̄] *
                   O[w, ū, w′, d] * conj(ACbar[ā, ū, b̄])
    return E[]
end

function expectationvalue(ψ::CanonicalIMPS, mpo::DenseIMPO,
                          envs::Environments = DMRGCache(ψ, mpo))
    N = length(ψ)
    E = zero(promote_type(scalartype(ψ), scalartype(mpo)))
    for ℓ in 1:N
        E += contract_mpo_expval(ψ.AC[ℓ], leftenv(envs, ℓ), mpo[ℓ], rightenv(envs, ℓ))
    end
    return E / dot(ψ, ψ)
end

"""
    expectationvalue(ψ, H::SparseIMPO, [envs])

Schur-Hamiltonian energy (mirrors MPSKit): per site only the **closed
column** `H[site][:, 1, 1, end]` is contracted (the on-site `D`, the closing
`B`, and the identity bookkeeping terms), i.e.

```julia
E = Σ_site Σ_l ⟨GL_l · W[l → end] · GR_end⟩
```

The fixed-point components of the identity-level environments were projected
out during environment construction, so the `(end → end)` terms vanish.
"""
function expectationvalue(ψ::CanonicalIMPS, H::SparseIMPO,
                          envs::Environments = DMRGCache(ψ, H))
    N = length(ψ)
    nl = bonddim(H)
    T = promote_type(scalartype(ψ), scalartype(H))
    E = zero(T)
    for ℓ in 1:N
        GL = leftenv(envs, ℓ)        # 键 ℓ-1 上的左环境
        GR = rightenv(envs, ℓ)       # 键 ℓ 上的右环境（非均匀键下与 GL 维数不同）
        d = size(ψ.AC[ℓ], 2)         # 逐站物理维
        GRe = reshape(GR[:, nl, :], size(GR, 1), 1, size(GR, 3))
        Wcol = zeros(T, nl, d, 1, d)
        for l in 1:nl
            Wcol[l, :, 1, :] .= H[ℓ][l, nl]
        end
        E += contract_mpo_expval(ψ.AC[ℓ], GL, Wcol, GRe)
    end
    return E / dot(ψ, ψ)
end
