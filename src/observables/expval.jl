# ---------------- 期望值（对标 MPSKit src/algorithms/expval.jl） ----------------

"""
    expectation_value(ψ, O, [environments])
    expectation_value(ψ, inds => O)

算符期望值：

- `O::InfiniteMPO`：MPO 全收缩，返回单位胞总期望（如 `expectation_value(ψ, H)` = 总能量）；
- `O = (i,) => A`：单 site 局域算符；
- `O = ((i, i+1),) => A12`：相邻双 site 算符（`A12` 为 `d²×d²` 矩阵，
  指标序 `(u1, u2; d1, d2)`）；
- `expectation_value(ψ)`：`⟨ψ|ψ⟩`。
"""
function expectation_value end

expectation_value(ψ::MixedCanonicalMPS) = dot(ψ, ψ)

expectation_value(ψ::MixedCanonicalMPS, operator::Nothing, envs...) = dot(ψ, ψ)

function expectation_value(ψ::MixedCanonicalMPS, (inds, O)::Pair)
    sites = Tuple(inds)
    (length(sites) == 1) && return _local_expectation1(ψ, sites[1], O)
    (length(sites) == 2) && return _local_expectation2(ψ, sites[1], sites[2], O)
    throw(ArgumentError("仅支持单 site 或相邻双 site 局域算符"))
end

function _local_expectation1(ψ::MixedCanonicalMPS, site::Int, O::AbstractMatrix)
    AC = ψ.AC[site]
    @tensor E[] := conj(AC[a, u, b]) * O[u, s] * AC[a, s, b]
    return E[] / dot(ψ, ψ)
end

function _local_expectation2(ψ::MixedCanonicalMPS, i::Int, j::Int, O12::AbstractMatrix)
    N = length(ψ)
    (j == _mod1(i + 1, N)) || throw(ArgumentError("双 site 算符仅支持相邻 site"))
    d1 = size(ψ.AC[i], 2)
    d2 = size(ψ.AR[j], 2)
    (size(O12, 1) == size(O12, 2) == d1 * d2) ||
        throw(DimensionMismatch("O12 应为 $(d1*d2)×$(d1*d2) 矩阵"))
    T = reshape(O12, d1, d2, d1, d2)   # (u1, u2; d1, d2)
    AC = ψ.AC[i]
    AR = ψ.AR[j]
    # 混合规范下 AC[i] 左环境与 AR[j] 右环境均为 I：bra/ket 链上
    # AC 右键与 AR 左键直接相连（对标 MPSKit expectation_value (i,j)=>O）
    @tensor E[] := conj(AC[a, u1, x]) * conj(AR[x, u2, b̄]) * T[u1, u2, d1, d2] *
                   AC[a, d1, y] * AR[y, d2, b̄]
    return E[] / dot(ψ, ψ)
end

"MPS-MPO-MPS 单 site 三明治收缩（对标 MPSKit 的 contract_mpo_expval）。"
function contract_mpo_expval(AC, GL, O, GR, ACbar = AC)
    @tensor E[] := GL[ā, w, b] * AC[b, d, b′] * GR[b′, w′, b̄] *
                   O[w, ū, w′, d] * conj(ACbar[ā, ū, b̄])
    return E[]
end

function expectation_value(ψ::MixedCanonicalMPS, mpo::InfiniteMPO,
                           envs::Environments = environments(ψ, mpo))
    N = length(ψ)
    E = zero(promote_type(scalartype(ψ), scalartype(mpo)))
    for ℓ in 1:N
        E += contract_mpo_expval(ψ.AC[ℓ], leftenv(envs, ℓ), mpo[ℓ], rightenv(envs, ℓ))
    end
    return E / dot(ψ, ψ)
end

"""
    expectation_value(ψ, H::MPOHamiltonian, [envs])

Jordan 哈密顿量能量（对标 MPSKit）：每 site 只收缩**闭合列**
`H[site][:, 1, 1, end]`（on-site `D`、闭合 `B`、恒等簿记项），即

```julia
E = Σ_site Σ_l ⟨GL_l · W[l → end] · GR_end⟩
```

恒等层环境的固定点分量已在环境构建中投影掉，`(end → end)` 项因此为零。
"""
function expectation_value(ψ::MixedCanonicalMPS, H::MPOHamiltonian,
                           envs::Environments = environments(ψ, H))
    N = length(ψ)
    nl = mpobond(H)
    d = phydim(H)
    T = promote_type(scalartype(ψ), scalartype(H))
    E = zero(T)
    for ℓ in 1:N
        GL = leftenv(envs, ℓ)
        GR = rightenv(envs, ℓ)
        Dl, _, Dk = size(GL)
        GRe = reshape(GR[:, nl, :], Dk, 1, Dl)
        Wcol = zeros(T, nl, d, 1, d)
        for l in 1:nl
            Wcol[l, :, 1, :] .= H[ℓ][l, nl]
        end
        E += contract_mpo_expval(ψ.AC[ℓ], GL, Wcol, GRe)
    end
    return E / dot(ψ, ψ)
end
