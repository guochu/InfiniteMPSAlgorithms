# ---------------- 环境求解 kernel（transfermatrix 的环境侧延伸） ----------------
#
# 哈密顿量通道环境构造（DMRGCache，见 groundstates/idmrg.jl）所需的求解原语：
# 非齐次线性解 `linsolve`（Schur 逐 level 通道）、恒等层固定点投影
# `regularize!`、以及 DenseMPO 通道的转移矩阵主本征向量 `dominant_env`。

"`_to3(L)`: `(bra, ket)` 矩阵环境的 rank-3 视图（w 维 = 1）。"
_to3(L::AbstractMatrix{T}) where {T} = reshape(L, size(L, 1), 1, size(L, 2))
_to3(L::AbstractArray{T,3}) where {T} = L

"""
    linsolve(operator, b, x₀, [alg]; a₀ = 1, a₁ = 1) -> (x, info)

Solve the linear system `a₀·x + a₁·A·x = b` (mirrors MPSKit's `linsolve`;
internally KrylovKit.linsolve; `alg` is `GMRES`/`BiCGStab`/`CG`).
"""
function linsolve(operator, b::AbstractVector, x₀::AbstractVector,
                  alg::KrylovKit.KrylovAlgorithm = KrylovKit.GMRES();
                  a₀ = 1, a₁ = 1)
    x, info = KrylovKit.linsolve(v -> operator(v), b, x₀, alg, a₀, a₁)
    return x, info
end

"""
    regularize!(v, lvec, rvec) -> v

Project out the identity-channel fixed-point component (mirrors MPSKit's
`regularize!`): `v ← v − rvec·⟨lvec, v⟩`. In this package's gauge
`lvec = rvec = I`, i.e. `v ← v − tr(v)·I`.
"""
function regularize!(v::AbstractMatrix, lvec::AbstractMatrix, rvec::AbstractMatrix)
    c = sum(lvec .* transpose(v))   # MPSKit semantics: Σ lvec[a,b]·v[b,a] (no conjugation)
    v .-= c .* rvec
    return v
end

function _dominant_env_matvec(op::Union{Nothing,DenseIMPO}, ψ::CanonicalIMPS, side::Symbol)
    N = length(ψ)
    identity = isnothing(op)
    return function matvec(v::AbstractVector)
        if identity
            L = reshape(v, size(ψ.AL[1], 1), size(ψ.AL[1], 1))
            if side === :left
                for ℓ in 1:N
                    L = push_env_left(L, ψ.AL[ℓ])
                end
            else
                R = L
                for ℓ in N:-1:1
                    R = push_env_right(R, ψ.AR[ℓ])
                end
                L = R
            end
            return vec(L)
        else
            W1 = op[1]
            L = reshape(v, size(ψ.AL[1], 1), size(W1, 1), size(ψ.AL[1], 1))
            if side === :left
                for ℓ in 1:N
                    L = push_env_left(L, op[ℓ], ψ.AL[ℓ])
                end
            else
                R = L
                for ℓ in N:-1:1
                    R = push_env_right(R, op[ℓ], ψ.AR[ℓ])
                end
                L = R
            end
            return vec(L)
        end
    end
end

"""
    dominant_env(ψ; side=:left, which=:LM, kwargs...) -> (λ, L)
    dominant_env(W, ψ; side=:left, which=:LM, kwargs...) -> (λ, L)

Dominant eigenvector of the identity/MPO-channel transfer matrix (tiled over
one unit cell). The identity channel uses AL/AR (strictly canonical) and
returns `λ ≈ 1`.
"""
function dominant_env(ψ::CanonicalIMPS; side::Symbol = :left, which::Symbol = :LM, kwargs...)
    return dominant_env(nothing, ψ; side = side, which = which, kwargs...)
end

function dominant_env(op::Union{Nothing,DenseIMPO}, ψ::CanonicalIMPS;
                      side::Symbol = :left, which::Symbol = :LM,
                      tol::Real = Defaults.tol, krylovdim::Int = Defaults.krylovdim,
                      maxiter::Int = Defaults.maxiter)
    identity = isnothing(op)
    D = size(ψ.AL[1], 1)
    dim = identity ? D * D : D * size(op[1], 1) * D
    T = scalartype(ψ)
    matvec = _dominant_env_matvec(op, ψ, side)
    v0 = ones(T, dim)
    λs, vs, _ = _eigsolve(matvec, v0, 1, which; ishermitian = false, tol = tol,
                          krylovdim = krylovdim, maxiter = maxiter)
    λ = λs[1]
    L = identity ? reshape(vs[1], D, D) : reshape(vs[1], D, size(op[1], 1), D)
    L ./= norm(L)
    return λ, L
end

# ---- per-level kernels of the Hamiltonian path (exclusive to the DMRGCache constructor) ----

"Left push of the channel slice (l → i): L′ = Σ conj(A[a,ū,a′])·L[a,b]·Wl[ū,d]·A[b,d,b′]."
function _push_slice_left(L::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, ū, a′]) * L[a, b] * Wl[ū, d] * A[b, d, b′]
    return L′
end

"Right push of the channel slice (i → l): mirrors MPSKit transfer_right; A
contracts with the ket (d), conj(A) with the bra (u)."
function _push_slice_right(R::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, d, a] * Wl[ū, d] * conj(A[b′, ū, b]) * R[a, b]
end

"Full-cell left sweep (mirrors MPSKit left_cyclethrough!):
`GL[site+1, i] = Σ_{l≤i} W_site[l→i]·GL[site, l]`."
function _left_cyclethrough!(lefts, Wds, ALs, i::Int, N::Int, Ds, T)
    for site in 1:N
        snext = site == N ? 1 : site + 1
        # 目标键 = site 的右键（逐站键维不同，取张量自身的实际尺寸）
        χs = size(ALs[site], 3)
        tgt = zeros(T, χs, χs)
        for l in 1:i
            tgt .+= _push_slice_left(lefts[site][:, l, :],
                                     view(Wds[site], l, :, i, :), ALs[site])
        end
        lefts[snext][:, i, :] .= tgt
    end
    return lefts
end

"Full-cell right sweep (mirrors MPSKit right_cyclethrough!):
`GR[site−1, i] = Σ_{l≥i} W_site[i→l]·GR[site, l]`."
function _right_cyclethrough!(rights, Wds, ARs, i::Int, N::Int, Ds, T)
    nl = size(Wds[1], 1)
    for site in N:-1:1
        sprev = site == 1 ? N : site - 1
        # 目标键 = site 的左键（逐站键维不同，取张量自身的实际尺寸）
        χs = size(ARs[site], 1)
        tgt = zeros(T, χs, χs)
        for l in i:nl
            tgt .+= _push_slice_right(rights[site][:, l, :],
                                      view(Wds[site], i, :, l, :), ARs[site])
        end
        rights[sprev][:, i, :] .= tgt
    end
    return rights
end

# ---- environment normalization and incremental pushes ----

"""
    normalize_envs!(envs, below, operator, above) -> envs

Mirrors MPSKit's `normalize!`:
- right environments are normalized to unit norm;
- left environments are scaled such that `dot(C, C_hamiltonian * C) = 1`.

Prevents environment-norm blowup during IDMRG sweeps.
"""
function normalize_envs!(envs::Environments, below::CanonicalIMPS,
                    operator::Union{Nothing,DenseIMPO,SparseIMPO},
                    above::CanonicalIMPS)
    N = length(below)
    for i in 1:N
        # normalize the right environment to unit norm
        r = envs.rights[i]
        nr = norm(r)
        if nr > 0
            r ./= nr
        end
        # λ = dot(C[i], C_hamiltonian(i) * C[i])
        hC = C_hamiltonian(i, below, operator, above, envs)
        Cproj = hC(below.C[i])
        λ = dot(below.C[i], Cproj)
        if abs(λ) > 0
            # GL[i+1] *= inv(λ)
            l = envs.lefts[_mod1(i + 1, N)]
            l ./= λ
        end
    end
    return envs
end

"""
    transfer_leftenv!(envs, ψ, operator, ψ2, site) -> envs

Push the left environment from `site − 1` to `site` (the incremental update of
the IDMRG sweep, mirroring MPSKit's `transfer_leftenv!`).
"""
function transfer_leftenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓm = _mod1(site - 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa SparseIMPO
        tompotensor(operator[ℓm])
    else
        operator[ℓm]
    end
    if isnothing(W)
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], ψ.AL[ℓm])
    else
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], W, ψ.AL[ℓm])
    end
    return envs
end

"Push the right environment from `site + 1` to `site`."
function transfer_rightenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓp = _mod1(site + 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa SparseIMPO
        tompotensor(operator[ℓp])
    else
        operator[ℓp]
    end
    if isnothing(W)
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ψ.AR[ℓp])
    else
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], W, ψ.AR[ℓp])
    end
    return envs
end
