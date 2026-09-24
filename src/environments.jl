# ---------------- environment machinery (mirroring MPSKit src/environments/infinite_envs.jl) ----------------
#
# Concrete cache types are defined together with their algorithms (the
# constructors in the algorithm files directly produce the caches):
# - `DMRGCache`: Hamiltonian channel ⟨ψ|H|ψ⟩ (ground-state VUMPS / IDMRG / TDVP
#   and energy evaluation), see algorithms/idmrg.jl;
# - `MultCache`: MPO-application channel ⟨bra|W|ket⟩ (iterative MPO
#   multiplication mult), see algorithms/mult.jl;
# - `OverlapCache`: pure overlap channel ⟨bra|ket⟩ (variational compression of
#   the algebra operations), see algorithms/arithmetics.jl.
#
# This file only keeps the shared machinery: the abstract supertype
# `Environments`, environment access and incremental pushes, and the fixed-point
# kernel shared by the cache constructors.

"""
    Environments

Abstract supertype of the left/right fixed-point environments of
⟨below|operator|ket⟩-type double-layer networks (mirroring MPSKit's
InfiniteEnvironments); concrete types: [`DMRGCache`](@ref) (Hamiltonian
channel), [`MultCache`](@ref) (MPO application) and [`OverlapCache`](@ref)
(pure overlap).
"""
abstract type Environments end

"leftenv(envs, ℓ): the left environment of site ℓ."
leftenv(envs::Environments, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs.ket))]
"rightenv(envs, ℓ): the right environment of site ℓ."
rightenv(envs::Environments, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs.ket))]

_to3(L::AbstractMatrix{T}) where {T} = reshape(L, size(L, 1), 1, size(L, 2))
_to3(L::AbstractArray{T,3}) where {T} = L

# ---- ternary fixed-point kernel (shared by the MultCache / OverlapCache constructors) ----

"Shared fixed-point solver of the ternary environments (left/right dominant
eigenvectors + MPSKit-style normalization). `GL0`/`GR0` optionally warm start
the eigsolves with the previous environments: for block-degenerate targets the
fixed-point space is multi-dimensional and a continuous initial guess keeps the
ALS iteration stable."
function _ternary_fixedpoints(below::CanonicalIMPS, operator, above::CanonicalIMPS;
                              tol::Real, krylovdim::Int, maxiter::Int,
                              GL0::Union{Nothing,AbstractArray} = nothing,
                              GR0::Union{Nothing,AbstractArray} = nothing)
    N = length(below)
    L = isnothing(operator) ? N : length(operator)
    (N % L == 0 && length(above) == N) ||
        throw(DimensionMismatch("incompatible unit-cell lengths of MPS and MPO"))
    T = promote_type(scalartype(below), scalartype(above))
    Dw = isnothing(operator) ? 1 : size(operator[1], 1)
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)
    Dr = size(below.AR[1], 3)
    Wop = isnothing(operator) ? (ℓ -> nothing) : (ℓ -> operator[_mod1(ℓ, L)])

    # ---- left fixed point: dominant eigenvector of T_L(above.AL, operator, below.AL) ----
    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Dw, Da)
        for ℓ in 1:N
            W = Wop(ℓ)
            GL = isnothing(W) ? push_env_left(GL, below.AL[ℓ], above.AL[ℓ]) :
                 push_env_left(GL, below.AL[ℓ], W, above.AL[ℓ])
        end
        return vec(GL)
    end
    v0L = GL0 === nothing ? ones(T, Dl * Dw * Da) : vec(copy(GL0))
    _, GL1 = eigsolve(Tleft, v0L, 1, :LM; ishermitian = false,
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    # 复环境提升（MPSKit 对齐：环境张量按 eigsolve 返回的实际 eltype 存放；
    # 实输入下融合转移的 leading vector 可为复，通道随后整体升为复算术）
    TCL = promote_type(T, eltype(GL1[1]))
    GLs = Vector{Array{TCL,3}}(undef, N)
    GLs[1] = reshape(GL1[1], Dl, Dw, Da)
    for ℓ in 2:N
        W = Wop(ℓ - 1)
        GLs[ℓ] = isnothing(W) ? push_env_left(GLs[ℓ-1], below.AL[ℓ-1], above.AL[ℓ-1]) :
                 push_env_left(GLs[ℓ-1], below.AL[ℓ-1], W, above.AL[ℓ-1])
    end

    # ---- right fixed point: dominant eigenvector of T_R(above.AR, operator, below.AR) ----
    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dw, Dr)
        for ℓ in N:-1:1
            W = Wop(ℓ)
            GR = isnothing(W) ? push_env_right(GR, above.AR[ℓ], below.AR[ℓ]) :
                 push_env_right(GR, above.AR[ℓ], W, below.AR[ℓ])
        end
        return vec(GR)
    end
    v0R = GR0 === nothing ? ones(T, Da * Dw * Dr) : vec(copy(GR0))
    _, GRN = eigsolve(Tright, v0R, 1, :LM; ishermitian = false,
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    TCR = promote_type(T, eltype(GRN[1]))
    GRs = Vector{Array{TCR,3}}(undef, N)
    GRs[N] = reshape(GRN[1], Da, Dw, Dr)
    for ℓ in N-1:-1:1
        W = Wop(ℓ + 1)
        GRs[ℓ] = isnothing(W) ? push_env_right(GRs[ℓ+1], above.AR[ℓ+1], below.AR[ℓ+1]) :
                 push_env_right(GRs[ℓ+1], above.AR[ℓ+1], W, below.AR[ℓ+1])
    end

    # ---- normalization (mirroring MPSKit: GR Frobenius-normalized, GL scaled
    #      by the local overlap λ) ----
    for ℓ in 1:N
        GRs[ℓ] .= GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        GLn = GLs[inext]
        GR = GRs[ℓ]
        Cnew = _mapC(GLn, GR, above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("ternary environment: local overlap λ = 0 at site $ℓ")
        GLs[inext] .= GLn ./ λ
    end
    return GLs, GRs
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
        tgt = zeros(T, Ds[snext], Ds[snext])
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
        tgt = zeros(T, Ds[sprev], Ds[sprev])
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
