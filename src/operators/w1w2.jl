# W^I / W^II time-evolution MPOs — interface aligned with MPSKit (`WI`/`WII`/
# `make_time_mpo`, acting on SchurMPOTensor). Reference: arXiv:1407.1832
# "Time-evolving a matrix product state with long-ranged interactions". The
# block-operator matrix exponential of W^II is assembled into a dense
# (4d)×(4d) matrix and passed to LinearAlgebra.exp, equivalent to TEMPO's
# ExpExp block exponential.

"""
    WI(; tol, maxiter)

W^I first-order time-evolution stepper (first-order MPO approximation;
mirrors MPSKit's `WI`).
"""
@kwdef struct WI <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

"""
    WII(; tol, maxiter)

W^II second-order time-evolution stepper (block-exponential MPO approximation;
mirrors MPSKit's `WII`).
"""
@kwdef struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

# ---- Schur block extraction ----

get_A(W::SchurMPOTensor) = [W.A[i, :, j, :] for i in 1:size(W.A, 1), j in 1:size(W.A, 3)]
get_B(W::SchurMPOTensor) = [W.B[i, :, :] for i in 1:size(W.B, 1)]
get_C(W::SchurMPOTensor) = [W.C[:, j, :] for j in 1:size(W.C, 2)]
get_D(W::SchurMPOTensor) = W.D

function _sqrt2(dt::Complex)
    r = sqrt(dt)
    return r, r
end

function _sqrt2(dt::Real)
    if dt >= zero(dt)
        r = sqrt(dt)
        return r, r
    else
        r = sqrt(-dt)
        return r, -r
    end
end

# The evolved identity-channel payload `WD` (`Wd[1,:,1,:] = WD`); propagation
# blocks are placed at their original Schur block positions.
function _timempo_dense(WA, WB, WC, WD)
    a1, a2 = size(WA)
    T = promote_type(eltype(WD), eltype(eltype(WA)),
                     eltype(eltype(WB)), eltype(eltype(WC)))
    d = size(WD, 1)
    n = a1 + 1
    Wd = zeros(T, n, d, n, d)
    Wd[1, :, 1, :] = WD
    for l in 1:a2
        Wd[1, :, l+1, :] = WC[l]
    end
    for l in 1:a1
        Wd[l+1, :, 1, :] = WB[l]
        for m in 1:a2
            Wd[l+1, :, m+1, :] = WA[l, m]
        end
    end
    return Wd
end

function _timempo_dense(W::SchurMPOTensor, dt::Number, alg::WI)
    WA = get_A(W)
    δ₁, δ₂ = _sqrt2(dt)
    WB = get_B(W) .* δ₁
    WC = get_C(W) .* δ₂
    D = get_D(W)
    WD = Matrix{promote_type(scalartype(W), typeof(dt))}(I, size(D, 1), size(D, 1)) + dt .* D
    return _timempo_dense(WA, WB, WC, WD)
end

function _timempo_dense(W::SchurMPOTensor, dt::Number, alg::WII)
    A, B, C, D = get_A(W), get_B(W), get_C(W), get_D(W)
    d = size(W.A, 2)
    T = promote_type(scalartype(W), typeof(dt))
    Ddt = dt .* D
    WD = exp(Matrix(Ddt))
    s1, s2 = size(A)
    δ₁, δ₂ = _sqrt2(dt)

    WA = Array{Matrix{T},2}(undef, size(A))
    WB = Array{Matrix{T},1}(undef, length(B))
    WC = Array{Matrix{T},1}(undef, length(C))

    for a in 1:s1, b in 1:s2
        M = zeros(T, 4d, 4d)
        M[1:d, 1:d] .= Ddt
        M[d+1:2d, 1:d] .= δ₂ .* C[b]
        M[d+1:2d, d+1:2d] .= Ddt
        M[2d+1:3d, 1:d] .= δ₁ .* B[a]
        M[2d+1:3d, 2d+1:3d] .= Ddt
        M[3d+1:4d, 1:d] .= A[a, b]
        M[3d+1:4d, d+1:2d] .= δ₁ .* B[a]
        M[3d+1:4d, 2d+1:3d] .= δ₂ .* C[b]
        M[3d+1:4d, 3d+1:4d] .= Ddt
        mexp = exp(M)[:, 1:d]
        WC[b] = mexp[(d+1):2d, :]
        WB[a] = mexp[(2d+1):3d, :]
        WA[a, b] = mexp[(3d+1):4d, :]
    end
    return _timempo_dense(WA, WB, WC, WD)
end

"""
    make_time_mpo(bulk::SchurMPOTensor, dt, alg::Union{WI,WII};
                  imaginary_evolution = false) -> DenseIMPO
    make_time_mpo(H::SparseIMPO, dt, alg; kwargs...) -> DenseIMPO

Build the periodic time-evolution MPO approximating `exp(-i·H·dt)` (mirrors
MPSKit's `make_time_mpo`; with `imaginary_evolution = true` it is
`exp(-H·dt)`). Internally implements the W^I/W^II block-exponential schemes;
for an `SparseIMPO`, the Schur tensor of **every site** in the unit cell
is evolved separately (mirroring MPSKit's
`tmap(parent(H)) do W ... end` + `DenseIMPO(PeriodicArray(O))`), supporting
arbitrary unit-cell lengths.
"""
function make_time_mpo(bulk::SchurMPOTensor, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    return DenseIMPO([_timempo_dense(bulk, δ, alg)])
end

function make_time_mpo(H::SparseIMPO, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    # mirroring MPSKit: evolve the Schur tensor of every site in the unit cell
    # (supports arbitrary unit-cell lengths)
    O = [_timempo_dense(W, δ, alg) for W in parent(H)]
    return DenseIMPO(O)
end
