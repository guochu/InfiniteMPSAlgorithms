# ---------------- 环境求解 kernel（transfermatrix 的环境侧延伸） ----------------
#
# 哈密顿量通道环境构造（DMRGCache，见 groundstates/idmrg.jl）所需的求解原语：
# 非齐次线性解 `linsolve`（Schur 逐 level 通道）、恒等层固定点投影
# `regularize!`、以及 DenseMPO 通道的转移矩阵主本征向量 `dominant_env`。

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
