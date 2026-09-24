# ---------------- transfer matrices and environment kernels ----------------

"`push_env_left(L, A)`: identity-channel left-environment push; `L` is a
`(bra bond, ket bond)` matrix."
function push_env_left(L::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, s, a′]) * L[a, b] * A[b, s, b′]
end

"`push_env_left(L, W, A)`: MPO-channel left-environment push; `L` is
`(bra bond, w, ket bond)` (参量允许不同标量类型，复环境 × 实链自动提升)."
function push_env_left(L::AbstractArray{TL,3}, W::AbstractArray{Tw,4}, A::AbstractArray{Ta,3}) where {TL,Tw,Ta}
    @tensor L′[a′, w′, b′] := conj(A[a, ū, a′]) * L[a, w, b] * W[w, ū, w′, d] * A[b, d, b′]
end

"Rank-3 form of the identity-channel (w dimension = 1) left-environment push."
function push_env_left(L::AbstractArray{T,3}, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, 1, b′] := conj(A[a, s, a′]) * L[a, 1, b] * A[b, s, b′]
end

"Identity-channel right-environment push (mirrors MPSKit transfer_right):
R′ = Σ_s A_s · R · A_s′, output (bra′, ket′)."
function push_env_right(R::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, s, a] * conj(A[b′, s, b]) * R[a, b]
end

"MPO-channel right-environment push (mirrors MPSKit transfer_right): `R` is
`(bra, w, ket)`; O's u contracts with below, d with above."
function push_env_right(R::AbstractArray{TR,3}, W::AbstractArray{Tw,4}, A::AbstractArray{Ta,3}) where {TR,Tw,Ta}
    @tensor R′[a′, w′, b′] := A[a′, d, a] * W[w′, ū, w, d] * conj(A[b′, ū, b]) * R[a, w, b]
end

"Rank-3 form of the identity-channel (w dimension = 1) right-environment push."
function push_env_right(R::AbstractArray{T,3}, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, 1, b′] := A[a′, s, a] * conj(A[b′, s, b]) * R[a, 1, b]
end

"Right-environment push with a local operator insertion: R′ = Σ A_s · O · R · A_s′."
function push_env_right(R::AbstractMatrix, O::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, u, a] * O[u, s] * conj(A[b′, s, b]) * R[a, b]
end

"`push_env_left(L, above, below)`: double-layer (above/below) fused transfer
push of a rank-2 environment."
function push_env_left(L::AbstractMatrix, above::AbstractArray{T,3}, below::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(below[a, s, a′]) * L[a, b] * above[b, s, b′]
end

"Double-layer fused transfer rightward push of a rank-2 environment (mirrors
transfer_right)."
function push_env_right(R::AbstractMatrix, above::AbstractArray{T,3}, below::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := above[a′, s, a] * conj(below[b′, s, b]) * R[a, b]
end

# ---- ternary channels (below = bra conjugated, above = ket unconjugated;
#      mirroring MPSKit's TransferMatrix(above.AL, operator, below.AL) contraction) ----

"Ternary MPO-channel left push: `L` is `(below bond, w, above bond)`.
各参量允许不同标量类型（复环境 × 实链，@tensor 自动提升——MPSKit 对齐：
环境取 eigsolve 返回的实际 eltype，实输入下 leading vector 可为复）。"
function push_env_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,3},
                       W::AbstractArray{Tw,4}, above::AbstractArray{Ta,3}) where {TL,Tb,Tw,Ta}
    @tensor L′[bl′, w′, al′] := conj(below[bl, ū, bl′]) * L[bl, w, al] * W[w, ū, w′, d] * above[al, d, al′]
end

"Ternary identity-channel left push (w dimension = 1)."
function push_env_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,3},
                       above::AbstractArray{Ta,3}) where {TL,Tb,Ta}
    @tensor L′[bl′, 1, al′] := conj(below[bl, s, bl′]) * L[bl, 1, al] * above[al, s, al′]
end

"Ternary MPO-channel right push: `R` is `(above bond, w, below bond)`."
function push_env_right(R::AbstractArray{TR,3}, above::AbstractArray{Ta,3},
                        W::AbstractArray{Tw,4}, below::AbstractArray{Tb,3}) where {TR,Ta,Tw,Tb}
    @tensor R′[al′, w′, bl′] := above[al′, d, al] * W[w′, ū, w, d] * conj(below[bl′, ū, bl]) * R[al, w, bl]
end

"Ternary identity-channel right push (w dimension = 1)."
function push_env_right(R::AbstractArray{TR,3}, above::AbstractArray{Ta,3},
                        below::AbstractArray{Tb,3}) where {TR,Ta,Tb}
    @tensor R′[al′, 1, bl′] := above[al′, s, al] * conj(below[bl′, s, bl]) * R[al, 1, bl]
end

# ---------------- TransferMatrix ----------------

"""
    TransferMatrix(above::AbstractVector, below::AbstractVector)
    TransferMatrix(a::AbstractArray{T,3}, b::AbstractArray{T,3})
    TransferMatrix(a::AbstractArray{T,3}, w::AbstractArray{T,4}, b::AbstractArray{T,3})
    TransferMatrix(ψ::CanonicalIMPS)

Fused transfer map tiled over one unit cell (mirrors MPSKit's `TransferMatrix`);
implements `size`, `getindex` (column-wise), and `*`; `side = :left/:right`
selects the acting direction. The identity channel acts on the vectorized
`(bra, ket)` matrices; the MPO channel on `(bra, w, ket)`.
"""
struct TransferMatrix{T,F<:Function}
    f::F
    sz::NTuple{2,Int}
    side::Symbol
end

Base.size(tm::TransferMatrix) = tm.sz
Base.size(tm::TransferMatrix, i::Int) = tm.sz[i]
Base.:*(tm::TransferMatrix, v::AbstractVector) = tm.f(v)
(tm::TransferMatrix)(v::AbstractVector) = tm.f(v)

function Base.getindex(tm::TransferMatrix{T}, i::Integer, j::Integer) where {T}
    e = zeros(T, tm.sz[2])
    e[j] = one(T)
    return tm.f(e)[i]
end

function TransferMatrix(above::AbstractVector{<:AbstractArray{T,3}},
                        below::AbstractVector{<:AbstractArray{T,3}};
                        side::Symbol = :left) where {T}
    N = length(above)
    (length(below) == N) || throw(DimensionMismatch("above and below must have equal lengths"))
    if side === :left
        f = function (v::AbstractVector)
            Dl, Dket = size(above[1], 1), size(above[1], 1)
            L = reshape(v, size(below[1], 1), Dket)
            for ℓ in 1:N
                L = push_env_left(L, above[ℓ], below[ℓ])
            end
            return vec(L)
        end
    else
        f = function (v::AbstractVector)
            Dl = size(above[1], 1)
            R = reshape(v, Dl, size(below[1], 1))
            for ℓ in N:-1:1
                R = push_env_right(R, above[ℓ], below[ℓ])
            end
            return vec(R)
        end
    end
    d = size(above[1], 1) * size(below[1], 1)
    return TransferMatrix{T,typeof(f)}(f, (d, d), side)
end

function TransferMatrix(a::AbstractArray{T,3}, b::AbstractArray{T,3}; side::Symbol = :left) where {T}
    return TransferMatrix([a], [b]; side = side)
end

function TransferMatrix(a::AbstractArray{T,3}, w::AbstractArray{T,4},
                        b::AbstractArray{T,3}; side::Symbol = :left) where {T}
    N = 1
    f = function (v::AbstractVector)
        if side === :left
            L = reshape(v, size(b, 1), size(w, 1), size(a, 1))
            L = push_env_left(L, w, a)
            return vec(L)
        else
            R = reshape(v, size(a, 1), size(w, 1), size(b, 1))
            R = push_env_right(R, w, a)
            return vec(R)
        end
    end
    d = size(a, 1) * size(w, 1) * size(b, 1)
    return TransferMatrix{T,typeof(f)}(f, (d, d), side)
end

TransferMatrix(ψ::CanonicalIMPS) = TransferMatrix(ψ.AL, ψ.AL)

# ---------------- eigen fixed points ----------------

"""
    fixedpoint(operator, x₀, which, alg) -> (λ, v)

Dominant eigenpair of the transfer map (mirrors MPSKit's `fixedpoint`;
internally KrylovKit.eigsolve). `alg` is a KrylovKit `Lanczos`/`Arnoldi`
algorithm object.
"""
function fixedpoint(operator, x₀, which::Symbol, alg::KrylovKit.KrylovAlgorithm)
    isherm = alg isa KrylovKit.Lanczos
    vals, vecs, _ = eigsolve(operator, x₀, 1, which;
                             ishermitian = isherm, tol = alg.tol,
                             krylovdim = alg.krylovdim, maxiter = alg.maxiter,
                             eager = true)
    return vals[1], vecs[1]
end

"DynamicTol wrapper: uses the inner Krylov algorithm's initial tolerance."
fixedpoint(operator, x₀, which::Symbol, alg::DynamicTol) =
    fixedpoint(operator, x₀, which, alg.alg)

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
                      tol::Real = 1.0e-13, krylovdim::Int = 12, maxiter::Int = 200)
    identity = isnothing(op)
    D = size(ψ.AL[1], 1)
    dim = identity ? D * D : D * size(op[1], 1) * D
    T = scalartype(ψ)
    matvec = _dominant_env_matvec(op, ψ, side)
    v0 = ones(T, dim)
    λs, vs, _ = eigsolve(matvec, v0, 1, which; ishermitian = false, tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    λ = λs[1]
    L = identity ? reshape(vs[1], D, D) : reshape(vs[1], D, size(op[1], 1), D)
    L ./= norm(L)
    return λ, L
end
