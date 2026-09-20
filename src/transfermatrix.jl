# ---------------- 转移矩阵与环境 kernel ----------------

"`push_env_left(L, A)`：恒等通道左环境推进，`L` 为 `(bra键, ket键)` 矩阵。"
function push_env_left(L::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, s, a′]) * L[a, b] * A[b, s, b′]
end

"`push_env_left(L, W, A)`：MPO 通道左环境推进，`L` 为 `(bra键, w, ket键)`。"
function push_env_left(L::AbstractArray{T,3}, W::AbstractArray{T,4}, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, w′, b′] := conj(A[a, ū, a′]) * L[a, w, b] * W[w, ū, w′, d] * A[b, d, b′]
end

"恒等通道的 rank-3 形式（w 维 = 1）左环境推进。"
function push_env_left(L::AbstractArray{T,3}, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, 1, b′] := conj(A[a, s, a′]) * L[a, 1, b] * A[b, s, b′]
end

"恒等通道右环境推进（对标 MPSKit transfer_right）：R′ = Σ_s A_s · R · A_s′，输出 (bra′, ket′)。"
function push_env_right(R::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, s, a] * conj(A[b′, s, b]) * R[a, b]
end

"MPO 通道右环境推进（对标 MPSKit transfer_right）：R 为 `(bra, w, ket)`，O 的 u 与 below 收缩、d 与 above 收缩。"
function push_env_right(R::AbstractArray{T,3}, W::AbstractArray{T,4}, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, w′, b′] := A[a′, d, a] * W[w′, ū, w, d] * conj(A[b′, ū, b]) * R[a, w, b]
end

"恒等通道的 rank-3 形式（w 维 = 1）右环境推进。"
function push_env_right(R::AbstractArray{T,3}, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, 1, b′] := A[a′, s, a] * conj(A[b′, s, b]) * R[a, 1, b]
end

"带局域算符插入的右环境推进：R′ = Σ A_s · O · R · A_s′。"
function push_env_right(R::AbstractMatrix, O::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, u, a] * O[u, s] * conj(A[b′, s, b]) * R[a, b]
end

"`push_env_left(L, above, below)`：双层（above/below）融合转移对 rank-2 环境的推进。"
function push_env_left(L::AbstractMatrix, above::AbstractArray{T,3}, below::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(below[a, s, a′]) * L[a, b] * above[b, s, b′]
end

"双层融合转移对 rank-2 环境的右向推进（对标 transfer_right）。"
function push_env_right(R::AbstractMatrix, above::AbstractArray{T,3}, below::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := above[a′, s, a] * conj(below[b′, s, b]) * R[a, b]
end

# ---- 三元通道（below = bra 共轭、above = ket 不共轭；对标 MPSKit 的
#      TransferMatrix(above.AL, operator, below.AL) 收缩） ----

"三元 MPO 通道左推进：L 为 `(below键, w, above键)`。"
function push_env_left(L::AbstractArray{T,3}, below::AbstractArray{T,3}, W::AbstractArray{T,4},
                       above::AbstractArray{T,3}) where {T}
    @tensor L′[bl′, w′, al′] := conj(below[bl, ū, bl′]) * L[bl, w, al] * W[w, ū, w′, d] * above[al, d, al′]
end

"三元恒等通道左推进（w 维 = 1）。"
function push_env_left(L::AbstractArray{T,3}, below::AbstractArray{T,3},
                       above::AbstractArray{T,3}) where {T}
    @tensor L′[bl′, 1, al′] := conj(below[bl, s, bl′]) * L[bl, 1, al] * above[al, s, al′]
end

"三元 MPO 通道右推进：R 为 `(above键, w, below键)`。"
function push_env_right(R::AbstractArray{T,3}, above::AbstractArray{T,3}, W::AbstractArray{T,4},
                        below::AbstractArray{T,3}) where {T}
    @tensor R′[al′, w′, bl′] := above[al′, d, al] * W[w′, ū, w, d] * conj(below[bl′, ū, bl]) * R[al, w, bl]
end

"三元恒等通道右推进（w 维 = 1）。"
function push_env_right(R::AbstractArray{T,3}, above::AbstractArray{T,3},
                        below::AbstractArray{T,3}) where {T}
    @tensor R′[al′, 1, bl′] := above[al′, s, al] * conj(below[bl′, s, bl]) * R[al, 1, bl]
end

# ---------------- TransferMatrix ----------------

"""
    TransferMatrix(above::AbstractVector, below::AbstractVector)
    TransferMatrix(a::AbstractArray{T,3}, b::AbstractArray{T,3})
    TransferMatrix(a::AbstractArray{T,3}, w::AbstractArray{T,4}, b::AbstractArray{T,3})
    TransferMatrix(ψ::InfiniteCanonicalMPS)

周期铺满一个单胞的融合转移映射（对标 MPSKit 的 `TransferMatrix`），
实现 `size`、`getindex`（逐列）与 `*`；`side = :left/:right` 选择左/右作用方向。
恒等通道作用在 `(bra, ket)` 矩阵向量化上；MPO 通道作用在 `(bra, w, ket)` 上。
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
    (length(below) == N) || throw(DimensionMismatch("above 与 below 长度必须相等"))
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

TransferMatrix(ψ::InfiniteCanonicalMPS) = TransferMatrix(ψ.AL, ψ.AL)

# ---------------- 本征固定点 ----------------

"""
    fixedpoint(operator, x₀, which, alg) -> (λ, v)

转移映射的主导本征对（对标 MPSKit 的 `fixedpoint`，内部为 KrylovKit.eigsolve）。
`alg` 为 KrylovKit 的 `Lanczos`/`Arnoldi` 算法对象。
"""
function fixedpoint(operator, x₀, which::Symbol, alg::KrylovKit.KrylovAlgorithm)
    isherm = alg isa KrylovKit.Lanczos
    vals, vecs, _ = eigsolve(operator, x₀, 1, which;
                             ishermitian = isherm, tol = alg.tol,
                             krylovdim = alg.krylovdim, maxiter = alg.maxiter,
                             eager = true)
    return vals[1], vecs[1]
end

"DynamicTol 包装器：使用内部 Krylov 算法的初始容差。"
fixedpoint(operator, x₀, which::Symbol, alg::DynamicTol) =
    fixedpoint(operator, x₀, which, alg.alg)

"""
    linsolve(operator, b, x₀, [alg]; a₀ = 1, a₁ = 1) -> (x, info)

解线性方程 `a₀·x + a₁·A·x = b`（对标 MPSKit 的 `linsolve`，内部为
KrylovKit.linsolve；`alg` 为 `GMRES`/`BiCGStab`/`CG`）。
"""
function linsolve(operator, b::AbstractVector, x₀::AbstractVector,
                  alg::KrylovKit.KrylovAlgorithm = KrylovKit.GMRES();
                  a₀ = 1, a₁ = 1)
    x, info = KrylovKit.linsolve(v -> operator(v), b, x₀, alg, a₀, a₁)
    return x, info
end

"""
    regularize!(v, lvec, rvec) -> v

投影掉恒等通道固定点分量（对标 MPSKit 的 `regularize!`）：
`v ← v − rvec·⟨lvec, v⟩`。本包规范下 `lvec = rvec = I`，即 `v ← v − tr(v)·I`。
"""
function regularize!(v::AbstractMatrix, lvec::AbstractMatrix, rvec::AbstractMatrix)
    c = sum(lvec .* transpose(v))   # MPSKit 语义：Σ lvec[a,b]·v[b,a]（不取共轭）
    v .-= c .* rvec
    return v
end

function _dominant_env_matvec(op::Union{Nothing,InfiniteMPO}, ψ::InfiniteCanonicalMPS, side::Symbol)
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

恒等/MPO 通道转移矩阵的主导本征向量（周期铺满一个单胞）。
恒等通道使用 AL/AR（严格规范），返回 `λ ≈ 1`。
"""
function dominant_env(ψ::InfiniteCanonicalMPS; side::Symbol = :left, which::Symbol = :LM, kwargs...)
    return dominant_env(nothing, ψ; side = side, which = which, kwargs...)
end

function dominant_env(op::Union{Nothing,InfiniteMPO}, ψ::InfiniteCanonicalMPS;
                      side::Symbol = :left, which::Symbol = :LM,
                      tol::Real = 1.0e-13, krylovdim::Int = 12, maxiter::Int = 200)
    identity = isnothing(op)
    D = size(ψ.AL[1], 1)
    dim = identity ? D * D : D * size(op[1], 1) * D
    T = scalartype(ψ)
    matvec = _dominant_env_matvec(op, ψ, side)
    v0 = ones(T, dim)
    λs, vs, _ = eigsolve(matvec, v0, 1, which; tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    λ = λs[1]
    L = identity ? reshape(vs[1], D, D) : reshape(vs[1], D, size(op[1], 1), D)
    L ./= norm(L)
    return λ, L
end
