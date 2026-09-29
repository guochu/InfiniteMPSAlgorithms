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
push of a rank-2 environment. 显式两步收缩：先收缩键指标、物理指标留到最后
（三张量一体的收缩序不保证不先消物理指标——那会付出 `O(d·D₁·D₂·D̃₁·D̃₂)`
的中间张量代价；键优先的两步始终是 `O(d·D·D′)` 平方级）。"
function push_env_left(L::AbstractMatrix, above::AbstractArray{Ta,3},
                       below::AbstractArray{Tb,3}) where {Ta,Tb}
    @tensor Y[a, s, b′] := L[a, b] * above[b, s, b′]          # 键指标 b 先收缩
    @tensor L′[a′, b′] := conj(below[a, s, a′]) * Y[a, s, b′] # 物理指标最后
end

"Double-layer fused transfer rightward push of a rank-2 environment (mirrors
transfer_right). 显式两步收缩（键指标优先，同 [`push_env_left`](@ref)）。"
function push_env_right(R::AbstractMatrix, above::AbstractArray{Ta,3},
                        below::AbstractArray{Tb,3}) where {Ta,Tb}
    @tensor Y[a, s, b′] := R[a, b] * conj(below[b′, s, b])    # 键指标 b 先收缩
    @tensor R′[a′, b′] := above[a′, s, a] * Y[a, s, b′]       # 物理指标最后
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

"Ternary identity-channel left push (w dimension = 1). 三张量一体的收缩由
TensorOperations 按 flops 最优排序（先键后物理）；不可拆步——L 的字面量
`1` 腿无法在中间步悬空。"
function push_env_left(L::AbstractArray{TL,3}, below::AbstractArray{Tb,3},
                       above::AbstractArray{Ta,3}) where {TL,Tb,Ta}
    @tensor L′[bl′, 1, al′] := conj(below[bl, s, bl′]) * L[bl, 1, al] * above[al, s, al′]
end

"Ternary MPO-channel right push: `R` is `(above bond, w, below bond)`."
function push_env_right(R::AbstractArray{TR,3}, above::AbstractArray{Ta,3},
                        W::AbstractArray{Tw,4}, below::AbstractArray{Tb,3}) where {TR,Ta,Tw,Tb}
    @tensor R′[al′, w′, bl′] := above[al′, d, al] * W[w′, ū, w, d] * conj(below[bl′, ū, bl]) * R[al, w, bl]
end

"Ternary identity-channel right push (w dimension = 1). 三张量一体的收缩由
TensorOperations 按 flops 最优排序（先键后物理）；不可拆步——L 的字面量
`1` 腿无法在中间步悬空。"
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

# linsolve / regularize! / dominant_env（哈密顿量通道环境求解 kernel）见
# algorithms/groundstates/envs.jl
