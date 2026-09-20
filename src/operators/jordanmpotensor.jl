# ---------------- JordanMPOTensor（移植自 MPSKit src/operators/jordanmpotensor.jl） ----------------
#
# MPO 的上三角块矩阵表示（无对称性、普通 Array 版本）。物理指标顺序与包内其余部分
# 一致（TEMPO 约定 `W[wl, u, wr, d]`，键指标在 1、3 位）：
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# 块存储（虚拟层数 nlvls = a + 2，首末层为单位层）：
# - `A::Array{T,4}`：`(a, d, a, d)`，中间块（第 2..end-1 行/列）；
# - `B::Array{T,3}`：`(a, d, d)`，中间 → 末列；
# - `C::Array{T,3}`：`(d, a, d)`，首行 → 中间；
# - `D::Array{T,2}`：`(d, d)`，首行末列（单 site 完整项）；
# - `[1,1]` 与 `[end,end]` 隐含为物理恒等算符（MPSKit 中的 `BraidingTensor`）。

"""
    JordanMPOTensor{T}

MPO 的 Jordan 上三角块矩阵张量（无对称性版本），见模块注释。支持
`W[i, j]` 的块矩阵式访问（返回 `(d, d)` 局域算符）。
"""
struct JordanMPOTensor{T}
    A::Array{T,4}
    B::Array{T,3}
    C::Array{T,3}
    D::Array{T,2}
end

Base.copy(W::JordanMPOTensor) = JordanMPOTensor(copy(W.A), copy(W.B), copy(W.C), copy(W.D))
scalartype(::Type{JordanMPOTensor{T}}) where {T} = T
scalartype(W::JordanMPOTensor) = scalartype(typeof(W))

"nlvls(W)：Jordan 张量的虚拟层数（= 键通道数 + 2 个单位层）。"
nlvls(W::JordanMPOTensor) = size(W.A, 1) + 2

# ---- 块矩阵式访问：W[i, j] → (d, d) 局域算符 ----

function Base.getindex(W::JordanMPOTensor{T}, i::Int, j::Int) where {T}
    n = nlvls(W)
    d = size(W.A, 2)
    if (i == 1 && j == 1) || (i == n && j == n)
        return Matrix{T}(I, d, d)
    elseif i == 1 && j == n
        return W.D
    elseif i == 1
        return W.C[:, j - 1, :]
    elseif j == n
        return W.B[i - 1, :, :]
    elseif 1 < i < n && 1 < j < n
        return W.A[i - 1, :, j - 1, :]
    end
    return zeros(T, d, d)
end

function Base.setindex!(W::JordanMPOTensor{T}, O::AbstractMatrix, i::Int, j::Int) where {T}
    (size(O, 1) == size(O, 2) == size(W.A, 2)) ||
        throw(DimensionMismatch("局域算符应为 $(size(W.A, 2))×$(size(W.A, 2))"))
    n = nlvls(W)
    if (i == 1 && j == 1) || (i == n && j == n)
        # 单位角块：不允许覆盖（保持恒等）
        return W
    elseif i == 1 && j == n
        W.D .= O
    elseif i == 1
        W.C[:, j - 1, :] .= O
    elseif j == n
        W.B[i - 1, :, :] .= O
    elseif 1 < i < n && 1 < j < n
        W.A[i - 1, :, j - 1, :] .= O
    else
        throw(ArgumentError("Jordan 张量下三角块 ($i, $j) 恒为零，不可赋值"))
    end
    return W
end

# ---- 构造器 ----

function _mpoham_scalar_type(W::AbstractMatrix)
    T = Union{}
    for v in W
        v isa Missing && continue
        T = promote_type(T, v isa Number ? typeof(v) : eltype(v))
    end
    return T === Union{} ? Float64 : T
end

"""
    JordanMPOTensor(W::AbstractMatrix) -> JordanMPOTensor

从 `n × n` 算符矩阵构造：`W[i, j]` 为第 `i` 行（左层）到第 `j` 列（右层）的
`(d, d)` 局域算符，条目可为 `Missing`、`Number` 或矩阵。`[1,1]` 与
`[end,end]` 隐含为恒等（条目应为 1 或 `Missing`）。
"""
function JordanMPOTensor(W::AbstractMatrix)
    (size(W, 1) == size(W, 2)) || throw(ArgumentError("W 应为方阵"))
    n = size(W, 1)
    (n >= 2) || throw(ArgumentError("W 至少要有 2 层（首末为单位层）"))
    T = _mpoham_scalar_type(W)
    # 物理维度：取第一个矩阵条目
    d = 0
    for v in W
        if v isa AbstractMatrix
            d = size(v, 1)
            break
        end
    end
    (d > 0) || throw(ArgumentError("W 中找不到 (d, d) 局域算符条目"))
    # 校验单位角
    for (i, j) in ((1, 1), (n, n))
        v = W[i, j]
        (v isa Missing || v isa Number && isone(v)) ||
            throw(ArgumentError("W[$i, $j] 应为 1 或 Missing（单位层隐含恒等）"))
    end
    J = JordanMPOTensor(zeros(T, n - 2, d, n - 2, d), zeros(T, n - 2, d, d),
                        zeros(T, d, n - 2, d), zeros(T, d, d))
    for i in 1:n, j in 1:n
        v = W[i, j]
        v isa Missing && continue
        iszero(v) && continue
        v isa Number ? J[i, j] = Matrix{T}(v * I, d, d) : (J[i, j] = v)
    end
    return J
end

"""
    tompotensor(W::JordanMPOTensor) -> Array{T,4}

稠密化为本包约定的 4 维 MPO 张量 `(wl, u, wr, d)`：恒等通道置于层 `1` 与层
`nlvls`，即 `Wd[1,:,1,:] = Wd[end,:,end,:] = I`。
"""
function tompotensor(W::JordanMPOTensor)
    T = scalartype(W)
    d = size(W.A, 2)
    n = nlvls(W)
    Wd = zeros(T, n, d, n, d)
    Wd[1, :, 1, :] .= Matrix{T}(I, d, d)
    Wd[n, :, n, :] .= Matrix{T}(I, d, d)
    Wd[1, :, n, :] .= W.D
    for j in 2:(n - 1)
        Wd[1, :, j, :] .= W.C[:, j - 1, :]
    end
    for i in 2:(n - 1)
        Wd[i, :, n, :] .= W.B[i - 1, :, :]
        for j in 2:(n - 1)
            Wd[i, :, j, :] .= W.A[i - 1, :, j - 1, :]
        end
    end
    return Wd
end

function Base.:+(W₁::JordanMPOTensor, W₂::JordanMPOTensor)
    n₁, n₂ = nlvls(W₁), nlvls(W₂)
    (n₁ == n₂ && size(W₁.A, 2) == size(W₂.A, 2)) ||
        throw(ArgumentError("Jordan 张量层数/物理维度不匹配"))
    return JordanMPOTensor(W₁.A + W₂.A, W₁.B + W₂.B, W₁.C + W₂.C, W₁.D + W₂.D)
end

function Base.:*(λ::Number, W::JordanMPOTensor)
    # 只缩放物理项（C、D 块与 B 的闭合端），恒等通道保持不变
    return JordanMPOTensor(copy(W.A), λ .* W.B, λ .* W.C, λ .* W.D)
end
