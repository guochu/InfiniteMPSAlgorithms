# ---------------- MPOHamiltonian（移植自 MPSKit src/operators/mpohamiltonian.jl） ----------------
#
# Jordan 上三角块矩阵形式的哈密顿量 MPO：
#
# ```math
# \begin{pmatrix}
# 1 & C & D \\
# 0 & A & B \\
# 0 & 0 & 1
# \end{pmatrix}
# ```
#
# 首末虚拟层为单位层（恒等通道），`isidentitylevel`/`isemptylevel` 支撑
# DMRGCache 的逐层线性求解（见 algorithms/idmrg.jl）。

"""
    MPOHamiltonian(W) -> MPOHamiltonian
    InfiniteMPOHamiltonian(Ws::Vector{<:Matrix}) -> InfiniteMPOHamiltonian

哈密顿量 MPO（`MPOHamiltonian` 的有限/无限别名）。`Ws[i][j, k]` 表示 site `i`
上左层 `j` 到右层 `k` 的局域算符，条目可为 `Missing`、`Number` 或 `(d, d)` 矩阵。
"""
struct MPOHamiltonian{TO<:JordanMPOTensor,V<:AbstractVector{TO}}
    W::V
end

const FiniteMPOHamiltonian{TO} = MPOHamiltonian{TO,Vector{TO}}
const InfiniteMPOHamiltonian{TO} = MPOHamiltonian{TO,PeriodicVector{TO}}

Base.length(H::MPOHamiltonian) = length(H.W)
Base.getindex(H::MPOHamiltonian, i::Int) = getindex(H.W, i)
Base.getindex(H::MPOHamiltonian, i::Int, j::Int, k::Int) = H[i][j, k]
Base.firstindex(H::MPOHamiltonian) = firstindex(H.W)
Base.lastindex(H::MPOHamiltonian) = lastindex(H.W)
Base.parent(H::MPOHamiltonian) = H.W
Base.copy(H::MPOHamiltonian) = MPOHamiltonian(map(copy, parent(H)))
Base.iterate(H::MPOHamiltonian, args...) = iterate(H.W, args...)
Base.eltype(::Type{MPOHamiltonian{TO,V}}) where {TO,V} = TO

function MPOHamiltonian(Ws::Vector{<:Matrix})
    return MPOHamiltonian([JordanMPOTensor(W) for W in Ws])
end

function FiniteMPOHamiltonian(Ws::Vector{TO}) where {TO<:JordanMPOTensor}
    return MPOHamiltonian{TO,Vector{TO}}(Ws)
end
function FiniteMPOHamiltonian(Ws::Vector{<:Matrix})
    return FiniteMPOHamiltonian([JordanMPOTensor(W) for W in Ws])
end

function InfiniteMPOHamiltonian(Ws::PeriodicVector{TO}) where {TO<:JordanMPOTensor}
    return MPOHamiltonian{TO,PeriodicVector{TO}}(Ws)
end
function InfiniteMPOHamiltonian(Ws::Vector{<:Matrix})
    N = length(Ws)
    for W in Ws
        (size(W, 1) == size(W, 2)) || throw(ArgumentError("无限哈密顿量的层矩阵应为方阵"))
        (size(W, 1) == size(Ws[1], 1)) || throw(ArgumentError("所有层矩阵尺寸应相同"))
    end
    return InfiniteMPOHamiltonian(PeriodicVector([JordanMPOTensor(W) for W in Ws]))
end

"bonddim(H, ℓ)：site ℓ 的 Jordan 虚拟层数（键维语义 per-bond，对标 MPSKit 的
`size(mpo[i], 1)`）。"
bonddim(H::MPOHamiltonian, ℓ::Integer) = nlvls(H[ℓ])
"bonddim(H)：单胞均匀层数（上三角方块结构 + 周期闭合要求各 site 层一致，由构造器保证）。"
bonddim(H::MPOHamiltonian) = nlvls(H[1])

scalartype(::Type{MPOHamiltonian{TO,V}}) where {TO,V} = scalartype(TO)
scalartype(H::MPOHamiltonian) = scalartype(typeof(H))

# MPSKit 风格的 A/B/C/D 块访问（对每个 site 返回对应块数组）
function Base.getproperty(H::MPOHamiltonian, sym::Symbol)
    if sym === :A
        return [getfield(W, :A) for W in parent(H)]
    elseif sym === :B
        return [getfield(W, :B) for W in parent(H)]
    elseif sym === :C
        return [getfield(W, :C) for W in parent(H)]
    elseif sym === :D
        return [getfield(W, :D) for W in parent(H)]
    end
    return getfield(H, sym)
end

"""
    isidentitylevel(H, i) -> Bool

层 `i` 是否为恒等层（转移仅含物理恒等算符）：首末层恒真；
中间层要求所有 site 的 `(i,i)` 对角块为恒等。
"""
function isidentitylevel(H::MPOHamiltonian, i::Int)
    n = bonddim(H)
    (i == 1 || i == n) && return true
    return all(parent(H)) do W
        block = W.A[i - 1, :, i - 1, :]
        return isapprox(block, Matrix{eltype(block)}(I, size(block)); atol = 1e-14)
    end
end

"""
    isemptylevel(H, i) -> Bool

层 `i` 是否为完全未使用的通道（对标 MPSKit：任意 site 上该层对角块结构性缺失即为空）。
本包稠密表示下等价于：所有 site 的对角块、首行 C 块、末列 B 块全为零。
注意显式存储的零对角块（如严格最近邻 MPO 的中间层）不算空层。
"""
function isemptylevel(H::MPOHamiltonian, i::Int)
    n = bonddim(H)
    (i == 1 || i == n) && return false
    return all(parent(H)) do W
        return iszero(W.A[i - 1, :, i - 1, :]) &&
               iszero(W.C[:, i - 1, :]) &&
               iszero(W.B[i - 1, :, :])
    end
end

# ---- 线性代数 ----

function Base.:+(H₁::MPOHamiltonian, H₂::MPOHamiltonian)
    (length(H₁) == length(H₂)) || throw(DimensionMismatch("单胞长度不匹配"))
    W = [H₁[i] + H₂[i] for i in 1:length(H₁)]
    return MPOHamiltonian(typeof(H₁.W)(W))
end

"""
    H + λs::AbstractVector（或 `λs + H`）

逐 site 加 `λᵢ·I`（对标 MPSKit 的 `H + λs`）。
"""
function Base.:+(H::InfiniteMPOHamiltonian, λs::AbstractVector{<:Number})
    (length(H) == length(λs)) || throw(DimensionMismatch("单胞长度不匹配"))
    Ws = Vector{Matrix{Any}}(undef, length(H))
    for i in 1:length(H)
        n = bonddim(H)
        W = Matrix{Any}(missing, n, n)
        W[1, 1] = one(scalartype(H))
        W[n, n] = one(scalartype(H))
        W[1, n] = λs[i] isa AbstractMatrix ? λs[i] : Matrix(λs[i] * I, phydim(H), phydim(H))
        Ws[i] = W
    end
    return H + InfiniteMPOHamiltonian(Ws)
end
Base.:+(λs::AbstractVector{<:Number}, H::InfiniteMPOHamiltonian) = H + λs

"phydim(H)：局域物理维度。"
phydim(H::MPOHamiltonian) = size(H[1].A, 2)

"""
    tompotensors(H::MPOHamiltonian) -> Vector{<:Array{T,4}}

稠密化为本包约定 `(wl, u, wr, d)` 的 MPO 张量串。
"""
tompotensors(H::MPOHamiltonian) = [tompotensor(H[i]) for i in 1:length(H)]

"""
    InfiniteMPO(H::MPOHamiltonian) -> InfiniteMPO

稠密周期 MPO 转换（对标 MPSKit 的 `DenseMPO(H)`）。
注意：恒等通道随之显式出现，环境将退化为转移矩阵主本征向量路径。
"""
InfiniteMPO(H::MPOHamiltonian) = InfiniteMPO(tompotensors(H))

"""
    infinite_mpo(bulk::JordanMPOTensor) -> InfiniteMPO

周期 bulk 的 Jordan 张量 → `InfiniteMPO`（键态 1 = identity 通道，on-site 项
`D` 并入 identity→identity 通道）：

- `W[1, ·, 1, ·] = I + D`（on-site 项并入 identity→identity 通道）；
- `W[1, ·, j+1, ·] = C[:, j, :]`，`W[j+1, ·, 1, ·] = B[j, :, :]`，
  `W[i+1, ·, j+1, ·] = A[i, :, j, :]`。
"""
function infinite_mpo(bulk::JordanMPOTensor)
    T = scalartype(bulk)
    d = size(bulk.A, 2)
    a = size(bulk.A, 1)
    nb = a + 1                       # 键态数：identity + 中间算符
    W = zeros(T, nb, d, nb, d)
    W[1, :, 1, :] = Matrix{T}(I, d, d) + bulk.D
    for j in 1:a
        W[1, :, j+1, :] = bulk.C[:, j, :]           # aⱼ（通道开启）
        W[j+1, :, 1, :] = bulk.B[j, :, :]           # bⱼ（通道闭合）
        for i in 1:a
            W[j+1, :, i+1, :] = bulk.A[j, :, i, :]  # 通道传播
        end
    end
    return InfiniteMPO([W])
end
