# MPSKit（TensorKit 张量）↔ InfiniteMPSAlgorithms（普通 Array）转换适配层。
#
# 指标约定对照：
#   MPS 张量：两包均为 (Dl, phys, Dr)，无需置换；
#   MPO 张量：MPSKit `W[wl, bra, ket, wr]`（键在 1、4 位），
#             本包   `W[wl, u, wr, d]`（键在 1、3 位）⇒ 置换 (1, 2, 4, 3)；
#   环境：GL = (bra, level, ket)、GR = (ket, level, bra)，两包一致，无需置换。

using TensorKit: TensorMap, Tensor, space

"TensorMap / BraidingTensor 的原始数据数组。"
_tkdata(x) = x.data

# Pauli 矩阵（MPSKit 0.13 已移除 Models 子模块，这里本地定义）
σx(T::Type{<:Number} = ComplexF64) = TensorMap(T[0 1; 1 0], ℂ^2, ℂ^2)
σy(T::Type{<:Number} = ComplexF64) = TensorMap(T[0 -im; im 0], ℂ^2, ℂ^2)
σz(T::Type{<:Number} = ComplexF64) = TensorMap(T[1 0; 0 -1], ℂ^2, ℂ^2)

_dims(A::TensorMap) = (dim(space(A, 1)), dim(space(A, 2)), dim(space(A, 3)))

"TensorMap(Dl⊗d ← Dr) → Array (Dl, d, Dr)。"
function mpsarray(A::TensorMap)
    Dl, d, Dr = _dims(A)
    return reshape(_tkdata(A), Dl, d, Dr)
end

"TensorMap(Dl⊗d ← d⊗Dr)（MPSKit MPO）→ 本包 (wl, u, wr, d)。"
function mpoarray(W::TensorMap)
    Dl, dbra, dket, Dr = (dim(space(W, 1)), dim(space(W, 2)), dim(space(W, 3)), dim(space(W, 4)))
    return permutedims(reshape(_tkdata(W), Dl, dbra, dket, Dr), (1, 2, 4, 3))
end

"本包 (wl, u, wr, d) → MPSKit TensorMap(Dl⊗d ← d⊗Dr)。"
function mkmpotensor(W::AbstractArray{T,4}) where {T}
    n, du, nb, dd = size(W)
    (du == dd) || throw(DimensionMismatch("物理维度不匹配"))
    data = permutedims(W, (1, 2, 4, 3))
    return TensorMap(reshape(data, n * du, dd * nb), ℂ^n * ℂ^du, ℂ^dd * ℂ^nb)
end

"本包 (Dl, d, Dr) → MPSKit TensorMap(Dl⊗d ← Dr)。"
function mkmpstensor(A::AbstractArray{T,3}) where {T}
    Dl, d, Dr = size(A)
    return TensorMap(reshape(A, Dl * d, Dr), ℂ^Dl * ℂ^d, ℂ^Dr)
end

"环境 TensorMap(D⊗level ← D) → Array (D, level, D)。"
function envarray(GL::TensorMap)
    D1, nl, D2 = _dims(GL)
    return reshape(_tkdata(GL), D1, nl, D2)
end

"本包 MixedCanonicalMPS → MPSKit InfiniteMPS（同一规范：直接给出 AL 与 C₀）。"
function mkinfinitemps(ψ::InfiniteMPSAlgorithms.MixedCanonicalMPS)
    ALs = [mkmpstensor(ψ.AL[ℓ]) for ℓ in 1:length(ψ)]
    C₀ = TensorKit.TensorMap(copy(ψ.C[1]), ℂ^size(ψ.C[1], 1), ℂ^size(ψ.C[1], 2))
    return MPSKit.InfiniteMPS(ALs, C₀)
end

"本包 Jordan MPOHamiltonian → MPSKit InfiniteMPOHamiltonian（同一条目矩阵重建）。"
function mkhamiltonian(H::InfiniteMPSAlgorithms.MPOHamiltonian)
    N = length(H)
    n = mpobond(H)
    d = phydim(H)
    Ws = [Matrix{Union{Missing,ComplexF64,Matrix{ComplexF64}}}(missing, n, n) for _ in 1:N]
    for ℓ in 1:N
        for i in 1:n, j in 1:n
            v = H[ℓ][i, j]
            if i == 1 && j == 1 || i == n && j == n
                Ws[ℓ][i, j] = one(ComplexF64)
            elseif any(!iszero, v)
                Ws[ℓ][i, j] = Matrix{ComplexF64}(v)
            end
        end
    end
    lattice = fill(ℂ^d, N)
    return MPSKit.InfiniteMPOHamiltonian(lattice, Ws)
end

"随机本包 MPS（由给定张量直接构造）。"
function our_mps(As::Vector{<:Array{T,3}}) where {T}
    return InfiniteMPSAlgorithms.MixedCanonicalMPS(As)
end
