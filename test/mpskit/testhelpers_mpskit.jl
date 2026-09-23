# =====================================================================
# InfiniteMPSAlgorithms（plain Array）↔ MPSKit（TensorKit）转换适配层
# （test/mpskit/ 对齐测试共用；移植自 debug/mpsconvert.jl 与
#   debug/testhelpers_concordance.jl，并合并去重）
#
# 指标约定对照（已与 MPSKit 0.13 对齐）：
# - MPS 张量：两包均为 (Dl, phys, Dr)，无需置换；
# - MPO 张量：MPSKit `W[wl, bra, ket, wr]`（键在 1、4 位），本包
#   `W[wl, u, wr, d]`（键在 1、3 位）⇒ 置换 (1, 2, 4, 3)；
#   （参照 MPSKit test/setup/testsetup.jl 的
#   `permute(TensorMap(d, ℂ²⊗ℂ², ℂ²⊗ℂ²), ((1,2),(4,3)))`）
# - 环境：GL = (bra, level, ket)、GR = (ket, level, bra)，两包一致。
# 故本包 Array ↔ MPSKit TensorMap 的 MPS 部分无需任何指标置换。
#
# 命名空间消歧：MPSKit 与本包存在导出名冲突（VUMPS / IDMRG / VOMPS / TDVP /
# find_groundstate / leftenv / rightenv / AC_hamiltonian / C_hamiltonian 等），
# 这里显式导入本包版本；MPSKit 一律加 `MPSKit.` 限定名使用。
# =====================================================================

using MPSKit
import TensorKit
using TensorKit: TensorMap, ComplexSpace, ℂ, dim, space, ⊗
using InfiniteMPSAlgorithms:  # 与 MPSKit 导出名冲突的本包接口
    VUMPS, IDMRG, VOMPS, TDVP, WI, WII,
    find_groundstate, timestep, leftenv, rightenv, AC_hamiltonian, C_hamiltonian

# ---- 预热 TensorKit 的 braid/repartition 生成内核 ----
# 在 --compiled-modules=no 的运行环境（如受限沙箱）下，该生成分派会在 MPSKit
# 构建 @async 任务的内部首次触发，兄弟任务随即因 world age 冲突无法调用
# （method too new）。在主任务先用同形张量触发一次 braid/repartition 内核
# 生成即可避免；正常预编译环境（内核随包预编译）不受影响。
let A = TensorMap(randn(ComplexF64, 2, 2, 2, 2), ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
    TensorKit.permute(A, ((2, 1), (4, 3)))
end

_tkdata(x) = x.data
_dims(A::TensorMap) = (dim(space(A, 1)), dim(space(A, 2)), dim(space(A, 3)))

# ---- Pauli（TensorMap 版；带 _tk 后缀避免与本包导出的 Matrix 版冲突）----
σx_tk(T::Type{<:Number} = ComplexF64) = TensorMap(T[0 1; 1 0], ℂ^2, ℂ^2)
σy_tk(T::Type{<:Number} = ComplexF64) = TensorMap(T[0 -im; im 0], ℂ^2, ℂ^2)
σz_tk(T::Type{<:Number} = ComplexF64) = TensorMap(T[1 0; 0 -1], ℂ^2, ℂ^2)

"TensorMap(Dl⊗d ← Dr) → Array (Dl, d, Dr)（复制数据，与 MPSKit 侧隔离）。"
mpsarray(A::TensorMap) = begin
    Dl, d, Dr = _dims(A)
    copy(reshape(_tkdata(A), Dl, d, Dr))
end

"TensorMap(Dl⊗d ← d⊗Dr)（MPSKit MPO）→ 本包 (wl, u, wr, d)（permutedims 已复制）。"
function mpoarray(W::TensorMap)
    Dl, dbra, dket, Dr = (dim(space(W, 1)), dim(space(W, 2)), dim(space(W, 3)),
                          dim(space(W, 4)))
    return permutedims(reshape(_tkdata(W), Dl, dbra, dket, Dr), (1, 2, 4, 3))
end

"TensorMap → 原始 Array（codomain 维在前、domain 维在后；复制数据）。"
tensor_to_array(t::TensorMap) = copy(reshape(t.data, dim.(space(t))...))

"MPOTensor → 本包 (wl, u, wr, d) 张量。"
mpo_from_mpskit(Wk) = permutedims(tensor_to_array(Wk), (1, 2, 4, 3))

"本包 (Dl, d, Dr) → MPSKit TensorMap(Dl⊗d ← Dr)（复制数据；MPSKit 的算法多为
in-place，绝不能与本包测试装置共享内存）。"
function mkmpstensor(A::AbstractArray{T,3}) where {T}
    Dl, d, Dr = size(A)
    return TensorMap(reshape(copy(A), Dl * d, Dr), ℂ^Dl * ℂ^d, ℂ^Dr)
end

"本包 (wl, u, wr, d) → MPSKit TensorMap(Dl⊗d ← d⊗Dr)（复制数据）。"
function mkmpotensor(W::AbstractArray{T,4}) where {T}
    n, du, nb, dd = size(W)
    (du == dd) || throw(DimensionMismatch("物理维度不匹配"))
    data = permutedims(W, (1, 2, 4, 3))
    return TensorMap(reshape(data, n * du, dd * nb), ℂ^n * ℂ^du, ℂ^dd * ℂ^nb)
end

"环境 TensorMap(D⊗level ← D) → Array (D, level, D)（复制数据）。"
function envarray(GL::TensorMap)
    D1, nl, D2 = _dims(GL)
    return copy(reshape(_tkdata(GL), D1, nl, D2))
end

"本包 CanonicalIMPS（AL 串）→ MPSKit InfiniteMPS。"
to_mpskit(ψ::CanonicalIMPS) =
    MPSKit.InfiniteMPS([mkmpstensor(a) for a in ψ.AL])

"本包 DenseIMPO → MPSKit DenseIMPO。"
to_mpskit(W::DenseIMPO) = MPSKit.InfiniteMPO([mkmpotensor(w) for w in W.Ws])

"本包 CanonicalIMPS → MPSKit InfiniteMPS（同一规范：直接给出 AL 与 C₀）。"
function mkinfinitemps(ψ::CanonicalIMPS)
    ALs = [mkmpstensor(ψ.AL[ℓ]) for ℓ in 1:length(ψ)]
    C₀ = TensorMap(copy(ψ.C[1]), ℂ^size(ψ.C[1], 1), ℂ^size(ψ.C[1], 2))
    return MPSKit.InfiniteMPS(ALs, C₀)
end

"MPSKit InfiniteMPS → 本包 CanonicalIMPS（保留 AL/AR/C/AC 全部四个场；无限 MPS
的态由 (AL, C) 共同决定，不能只用 AL 重建）。"
function from_mpskit(ϕ::MPSKit.InfiniteMPS)
    P = InfiniteMPSAlgorithms.PeriodicVector
    return CanonicalIMPS(P([tensor_to_array(a) for a in ϕ.AL]),
                         P([tensor_to_array(a) for a in ϕ.AR]),
                         P([tensor_to_array(c) for c in ϕ.C]),
                         P([tensor_to_array(ac) for ac in ϕ.AC]))
end

"MPSKit DenseIMPO → 本包 DenseIMPO。"
from_mpskit(O::MPSKit.InfiniteMPO) = DenseIMPO([mpo_from_mpskit(w) for w in parent(O)])

"两 MPO 的稠密周期 trace 表示的复比例残差（0 = 平行，规范/尺度不变）"
mpo_ray_residual(O1, O2) = begin
    d1 = vec(_dense_mpo_repr(O1))
    d2 = vec(_dense_mpo_repr(O2))
    ls = dot(d2, d1) / dot(d2, d2)
    norm(d1 .- ls .* d2) / norm(d1)
end
