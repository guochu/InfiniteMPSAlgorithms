"""
    InfiniteCanonicalMPO{T}

与 `InfiniteCanonicalMPS` 同布局的 MPO 混合规范存储（把 MPO 当 MPS 用，
可用于密度矩阵等；储存与使用规则参考 `InfiniteCanonicalMPS`）：

- `AL[ℓ]::Array{T,4}`：MPO 张量 `(wl, u, wr, d)`，MPS 视图下左正交；
- `AR[ℓ]::Array{T,4}`：MPS 视图下右正交；
- `C[ℓ]::Array{T,2}`：bond ℓ 中心矩阵；
- `AC[ℓ]::Array{T,4}`：中心规范 MPO 张量，`AC[ℓ] = AL[ℓ]·C[ℓ] = C[ℓ-1]·AR[ℓ]`
  （在 MPS 视图 `(wl, u*d, wr)` 意义下，约定与 `InfiniteCanonicalMPS` 相同）。

用途：对 MPO 使用 MPS 算法（主导本征向量 = 最优低键维逼近 → MPO 压缩；
`hadamard(ψ, dag(ψ))` 型密度矩阵的规范存储）。
MPS 视图：`(wl, u, wr, d)` → permute `(1,2,4,3)` → reshape `(wl, u*d, wr)`。

注意：构造经 `InfiniteCanonicalMPS` 的 `gaugefix!` 混合规范化。规范变换在周期
trace 表示下望远相消（`tr(C⁻¹·X·C) = tr(X)`），故算符值（周期收缩）被精确保留。
"""
struct InfiniteCanonicalMPO{T}
    AL::PeriodicVector{Array{T,4}}
    AR::PeriodicVector{Array{T,4}}
    C::PeriodicVector{Array{T,2}}
    AC::PeriodicVector{Array{T,4}}

    function InfiniteCanonicalMPO{T}(AL::PeriodicVector{Array{T,4}},
                                     AR::PeriodicVector{Array{T,4}},
                                     C::PeriodicVector{Array{T,2}},
                                     AC::PeriodicVector{Array{T,4}}) where {T}
        L = length(AL)
        (L == length(AR) == length(C) == length(AC)) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        for ℓ in 1:L
            ℓ1 = _mod1(ℓ + 1, L)
            size(AC[ℓ], 1) == size(AL[ℓ], 1) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
            size(AL[ℓ], 3) == size(C[ℓ], 1) == size(AC[ℓ], 3) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
            size(C[ℓ - 1], 2) == size(AR[ℓ], 1) ||
                throw(DimensionMismatch("bond 不匹配 at site $ℓ"))
            size(AC[ℓ], 2) == size(AC[ℓ], 4) ||
                throw(DimensionMismatch("物理 (u,d) 不匹配 at site $ℓ"))
        end
        new{T}(AL, AR, C, AC)
    end
end

"""
    _mpo_from_mps(ψ::InfiniteCanonicalMPS, dus, dds) -> InfiniteCanonicalMPO

[`InfiniteCanonicalMPS`](@ref) → [`InfiniteCanonicalMPO`](@ref)：rank-3 场经
[`mps_view_to_mpo`](@ref) 逆视图变换转 rank-4（`dus`/`dds` 为每个 site 的 u/d 物理维）。
"""
function _mpo_from_mps(ψ::InfiniteCanonicalMPS{T}, dus::AbstractVector{Int},
                       dds::AbstractVector{Int}) where {T}
    to4 = As -> mps_view_to_mpo(collect(As); dus = dus, dds = dds)
    return InfiniteCanonicalMPO{T}(PeriodicVector(to4(ψ.AL)), PeriodicVector(to4(ψ.AR)),
                                   copy(ψ.C), PeriodicVector(to4(ψ.AC)))
end

"""
    InfiniteCanonicalMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...)

由普通 MPO 张量串构造（对标 `InfiniteCanonicalMPS(As)`）：经 `asmps_view` 转 MPS 后
`gaugefix!` 混合规范化（周期 trace 表示下算符值精确保留，见类型文档）。
"""
function InfiniteCanonicalMPO(Ws::AbstractVector{<:Array{T,4}}; kwargs...) where {T}
    N = length(Ws)
    ψ = InfiniteCanonicalMPS(asmps_view(Ws); kwargs...)
    return _mpo_from_mps(ψ, [size(Ws[ℓ], 2) for ℓ in 1:N], [size(Ws[ℓ], 4) for ℓ in 1:N])
end

InfiniteCanonicalMPO(W::InfiniteMPO; kwargs...) = InfiniteCanonicalMPO(W.Ws; kwargs...)

# ---------------- 接口（对标 InfiniteCanonicalMPS） ----------------

Base.length(W::InfiniteCanonicalMPO) = length(W.AL)
Base.size(W::InfiniteCanonicalMPO, args...) = size(W.AL, args...)
Base.getindex(W::InfiniteCanonicalMPO, ℓ::Integer) = W.AC[_mod1(ℓ, length(W))]
Base.setindex!(W::InfiniteCanonicalMPO, v::Array, ℓ::Integer) = (W.AC[ℓ] = v; W)
Base.firstindex(W::InfiniteCanonicalMPO) = 1
Base.lastindex(W::InfiniteCanonicalMPO) = length(W)
Base.iterate(W::InfiniteCanonicalMPO, args...) = iterate(W.AC, args...)
eachsite(W::InfiniteCanonicalMPO) = 1:length(W)

function Base.copy(W::InfiniteCanonicalMPO{T}) where {T}
    return InfiniteCanonicalMPO{T}(PeriodicVector([copy(a) for a in W.AL]),
                                   PeriodicVector([copy(a) for a in W.AR]),
                                   PeriodicVector([copy(c) for c in W.C]),
                                   PeriodicVector([copy(a) for a in W.AC]))
end
function Base.similar(W::InfiniteCanonicalMPO{T}) where {T}
    return InfiniteCanonicalMPO{T}(similar(W.AL), similar(W.AR), similar(W.C), similar(W.AC))
end
function Base.circshift(W::InfiniteCanonicalMPO, n)
    return InfiniteCanonicalMPO{T}(circshift(W.AL, n), circshift(W.AR, n),
                                   circshift(W.C, n), circshift(W.AC, n))
end

scalartype(::Type{InfiniteCanonicalMPO{T}}) where {T} = T
scalartype(W::InfiniteCanonicalMPO) = scalartype(typeof(W))

phydims(W::InfiniteCanonicalMPO) =
    [size(W.AL[ℓ], 2) * size(W.AL[ℓ], 4) for ℓ in 1:length(W)]
bonddim(W::InfiniteCanonicalMPO, ℓ::Integer) = size(W.C[_mod1(ℓ, length(W))], 1)
max_bonddim(W::InfiniteCanonicalMPO) = maximum(bonddim(W, ℓ) for ℓ in 1:length(W))

"`dag(W)`：逐张量共轭（用于重叠型收缩；非算符伴随网络）。"
dag(W::InfiniteCanonicalMPO{T}) where {T} =
    InfiniteCanonicalMPO{T}(PeriodicVector(conj.(parent(W.AL))), PeriodicVector(conj.(parent(W.AR))),
                            PeriodicVector(conj.(parent(W.C))), PeriodicVector(conj.(parent(W.AC))))

"`LinearAlgebra.norm(W) = norm(W.AC[1])`（与 InfiniteCanonicalMPS 一致）。"
LinearAlgebra.norm(W::InfiniteCanonicalMPO) = norm(W.AC[1])

"占位：MPO 整体 scale 有物理意义，由压缩/代数流程控制归一化。"
LinearAlgebra.normalize!(W::InfiniteCanonicalMPO) = W

"`InfiniteMPO(W)`：取左规范张量串 `W.AL` 转回普通 MPO。`tr(∏AL)` 是规范变换
（含相位）下不变的算符幅值（= 构造输入幅值 / 实正 λ），而 `tr(∏AC)` 被 C
矩阵插入加权、依赖规范，故这里用 `AL` 保证转换的唯一性。"
InfiniteMPO(W::InfiniteCanonicalMPO) = InfiniteMPO(collect(W.AL))

"""
    asmps_view(Ws::Vector{<:Array{T,4}}) -> Vector{Array{T,3}}

MPO 张量串的 MPS 视图：`(wl, u, wr, d)` → `(wl, u*d, wr)`。
"""
function asmps_view(Ws::Vector{<:Array{T,4}}) where {T}
    out = Vector{Array{T,3}}(undef, length(Ws))
    for (ℓ, W) in enumerate(Ws)
        wl, u, wr, d = size(W)
        out[ℓ] = reshape(permutedims(W, (1, 2, 4, 3)), wl, u * d, wr)
    end
    return out
end
asmps_view(W::InfiniteMPO) = asmps_view(W.Ws)
asmps_view(W::InfiniteCanonicalMPO) = asmps_view(collect(W.AC))

"""
    mps_view_to_mpo(As::Vector{<:Array{T,3}}; dus, dds) -> Vector{Array{T,4}}

[`asmps_view`](@ref) 的逆变换：`(wl, u*d, wr)` → `(wl, u, wr, d)`。
`dus`/`dds` 给出每个 site 的 u/d 物理维。
"""
function mps_view_to_mpo(As::Vector{<:Array{T,3}}; dus::AbstractVector{Int}, dds::AbstractVector{Int}) where {T}
    length(As) == length(dus) == length(dds) || throw(DimensionMismatch())
    out = Vector{Array{T,4}}(undef, length(As))
    for (ℓ, A) in enumerate(As)
        wl, p, wr = size(A)
        (p == dus[ℓ] * dds[ℓ]) || throw(DimensionMismatch("物理维不匹配"))
        out[ℓ] = permutedims(reshape(A, wl, dus[ℓ], dds[ℓ], wr), (1, 2, 4, 3))
    end
    return out
end

"""
    _align_scale!(x::InfiniteCanonicalMPS, K::Vector{<:Array{T,3}}) -> x

变分解 `x` 与目标张量串 `K` 的 scale/相位对齐（供代数运算的 MPS 结果使用）：

- 键维与 `K` 完全一致时：逐 site Frobenius 最优标量 `c_ℓ = ⟨x_AC|K_ℓ⟩/⟨x_AC|x_AC⟩`
  （键维相同时精确复原原始 scale 与相位）；
- 否则：环迹 `⟨x|K⟩` 是 `x` 的 N 次型，取主值 N 次根 `c = (⟨x|K⟩/⟨x|x⟩)^(1/N)`
  均匀分配到每个 site（使环重叠 ⟨x|K⟩ 实正且 ring⟨x|x⟩ 与之相等）；
  同时同步缩放 `C` 链，保持 `AC = AL·C = C·AR` 规范一致性。
"""
function _align_scale!(x::InfiniteCanonicalMPS, K::Vector{<:Array{T,3}}) where {T}
    N = length(x)
    if all(size(x.AC[ℓ]) == size(K[ℓ]) for ℓ in 1:N)
        for ℓ in 1:N
            c = dot(x.AC[ℓ], K[ℓ]) / dot(x.AC[ℓ], x.AC[ℓ])
            x.AC[ℓ] .= x.AC[ℓ] .* c
        end
    else
        xALs = [x.AL[ℓ] for ℓ in 1:N]
        c = (_ring_overlap(xALs, collect(K)) / _ring_overlap(xALs, xALs))^(1 / N)
        for ℓ in 1:N
            x.AC[ℓ] .= x.AC[ℓ] .* c
            x.C[ℓ] .= x.C[ℓ] .* c
        end
    end
    return x
end

"""
    mpo_compress(W::InfiniteMPO, D; tol=1e-10, maxiter=100, verbosity=0) -> (; W, overlap)

把 MPO 变分压缩到键维 `D`：把 MPO 视作 MPS（`asmps_view`），在恒等通道上做
VOMPS 重叠最大化扫描（等价于双层转移 `W⊗W̄` 主本征向量的键维 `D` 变分逼近）。
输出取环迹对齐标量 `c` 乘左规范张量 `AL`（[`_ring_scale`](@ref)）：输出幅值
精确恢复目标在压缩射线上的投影（含相位；收集 `AC` 会被 `C` 加权污染）。
压缩收敛态的逐 site 相位是本征解的规范自由度（MPSKit 同样不钉定本征解相位）。
返回压缩后的 `InfiniteMPO` 与最终重叠（归一化保真度 × N，见 `_overlap_sweeps`）。
"""
function mpo_compress(W::InfiniteMPO, D::Int;
                      tol::Real = 1.0e-10, maxiter::Int = 100, verbosity::Int = 0)
    N = length(W)
    dus = [size(W[ℓ], 2) for ℓ in 1:N]
    dds = [size(W[ℓ], 4) for ℓ in 1:N]
    K = asmps_view(W.Ws)
    ket = InfiniteCanonicalMPS(K)       # MPO 的 MPS 视图规范化为 ket
    x0 = randommps(scalartype(W), [dus[ℓ] * dds[ℓ] for ℓ in 1:N], D)
    x, overlap = _overlap_sweeps(nothing, ket, x0, K;
                                 tol = tol, maxiter = maxiter, verbosity = verbosity)
    c = _ring_scale(x, K)
    ALs4 = mps_view_to_mpo(collect(x.AL); dus = dus, dds = dds)
    return (; W = InfiniteMPO([c * A for A in ALs4]), overlap = overlap)
end

"""
    mixedcanonical_error(W) -> (ϵ_left, ϵ_right, ϵ_mixed)
    ismixedcanonical(W; tol = 1e-8, verbosity = 0) -> Bool

[`InfiniteCanonicalMPO`](@ref) 的混合规范诊断：在 MPS 视图 `(wl, u·d, wr)`
意义下检查（核与约定见 `InfiniteCanonicalMPS` 方法）。
"""
mixedcanonical_error(W::InfiniteCanonicalMPO) =
    _mixedcanonical_error(asmps_view(collect(W.AL)), asmps_view(collect(W.AR)), collect(W.C))

function ismixedcanonical(W::InfiniteCanonicalMPO; tol::Real = 1.0e-8, verbosity::Int = 0)
    ϵ_left, ϵ_right, ϵ_mixed = mixedcanonical_error(W)
    if verbosity > 0
        println("ismixedcanonical: ‖ΣAL†AL−I‖ = ", ϵ_left,
                ", ‖ΣAR·AR†−I‖ = ", ϵ_right,
                ", ‖AL·C−C·AR‖ = ", ϵ_mixed, " (tol = ", tol, ")")
    end
    return max(ϵ_left, ϵ_right, ϵ_mixed) ≤ tol
end
