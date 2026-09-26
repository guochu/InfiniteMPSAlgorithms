# ---- 周期 trace 表示的稠密辅助（algebra 与 api 共用）----
# MPS：c[(s1…sN)] = tr(∏ AL[ℓ][:, s_ℓ, :])（左规范张量的周期 trace）。
# 注意必须用 AL 而非 AC：混合规范下 AC = AL·C 在张量间插入 C 矩阵，
# tr(∏AC) 是规范依赖的 C 加权量；而 tr(∏AL) = tr(∏A_raw)/λ（λ 实正），
# 是构造器确定性产出、规范变换（含相位）下不变的射线幅值。
function _dense_trace(ALs)
    N = length(ALs)
    dims = Tuple(size(ALs[ℓ], 2) for ℓ in 1:N)
    T = eltype(ALs[1])
    c = zeros(T, dims)
    for idx in CartesianIndices(dims)
        s = Tuple(idx)
        M = Matrix{T}(I, size(ALs[1], 1), size(ALs[1], 1))
        for ℓ in 1:N
            M = M * ALs[ℓ][:, s[ℓ], :]
        end
        c[idx] = tr(M)
    end
    return c
end

_dense_mps_repr(ψ) = _dense_trace(collect(ψ.AL))

# MPO：O[(u1…uN), (d1…dN)] = tr(∏ W[ℓ][:, u_ℓ, :, d_ℓ])
function _dense_mpo_repr(W)
    N = length(W)
    dus = Tuple(size(W[ℓ], 2) for ℓ in 1:N)
    dds = Tuple(size(W[ℓ], 4) for ℓ in 1:N)
    T = eltype(W[1])
    O = zeros(T, dus..., dds...)
    for uidx in CartesianIndices(dus), didx in CartesianIndices(dds)
        u, d = Tuple(uidx), Tuple(didx)
        M = Matrix{T}(I, size(W[1], 1), size(W[1], 1))
        for ℓ in 1:N
            M = M * W[ℓ][:, u[ℓ], :, d[ℓ]]
        end
        O[uidx, didx] = tr(M)
    end
    return O
end

# ---- 非均匀键 profile 的辅助（unit cell > 1 且各 bond 键维不同）----

"在维度 `dim` 的尾部插入 `m` 维零块。"
_zextdim(A::AbstractArray{T}, m::Int, dim::Int) where {T} =
    cat(A, zeros(T, ntuple(i -> i == dim ? m : size(A, i), ndims(A))...); dims = dim)

"把 Q 的列从 χ 补全到 χ+n（新列与旧列正交归一）。"
function _coldone(Q::AbstractMatrix, n::Int)
    Y = randn(eltype(Q), size(Q, 1), n)
    return hcat(Q, Matrix(qr(Y - Q * (Q' * Y)).Q))
end

"把 R 的行从 χ 补全到 χ+n（新行与旧行正交归一）。"
function _rowdone(R::AbstractMatrix, n::Int)
    Y = randn(eltype(R), size(R, 2), n)
    return vcat(R, Matrix(qr(Y - R' * (R * Y)).Q)')
end

"""
键 profile `χ`（各 bond 的键维）的最小非均匀混合规范态：随机左等距 AL 串 +
`C₀ = I_{χ[N]}`，交给 `CanonicalIMPS(ALs, C₀)` 求右规范形式。
"""
function _nonuniform_mps(χ::Vector{Int}, d::Int; T::Type = ComplexF64)
    N = length(χ)
    ALs = Vector{Array{T,3}}(undef, N)
    for ℓ in 1:N
        χl = χ[mod1(ℓ - 1, N)]
        Q, _ = qr(randn(T, χl * d, χ[ℓ]))
        ALs[ℓ] = reshape(Matrix(Q)[:, 1:χ[ℓ]], χl, d, χ[ℓ])
    end
    return CanonicalIMPS(ALs, Matrix{T}(I, χ[N], χ[N]))
end

"""
把 MPS 第 `ℓ` 条 bond 的键维从 χ 零填充放大到 χ+n：**同一物理态**（新增的
Schmidt 方向权重为零），AL/AR 保持等距（新列/新行取正交归一补全），新指标一律
追加在末尾。可用于检验非均匀键下所有观测量必须与填充前逐一相等。
"""
function _padbond!(ψ::CanonicalIMPS, ℓ::Int, n::Int)
    N = length(ψ)
    ℓn = mod1(ℓ + 1, N)
    χ = size(ψ.C[ℓ], 1)
    dl, d, _ = size(ψ.AL[ℓ])
    ψ.AL[ℓ] = reshape(_coldone(reshape(ψ.AL[ℓ], dl * d, χ), n), dl, d, χ + n)
    ψ.AR[ℓ] = _zextdim(ψ.AR[ℓ], n, 3)
    ψ.AL[ℓn] = _zextdim(ψ.AL[ℓn], n, 1)
    dl2, d2, dr2 = size(ψ.AR[ℓn])
    ψ.AR[ℓn] = reshape(_rowdone(reshape(ψ.AR[ℓn], χ, d2 * dr2), n), χ + n, d2, dr2)
    C = zeros(eltype(ψ.C[ℓ]), χ + n, χ + n)
    C[1:χ, 1:χ] = ψ.C[ℓ]
    ψ.C[ℓ] = C
    ψ.AC[ℓ] = _zextdim(ψ.AC[ℓ], n, 3)
    ψ.AC[ℓn] = _zextdim(ψ.AC[ℓn], n, 1)
    return ψ
end

"""
把 `DenseIMPO` 第 `ℓ` 条 bond 的键维零填充放大到 χ+n：**同一算符**（新增 MPO
通道全零），新指标一律追加在末尾（两侧位置必须一致）。
"""
function _padbond(W::DenseIMPO, ℓ::Int, n::Int)
    N = length(W)
    ℓn = mod1(ℓ + 1, N)
    Ws = [copy(w) for w in W.Ws]
    Ws[ℓ] = _zextdim(Ws[ℓ], n, 3)
    Ws[ℓn] = _zextdim(Ws[ℓn], n, 1)
    return DenseIMPO(Ws)
end

"3 站点胞 TFI 的 `SparseIMPO`（unit cell = 3，用于非均匀键的单元胞测试）。"
function _tfim3(; J::Real = 1.0, h::Real = 1.3, T::Type = ComplexF64)
    W = mpohamiltonian(-h * σz(T), [(-J, σx(T), σx(T))])
    return SparseIMPO([W, W, W])
end

