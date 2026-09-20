# ---- 周期 trace 表示的稠密辅助（algebra 与 api 共用）----
# MPS：c[(s1…sN)] = tr(∏ AL[ℓ][:, s_ℓ, :])（左规范张量的周期 trace）。
# 注意必须用 AL 而非 AC：混合规范下 AC = AL·C 在张量间插入 C 矩阵，
# tr(∏AC) 是规范依赖的 C 加权量；而 tr(∏AL) = tr(∏A_raw)/λ（λ 实正），
# 是构造器确定性产出、规范变换（含相位）下不变的射线幅值。
function _dense_mps_repr(ψ)
    N = length(ψ)
    dims = Tuple(size(ψ.AL[ℓ], 2) for ℓ in 1:N)
    T = eltype(ψ.AL[1])
    c = zeros(T, dims)
    for idx in CartesianIndices(dims)
        s = Tuple(idx)
        M = Matrix{T}(I, size(ψ.AL[1], 1), size(ψ.AL[1], 1))
        for ℓ in 1:N
            M = M * ψ.AL[ℓ][:, s[ℓ], :]
        end
        c[idx] = tr(M)
    end
    return c
end

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
