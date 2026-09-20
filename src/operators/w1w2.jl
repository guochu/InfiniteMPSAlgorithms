# W^I / W^II 时间演化 MPO —— 接口对齐 MPSKit（`WI`/`WII`/`make_time_mpo`，作用于
# JordanMPOTensor）。参考 arXiv:1407.1832 "Time-evolving a matrix product state
# with long-ranged interactions"。W^II 的块算符矩阵指数：拼成稠密 (4d)×(4d) 矩阵
# 调 LinearAlgebra.exp，与 TEMPO 经 ExpExp 的块指数等价。

"""
    WI(; tol, maxiter)

W^I 一阶时间演化步进器（一阶 MPO 近似；对标 MPSKit 的 `WI`）。
"""
@kwdef struct WI <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

"""
    WII(; tol, maxiter)

W^II 二阶时间演化步进器（块指数 MPO 近似；对标 MPSKit 的 `WII`）。
"""
@kwdef struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

# ---- Jordan 块提取 ----

get_A(W::JordanMPOTensor) = [W.A[i, :, j, :] for i in 1:size(W.A, 1), j in 1:size(W.A, 3)]
get_B(W::JordanMPOTensor) = [W.B[i, :, :] for i in 1:size(W.B, 1)]
get_C(W::JordanMPOTensor) = [W.C[:, j, :] for j in 1:size(W.C, 2)]
get_D(W::JordanMPOTensor) = W.D

function _sqrt2(dt::Complex)
    r = sqrt(dt)
    return r, r
end

function _sqrt2(dt::Real)
    if dt >= zero(dt)
        r = sqrt(dt)
        return r, r
    else
        r = sqrt(-dt)
        return r, -r
    end
end

# 演化后的恒等通道载荷 `WD`（`Wd[1,:,1,:] = WD`），传播块按原 Jordan 块位置放置。
function _timempo_dense(WA, WB, WC, WD)
    a1, a2 = size(WA)
    T = promote_type(eltype(WD), eltype(eltype(WA)),
                     eltype(eltype(WB)), eltype(eltype(WC)))
    d = size(WD, 1)
    n = a1 + 1
    Wd = zeros(T, n, d, n, d)
    Wd[1, :, 1, :] = WD
    for l in 1:a2
        Wd[1, :, l+1, :] = WC[l]
    end
    for l in 1:a1
        Wd[l+1, :, 1, :] = WB[l]
        for m in 1:a2
            Wd[l+1, :, m+1, :] = WA[l, m]
        end
    end
    return Wd
end

function _timempo_dense(W::JordanMPOTensor, dt::Number, alg::WI)
    WA = get_A(W)
    δ₁, δ₂ = _sqrt2(dt)
    WB = get_B(W) .* δ₁
    WC = get_C(W) .* δ₂
    D = get_D(W)
    WD = Matrix{promote_type(scalartype(W), typeof(dt))}(I, size(D, 1), size(D, 1)) + dt .* D
    return _timempo_dense(WA, WB, WC, WD)
end

function _timempo_dense(W::JordanMPOTensor, dt::Number, alg::WII)
    A, B, C, D = get_A(W), get_B(W), get_C(W), get_D(W)
    d = size(W.A, 2)
    T = promote_type(scalartype(W), typeof(dt))
    Ddt = dt .* D
    WD = exp(Matrix(Ddt))
    s1, s2 = size(A)
    δ₁, δ₂ = _sqrt2(dt)

    WA = Array{Matrix{T},2}(undef, size(A))
    WB = Array{Matrix{T},1}(undef, length(B))
    WC = Array{Matrix{T},1}(undef, length(C))

    for a in 1:s1, b in 1:s2
        M = zeros(T, 4d, 4d)
        M[1:d, 1:d] .= Ddt
        M[d+1:2d, 1:d] .= δ₂ .* C[b]
        M[d+1:2d, d+1:2d] .= Ddt
        M[2d+1:3d, 1:d] .= δ₁ .* B[a]
        M[2d+1:3d, 2d+1:3d] .= Ddt
        M[3d+1:4d, 1:d] .= A[a, b]
        M[3d+1:4d, d+1:2d] .= δ₁ .* B[a]
        M[3d+1:4d, 2d+1:3d] .= δ₂ .* C[b]
        M[3d+1:4d, 3d+1:4d] .= Ddt
        mexp = exp(M)[:, 1:d]
        WC[b] = mexp[(d+1):2d, :]
        WB[a] = mexp[(2d+1):3d, :]
        WA[a, b] = mexp[(3d+1):4d, :]
    end
    return _timempo_dense(WA, WB, WC, WD)
end

"""
    make_time_mpo(bulk::JordanMPOTensor, dt, alg::Union{WI,WII};
                  imaginary_evolution = false) -> InfiniteMPO
    make_time_mpo(H::MPOHamiltonian, dt, alg; kwargs...) -> InfiniteMPO

构造近似 `exp(-i·H·dt)` 的周期时间演化 MPO（对标 MPSKit 的 `make_time_mpo`；
`imaginary_evolution = true` 时为 `exp(-H·dt)`）。内部为 W^I/W^II 块指数算法；
`MPOHamiltonian` 对单胞内**每个 site** 的 Jordan 张量分别演化（对标 MPSKit 的
`tmap(parent(H)) do W ... end` + `InfiniteMPO(PeriodicArray(O))`），支持任意
单胞长度。
"""
function make_time_mpo(bulk::JordanMPOTensor, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    return InfiniteMPO([_timempo_dense(bulk, δ, alg)])
end

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::Union{WI,WII};
                       imaginary_evolution::Bool = false)
    δ = imaginary_evolution ? -dt : -im * dt
    # 对标 MPSKit：逐 site 演化单胞内每个 Jordan 张量（支持任意单胞长度）
    O = [_timempo_dense(W, δ, alg) for W in parent(H)]
    return InfiniteMPO(O)
end
