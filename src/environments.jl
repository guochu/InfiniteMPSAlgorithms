# ---------------- 环境机制（对标 MPSKit src/environments/infinite_envs.jl） ----------------
#
# 具体缓存类型的定义与所属算法放在一起（算法文件的构造器直接产出缓存）：
# - `DMRGCache`：哈密顿量通道 ⟨ψ|H|ψ⟩（基态搜索 VUMPS / IDMRG / TDVP 与能量计算），
#   见 algorithms/idmrg.jl；
# - `MultCache`：MPO 施加通道 ⟨bra|W|ket⟩（迭代 MPO 乘法 mult），
#   见 algorithms/mult.jl；
# - `OverlapCache`：纯重叠通道 ⟨bra|ket⟩（代数运算的变分压缩），
#   见 algorithms/arithmetics.jl。
#
# 本文件只保留共享机制：抽象父类型 `Environments`、环境访问与增量推进、
# 以及各缓存构造器共用的固定点 kernel。

"""
    Environments

⟨below|operator|ket⟩ 型双层网络左右固定点环境的抽象父类型（对标 MPSKit 的
InfiniteEnvironments）；具体类型：[`DMRGCache`](@ref)（哈密顿量通道）、
[`MultCache`](@ref)（MPO 施加）与 [`OverlapCache`](@ref)（纯重叠）。
"""
abstract type Environments end

"leftenv(envs, ℓ)：site ℓ 左环境。"
leftenv(envs::Environments, ℓ::Integer) = envs.lefts[_mod1(ℓ, length(envs.ket))]
"rightenv(envs, ℓ)：site ℓ 右环境。"
rightenv(envs::Environments, ℓ::Integer) = envs.rights[_mod1(ℓ, length(envs.ket))]

_to3(L::AbstractMatrix{T}) where {T} = reshape(L, size(L, 1), 1, size(L, 2))
_to3(L::AbstractArray{T,3}) where {T} = L

# ---- 三元固定点 kernel（MultCache / OverlapCache 构造共用） ----

"三元环境共享的固定点求解（左右主本征向量 + MPSKit 型归一化）。"
function _ternary_fixedpoints(below::InfiniteCanonicalMPS, operator, above::InfiniteCanonicalMPS;
                              tol::Real, krylovdim::Int, maxiter::Int)
    N = length(below)
    L = isnothing(operator) ? N : length(operator)
    (N % L == 0 && length(above) == N) ||
        throw(DimensionMismatch("MPS 与 MPO 单胞长度不兼容"))
    T = promote_type(scalartype(below), scalartype(above))
    Dw = isnothing(operator) ? 1 : size(operator[1], 1)
    Dl = size(below.AL[1], 1)
    Da = size(above.AL[1], 1)
    Dr = size(below.AR[1], 3)
    Wop = isnothing(operator) ? (ℓ -> nothing) : (ℓ -> operator[_mod1(ℓ, L)])

    # ---- 左固定点：T_L(above.AL, operator, below.AL) 的主本征向量 ----
    Tleft = function (v::AbstractVector)
        GL = reshape(v, Dl, Dw, Da)
        for ℓ in 1:N
            W = Wop(ℓ)
            GL = isnothing(W) ? push_env_left(GL, below.AL[ℓ], above.AL[ℓ]) :
                 push_env_left(GL, below.AL[ℓ], W, above.AL[ℓ])
        end
        return vec(GL)
    end
    _, GL1 = eigsolve(Tleft, ones(T, Dl * Dw * Da), 1, :LM;
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    GLs = Vector{Array{T,3}}(undef, N)
    GLs[1] = reshape(GL1[1], Dl, Dw, Da)
    for ℓ in 2:N
        W = Wop(ℓ - 1)
        GLs[ℓ] = isnothing(W) ? push_env_left(GLs[ℓ-1], below.AL[ℓ-1], above.AL[ℓ-1]) :
                 push_env_left(GLs[ℓ-1], below.AL[ℓ-1], W, above.AL[ℓ-1])
    end

    # ---- 右固定点：T_R(above.AR, operator, below.AR) 的主本征向量 ----
    Tright = function (v::AbstractVector)
        GR = reshape(v, Da, Dw, Dr)
        for ℓ in N:-1:1
            W = Wop(ℓ)
            GR = isnothing(W) ? push_env_right(GR, above.AR[ℓ], below.AR[ℓ]) :
                 push_env_right(GR, above.AR[ℓ], W, below.AR[ℓ])
        end
        return vec(GR)
    end
    _, GRN = eigsolve(Tright, ones(T, Da * Dw * Dr), 1, :LM;
                      tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    GRs = Vector{Array{T,3}}(undef, N)
    GRs[N] = reshape(GRN[1], Da, Dw, Dr)
    for ℓ in N-1:-1:1
        W = Wop(ℓ + 1)
        GRs[ℓ] = isnothing(W) ? push_env_right(GRs[ℓ+1], above.AR[ℓ+1], below.AR[ℓ+1]) :
                 push_env_right(GRs[ℓ+1], above.AR[ℓ+1], W, below.AR[ℓ+1])
    end

    # ---- 归一化（对标 MPSKit：GR Frobenius 归一、GL 按局部重叠 λ 缩放）----
    for ℓ in 1:N
        GRs[ℓ] .= GRs[ℓ] ./ norm(GRs[ℓ])
    end
    for ℓ in 1:N
        inext = _mod1(ℓ + 1, N)
        GLn = GLs[inext]
        GR = GRs[ℓ]
        Cnew = _mapC(GLn, GR, above.C[ℓ])
        λ = dot(below.C[ℓ], Cnew)
        λ == 0 && error("三元环境：site $ℓ 局部重叠 λ = 0")
        GLs[inext] .= GLn ./ λ
    end
    return GLs, GRs
end

# ---- Hamiltonian 路径的逐层 kernel（DMRGCache 构造专用） ----

"通道切片 (l → i) 的左推进：L′ = Σ conj(A[a,ū,a′])·L[a,b]·Wl[ū,d]·A[b,d,b′]。"
function _push_slice_left(L::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor L′[a′, b′] := conj(A[a, ū, a′]) * L[a, b] * Wl[ū, d] * A[b, d, b′]
    return L′
end

"通道切片 (i → l) 的右推进：对标 MPSKit transfer_right，A 与 ket(d) 收缩、conj(A) 与 bra(u) 收缩。"
function _push_slice_right(R::AbstractMatrix, Wl::AbstractMatrix, A::AbstractArray{T,3}) where {T}
    @tensor R′[a′, b′] := A[a′, d, a] * Wl[ū, d] * conj(A[b′, ū, b]) * R[a, b]
end

"全胞左扫描（对标 MPSKit left_cyclethrough!）：`GL[site+1, i] = Σ_{l≤i} W_site[l→i]·GL[site, l]`。"
function _left_cyclethrough!(lefts, Wds, ALs, i::Int, N::Int, Ds, T)
    for site in 1:N
        snext = site == N ? 1 : site + 1
        tgt = zeros(T, Ds[snext], Ds[snext])
        for l in 1:i
            tgt .+= _push_slice_left(lefts[site][:, l, :],
                                     view(Wds[site], l, :, i, :), ALs[site])
        end
        lefts[snext][:, i, :] .= tgt
    end
    return lefts
end

"全胞右扫描（对标 MPSKit right_cyclethrough!）：`GR[site−1, i] = Σ_{l≥i} W_site[i→l]·GR[site, l]`。"
function _right_cyclethrough!(rights, Wds, ARs, i::Int, N::Int, Ds, T)
    nl = size(Wds[1], 1)
    for site in N:-1:1
        sprev = site == 1 ? N : site - 1
        tgt = zeros(T, Ds[sprev], Ds[sprev])
        for l in i:nl
            tgt .+= _push_slice_right(rights[site][:, l, :],
                                      view(Wds[site], i, :, l, :), ARs[site])
        end
        rights[sprev][:, i, :] .= tgt
    end
    return rights
end

# ---- 环境的规范化与增量推进 ----

"""
    normalize_envs!(envs, below, operator, above) -> envs

对标 MPSKit `normalize!`：
- 右环境归一化到 norm 1；
- 左环境缩放使得 `dot(C, C_hamiltonian * C) = 1`。

避免 IDMRG 扫描中环境范数爆炸。
"""
function normalize_envs!(envs::Environments, below::InfiniteCanonicalMPS,
                    operator::Union{Nothing,InfiniteMPO,MPOHamiltonian},
                    above::InfiniteCanonicalMPS)
    N = length(below)
    for i in 1:N
        # 右环境归一化到 norm 1
        r = envs.rights[i]
        nr = norm(r)
        if nr > 0
            r ./= nr
        end
        # λ = dot(C[i], C_hamiltonian(i) * C[i])
        hC = C_hamiltonian(i, below, operator, above, envs)
        Cproj = hC(below.C[i])
        λ = dot(below.C[i], Cproj)
        if abs(λ) > 0
            # GL[i+1] *= inv(λ)
            l = envs.lefts[_mod1(i + 1, N)]
            l ./= λ
        end
    end
    return envs
end

"""
    transfer_leftenv!(envs, ψ, operator, ψ2, site) -> envs

把左环境从 `site − 1` 推进到 `site`（IDMRG 扫描的增量更新，对标 MPSKit 的
`transfer_leftenv!`）。
"""
function transfer_leftenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓm = _mod1(site - 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa MPOHamiltonian
        tompotensor(operator[ℓm])
    else
        operator[ℓm]
    end
    if isnothing(W)
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], ψ.AL[ℓm])
    else
        envs.lefts[ℓ] = push_env_left(envs.lefts[ℓm], W, ψ.AL[ℓm])
    end
    return envs
end

"把右环境从 `site + 1` 推进到 `site`。"
function transfer_rightenv!(envs::Environments, ψ, operator, ψ2, site::Int)
    ℓ = _mod1(site, length(ψ))
    ℓp = _mod1(site + 1, length(ψ))
    W = if isnothing(operator)
        nothing
    elseif operator isa MPOHamiltonian
        tompotensor(operator[ℓp])
    else
        operator[ℓp]
    end
    if isnothing(W)
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], ψ.AR[ℓp])
    else
        envs.rights[ℓ] = push_env_right(envs.rights[ℓp], W, ψ.AR[ℓp])
    end
    return envs
end
