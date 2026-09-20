# ---------------- 规范化算法（对标 MPSKit src/states/ortho.jl） ----------------

"""
    LeftCanonical(; tol, maxiter, verbosity, alg_orth, alg_eigsolve, eig_miniter)

将 `InfiniteCanonicalMPS` 化为左规范形式的算法（对标 MPSKit 的 `LeftCanonical`）。
"""
@kwdef struct LeftCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.VERBOSE_WARN
    alg_orth = Defaults.alg_orth()
    alg_eigsolve = Defaults.alg_eigsolve(; ishermitian = false, tol = Defaults.tolgauge)
    eig_miniter::Int = 3
end

"""
    RightCanonical(; tol, maxiter, verbosity, alg_orth, alg_eigsolve, eig_miniter)

将 `InfiniteCanonicalMPS` 化为右规范形式的算法（对标 MPSKit 的 `RightCanonical`）。
"""
@kwdef struct RightCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.VERBOSE_WARN
    alg_orth = Defaults.alg_orth()
    alg_eigsolve = Defaults.alg_eigsolve(; ishermitian = false, tol = Defaults.tolgauge)
    eig_miniter::Int = 3
end

"""
    MixedCanonical(; order = :LR, kwargs...)

混合规范化算法（对标 MPSKit 的 `MixedCanonical`）：先左后右（`:LR`）或先右后左（`:RL`）。
"""
struct MixedCanonical <: Algorithm
    alg_leftcanonical::LeftCanonical
    alg_rightcanonical::RightCanonical
    order::Symbol
end
function MixedCanonical(; tol::Real = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
                         order::Symbol = :LR, kwargs...)
    left = LeftCanonical(; tol = tol, maxiter = maxiter, kwargs...)
    right = RightCanonical(; tol = tol, maxiter = maxiter, kwargs...)
    return MixedCanonical(left, right, order)
end

"""
    gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀ = ψ.C[end]; order = :LR, kwargs...) -> ψ
    gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀, alg::Algorithm) -> ψ

把 `A`（普通 site 张量串或左/右规范张量串）的规范信息写入 `ψ`
（对标 MPSKit 的 `gaugefix!`）。`order` 可为 `:L`、`:R`、`:LR`、`:RL`。
"""
function gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀ = ψ.C[end]; order = :LR, kwargs...)
    alg = if order === :LR || order === :RL
        MixedCanonical(; order = order, kwargs...)
    elseif order === :L
        LeftCanonical(; kwargs...)
    elseif order === :R
        RightCanonical(; kwargs...)
    else
        throw(ArgumentError("Invalid order: $order"))
    end
    return gaugefix!(ψ, A, C₀, alg)
end

function gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀, alg::MixedCanonical)
    if alg.order === :LR
        gaugefix!(ψ, A, C₀, alg.alg_leftcanonical)
        gaugefix!(ψ, ψ.AL, ψ.C[end], alg.alg_rightcanonical)
    elseif alg.order === :RL
        gaugefix!(ψ, A, C₀, alg.alg_rightcanonical)
        gaugefix!(ψ, ψ.AR, ψ.C[end], alg.alg_leftcanonical)
    else
        throw(ArgumentError("Invalid order: $(alg.order)"))
    end
    return ψ
end

function gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀, alg::LeftCanonical)
    uniform_leftorth!((ψ.AL, ψ.C), A, C₀, alg)
    return ψ
end
function gaugefix!(ψ::InfiniteCanonicalMPS, A, C₀, alg::RightCanonical)
    uniform_rightorth!((ψ.AR, ψ.C), A, C₀, alg)
    return ψ
end

# ---------------- uniform 正交化迭代（对标 uniform_leftorth!/uniform_rightorth!） ----------------

function uniform_leftorth!((AL, C), A, C₀, alg::LeftCanonical)
    N = length(AL)
    T = eltype(A[1])
    C[N] = normalize!(copy(C₀))
    Dl = size(A[1], 1)
    iter = 0
    ϵ = Inf
    while true
        iter += 1
        C_old = copy(C[N])
        # 逐 site 左正交化：C[i-1]·A[i] → QR → AL[i], C[i]
        for i in 1:N
            Ai = A[i]
            Dli, d, Dri = size(Ai)
            @tensor M[a, s, b] := C[i - 1][a, ā] * Ai[ā, s, b]
            Q, Rf = leftorth(reshape(M, Dli * d, Dri); alg = alg.alg_orth)
            AL[i] = reshape(Q, Dli, d, size(Q, 2))
            C[i] = Rf
        end
        normalize!(C[N])
        ϵ = norm(C[N] - C_old)
        (ϵ < alg.tol || iter >= alg.maxiter) && break
    end
    return AL, C
end

function uniform_rightorth!((AR, C), A, C₀, alg::RightCanonical)
    N = length(AR)
    C[N] = normalize!(copy(C₀))
    iter = 0
    ϵ = Inf
    while true
        iter += 1
        C_old = copy(C[N])
        # 逐 site 右正交化：A[i]·C[i] → LQ → C[i-1], AR[i]
        alg_right = LQpos()
        for i in N:-1:1
            Ai = A[i]
            Dli, d, Dri = size(Ai)
            @tensor M[a, s, b] := Ai[a, s, ā] * C[i][ā, b]
            Lf, Q = rightorth(reshape(M, Dli, d * Dri); alg = alg_right)
            AR[i] = reshape(Q, size(Lf, 2), d, Dri)
            C[i - 1] = Lf
        end
        normalize!(C[N])
        ϵ = norm(C[N] - C_old)
        (ϵ < alg.tol || iter >= alg.maxiter) && break
    end
    return AR, C
end

# ---------------- regauge!（对标 MPSKit 的 regauge!） ----------------

"""
    regauge!(AC, C; alg = Defaults.alg_orth()) -> AL
    regauge!(CL, AC; alg = Defaults.alg_orth()) -> AR

把更新后的 `(AC, C)` 张量对重新规范成一致的 `AL`（或 `(CL, AC)` → `AR`），
最小化 `‖AC_i − AL_i·C_i‖`（或 `‖AC_i − C_{i-1}·AR_i‖`）。
"""
function regauge!(AC::AbstractArray{T,3}, C::AbstractMatrix{T}; alg = Defaults.alg_orth()) where {T}
    Dl, d, Dr = size(AC)
    Q_AC, _ = leftorth(reshape(AC, Dl * d, Dr); alg = alg)
    Q_C, _ = leftorth(copy(C); alg = alg)
    return reshape(Q_AC * Q_C', Dl, d, Dr)
end

function regauge!(CL::AbstractMatrix{T}, AC::AbstractArray{T,3}; alg = Defaults.alg_orth()) where {T}
    Dl, d, Dr = size(AC)
    _, Q_AC = rightorth(reshape(AC, Dl, d * Dr); alg = alg)
    _, Q_C = rightorth(copy(CL); alg = alg)
    return reshape(Q_C' * Q_AC, size(Q_C, 2), d, Dr)
end

function regauge!(ACs::AbstractVector, Cs::AbstractVector; kwargs...)
    return Array{eltype(ACs[1]),3}[regauge!(ACs[i], Cs[i]; kwargs...) for i in eachindex(ACs)]
end
