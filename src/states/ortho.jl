# ---------------- gauge-fixing algorithms (mirroring MPSKit src/states/ortho.jl) ----------------

# mirrors MPSKit's _GAUGE_ALG_EIGSOLVE: non-Hermitian (the mixed transfer map
# is not normal), with the dynamic tolerance factor fixed to 1 so the inner
# solve runs at clamp(ϵ²) each gauge iteration
const _GAUGE_ALG_EIGSOLVE = Defaults.alg_eigsolve(; ishermitian = false, tol_factor = 1)

"""
    LeftCanonical(; tol, maxiter, verbosity, alg_orth, alg_eigsolve, eig_miniter)

Algorithm bringing an `CanonicalIMPS` to the left-canonical form
(mirrors MPSKit's `LeftCanonical`).
"""
@kwdef struct LeftCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.VERBOSE_WARN
    alg_orth = Defaults.alg_orth()
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    eig_miniter::Int = 10
end

"""
    RightCanonical(; tol, maxiter, verbosity, alg_orth, alg_eigsolve, eig_miniter)

Algorithm bringing an `CanonicalIMPS` to the right-canonical form
(mirrors MPSKit's `RightCanonical`).
"""
@kwdef struct RightCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.VERBOSE_WARN
    alg_orth = Defaults.alg_orth()
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    eig_miniter::Int = 10
end

"""
    MixedCanonical(; order = :LR, kwargs...)

Mixed-canonicalization algorithm (mirrors MPSKit's `MixedCanonical`):
left-then-right (`:LR`) or right-then-left (`:RL`).
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
    gaugefix!(ψ::CanonicalIMPS, A, C₀ = ψ.C[end]; order = :LR, kwargs...) -> ψ
    gaugefix!(ψ::CanonicalIMPS, A, C₀, alg::Algorithm) -> ψ

Write the gauge information of `A` (plain site tensors or left/right-canonical
tensors) into `ψ` (mirrors MPSKit's `gaugefix!`). `order` is one of
`:L`, `:R`, `:LR`, `:RL`.
"""
function gaugefix!(ψ::CanonicalIMPS, A, C₀ = ψ.C[end]; order = :LR, kwargs...)
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

function gaugefix!(ψ::CanonicalIMPS, A, C₀, alg::MixedCanonical)
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

function gaugefix!(ψ::CanonicalIMPS, A, C₀, alg::LeftCanonical)
    uniform_leftorth!((ψ.AL, ψ.C), A, C₀, alg)
    ψ.AC .= _mul_ALC(ψ.AL, ψ.C)
    return ψ
end
function gaugefix!(ψ::CanonicalIMPS, A, C₀, alg::RightCanonical)
    uniform_rightorth!((ψ.AR, ψ.C), A, C₀, alg)
    ψ.AC .= _mul_CAR(ψ.C, ψ.AR)
    return ψ
end

"`AC[ℓ] = C[ℓ-1]·AR[ℓ]` (AR-side definition of the center tensors)."
function _mul_CAR(C::PeriodicVector{B}, AR::PeriodicVector{A}) where {A<:Array{T,3},B<:Matrix{T}} where {T}
    PeriodicVector(Array{T,3}[_mul_CAR(C[mod1(ℓ - 1, length(C))], AR[ℓ]) for ℓ in 1:length(AR)])
end
_mul_CAR(C::AbstractMatrix{T}, AR::AbstractArray{T,3}) where {T} =
    begin
        @tensor AC[x, s, y] := C[x, a] * AR[a, s, y]
    end

# ---------------- uniform orthogonalization iterations (mirroring uniform_leftorth!/uniform_rightorth!) ----------------
#
# Each iteration (mirrors MPSKit's IterativeSolver{LeftCanonical}):
#   1. gauge_eigsolve_step! (once iter ≥ eig_miniter): refresh C[N] from the
#      dominant fixed point of the MIXED transfer map flip(T(A, AL)) via the
#      eigensolver, brought to the positive representative by a QR factor.
#      This is what makes the iteration converge reliably even when the
#      transfer map is non-normal (pure QR power sweeps can stall there).
#   2. gauge_orth_step!: one periodic QR sweep.
#   3. ϵ = ‖C_eigsolve − C_sweep‖ (NOT the successive-iterate difference).
#
# The sweep acts on a private workspace copy of A (mirrors MPSKit's
# pre-allocated A_tail), so callers may pass ψ.AL itself (A may alias the
# output storage) without having the problem tensors mutated mid-iteration.

function uniform_leftorth!((AL, C), A, C₀, alg::LeftCanonical)
    N = length(AL)
    C[N] = normalize!(copy(C₀))
    Awork = [copy(A[i]) for i in 1:N]
    iter = 0
    ϵ = float(real(one(real(scalartype(A[1]))))) * Inf
    while true
        # gauge_eigsolve_step!: C[N] = R factor of the mixed-transfer fixed point
        if iter ≥ alg.eig_miniter
            ealg = updatetol(alg.alg_eigsolve, 1, ϵ^2)
            _, evec = fixedpoint(TransferMatrix(Awork, AL; side = :left),
                                 vec(C[N]), :LM, ealg)
            _, C[N] = leftorth(reshape(evec, size(C[N])...); alg = alg.alg_orth)
        end
        C_pre = copy(C[N])
        # gauge_orth_step!: per-site C[i-1]·A[i] → QR → AL[i], C[i]
        for i in 1:N
            Dli, d, Dri = size(Awork[i])
            @tensor M[a, s, b] := C[i - 1][a, ā] * Awork[i][ā, s, b]
            Q, Rf = leftorth(reshape(M, Dli * d, Dri); alg = alg.alg_orth)
            AL[i] = reshape(Q, Dli, d, size(Q, 2))
            C[i] = Rf
        end
        normalize!(C[N])
        ϵ = norm(C[N] - C_pre)
        iter += 1
        ϵ < alg.tol && return AL, C
        if iter > alg.maxiter
            alg.verbosity ≥ Defaults.VERBOSE_WARN &&
                @warn "uniform_leftorth!: not converged" maxiter = alg.maxiter ϵ
            return AL, C
        end
    end
end

function uniform_rightorth!((AR, C), A, C₀, alg::RightCanonical)
    N = length(AR)
    C[N] = normalize!(copy(C₀))
    Awork = [copy(A[i]) for i in 1:N]
    iter = 0
    ϵ = float(real(one(real(scalartype(A[1]))))) * Inf
    while true
        # gauge_eigsolve_step!: C[N] = L factor of the mixed-transfer fixed point
        if iter ≥ alg.eig_miniter
            ealg = updatetol(alg.alg_eigsolve, 1, ϵ^2)
            _, evec = fixedpoint(TransferMatrix(Awork, AR; side = :right),
                                 vec(C[N]), :LM, ealg)
            C[N], _ = rightorth(reshape(evec, size(C[N])...); alg = alg.alg_orth')
        end
        C_pre = copy(C[N])
        # gauge_orth_step!: per-site A[i]·C[i] → LQ → C[i-1], AR[i]
        for i in N:-1:1
            Dli, d, Dri = size(Awork[i])
            @tensor M[a, s, b] := Awork[i][a, s, ā] * C[i][ā, b]
            Lf, Q = rightorth(reshape(M, Dli, d * Dri); alg = alg.alg_orth')
            AR[i] = reshape(Q, size(Lf, 2), d, Dri)
            C[i - 1] = Lf
        end
        normalize!(C[N])
        ϵ = norm(C[N] - C_pre)
        iter += 1
        ϵ < alg.tol && return AR, C
        if iter > alg.maxiter
            alg.verbosity ≥ Defaults.VERBOSE_WARN &&
                @warn "uniform_rightorth!: not converged" maxiter = alg.maxiter ϵ
            return AR, C
        end
    end
end

# ---------------- regauge! (mirroring MPSKit's regauge!) ----------------

"""
    regauge!(AC, C; alg = Defaults.alg_orth()) -> AL
    regauge!(CL, AC; alg = Defaults.alg_orth()) -> AR

Re-canonicalize an updated `(AC, C)` tensor pair into a consistent `AL`
(or `(CL, AC)` into `AR`), minimizing `‖AC_i − AL_i·C_i‖`
(respectively `‖AC_i − C_{i-1}·AR_i‖`).
"""
function regauge!(AC::AbstractArray{T,3}, C::AbstractMatrix{T}; alg = Defaults.alg_orth()) where {T}
    Dl, d, Dr = size(AC)
    Q_AC, _ = leftorth(reshape(AC, Dl * d, Dr); alg = alg)
    Q_C, _ = leftorth(copy(C); alg = alg)
    return reshape(Q_AC * Q_C', Dl, d, Dr)
end

function regauge!(CL::AbstractMatrix{T}, AC::AbstractArray{T,3}; alg = Defaults.alg_orth()) where {T}
	Dl, d, Dr = size(AC)
	_, Q_AC = rightorth(reshape(AC, Dl, d * Dr); alg = alg')
	_, Q_C = rightorth(copy(CL); alg = alg')
	return reshape(Q_C' * Q_AC, size(Q_C, 2), d, Dr)
end

function regauge!(ACs::AbstractVector, Cs::AbstractVector; kwargs...)
    return Array{eltype(ACs[1]),3}[regauge!(ACs[i], Cs[i]; kwargs...) for i in eachindex(ACs)]
end

# The gaugefix!/rank-4 regauge! methods for CanonicalIMPO live in
# states/canonicalmpo.jl (that file is included after this one — the
# CanonicalIMPO type is defined there)
