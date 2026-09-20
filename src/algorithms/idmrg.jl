# ---------------- IDMRG（严格对标 MPSKit src/algorithms/groundstate/idmrg.jl，single-site） ----------------

"""
    IDMRG(; tol, maxiter, verbosity, alg_gauge, alg_eigsolve)

single-site 无限 DMRG（对标 MPSKit 的 `IDMRG`）。

每轮迭代（MPSKit 模板）：
1. 前向扫描：逐 site 解 `AC_hamiltonian` 最小本征对，`left_orth` 分裂为 `AL/C`，
   `transfer_leftenv!` 增量推进环境；
2. 后向扫描：逐 site 再解 AC，`right_orth` 分裂为 `C/AR`，
   `transfer_rightenv!` 增量推进环境；
3. 收敛判据 `ϵ = ‖C − C_old‖`（取 bond 0 的中心矩阵），能量增量 `ΔE = ΔE_iter/2`。

结束后从 `AR` 重建混合规范态（对标 `InfiniteMPS(mps.AR)`）并重算环境。
"""
@kwdef struct IDMRG{A} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve::A = Defaults.alg_eigsolve()
end

"_transpose_tail(A) / _transpose_front(A)：前后端指标交换 `(Dl, d, Dr) ↔ (Dr, d, Dl)`。"
_transpose_tail(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))
_transpose_front(A::AbstractArray{T,3}) where {T} = permutedims(A, (3, 2, 1))

"_left_orth3(AC; alg)：`(Dl·d, Dr)` QR 分裂 → `(AL, C)`（`positive = true` 即 `QRpos`）。"
function _left_orth3(AC::AbstractArray{T,3}; alg = Defaults.alg_orth()) where {T}
    Dl, d, Dr = size(AC)
    Q, R = leftorth(reshape(AC, Dl * d, Dr); alg = alg)
    return reshape(Q, Dl, d, size(Q, 2)), R
end

"_right_orth3(AC; alg)：`(Dl, d·Dr)` LQ 分裂 → `(C, AR)`。"
function _right_orth3(AC::AbstractArray{T,3}; alg = LQpos()) where {T}
    Dl, d, Dr = size(AC)
    L, Q = rightorth(reshape(AC, Dl, d * Dr); alg = alg)
    return L, reshape(Q, size(L, 2), d, Dr)
end

"MPSKit 的 `_localupdate_sweep_idmrg!`：前向 + 后向扫描，返回 `(ψ, envs, C_old, E)`。"
function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve)
    N = length(ψ)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:N
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = _left_orth3(ψ.AC[pos])
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end
    # right to left sweep
    for pos in N:-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.C[pos - 1], ψ.AR[pos] = _right_orth3(ψ.AC[pos])
        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end

function find_groundstate(ψ₀::MixedCanonicalMPS, operator, alg::IDMRG,
                          envs::Environments = environments(ψ₀, operator))
    ψ = copy(ψ₀)
    ϵ = calc_galerkin(ψ, operator, envs)
    E = expectation_value(ψ, operator, envs)
    alg.verbosity > 0 && _logiter(stdout, "IDMRG", 0, ϵ, "f" => E)
    iter = 0
    for outer iter in 1:alg.maxiter
        # MPSKit 第 iter 次扫描时 state.iter = iter-1
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter - 1, ϵ)
        ψ, envs, C_old, E_new = _localupdate_sweep_idmrg!(ψ, operator, envs, alg_eigsolve)
        # error criterion（bond 0 中心矩阵之差）
        ϵ = norm(ψ.C[0] - C_old)
        # new energy
        ΔE = (E_new - E) / 2
        E = E_new
        alg.verbosity > 0 && _logiter(stdout, "IDMRG", iter, ϵ, "f" => E, "ΔE" => ΔE)
        ϵ ≤ alg.tol && break
    end
    # 规范恢复：从 AR 重建（对标 MPSKit 的 `InfiniteMPS(mps.AR)`）
    N = length(ψ)
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = MixedCanonicalMPS([ψ.AR[ℓ] for ℓ in 1:N];
                           tol = alg_gauge.tol, maxiter = alg_gauge.maxiter)
    recalculate!(envs, ψ′, operator)
    return ψ′, envs, ϵ
end
