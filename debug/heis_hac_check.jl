using MPSKit, TensorKit, Random, LinearAlgebra
using InfiniteMPSAlgorithms
include(joinpath(@__DIR__, "mpsconvert.jl"))
using TensorOperations

function hac_dense(GL, W, GR, x)
    @tensor y[b, u, b′] := GL[b, wl, a] * x[a, vi, c] *
                           W[wl, u, wr, vi] * GR[c, wr, b′]
    return y
end

function hc_dense(GL, GR, x)
    @tensor y[α, β] := GL[α, w, α′] * x[α′, β′] * GR[β′, w, β]
    return y
end

function solve_channel(X_our, X_mk, mid)
    # X_mk[m] = Σ_k R[m,k] X_our[k]; 向量化 D×D
    Vo = hcat([vec(X_our[:, k, :]) for k in mid]...)   # (D², 3)
    Vm = hcat([vec(X_mk[:, m, :]) for m in mid]...)   # (D², 3)
    return transpose(pinv(transpose(Vo)) * transpose(Vm))  # R (3,3) s.t. Vm ≈ Vo R^T? 下面直接验证
end

function check()
    T = ComplexF64
    D, d = 6, 2
    Random.seed!(20260918)
    ψ = MixedCanonicalMPS([randn(T, D, d, D)])
    lattice = fill(ℂ^d, 1)
    H_our = heisenberg_hamiltonian(T = T)
    H_mk = MPSKit.InfiniteMPOHamiltonian(
        lattice,
        ((1, 2) => 0.25 * (σx(T) ⊗ σx(T) + σy(T) ⊗ σy(T) + σz(T) ⊗ σz(T))),
    )
    ψ_mk = mkinfinitemps(ψ)

    W = tompotensor(H_our[1])
    Wm = mpoarray(convert(TensorMap, H_mk[1]))
    n = size(W, 1)
    mid = 2:(n - 1)

    envs_our = InfiniteMPSAlgorithms.environments(ψ, H_our)
    envs_mk = MPSKit.environments(ψ_mk, H_mk)
    GL = InfiniteMPSAlgorithms.leftenv(envs_our, 1)
    GR = InfiniteMPSAlgorithms.rightenv(envs_our, 1)
    GL_mk0 = envarray(convert(TensorMap, MPSKit.leftenv(envs_mk, 1, ψ_mk)))
    GR_mk0 = envarray(convert(TensorMap, MPSKit.rightenv(envs_mk, 1, ψ_mk)))

    VoL = hcat([vec(GL[:, k, :]) for k in mid]...)
    VmL = hcat([vec(GL_mk0[:, m, :]) for m in mid]...)
    VoR = hcat([vec(GR[:, k, :]) for k in mid]...)
    VmR = hcat([vec(GR_mk0[:, m, :]) for m in mid]...)
    # Vm = Vo * S^T  → S^T = pinv(Vo)*Vm
    SLt = pinv(VoL) * VmL
    SRt = pinv(VoR) * VmR
    SL = transpose(SLt)
    SR = transpose(SRt)
    println("GL channel fit reldiff = ", norm(VoL * SLt - VmL) / norm(VmL))
    println("GR channel fit reldiff = ", norm(VoR * SRt - VmR) / norm(VmR))
    println("SL = ", SL)
    println("SR = ", SR)
    println("SL*SR = ", SL * SR)
    println("SL*conj(SR) = ", SL * conj(SR))

    GL_pred = copy(GL); GR_pred = copy(GR)
    for (im, m) in enumerate(mid)
        GL_pred[:, m, :] = reshape(VmL[:, im], D, D)
        GR_pred[:, m, :] = reshape(VmR[:, im], D, D)
    end

    Random.seed!(42)
    x = randn(T, D, d, D)
    y_pred = hac_dense(GL_pred, Wm, GR_pred, x)
    y_mk = mpsarray(MPSKit.AC_hamiltonian(1, ψ_mk, H_mk, ψ_mk, envs_mk)(mkmpstensor(x)))
    println("HAC: fitted-env + Wm vs MPSKit reldiff = ", norm(y_pred - y_mk) / norm(y_mk))

    y_our = hac_dense(GL, W, GR, x)
    println("HAC: our vs MPSKit reldiff = ", norm(y_our - y_mk) / norm(y_mk))

    c = randn(T, D, D)
    z_pred = hc_dense(GL_pred, GR_pred, c)
    z_mk = reshape(_tkdata(MPSKit.C_hamiltonian(1, ψ_mk, H_mk, ψ_mk, envs_mk)(
        TensorKit.TensorMap(copy(c), ℂ^D, ℂ^D))), D, D)
    println("HC : fitted-env vs MPSKit reldiff = ", norm(z_pred - z_mk) / norm(z_mk))
end

check()
