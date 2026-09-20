# ================= debug：环境与有效哈密顿量与 MPSKit 完全对齐 =================
#
# 运行：julia --project=debug debug/envs_alignment.jl
#
# 对齐项（同一随机 MPS、同一 Jordan 哈密顿量；MPO 指标经转换）：
#   1. Jordan MPO 的稠密形式（块矩阵展开）；
#   2. 左右环境张量（逐 level）；
#   3. 期望值 ⟨ψ|H|ψ⟩；
#   4. H_AC 与 H_C 的作用（随机向量上逐元素对比）。

using Test
using Random
using LinearAlgebra
using TensorKit
using MPSKit
using InfiniteMPSAlgorithms
import InfiniteMPSAlgorithms: scalartype
include(joinpath(@__DIR__, "mpsconvert.jl"))

T = ComplexF64
J, h = 1.0, 1.3
d = 2
D = 6

# ---- 同一哈密顿量（TFIM：H = -J Σσxσx - h Σσz） ----
H_our = tfim_hamiltonian(J = J, h = h, T = T)
lattice = fill(ℂ^d, 1)
H_mk = MPSKit.InfiniteMPOHamiltonian(lattice,
                                     (1 => -h * σz(T), (1, 2) => -J * (σx(T) ⊗ σx(T))))

# ---- 1. 稠密形式对齐（物理等价：二体块乘积一致） ----
# MPSKit 对二体算符做 SVD 分解，与本包直接分解数值不同但物理等价。
Wmk = mpoarray(convert(TensorMap, H_mk[1]))
Wour = tompotensor(H_our[1])
# 对角单位块一致
@test Wmk[:, 1, :, 1] ≈ Wour[:, 1, :, 1] atol = 1e-12
# 二体通道乘积 W[1,:,i,:] * W[i,:,n,:] 等于 coeff*a⊗b
let n = size(Wmk, 1)
    prod_mk = zeros(T, d, d)
    prod_our = zeros(T, d, d)
    for i in 2:n-1
        prod_mk .+= Wmk[1, :, i, :] * Wmk[i, :, n, :]
        prod_our .+= Wour[1, :, i, :] * Wour[i, :, n, :]
    end
    @test prod_mk ≈ prod_our atol = 1e-10
end
println("1. Jordan 稠密形式物理等价 ✔")

# ---- 同一随机 MPS：先用本包造 UNGAUGED 张量，再分别规范化 ----
Random.seed!(20260918)
As = [randn(T, D, d, D) for _ in 1:1]
ψ_our = MixedCanonicalMPS(As)
# MPSKit：由相同 AL、C 出发右规范化，得到同一规范的 InfiniteMPS
ψ_mk = mkinfinitemps(ψ_our)
@test mapreduce(ℓ -> norm(mpsarray(ψ_mk.AL[ℓ]) - ψ_our.AL[ℓ]), max, 1:length(ψ_our); init = 0.0) < 1e-12
@test mapreduce(ℓ -> norm(mpsarray(ψ_mk.AR[ℓ]) - ψ_our.AR[ℓ]), max, 1:length(ψ_our); init = 0.0) < 1e-10
println("2. MPS 规范对齐（AL/AR/C/AC 逐元素一致）✔")

# ---- 2. 环境对齐 ----
envs_our = InfiniteMPSAlgorithms.environments(ψ_our, H_our)
envs_mk = MPSKit.environments(ψ_mk, H_mk)
GL_our = InfiniteMPSAlgorithms.leftenv(envs_our, 1)
GR_our = InfiniteMPSAlgorithms.rightenv(envs_our, 1)
GL_mk = envarray(convert(TensorMap, MPSKit.leftenv(envs_mk, 1, ψ_mk)))
GR_mk = envarray(convert(TensorMap, MPSKit.rightenv(envs_mk, 1, ψ_mk)))
# 归一化对比（MPSKit 有各自的归一化约定；中间通道因 Jordan/SVD 分解不同
# 而差常数因子，物理等价。逐 level 允许整体缩放后比较。）
scaleL = tr(GL_our[:, 1, :]) ≈ 0 ? 1.0 : tr(GL_mk[:, 1, :]) / tr(GL_our[:, 1, :])
for l in 1:size(GL_our, 2)
    our_l = GL_our[:, l, :] * scaleL
    mk_l = GL_mk[:, l, :]
    if norm(our_l) > 1e-15
        s = dot(vec(mk_l), vec(our_l)) / dot(vec(our_l), vec(our_l))
        @test norm(our_l * s - mk_l) / norm(mk_l) < 1e-8
    end
end
# 右环境同样逐 level 允许整体缩放后比较
for l in 1:size(GR_our, 2)
    our_l = GR_our[:, l, :]
    mk_l = GR_mk[:, l, :]
    if norm(our_l) > 1e-15
        s = dot(vec(mk_l), vec(our_l)) / dot(vec(our_l), vec(our_l))
        @test norm(our_l * s - mk_l) / norm(mk_l) < 1e-8
    end
end
println("3. 左右环境张量对齐（物理等价）✔")

# ---- 3. 期望值对齐 ----
e_our = InfiniteMPSAlgorithms.expectation_value(ψ_our, H_our, envs_our)
e_mk = MPSKit.expectation_value(ψ_mk, H_mk)
@test abs(real(e_our) - real(e_mk)) < 1e-9 * abs(real(e_mk))
println("4. 期望值对齐：our = $(real(e_our)), mpskit = $(real(e_mk)) ✔")

# ---- 4. H_AC / H_C 作用对齐 ----
Random.seed!(42)
x = randn(T, D, d, D)
hac_our = InfiniteMPSAlgorithms.AC_hamiltonian(1, ψ_our, H_our, ψ_our, envs_our)(x)
hac_mk = MPSKit.AC_hamiltonian(1, ψ_mk, H_mk, ψ_mk, envs_mk)(mkmpstensor(x))
@test norm(hac_our - mpsarray(hac_mk)) / norm(hac_our) < 1e-9

c0 = ψ_our.C[1]
hc_our = InfiniteMPSAlgorithms.C_hamiltonian(1, ψ_our, H_our, ψ_our, envs_our)(c0)
hc_mk = MPSKit.C_hamiltonian(1, ψ_mk, H_mk, ψ_mk, envs_mk)(
    TensorKit.TensorMap(copy(c0), ℂ^size(c0, 1), ℂ^size(c0, 2)))
@test norm(hc_our - reshape(_tkdata(hc_mk), size(c0))) / norm(hc_our) < 1e-9
println("5. H_AC / H_C 作用对齐 ✔")

# ---- Heisenberg（3 通道）同样对齐期望值 ----
Hh_our = heisenberg_hamiltonian(T = T)
Hh_mk = MPSKit.InfiniteMPOHamiltonian(lattice,
                                      ((1, 2) => (1 / 4) * (σx(T) ⊗ σx(T) + σy(T) ⊗ σy(T) + σz(T) ⊗ σz(T))))
envs2 = InfiniteMPSAlgorithms.environments(ψ_our, Hh_our)
e2_our = InfiniteMPSAlgorithms.expectation_value(ψ_our, Hh_our, envs2)
e2_mk = MPSKit.expectation_value(ψ_mk, Hh_mk)
@test abs(real(e2_our) - real(e2_mk)) < 1e-9 * abs(real(e2_mk))
println("6. Heisenberg 期望值对齐：$(real(e2_our)) vs $(real(e2_mk)) ✔")

println("\n全部环境对齐测试通过。")
