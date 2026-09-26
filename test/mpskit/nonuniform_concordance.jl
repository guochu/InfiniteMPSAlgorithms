# =====================================================================
# 非均匀键 profile（unit cell > 1，各 bond 键维不同）与 MPSKit 的对比
#
# 覆盖：构造/规范化、SparseIMPO 哈密顿量的期望值与环境、零填充扩键的不变性、
#       VUMPS / IDMRG 从非均匀初态出发的收敛能量。
# 对齐约定：MPSKit 的 InfiniteMPOHamiltonian 在 L 站点胞上的周期闭合键写作
#       `(L, L+1)`（不是 `(L, 1)`）。
# =====================================================================

@testset "非均匀键 profile ≡ MPSKit" begin
    T = ComplexF64
    J, h = 1.0, 1.3

    # ---- 不可行键 profile 的清理：与 MPSKit `InfiniteMPS(A)` 的 makefullrank! 一致 ----
    # （右键过大 / 左键过大两种方向都要走对分支，且周期 trace 即态不变）
    for (Ds, expect) in (([2, 4, 2], [2, 4, 2]), ([2, 5, 2], [2, 4, 2]),
                         ([5, 2, 2], [4, 2, 2]))
        A = [randn(T, Ds[mod1(ℓ - 1, 3)], 2, Ds[ℓ]) for ℓ in 1:3]
        ψr = CanonicalIMPS([copy(a) for a in A])
        ψk = MPSKit.InfiniteMPS([mkmpstensor(copy(a)) for a in A])
        @test [bonddim(ψr, ℓ) for ℓ in 1:3] == expect
        @test [dim(space(ψk.AL[ℓ], 3)) for ℓ in 1:3] == expect
        @test ismixedcanonical(ψr)
    end

    Hs = _tfim3(J = J, h = h, T = T)
    σx_t = σx_tk(T); σz_t = σz_tk(T)
    Hmk = MPSKit.InfiniteMPOHamiltonian(fill(ℂ^2, 3), 1 => -h * σz_t,
                                        (1, 2) => -J * (σx_t ⊗ σx_t),
                                        2 => -h * σz_t, (2, 3) => -J * (σx_t ⊗ σx_t),
                                        3 => -h * σz_t, (3, 4) => -J * (σx_t ⊗ σx_t))

    # ---- 均匀键 D = 4 的基准 ----
    Random.seed!(20260925)
    ψu = randomimps(T, fill(2, 3), 4)
    e_u = real(expectationvalue(ψu, Hs))
    @test abs(e_u - real(MPSKit.expectation_value(to_mpskit(ψu), Hmk))) < 1e-10

    # ---- 最小非均匀键态 χ = [2, 4, 2] ----
    ψnu = _nonuniform_mps([2, 4, 2], 2)
    @test [bonddim(ψnu, ℓ) for ℓ in 1:3] == [2, 4, 2]
    @test ismixedcanonical(ψnu)
    e_nu = real(expectationvalue(ψnu, Hs))
    @test abs(e_nu - real(MPSKit.expectation_value(to_mpskit(ψnu), Hmk))) < 1e-10

    # ---- 零填充扩键（同一物理态，profile 4,4,4 → 4,5,4）：能量逐位不变 ----
    ψp = _padbond!(copy(ψu), 2, 1)
    @test [bonddim(ψp, ℓ) for ℓ in 1:3] == [4, 5, 4]
    @test ismixedcanonical(ψp)
    @test abs(dot(ψp, ψu) / (norm(ψp) * norm(ψu)) - 1) < 1e-12
    @test abs(real(expectationvalue(ψp, Hs)) - e_u) < 1e-10
    @test abs(real(expectationvalue(ψp, Hs)) -
              real(MPSKit.expectation_value(to_mpskit(ψp), Hmk))) < 1e-10

    # ---- 非均匀初态上的基态搜索：与 MPSKit 逐位一致，且保留非均匀 profile ----
    for (ouralg, mkalg) in ((VUMPS(D = 4, maxiter = 100, tol = 1e-10, verbosity = 0),
                             MPSKit.VUMPS(maxiter = 100, tol = 1e-10, verbosity = 0)),
                            (IDMRG(D = 4, maxiter = 100, tol = 1e-10, verbosity = 0),
                             MPSKit.IDMRG(maxiter = 100, tol = 1e-10, verbosity = 0)))
        Random.seed!(11)
        ψ1, e1, _ = find_groundstate(copy(ψnu), Hs, ouralg)
        Random.seed!(11)
        ψ2, _, _ = MPSKit.find_groundstate(to_mpskit(ψnu), Hmk, mkalg)
        @test abs(real(expectationvalue(ψ1, Hs, e1)) -
                  real(MPSKit.expectation_value(ψ2, Hmk))) < 1e-9
        @test [bonddim(ψ1, ℓ) for ℓ in 1:3] == [2, 4, 2]
    end
end

@testset "异构物理维（unit cell 内各站 d 不同）≡ MPSKit" begin
    T = ComplexF64
    dims = [2, 3, 2]
    Random.seed!(3)
    ψ = randomimps(T, dims, 4)
    # 解析参考：on-site-only 模型 H = Σ_ℓ h1_ℓ ⇒ E = Σ_ℓ ⟨h1_ℓ⟩（无环境路径）
    h1 = [Matrix{T}(h + h') for h in (randn(T, dims[ℓ], dims[ℓ]) for ℓ in 1:3)]
    Hs = SparseIMPO([mpohamiltonian(h1[ℓ], Tuple{Float64,Matrix{T},Matrix{T}}[]) for ℓ in 1:3])
    E_ref = sum(real(expectationvalue(ψ, (ℓ,) => h1[ℓ])) for ℓ in 1:3)
    @test abs(real(expectationvalue(ψ, Hs)) - E_ref) < 1e-10

    lattice = [ℂ^dims[1], ℂ^dims[2], ℂ^dims[3]]
    h1_tk = [TensorMap(copy(h1[ℓ]), ℂ^dims[ℓ], ℂ^dims[ℓ]) for ℓ in 1:3]
    Hmk = MPSKit.InfiniteMPOHamiltonian(lattice, 1 => h1_tk[1], 2 => h1_tk[2], 3 => h1_tk[3])
    @test abs(real(expectationvalue(ψ, Hs)) -
              real(MPSKit.expectation_value(to_mpskit(ψ), Hmk))) < 1e-10

    # VUMPS / IDMRG：异构 physical space 下逐位一致
    for (ouralg, mkalg) in ((VUMPS(D = 4, maxiter = 100, tol = 1e-10, verbosity = 0),
                             MPSKit.VUMPS(maxiter = 100, tol = 1e-10, verbosity = 0)),
                            (IDMRG(D = 4, maxiter = 100, tol = 1e-10, verbosity = 0),
                             MPSKit.IDMRG(maxiter = 100, tol = 1e-10, verbosity = 0)))
        Random.seed!(9)
        ψ1, e1, _ = find_groundstate(copy(ψ), Hs, ouralg)
        Random.seed!(9)
        ψ2, _, _ = MPSKit.find_groundstate(to_mpskit(ψ), Hmk, mkalg)
        @test abs(real(expectationvalue(ψ1, Hs, e1)) -
                  real(MPSKit.expectation_value(ψ2, Hmk))) < 1e-9
        @test phydims(ψ1) == dims
    end
end
