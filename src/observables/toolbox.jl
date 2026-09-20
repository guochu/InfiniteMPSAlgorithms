# ---------------- 纠缠与谱（对标 MPSKit src/algorithms/toolbox.jl） ----------------

"""
    entanglement_spectrum(ψ, loc = 1) -> Vector{Float64}

bond `loc` 的纠缠谱（Schmidt 值平方，降序），由中心矩阵 `ψ.C[loc]` 的奇异值给出。
"""
function entanglement_spectrum(ψ::InfiniteCanonicalMPS, loc::Int = 1)
    s = svdvals(ψ.C[loc])
    p = abs2.(s)
    p = p ./ sum(p)
    return sort(p; rev = true)
end

"""
    entropy(ψ, loc = 1; α = 1) -> Real

bond `loc` 的纠缠熵。`α = 1` 为 von Neumann 熵（与 MPSKit 的 `entropy` 一致），
否则为 α 阶 Rényi 熵（自然对数）。
"""
function entropy(ψ::InfiniteCanonicalMPS, loc::Int = 1; α::Real = 1)
    return renyi_entropy(entanglement_spectrum(ψ, loc); α = α)
end
