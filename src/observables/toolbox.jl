# ---------------- entanglement and spectra (mirrors MPSKit src/algorithms/toolbox.jl) ----------------

"""
    entanglement_spectrum(ψ, loc = 1) -> Vector{Float64}

Entanglement spectrum on bond `loc` (squared Schmidt values, in descending
order), given by the singular values of the center matrix `ψ.C[loc]`.
"""
function entanglement_spectrum(ψ::CanonicalIMPS, loc::Int = 1)
    s = svdvals(ψ.C[loc])
    p = abs2.(s)
    p = p ./ sum(p)
    return sort(p; rev = true)
end

"""
    entropy(ψ, loc = 1; α = 1) -> Real

Entanglement entropy on bond `loc`. `α = 1` gives the von Neumann entropy
(consistent with MPSKit's `entropy`); otherwise the α-order Rényi entropy
(natural logarithm).
"""
function entropy(ψ::CanonicalIMPS, loc::Int = 1; α::Real = 1)
    return renyi_entropy(entanglement_spectrum(ψ, loc); α = α)
end
