# ---------------- correlation functions (mirrors MPSKit src/algorithms/correlators.jl) ----------------

"""
    correlator(ψ, O1, O2, i, j) -> Number
    correlator(ψ, O1, O2, i, js::AbstractRange{Int}) -> Vector{Number}

Two-point correlator `⟨ψ|O1[i]·O2[j]|ψ⟩` (non-connected, consistent with
MPSKit). Translation-invariant systems require `j > i`; `js` provides a series
of `j`s (with incremental propagation).
"""
function correlator end

function correlator(ψ::CanonicalIMPS, O1::AbstractMatrix, O2::AbstractMatrix,
                    i::Int, j::Int)
    return first(correlator(ψ, O1, O2, i, j:j))
end

function correlator(ψ::CanonicalIMPS, O1::AbstractMatrix, O2::AbstractMatrix,
                    i::Int, js::AbstractRange{Int})
    N = length(ψ)
    (first(js) > i) || throw(ArgumentError("i should be smaller than j ($i, $(first(js)))"))
    # insert O1 at site i: V[(bra right bond, ket right bond)]
    AC = ψ.AC[i]
    V = @tensor V0[b̄, β] := conj(AC[a, ū, b̄]) * O1[ū, d] * AC[a, d, β]
    G = similar(collect(js), promote_type(scalartype(ψ), eltype(O1), eltype(O2)))
    ctr = i
    for (k, j) in enumerate(js)
        (j > ctr) || (k > 1 && continue)
        # propagate to site j (identity channel; V is in (bra, ket) order)
        for ℓ in (ctr+1):(j-1)
            A = ψ.AR[_mod1(ℓ, N)]
            V = @tensor V′[b̄′, β′] := conj(A[b̄, s, b̄′]) * A[β̄, s, β′] * V[b̄, β̄]
        end
        jj = _mod1(j, N)
        AR = ψ.AR[jj]
        @tensor E[] := V[b̄, β] * conj(AR[b̄, ū, c̄]) * O2[ū, d] * AR[β, d, c̄]
        G[k] = E[]
        ctr = j
    end
    return G ./ dot(ψ, ψ)
end
