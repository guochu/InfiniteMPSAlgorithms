# TEBD: quantum gates (AbstractGate / UnitaryGate / GeneralGate) + apply! + swap!
# Minimal TEBD building blocks; the time-evolution loop is driven by the caller.
#
# The Hastings update is adapted to the mixed-canonical AL/C/AR representation:
# the two-site window `AC[i]·AR[i+1]` (exact local representation, unit left
# environment) is decomposed with a single truncated SVD
#     gated = U·S·V†,
# after which
#     AL[i] ← U        (left-orthogonal by construction),
#     AR[i+1] ← V†     (right-orthogonal by construction),
#     C[i] ← S         (all Schmidt weight goes to the gate bond — the SVD
#                       spectrum is never divided),
#     AC[i] ← AL[i]·C[i],  AC[i+1] ← C[i]·AR[i+1]   (exact by construction),
# and the two seam tensors follow from the *old* canonical identities
#     AR[i] ← C[i-1] \ AC[i],   AL[i+1] ← AC[i+1] / C[i+1],
# which are exact (no bond re-canonicalization sweep, no global gauge fix).
# With no truncation the window `AC[i]·AR[i+1]` is preserved exactly, so the
# physical state is preserved exactly for any gate; `swap!` (an identity gate)
# additionally preserves the canonical form exactly.

"""
	AbstractGate{N,T}

Abstract supertype of quantum gates acting on `N` sites: `positions(g)::NTuple{N,Int}`
(ascending), `operator(g)::Array{T,M}` — the rank-`M = 2N` operator tensor in the index
convention `(i1', i2', …, iN', i1, …, iN)`, i.e. all bra (output) indices first and all
ket (input) indices second, each block ordered with site 1 the slowest index.
"""
abstract type AbstractGate{N, T} end

positions(g::AbstractGate) = g.positions
operator(g::AbstractGate) = g.op
scalartype(::Type{<:AbstractGate{N, T}}) where {N, T} = T

"""
	shift(g, by)

Shift all support positions of gate `g` by `by` sites.
"""
shift(g::G, by::Integer) where {G<:AbstractGate} =
	typeof(g)(ntuple(i -> g.positions[i] + by, Val(length(g.positions))), g.op)

function Base.adjoint(g::G) where {G<:AbstractGate}
	N = length(g.positions)
	# U† : swap the ket/bra index blocks and conjugate; support positions unchanged
	perm = (ntuple(i -> N + i, Val(N))..., ntuple(i -> i, Val(N))...)
	return typeof(g)(g.positions, permutedims(conj(g.op), perm))
end

_isunitary(m::AbstractMatrix; atol::Real) =
	norm(m' * m - I(size(m, 1))) <= atol * max(1.0, size(m, 1))

"""
	UnitaryGate(positions, op; atol=1e-10)
	UnitaryGate(positions::Pair{Int,Int}, op::AbstractMatrix; atol=1e-10)

A unitary gate acting on the `N` ascending sites `positions`. The operator `op` may be
given in either of two index conventions:

* a rank-`2N` tensor `(i1', i2', …, iN', i1, …, iN)` — all bra (output) indices first,
  all ket (input) indices second, each block ordered with site 1 the slowest index;
* for `positions::Pair{Int,Int}` (`N = 2`), a `d²×d²` matrix in the Kronecker
  convention `(i1 i2)', (i1 i2)` with site `i1` the slower index; it is permuted into
  the tensor convention on construction.

The input is copied and materialized as a dense array of concrete element type, then
checked for unitarity (`op'*op ≈ I` within `atol`), throwing `ArgumentError` otherwise.
"""
struct UnitaryGate{N, T, M} <: AbstractGate{N, T}
	positions::NTuple{N, Int}
	op::Array{T, M}   # (i1', …, iN', i1, …, iN) with M == 2N (checked at construction)

	function UnitaryGate{N, T, M}(positions::NTuple{N, Int}, op::Array{<:Any, M}; atol::Real=1.0e-10) where {N, T, M}
		M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
		issorted(collect(positions)) || throw(ArgumentError("positions must be ascending"))
		op = convert(Array{T, M}, op)
		m = tie(op, (N, N))
		_isunitary(m; atol) || throw(ArgumentError("the gate operator is not unitary"))
		return new{N, T, M}(positions, op)
	end
end

function UnitaryGate(positions::NTuple{N, Int}, op::AbstractArray; atol::Real=1.0e-10) where {N}
	M = ndims(op)
	M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
	T = scalartype(op)
	return UnitaryGate{N, T, M}(positions, Array{T, M}(op); atol)
end
function UnitaryGate(positions::Pair{Int, Int}, op::AbstractMatrix; atol::Real=1.0e-10)
	d2 = size(op, 1)
	d = isqrt(d2)
	d^2 == d2 || throw(ArgumentError("operator dimension must be a perfect square"))
	positions.first < positions.second || throw(ArgumentError("positions must be ascending"))
	# matrix convention (i1 i2)',(i1 i2) with i1 the slowest index; the column-major
	# reshape yields (i2', i1', i2, i1), so permute to the documented (i1', i2', i1, i2)
	t = reshape(Matrix{scalartype(op)}(op), d, d, d, d)
	t = permutedims(t, (2, 1, 4, 3))
	return UnitaryGate((positions.first, positions.second), t; atol)
end

"""
	GeneralGate(positions, op)
	GeneralGate(positions::Pair{Int,Int}, op::AbstractMatrix)

A general gate with the same data storage and index conventions as
[`UnitaryGate`](@ref) (rank-`2N` tensor `(i1', …, iN', i1, …, iN)`, or for
`positions::Pair{Int,Int}` a `d²×d²` Kronecker-convention matrix `(i1 i2)', (i1 i2)`
with i1 the slowest index), but the input is **not** checked for unitarity. The input
is copied and materialized as a dense array of concrete element type. Since a
non-unitary gate does not preserve the canonical form, `apply!` re-canonicalizes the
state with `gaugefix!` afterwards.
"""
struct GeneralGate{N, T, M} <: AbstractGate{N, T}
	positions::NTuple{N, Int}
	op::Array{T, M}   # (i1', …, iN', i1, …, iN) with M == 2N (checked at construction)

	function GeneralGate{N, T, M}(positions::NTuple{N, Int}, op::Array{<:Any, M}) where {N, T, M}
		M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
		issorted(collect(positions)) || throw(ArgumentError("positions must be ascending"))
		return new{N, T, M}(positions, convert(Array{T, M}, op))
	end
end

function GeneralGate(positions::NTuple{N, Int}, op::AbstractArray) where {N}
	M = ndims(op)
	M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
	T = scalartype(op)
	return GeneralGate{N, T, M}(positions, Array{T, M}(op))
end
function GeneralGate(positions::Pair{Int, Int}, op::AbstractMatrix)
	d2 = size(op, 1)
	d = isqrt(d2)
	d^2 == d2 || throw(ArgumentError("operator dimension must be a perfect square"))
	positions.first < positions.second || throw(ArgumentError("positions must be ascending"))
	# matrix convention (i1 i2)',(i1 i2) with i1 the slowest index; the column-major
	# reshape yields (i2', i1', i2, i1), so permute to the documented (i1', i2', i1, i2)
	t = reshape(Matrix{scalartype(op)}(op), d, d, d, d)
	t = permutedims(t, (2, 1, 4, 3))
	return GeneralGate((positions.first, positions.second), t)
end

# nearest-neighbor Hastings gate core (shared by UnitaryGate, GeneralGate and the
# identity-gauge `swap!`): the two-site window `AC[i]·AR[i+1]` (unit left environment)
# is SVD-decomposed; the left factor becomes the new `AL[i]`, the right factor the new
# `AR[i+1]`, the spectrum the new `C[i]` (never divided), and the center / seam tensors
# are restored from the canonical identities. With no truncation the window is
# preserved exactly, so the physical state is preserved exactly for any gate.
function _hastings_update!(ψ::CanonicalIMPS, Θ::Array, i::Integer; trunc::TruncationScheme=NoTruncation())
	j = i + 1
	gmat = reshape(Θ, (size(Θ, 1) * size(Θ, 2), size(Θ, 3) * size(Θ, 4)))
	u, s, v, err = tsvd(gmat; trunc)
	k = length(s)
	ALi = reshape(u, size(Θ, 1), size(Θ, 2), k)             # left-orthogonal by construction
	ARj = reshape(v, k, size(Θ, 3), size(Θ, 4))             # right-orthogonal by construction
	Ci = Matrix(Diagonal(s))
	ACi = reshape(reshape(ALi, :, k) * Ci, size(Θ, 1), size(Θ, 2), k)              # AL·C
	ACj = reshape(Ci * reshape(ARj, k, :), k, size(Θ, 3), size(Θ, 4))              # C·AR
	# seam tensors from the old canonical identities (exact; O(trunc err) when truncated)
	ALj = reshape(reshape(ACj, k * size(Θ, 3), :) / ψ.C[j], k, size(Θ, 3), size(ψ.C[j], 2))
	ARi = reshape(ψ.C[i - 1] \ reshape(ACi, size(Θ, 1), :), size(Θ, 1), size(Θ, 2), k)
	ψ.AL[i] = ALi
	ψ.AR[j] = ARj
	ψ.C[i] = Ci
	ψ.AC[i] = ACi
	ψ.AC[j] = ACj
	ψ.AL[j] = ALj
	ψ.AR[i] = ARi
	return err
end

function _nn_gate_apply!(g::AbstractGate{2}, ψ::CanonicalIMPS, i::Integer; trunc::TruncationScheme=NoTruncation())
	@tensor Θ[a, p, q, b] := ψ.AC[i][a, p, c] * ψ.AR[i + 1][c, q, b]
	@tensor gated[a, p′, q′, b] := Θ[a, p, q, b] * operator(g)[p′, q′, p, q]
	return _hastings_update!(ψ, gated, i; trunc)
end

# Hastings content swap of the neighboring sites `i`, `i+1`: the two-site window is
# formed with the physical legs crossed (a SWAP gate), so the site contents exchange
# while the canonical form is preserved (the SWAP gate is unitary).
function _swap_content!(ψ::CanonicalIMPS, i::Integer; trunc::TruncationScheme=NoTruncation())
	@tensor Θ[a, q, p, b] := ψ.AC[i][a, p, c] * ψ.AR[i + 1][c, q, b]
	return _hastings_update!(ψ, Θ, i; trunc)
end

"""
	apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS; trunc=NoTruncation()) -> ψ

Apply a two-site unitary gate to `ψ` with the Hastings update adapted to the
mixed-canonical AL/C/AR representation. The two gate sites may be any ascending pair
`(i, j)`: non-adjacent gates are first moved next to each other with exact unitary
content swaps, applied at the neighboring bond, and moved back. The two-site window
`AC[i]·AR[i+1]` (unit left environment) is SVD-decomposed; the right factor is written
directly to `AR[i+1]`, the new spectrum to `C[i]` (never divided), and the center and
seam tensors are restored from the canonical identities. With no truncation the
physical state is preserved exactly and the canonical form is preserved by `swap!`;
seam orthogonality degrades by `O(gate)` and can be restored with a (local or global)
re-canonicalization if needed. The bond dimension may grow up to
`min(Dl·d, d·Dr)` on the gate bond before truncation.
"""
function apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS; trunc::TruncationScheme=NoTruncation())
	i, j = g.positions
	L = length(ψ)
	(1 <= i < j <= L) || throw(BoundsError())
	for b in j-1:-1:i+1
		_swap_content!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		_swap_content!(ψ, b; trunc)
	end
	return ψ
end

"""
	apply!(g::GeneralGate{2}, ψ::CanonicalIMPS; trunc=NoTruncation(), kwargs...) -> ψ

Apply a two-site gate (possibly non-unitary) to `ψ` with the Hastings update, identical
to the [`UnitaryGate`](@ref) case; the two gate sites may be any ascending pair
(non-adjacent gates are moved next to each other with exact unitary content swaps,
applied, and moved back). Because a non-unitary gate does not preserve the canonical
form, the state is re-canonicalized with `gaugefix!` afterwards (`kwargs...` are
forwarded). Initializes the canonical form if needed.
"""
function apply!(g::GeneralGate{2}, ψ::CanonicalIMPS; trunc::TruncationScheme=NoTruncation(), kwargs...)
	i, j = g.positions
	L = length(ψ)
	(1 <= i < j <= L) || throw(BoundsError())
	for b in j-1:-1:i+1
		_swap_content!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		_swap_content!(ψ, b; trunc)
	end
	gaugefix!(ψ, parent(ψ.AR); order = :RL, kwargs...)
	return ψ
end

"""
	swap!(ψ::CanonicalIMPS, i::Integer; trunc=NoTruncation()) -> ψ

Re-gauge the bond between the neighboring sites `i` and `i+1` of `ψ` with the Hastings
identity-gate update: the two-site window `AC[i]·AR[i+1]` (physical legs uncrossed) is
re-decomposed with one SVD; the right factor is written directly to `AR[i+1]`, the
renewed spectrum to `C[i]` (never divided), and the center and seam tensors are
restored from the canonical identities. The physical state is unchanged; with no
truncation the canonical form is preserved exactly (an exact unitary re-gauging).
Sequences of `swap!` are thus lossless re-gaugings.
"""
function swap!(ψ::CanonicalIMPS, i::Integer; trunc::TruncationScheme=NoTruncation())
	(1 <= i <= length(ψ)) || throw(BoundsError())
	@tensor Θ[a, p, q, b] := ψ.AC[i][a, p, c] * ψ.AR[i + 1][c, q, b]
	_hastings_update!(ψ, Θ, i; trunc)
	return ψ
end

"""
	swap!(ψ::CanonicalIMPS, i::Integer, j::Integer; trunc=NoTruncation()) -> ψ

Exchange the physical content of the two (generally non-adjacent) sites `i` and `j` of
`ψ`; the physical dimensions of the two sites may differ, in which case the
physical-dimension list of `ψ` changes accordingly. The exchange is performed as a
sequence of adjacent content swaps (the Hastings SWAP gate: the two-site window with
the physical legs crossed), each an exact unitary re-gauging, so the canonical form is
preserved when the truncation error is small.
"""
function swap!(ψ::CanonicalIMPS, i::Integer, j::Integer; trunc::TruncationScheme=NoTruncation())
	L = length(ψ)
	(1 <= i <= L && 1 <= j <= L) || throw(BoundsError())
	i == j && return ψ
	lo, hi = minmax(i, j)
	for b in hi-1:-1:lo        # bubble the content of the higher site down to lo
		_swap_content!(ψ, b; trunc)
	end
	for b in lo+1:hi-1         # bubble the original lo content up to hi
		_swap_content!(ψ, b; trunc)
	end
	return ψ
end
