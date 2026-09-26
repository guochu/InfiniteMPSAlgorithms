# TEBD: quantum gates (AbstractGate / UnitaryGate / GeneralGate) + apply! + swap!
# Minimal TEBD building blocks; the time-evolution loop is driven by the caller.
#
# The Hastings update (aligned with TEMPO/GTEMPO) acts on the two-site window
# AC[i]·AR[i+1] (exact local representation, unit left environment). A single
# truncated SVD of the post-gate window,
#     AC[i]·G·AR[i+1] = U·S·V†,
# yields the three Hastings slots without ever dividing the spectrum:
#     AL[i]   ← U      (exactly left-orthogonal),
#     AR[i+1] ← V†     (exactly right-orthogonal),
#     C[i]    ← S      (the whole spectrum goes to the gate bond),
# and the remaining tensors follow from the canonical identities
#     AC[i] = AL[i]·C[i],  AC[i+1] = C[i]·AR[i+1]   (exact by construction),
#     AR[i] from C[i-1]·AR[i] = AC[i],  AL[i+1] from AC[i+1] = AL[i+1]·C[i+1]
# (a gauge conversion of the SVD factors onto the neighbouring bonds' gauges;
# exact for identity gates, O(gate) orthogonality error otherwise, as in
# TEMPO/GTEMPO). These solves are what keeps the mixed-canonical identity
# network AC = AL·C = C·AR exact without any global re-canonicalization
# sweep. With no truncation the post-gate window is reproduced exactly, so
# the physical state is preserved exactly for any gate, and identity gates
# (swaps) are exact lossless re-gaugings.
#
# The seam gauge solves are exact on any mixed-canonical input: `regauge!`
# returns the procrustes-optimal left/right-orthogonal solution, and the
# canonical identity network is reproduced at machine precision — verified
# over random gate sequences (including the wrapping bond) directly on
# states that never went through any special initialization.

"""
	AbstractGate{N,T}

Abstract supertype of quantum gates acting on `N` sites: `positions(g)::NTuple{N,Int}`
(ascending), `g.op::Array{T,M}` — the rank-`M = 2N` operator tensor in the index
convention `(i1', i2', …, iN', i1, …, iN)`, i.e. all bra (output) indices first and all
ket (input) indices second, each block ordered with site 1 the slowest index.
"""
abstract type AbstractGate{N, T} end

positions(g::AbstractGate) = g.positions
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
  the tensor convention on construction. **该形式要求两侧物理维相同**（把矩阵按
  `d = isqrt` 拆开）；unit cell 内各站物理维不同时请用上面的 rank-4 张量形式
  （形状 `(d_i, d_j, d_i, d_j)`）。

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

# Hastings gate core (aligned with TEMPO/GTEMPO). The two-site window
# AC[i]·G·AR[i+1] is decomposed with a single truncated SVD,
#     AC[i]·G·AR[i+1] = U·S·V†,
# and the three Hastings slots are written without ever dividing the spectrum:
#     AL[i]   ← U      (exactly left-orthogonal),
#     AR[i+1] ← V†     (exactly right-orthogonal),
#     C[i]    ← S      (the whole spectrum goes to the gate bond).
# The center tensors follow from the AR-side definition AC = C·AR, and the two
# seam tensors are obtained by solving the *other* canonical identities
#     AR[i]  from C[i-1]·AR[i] = AC[i],      AL[i+1] from AC[i+1] = AL[i+1]·C[i+1],
# i.e. a gauge conversion of the SVD factors onto the neighbouring bonds' gauges.
# These solves are what preserves the mixed-canonical identity network
# AC = AL·C = C·AR exactly (no global re-canonicalization sweep). With no
# truncation the window AC[i]·AR[i+1]·G is reproduced exactly, so the physical
# state is preserved exactly for any gate, and identity gates (swaps) are
# exact lossless re-gaugings.
function _hastings_update!(ψ::CanonicalIMPS, gated::Array, i::Integer; trunc::TruncationScheme=NoTruncation())
	j = i + 1
	gmat = reshape(gated, (size(gated, 1) * size(gated, 2), size(gated, 3) * size(gated, 4)))
	u, s, v, err = tsvd(gmat; trunc)
	k = length(s)
	ALi = reshape(u, size(gated, 1), size(gated, 2), k)             # exactly left-orthogonal
	ARj = reshape(v, k, size(gated, 3), size(gated, 4))             # exactly right-orthogonal
	Ci = Matrix(Diagonal(s))
	ACi = reshape(reshape(ALi, :, k) * Ci, size(gated, 1), size(gated, 2), k)              # AL·C
	ACj = reshape(Ci * reshape(ARj, k, :), k, size(gated, 3), size(gated, 4))              # C·AR
	# seam tensors: orthogonal gauge conversion of the SVD factors onto the
	# neighbouring bonds' gauges (`regauge!` = the procrustes-optimal
	# left/right-orthogonal solution; no division of the bond matrices).
	# The seams are always exactly orthogonal, and on any mixed-canonical
	# input the canonical identity network is reproduced at machine
	# precision (verified over random gate sequences).
	ALj = regauge!(ACj, ψ.C[j])
	ARi = regauge!(ψ.C[i - 1], ACi)
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
	@tensor gated[a, p′, q′, b] := ψ.AC[i][a, p, c] * g.op[p′, q′, p, q] * ψ.AR[i + 1][c, q, b]
	return _hastings_update!(ψ, gated, i; trunc)
end

"""
	apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS; trunc=NoTruncation()) -> ψ

Apply a two-site unitary gate to `ψ` with the Hastings update (aligned with
TEMPO/GTEMPO). The two gate sites may be any ascending pair `(i, j)` — with `j`
possibly `L + 1` for the wrapping bond `(L, 1)`: non-adjacent gates are first
moved next to each other with exact unitary content swaps, applied at the
neighboring bond, and moved back. The post-gate window `AC[i]·G·AR[i+1]` is
SVD-decomposed; the left factor is written directly to `AL[i]` (exactly
left-orthogonal), the right factor to `AR[i+1]` (exactly right-orthogonal), the
new spectrum to `C[i]` (never divided), and the center/seam tensors follow from
the canonical identities (no global re-canonicalization sweep).

No initialization is needed: on any mixed-canonical input the gate acts exactly
(`NoTruncation` reproduces the post-gate window exactly) and the canonical form
is preserved at machine precision — the seams are exactly orthogonal (`regauge!`
gauge conversion) and the canonical identity `AC = C·AR` holds at all bonds
(verified over random gate sequences, including the wrapping bond). The bond
dimension may grow up to `min(Dl·d, d·Dr)` on the gate bond before truncation.
"""
function apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS; trunc::TruncationScheme=NoTruncation())
	i, j = g.positions
	L = length(ψ)
	# positions may run one past the cell (bond wrapping: (L, L+1) ≡ the
	# bond between sites L and 1)
	(1 <= i < j <= L + 1) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		swap!(ψ, b; trunc)
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
	# positions may run one past the cell (bond wrapping, as in apply!(::UnitaryGate))
	(1 <= i < j <= L + 1) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		swap!(ψ, b; trunc)
	end
	gaugefix!(ψ, parent(ψ.AR); order = :RL, kwargs...)
	return ψ
end

"""
	swap!(ψ::CanonicalIMPS, i::Integer; trunc=NoTruncation()) -> ψ

Exchange the physical content of the neighboring sites `i` and `i+1` of `ψ`
with the Hastings SWAP gate: the bare block `AR[i]·AR[i+1]` is formed with the
physical legs crossed and folded with the left bond spectrum `C[i-1]`; one SVD
yields the right factor (written directly to `AR[i+1]`, exactly
right-orthogonal), the renewed spectrum (written directly to `C[i]`, never
divided) and the back-projected left site. The center tensors follow from the
AR-side definition `AC = C·AR`. With no truncation this is an exact lossless
unitary re-gauging on any mixed-canonical input: the physical state is
preserved exactly and the canonical form at machine precision — no special
initialization required.
"""
function swap!(ψ::CanonicalIMPS, i::Integer; trunc::TruncationScheme=NoTruncation())
	(1 <= i <= length(ψ)) || throw(BoundsError())
	@tensor gated[a, q, p, b] := ψ.AC[i][a, p, c] * ψ.AR[i + 1][c, q, b]   # legs crossed
	_hastings_update!(ψ, gated, i; trunc)
	return ψ
end

