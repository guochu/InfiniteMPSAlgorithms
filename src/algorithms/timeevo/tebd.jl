# TEBD: quantum gates (AbstractGate / UnitaryGate / GeneralGate) + apply! + swap!
# Minimal TEBD building blocks; the time-evolution loop is driven by the caller.
#
# 门类型与基础接口（struct 定义、positions/shift/adjoint/scalartype、NTuple
# 构造与 unitarity 检查）由 FiniteMPSAlgorithms 提供并经主文件 import 再导出
# （两侧本就是同一份代码的移植）。本文件只保留本包侧的扩展：
# Pair{Int,Int} 矩阵约定的便捷构造（FMA 无）与 CanonicalIMPS（infinite 链）
# 上的 apply!/swap!（Hastings 更新）。
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
# sweep. With no truncation the post-gate window is reproduced exactly, so the
# physical state is preserved exactly for any gate, and identity gates
# (swaps) are exact lossless re-gaugings.
#
# The seam gauge solves are exact on any mixed-canonical input: `regauge!`
# returns the procrustes-optimal left/right-orthogonal solution, and the
# canonical identity network is reproduced at machine precision — verified
# over random gate sequences (including the wrapping bond) directly on
# states that never went through any special initialization.

"""
	UnitaryGate(positions::Pair{Int,Int}, op::AbstractMatrix; atol=1e-10)

本包对 FMA [`UnitaryGate`](@ref) 的便捷构造扩展：`d²×d²` 矩阵按 Kronecker
约定 `(i1 i2)', (i1 i2)`（site i1 为慢指标），permute 成张量约定
`(i1', i2', i1, i2)` 后委托 FMA 的 NTuple 构造（含 unitarity 检查）。
**该形式要求两侧物理维相同**（按 `d = isqrt` 拆开）；unit cell 内各站物理维
不同时请直接用 rank-4 张量形式 `UnitaryGate((i, j), t)`（形状
`(d_i, d_j, d_i, d_j)`）。
"""
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
	GeneralGate(positions::Pair{Int,Int}, op::AbstractMatrix)

本包对 FMA [`GeneralGate`](@ref) 的便捷构造扩展：`d²×d²` Kronecker 约定矩阵
（site i1 为慢指标）permute 成张量约定后委托 FMA 的 NTuple 构造（不做
unitarity 检查）。物理维不同的站点对请直接用 rank-4 张量形式。
"""
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
function _hastings_update!(ψ::CanonicalIMPS, gated::Array, i::Integer;
                           trunc::TruncationScheme = alg_orth_trunc())
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

function _nn_gate_apply!(g::AbstractGate{2}, ψ::CanonicalIMPS, i::Integer;
                         trunc::TruncationScheme = alg_orth_trunc())
	@tensor gated[a, p′, q′, b] := ψ.AC[i][a, p, c] * g.op[p′, q′, p, q] * ψ.AR[i + 1][c, q, b]
	return _hastings_update!(ψ, gated, i; trunc)
end

"""
	apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS;
	       trunc = alg_orth_trunc()) -> ψ

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
(the default `alg_orth_trunc()` only cleans numerically-zero spectral directions;
`NoTruncation()` reproduces the post-gate window bit-for-bit) and the canonical form
is preserved at machine precision — the seams are exactly orthogonal (`regauge!`
gauge conversion) and the canonical identity `AC = C·AR` holds at all bonds
(verified over random gate sequences, including the wrapping bond). The bond
dimension may grow up to `min(Dl·d, d·Dr)` on the gate bond before truncation.
"""
function apply!(g::UnitaryGate{2}, ψ::CanonicalIMPS;
                trunc::TruncationScheme = alg_orth_trunc())
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
	apply!(g::GeneralGate{2}, ψ::CanonicalIMPS;
	       trunc = alg_orth_trunc(), kwargs...) -> ψ

Apply a two-site gate (possibly non-unitary) to `ψ` with the Hastings update, identical
to the [`UnitaryGate`](@ref) case; the two gate sites may be any ascending pair
(non-adjacent gates are moved next to each other with exact unitary content swaps,
applied, and moved back). Because a non-unitary gate does not preserve the canonical
form, the state is re-canonicalized with `gaugefix!` afterwards (`kwargs...` are
forwarded). Initializes the canonical form if needed.
"""
function apply!(g::GeneralGate{2}, ψ::CanonicalIMPS;
                trunc::TruncationScheme = alg_orth_trunc(), kwargs...)
	i, j = g.positions
	L = length(ψ)
	# positions may run one past the cell (bond wrapping, as in apply!(::UnitaryGate))
	(1 <= i < j <= L + 1) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	# re-canonicalize immediately after the non-unitary gate, so that the return
	# swaps act on a canonical state (the Hastings update assumes mixed-canonical
	# input; acting on the gauge-broken two-site tensor would accumulate error)
	gaugefix!(ψ, parent(ψ.AR); order = :RL, kwargs...)
	for b in i+1:j-1
		swap!(ψ, b; trunc)
	end
	return ψ
end

"""
	swap!(ψ::CanonicalIMPS, i::Integer;
	      trunc = alg_orth_trunc()) -> ψ

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
function swap!(ψ::CanonicalIMPS, i::Integer;
                trunc::TruncationScheme = alg_orth_trunc())
	(1 <= i <= length(ψ)) || throw(BoundsError())
	@tensor gated[a, q, p, b] := ψ.AC[i][a, p, c] * ψ.AR[i + 1][c, q, b]   # legs crossed
	_hastings_update!(ψ, gated, i; trunc)
	return ψ
end
