# TEBD: quantum gates (AbstractGate / Gate) + apply! + swap!
# Minimal TEBD building blocks; the time-evolution loop is driven by the caller.
#
# 门类型与基础接口（AbstractGate/Gate 的 struct 定义、positions/shift/
# adjoint、NTuple/矩阵构造与内部排序）由 FiniteMPSAlgorithms 提供
# 并经主文件 import 再导出（FMA 的统一门接口：单一 Gate 类型取代旧的
# UnitaryGate/GeneralGate 双类型；GateOp 已更名为 Gate）。本文件只保留本包
# 侧的扩展：CanonicalIMPS（infinite 链）上的 apply!（单站门 + 两站门 +
# canonicalize 重正则化）与 swap!（Hastings 更新）——门操作支持单比特与
# 两比特门（AbstractGate{1}/AbstractGate{2}），FMA 的 N ≥ 3 block sweep 不在
# infinite 侧实现。
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

# Single-site gate core. The gate acts on the center tensor of site `i`,
#     AC[i] ← G·AC[i],
# and the bond spectra are untouched: a unitary gate preserves the Schmidt
# values of both adjacent bonds, and the neighboring tensors follow from the
# canonical identities AC = AL·C = C·AR with a `regauge!` gauge conversion
#     AL[i] from AL[i]·C[i] = AC[i],   AR[i] from C[i-1]·AR[i] = AC[i]
# (exactly the new left/right isometries G·AL[i] and G·AR[i] for unitary G).
# A non-unitary gate changes the spectra; `apply!` re-canonicalizes by default
# (`canonicalize = !g.isunitary`), matching FMA's semantics.
function _single_site_gate_apply!(g::AbstractGate{1}, ψ::CanonicalIMPS, i::Integer)
	@tensor ACi[a, p′, c] := g.op[p′, p] * ψ.AC[i][a, p, c]
	ψ.AC[i] = ACi
	ψ.AL[i] = regauge!(ACi, ψ.C[i])
	ψ.AR[i] = regauge!(ψ.C[i - 1], ACi)
	return ψ
end

"""
	apply!(g::AbstractGate{2}, ψ::CanonicalIMPS;
	       trunc = alg_orth_trunc(), canonicalize = !g.isunitary) -> ψ

Apply a two-site gate to `ψ` with the Hastings update (aligned with
TEMPO/GTEMPO; FMA 的统一门接口：门一律为 [`Gate`](@ref)，unitarity 读取
`g.isunitary` 字段——FMA 已删除导出的 `isunitary(g)`；本方法不再区分
幺正/非幺正类型分派). The
support must satisfy `1 <= i <= L` and `j - i < L` (`j > i` is automatic from
the ascending [`Gate`](@ref) support); `j = L + 1` is the wrapping bond `(L, 1)`.
Non-adjacent gates are first moved next to each other with exact unitary
content swaps, applied at the neighboring bond, and moved back. The post-gate
window `AC[i]·G·AR[i+1]` is SVD-decomposed; the left
factor is written directly to `AL[i]` (exactly left-orthogonal for unitary
gates), the right factor to `AR[i+1]` (exactly right-orthogonal for unitary
gates), the new spectrum to `C[i]` (never divided), and the center/seam
tensors follow from the canonical identities (no global re-canonicalization
sweep).

A non-unitary gate breaks the exact orthogonality of the SVD factors (the
`gauge` property of the Hastings update), so `canonicalize` defaults to
`!g.isunitary` (the FMA interface): a non-unitary gate re-canonicalizes the
final state with `truncate!(ψ; trunc)`（逐键谱截断 + 重正则化的 infinite 版，
对齐 FMA finite `apply!(...; canonicalize = !g.isunitary)` 的语义）; pass
`canonicalize = false` to skip it.

No initialization is needed: on any mixed-canonical input a unitary gate acts
exactly (the default `alg_orth_trunc()` only cleans numerically-zero spectral
directions; `NoTruncation()` reproduces the post-gate window bit-for-bit) and
the canonical form is preserved at machine precision — the seams are exactly
orthogonal (`regauge!` gauge conversion) and the canonical identity
`AC = C·AR` holds at all bonds (verified over random gate sequences, including
the wrapping bond). The bond dimension may grow up to `min(Dl·d, d·Dr)` on the
gate bond before truncation.
"""
function apply!(g::AbstractGate{2}, ψ::CanonicalIMPS;
                trunc::TruncationScheme = alg_orth_trunc(),
                canonicalize::Bool = !g.isunitary)
	i, j = g.positions
	L = length(ψ)
	# support bounds: 1 <= i <= L and j - i < L (j > i is automatic from the
	# ascending Gate support). j == L + 1 is the wrapping bond (L, 1).
	(1 <= i <= L && j - i < L) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	if canonicalize
		# non-unitary gate: re-canonicalize immediately (before the return
		# swaps, so they act on a canonical state — the Hastings update
		# assumes mixed-canonical input), mirroring FMA's
		# `apply!(...; canonicalize = !g.isunitary)` → `truncate!(ψ; trunc)`
		truncate!(ψ; trunc)
	end
	for b in i+1:j-1
		swap!(ψ, b; trunc)
	end
	return ψ
end

"""
	apply!(g::AbstractGate{1}, ψ::CanonicalIMPS;
	       trunc = alg_orth_trunc(), canonicalize = !g.isunitary) -> ψ

Apply a single-site gate to `ψ` in place. The operator acts on the center
tensor `AC[i]` (`i = g.positions[1]`, with `1 <= i <= L`), and the bond
spectra `C[i-1]`, `C[i]` are kept: a unitary gate
does not change the Schmidt values, and the neighboring tensors follow from
the canonical identities `AC = AL·C = C·AR` via a `regauge!` gauge conversion
(`AL[i] ← G·AL[i]`, `AR[i] ← G·AR[i]` for unitary `G`). No swaps are needed
and the canonical form is preserved at machine precision.

A non-unitary gate changes the spectra, so `canonicalize` defaults to
`!g.isunitary` (the FMA interface): the final state is re-canonicalized with
`truncate!(ψ; trunc)`; pass `canonicalize = false` to skip it.
"""
function apply!(g::AbstractGate{1}, ψ::CanonicalIMPS;
                trunc::TruncationScheme = alg_orth_trunc(),
                canonicalize::Bool = !g.isunitary)
	i = g.positions[1]
	(1 <= i <= length(ψ)) || throw(BoundsError())
	_single_site_gate_apply!(g, ψ, i)
	canonicalize && truncate!(ψ; trunc)
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
