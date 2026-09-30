# ---------------- 压缩/代数通道的环境缓存抽象 ----------------
#
# VOMPS/IDMRG 压缩引擎（mult / hadamard / compress 共用）的环境缓存层次：
# OverlapCache（恒等通道）、MultCache（mpo·mps 施加 / mpo·mpo 组合通道）、
# HadamardCache（zip 通道）——与哈密顿量通道的 DMRGCache
# （groundstates/envs.jl）相区分。

abstract type CompressionEnvironments <: Environments end
