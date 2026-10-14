# 内存压力基准（memory pressure baseline）

全通道压缩引擎的内存压力基准：{compress, mult·mps, mult·mpo, hadamard} ×
{IDMRG1, IDMRG2, VOMPS} × D = {25, 50, 100}，unit cell = 2、d = 2。
**每次影响收缩管线 / 环境求解 / 扫掠引擎的源代码调整后，重跑本基准并与
本文件数据对照**，确认无内存回归（HWM 峰值暴涨、驻留泄漏）。

## 复现方法

```bash
# 完整基准（36 case，输入键 32 → 乘积键 1024 全截断的真实压缩压力；
# 总耗时 ~3.5 h，mult·mps VOMPS D=100 单 case 最长 ~41 min）
julia --project=. performance/memory_presure_tests/memory_audit_stress.jl

# 快速回归（36 case，输入键 8——注意 D=100 时乘积键 64 < D 退化为无损
# 恒等问题、iters=1 未构成压缩压力，仅作冒烟测试；总耗时 ~10 min）
julia --project=. performance/memory_presure_tests/memory_audit_light.jl
```

指标说明：`alloc` = GC 吞吐分配（密集 GEMM 管线的固有形态，即时回收，
**不是驻留**）；`RSS`/`HWM` = 进程驻留/峰值增量（**暴涨判据**）；
`驻留` = 终态 + 环境 summarysize。回归判据：HWM 增量显著超出下表同 case
基线（> 2×），或 RSS 随 case 间持续增长（泄漏）。

## 基线数据（Julia 1.10.11，2026-10-13，commit 477fa05）

### memory_audit_stress.jl（输入键 32，36 case 全截断）

| case | t (s) | alloc (GiB) | RSS + (MB) | HWM + (MB) | 驻留 (MB) | iters |
|---|---|---|---|---|---|---|
| compress IDMRG1 D=25 | 7.5 | 0.355 | 42.9 | 35.4 | 0.5 | 30 unc |
| compress IDMRG2 D=25 | 2.7 | 0.192 | 1.1 | 0.0 | 0.5 | 8 conv |
| compress VOMPS D=25 | 1.6 | 0.445 | 5.5 | 4.8 | 0.5 | 30 unc |
| compress IDMRG1 D=50 | 0.3 | 0.076 | 0.9 | 0.9 | 0.7 | 30 unc |
| compress IDMRG2 D=50 | 0.3 | 0.034 | 0.2 | 0.0 | 1.4 | 1 conv |
| compress VOMPS D=50 | 0.1 | 0.03 | 0.0 | 0.0 | 0.7 | 1 conv |
| compress IDMRG1 D=100 | 0.4 | 0.076 | 4.6 | 0.1 | 0.7 | 30 unc |
| compress IDMRG2 D=100 | 0.4 | 0.053 | 5.3 | 0.6 | 4.7 | 1 conv |
| compress VOMPS D=100 | 0.5 | 0.03 | 0.0 | 0.0 | 0.7 | 1 conv |
| mult·mps IDMRG1 D=25 | 114.5 | 68.6 | 117.4 | 296.6 | 2.2 | 30 unc |
| mult·mps IDMRG2 D=25 | 177.8 | 158.4 | -8.3 | 57.2 | 2.2 | 30 unc |
| mult·mps VOMPS D=25 | 235.2 | 203.7 | 19.6 | 0.0 | 2.2 | 30 unc |
| mult·mps IDMRG1 D=50 | 426.7 | 355.1 | -24.2 | 195.2 | 4.5 | 30 unc |
| mult·mps IDMRG2 D=50 | 698.5 | 640.0 | 127.5 | 0.0 | 4.5 | 30 unc |
| mult·mps VOMPS D=50 | 837.2 | 812.8 | -16.0 | 40.7 | 4.5 | 30 unc |
| mult·mps IDMRG1 D=100 | 756.7 | 969.3 | 133.1 | **1390.4** | 10.9 | 30 unc |
| mult·mps IDMRG2 D=100 | 1853.6 | 1951.8 | 135.0 | 0.0 | 10.9 | 30 unc |
| mult·mps VOMPS D=100 | 2484.9 | 2068.5 | 0.0 | 0.0 | 10.9 | 30 unc |
| mult·mpo IDMRG1 D=25 | 148.5 | 53.6 | 46.2 | 0.0 | 2.3 | 30 unc |
| mult·mpo IDMRG2 D=25 | 172.5 | 84.1 | -1.4 | 0.0 | 2.3 | 30 unc |
| mult·mpo VOMPS D=25 | 3021.3 | 1577.2 | 30.5 | 0.0 | 2.3 | 30 unc |
| mult·mpo IDMRG1 D=50 | 44.7 | 22.4 | 0.7 | 0.0 | 5.4 | 30 unc |
| mult·mpo IDMRG2 D=50 | 125.1 | 75.5 | 59.4 | 0.0 | 5.4 | 30 unc |
| mult·mpo VOMPS D=50 | 5038.3 | 4155.0 | 54.4 | 0.0 | 5.4 | 30 unc |
| mult·mpo IDMRG1 D=100 | 43.2 | 20.6 | 64.4 | 0.0 | 14.4 | 30 unc |
| mult·mpo IDMRG2 D=100 | 217.7 | 117.9 | 8.9 | 0.0 | 14.4 | 30 unc |
| mult·mpo VOMPS D=100 | 125.6 | 108.3 | 21.7 | 0.0 | 14.4 | 30 unc |
| hadamard IDMRG1 D=25 | 50.5 | 43.9 | 6.9 | 0.0 | 2.3 | 30 unc |
| hadamard IDMRG2 D=25 | 136.4 | 136.9 | -50.9 | 0.0 | 2.3 | 30 unc |
| hadamard VOMPS D=25 | 277.4 | 293.0 | 48.1 | 0.0 | 2.3 | 30 unc |
| hadamard IDMRG1 D=50 | 38.4 | 36.5 | 45.2 | 0.0 | 4.6 | 30 unc |
| hadamard IDMRG2 D=50 | 130.3 | 129.7 | -18.0 | 0.0 | 4.6 | 30 unc |
| hadamard VOMPS D=50 | 131.9 | 133.2 | 0.0 | 0.0 | 4.6 | 30 unc |
| hadamard IDMRG1 D=100 | 33.7 | 26.3 | 242.3 | 0.0 | 11.0 | 30 unc |
| hadamard IDMRG2 D=100 | 138.5 | 130.5 | 0.0 | 0.0 | 11.0 | 30 unc |
| hadamard VOMPS D=100 | 142.0 | 128.0 | 73.2 | 0.0 | 11.0 | 30 unc |

### memory_audit_light.jl（输入键 8，快速冒烟）

结论同上、量级低 1–2 个数量级（alloc ≤ 12.3 GiB @ mult·mps VOMPS D=50，
HWM ≤ 50 MB，驻留 ≤ 8.3 MB）；注意其 compress 与 D=100 的多数 case 因
乘积键 64 < D 退化为无损恒等问题（iters=1 conv、t≈0），不构成压缩压力。

## 基线结论（2026-10-13 审计）

- **36/36 无内存暴涨**：HWM 峰值增量全局最大 **1.4 GB**（mult·mps IDMRG1
  D=100——BLAS 工作集 + 初猜一次性乘积键 1024 规范化），其余 ≤300 MB、
  过半 +0；终态+环境驻留全部 ≤14.4 MB；RSS 增量 ≤250 MB 且 case 间回落
  （无泄漏累积）；
- **吞吐分配**（GC 即时回收，非驻留）：compress ≤0.45 GiB；mult·mps 随
  D 增长（D=100 达 1–2 TB/30 轮——每轮环境 Lanczos + 大 GEMM 中间量）；
  mult·mpo VOMPS 峰值 4155 GiB（D=50）；hadamard ≤137 GiB；
- **时间维度**：mpo·mpo VOMPS 计算最重（D=50 84 min/30 轮）——环境
  Arnoldi 在随机密谱上的计算量问题（非内存）；D=100 的 IDMRG 系因随机谱
  Krylov 收敛特性反而快于 D=50（43–218 s）。

## 历史背景

D=100 的 mult IDMRG2 曾存在真实内存爆炸（mpo·mpo 通道 `OutOfMemoryError`，
根因为多因子 `@tensor` 收缩的外积中间张量 ~334 GiB 单次申请），已于
2026-10-12 修复（多因子热点拆二元链式收缩，见 changes.md「内存审计」条目）。
本基准即其回归测试的全通道扩展。
