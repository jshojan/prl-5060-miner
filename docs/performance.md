# RTX 5060 Ti performance evidence

Measured on this project's 16 GB RTX 5060 Ti with CUDA 13, `sm_120`, cuBLAS period GEMM, step-major panels, and row/column batches of 4/32. These are short measurements on one computer, dated 2026-10-04.

## Warp jackpot change

The previous kernel assigned 128 threads to a hash tile. Every rank step used two block barriers and a single thread to XOR 128 cumulative cells. The new kernel assigns one warp to each tile, with four cells per lane and four independent tiles per block. Each lane holds one jackpot word; a warp XOR replaces the serial reduction and block barriers. The early exit for an already found share is uniform within a warp.

| Measurement | Previous kernel | Warp kernel |
| --- | ---: | ---: |
| Jackpot time per batch | 1.807 ms | 1.333 ms |
| Full small-matrix sweep, wall rate | 4.55 TMAC/s | 5.31 TMAC/s |
| Short live pool scan after warmup | about 4.7 TH/s | about 5.5 TH/s |

The small sweep improved about 17% in this measurement. It excludes matrix preparation and proof construction. The live miner overlaps proof work with scanning, but short internal rate readings do not independently establish earnings or long-term accepted work per second.

Reproduce the small profile with:

```bash
build/cmake/cppminer --backend cuda --cublas-period --no-cutlass-fused \
  --row-period-batch 4 --col-period-batch 32 --profile-scan 6 --dev
```

## Correctness and remaining work

A subsequent tuning check compared separate rank GEMMs against `cublasGemmStridedBatchedEx` over three alternating trials. At 4/32 batches the separate path gave 5.35, 5.32, and 5.53 TMAC/s wall sweep rates, versus 5.30, 5.14, and 5.36 for the batched path. At 16/32 both paths were about 5.9 TMAC/s. The batched experiment was removed because it gave no reliable gain. Larger row batches offer a small speed improvement in this small profile, but have not been selected for live operation or checked for whole-system energy efficiency.

The CUDA differential test compares all 16 jackpot words for 257 synthetic tiles against a separate serial CPU calculation over 64 cumulative rank steps. The optimized miner also builds a mock share that passes the ZK verifier. A 120-second LuckyPool test received five successful `plain_proof` share replies (IDs 2 through 6) and was then stopped. Fee switching was disabled because the payout and project fee addresses were identical.

This rate is still far below the historical closed miner's 91.3 TH/s on the same GPU. Whole-system wall power and a completed payout remain unverified. The dashboard keeps automatic mining disabled. The next performance work should reduce GEMM and rank-history memory traffic while preserving these proof checks.
