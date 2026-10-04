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

## Experimental fused tensor-core path

The selected MIT-licensed kernels from [pearl-hashrate-miner](https://github.com/puneet-mehta/pearl-hashrate-miner) are bound to revision `a6574254cb1599046b174236ee5bea1070534517` under `third_party/pearl-sm120`. This is a prototype tested separately from the pool miner. It uses native signed int8 tensor instructions, three stages of shared-memory prefetching, and register-resident cumulative products instead of writing rank-history matrices to global memory.

`tensor_kernel_test` checks all 1,024 transcripts on a 256x256x4096 input against an independent CPU matrix calculation, then checks all keyed digests against the portable BLAKE3 library. Maximum and zero targets exercise acceptance and rejection. Signed inputs are independently seeded in [-63,63]. CUDA memcheck and synccheck reported zero errors; racecheck reported zero errors and warnings on this correctness fixture.

The isolated 8192x8192x4096 scan measured **15.035 ms, 18.28 TMAC/s**, averaged over ten launches after three warmups, with independently seeded random matrices. Allocation, input generation, host transfers, noise preparation, and proof generation are excluded. This is a CUDA-event kernel rate, not live accepted mining throughput. Timing fixtures use the same kernel but larger matrices than the CPU correctness fixture.

```bash
build/cmake/tensor_kernel_test --bench
ctest --test-dir build/cmake --output-on-failure
```

Its 2x32 hash pattern (rows `[0,8]`, columns `[0,1,8,9,...,120,121]`) differs from the current miner's 8x16 pattern. The Rust proof builder now supports layout 5, and independent proof fixtures pass the full ZK verifier for certificate versions 2 and 3 at both interior and boundary anchors. Tests cover all 256 lane anchors, the 262144 target normalization factor, and invalid matrix bounds. The inherited proof fixtures were corrected to use matrix sizes that fit their tiles and valid verification dimensions; the 11-test Rust suite now runs through CTest.

The opt-in `--tensor-fused` mode now connects GPU candidate selection, job configuration, proof metadata, target scaling, and fee work accounting to that pattern. On the 5060 Ti, a dev mock found a candidate at nonce 308; the full in-process ZK verifier accepted its certificate version 3 proof. A short production-dimension LuckyPool run then returned seven successful submit responses (IDs 2–8, all `{"error":null,"result":true}`). This is evidence of pool acceptance; the local mock independently checks the full proof. The median attempt rate across 15 production-size attempts was 12.15 TMAC/s. The mode remains opt-in while longer runs, power draw, and payout behavior are measured.

```bash
build/cmake/cppminer --backend cuda --tensor-fused \
  --row-period-batch 4 --col-period-batch 32 --mock --dev
```

## Correctness and remaining work

A subsequent tuning check compared separate rank GEMMs against `cublasGemmStridedBatchedEx` over three alternating trials. At 4/32 batches the separate path gave 5.35, 5.32, and 5.53 TMAC/s wall sweep rates, versus 5.30, 5.14, and 5.36 for the batched path. At 16/32 both paths were about 5.9 TMAC/s. The batched experiment was removed because it gave no reliable gain. Larger row batches offer a small speed improvement in this small profile, but have not been selected for live operation or checked for whole-system energy efficiency.

The CUDA differential test compares all 16 jackpot words for 257 synthetic tiles against a separate serial CPU calculation over 64 cumulative rank steps. The optimized miner also builds a mock share that passes the ZK verifier. A 120-second LuckyPool test received five successful `plain_proof` share replies (IDs 2 through 6) and was then stopped. Fee switching was disabled because the payout and project fee addresses were identical.

This rate is still far below the historical closed miner's 91.3 TH/s on the same GPU. Whole-system wall power and a completed payout remain unverified. The dashboard keeps automatic mining disabled. The next performance work should reduce GEMM and rank-history memory traffic while preserving these proof checks.
