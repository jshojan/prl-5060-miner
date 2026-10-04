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

The opt-in `--tensor-fused` mode now connects GPU candidate selection, job configuration, proof metadata, target scaling, and fee work accounting to that pattern. On the 5060 Ti, a dev mock found a candidate at nonce 308; the full in-process ZK verifier accepted its certificate version 3 proof. A short production-dimension LuckyPool run then returned seven successful submit responses (IDs 2–8, all `{"error":null,"result":true}`). This is evidence of pool acceptance; the local mock independently checks the full proof. The median attempt rate across 15 production-size attempts was 12.15 TMAC/s. A second 30-second production-size run returned two further successful submit responses. During it, 27 active 1-second `nvidia-smi` samples showed 87.22 W median and 70.99 W minimum GPU draw. This is GPU telemetry, not whole-computer power. The mode remains opt-in while longer runs and payout behavior are measured.

```bash
build/cmake/cppminer --backend cuda --tensor-fused \
  --row-period-batch 4 --col-period-batch 32 --mock --dev
```

## Direct tensor hit selection and target upload

The fused kernel now records the first qualifying 2x32 candidate directly with an atomic flag and its proof coordinates. Normal mining omits the per-thread diagnostic hash and hit buffers and the separate hit-collection kernel. It also uploads the unchanged 32-byte target once per attempt instead of once per row/column batch. The diagnostic outputs remain available to the independent CUDA test. That test still checks all 1,024 transcripts and keyed digests for both rank 64 and 128, and checks that a target set to the minimum digest returns its exact row and column even with nonzero period offsets. All four CTest checks passed. An offline certificate-version-3 mock share passed the full verifier after both changes.

Two baseline/new/new/baseline sequences measured full 131072² production-size sweeps at the 4/32 batch setting. The first gave **20.18 / 20.68 / 20.58 / 20.29 TMAC/s** wall scan rates; the second gave **20.48 / 21.44 / 21.20 / 20.33 TMAC/s**. The change improved the measured scan rate by roughly 4–6% in these short runs. A third alternating sequence with GPU power sampling gave baseline **19.48 / 19.59 TMAC/s** at **99.29 / 97.82 W** median active GPU draw, and changed **20.66 / 20.74 TMAC/s** at **101.43 / 101.75 W**. Those samples imply about 3% more GPU-only work per watt, with whole-computer power still unknown. Each scan ran with the Quantus pilot briefly paused; the pilot resumed under its original deadline after every sequence.

A 55-second LuckyPool check of the changed production miner found and submitted **12 shares**, with **12 `result:true` pool responses and no rejected response** in the captured log. The full proof mock and pool replies support the new coordinate path. The short test does not prove a Pearl payout or profitable long-term operation. The local dashboard retains the conservative 22 TH/s planning rate and Quantus pilot while Pearl whole-computer power remains unmeasured.

## Correctness and remaining work

### 64-wide K tile on the RTX 5060 Ti

The fused kernel now uses a 64-wide K tile and three 16 KiB shared-memory stages (48 KiB per CTA), rather than a 128-wide tile and 96 KiB per CTA. A rank-128 checkpoint spans two K tiles, so the transcript cursor advances only after all four 32-wide MMA blocks complete. The independent CPU fixture checks 1,024 full transcripts and keyed digests at **both** rank 64 and rank 128. CUDA memcheck, racecheck, and synccheck found zero errors (racecheck: zero warnings) on that fixture. The offline mock built and fully verified a certificate-version-3 share.

The isolated 8192² × 4096 CUDA-event scan fell from about 15.0 ms (18.3 TMAC/s) to **7.66–8.14 ms** (33.8–35.9 TMAC/s). A separate two-stage 128-wide trial varied between 14.60 and 14.98 ms across six alternating measurements against the original three-stage 128-wide build, which varied between 14.55 and 15.06 ms. Those ranges overlap, so the two-stage trial was not retained.

A 32-wide K tile also passed both rank references, but its first isolated scan took 10.45 ms (26.3 TMAC/s). It was slower than the 64-wide candidate and was reverted before production mining.

At the selected 64-wide tile, a two-stage pipeline also passed both rank references. Four alternating timings favored three stages in every pair: two stages took 7.86–8.38 ms, while three took 7.59–8.02 ms. The three-stage build remains selected.

### Fused miner batch size and power

`--profile-scan` now accepts `--tensor-fused` and profiles the actual fused batch and full sweep. The old profile path called the separate unfused GEMM with no output buffer when the tensor flag was enabled; it now dispatches the fused kernel. On production 131072² dimensions, one full sweep measured **29.94 TH/s** at row/column batch 32/256, **32.90 TH/s** at 64/64, **34.39 TH/s** at 128/64, and **35.00 TH/s** at 256/64. Increasing the row batch to 512/64 gave 34.87 TH/s. The smaller 8192² profile rose from 20.10 TH/s at 4/32 to 33.10 TH/s at 64/64. These are profiling sweeps on prepared matrices, not live earnings.

A 45-second LuckyPool run at 256/64 produced 15 full production scans at **34.79 TH/s median** and **12 accepted shares** with no rejected submit response. Thirty-three active one-second GPU samples had **169.64 W median**. Compared with the earlier 4/32 live sample (22.9 TH/s, 105.1 W GPU median), this configuration increases speed but reduces the observed GPU-only work per watt from about 0.218 to 0.205 TH/s/W. The local dashboard therefore retains the conservative 4/32 configuration and 22 TH/s planning rate. Whole-computer power and a positive net return remain unverified.

A 45-second production-size LuckyPool check of the 64-wide kernel had 19 attempt timings; eight full scans (at least 2.5 seconds) had **22.9 TMAC/s median**. The pool returned **11 accepted share responses**, with no rejected response in the captured log. Of 44 one-second GPU telemetry samples, 36 with at least 50% utilization had **105.1 W median**. This excludes the rest of the computer. The root dashboard uses a conservative **22 TH/s** planning rate and remains disarmed pending measured whole-system power and positive net returns. These figures are short-run results, not a payout or sustained earnings record.

A subsequent tuning check compared separate rank GEMMs against `cublasGemmStridedBatchedEx` over three alternating trials. At 4/32 batches the separate path gave 5.35, 5.32, and 5.53 TMAC/s wall sweep rates, versus 5.30, 5.14, and 5.36 for the batched path. At 16/32 both paths were about 5.9 TMAC/s. The batched experiment was removed because it gave no reliable gain. Larger row batches offer a small speed improvement in this small profile, but have not been selected for live operation or checked for whole-system energy efficiency.

### 256-row tensor tile experiment, 2026-10-04

An isolated build doubled the fused kernel's row tile from 128 to 256, using 512 threads, 16 warps, and 72 KiB dynamic shared memory per CTA. It adjusted B-tile loading so only the first 256 threads load the 128 B rows; CUDA memcheck first identified the otherwise out-of-bounds shared write. The updated independent reference test covered all 1,024 transcripts and digests at ranks 64 and 128, and all four CTest checks passed. The 8192² isolated kernel scan was slower than baseline in adjacent runs: **8.516 / 8.975 ms** versus **8.346 / 8.450 ms**.

Production-size 131072² baseline/variant/variant/baseline sweeps at the selected 4/32 batch setting reported **23.50 / 22.85 / 22.82 / 23.81 TMAC/s** wall rates. Median active GPU telemetry samples were **102.55 / 92.87 / 93.56 / 100.29 W**, with 15–16 samples per run. The larger tile traded roughly 3–4% throughput for 7–10 W less GPU draw. At the dashboard's unmeasured 300 W whole-computer Pearl scenario, that trade does not improve modeled whole-computer work per watt, even if the full GPU-power difference reached the outlet. The change was not selected for release. The isolated source, CTest and memcheck logs, and benchmark outputs are retained under the dashboard project's ignored `.runtime/research/prl-tile-m256*` paths.

The CUDA differential test compares all 16 jackpot words for 257 synthetic tiles against a separate serial CPU calculation over 64 cumulative rank steps. The optimized miner also builds a mock share that passes the ZK verifier. A 120-second LuckyPool test received five successful `plain_proof` share replies (IDs 2 through 6) and was then stopped. Fee switching was disabled because the payout and project fee addresses were identical.

This rate is still far below the historical closed miner's 91.3 TH/s on the same GPU. Whole-system wall power and a completed payout remain unverified. The dashboard keeps automatic mining disabled. The next performance work should reduce GEMM and rank-history memory traffic while preserving these proof checks.
