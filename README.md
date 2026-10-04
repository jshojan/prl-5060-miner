# PRL 5060 Miner

An open source PearlHash miner under development for the RTX 5060 Ti. This code is derived from [CPPminer](https://github.com/1640675651/CPPminer) (MIT); the original copyright notice remains in `LICENSE`. See `NOTICE.md` for the upstream revision and third party components.

## Current status

- CUDA 13.0 `sm_120` build succeeds on an RTX 5060 Ti.
- Offline mock mining builds a Pearl proof and passes the included ZK verifier.
- On the RTX 5060 Ti, the warp jackpot kernel reports about 5.5 TH/s after warmup at LuckyPool, up from about 4.7 TH/s before this change. The original run accepted five `plain_proof` shares; the new kernel also passes the full offline proof verifier and has accepted live shares. This remains far below the historical 91.3 TH/s closed miner baseline; pool acceptance does not establish profitability or a payout. See [performance evidence](docs/performance.md).
- The source contains a transparent 1% project fee routed to the project owner's Pearl wallet. When the mining wallet is the same address, fee switching is disabled. No third party developer fee address is present.
- The source is public for review and development. Whole-system power draw and a completed payout remain unmeasured.
- The opt-in `--tensor-fused` path uses the separately checked signed-int8 tensor kernel. A local mock proof passed the full ZK verifier, and a short LuckyPool run returned seven accepted submit responses. Across 15 production-size attempts the median rate was 12.15 TH/s. The isolated kernel benchmark reaches about 18.3 TMAC/s; see [performance evidence](docs/performance.md).

## Build on the 5060 Ti

Requires CUDA 13, CMake, a C++ compiler, Rust, and the build script's downloaded BLAKE3 and CUTLASS dependencies.

```bash
CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc CUDAARCHS=120 \
  ./build.sh --backend cuda --cuda-arch 120
cmake -S . -B build/cmake -DCP_ENABLE_CUBLAS=ON -DCP_ENABLE_CUDA=ON \
  -DCP_ENABLE_CPU=OFF -DCP_CUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j 4
```

## Offline proof check

```bash
build/cmake/cppminer --backend cuda --mock --dev --cublas-period \
  --no-cutlass-fused --row-period-batch 1 --col-period-batch 8 \
  --python /usr/bin/python3
```

The test succeeds only if it reports `verify OK` and `[mock] PASS`. It does not prove pool compatibility or profitable speed.

## Live pool check

The accepted shares used the cuBLAS path with `--row-period-batch 4 --col-period-batch 32` at LuckyPool's Pearl endpoint. The miner was stopped after each short check. Its displayed speed is an internal work-rate estimate, not independently measured earnings. LuckyPool currently advertises a 1% pool fee; confirm the pool terms before mining.

Run `ctest --test-dir build/cmake --output-on-failure` for the fee scheduler and CUDA jackpot tests. The latter compares 257 jackpot messages with an independent CPU calculation across 64 cumulative rank steps, including zero, all-one, sequential, and random partials. It is skipped when no CUDA device is available.

## Fee and attribution

`src/common/cp_fee.cpp` contains the fee address in plain text. The tile debt scheduler gives the project wallet roughly one tile per hundred user tiles over sustained mining. The miner logs wallet switches. The fee recipient is a public Pearl address, never a seed or private key. See `NOTICE.md` and the license files before distributing a build.
