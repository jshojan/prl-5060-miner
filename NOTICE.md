# Attribution and provenance

This miner began from `1640675651/CPPminer` commit `6785ad3` (MIT, copyright 2026 foolzhz). The original MIT license is retained in `LICENSE`. Changes in this project include the public project fee address and scheduler behavior, a CUDA build fix, and Blackwell build and proof validation work.

The build downloads BLAKE3 1.8.5 and CUTLASS 2.11.0. The source tree includes Pearl proof components (`third_party/PEARL-LICENSE`), Plonky2 (its MIT/Apache licenses), and OpenCL headers. Those components retain their own terms. No CPPminer developer fee wallet is included in this derivative.

`third_party/pearl-sm120/` contains two CUDA headers from `puneet-mehta/pearl-hashrate-miner` revision `a6574254cb1599046b174236ee5bea1070534517`, used under its MIT option. The upstream license and source binding are preserved in that directory. These headers are currently used only by the experimental tensor-kernel reference test; they are not connected to the miner's pool path.
