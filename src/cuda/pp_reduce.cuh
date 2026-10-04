#ifndef PP_REDUCE_CUH
#define PP_REDUCE_CUH

#include <stdint.h>

/* All 32 lanes participate; each lane receives the full XOR. */
__device__ __forceinline__ uint32_t pp_warp_xor(uint32_t value)
{
#if __CUDA_ARCH__ >= 800
    value = __reduce_xor_sync(0xffffffffu, value);
#else
    for(int offset = 16; offset > 0; offset >>= 1)
        value ^= __shfl_xor_sync(0xffffffffu, value, offset);
#endif
    return value;
}

/* Lanes 0..15 own one jackpot word each. All lanes run every step. */
__device__ __forceinline__ uint32_t pp_warp_jackpot_step(
    uint32_t cell_xor, int step, uint32_t word)
{
    const uint32_t xored = pp_warp_xor(cell_xor);
    if((threadIdx.x & 31) == (step & 15))
        word = ((word << 13) | (word >> 19)) ^ xored;
    return word;
}

#endif
