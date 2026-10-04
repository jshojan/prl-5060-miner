#include <cuda_runtime.h>
#include "pp_reduce.cuh"

#include <cstdio>
#include <vector>

constexpr int tiles = 257;
constexpr int steps = 64;
constexpr int threads = 128;
constexpr int words = 16;

/* Exercise cumulative rank partials and repeated updates of all 16 words,
 * using the same reduction helper as the mining kernel. */
__global__ void messages(const uint32_t* partials, uint32_t* output)
{
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * 4 + (threadIdx.x >> 5);
    if(tile >= tiles) return;
    uint32_t cells[4] = {};
    uint32_t word = 0;
    for(int step = 0; step < steps; ++step) {
        uint32_t x = 0;
        for(int i = 0; i < 4; ++i) {
            cells[i] += partials[(tile * steps + step) * threads + lane + 32 * i];
            x ^= cells[i];
        }
        word = pp_warp_jackpot_step(x, step, word);
    }
    if(lane < words) output[tile * words + lane] = word;
}

#define CUDA_OK(call) do { \
    const cudaError_t error = (call); \
    if(error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        return 1; \
    } \
} while(0)

int main()
{
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::vector<uint32_t> partials(tiles * steps * threads);
    uint32_t random = 0x12345678u;
    for(size_t i = 0; i < partials.size(); ++i) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        const size_t tile = i / (steps * threads);
        partials[i] = tile == 0 ? 0u : tile == 1 ? 0xffffffffu
            : tile == 2 ? uint32_t(i) : random;
    }
    std::vector<uint32_t> expected(tiles * words, 0), actual(tiles * words);
    for(int tile = 0; tile < tiles; ++tile) {
        uint32_t cells[threads] = {};
        for(int step = 0; step < steps; ++step) {
            uint32_t x = 0;
            for(int thread = 0; thread < threads; ++thread) {
                cells[thread] += partials[(tile * steps + step) * threads + thread];
                x ^= cells[thread];
            }
            uint32_t& word = expected[tile * words + step % words];
            word = ((word << 13) | (word >> 19)) ^ x;
        }
    }
    uint32_t *input = nullptr, *output = nullptr;
    CUDA_OK(cudaMalloc(&input, partials.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&output, actual.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMemcpy(input, partials.data(), partials.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    messages<<<(tiles + 3) / 4, 128>>>(input, output);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(actual.data(), output, actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(input));
    CUDA_OK(cudaFree(output));
    for(size_t i = 0; i < actual.size(); ++i) {
        if(actual[i] != expected[i]) {
            std::fprintf(stderr, "jackpot word %zu: GPU %08x CPU %08x\n", i, actual[i], expected[i]);
            return 1;
        }
    }
    std::puts("257 jackpot messages match CPU across 64 cumulative rank steps");
    return 0;
}
