#include <cuda_runtime.h>
#include "pearl_gemm_search_perthread_smem_pipelined_sm80.cuh"
#include "blake3.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
static void launch(int m, int n, int k, const int8_t* a, const int8_t* b,
                   const uint32_t* key, const uint32_t* target,
                   uint32_t* hashes, uint8_t* hits, uint32_t* words = nullptr) {
    pearl::sm80::search_perthread_smem_pipelined::
        launch_pearl_gemm_search_perthread_smem_pipelined_R<128>(
            m, n, k, a, b, key, target, hashes, hits, nullptr, words);
    check(cudaGetLastError());
}
static uint32_t random_word(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static void verify() {
    constexpr int m = 256, n = 256, k = 4096, rank = 128;
    constexpr int candidates = (m / 128) * (n / 128) * 256;
    std::vector<int8_t> a(m*k), b(n*k);
    uint32_t random = 0x83a19820;
    for(auto& value : a) value = int(random_word(random) % 127) - 63;
    for(auto& value : b) value = int(random_word(random) % 127) - 63;
    std::array<uint32_t,8> key = {0x12345678,9,10,11,12,13,14,15}, target;
    target.fill(0xffffffffu);
    Buffer<int8_t> da(a.size()), db(b.size());
    Buffer<uint32_t> dkey(8), dtarget(8), hashes(candidates*8), words(candidates*16);
    Buffer<uint8_t> hits(candidates);
    check(cudaMemcpy(da.data,a.data(),a.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data,b.data(),b.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dkey.data,key.data(),32,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dtarget.data,target.data(),32,cudaMemcpyHostToDevice));
    launch(m,n,k,da.data,db.data,dkey.data,dtarget.data,hashes.data,hits.data,words.data);
    std::vector<uint32_t> actual_words(candidates*16), actual_hashes(candidates*8);
    std::vector<uint8_t> actual_hits(candidates);
    check(cudaMemcpy(actual_words.data(),words.data,actual_words.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual_hashes.data(),hashes.data,actual_hashes.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual_hits.data(),hits.data,actual_hits.size(),cudaMemcpyDeviceToHost));

    // Serial reference constructs every matrix cell and then gathers each
    // 2x32 hash tile. It shares neither tensor lane mapping nor GPU reduction.
    std::vector<int32_t> cells(m*n,0);
    std::vector<uint32_t> expected(candidates*16,0);
    for(int step=0;step<k/rank;++step) {
        for(int row=0;row<m;++row) for(int col=0;col<n;++col) {
            int32_t partial=0;
            for(int pos=step*rank;pos<(step+1)*rank;++pos)
                partial+=int32_t(a[row*k+pos])*int32_t(b[col*k+pos]);
            cells[row*n+col]+=partial;
        }
        for(int tile_m=0;tile_m<m/128;++tile_m)
            for(int tile_n=0;tile_n<n/128;++tile_n)
                for(int thread=0;thread<256;++thread) {
                    const int row=tile_m*128+(thread/32)*16+(thread%32)/4;
                    const int col=tile_n*128+(thread%4)*2;
                    uint32_t x=0;
                    for(int r : {row,row+8}) for(int pair=0;pair<16;++pair)
                        for(int adjacent=0;adjacent<2;++adjacent)
                            x^=uint32_t(cells[r*n+col+pair*8+adjacent]);
                    const int candidate=(tile_m*(n/128)+tile_n)*256+thread;
                    uint32_t& word=expected[candidate*16+(step%16)];
                    word=((word<<13)|(word>>19))^x;
                }
    }
    if(expected!=actual_words) throw std::runtime_error("tensor transcript differs from CPU");
    uint8_t key_bytes[32];
    for(int i=0;i<32;++i) key_bytes[i]=uint8_t(key[i/4]>>(8*(i%4)));
    for(int candidate=0;candidate<candidates;++candidate) {
        uint8_t message[64], digest[32];
        for(int i=0;i<64;++i) message[i]=uint8_t(expected[candidate*16+i/4]>>(8*(i%4)));
        blake3_hasher hasher;
        blake3_hasher_init_keyed(&hasher,key_bytes);
        blake3_hasher_update(&hasher,message,64);
        blake3_hasher_finalize(&hasher,digest,32);
        for(int i=0;i<32;++i)
            if(digest[i]!=uint8_t(actual_hashes[candidate*8+i/4]>>(8*(i%4))))
                throw std::runtime_error("tensor digest differs from BLAKE3 reference");
        if(actual_hits[candidate]!=1) throw std::runtime_error("maximum target did not accept");
    }
    check(cudaMemset(dtarget.data,0,32));
    launch(m,n,k,da.data,db.data,dkey.data,dtarget.data,hashes.data,hits.data);
    check(cudaMemcpy(actual_hits.data(),hits.data,actual_hits.size(),cudaMemcpyDeviceToHost));
    for(auto hit : actual_hits) if(hit) throw std::runtime_error("zero target accepted");
    std::puts("1024 tensor transcripts and digests match independent CPU references");
}
static void benchmark() {
    constexpr int m=8192,n=8192,k=4096,candidates=(m/128)*(n/128)*256;
    Buffer<int8_t> a(size_t(m)*k), b(size_t(n)*k);
    Buffer<uint32_t> key(8),target(8),hashes(size_t(candidates)*8);
    Buffer<uint8_t> hits(candidates);
    std::vector<int8_t> input(size_t(m)*k);
    uint32_t random=0x91374fac;
    for(auto& value:input) value=int(random_word(random)%127)-63;
    check(cudaMemcpy(a.data,input.data(),input.size(),cudaMemcpyHostToDevice));
    for(auto& value:input) value=int(random_word(random)%127)-63;
    check(cudaMemcpy(b.data,input.data(),input.size(),cudaMemcpyHostToDevice));
    check(cudaMemset(key.data,0,32)); check(cudaMemset(target.data,0,32));
    for(int i=0;i<3;++i) launch(m,n,k,a.data,b.data,key.data,target.data,hashes.data,hits.data);
    check(cudaDeviceSynchronize());
    cudaEvent_t start,end;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    check(cudaEventRecord(start));
    for(int i=0;i<10;++i) launch(m,n,k,a.data,b.data,key.data,target.data,hashes.data,hits.data);
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float elapsed; check(cudaEventElapsedTime(&elapsed,start,end));
    std::printf("Fused tensor scan 8192x8192x4096: %.3f ms, %.2f TMAC/s (10 runs)\n",
                elapsed/10,double(m)*n*k/(elapsed/10*1e9));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
}
int main(int argc,char** argv) {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    cudaDeviceProp device;
    check(cudaGetDeviceProperties(&device,0));
    if(device.major<8) return 77;
    try { verify(); if(argc>1 && std::strcmp(argv[1],"--bench")==0) benchmark(); }
    catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
    return 0;
}
