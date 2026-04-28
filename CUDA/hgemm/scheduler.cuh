#pragma once
#include <cuda.h>                          // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda/barrier>                    // cuda::barrier
#include <cuda/pipeline>                   // (optional, for async pipeline)
#include <cuda_bf16.h>                     // bf16 (__nv_bfloat16)
#include <cassert>                         // assert

namespace scheduler {

template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<0, NUM_SM, BM, BN, TM, TN> {
    // start and end index
    int st, en;

    // assign continues blocks to each SM
    __device__ __forceinline__ Schedule(int M, int N, int block) {
        int total_blocks = M*N/(BM*BN);
        int blocks_per_sm = total_blocks / NUM_SM;
        int extra_blocks = total_blocks % NUM_SM;
        if (block < extra_blocks) {
            st = block*(blocks_per_sm + 1);
            en = st + blocks_per_sm + 1;
        } else {
            st = extra_blocks + block*blocks_per_sm;
            en = st + blocks_per_sm;
        }
    }

    __device__ __forceinline__ int next() {
        if (en == st) return -1;
        return st++;
    }
};

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
    int block;
    int it;
    int total_blocks_m;
    int total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = M/BM;
        total_blocks_n = N/BN;
    }

    // super tiles: TM x TN, eg: 16x8 super tiles, as a unit for all CTA (SM)
    // inside each super tile, blocks are assigned in round robin way
    __device__ __forceinline__ int next() {
        int num = it*NUM_SM + block;    // global sequence number for this SM, round-robin across SMs
        if (num >= total_blocks_m*total_blocks_n) return -1;

        int cur_tile = num / (TM*TN);       // which super-tile? (0, 1, 2, ...)
        int cur_tile_pos = num % (TM*TN);   // position within the super-tile (0..127)

        // Super-tile origin in tile coordinates:
        int m = TM * (cur_tile / (total_blocks_n/TN));   // super-tile row
        int n = TN * (cur_tile % (total_blocks_n/TN));   // super-tile col

        // Offset within super-tile (row-major within TM×TN):
        m += cur_tile_pos / TN;    // local row
        n += cur_tile_pos % TN;    // local col
        ++it;

        return m*total_blocks_n + n;   // re-linearize to 1D tile index
    }
};

}
