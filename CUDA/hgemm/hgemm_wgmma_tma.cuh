#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <mma.h>
#include <cassert>
#include "ptx_wrapper.cuh"

/*
 * HGEMM via TMA + WGMMA
 * C = alpha * A^T * B + beta * C   (TN col-major).
 */

namespace hgemm_wgmma_tma {

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

template<int BM, int BN, int BK, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void hgemm_wgmma_tma_device(
    const __grid_constant__ CUtensorMap tmapA,
    const __grid_constant__ CUtensorMap tmapB,
    int M, int N, int K,
    float alpha, float beta,
    half* __restrict__ C, int ldC)
{
    using namespace ptx_wrapper;
    // MMA atom size 
    constexpr int WM = 64;
    constexpr int WN = BN;
    constexpr int WK = 16;
    static_assert(BM % WM == 0 && BN % WN == 0, "tile sizes must divide block sizes");

    // number of warpgroups 
    constexpr int NWG = NUM_THREADS / 128;
    // split along M direction only
    constexpr int WGM = WM * NWG;
    // constexpr int WGN = BN;
    // constexpr int WKG = WK;
    
    extern __shared__ __align__(128) char smem_buf[];
    half* sA = reinterpret_cast<half*>(smem_buf);                          // (BM, BK):(BK, 1)
    half* sB = reinterpret_cast<half*>(smem_buf + BM * BK * sizeof(half)); // (BN, BK):(BK, 1)

    // block shift 
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;
    // warpgroup id 
    int wgid = threadIdx.x / 128;

    // register  accumulator fragments 
    // since entire smem is loaded once by TMA, need to save all accumualtors in registers
    // BN/WGN = 1 which is ignored
    float d[BM/WGM][WN/16][8];
    // float d[BM/WGM][WN/8][4];
    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    // barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;
    // tokens for producer-consumer synchronization
    barrier::arrival_token tokenA, tokenB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // mainloop over K (all but final tile)
    #pragma nounroll
    for (int bk = 0; bk < K; bk += BK) {
        // tma load 
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(sA, &tmapA, bk, bm, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK*BM*sizeof(half));
            cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &tmapB, bk, bn, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK*BN*sizeof(half));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        // compte 
        // fence: smem writes complete before wgmma reads
        warpgroup_arrive();
        #pragma unroll 
        for (int wm = 0; wm < BM; wm += WGM) {
            // BN loop is ignored since WN = BN
            half* sAm = sA + (wm + wgid * WM) * BK;
            #pragma unroll
            for (int wk = 0; wk < BK; wk += WK) {
                wgmma<WN, 1, 1, 1, 0, 0>(d[wm/WGM], &sAm[wk], &sB[wk]);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // epilogue
    // CLayout of MMA atom for each warpgroup
    // using CLayout_64xN = Layout<Shape <Shape <_4,_8,_4>,Shape <_2,_2,Int<N/8>>>,
    //                      Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;
    // m = t1 + 16*t2 + 8*v1
    // n = 2*t0 + v0 + 8*v2
    {
        int tid = threadIdx.x % 128;
        int t0 = tid %4;
        int t1 = (tid /4) % 8;
        int t2 = (tid /32) % 4;

        int mo = t1 + 16*t2 + wgid * WM;
        int no = 2*t0;
        // advance C pointer 
        C += bm + bn * M;
        // store 
        #pragma unroll
        for (int wm = 0; wm < BM; wm += WGM) {
            int m = mo + wm;
            #pragma unroll
            for (int w = 0; w < WN/16; w++) {
                int n = no + 16*w;
                #define IDX(i,j) ((i) + (j) * M)
                #define ST(i, j, v) C[IDX(i, j)] = (half)(alpha * (v) + beta * (float)C[IDX(i, j)])
                ST(m,   n,   d[wm/WGM][w][0]);
                ST(m,   n+1, d[wm/WGM][w][1]);
                ST(m+8, n,   d[wm/WGM][w][2]);
                ST(m+8, n+1, d[wm/WGM][w][3]);
                ST(m,   n+8, d[wm/WGM][w][4]);
                ST(m,   n+9, d[wm/WGM][w][5]);
                ST(m+8, n+8, d[wm/WGM][w][6]);
                ST(m+8, n+9, d[wm/WGM][w][7]);
                #undef ST
                #undef IDX
            }
        }
    }
}


template<int BM, int BN, int BK, int NUM_WARPGROUPS = 1>
void hgemm_wgmma_tma(
    int M, int N, int K,
    float alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    float beta,
    half* C, int ldC)
{
    static_assert(NUM_WARPGROUPS >= 1, "need at least 1 warpgroup");
    constexpr int NUM_THREADS = NUM_WARPGROUPS * 128;

    CUtensorMap tmap_A{}, tmap_B{};
    (void) ptx_wrapper::build_tma_descriptor(&tmap_A, A, /*outer=*/M, /*inner=*/K, BK, BM);
    (void) ptx_wrapper::build_tma_descriptor(&tmap_B, B, /*outer=*/N, /*inner=*/K, BK, BN);

    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    auto kernel = hgemm_wgmma_tma_device<BM, BN, BK, NUM_THREADS>;

    constexpr int kSmemBytes = (BM * BK + BN * BK) * (int) sizeof(half);
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    kernel<<<grid, block, kSmemBytes>>>(tmap_A, tmap_B, M, N, K, alpha, beta, C, ldC);
}

}  // namespace hgemm_wgmma_tma
