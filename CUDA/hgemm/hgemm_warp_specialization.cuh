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
 * WARP SPECIALIZED HGEMM via TMA + WGMMA
 * C = alpha * A^T * B + beta * C   (TN col-major).
 *
 * Warp-specialized GEMM using a producer-consumer pattern: one warpgroup
 * issues TMA bulk copies from global to shared memory through a circular
 * queue, while the remaining warpgroup(s) consume the tiles via WGMMA
 * asynchronous tensor-core operations, synchronized through arrive/wait
 * barriers on the shared-memory queue slots.
 * 
 * Each TMA takes ~200-400 cycles 
 * Each WGMMA takes ~64 cycles, but for each TMA load, 8-16 WGMMA are called
 * Thus, time for WGMMA is practically longer than time for TMA
 */

namespace hgemm_warp_specialization {

namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

template <int BM, int BN, int BK, int QSIZE>
struct smem {
    alignas(128) half A[BM*BK*QSIZE];
    alignas(128) half B[BK*BN*QSIZE];
};

template<int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ __launch_bounds__(NUM_THREADS)
void hgemm_warp_specialization_device(
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
    // number of consumers 
    constexpr int NCS = NWG - 1;
    // split along M direction only
    constexpr int WGM = WM * NCS;
    // constexpr int WGN = BN;
    // constexpr int WKG = WK;
    
    extern __shared__ __align__(128) char smem_buf[];
    auto &s = *reinterpret_cast<smem<BM, BN, BK, QSIZE>*>(smem_buf);
    half* sA = reinterpret_cast<half*>(s.A);  // (BM, BK):(BK, 1)
    half* sB = reinterpret_cast<half*>(s.B);  // (BN, BK):(BK, 1)

    // block shift 
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;
    // warpgroup id 
    int wgid = threadIdx.x / 128;
    // consumer id 
    int csid = wgid - 1;
    // thread id within warpgroup
    int tid = threadIdx.x % 128;

    // barrier
    // full: is slot i ready to read 
    // empty: is slot i ready to write
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[QSIZE], empty[QSIZE];

    if (threadIdx.x == 0) {
        // each full/empty only need to init for 1 in producer and all in consumer
        for (int i = 0; i < QSIZE; i++) {
            init(&full[i], NCS * 128 + 1);
            init(&empty[i], NCS * 128 + 1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (wgid == 0) { // producer 
        // deallocate registers and keep upto NRG
        constexpr int NRG = (NCS <= 2 ? 24: 32);
        warpgroup_reg_dealloc<NRG>();

        if (tid == 0) {
            for (int bk = 0, qid = 0; bk < K; bk += BK, qid = (qid + 1) % QSIZE) {
                // make sure the qid slot is consumed by all consumers before overwriting
                empty[qid].wait(empty[qid].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[qid * BM * BK], &tmapA, bk, bm, full[qid]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[qid * BK * BN], &tmapB, bk, bn, full[qid]);
                barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full[qid], 1, (BM*BK + BK*BN)*sizeof(half));
            }
        }
    } else { // consumer
        // allocate registers upto NRG for WGMMA
        constexpr int NRG = (NCS == 1 ? 256 : (NCS == 2 ? 240 : 160));
        warpgroup_reg_alloc<NRG>();

        // signal all ready to write 
        for (int i = 0; i < QSIZE; ++i) {
            barrier::arrival_token _ = empty[i].arrive();
        }
        
        // register  accumulator fragments 
        // since entire smem is loaded once by TMA, need to save all accumualtors in registers
        // BN/WGN = 1 which is ignored
        float d[BM/WGM][WN/16][8];
        memset(d, 0, sizeof(d));

        for (int bk = 0, qid = 0; bk < K; bk += BK, qid = (qid + 1) % QSIZE) {
            // wait for producer to fill the qid slot
            full[qid].wait(full[qid].arrive());
            // compute 
            // fence: smem writes complete before wgmma reads
            warpgroup_arrive();
            #pragma unroll 
            for (int wm = 0; wm < BM; wm += WGM) {
                // BN loop is ignored since WN = BN
                half* sAm = &sA[qid * BM * BK + (wm + csid * WM) * BK];
                half* sBm = &sB[qid * BK * BN];
                #pragma unroll
                for (int wk = 0; wk < BK; wk += WK) {
                    wgmma<WN, 1, 1, 1, 0, 0>(d[wm/WGM], &sAm[wk], &sBm[wk]);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();

            // signal producer that qid slot is consumed and ready for next write
            barrier::arrival_token _ = empty[qid].arrive();
        }

        // epilogue
        // CLayout of MMA atom for each warpgroup
        // using CLayout_64xN = Layout<Shape <Shape <_4,_8,_4>,Shape <_2,_2,Int<N/8>>>,
        //                      Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>;
        // m = t1 + 16*t2 + 8*v1
        // n = 2*t0 + v0 + 8*v2
        {
            int t0 = tid %4;
            int t1 = (tid /4) % 8;
            int t2 = (tid /32) % 4;

            int mo = t1 + 16*t2 + csid * WM;
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
}


template<int BM, int BN, int BK, int NUM_CONSUMERS = 1, int QSIZE = 2>
void hgemm_warp_specialization(
    int M, int N, int K,
    float alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    float beta,
    half* C, int ldC)
{
    static_assert(NUM_CONSUMERS >= 1, "need at least 1 consumer");
    constexpr int NUM_THREADS = (NUM_CONSUMERS + 1) * 128;

    CUtensorMap tmap_A{}, tmap_B{};
    (void) ptx_wrapper::build_tma_descriptor(&tmap_A, A, /*outer=*/M, /*inner=*/K, BK, BM);
    (void) ptx_wrapper::build_tma_descriptor(&tmap_B, B, /*outer=*/N, /*inner=*/K, BK, BN);

    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    auto kernel = hgemm_warp_specialization_device<BM, BN, BK, NUM_THREADS, QSIZE>;

    constexpr int kSmemBytes = sizeof(smem<BM, BN, BK, QSIZE>);
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    kernel<<<grid, block, kSmemBytes>>>(tmap_A, tmap_B, M, N, K, alpha, beta, C, ldC);
}

}  // namespace hgemm_warp_specialization
