// Instantiates CUTLASS/CuTe Flash Attention kernels as extern "C" for Python ctypes.
// Layout: [batch, heads, seq_len, d_model] row-major, cute::half_t storage, float compute.

#include <cuda_fp16.h>
#include "flash_attention_naive.cuh"
#include "flash_attention_register.cuh"
#include "flash_attention_mma.cuh"
#include "flash_attention_multistage.cuh"
#include "flash_attention_tma.cuh"
#include "flash_attention_pregs.cuh"
#include "flash_attention_wsp.cuh"
#include "flash_attention_fa3.cuh"
#include "flash_attention_wgmma.cuh"

#define INSTANTIATE_FLASH_ATTENTION_NAIVE(BC, BR)                         \
  extern "C" void cutlass_flash_attention_naive_##BC##x##BR(              \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_naive<BC, BR>(                                        \
        batch_size, num_heads, seq_len, d_model,                          \
        reinterpret_cast<const cute::half_t*>(Q),                         \
        reinterpret_cast<const cute::half_t*>(K),                         \
        reinterpret_cast<const cute::half_t*>(V),                         \
        reinterpret_cast<const cute::half_t*>(mask),                      \
        reinterpret_cast<cute::half_t*>(out));                            \
  }

// INSTANTIATE_FLASH_ATTENTION_NAIVE(8, 32)

#undef INSTANTIATE_FLASH_ATTENTION_NAIVE


#define INSTANTIATE_FLASH_ATTENTION_REGISTER(BC, BR)                      \
  extern "C" void cutlass_flash_attention_register_##BC##x##BR(           \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_register<BC, BR>(                                     \
        batch_size, num_heads, seq_len, d_model,                          \
        reinterpret_cast<const cute::half_t*>(Q),                         \
        reinterpret_cast<const cute::half_t*>(K),                         \
        reinterpret_cast<const cute::half_t*>(V),                         \
        reinterpret_cast<const cute::half_t*>(mask),                      \
        reinterpret_cast<cute::half_t*>(out));                            \
  }

// INSTANTIATE_FLASH_ATTENTION_REGISTER(32, 8)

#undef INSTANTIATE_FLASH_ATTENTION_REGISTER

#define INSTANTIATE_FLASH_ATTENTION_MMA_D64(BC, BR)                       \
  extern "C" void cutlass_flash_attention_mma_d64_##BC##x##BR(            \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_mma<BC, BR, 64>(                                      \
        batch_size, num_heads, seq_len, d_model,                          \
        reinterpret_cast<const cute::half_t*>(Q),                         \
        reinterpret_cast<const cute::half_t*>(K),                         \
        reinterpret_cast<const cute::half_t*>(V),                         \
        reinterpret_cast<const cute::half_t*>(mask),                      \
        reinterpret_cast<cute::half_t*>(out));                            \
  }

#define INSTANTIATE_FLASH_ATTENTION_MMA_D128(BC, BR)                      \
  extern "C" void cutlass_flash_attention_mma_d128_##BC##x##BR(           \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_mma<BC, BR, 128>(                                     \
        batch_size, num_heads, seq_len, d_model,                          \
        reinterpret_cast<const cute::half_t*>(Q),                         \
        reinterpret_cast<const cute::half_t*>(K),                         \
        reinterpret_cast<const cute::half_t*>(V),                         \
        reinterpret_cast<const cute::half_t*>(mask),                      \
        reinterpret_cast<cute::half_t*>(out));                            \
  }


// mma sweep: (Bc, Br).  NUM_WARPS = Br/16.  d64 = d_model=64, d128 = d_model=128.
INSTANTIATE_FLASH_ATTENTION_MMA_D64(32, 16)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(32, 32)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(32, 64)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(64, 32)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(64, 64)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(128, 64)
INSTANTIATE_FLASH_ATTENTION_MMA_D64(128, 128)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(32, 32)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(32, 64)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(64, 32)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(64, 64)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(64, 128)
INSTANTIATE_FLASH_ATTENTION_MMA_D128(128, 64)

// multistage (cp.async + NUM_STAGES-deep pipeline). D=128. NS in macro name.
#define INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S2(BC, BR)                         \
  extern "C" void cutlass_flash_attention_multistage_d128_s2_##BC##x##BR(              \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_multistage<BC, BR, 128, 2>(                                        \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

#define INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S3(BC, BR)                         \
  extern "C" void cutlass_flash_attention_multistage_d128_s3_##BC##x##BR(              \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_multistage<BC, BR, 128, 3>(                                        \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

#define INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S4(BC, BR)                         \
  extern "C" void cutlass_flash_attention_multistage_d128_s4_##BC##x##BR(              \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_multistage<BC, BR, 128, 4>(                                        \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

// Smem (D=128, per stage = 16KB; fixed = 2*Br*D + 2*D*Bc + 2*Br*Bc bytes).
// SM120 cap is ~99KB/block (opt-in).
// Br=128: fixed=48KB → NS=2: 80KB, NS=3: 96KB, NS=4 exceeds.
// Br=64:  fixed=28KB → NS=2: 60KB, NS=3: 76KB, NS=4: 92KB.
// Br=32:  fixed=18KB → NS=2..5 all fit.
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S2(32, 32)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S3(32, 32)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S4(32, 32)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S2(32, 64)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S3(32, 64)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S4(32, 64)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S2(32, 128)
INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S3(32, 128)

// TMA (cp.async.bulk.tensor). D=128. NUM_STAGES in macro name.
#define INSTANTIATE_FLASH_ATTENTION_TMA_D128_S2(BC, BR)                                \
  extern "C" void cutlass_flash_attention_tma_d128_s2_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_tma<BC, BR, 128, 2>(                                               \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }
#define INSTANTIATE_FLASH_ATTENTION_TMA_D128_S3(BC, BR)                                \
  extern "C" void cutlass_flash_attention_tma_d128_s3_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_tma<BC, BR, 128, 3>(                                               \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

INSTANTIATE_FLASH_ATTENTION_TMA_D128_S2(32, 64)
INSTANTIATE_FLASH_ATTENTION_TMA_D128_S3(32, 64)
INSTANTIATE_FLASH_ATTENTION_TMA_D128_S2(32, 128)
INSTANTIATE_FLASH_ATTENTION_TMA_D128_S3(32, 128)

#undef INSTANTIATE_FLASH_ATTENTION_TMA_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_TMA_D128_S3

// pregs: multistage + P in registers. D=128.
#define INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S2(BC, BR)                              \
  extern "C" void cutlass_flash_attention_pregs_d128_s2_##BC##x##BR(                   \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_pregs<BC, BR, 128, 2>(                                             \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }
#define INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S3(BC, BR)                              \
  extern "C" void cutlass_flash_attention_pregs_d128_s3_##BC##x##BR(                   \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_pregs<BC, BR, 128, 3>(                                             \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S2(32, 64)
INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S3(32, 64)
INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S2(32, 128)
INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S3(32, 128)

#undef INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_PREGS_D128_S3

// wsp: warp-specialized (1 producer warp + Br/16 consumer warps).
#define INSTANTIATE_FLASH_ATTENTION_WSP_D128_S2(BC, BR)                                \
  extern "C" void cutlass_flash_attention_wsp_d128_s2_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_wsp<BC, BR, 128, 2>(                                               \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }
#define INSTANTIATE_FLASH_ATTENTION_WSP_D128_S3(BC, BR)                                \
  extern "C" void cutlass_flash_attention_wsp_d128_s3_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_wsp<BC, BR, 128, 3>(                                               \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

INSTANTIATE_FLASH_ATTENTION_WSP_D128_S2(32, 64)
INSTANTIATE_FLASH_ATTENTION_WSP_D128_S2(32, 128)
INSTANTIATE_FLASH_ATTENTION_WSP_D128_S3(32, 64)
INSTANTIATE_FLASH_ATTENTION_WSP_D128_S3(32, 128)

#undef INSTANTIATE_FLASH_ATTENTION_WSP_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_WSP_D128_S3

// fa3: ping-pong WG_A (QK^T) + WG_B (softmax+PV). Br fixed at 64.
#define INSTANTIATE_FLASH_ATTENTION_FA3_D128_S2(BC, BR)                                \
  extern "C" void cutlass_flash_attention_fa3_d128_s2_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_fa3<BC, 128, 2>(                                                   \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }
#define INSTANTIATE_FLASH_ATTENTION_FA3_D128_S3(BC, BR)                                \
  extern "C" void cutlass_flash_attention_fa3_d128_s3_##BC##x##BR(                     \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_fa3<BC, 128, 3>(                                                   \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

INSTANTIATE_FLASH_ATTENTION_FA3_D128_S2(32, 64)
INSTANTIATE_FLASH_ATTENTION_FA3_D128_S3(32, 64)

#undef INSTANTIATE_FLASH_ATTENTION_FA3_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_FA3_D128_S3

// wgmma: TMA + WGMMA (SM90) warp-specialized. Br fixed at 64.
#define INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S2(BC, BR)                              \
  extern "C" void cutlass_flash_attention_wgmma_d128_s2_##BC##x##BR(                   \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_wgmma<BC, 128, 2>(                                                 \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }
#define INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S3(BC, BR)                              \
  extern "C" void cutlass_flash_attention_wgmma_d128_s3_##BC##x##BR(                   \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_wgmma<BC, 128, 3>(                                                 \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

#define INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S4(BC, BR)                              \
  extern "C" void cutlass_flash_attention_wgmma_d128_s4_##BC##x##BR(                   \
      int batch_size, int num_heads, int seq_len, int d_model,                         \
      const void* Q, const void* K, const void* V,                                     \
      const void* mask, void* out) {                                                   \
    flash_attention_wgmma<BC, 128, 4>(                                                 \
        batch_size, num_heads, seq_len, d_model,                                       \
        reinterpret_cast<const cute::half_t*>(Q),                                      \
        reinterpret_cast<const cute::half_t*>(K),                                      \
        reinterpret_cast<const cute::half_t*>(V),                                      \
        reinterpret_cast<const cute::half_t*>(mask),                                   \
        reinterpret_cast<cute::half_t*>(out));                                         \
  }

INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S2(64, 64)
INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S3(64, 64)
INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S4(64, 64)
INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S2(128, 64)
INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S3(128, 64)

#undef INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S3
#undef INSTANTIATE_FLASH_ATTENTION_WGMMA_D128_S4

#undef INSTANTIATE_FLASH_ATTENTION_MMA_D64
#undef INSTANTIATE_FLASH_ATTENTION_MMA_D128
#undef INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S2
#undef INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S3
#undef INSTANTIATE_FLASH_ATTENTION_MULTISTAGE_D128_S4
