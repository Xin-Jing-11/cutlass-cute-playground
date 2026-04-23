// Instantiates CUDA Flash Attention kernels as extern "C" for Python ctypes.
// Layout: [batch, heads, seq_len, d_model], row-major.
// __half pointers are exposed as void* at the ABI boundary.

#include "flash_attention_naive.cuh"
#include "flash_attention_register.cuh"

#define INSTANTIATE_FLASH_ATTENTION_NAIVE(BC, BR)                         \
  extern "C" void cuda_flash_attention_naive_##BC##x##BR(                 \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_naive<BC, BR>(                                        \
        reinterpret_cast<const __half*>(Q),                               \
        reinterpret_cast<const __half*>(K),                               \
        reinterpret_cast<const __half*>(V),                               \
        reinterpret_cast<const __half*>(mask),                            \
        reinterpret_cast<__half*>(out),                                   \
        batch_size, num_heads, seq_len, d_model);                         \
  }

// INSTANTIATE_FLASH_ATTENTION_NAIVE(8, 32)

#undef INSTANTIATE_FLASH_ATTENTION_NAIVE

#define INSTANTIATE_FLASH_ATTENTION_REGISTER(BC, BR)                      \
  extern "C" void cuda_flash_attention_register_##BC##x##BR(              \
      int batch_size, int num_heads, int seq_len, int d_model,            \
      const void* Q, const void* K, const void* V,                        \
      const void* mask, void* out) {                                      \
    flash_attention_register<BC, BR>(                                     \
        reinterpret_cast<const __half*>(Q),                               \
        reinterpret_cast<const __half*>(K),                               \
        reinterpret_cast<const __half*>(V),                               \
        reinterpret_cast<const __half*>(mask),                            \
        reinterpret_cast<__half*>(out),                                   \
        batch_size, num_heads, seq_len, d_model);                         \
  }

// INSTANTIATE_FLASH_ATTENTION_REGISTER(32, 8)

#undef INSTANTIATE_FLASH_ATTENTION_REGISTER
