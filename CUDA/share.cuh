#pragma once
#include <cstdint>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__host__ __device__ 
inline bool is_pow2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// Identical to cute::Swizzle<B, M, S>.
// XORs B bits at position [M, M+B) with B bits at position [M+S, M+S+B).
//   B = number of bits to swizzle
//   M = bit position of the destination (base) bits
//   S = shift distance from destination to source bits (S >= 0)
template <int B, int M, int S>
__host__ __device__ __forceinline__ constexpr
int swizzle(int offset) {
    static_assert(B >= 0 && M >= 0 && S >= 0, "B, M, S must be non-negative");
    constexpr int bit_msk = (1 << B) - 1;
    constexpr int zzz_msk = bit_msk << (M + S);
    return offset ^ ((offset & zzz_msk) >> S);
}