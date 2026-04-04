#pragma once

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__host__ __device__ 
inline bool is_pow2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}
