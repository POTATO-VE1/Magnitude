// simd_neon.h — declarations for ARM64 NEON SIMD functions
#ifndef SIMD_NEON_H
#define SIMD_NEON_H

void l2_batch_neon(const float* query, const float* matrix,
                   int n, int dim, float* results);

void dot_batch_neon(const float* query, const float* matrix,
                    int n, int dim, float* results);

void cosine_batch_neon(const float* query, const float* matrix,
                       int n, int dim, float query_norm, float* results);

#endif
