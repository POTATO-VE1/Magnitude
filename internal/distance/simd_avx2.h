// simd_avx2.h — declarations for AVX2 SIMD functions
#ifndef SIMD_AVX2_H
#define SIMD_AVX2_H

void l2_batch_avx2(const float* query, const float* matrix,
                   int n, int dim, float* results);

void dot_batch_avx2(const float* query, const float* matrix,
                    int n, int dim, float* results);

void cosine_batch_avx2(const float* query, const float* matrix,
                       int n, int dim, float query_norm, float* results);

#endif
