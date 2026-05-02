// internal/distance/simd.c
// Compiled automatically by cgo. No separate build step.
#include <stdint.h>
#include <immintrin.h>  // AVX2 intrinsics

// Computes L2 squared distance between query and n vectors.
// dim need NOT be a multiple of 8 — scalar tail loop handles the remainder.
// Inputs do NOT need 32-byte alignment (_loadu_ handles unaligned loads).
void l2_batch_avx2(const float* query, const float* matrix,
                   int n, int dim, float* results) {
    int vec_width = 8;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        __m256 sum = _mm256_setzero_ps();

        for (int j = 0; j < aligned_dim; j += vec_width) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 r = _mm256_loadu_ps(row + j);
            __m256 diff = _mm256_sub_ps(q, r);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        // Horizontal sum: reduce 8 floats → 1 float
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float scalar_sum = _mm_cvtss_f32(lo);

        // Scalar tail — handles dim % 8 remaining elements
        for (int j = aligned_dim; j < dim; j++) {
            float diff = query[j] - row[j];
            scalar_sum += diff * diff;
        }
        results[i] = scalar_sum;
    }
}
