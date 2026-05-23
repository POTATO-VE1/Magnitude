// simd_avx2.c — AVX2-accelerated batch distance functions.
// Compiled with -mavx2 -mfma by the go build system via CGO_CFLAGS.
#include <immintrin.h>
#include <stdint.h>

// l2_batch_avx2 computes L2 squared distance from query to n vectors.
// Processes 8 floats per iteration using 256-bit YMM registers.
void l2_batch_avx2(const float* query, const float* matrix,
                   int n, int dim, float* results) {
    int vec_width = 8;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        __m256 acc = _mm256_setzero_ps();

        // Vectorized loop: 8 floats per iteration
        for (int j = 0; j < aligned_dim; j += vec_width) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 r = _mm256_loadu_ps(row + j);
            __m256 diff = _mm256_sub_ps(q, r);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        // Horizontal sum: reduce 8 floats to 1
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        // Scalar tail for remaining elements (dim % 8)
        for (int j = aligned_dim; j < dim; j++) {
            float diff = query[j] - row[j];
            sum += diff * diff;
        }
        results[i] = sum;
    }
}

// dot_batch_avx2 computes dot product of query with n vectors.
void dot_batch_avx2(const float* query, const float* matrix,
                    int n, int dim, float* results) {
    int vec_width = 8;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        __m256 acc = _mm256_setzero_ps();

        for (int j = 0; j < aligned_dim; j += vec_width) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 r = _mm256_loadu_ps(row + j);
            acc = _mm256_fmadd_ps(q, r, acc);
        }

        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        for (int j = aligned_dim; j < dim; j++) {
            sum += query[j] * row[j];
        }
        results[i] = sum;
    }
}

// cosine_batch_avx2 computes cosine distance from query to n vectors.
// query_norm must be precomputed: sum(query[i]^2).
void cosine_batch_avx2(const float* query, const float* matrix,
                       int n, int dim, float query_norm, float* results) {
    int vec_width = 8;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        __m256 dot_acc = _mm256_setzero_ps();
        __m256 norm_acc = _mm256_setzero_ps();

        for (int j = 0; j < aligned_dim; j += vec_width) {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 r = _mm256_loadu_ps(row + j);
            dot_acc = _mm256_fmadd_ps(q, r, dot_acc);
            norm_acc = _mm256_fmadd_ps(r, r, norm_acc);
        }

        // Horizontal sum for dot
        __m128 hi = _mm256_extractf128_ps(dot_acc, 1);
        __m128 lo = _mm256_castps256_ps128(dot_acc);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float dot = _mm_cvtss_f32(s);

        // Horizontal sum for norm
        hi = _mm256_extractf128_ps(norm_acc, 1);
        lo = _mm256_castps256_ps128(norm_acc);
        s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float norm_b = _mm_cvtss_f32(s);

        // Scalar tail
        for (int j = aligned_dim; j < dim; j++) {
            dot += query[j] * row[j];
            norm_b += row[j] * row[j];
        }

        if (query_norm == 0.0f || norm_b == 0.0f) {
            results[i] = 1.0f;
            continue;
        }
        float similarity = dot / __builtin_sqrtf(query_norm * norm_b);
        if (similarity > 1.0f) similarity = 1.0f;
        if (similarity < -1.0f) similarity = -1.0f;
        results[i] = 1.0f - similarity;
    }
}
