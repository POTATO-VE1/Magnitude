// simd_neon.c — ARM64 NEON-accelerated batch distance functions.
#include "simd_neon.h"
#include <arm_neon.h>
#include <stdint.h>

void l2_batch_neon(const float* query, const float* matrix,
                   int n, int dim, float* results) {
    int vec_width = 4;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (int j = 0; j < aligned_dim; j += vec_width) {
            float32x4_t q = vld1q_f32(query + j);
            float32x4_t r = vld1q_f32(row + j);
            float32x4_t diff = vsubq_f32(q, r);
            acc = vfmaq_f32(acc, diff, diff);
        }

        float sum = vaddvq_f32(acc);

        for (int j = aligned_dim; j < dim; j++) {
            float d = query[j] - row[j];
            sum += d * d;
        }
        results[i] = sum;
    }
}

void dot_batch_neon(const float* query, const float* matrix,
                    int n, int dim, float* results) {
    int vec_width = 4;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (int j = 0; j < aligned_dim; j += vec_width) {
            float32x4_t q = vld1q_f32(query + j);
            float32x4_t r = vld1q_f32(row + j);
            acc = vfmaq_f32(acc, q, r);
        }

        float sum = vaddvq_f32(acc);

        for (int j = aligned_dim; j < dim; j++) {
            sum += query[j] * row[j];
        }
        results[i] = sum;
    }
}

void cosine_batch_neon(const float* query, const float* matrix,
                       int n, int dim, float query_norm, float* results) {
    int vec_width = 4;
    int aligned_dim = (dim / vec_width) * vec_width;

    for (int i = 0; i < n; i++) {
        const float* row = matrix + (size_t)i * dim;
        float32x4_t dot_acc = vdupq_n_f32(0.0f);
        float32x4_t norm_acc = vdupq_n_f32(0.0f);

        for (int j = 0; j < aligned_dim; j += vec_width) {
            float32x4_t q = vld1q_f32(query + j);
            float32x4_t r = vld1q_f32(row + j);
            dot_acc = vfmaq_f32(dot_acc, q, r);
            norm_acc = vfmaq_f32(norm_acc, r, r);
        }

        float dot = vaddvq_f32(dot_acc);
        float norm_b = vaddvq_f32(norm_acc);

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
