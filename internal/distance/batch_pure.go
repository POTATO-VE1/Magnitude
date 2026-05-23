package distance

import "math"

// ── L2 Squared ──────────────────────────────────────────────────────────────

// L2BatchPure computes L2 squared distance from query to n vectors in matrix.
// 4-wide accumulator pattern — Go compiler auto-vectorizes to VFMADD on amd64,
// FMLA on arm64. Works on ALL architectures, no cgo, no build tags.
func L2BatchPure(query, matrix []float32, n, dim int, results []float32) {
	for i := 0; i < n; i++ {
		base := i * dim
		var s0, s1, s2, s3 float32
		j := 0
		for ; j <= dim-4; j += 4 {
			d0 := query[j] - matrix[base+j]
			d1 := query[j+1] - matrix[base+j+1]
			d2 := query[j+2] - matrix[base+j+2]
			d3 := query[j+3] - matrix[base+j+3]
			s0 += d0 * d0
			s1 += d1 * d1
			s2 += d2 * d2
			s3 += d3 * d3
		}
		sum := s0 + s1 + s2 + s3
		for ; j < dim; j++ {
			d := query[j] - matrix[base+j]
			sum += d * d
		}
		results[i] = sum
	}
}

// ── Dot Product ─────────────────────────────────────────────────────────────

// DotBatchPure computes dot product of query with n vectors in matrix.
// Higher = more similar. Returns similarity, not distance.
func DotBatchPure(query, matrix []float32, n, dim int, results []float32) {
	for i := 0; i < n; i++ {
		base := i * dim
		var s0, s1, s2, s3 float32
		j := 0
		for ; j <= dim-4; j += 4 {
			s0 += query[j] * matrix[base+j]
			s1 += query[j+1] * matrix[base+j+1]
			s2 += query[j+2] * matrix[base+j+2]
			s3 += query[j+3] * matrix[base+j+3]
		}
		sum := s0 + s1 + s2 + s3
		for ; j < dim; j++ {
			sum += query[j] * matrix[base+j]
		}
		results[i] = sum
	}
}

// ── Cosine Distance ─────────────────────────────────────────────────────────

// CosineBatchPure computes cosine distance from query to n vectors in matrix.
// Returns distance in [0, 2] where 0 = identical.
// 3 accumulators: dot, normA, normB.
func CosineBatchPure(query, matrix []float32, n, dim int, results []float32) {
	// Precompute query norm
	var queryNorm float32
	for j := 0; j < dim; j++ {
		queryNorm += query[j] * query[j]
	}

	for i := 0; i < n; i++ {
		base := i * dim
		var dot, normB0, normB1, normB2, normB3 float32
		j := 0
		for ; j <= dim-4; j += 4 {
			q0, q1, q2, q3 := query[j], query[j+1], query[j+2], query[j+3]
			r0, r1, r2, r3 := matrix[base+j], matrix[base+j+1], matrix[base+j+2], matrix[base+j+3]
			dot += q0*r0 + q1*r1 + q2*r2 + q3*r3
			normB0 += r0 * r0
			normB1 += r1 * r1
			normB2 += r2 * r2
			normB3 += r3 * r3
		}
		normB := normB0 + normB1 + normB2 + normB3
		for ; j < dim; j++ {
			r := matrix[base+j]
			dot += query[j] * r
			normB += r * r
		}

		if queryNorm == 0 || normB == 0 {
			results[i] = 1.0
			continue
		}
		similarity := dot / float32(math.Sqrt(float64(queryNorm)*float64(normB)))
		if similarity > 1.0 {
			similarity = 1.0
		} else if similarity < -1.0 {
			similarity = -1.0
		}
		results[i] = 1.0 - similarity
	}
}

// ── Manhattan (L1) ──────────────────────────────────────────────────────────

// ManhattanBatchPure computes Manhattan (L1) distance from query to n vectors.
// Uses IEEE754 bit manipulation for branchless abs.
func ManhattanBatchPure(query, matrix []float32, n, dim int, results []float32) {
	const signMask = 0x7FFFFFFF
	for i := 0; i < n; i++ {
		base := i * dim
		var s0, s1, s2, s3 float32
		j := 0
		for ; j <= dim-4; j += 4 {
			d0 := query[j] - matrix[base+j]
			d1 := query[j+1] - matrix[base+j+1]
			d2 := query[j+2] - matrix[base+j+2]
			d3 := query[j+3] - matrix[base+j+3]
			// Branchless abs via IEEE754 bit manipulation
			s0 += math.Float32frombits(math.Float32bits(d0) & signMask)
			s1 += math.Float32frombits(math.Float32bits(d1) & signMask)
			s2 += math.Float32frombits(math.Float32bits(d2) & signMask)
			s3 += math.Float32frombits(math.Float32bits(d3) & signMask)
		}
		sum := s0 + s1 + s2 + s3
		for ; j < dim; j++ {
			d := query[j] - matrix[base+j]
			sum += math.Float32frombits(math.Float32bits(d) & signMask)
		}
		results[i] = sum
	}
}
