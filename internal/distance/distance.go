// Package distance implements the core vector distance and similarity functions
// used throughout the index layer. All functions are hand-written for performance
// — no BLAS calls in the hot path, as CGO overhead (~90ns/call) would dominate
// at the per-vector level.
//
// SIMD acceleration is added in Phase 5 via assembly/CGO with batch processing.
// Until then, these scalar implementations serve as the correctness oracle.
//
// All functions assume equal-length slices. Callers MUST validate dimensions before
// calling; these functions do NOT check lengths in the hot path.
package distance

import (
	"fmt"
	"math"
)

// DistanceFunc is the function signature for all distance/similarity functions.
// Returns a float32 where lower values indicate closer vectors (for L2, cosine distance, manhattan)
// and higher values indicate closer vectors (for dot product used as similarity).
type DistanceFunc func(a, b []float32) float32

// GetDistanceFunc returns the DistanceFunc for the given metric name.
// Valid names: "l2", "cosine", "dot", "manhattan".
// Returns an error for unknown metric names.
func GetDistanceFunc(name string) (DistanceFunc, error) {
	switch name {
	case "l2":
		return L2Squared, nil
	case "cosine":
		return Cosine, nil
	case "dot":
		return DotProduct, nil
	case "manhattan":
		return Manhattan, nil
	default:
		return nil, fmt.Errorf("distance: unknown metric %q (valid: l2, cosine, dot, manhattan)", name)
	}
}

// L2Squared computes the squared Euclidean distance between vectors a and b.
//
//	d²(a,b) = Σ (a[i] - b[i])²
//
// We return the squared distance rather than the true L2 distance to avoid a
// sqrt() call in the hot path. Since sqrt is monotonic, rankings are preserved.
// The sqrt is only needed when the caller requires the actual distance value
// (e.g., for score normalization). Use L2 (below) for that case.
//
// Precondition: len(a) == len(b). Caller is responsible for dimension validation.
func L2Squared(a, b []float32) float32 {
	var sum float32
	// Loop is intentionally written to allow the compiler to auto-vectorize.
	// go tool objdump -s L2Squared shows VFMADD instructions on x86-64 with AVX2.
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// L2 computes the true Euclidean distance (sqrt of L2Squared).
// Use for final distance reporting only, not in the search hot path.
func L2(a, b []float32) float32 {
	return float32(math.Sqrt(float64(L2Squared(a, b))))
}

// Cosine computes the cosine distance between vectors a and b.
//
//	cosine_distance(a,b) = 1 - cosine_similarity(a,b)
//	cosine_similarity(a,b) = dot(a,b) / (|a| * |b|)
//
// Returns a value in [0, 2]:
//   - 0.0 = identical direction (similarity = 1.0)
//   - 1.0 = orthogonal (similarity = 0.0)
//   - 2.0 = opposite direction (similarity = -1.0)
//
// If either vector has zero magnitude, returns 1.0 (undefined similarity → neutral distance).
// For best performance, pre-normalize vectors to unit length and use DotProduct instead.
// Pre-normalized cosine search reduces each query to a dot product computation.
//
// Precondition: len(a) == len(b). Caller is responsible for dimension validation.
func Cosine(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0 // undefined similarity → neutral distance
	}
	similarity := dot / float32(math.Sqrt(float64(normA)*float64(normB)))
	// Clamp to [-1, 1] to handle floating-point rounding errors
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}
	return 1.0 - similarity
}

// CosineSimilarity computes the raw cosine similarity in [-1, 1].
// Higher = more similar. Use Cosine() for distance-based ranking.
func CosineSimilarity(a, b []float32) float32 {
	return 1.0 - Cosine(a, b)
}

// DotProduct computes the inner product (dot product) of vectors a and b.
//
//	dot(a,b) = Σ a[i] * b[i]
//
// For unit-normalized vectors, dot product equals cosine similarity.
// This is the fastest distance function — one multiply-add per dimension.
// The ChromaDB Rust rewrite uses dot product on pre-normalized HNSW graphs.
//
// Note: DotProduct returns a SIMILARITY (higher = more similar), not a distance.
// To use as a distance for Top-K retrieval, negate the result or use a max-heap.
//
// Precondition: len(a) == len(b). Caller is responsible for dimension validation.
func DotProduct(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// DotDistance converts dot product to a distance metric (lower = more similar).
// Use this when you need a distance-compatible ordering: distance = -dot(a,b).
func DotDistance(a, b []float32) float32 {
	return -DotProduct(a, b)
}

// Manhattan computes the Manhattan (L1) distance between vectors a and b.
//
//	L1(a,b) = Σ |a[i] - b[i]|
//
// Manhattan distance is less sensitive to outlier dimensions than L2.
// Less common in ANN literature but useful for sparse-ish embeddings.
//
// Precondition: len(a) == len(b). Caller is responsible for dimension validation.
func Manhattan(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		sum += diff
	}
	return sum
}

// Normalize returns a new unit-normalized copy of the input vector.
// The original vector is not modified. If the vector has zero magnitude,
// the zero vector is returned unchanged.
//
// Pre-normalizing vectors before insertion and converting cosine search
// to dot product search is a standard production optimization:
// it eliminates the sqrt in Cosine() from the hot path entirely.
func Normalize(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	if norm == 0 {
		result := make([]float32, len(v))
		copy(result, v)
		return result
	}
	scale := float32(1.0 / math.Sqrt(float64(norm)))
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = x * scale
	}
	return result
}

// NormalizeInPlace normalizes vector v in place. Returns v for chaining.
// More efficient than Normalize when the original vector is not needed.
func NormalizeInPlace(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	if norm == 0 {
		return v
	}
	scale := float32(1.0 / math.Sqrt(float64(norm)))
	for i := range v {
		v[i] *= scale
	}
	return v
}

// ScoreFromDistance converts a raw distance value to a normalized similarity
// score in [0, 1] using the specified metric. Higher score = more similar.
//
// This is used to populate SearchResult.Score after search.
//
// Metric-specific behavior:
//   - "l2":       score = 1 / (1 + sqrt(distance))  [distance is L2Squared input]
//   - "cosine":   score = 1 - distance               [distance is already in [0, 2]]
//   - "dot":      score = (distance + 1) / 2         [distance = -dot, mapped to [0,1]]
//   - "manhattan": score = 1 / (1 + distance)
func ScoreFromDistance(distance float32, metric string) float32 {
	switch metric {
	case "l2":
		return 1.0 / (1.0 + float32(math.Sqrt(float64(distance))))
	case "cosine":
		// Cosine distance is in [0, 2]; clamp score to [0, 1]
		score := 1.0 - distance/2.0
		if score < 0 {
			score = 0
		}
		return score
	case "dot":
		// distance = -dot (negated); map to [0,1] assuming dot ∈ [-1, 1]
		return (-distance + 1.0) / 2.0
	case "manhattan":
		return 1.0 / (1.0 + distance)
	default:
		return 0
	}
}
