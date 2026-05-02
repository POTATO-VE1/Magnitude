package distance

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── Helpers ──────────────────────────────────────────────────────────────────

const floatTol = 1e-5 // tolerance for float32 comparisons

func assertFloat32(t *testing.T, expected, actual float32, msgAndArgs ...any) {
	t.Helper()
	assert.InDelta(t, float64(expected), float64(actual), floatTol, msgAndArgs...)
}

// ── L2Squared ────────────────────────────────────────────────────────────────

func TestL2Squared_ZeroDistance(t *testing.T) {
	a := []float32{1, 2, 3}
	assertFloat32(t, 0.0, L2Squared(a, a), "identical vectors must have zero distance")
}

func TestL2Squared_KnownValue(t *testing.T) {
	// (3,4) vs (0,0): L2² = 9 + 16 = 25
	a := []float32{3, 4}
	b := []float32{0, 0}
	assertFloat32(t, 25.0, L2Squared(a, b))
}

func TestL2Squared_UnitVectors(t *testing.T) {
	// (1,0) vs (0,1): L2² = 1 + 1 = 2
	a := []float32{1, 0}
	b := []float32{0, 1}
	assertFloat32(t, 2.0, L2Squared(a, b))
}

func TestL2Squared_NegativeComponents(t *testing.T) {
	a := []float32{-1, -2, -3}
	b := []float32{1, 2, 3}
	// diffs: 2,4,6 → squares: 4,16,36 → sum: 56
	assertFloat32(t, 56.0, L2Squared(a, b))
}

func TestL2Squared_HighDimensional(t *testing.T) {
	// 768-dim vectors of all 1s vs all 2s: each diff = 1, sum = 768
	d := 768
	a := make([]float32, d)
	b := make([]float32, d)
	for i := range a {
		a[i] = 1.0
		b[i] = 2.0
	}
	assertFloat32(t, float32(d), L2Squared(a, b))
}

func TestL2_KnownValue(t *testing.T) {
	// (3,4) vs (0,0): L2 = sqrt(25) = 5
	a := []float32{3, 4}
	b := []float32{0, 0}
	assertFloat32(t, 5.0, L2(a, b))
}

// ── Cosine ───────────────────────────────────────────────────────────────────

func TestCosine_IdenticalVectors(t *testing.T) {
	// Identical vectors → cosine similarity = 1 → distance = 0
	a := []float32{1, 2, 3}
	assertFloat32(t, 0.0, Cosine(a, a), "identical vectors → cosine distance = 0")
}

func TestCosine_OrthogonalVectors(t *testing.T) {
	// (1,0) ⊥ (0,1) → similarity = 0 → distance = 1
	a := []float32{1, 0}
	b := []float32{0, 1}
	assertFloat32(t, 1.0, Cosine(a, b), "orthogonal vectors → cosine distance = 1")
}

func TestCosine_OppositeVectors(t *testing.T) {
	// (1,0) vs (-1,0) → similarity = -1 → distance = 2
	a := []float32{1, 0}
	b := []float32{-1, 0}
	assertFloat32(t, 2.0, Cosine(a, b), "opposite vectors → cosine distance = 2")
}

func TestCosine_KnownAngle(t *testing.T) {
	// Two unit vectors at 60°: cosine_similarity = cos(60°) = 0.5 → distance = 0.5
	// (1, 0) vs (0.5, sqrt(3)/2) — both unit vectors, angle = 60°
	a := []float32{1, 0}
	b := []float32{0.5, float32(math.Sqrt(3) / 2)}
	assertFloat32(t, 0.5, Cosine(a, b), "60° apart → cosine distance = 0.5")
}

func TestCosine_ZeroVector(t *testing.T) {
	// Zero magnitude → undefined → should return 1.0 (neutral)
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	assertFloat32(t, 1.0, Cosine(a, b), "zero vector → neutral distance = 1.0")
}

func TestCosine_ScaleInvariant(t *testing.T) {
	// Cosine distance is scale-invariant: (1,2,3) vs (2,4,6) → distance = 0
	a := []float32{1, 2, 3}
	b := []float32{2, 4, 6}
	assertFloat32(t, 0.0, Cosine(a, b), "parallel vectors (scaled) → distance = 0")
}

func TestCosineSimilarity_Complement(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	// Cosine distance = 1 → similarity = 0
	assertFloat32(t, 0.0, CosineSimilarity(a, b))
}

// ── DotProduct ───────────────────────────────────────────────────────────────

func TestDotProduct_KnownValue(t *testing.T) {
	// [1,2,3] · [4,5,6] = 4 + 10 + 18 = 32
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	assertFloat32(t, 32.0, DotProduct(a, b))
}

func TestDotProduct_Orthogonal(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	assertFloat32(t, 0.0, DotProduct(a, b), "orthogonal vectors → dot product = 0")
}

func TestDotProduct_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	assertFloat32(t, 0.0, DotProduct(a, b), "zero vector → dot product = 0")
}

func TestDotProduct_UnitVectorsEqualsCosine(t *testing.T) {
	// For unit vectors: dot product = cosine similarity
	a := []float32{1, 0}
	b := []float32{0.5, float32(math.Sqrt(3) / 2)}
	dot := DotProduct(a, b)
	cosSimil := CosineSimilarity(a, b)
	assertFloat32(t, cosSimil, dot, "dot product on unit vectors should equal cosine similarity")
}

func TestDotDistance_NegatesDot(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	assertFloat32(t, -32.0, DotDistance(a, b))
}

// ── Manhattan ────────────────────────────────────────────────────────────────

func TestManhattan_ZeroDistance(t *testing.T) {
	a := []float32{1, 2, 3}
	assertFloat32(t, 0.0, Manhattan(a, a), "identical vectors → zero Manhattan distance")
}

func TestManhattan_KnownValue(t *testing.T) {
	// |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	assertFloat32(t, 9.0, Manhattan(a, b))
}

func TestManhattan_NegativeComponents(t *testing.T) {
	// |-3 - 3| + |-4 - 4| = 6 + 8 = 14
	a := []float32{-3, -4}
	b := []float32{3, 4}
	assertFloat32(t, 14.0, Manhattan(a, b))
}

// ── GetDistanceFunc ───────────────────────────────────────────────────────────

func TestGetDistanceFunc_ValidMetrics(t *testing.T) {
	for _, metric := range []string{"l2", "cosine", "dot", "manhattan"} {
		fn, err := GetDistanceFunc(metric)
		require.NoError(t, err, "metric %q should be valid", metric)
		require.NotNil(t, fn, "metric %q should return non-nil function", metric)
	}
}

func TestGetDistanceFunc_InvalidMetric(t *testing.T) {
	_, err := GetDistanceFunc("euclidean")
	require.Error(t, err, "unknown metric should return error")
}

func TestGetDistanceFunc_L2Works(t *testing.T) {
	fn, _ := GetDistanceFunc("l2")
	a := []float32{3, 4}
	b := []float32{0, 0}
	assertFloat32(t, 25.0, fn(a, b))
}

// ── Normalize ─────────────────────────────────────────────────────────────────

func TestNormalize_UnitLength(t *testing.T) {
	a := []float32{3, 4}
	n := Normalize(a)
	// |n| should be 1.0
	mag := float32(math.Sqrt(float64(n[0]*n[0] + n[1]*n[1])))
	assertFloat32(t, 1.0, mag, "normalized vector should have unit length")
}

func TestNormalize_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	n := Normalize(a)
	// Should return zero vector unchanged
	assertFloat32(t, 0.0, n[0])
	assertFloat32(t, 0.0, n[1])
	assertFloat32(t, 0.0, n[2])
}

func TestNormalize_DoesNotModifyOriginal(t *testing.T) {
	a := []float32{3, 4}
	original := []float32{3, 4}
	_ = Normalize(a)
	assert.Equal(t, original[0], a[0], "Normalize should not modify original")
	assert.Equal(t, original[1], a[1], "Normalize should not modify original")
}

func TestNormalizeInPlace_ModifiesOriginal(t *testing.T) {
	a := []float32{3, 4}
	NormalizeInPlace(a)
	mag := float32(math.Sqrt(float64(a[0]*a[0] + a[1]*a[1])))
	assertFloat32(t, 1.0, mag, "NormalizeInPlace should make vector unit length in place")
}

// ── ScoreFromDistance ─────────────────────────────────────────────────────────

func TestScoreFromDistance_Cosine(t *testing.T) {
	// cosine distance = 0 → score = 1 (perfect match)
	assertFloat32(t, 1.0, ScoreFromDistance(0.0, "cosine"))
	// cosine distance = 1 → score = 0.5 (orthogonal)
	assertFloat32(t, 0.5, ScoreFromDistance(1.0, "cosine"))
	// cosine distance = 2 → score = 0 (opposite)
	assertFloat32(t, 0.0, ScoreFromDistance(2.0, "cosine"))
}

func TestScoreFromDistance_L2(t *testing.T) {
	// L2² = 0 → score = 1/(1+0) = 1
	assertFloat32(t, 1.0, ScoreFromDistance(0.0, "l2"))
}

func TestScoreFromDistance_UnknownMetric(t *testing.T) {
	// Unknown metric → score = 0 (safe default)
	assertFloat32(t, 0.0, ScoreFromDistance(0.5, "unknown"))
}

// ── Benchmarks ────────────────────────────────────────────────────────────────
// These become the correctness oracle for future SIMD implementations in Phase 5.
// Run with: go test -bench=. -benchmem ./internal/distance/

func BenchmarkL2Squared_128d(b *testing.B) {
	a, bv := makeVectors(128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Squared(a, bv)
	}
}

func BenchmarkL2Squared_768d(b *testing.B) {
	a, bv := makeVectors(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Squared(a, bv)
	}
}

func BenchmarkL2Squared_1536d(b *testing.B) {
	a, bv := makeVectors(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Squared(a, bv)
	}
}

func BenchmarkCosine_768d(b *testing.B) {
	a, bv := makeVectors(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Cosine(a, bv)
	}
}

func BenchmarkDotProduct_768d(b *testing.B) {
	a, bv := makeVectors(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(a, bv)
	}
}

func BenchmarkManhattan_768d(b *testing.B) {
	a, bv := makeVectors(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Manhattan(a, bv)
	}
}

// makeVectors creates two reproducible test vectors of dimension d.
func makeVectors(d int) ([]float32, []float32) {
	a := make([]float32, d)
	b := make([]float32, d)
	for i := range a {
		a[i] = float32(i) * 0.001
		b[i] = float32(d-i) * 0.001
	}
	return a, b
}
