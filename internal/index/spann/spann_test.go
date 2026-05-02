package spann

import (
	"context"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/veda/vectordb/internal/index"
	"github.com/veda/vectordb/internal/index/flat"
)

func newTestSPANN(t *testing.T, dim int) *SPANNIndex {
	t.Helper()
	idx, err := NewSPANNIndex(dim, 16, 4, "l2")
	require.NoError(t, err)
	return idx
}

func TestNewSPANN(t *testing.T) {
	idx := newTestSPANN(t, 32)
	assert.Equal(t, 0, idx.Len())
}

func TestNewSPANN_InvalidDim(t *testing.T) {
	_, err := NewSPANNIndex(0, 16, 4, "l2")
	require.Error(t, err)
}

func TestNewSPANN_InvalidMetric(t *testing.T) {
	_, err := NewSPANNIndex(32, 16, 4, "invalid")
	require.Error(t, err)
}

func TestInsert_BeforeBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	require.NoError(t, idx.Insert(2, []float32{0, 1, 0, 0}))
	assert.Equal(t, 2, idx.Len())
}

func TestInsert_DuplicateID(t *testing.T) {
	idx := newTestSPANN(t, 4)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	err := idx.Insert(1, []float32{0, 1, 0, 0})
	require.Error(t, err)
}

func TestInsert_DimensionMismatch(t *testing.T) {
	idx := newTestSPANN(t, 4)
	err := idx.Insert(1, []float32{1, 0})
	require.Error(t, err)
}

func TestSearch_BeforeBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})
	idx.Insert(3, []float32{0, 0, 1, 0})

	results, err := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 2, 0)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, uint64(1), results[0].ID)
}

func TestSearch_AfterBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	rng := rand.New(rand.NewSource(42))
	n := 200
	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVector(rng, 4))
	}

	require.NoError(t, idx.Rebuild())

	results, err := idx.Search(context.Background(), randomVector(rng, 4), 5, 4)
	require.NoError(t, err)
	assert.Len(t, results, 5)

	// Distance ordering
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i-1].Distance, results[i].Distance)
	}
}

func TestSearch_EmptyIndex(t *testing.T) {
	idx := newTestSPANN(t, 4)
	results, err := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 5, 0)
	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestDelete_BeforeBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})

	require.NoError(t, idx.Delete(1))
	assert.Equal(t, 1, idx.Len())

	results, _ := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 2, 0)
	for _, r := range results {
		assert.NotEqual(t, uint64(1), r.ID)
	}
}

func TestDelete_AfterBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		idx.Insert(uint64(i), randomVector(rng, 4))
	}
	idx.Rebuild()

	require.NoError(t, idx.Delete(0))
	results, _ := idx.Search(context.Background(), randomVector(rng, 4), 10, 4)
	for _, r := range results {
		assert.NotEqual(t, uint64(0), r.ID)
	}
}

func TestDelete_NotFound(t *testing.T) {
	idx := newTestSPANN(t, 4)
	err := idx.Delete(999)
	require.Error(t, err)
}

func TestRebuild(t *testing.T) {
	idx := newTestSPANN(t, 8)
	rng := rand.New(rand.NewSource(42))
	n := 500
	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVector(rng, 8))
	}

	require.NoError(t, idx.Rebuild())
	assert.Equal(t, n, idx.Len())
	assert.True(t, idx.built)
}

func TestInsert_AfterBuild(t *testing.T) {
	idx := newTestSPANN(t, 4)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		idx.Insert(uint64(i), randomVector(rng, 4))
	}
	idx.Rebuild()

	// Insert new vectors after build — should go to nearest centroid
	require.NoError(t, idx.Insert(1000, randomVector(rng, 4)))
	assert.Equal(t, 101, idx.Len())

	// Should be searchable
	results, err := idx.Search(context.Background(), randomVector(rng, 4), 5, 4)
	require.NoError(t, err)
	assert.Greater(t, len(results), 0)
}

func TestRecallAtK_SPANN(t *testing.T) {
	dim := 32
	n := 2000

	rng := rand.New(rand.NewSource(42))
	spannIdx, err := NewSPANNIndex(dim, 48, 12, "l2")
	require.NoError(t, err)
	flatIdx, _ := flat.NewFlatIndex(dim, "l2")

	for i := uint64(0); i < uint64(n); i++ {
		v := randomVector(rng, dim)
		require.NoError(t, spannIdx.Insert(i, v))
		require.NoError(t, flatIdx.Insert(i, v))
	}

	require.NoError(t, spannIdx.Rebuild())

	numQueries := 30
	var totalRecall float64
	ctx := context.Background()

	for q := 0; q < numQueries; q++ {
		query := randomVector(rng, dim)

		truth, err := flatIdx.Search(ctx, query, 10, 0)
		require.NoError(t, err)

		approx, err := spannIdx.Search(ctx, query, 10, 12)
		require.NoError(t, err)

		recall := index.RecallAtK(truth, approx)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("SPANN Recall@10 = %.4f (n=%d, K=%d, nprobe=%d)", avgRecall, n, 48, 12)
	assert.GreaterOrEqual(t, avgRecall, 0.65, "SPANN recall@10 should be >= 0.65")
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

func BenchmarkSPANN_Search_10K_32D(b *testing.B) {
	dim := 32
	n := 10000
	rng := rand.New(rand.NewSource(42))

	idx, _ := NewSPANNIndex(dim, 64, 5, "l2")
	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVector(rng, dim))
	}
	idx.Rebuild()

	query := randomVector(rng, dim)
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, 10, 5)
	}
}

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}
