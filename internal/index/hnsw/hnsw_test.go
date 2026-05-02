package hnsw

import (
	"context"
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/veda/vectordb/internal/index"
	"github.com/veda/vectordb/internal/index/flat"
)

func newTestHNSW(t *testing.T, dim int) *HNSWIndex {
	t.Helper()
	idx, err := NewHNSWIndex(dim, 16, 200, 50, "l2")
	require.NoError(t, err)
	return idx
}

func TestNewHNSW(t *testing.T) {
	idx := newTestHNSW(t, 128)
	assert.Equal(t, 0, idx.Len())
}

func TestNewHNSW_InvalidDimension(t *testing.T) {
	_, err := NewHNSWIndex(0, 16, 200, 50, "l2")
	require.Error(t, err)
}

func TestNewHNSW_InvalidMetric(t *testing.T) {
	_, err := NewHNSWIndex(128, 16, 200, 50, "invalid")
	require.Error(t, err)
}

func TestInsert_Basic(t *testing.T) {
	idx := newTestHNSW(t, 4)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	require.NoError(t, idx.Insert(2, []float32{0, 1, 0, 0}))
	assert.Equal(t, 2, idx.Len())
}

func TestInsert_DuplicateID(t *testing.T) {
	idx := newTestHNSW(t, 4)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	err := idx.Insert(1, []float32{0, 1, 0, 0})
	require.Error(t, err)
}

func TestInsert_DimensionMismatch(t *testing.T) {
	idx := newTestHNSW(t, 4)
	err := idx.Insert(1, []float32{1, 0})
	require.Error(t, err)
}

func TestSearch_ExactMatch(t *testing.T) {
	idx := newTestHNSW(t, 4)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})
	idx.Insert(3, []float32{0, 0, 1, 0})

	results, err := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 1, 0)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.InDelta(t, 0.0, results[0].Distance, 1e-6)
}

func TestSearch_TopKOrdering(t *testing.T) {
	idx := newTestHNSW(t, 2)
	idx.Insert(1, []float32{0, 0})
	idx.Insert(2, []float32{1, 0})
	idx.Insert(3, []float32{2, 0})
	idx.Insert(4, []float32{3, 0})
	idx.Insert(5, []float32{4, 0})

	results, err := idx.Search(context.Background(), []float32{0, 0}, 3, 0)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// Results must be sorted by distance ascending
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i-1].Distance, results[i].Distance,
			"results should be sorted ascending by distance")
	}
}

func TestSearch_EmptyIndex(t *testing.T) {
	idx := newTestHNSW(t, 4)
	results, err := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 5, 0)
	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestSearch_KLargerThanN(t *testing.T) {
	idx := newTestHNSW(t, 2)
	idx.Insert(1, []float32{1, 0})
	idx.Insert(2, []float32{0, 1})

	results, err := idx.Search(context.Background(), []float32{0, 0}, 10, 0)
	require.NoError(t, err)
	assert.Len(t, results, 2)
}

func TestSearch_EfSearchOverride(t *testing.T) {
	idx := newTestHNSW(t, 32)
	rng := rand.New(rand.NewSource(42))
	for i := uint64(0); i < 500; i++ {
		idx.Insert(i, randomVector(rng, 32))
	}

	query := randomVector(rng, 32)

	// Low ef → faster but potentially lower recall
	results1, err := idx.Search(context.Background(), query, 10, 20)
	require.NoError(t, err)

	// High ef → slower but higher recall
	results2, err := idx.Search(context.Background(), query, 10, 200)
	require.NoError(t, err)

	assert.Len(t, results1, 10)
	assert.Len(t, results2, 10)
}

func TestDelete_Basic(t *testing.T) {
	idx := newTestHNSW(t, 4)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})
	idx.Insert(3, []float32{0, 0, 1, 0})

	require.NoError(t, idx.Delete(2))
	assert.Equal(t, 2, idx.Len())

	// Search should not return deleted vector
	results, _ := idx.Search(context.Background(), []float32{0, 1, 0, 0}, 3, 0)
	for _, r := range results {
		assert.NotEqual(t, uint64(2), r.ID, "deleted vector should not appear")
	}
}

func TestDelete_NotFound(t *testing.T) {
	idx := newTestHNSW(t, 4)
	err := idx.Delete(999)
	require.Error(t, err)
}

func TestRebuild(t *testing.T) {
	idx := newTestHNSW(t, 4)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})
	idx.Insert(3, []float32{0, 0, 1, 0})

	idx.Delete(2)
	require.NoError(t, idx.Rebuild())
	assert.Equal(t, 2, idx.Len())

	// Should still find the non-deleted vectors
	results, _ := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 2, 0)
	require.Len(t, results, 2)
}

func TestRecallAtK_HNSW(t *testing.T) {
	dim := 32
	n := 5000

	rng := rand.New(rand.NewSource(42))
	hnswIdx := newTestHNSW(t, dim)
	flatIdx, _ := flat.NewFlatIndex(dim, "l2")

	for i := uint64(0); i < uint64(n); i++ {
		v := randomVector(rng, dim)
		require.NoError(t, hnswIdx.Insert(i, v))
		require.NoError(t, flatIdx.Insert(i, v))
	}

	numQueries := 50
	var totalRecall float64
	ctx := context.Background()

	for q := 0; q < numQueries; q++ {
		query := randomVector(rng, dim)

		truth, err := flatIdx.Search(ctx, query, 10, 0)
		require.NoError(t, err)

		// Use higher ef for better recall in test
		approx, err := hnswIdx.Search(ctx, query, 10, 100)
		require.NoError(t, err)

		recall := index.RecallAtK(truth, approx)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("HNSW Recall@10 = %.4f (n=%d, M=%d, efSearch=%d)", avgRecall, n, 16, 100)
	assert.GreaterOrEqual(t, avgRecall, 0.90, "HNSW recall@10 should be >= 0.90")
}

func TestConcurrent_InsertAndSearch(t *testing.T) {
	idx := newTestHNSW(t, 8)
	ctx := context.Background()

	// Insert initial batch
	rng := rand.New(rand.NewSource(42))
	for i := uint64(0); i < 100; i++ {
		idx.Insert(i, randomVector(rng, 8))
	}

	var wg sync.WaitGroup

	// Concurrent searches
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := randomVector(rand.New(rand.NewSource(99)), 8)
			results, err := idx.Search(ctx, q, 5, 0)
			assert.NoError(t, err)
			assert.LessOrEqual(t, len(results), 5)
		}()
	}
	wg.Wait()
}

func TestHNSW_ManyInserts(t *testing.T) {
	idx := newTestHNSW(t, 16)
	rng := rand.New(rand.NewSource(42))
	n := 2000
	for i := 0; i < n; i++ {
		require.NoError(t, idx.Insert(uint64(i), randomVector(rng, 16)))
	}
	assert.Equal(t, n, idx.Len())

	// Search should return valid results
	results, err := idx.Search(context.Background(), randomVector(rng, 16), 10, 0)
	require.NoError(t, err)
	assert.Len(t, results, 10)
}

func TestHNSW_CosineMetric(t *testing.T) {
	idx, err := NewHNSWIndex(2, 16, 200, 50, "cosine")
	require.NoError(t, err)

	idx.Insert(1, []float32{1, 0})
	idx.Insert(2, []float32{0, 1})
	idx.Insert(3, []float32{-1, 0})

	results, err := idx.Search(context.Background(), []float32{1, 0}, 3, 0)
	require.NoError(t, err)
	require.Len(t, results, 3)
	assert.Equal(t, uint64(1), results[0].ID)
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

func BenchmarkHNSW_Insert_128D(b *testing.B) {
	idx, _ := NewHNSWIndex(128, 16, 200, 50, "l2")
	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = randomVector(rng, 128)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Insert(uint64(i), vecs[i])
	}
}

func BenchmarkHNSW_Search_10K_128D(b *testing.B) {
	benchHNSWSearch(b, 10000, 128, 50)
}

func BenchmarkHNSW_Search_10K_128D_ef100(b *testing.B) {
	benchHNSWSearch(b, 10000, 128, 100)
}

func benchHNSWSearch(b *testing.B, n, dim, ef int) {
	b.Helper()
	rng := rand.New(rand.NewSource(42))
	idx, _ := NewHNSWIndex(dim, 16, 200, ef, "l2")

	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVector(rng, dim))
	}

	query := randomVector(rng, dim)
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, 10, ef)
	}
}

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}
