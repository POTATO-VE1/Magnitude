package ivf

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

func newTestIVF(t *testing.T, dim, k, nprobe int) *IVFIndex {
	t.Helper()
	idx, err := NewIVFIndex(dim, k, nprobe, "l2", 0.10)
	require.NoError(t, err)
	t.Cleanup(func() { idx.Close() })
	return idx
}

func TestIVF_InsertAndLen(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	require.NoError(t, idx.Insert(2, []float32{0, 1, 0, 0}))
	assert.Equal(t, 2, idx.Len())
}

func TestIVF_DuplicateID(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	require.NoError(t, idx.Insert(1, []float32{1, 0, 0, 0}))
	err := idx.Insert(1, []float32{0, 1, 0, 0})
	require.Error(t, err)
}

func TestIVF_DimensionMismatch(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	err := idx.Insert(1, []float32{1, 0})
	require.Error(t, err)
}

func TestIVF_SearchBeforeRebuild(t *testing.T) {
	// Before rebuild, search should still work via dirty buffer (brute force)
	idx := newTestIVF(t, 4, 4, 2)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})

	results, err := idx.Search(context.Background(), []float32{1, 0, 0, 0}, 2, 0)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, uint64(1), results[0].ID) // exact match first
}

func TestIVF_RebuildAndSearch(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)

	// Insert 100 vectors
	rng := rand.New(rand.NewSource(42))
	for i := uint64(0); i < 100; i++ {
		v := randomVec(rng, 4)
		require.NoError(t, idx.Insert(i, v))
	}

	// Rebuild index
	require.NoError(t, idx.Rebuild())

	// Search should work on rebuilt index
	results, err := idx.Search(context.Background(), []float32{0.5, 0.5, 0.5, 0.5}, 10, 4)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(results), 10)
	assert.Greater(t, len(results), 0)
}

func TestIVF_RecallAtK(t *testing.T) {
	dim := 32
	n := 5000
	k := 8
	nprobe := 4

	rng := rand.New(rand.NewSource(99))
	ivfIdx := newTestIVF(t, dim, k, nprobe)
	flatIdx, _ := flat.NewFlatIndex(dim, "l2")

	// Insert same vectors into both
	for i := uint64(0); i < uint64(n); i++ {
		v := randomVec(rng, dim)
		require.NoError(t, ivfIdx.Insert(i, v))
		require.NoError(t, flatIdx.Insert(i, v))
	}

	// Build IVF
	require.NoError(t, ivfIdx.Rebuild())

	// Run 50 queries and compute recall
	numQueries := 50
	var totalRecall float64
	for q := 0; q < numQueries; q++ {
		query := randomVec(rng, dim)
		ctx := context.Background()

		truth, err := flatIdx.Search(ctx, query, 10, 0)
		require.NoError(t, err)

		approx, err := ivfIdx.Search(ctx, query, 10, nprobe)
		require.NoError(t, err)

		recall := index.RecallAtK(truth, approx)
		totalRecall += recall
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("IVF Recall@10 = %.4f (nprobe=%d, k=%d, n=%d)", avgRecall, nprobe, k, n)
	assert.GreaterOrEqual(t, avgRecall, 0.70, "IVF recall@10 should be >= 0.70")
}

func TestIVF_Delete(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})

	// Delete from dirty buffer
	require.NoError(t, idx.Delete(1))
	assert.Equal(t, 1, idx.Len())

	// Rebuild then delete from main index
	idx.Insert(3, []float32{0, 0, 1, 0})
	require.NoError(t, idx.Rebuild())
	require.NoError(t, idx.Delete(2))
	assert.Equal(t, 1, idx.Len())
}

func TestIVF_DeleteNotFound(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	err := idx.Delete(999)
	require.Error(t, err)
}

func TestIVF_ConcurrentInsertSearch(t *testing.T) {
	idx := newTestIVF(t, 8, 4, 2)
	ctx := context.Background()

	// Insert initial batch and rebuild
	rng := rand.New(rand.NewSource(42))
	for i := uint64(0); i < 200; i++ {
		idx.Insert(i, randomVec(rng, 8))
	}
	idx.Rebuild()

	var wg sync.WaitGroup

	// Concurrent inserts (into dirty buffer)
	for i := uint64(200); i < 300; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			idx.Insert(id, randomVec(rand.New(rand.NewSource(int64(id))), 8))
		}(i)
	}

	// Concurrent searches
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := randomVec(rand.New(rand.NewSource(42)), 8)
			results, err := idx.Search(ctx, q, 5, 2)
			assert.NoError(t, err)
			assert.LessOrEqual(t, len(results), 5)
		}()
	}
	wg.Wait()
}

func TestIVF_SearchDimensionMismatch(t *testing.T) {
	idx := newTestIVF(t, 4, 4, 2)
	_, err := idx.Search(context.Background(), []float32{1, 0}, 5, 0)
	require.Error(t, err)
}

func TestIVF_RebuildFewVectors(t *testing.T) {
	// Fewer vectors than clusters — should handle gracefully
	idx := newTestIVF(t, 4, 100, 5)
	idx.Insert(1, []float32{1, 0, 0, 0})
	idx.Insert(2, []float32{0, 1, 0, 0})
	require.NoError(t, idx.Rebuild())
	assert.Equal(t, 2, idx.Len())
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkIVF_Search_10K_128D_nprobe5(b *testing.B) {
	benchIVFSearch(b, 10000, 128, 16, 5)
}

func BenchmarkIVF_Search_100K_128D_nprobe5(b *testing.B) {
	benchIVFSearch(b, 100000, 128, 64, 5)
}

func benchIVFSearch(b *testing.B, n, dim, k, nprobe int) {
	rng := rand.New(rand.NewSource(42))
	idx, _ := NewIVFIndex(dim, k, nprobe, "l2", 0.10)
	defer idx.Close()

	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVec(rng, dim))
	}
	idx.Rebuild()

	query := randomVec(rng, dim)
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, 10, nprobe)
	}
}

func randomVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}
