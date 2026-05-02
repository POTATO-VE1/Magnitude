package flat

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/veda/vectordb/internal/index"
)

func TestNewFlatIndex(t *testing.T) {
	idx, err := NewFlatIndex(128, "l2")
	require.NoError(t, err)
	assert.Equal(t, 128, idx.Dim())
	assert.Equal(t, "l2", idx.Metric())
	assert.Equal(t, 0, idx.Len())
}

func TestNewFlatIndex_InvalidDimension(t *testing.T) {
	_, err := NewFlatIndex(0, "l2")
	require.Error(t, err)
}

func TestNewFlatIndex_InvalidMetric(t *testing.T) {
	_, err := NewFlatIndex(128, "invalid")
	require.Error(t, err)
}

func TestInsert_Basic(t *testing.T) {
	idx, _ := NewFlatIndex(3, "l2")
	require.NoError(t, idx.Insert(1, []float32{1, 2, 3}))
	assert.Equal(t, 1, idx.Len())
}

func TestInsert_DimensionMismatch(t *testing.T) {
	idx, _ := NewFlatIndex(3, "l2")
	err := idx.Insert(1, []float32{1, 2})
	require.Error(t, err)
	assert.Equal(t, 0, idx.Len())
}

func TestInsert_DuplicateID(t *testing.T) {
	idx, _ := NewFlatIndex(3, "l2")
	require.NoError(t, idx.Insert(1, []float32{1, 2, 3}))
	err := idx.Insert(1, []float32{4, 5, 6})
	require.Error(t, err)
	assert.Equal(t, 1, idx.Len())
}

func TestInsert_CapacityGrowth(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	// Insert more than initial capacity to trigger grow()
	for i := uint64(0); i < 2000; i++ {
		require.NoError(t, idx.Insert(i, []float32{float32(i), float32(i + 1)}))
	}
	assert.Equal(t, 2000, idx.Len())
}

func TestSearch_ExactMatch(t *testing.T) {
	idx, _ := NewFlatIndex(3, "l2")
	idx.Insert(1, []float32{1, 0, 0})
	idx.Insert(2, []float32{0, 1, 0})
	idx.Insert(3, []float32{0, 0, 1})

	results, err := idx.Search(context.Background(), []float32{1, 0, 0}, 1, 0)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.InDelta(t, 0.0, results[0].Distance, 1e-6)
}

func TestSearch_TopKOrdering(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{0, 0})
	idx.Insert(2, []float32{1, 0})
	idx.Insert(3, []float32{2, 0})
	idx.Insert(4, []float32{3, 0})
	idx.Insert(5, []float32{4, 0})

	results, err := idx.Search(context.Background(), []float32{0, 0}, 3, 0)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// Results must be sorted by distance ascending
	assert.Equal(t, uint64(1), results[0].ID)
	assert.Equal(t, uint64(2), results[1].ID)
	assert.Equal(t, uint64(3), results[2].ID)
	assert.True(t, results[0].Distance <= results[1].Distance)
	assert.True(t, results[1].Distance <= results[2].Distance)
}

func TestSearch_KLargerThanN(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{1, 0})
	idx.Insert(2, []float32{0, 1})

	results, err := idx.Search(context.Background(), []float32{0, 0}, 10, 0)
	require.NoError(t, err)
	assert.Len(t, results, 2) // only 2 vectors exist
}

func TestSearch_EmptyIndex(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	results, err := idx.Search(context.Background(), []float32{0, 0}, 5, 0)
	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestSearch_CosineMetric(t *testing.T) {
	idx, _ := NewFlatIndex(2, "cosine")
	idx.Insert(1, []float32{1, 0})    // unit vector along x
	idx.Insert(2, []float32{0, 1})    // unit vector along y
	idx.Insert(3, []float32{-1, 0})   // opposite direction

	results, err := idx.Search(context.Background(), []float32{1, 0}, 3, 0)
	require.NoError(t, err)
	require.Len(t, results, 3)
	// Closest by cosine: ID 1 (same direction), then ID 2 (orthogonal), then ID 3 (opposite)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.Equal(t, uint64(2), results[1].ID)
	assert.Equal(t, uint64(3), results[2].ID)
}

func TestSearch_ContextCancellation(t *testing.T) {
	idx, _ := NewFlatIndex(128, "l2")
	for i := uint64(0); i < 50000; i++ {
		v := make([]float32, 128)
		v[0] = float32(i)
		idx.Insert(i, v)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := idx.Search(ctx, make([]float32, 128), 10, 0)
	require.Error(t, err) // should return context.Canceled
}

func TestDelete_Basic(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{1, 0})
	idx.Insert(2, []float32{0, 1})
	idx.Insert(3, []float32{1, 1})

	require.NoError(t, idx.Delete(2))
	assert.Equal(t, 2, idx.Len())

	// Search should not find deleted vector
	results, _ := idx.Search(context.Background(), []float32{0, 1}, 3, 0)
	for _, r := range results {
		assert.NotEqual(t, uint64(2), r.ID, "deleted vector should not appear in results")
	}
}

func TestDelete_SwapCorrectness(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{1, 0})
	idx.Insert(2, []float32{0, 1})
	idx.Insert(3, []float32{1, 1})

	// Delete the first one — should swap with last
	require.NoError(t, idx.Delete(1))
	assert.Equal(t, 2, idx.Len())

	// Remaining vectors should still be searchable
	results, _ := idx.Search(context.Background(), []float32{0, 1}, 2, 0)
	require.Len(t, results, 2)

	resultIDs := map[uint64]bool{}
	for _, r := range results {
		resultIDs[r.ID] = true
	}
	assert.True(t, resultIDs[2], "ID 2 should be findable")
	assert.True(t, resultIDs[3], "ID 3 should be findable")
}

func TestDelete_NotFound(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	err := idx.Delete(999)
	require.Error(t, err)
}

func TestDelete_LastElement(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{1, 0})
	require.NoError(t, idx.Delete(1))
	assert.Equal(t, 0, idx.Len())
}

func TestGetVector(t *testing.T) {
	idx, _ := NewFlatIndex(3, "l2")
	idx.Insert(42, []float32{1, 2, 3})

	vec, ok := idx.GetVector(42)
	require.True(t, ok)
	assert.Equal(t, []float32{1, 2, 3}, vec)

	_, ok = idx.GetVector(999)
	assert.False(t, ok)
}

func TestRecallAtK_PerfectRecall(t *testing.T) {
	// Flat index must have perfect recall (it's exact search)
	idx, _ := NewFlatIndex(128, "l2")
	rng := rand.New(rand.NewSource(42))

	for i := uint64(0); i < 1000; i++ {
		v := randomVector(rng, 128)
		idx.Insert(i, v)
	}

	query := randomVector(rng, 128)
	results, err := idx.Search(context.Background(), query, 10, 0)
	require.NoError(t, err)

	// For flat index, recall must be exactly 1.0 (it IS the ground truth)
	recall := index.RecallAtK(results, results)
	assert.Equal(t, 1.0, recall)
}

func TestConcurrent_InsertAndSearch(t *testing.T) {
	idx, _ := NewFlatIndex(8, "l2")
	ctx := context.Background()

	var wg sync.WaitGroup
	// Insert 100 vectors concurrently
	for i := uint64(0); i < 100; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			v := make([]float32, 8)
			v[0] = float32(id)
			idx.Insert(id, v)
		}(i)
	}
	wg.Wait()

	assert.Equal(t, 100, idx.Len())

	// Run 50 concurrent searches
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := make([]float32, 8)
			results, err := idx.Search(ctx, q, 5, 0)
			assert.NoError(t, err)
			assert.LessOrEqual(t, len(results), 5)
		}()
	}
	wg.Wait()
}

func TestScore_L2(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(1, []float32{0, 0})
	idx.Insert(2, []float32{3, 4}) // L2² = 25

	results, _ := idx.Search(context.Background(), []float32{0, 0}, 2, 0)
	require.Len(t, results, 2)

	// First result (distance=0) should have score close to 1.0
	assert.InDelta(t, 1.0, results[0].Score, 0.01)
	// Second result (distance=25, sqrt=5) → score = 1/(1+5) ≈ 0.167
	assert.InDelta(t, 1.0/(1.0+5.0), results[1].Score, 0.01)
}

func TestAllVectors(t *testing.T) {
	idx, _ := NewFlatIndex(2, "l2")
	idx.Insert(10, []float32{1, 2})
	idx.Insert(20, []float32{3, 4})

	ids, vecs, count := idx.AllVectors()
	assert.Equal(t, 2, count)
	assert.Len(t, ids, 2)
	assert.Len(t, vecs, 4)
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkFlatSearch_1K_128D(b *testing.B) {
	benchSearch(b, 1000, 128, 10)
}

func BenchmarkFlatSearch_10K_128D(b *testing.B) {
	benchSearch(b, 10000, 128, 10)
}

func BenchmarkFlatSearch_100K_128D(b *testing.B) {
	benchSearch(b, 100000, 128, 10)
}

func BenchmarkFlatSearch_10K_768D(b *testing.B) {
	benchSearch(b, 10000, 768, 10)
}

func BenchmarkFlatInsert_128D(b *testing.B) {
	idx, _ := NewFlatIndex(128, "l2")
	v := make([]float32, 128)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Insert(uint64(i), v)
	}
}

func benchSearch(b *testing.B, n, dim, k int) {
	b.Helper()
	rng := rand.New(rand.NewSource(42))
	idx, _ := NewFlatIndex(dim, "l2")
	for i := 0; i < n; i++ {
		idx.Insert(uint64(i), randomVector(rng, dim))
	}
	query := randomVector(rng, dim)
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, k, 0)
	}
}

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// Verify float tolerance helper
func assertFloat(t *testing.T, expected, actual float64) {
	t.Helper()
	assert.InDelta(t, expected, actual, 1e-5)
}

// Suppress unused import
var _ = math.Abs
