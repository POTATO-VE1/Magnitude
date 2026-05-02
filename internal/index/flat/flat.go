// Package flat implements a brute-force flat index for exact nearest-neighbor search.
// Every query computes the distance to every vector in the index — O(N*D) time.
// This is the correctness oracle for all other index types.
package flat

import (
	"context"
	"sync"

	"github.com/veda/vectordb/internal/distance"
	vdberrors "github.com/veda/vectordb/internal/errors"
	"github.com/veda/vectordb/internal/index"
)

const defaultInitialCapacity = 1024

// FlatIndex is a brute-force exact nearest-neighbor index.
// It stores all vectors in a single contiguous []float32 slice (row-major)
// for cache-friendly sequential scans and potential BLAS acceleration.
//
// Concurrency: Insert/Delete acquire a write lock; Search acquires a read lock.
// Multiple searches run in parallel. Writes are serialized.
type FlatIndex struct {
	mu       sync.RWMutex
	dim      int
	metric   string
	distFn   distance.DistanceFunc
	capacity int
	count    int
	vectors  []float32        // shape [capacity * dim], row-major
	ids      []uint64         // shape [capacity], id at row i
	idToRow  map[uint64]int   // id → row index, O(1) lookup
}

// NewFlatIndex creates a new flat index for vectors of the given dimension.
// metric must be one of: "l2", "cosine", "dot", "manhattan".
func NewFlatIndex(dim int, metric string) (*FlatIndex, error) {
	if dim <= 0 {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch, "dimension must be > 0, got %d", dim)
	}
	distFn, err := distance.GetDistanceFunc(metric)
	if err != nil {
		return nil, err
	}
	return &FlatIndex{
		dim:      dim,
		metric:   metric,
		distFn:   distFn,
		capacity: defaultInitialCapacity,
		count:    0,
		vectors:  make([]float32, defaultInitialCapacity*dim),
		ids:      make([]uint64, defaultInitialCapacity),
		idToRow:  make(map[uint64]int),
	}, nil
}

// Dim returns the vector dimension this index was created for.
func (idx *FlatIndex) Dim() int {
	return idx.dim
}

// Metric returns the distance metric name.
func (idx *FlatIndex) Metric() string {
	return idx.metric
}

// Insert adds a vector with the given ID into the index.
// Returns ErrDuplicateID if the ID already exists.
// Returns ErrDimensionMismatch if len(vector) != index dimension.
func (idx *FlatIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != idx.dim {
		return vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"expected dimension %d, got %d", idx.dim, len(vector))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.idToRow[id]; exists {
		return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists", id)
	}

	// Grow if at capacity (amortized O(1) via doubling)
	if idx.count >= idx.capacity {
		idx.grow()
	}

	// Copy vector into flat slice at row idx.count
	row := idx.count
	copy(idx.vectors[row*idx.dim:(row+1)*idx.dim], vector)
	idx.ids[row] = id
	idx.idToRow[id] = row
	idx.count++

	return nil
}

// grow doubles the capacity of the backing arrays.
// Must be called with write lock held.
func (idx *FlatIndex) grow() {
	newCap := idx.capacity * 2
	if newCap == 0 {
		newCap = defaultInitialCapacity
	}

	newVec := make([]float32, newCap*idx.dim)
	copy(newVec, idx.vectors[:idx.count*idx.dim])
	idx.vectors = newVec

	newIDs := make([]uint64, newCap)
	copy(newIDs, idx.ids[:idx.count])
	idx.ids = newIDs

	idx.capacity = newCap
}

// Search performs exact nearest-neighbor search, returning up to k results.
// nprobe is ignored by FlatIndex (always searches everything).
// ctx cancellation is respected between distance computation batches.
func (idx *FlatIndex) Search(ctx context.Context, query []float32, k int, nprobe int) ([]index.SearchResult, error) {
	if len(query) != idx.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), idx.dim)
	}
	if k <= 0 {
		return nil, nil
	}

	idx.mu.RLock()
	n := idx.count
	if n == 0 {
		idx.mu.RUnlock()
		return nil, nil
	}

	// Compute distances from query to all stored vectors
	distances := make([]float32, n)
	liveIDs := make([]uint64, n)
	copy(liveIDs, idx.ids[:n])

	for i := 0; i < n; i++ {
		// Check context cancellation every 10K vectors to avoid unbounded loops
		if i%10000 == 0 {
			if err := ctx.Err(); err != nil {
				idx.mu.RUnlock()
				return nil, err
			}
		}
		row := idx.vectors[i*idx.dim : (i+1)*idx.dim]
		distances[i] = idx.distFn(query, row)
	}
	idx.mu.RUnlock()

	// Top-K selection via max-heap (heap.go)
	results := TopK(distances, liveIDs, k, idx.metric)

	// Populate Score field
	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, idx.metric)
	}

	return results, nil
}

// Delete removes a vector by ID using swap-with-last for O(1) deletion.
// The vector at the target row is overwritten with the last vector,
// and the count is decremented.
func (idx *FlatIndex) Delete(id uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	row, exists := idx.idToRow[id]
	if !exists {
		return vdberrors.Newf(vdberrors.ErrVectorNotFound, "vector ID %d not found", id)
	}

	lastRow := idx.count - 1
	if row != lastRow {
		// Swap target row with last row
		copy(
			idx.vectors[row*idx.dim:(row+1)*idx.dim],
			idx.vectors[lastRow*idx.dim:(lastRow+1)*idx.dim],
		)
		lastID := idx.ids[lastRow]
		idx.ids[row] = lastID
		idx.idToRow[lastID] = row
	}

	// Remove the (now last) row
	delete(idx.idToRow, id)
	idx.count--

	return nil
}

// Len returns the number of live vectors in the index.
func (idx *FlatIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.count
}

// Rebuild is a no-op for FlatIndex. Brute-force search needs no index structure.
func (idx *FlatIndex) Rebuild() error {
	return nil
}

// Flush is a no-op for FlatIndex. In-memory only — no persistence.
func (idx *FlatIndex) Flush() error {
	return nil
}

// GetVector returns a copy of the vector with the given ID.
// Used by IVF for rebuilding and by tests for verification.
func (idx *FlatIndex) GetVector(id uint64) ([]float32, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	row, exists := idx.idToRow[id]
	if !exists {
		return nil, false
	}

	vec := make([]float32, idx.dim)
	copy(vec, idx.vectors[row*idx.dim:(row+1)*idx.dim])
	return vec, true
}

// AllVectors returns copies of all vectors and their IDs.
// Used by IVF for K-Means clustering. Returns (ids, vectors_flat, count).
func (idx *FlatIndex) AllVectors() ([]uint64, []float32, int) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	ids := make([]uint64, idx.count)
	copy(ids, idx.ids[:idx.count])

	vecs := make([]float32, idx.count*idx.dim)
	copy(vecs, idx.vectors[:idx.count*idx.dim])

	return ids, vecs, idx.count
}

// Compile-time interface check
var _ index.Index = (*FlatIndex)(nil)
