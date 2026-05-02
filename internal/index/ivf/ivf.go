// Package ivf implements an Inverted File Index (IVF) for approximate nearest-neighbor search.
// Vectors are assigned to Voronoi cells via K-Means++ clustering. Queries probe
// the nprobe nearest centroids, trading recall for sub-linear search time.
package ivf

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/veda/vectordb/internal/distance"
	vdberrors "github.com/veda/vectordb/internal/errors"
	"github.com/veda/vectordb/internal/index"
	"github.com/veda/vectordb/internal/index/flat"
)

// IVFIndex implements approximate nearest-neighbor search using inverted file indexing.
// Vectors are partitioned into K Voronoi cells. Queries probe nprobe cells.
type IVFIndex struct {
	mu sync.RWMutex

	dim     int
	k       int     // number of clusters
	nprobe  int     // default number of clusters to probe
	metric  string
	distFn  distance.DistanceFunc

	// K-Means state
	kmeans      *KMeans
	postingList [][]uint64 // postingList[cluster] = vector IDs in that cluster
	built       bool       // true after first Build/Rebuild

	// Contiguous vector storage (same as FlatIndex: one allocation, GC-friendly)
	vectors  []float32       // shape [capacity * dim], row-major
	ids      []uint64        // shape [capacity]
	idToRow  map[uint64]int  // id → row index
	count    int
	capacity int
	clusterOf map[uint64]int // id → cluster assignment

	// Two-tiered dirty buffer: new inserts go here until rebuild
	dirty          *flat.FlatIndex
	dirtyThreshold int // rebuild when dirty.Len() > this

	// Background rebuilder
	cancel context.CancelFunc
	done   chan struct{}
}

// NewIVFIndex creates a new IVF index.
// k = number of clusters, nprobe = default clusters to probe at search time.
func NewIVFIndex(dim, k, nprobe int, metric string, dirtyThreshold float64) (*IVFIndex, error) {
	if dim <= 0 {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch, "dimension must be > 0, got %d", dim)
	}
	distFn, err := distance.GetDistanceFunc(metric)
	if err != nil {
		return nil, err
	}
	if k <= 0 {
		k = 256
	}
	if nprobe <= 0 {
		nprobe = 5
	}

	dirty, _ := flat.NewFlatIndex(dim, metric)

	initCap := 1024
	idx := &IVFIndex{
		dim:            dim,
		k:              k,
		nprobe:         nprobe,
		metric:         metric,
		distFn:         distFn,
		postingList:    make([][]uint64, k),
		vectors:        make([]float32, initCap*dim),
		ids:            make([]uint64, initCap),
		idToRow:        make(map[uint64]int),
		clusterOf:      make(map[uint64]int),
		count:          0,
		capacity:       initCap,
		dirty:          dirty,
		dirtyThreshold: int(float64(initCap) * dirtyThreshold),
	}
	if idx.dirtyThreshold < 100 {
		idx.dirtyThreshold = 100
	}

	// Start background rebuilder
	ctx, cancel := context.WithCancel(context.Background())
	idx.cancel = cancel
	idx.done = make(chan struct{})
	go idx.backgroundRebuilder(ctx)

	return idx, nil
}

// Dim returns the vector dimension.
func (idx *IVFIndex) Dim() int { return idx.dim }

// Insert adds a vector. Before the first Rebuild, all vectors go into the dirty buffer.
// After Rebuild, new vectors go into the dirty buffer until the next rebuild.
func (idx *IVFIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != idx.dim {
		return vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"expected dimension %d, got %d", idx.dim, len(vector))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Check for duplicates in both main index and dirty buffer
	if _, exists := idx.idToRow[id]; exists {
		return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists", id)
	}
	if _, exists := idx.dirty.GetVector(id); exists {
		return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists in dirty buffer", id)
	}

	// Always insert into dirty buffer; rebuild will merge
	return idx.dirty.Insert(id, vector)
}

// Search queries both the IVF index and the dirty buffer, merging results.
func (idx *IVFIndex) Search(ctx context.Context, query []float32, k int, nprobe int) ([]index.SearchResult, error) {
	if len(query) != idx.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), idx.dim)
	}
	if k <= 0 {
		return nil, nil
	}
	if nprobe <= 0 {
		nprobe = idx.nprobe
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var ivfResults []index.SearchResult

	if idx.built && idx.count > 0 {
		// Find top-nprobe centroids
		topCentroids := idx.kmeans.NearestKCentroids(query, nprobe)

		// Gather candidate IDs from probed posting lists
		var candidateIDs []uint64
		for _, cd := range topCentroids {
			candidateIDs = append(candidateIDs, idx.postingList[cd.Index]...)
		}

		if len(candidateIDs) > 0 {
			// Compute exact distances for candidates
			distances := make([]float32, len(candidateIDs))
			ids := make([]uint64, len(candidateIDs))
			for i, cid := range candidateIDs {
				row, exists := idx.idToRow[cid]
				if !exists {
					continue
				}
				ids[i] = cid
				v := idx.vectors[row*idx.dim : (row+1)*idx.dim]
				distances[i] = idx.distFn(query, v)
			}
			ivfResults = flat.TopK(distances, ids, k, idx.metric)
		}
	}

	// Also search the dirty buffer (brute force on recent inserts)
	dirtyResults, err := idx.dirty.Search(ctx, query, k, 0)
	if err != nil {
		return nil, err
	}

	// Merge IVF + dirty results
	merged := mergeResults(ivfResults, dirtyResults, k)

	// Populate scores
	for i := range merged {
		merged[i].Score = distance.ScoreFromDistance(merged[i].Distance, idx.metric)
	}

	return merged, nil
}

// mergeResults merges two sorted result lists and returns the top-k.
func mergeResults(a, b []index.SearchResult, k int) []index.SearchResult {
	if len(a) == 0 {
		if len(b) > k {
			return b[:k]
		}
		return b
	}
	if len(b) == 0 {
		if len(a) > k {
			return a[:k]
		}
		return a
	}

	// Deduplicate by ID (dirty buffer may contain vectors that are also in IVF)
	seen := make(map[uint64]bool, len(a)+len(b))
	all := make([]index.SearchResult, 0, len(a)+len(b))
	for _, r := range a {
		if !seen[r.ID] {
			seen[r.ID] = true
			all = append(all, r)
		}
	}
	for _, r := range b {
		if !seen[r.ID] {
			seen[r.ID] = true
			all = append(all, r)
		}
	}

	// Use TopK to select best k from merged set
	distances := make([]float32, len(all))
	ids := make([]uint64, len(all))
	for i, r := range all {
		distances[i] = r.Distance
		ids[i] = r.ID
	}
	return flat.TopK(distances, ids, k, "")
}

// Delete removes a vector by ID from either the main index or dirty buffer.
func (idx *IVFIndex) Delete(id uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Try main index first
	if row, exists := idx.idToRow[id]; exists {
		// Remove from posting list
		if cluster, ok := idx.clusterOf[id]; ok {
			idx.removeFromPostingList(cluster, id)
			delete(idx.clusterOf, id)
		}
		// Swap-with-last in vectors array
		lastRow := idx.count - 1
		if row != lastRow {
			copy(idx.vectors[row*idx.dim:(row+1)*idx.dim],
				idx.vectors[lastRow*idx.dim:(lastRow+1)*idx.dim])
			lastID := idx.ids[lastRow]
			idx.ids[row] = lastID
			idx.idToRow[lastID] = row
		}
		delete(idx.idToRow, id)
		idx.count--
		return nil
	}

	// Try dirty buffer
	return idx.dirty.Delete(id)
}

// removeFromPostingList removes an ID from a cluster's posting list.
func (idx *IVFIndex) removeFromPostingList(cluster int, id uint64) {
	list := idx.postingList[cluster]
	for i, v := range list {
		if v == id {
			// Swap with last and truncate
			list[i] = list[len(list)-1]
			idx.postingList[cluster] = list[:len(list)-1]
			return
		}
	}
}

// Len returns the total number of live vectors (main + dirty buffer).
func (idx *IVFIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.count + idx.dirty.Len()
}

// Rebuild triggers a full index reconstruction: merges dirty buffer into main,
// runs K-Means++, and rebuilds posting lists.
func (idx *IVFIndex) Rebuild() error {
	// Snapshot all vectors (dirty + main) outside write lock
	idx.mu.RLock()
	dirtyIDs, dirtyVecs, dirtyCount := idx.dirty.AllVectors()
	mainCount := idx.count
	totalN := mainCount + dirtyCount

	allVecs := make([]float32, totalN*idx.dim)
	allIDs := make([]uint64, totalN)

	// Copy main vectors
	copy(allVecs, idx.vectors[:mainCount*idx.dim])
	copy(allIDs, idx.ids[:mainCount])

	// Append dirty vectors
	copy(allVecs[mainCount*idx.dim:], dirtyVecs[:dirtyCount*idx.dim])
	copy(allIDs[mainCount:], dirtyIDs[:dirtyCount])
	idx.mu.RUnlock()

	if totalN == 0 {
		return nil
	}

	// Adjust K if we have fewer vectors than clusters
	effectiveK := idx.k
	if totalN < effectiveK {
		effectiveK = totalN
	}

	// Run K-Means++ outside any lock (this is the slow part)
	km := NewKMeans(effectiveK, idx.dim, idx.distFn, time.Now().UnixNano())
	assignments := km.Fit(allVecs, totalN)

	// Build new posting lists
	newPostingList := make([][]uint64, effectiveK)
	for i := range newPostingList {
		newPostingList[i] = make([]uint64, 0)
	}
	newClusterOf := make(map[uint64]int, totalN)
	for i, cluster := range assignments {
		id := allIDs[i]
		newPostingList[cluster] = append(newPostingList[cluster], id)
		newClusterOf[id] = cluster
	}

	// Swap in atomically under write lock
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Rebuild main storage
	if totalN > idx.capacity {
		idx.capacity = totalN * 2
		idx.vectors = make([]float32, idx.capacity*idx.dim)
		idx.ids = make([]uint64, idx.capacity)
	}
	copy(idx.vectors, allVecs[:totalN*idx.dim])
	copy(idx.ids, allIDs[:totalN])
	idx.count = totalN

	// Rebuild id→row map
	idx.idToRow = make(map[uint64]int, totalN)
	for i := 0; i < totalN; i++ {
		idx.idToRow[allIDs[i]] = i
	}

	idx.kmeans = km
	idx.postingList = newPostingList
	idx.clusterOf = newClusterOf
	idx.built = true

	// Clear dirty buffer by creating a fresh one
	idx.dirty, _ = flat.NewFlatIndex(idx.dim, idx.metric)

	return nil
}

// Flush is a no-op for in-memory IVFIndex.
func (idx *IVFIndex) Flush() error {
	return nil
}

// Close stops the background rebuilder.
func (idx *IVFIndex) Close() {
	if idx.cancel != nil {
		idx.cancel()
		<-idx.done
	}
}

// backgroundRebuilder watches the dirty buffer and triggers rebuilds.
func (idx *IVFIndex) backgroundRebuilder(ctx context.Context) {
	defer close(idx.done)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			idx.mu.RLock()
			dirtyLen := idx.dirty.Len()
			threshold := idx.dirtyThreshold
			idx.mu.RUnlock()

			if dirtyLen > threshold {
				slog.Info("triggering IVF rebuild",
					"dirty_count", dirtyLen,
					"threshold", threshold,
				)
				if err := idx.Rebuild(); err != nil {
					slog.Error("IVF rebuild failed", "error", err)
				}
			}
		}
	}
}

// Compile-time interface check
var _ index.Index = (*IVFIndex)(nil)
