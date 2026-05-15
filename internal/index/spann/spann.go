// Package spann implements the SPANN (Space Partition And Nearest Neighbor) index
// for billion-scale vector search. The centroid HNSW index fits in RAM; posting lists
// of assigned vectors live on disk or in-memory pages.
//
// Architecture (modeled after ChromaDB Cloud's production SPANN):
//
//   ┌──────────────────────────────────┐
//   │         Centroid HNSW            │  ← in-RAM graph of K centroids
//   │  (small: K nodes, each = centroid)│
//   └──────────┬───────────────────────┘
//              │ search yields top-C closest centroids
//              ▼
//   ┌──────────────────────────────────┐
//   │       Posting Lists              │  ← per-centroid vector lists
//   │  postings[c] = [(id, vec), ...]  │     (could be on disk for billion-scale)
//   └──────────────────────────────────┘
//
// Query flow:
//  1. Search centroid HNSW for closest C centroids (C = nprobe)
//  2. Scan posting lists of those C centroids
//  3. Compute exact distances, return top-K
//
// This is essentially IVF with an HNSW navigator instead of brute-force centroid search.
// At billion scale, the centroid HNSW has ~sqrt(N) centroids and fits in < 1GB RAM.
package spann

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/POTATO-VE1/Magnitude/internal/distance"
	vdberrors "github.com/POTATO-VE1/Magnitude/internal/errors"
	"github.com/POTATO-VE1/Magnitude/internal/index"
	"github.com/POTATO-VE1/Magnitude/internal/index/hnsw"
)

// posting is a single vector in a posting list.
type posting struct {
	id     uint64
	vector []float32
}

// SPANNIndex implements the Index interface using centroid HNSW + posting lists.
type SPANNIndex struct {
	mu         sync.RWMutex
	dim        int
	metric     string
	distFn     distance.DistanceFunc
	numCents   int // number of centroids
	nprobe     int // default number of posting lists to scan

	// centroidHNSW navigates to the nearest centroids in O(log K)
	centroidHNSW *hnsw.HNSWIndex
	// centroids[i] = centroid vector for cluster i
	centroids [][]float32
	
	// Disk-backed posting lists
	store     *MmapPostingStore
	storePath string

	// idToCentroid maps vector ID → which centroid it was assigned to
	idToCentroid map[uint64]int
	// deleted tracks soft-deleted IDs
	deleted map[uint64]bool

	// dirty buffer for vectors inserted before first build
	dirtyBuf []posting
	built    bool
}

// NewSPANNIndex creates a new SPANN index.
//   - dim: vector dimension
//   - numCentroids: number of centroids (rule of thumb: sqrt(N))
//   - nprobe: default number of posting lists to scan during search
//   - metric: distance metric name
func NewSPANNIndex(dim, numCentroids, nprobe int, metric string) (*SPANNIndex, error) {
	if dim <= 0 {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch, "dimension must be > 0, got %d", dim)
	}
	distFn, err := distance.GetDistanceFunc(metric)
	if err != nil {
		return nil, err
	}
	if numCentroids <= 0 {
		numCentroids = 256
	}
	if nprobe <= 0 {
		nprobe = 5
	}

	return &SPANNIndex{
		dim:          dim,
		metric:       metric,
		distFn:       distFn,
		numCents:     numCentroids,
		nprobe:       nprobe,
		idToCentroid: make(map[uint64]int),
		deleted:      make(map[uint64]bool),
		dirtyBuf:     make([]posting, 0, 1024),
		storePath:    filepath.Join(os.TempDir(), fmt.Sprintf("spann_postings_%d.bin", rand.Int63())),
	}, nil
}

// Insert adds a vector. Before the first Rebuild, vectors go to a dirty buffer.
// After Rebuild, vectors are assigned to the nearest centroid's posting list.
func (s *SPANNIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != s.dim {
		return vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"expected dimension %d, got %d", s.dim, len(vector))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.idToCentroid[id]; exists {
		return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists", id)
	}

	// Check dirty buffer for duplicates too
	for _, p := range s.dirtyBuf {
		if p.id == id {
			return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists in buffer", id)
		}
	}

	vec := make([]float32, s.dim)
	copy(vec, vector)

	if !s.built {
		s.dirtyBuf = append(s.dirtyBuf, posting{id: id, vector: vec})
		return nil
	}

	// We cannot directly append to disk in SPANN. 
	// In a real implementation we would buffer to a delta index.
	// For simplicity, we just add it to dirtyBuf until next rebuild.
	s.dirtyBuf = append(s.dirtyBuf, posting{id: id, vector: vec})
	s.idToCentroid[id] = -1 // -1 implies not in a centroid yet

	return nil
}

// Search performs ANN search: find nearest centroids via HNSW, then scan their posting lists.
func (s *SPANNIndex) Search(ctx context.Context, query []float32, k int, nprobe int) ([]index.SearchResult, error) {
	if len(query) != s.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), s.dim)
	}
	if k <= 0 {
		return nil, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if nprobe <= 0 {
		nprobe = s.nprobe
	}

	// If not built yet, brute-force the dirty buffer
	if !s.built {
		return s.bruteForceSearch(query, k)
	}

	// Phase 1: Find nearest C centroids via HNSW
	centroidResults, err := s.centroidHNSW.Search(ctx, query, nprobe, nprobe*2)
	if err != nil {
		return nil, err
	}

	// Phase 2: Scan posting lists of those centroids
	type candidate struct {
		id   uint64
		dist float32
	}
	var candidates []candidate

	for _, cr := range centroidResults {
		centIdx := int(cr.ID)
		
		posts, err := s.store.GetPostings(centIdx)
		if err != nil {
			return nil, err
		}

		for _, p := range posts {
			if s.deleted[p.id] {
				continue
			}
			d := s.distFn(query, p.vector)
			candidates = append(candidates, candidate{id: p.id, dist: d})
		}
	}

	// Also scan dirty buffer
	for _, p := range s.dirtyBuf {
		if s.deleted[p.id] {
			continue
		}
		d := s.distFn(query, p.vector)
		candidates = append(candidates, candidate{id: p.id, dist: d})
	}

	// Select top-K (partial sort)
	if len(candidates) > k {
		for i := 0; i < k; i++ {
			minIdx := i
			for j := i + 1; j < len(candidates); j++ {
				if candidates[j].dist < candidates[minIdx].dist {
					minIdx = j
				}
			}
			candidates[i], candidates[minIdx] = candidates[minIdx], candidates[i]
		}
		candidates = candidates[:k]
	} else {
		// Sort what we have
		for i := 0; i < len(candidates); i++ {
			for j := i + 1; j < len(candidates); j++ {
				if candidates[j].dist < candidates[i].dist {
					candidates[i], candidates[j] = candidates[j], candidates[i]
				}
			}
		}
	}

	results := make([]index.SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = index.SearchResult{
			ID:       c.id,
			Distance: c.dist,
			Score:    distance.ScoreFromDistance(c.dist, s.metric),
		}
	}

	return results, nil
}

// Delete soft-deletes a vector by ID.
func (s *SPANNIndex) Delete(id uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if ID exists in postings or dirty buffer
	if _, exists := s.idToCentroid[id]; !exists {
		found := false
		for _, p := range s.dirtyBuf {
			if p.id == id {
				found = true
				break
			}
		}
		if !found {
			return vdberrors.Newf(vdberrors.ErrVectorNotFound, "vector ID %d not found", id)
		}
	}

	s.deleted[id] = true
	return nil
}

// Len returns the number of live vectors.
func (s *SPANNIndex) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := len(s.dirtyBuf)
	if s.store != nil {
		for _, meta := range s.store.offsets {
			total += meta.count
		}
	}
	return total - len(s.deleted)
}

// Rebuild builds the centroid HNSW and assigns all vectors to posting lists.
// This uses simple K-Means to find centroids, then builds an HNSW index over them.
func (s *SPANNIndex) Rebuild() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Collect all live vectors
	var allVecs []posting
	for _, p := range s.dirtyBuf {
		if !s.deleted[p.id] {
			allVecs = append(allVecs, p)
		}
	}
	if s.store != nil {
		for cid := range s.store.offsets {
			posts, _ := s.store.GetPostings(cid)
			for _, p := range posts {
				if !s.deleted[p.id] {
					allVecs = append(allVecs, p)
				}
			}
		}
	}

	n := len(allVecs)
	if n == 0 {
		return nil
	}

	// Determine number of centroids: min(numCents, n)
	k := s.numCents
	if k > n {
		k = n
	}

	// Simple K-Means to find centroids
	centroids := s.kmeans(allVecs, k, 20)

	// Build HNSW index over centroids
	centHNSW, err := hnsw.NewHNSWIndex(s.dim, 16, 200, 50, s.metric)
	if err != nil {
		return err
	}

	for i, c := range centroids {
		if err := centHNSW.Insert(uint64(i), c); err != nil {
			return err
		}
	}

	// Assign all vectors to nearest centroid
	postings := make([][]posting, k)
	for i := range postings {
		postings[i] = make([]posting, 0)
	}

	idToCentroid := make(map[uint64]int, n)
	for _, p := range allVecs {
		nearest := 0
		nearestDist := float32(math.MaxFloat32)
		for c := 0; c < k; c++ {
			d := s.distFn(p.vector, centroids[c])
			if d < nearestDist {
				nearestDist = d
				nearest = c
			}
		}
		postings[nearest] = append(postings[nearest], p)
		idToCentroid[p.id] = nearest
	}

	// Write to disk
	if s.store != nil {
		s.store.Close()
	}
	if err := WritePostings(s.storePath, postings, s.dim); err != nil {
		return err
	}
	
	newStore, err := NewMmapPostingStore(s.storePath, s.dim)
	if err != nil {
		return err
	}

	// Commit
	s.centroidHNSW = centHNSW
	s.centroids = centroids
	s.store = newStore
	s.idToCentroid = idToCentroid
	s.dirtyBuf = s.dirtyBuf[:0]
	s.deleted = make(map[uint64]bool)
	s.built = true

	return nil
}

// Flush is a no-op for in-memory SPANN.
func (s *SPANNIndex) Flush() error {
	return nil
}

// Close cleans up the mmap store.
func (s *SPANNIndex) Close() {
	if s.store != nil {
		s.store.Close()
	}
	os.Remove(s.storePath)
}

// ── Internal helpers ────────────────────────────────────────────────────────

// nearestCentroid finds the index of the nearest centroid for a vector.
func (s *SPANNIndex) nearestCentroid(vec []float32) int {
	nearest := 0
	nearestDist := float32(math.MaxFloat32)
	for i, c := range s.centroids {
		d := s.distFn(vec, c)
		if d < nearestDist {
			nearestDist = d
			nearest = i
		}
	}
	return nearest
}

// bruteForceSearch scans all vectors in the dirty buffer.
func (s *SPANNIndex) bruteForceSearch(query []float32, k int) ([]index.SearchResult, error) {
	type candidate struct {
		id   uint64
		dist float32
	}
	var candidates []candidate
	for _, p := range s.dirtyBuf {
		if s.deleted[p.id] {
			continue
		}
		d := s.distFn(query, p.vector)
		candidates = append(candidates, candidate{id: p.id, dist: d})
	}

	// Sort by distance
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].dist < candidates[i].dist {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	results := make([]index.SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = index.SearchResult{
			ID:       c.id,
			Distance: c.dist,
			Score:    distance.ScoreFromDistance(c.dist, s.metric),
		}
	}
	return results, nil
}

// kmeans performs simple K-Means clustering to find centroids.
func (s *SPANNIndex) kmeans(data []posting, k, maxIter int) [][]float32 {
	rng := rand.New(rand.NewSource(42))
	n := len(data)

	// Initialize centroids with random selection
	centroids := make([][]float32, k)
	used := make(map[int]bool)
	for i := 0; i < k; i++ {
		idx := rng.Intn(n)
		for used[idx] {
			idx = rng.Intn(n)
		}
		used[idx] = true
		centroids[i] = make([]float32, s.dim)
		copy(centroids[i], data[idx].vector)
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIter; iter++ {
		// Assignment
		changes := 0
		for i, p := range data {
			nearest := 0
			nearestDist := float32(math.MaxFloat32)
			for c := 0; c < k; c++ {
				d := s.distFn(p.vector, centroids[c])
				if d < nearestDist {
					nearestDist = d
					nearest = c
				}
			}
			if nearest != assignments[i] {
				assignments[i] = nearest
				changes++
			}
		}

		if changes == 0 {
			break
		}

		// Update
		newCentroids := make([][]float32, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			newCentroids[c] = make([]float32, s.dim)
		}
		for i, p := range data {
			c := assignments[i]
			counts[c]++
			for d := 0; d < s.dim; d++ {
				newCentroids[c][d] += p.vector[d]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				invN := 1.0 / float32(counts[c])
				for d := 0; d < s.dim; d++ {
					newCentroids[c][d] *= invN
				}
			} else {
				// Dead centroid: keep old
				copy(newCentroids[c], centroids[c])
			}
		}
		centroids = newCentroids
	}

	return centroids
}

// Compile-time interface check
var _ index.Index = (*SPANNIndex)(nil)
var _ index.VectorExporter = (*SPANNIndex)(nil)

// ExportVectors returns all live vectors across posting lists and dirty buffer.
func (s *SPANNIndex) ExportVectors() []index.ExportedVector {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []index.ExportedVector

	// Export from posting store
	if s.store != nil {
		for cid := range s.store.offsets {
			posts, err := s.store.GetPostings(cid)
			if err != nil {
				continue
			}
			for _, p := range posts {
				if s.deleted[p.id] {
					continue
				}
				vec := make([]float32, len(p.vector))
				copy(vec, p.vector)
				result = append(result, index.ExportedVector{ID: p.id, Vector: vec})
			}
		}
	}

	// Export from dirty buffer
	for _, p := range s.dirtyBuf {
		if s.deleted[p.id] {
			continue
		}
		vec := make([]float32, len(p.vector))
		copy(vec, p.vector)
		result = append(result, index.ExportedVector{ID: p.id, Vector: vec})
	}

	return result
}
