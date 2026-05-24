// Package hnsw implements a Hierarchical Navigable Small World (HNSW) graph index.
// HNSW provides state-of-the-art approximate nearest-neighbor search with O(log N)
// query time and high recall. The graph fits entirely in RAM.
//
// Algorithm summary (Malkov & Yashunin, 2016):
//   - Multi-layer skip-list-like graph. Layer 0 has all nodes.
//   - Higher layers act as express lanes for coarse navigation.
//   - Insert: enter at the topmost layer, greedily descend to layer 0,
//     build edges at each layer via beam search (efConstruction width).
//   - Search: enter at top, greedily descend, widen search at layer 0
//     with beam width ef (efSearch).
//
// Key parameters:
//   - M: max bidirectional connections per node per layer (default: 16)
//   - efConstruction: beam width during insert (default: 200)
//   - efSearch: beam width during query (default: 50)
//
// The level assignment follows the exponential distribution:
//
//	level = floor(-ln(uniform(0,1)) * mL), where mL = 1/ln(M).
package hnsw

import (
	"container/heap"
	"context"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/POTATO-VE1/Magnitude/internal/distance"
	vdberrors "github.com/POTATO-VE1/Magnitude/internal/errors"
	"github.com/POTATO-VE1/Magnitude/internal/index"
)

// node represents a single vertex in the HNSW graph.
type node struct {
	id      uint64
	level   int     // max layer this node exists on
	friends [][]int // friends[layer] = list of neighbor node indices
}

// HNSWIndex implements the Index interface using a multi-layer navigable small world graph.
type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	metric         string
	distFn         distance.DistanceFunc
	m              int            // max connections per layer
	mMax0          int            // max connections on layer 0 (2*M)
	efConstruction int            // beam width during construction
	efSearch       int            // default beam width during search
	mL             float64        // normalization factor: 1/ln(M)
	maxLevel       int            // current max level in the graph
	entryPoint     int            // index of the entry point node
	nodes          []node         // all nodes (index = internal node ID)
	idToNode       map[uint64]int // external ID → internal node index
	vectorSlab     []float32      // contiguous vector storage [nodeCount * dim]
	vectorOffsets  []int          // node index → offset in vectorSlab
	rng            *rand.Rand
	deleted        map[int]bool // soft-delete tombstones by internal nodeIdx
	snapshotPath   string       // file path for snapshot persistence
	snapshotSeqID  uint64       // WAL seqID at last snapshot time

	// Async indexing: dirty buffer holds pending inserts
	dirtyMu   sync.Mutex
	dirty     []pendingInsert // buffered inserts not yet in the graph
	cancel    context.CancelFunc
	done      chan struct{}
	applierWg sync.WaitGroup

	// OnDrain is called after the dirty buffer is drained to the main graph.
	// Receives the mapping of external IDs to their newly assigned node indices.
	// Used by the Collection layer to update the bitmap index.
	OnDrain func(drained map[uint64]int)
}

// pendingInsert holds a vector waiting to be applied to the HNSW graph.
type pendingInsert struct {
	id     uint64
	vector []float32
}

// nodeVector returns the vector for the given node index from the contiguous slab.
func (h *HNSWIndex) nodeVector(nodeIdx int) []float32 {
	offset := h.vectorOffsets[nodeIdx]
	return h.vectorSlab[offset : offset+h.dim]
}

// visitedTracker is a bitmap-based visited tracker that avoids map overhead.
// Uses a generation counter to avoid clearing the entire bitmap on each search.
type visitedTracker struct {
	generations []uint32
	currentGen  uint32
}

// nextGen increments the generation counter, resetting the tracker on wraparound.
func (vt *visitedTracker) nextGen() {
	vt.currentGen++
	if vt.currentGen == 0 {
		// Wraparound: reset all generations to 0 so old entries aren't falsely "visited"
		for i := range vt.generations {
			vt.generations[i] = 0
		}
		vt.currentGen = 1
	}
}

// visitedPool is a sync.Pool of visited tracker bitmaps to avoid allocation overhead during searches.
var visitedPool = sync.Pool{
	New: func() any {
		return &visitedTracker{
			generations: make([]uint32, 1024),
		}
	},
}

// NewHNSWIndex creates a new HNSW index.
//   - dim: vector dimension
//   - m: max connections per layer (typical: 8–48)
//   - efConstruction: beam width during insert (typical: 100–400)
//   - efSearch: default beam width during query (typical: 50–200)
//   - metric: distance metric name ("l2", "cosine", "dot", "manhattan")
func NewHNSWIndex(dim, m, efConstruction, efSearch int, metric string) (*HNSWIndex, error) {
	if dim <= 0 {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch, "dimension must be > 0, got %d", dim)
	}
	distFn, err := distance.GetDistanceFunc(metric)
	if err != nil {
		return nil, err
	}
	if m <= 0 {
		m = 16
	}
	if efConstruction <= 0 {
		efConstruction = 400
	}
	if efSearch <= 0 {
		efSearch = 128
	}

	h := &HNSWIndex{
		dim:            dim,
		metric:         metric,
		distFn:         distFn,
		m:              m,
		mMax0:          2 * m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		mL:             1.0 / math.Log(float64(m)),
		maxLevel:       -1,
		entryPoint:     -1,
		nodes:          make([]node, 0, 1024),
		idToNode:       make(map[uint64]int),
		vectorSlab:     make([]float32, 0, 1024*dim),
		vectorOffsets:  make([]int, 0, 1024),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
		deleted:        make(map[int]bool),
		dirty:          make([]pendingInsert, 0, 256),
		done:           make(chan struct{}),
	}

	// Start background applier
	ctx, cancel := context.WithCancel(context.Background())
	h.cancel = cancel
	go h.backgroundApplier(ctx)

	return h, nil
}

// randomLevel generates a random level using the exponential distribution.
// P(level=l) = (1/M)^l, truncated at a reasonable maximum.
func (h *HNSWIndex) randomLevel() int {
	r := h.rng.Float64()
	if r == 0 {
		r = 1e-15
	}
	level := int(-math.Log(r) * h.mL)
	// Cap at a reasonable max to prevent degenerate graphs
	maxPossible := int(math.Log(float64(len(h.nodes)+1))*h.mL) + 2
	if maxPossible < 6 {
		maxPossible = 6
	}
	if level > maxPossible {
		level = maxPossible
	}
	return level
}

// Insert adds a vector to the dirty buffer. The background applier
// will insert it into the HNSW graph asynchronously.
func (h *HNSWIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != h.dim {
		return vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"expected dimension %d, got %d", h.dim, len(vector))
	}

	// Copy the vector so the caller can't mutate it after Insert returns
	vec := make([]float32, h.dim)
	copy(vec, vector)

	h.dirtyMu.Lock()
	h.dirty = append(h.dirty, pendingInsert{id: id, vector: vec})
	h.dirtyMu.Unlock()

	return nil
}

// insertLocked performs the actual insert into the main graph.
// Must be called with h.mu held.
func (h *HNSWIndex) insertLocked(id uint64, vector []float32) error {
	if nodeIdx, exists := h.idToNode[id]; exists {
		if h.deleted[nodeIdx] {
			delete(h.idToNode, id)
		} else {
			return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists", id)
		}
	}

	// Assign level
	level := h.randomLevel()
	nodeIdx := len(h.nodes)

	// Create node with friend lists for each layer
	n := node{
		id:      id,
		level:   level,
		friends: make([][]int, level+1),
	}
	for i := range n.friends {
		n.friends[i] = make([]int, 0, h.m)
	}

	// Append vector to contiguous slab
	offset := len(h.vectorSlab)
	h.vectorSlab = append(h.vectorSlab, vector...)
	h.vectorOffsets = append(h.vectorOffsets, offset)

	h.nodes = append(h.nodes, n)
	h.idToNode[id] = nodeIdx

	// First node: set as entry point
	if h.entryPoint == -1 {
		h.entryPoint = nodeIdx
		h.maxLevel = level
		return nil
	}

	// Phase 1: Greedy descent from top to level+1
	ep := h.entryPoint
	for lc := h.maxLevel; lc > level; lc-- {
		ep = h.greedyClosest(vector, ep, lc)
	}

	// Phase 2: Insert with beam search at each layer from min(level, maxLevel) down to 0
	topLayer := level
	if topLayer > h.maxLevel {
		topLayer = h.maxLevel
	}
	vt := visitedPool.Get().(*visitedTracker)
	defer func() {
		vt.nextGen()
		visitedPool.Put(vt)
	}()

	for lc := topLayer; lc >= 0; lc-- {
		vt.nextGen()
		candidates := h.searchLayer(context.Background(), vector, ep, h.efConstruction, lc, vt, nil)
		// Select neighbors using the diversity heuristic
		maxConn := h.m
		if lc == 0 {
			maxConn = h.mMax0
		}
		neighbors := h.selectNeighborsHeuristic(vector, candidates, maxConn)

		// Add bidirectional edges
		h.nodes[nodeIdx].friends[lc] = neighbors
		for _, neighborIdx := range neighbors {
			h.nodes[neighborIdx].friends[lc] = append(h.nodes[neighborIdx].friends[lc], nodeIdx)
			// Prune neighbor if it exceeds max connections
			if len(h.nodes[neighborIdx].friends[lc]) > maxConn {
				h.nodes[neighborIdx].friends[lc] = h.pruneConnections(
					h.nodeVector(neighborIdx), h.nodes[neighborIdx].friends[lc], maxConn)
			}
		}

		// Update entry point for next layer
		if len(candidates) > 0 {
			ep = candidates[0].nodeIdx
		}
	}

	// Update entry point if new node has a higher level
	if level > h.maxLevel {
		h.maxLevel = level
		h.entryPoint = nodeIdx
	}

	return nil
}

// Search performs approximate nearest-neighbor search via HNSW graph traversal.
// k = number of results, nprobe is interpreted as efSearch override.
func (h *HNSWIndex) Search(ctx context.Context, query []float32, k int, nprobe int) ([]index.SearchResult, error) {
	if len(query) != h.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), h.dim)
	}
	if k <= 0 {
		return nil, nil
	}

	// Search dirty buffer FIRST (has its own dirtyMu) to avoid ABBA deadlock
	// with drainDirtyBuffer which acquires dirtyMu then mu.
	dirtyResults := h.searchDirtyBuffer(query, k)

	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		// No graph yet — return dirty buffer results only
		return dirtyResults, nil
	}

	ef := h.efSearch
	if nprobe > 0 {
		ef = nprobe
	}
	if ef < k {
		ef = k
	}

	// Phase 1: Greedy descent from top to layer 1
	ep := h.entryPoint
	for lc := h.maxLevel; lc >= 1; lc-- {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		ep = h.greedyClosest(query, ep, lc)
	}

	vt := visitedPool.Get().(*visitedTracker)
	defer func() {
		vt.nextGen()
		visitedPool.Put(vt)
	}()

	// Phase 2: Beam search at layer 0
	candidates := h.searchLayer(ctx, query, ep, ef, 0, vt, nil)

	// Extract top-k, filtering deleted nodes
	var results []index.SearchResult
	for _, c := range candidates {
		if h.deleted[c.nodeIdx] {
			continue
		}
		results = append(results, index.SearchResult{
			ID:       h.nodes[c.nodeIdx].id,
			Distance: c.dist,
		})
		if len(results) >= k {
			break
		}
	}

	// Populate scores
	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	// Merge with dirty buffer results
	if len(dirtyResults) > 0 {
		results = mergeResults(results, dirtyResults, k)
	}

	return results, nil
}

// SearchFiltered performs pre-filtered HNSW search. Only nodes whose IDs
// are in validIDs are explored during graph traversal. This is much faster
// than post-filtering when the filter is selective (e.g., 1% match rate).
func (h *HNSWIndex) SearchFiltered(ctx context.Context, query []float32, k, nprobe int, validIDs map[uint64]bool) ([]index.SearchResult, error) {
	if len(query) != h.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), h.dim)
	}
	if k <= 0 || len(validIDs) == 0 {
		return nil, nil
	}

	// Search dirty buffer FIRST (has its own dirtyMu) to avoid ABBA deadlock
	dirtyResults := h.searchDirtyBufferFiltered(query, k, validIDs)

	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		return dirtyResults, nil
	}

	ef := h.efSearch
	if nprobe > 0 {
		ef = nprobe
	}
	if ef < k {
		ef = k
	}

	// Phase 1: Greedy descent — find closest valid entry point
	ep := h.entryPoint
	for lc := h.maxLevel; lc >= 1; lc-- {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		ep = h.greedyClosestFiltered(query, ep, lc, validIDs)
	}

	vt := visitedPool.Get().(*visitedTracker)
	defer func() {
		vt.nextGen()
		visitedPool.Put(vt)
	}()

	// Phase 2: Beam search at layer 0 with filter
	candidates := h.searchLayer(ctx, query, ep, ef, 0, vt, validIDs)

	// Extract top-k (already filtered, no deleted check needed)
	var results []index.SearchResult
	for _, c := range candidates {
		results = append(results, index.SearchResult{
			ID:       h.nodes[c.nodeIdx].id,
			Distance: c.dist,
		})
		if len(results) >= k {
			break
		}
	}

	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	// Merge with dirty buffer results
	if len(dirtyResults) > 0 {
		results = mergeResults(results, dirtyResults, k)
	}

	return results, nil
}

// SearchFilteredBitmap is the optimized pre-filtered search that uses a
// bitset bitmap of internal node indices instead of a map of external IDs.
// The bitmap is produced by MetadataBitmapIndex.Resolve() and allows the
// searchLayer to filter nodes with a single bit test (~1 CPU cycle) instead
// of a map lookup (hash + compare, ~10-20 CPU cycles).
func (h *HNSWIndex) SearchFilteredBitmap(ctx context.Context, query []float32, k, nprobe int, filter *index.FilterBitmap) ([]index.SearchResult, error) {
	if len(query) != h.dim {
		return nil, vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"query dimension %d != index dimension %d", len(query), h.dim)
	}
	if k <= 0 || filter == nil || filter.None() {
		return nil, nil
	}

	// Search dirty buffer FIRST (has its own dirtyMu) to avoid ABBA deadlock
	dirtyResults := h.searchDirtyBufferBitmap(query, k, filter)

	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		return dirtyResults, nil
	}

	ef := h.efSearch
	if nprobe > 0 {
		ef = nprobe
	}
	if ef < k {
		ef = k
	}

	// Phase 1: Greedy descent — find closest valid entry point using bitmap
	ep := h.entryPoint
	for lc := h.maxLevel; lc >= 1; lc-- {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		ep = h.greedyClosestBitmap(query, ep, lc, filter)
	}

	vt := visitedPool.Get().(*visitedTracker)
	defer func() {
		vt.nextGen()
		visitedPool.Put(vt)
	}()

	// Phase 2: Beam search at layer 0 with bitmap filter
	candidates := h.searchLayerBitmap(ctx, query, ep, ef, 0, vt, filter)

	// Extract top-k
	var results []index.SearchResult
	for _, c := range candidates {
		results = append(results, index.SearchResult{
			ID:       h.nodes[c.nodeIdx].id,
			Distance: c.dist,
		})
		if len(results) >= k {
			break
		}
	}

	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	// Merge with dirty buffer results
	if len(dirtyResults) > 0 {
		results = mergeResults(results, dirtyResults, k)
	}

	return results, nil
}

// searchDirtyBufferBitmap performs brute-force filtered search on the dirty buffer.
// The bitmap uses internal node indices, but dirty buffer entries don't have them yet.
// So we search all dirty entries and let the caller merge/filter.
func (h *HNSWIndex) searchDirtyBufferBitmap(query []float32, k int, filter *index.FilterBitmap) []index.SearchResult {
	h.dirtyMu.Lock()
	dirty := make([]pendingInsert, len(h.dirty))
	copy(dirty, h.dirty)
	h.dirtyMu.Unlock()

	if len(dirty) == 0 {
		return nil
	}

	dim := h.dim
	n := len(dirty)
	batchVecs := make([]float32, n*dim)
	ids := make([]uint64, n)
	for i, d := range dirty {
		copy(batchVecs[i*dim:], d.vector)
		ids[i] = d.id
	}

	dists := make([]float32, n)
	distance.BatchDistance(query, batchVecs, n, dim, h.metric, dists)

	results := make([]index.SearchResult, 0, n)
	for i, id := range ids {
		results = append(results, index.SearchResult{
			ID:       id,
			Distance: dists[i],
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if len(results) > k {
		results = results[:k]
	}

	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	return results
}

// greedyClosestBitmap walks greedily from ep, only moving to neighbors
// whose node index is set in the bitmap filter.
func (h *HNSWIndex) greedyClosestBitmap(query []float32, ep int, layer int, filter *index.FilterBitmap) int {
	epDist := h.distFn(query, h.nodeVector(ep))
	changed := true
	for changed {
		changed = false
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				if h.deleted[friendIdx] {
					continue
				}
				// Bitmap check: O(1) bit test instead of map lookup
				if !filter.Test(uint(friendIdx)) {
					continue
				}
				d := h.distFn(query, h.nodeVector(friendIdx))
				if d < epDist {
					ep = friendIdx
					epDist = d
					changed = true
				}
			}
		}
	}
	// Verify entry point is in the filter set
	if !filter.Test(uint(ep)) {
		// Find closest valid neighbor
		bestDist := float32(math.MaxFloat32)
		bestIdx := ep
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				if h.deleted[friendIdx] || !filter.Test(uint(friendIdx)) {
					continue
				}
				d := h.distFn(query, h.nodeVector(friendIdx))
				if d < bestDist {
					bestDist = d
					bestIdx = friendIdx
				}
			}
		}
		ep = bestIdx
	}
	return ep
}

// searchLayerBitmap performs beam search at a single layer using bitmap filtering.
// The bitmap allows checking node validity with a single bit test instead of a map lookup.
func (h *HNSWIndex) searchLayerBitmap(ctx context.Context, query []float32, ep int, ef int, layer int, vt *visitedTracker, filter *index.FilterBitmap) []candidate {
	// Ensure tracker is large enough
	if ep >= len(vt.generations) {
		newGen := make([]uint32, ep+1024)
		copy(newGen, vt.generations)
		vt.generations = newGen
	}
	vt.generations[ep] = vt.currentGen

	epDist := h.distFn(query, h.nodeVector(ep))

	cands := &minCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(cands)

	results := &maxCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(results)

	maxFriends := h.mMax0
	unvisited := make([]int, 0, maxFriends)
	batchVecs := make([]float32, 0, maxFriends*h.dim)
	dists := make([]float32, 0, maxFriends)

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)

		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}

		if layer < len(h.nodes[c.nodeIdx].friends) {
			friends := h.nodes[c.nodeIdx].friends[layer]

			unvisited = unvisited[:0]
			for _, friendIdx := range friends {
				if friendIdx >= len(vt.generations) {
					newGen := make([]uint32, friendIdx+1024)
					copy(newGen, vt.generations)
					vt.generations = newGen
				}
				if vt.generations[friendIdx] != vt.currentGen {
					vt.generations[friendIdx] = vt.currentGen
					// Bitmap filter: single bit test (~1 CPU cycle)
					if !filter.Test(uint(friendIdx)) {
						continue
					}
					unvisited = append(unvisited, friendIdx)
				}
			}

			if len(unvisited) > 0 {
				batchVecs = batchVecs[:len(unvisited)*h.dim]
				for i, fi := range unvisited {
					copy(batchVecs[i*h.dim:], h.nodeVector(fi))
				}
				dists = dists[:len(unvisited)]
				distance.BatchDistance(query, batchVecs, len(unvisited), h.dim, h.metric, dists)

				for i, fi := range unvisited {
					if i%100 == 0 {
						select {
						case <-ctx.Done():
							sorted := make([]candidate, results.Len())
							for j := len(sorted) - 1; j >= 0; j-- {
								sorted[j] = heap.Pop(results).(candidate)
							}
							return sorted
						default:
						}
					}
					d := dists[i]
					friend := candidate{nodeIdx: fi, dist: d}
					if results.Len() < ef || d < (*results)[0].dist {
						heap.Push(cands, friend)
						heap.Push(results, friend)
						if results.Len() > ef {
							heap.Pop(results)
						}
					}
				}
			}
		}
	}

	sorted := make([]candidate, results.Len())
	for i := len(sorted) - 1; i >= 0; i-- {
		sorted[i] = heap.Pop(results).(candidate)
	}
	return sorted
}

// searchDirtyBufferFiltered performs brute-force filtered search on the dirty buffer.
func (h *HNSWIndex) searchDirtyBufferFiltered(query []float32, k int, validIDs map[uint64]bool) []index.SearchResult {
	h.dirtyMu.Lock()
	dirty := make([]pendingInsert, len(h.dirty))
	copy(dirty, h.dirty)
	h.dirtyMu.Unlock()

	if len(dirty) == 0 {
		return nil
	}

	dim := h.dim
	n := len(dirty)
	batchVecs := make([]float32, 0, n*dim)
	var filteredIDs []uint64

	for _, d := range dirty {
		if validIDs[d.id] {
			batchVecs = append(batchVecs, d.vector...)
			filteredIDs = append(filteredIDs, d.id)
		}
	}

	if len(filteredIDs) == 0 {
		return nil
	}

	dists := make([]float32, len(filteredIDs))
	distance.BatchDistance(query, batchVecs, len(filteredIDs), dim, h.metric, dists)

	results := make([]index.SearchResult, 0, len(filteredIDs))
	for i, id := range filteredIDs {
		results = append(results, index.SearchResult{
			ID:       id,
			Distance: dists[i],
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if len(results) > k {
		results = results[:k]
	}

	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	return results
}

// greedyClosestFiltered walks greedily from ep, only moving to neighbors
// that are in the valid set. At layer 0, verifies the entry point is valid.
func (h *HNSWIndex) greedyClosestFiltered(query []float32, ep int, layer int, validIDs map[uint64]bool) int {
	epDist := h.distFn(query, h.nodeVector(ep))
	changed := true
	for changed {
		changed = false
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				if h.deleted[friendIdx] || !validIDs[h.nodes[friendIdx].id] {
					continue
				}
				d := h.distFn(query, h.nodeVector(friendIdx))
				if d < epDist {
					ep = friendIdx
					epDist = d
					changed = true
				}
			}
		}
	}
	// At layer 0, verify the entry point is valid. If not, find closest valid neighbor.
	if layer == 0 && !validIDs[h.nodes[ep].id] {
		bestDist := float32(math.MaxFloat32)
		bestIdx := ep
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				if h.deleted[friendIdx] || !validIDs[h.nodes[friendIdx].id] {
					continue
				}
				d := h.distFn(query, h.nodeVector(friendIdx))
				if d < bestDist {
					bestDist = d
					bestIdx = friendIdx
				}
			}
		}
		ep = bestIdx
	}
	return ep
}

// Delete soft-deletes a vector by ID. The node remains in the graph
// but is excluded from search results. A Rebuild() removes it permanently.
func (h *HNSWIndex) Delete(id uint64) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	nodeIdx, exists := h.idToNode[id]
	if !exists {
		return vdberrors.Newf(vdberrors.ErrVectorNotFound, "vector ID %d not found", id)
	}
	h.deleted[nodeIdx] = true
	return nil
}

// Len returns the number of live (non-deleted) vectors including dirty buffer.
func (h *HNSWIndex) Len() int {
	h.mu.RLock()
	graphLen := len(h.nodes) - len(h.deleted)
	h.mu.RUnlock()

	h.dirtyMu.Lock()
	dirtyLen := len(h.dirty)
	h.dirtyMu.Unlock()

	return graphLen + dirtyLen
}

// NodeIndex returns the internal HNSW node index for a given external vector ID.
// Returns -1, false if the ID is not in the main graph (may be in dirty buffer).
// Used by the bitmap index to map external IDs to internal node indices.
func (h *HNSWIndex) NodeIndex(id uint64) (int, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	nodeIdx, exists := h.idToNode[id]
	if !exists || h.deleted[nodeIdx] {
		return -1, false
	}
	return nodeIdx, true
}

// Rebuild reconstructs the HNSW graph from scratch, excluding deleted nodes.
func (h *HNSWIndex) Rebuild() error {
	// Drain dirty buffer first to avoid losing pending inserts
	h.drainDirtyBuffer()

	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.deleted) == 0 {
		return nil
	}

	// Collect live nodes (reuse existing slices, no copy needed)
	liveNodes := make([]struct {
		id  uint64
		vec []float32
	}, 0, len(h.nodes)-len(h.deleted))

	for nodeIdx, n := range h.nodes {
		if !h.deleted[nodeIdx] {
			liveNodes = append(liveNodes, struct {
				id  uint64
				vec []float32
			}{n.id, h.nodeVector(nodeIdx)})
		}
	}

	// Reset graph state
	h.nodes = h.nodes[:0]
	h.idToNode = make(map[uint64]int)
	h.deleted = make(map[int]bool)
	h.maxLevel = -1
	h.entryPoint = -1
	h.rng = rand.New(rand.NewSource(time.Now().UnixNano()))

	// Re-insert all live nodes (lock already held)
	for _, ln := range liveNodes {
		if err := h.insertLocked(ln.id, ln.vec); err != nil {
			return err
		}
	}

	return nil
}

// Flush persists the in-memory HNSW graph to disk via snapshot.
// SnapshotPath and SnapshotSeqID must be set before calling Flush.
func (h *HNSWIndex) Flush() error {
	// Drain dirty buffer synchronously before snapshotting
	h.drainDirtyBuffer()

	if h.snapshotPath == "" {
		return nil
	}
	return h.SnapshotToFile(h.snapshotPath, h.snapshotSeqID)
}

// Close stops the background applier and drains remaining dirty inserts.
func (h *HNSWIndex) Close() {
	if h.cancel != nil {
		h.cancel()
		<-h.done
	}
	// Drain any remaining dirty inserts
	h.drainDirtyBuffer()
}

// searchDirtyBuffer performs brute-force search on the dirty buffer.
// Note: Does NOT check h.idToNode/h.deleted — those are protected by h.mu.
// Deleted nodes in the dirty buffer are filtered by the main graph's deleted set
// after merging, or by the caller.
func (h *HNSWIndex) searchDirtyBuffer(query []float32, k int) []index.SearchResult {
	h.dirtyMu.Lock()
	dirty := make([]pendingInsert, len(h.dirty))
	copy(dirty, h.dirty)
	h.dirtyMu.Unlock()

	if len(dirty) == 0 {
		return nil
	}

	// Build contiguous vector buffer for batch distance
	dim := h.dim
	n := len(dirty)
	batchVecs := make([]float32, n*dim)
	for i, d := range dirty {
		copy(batchVecs[i*dim:], d.vector)
	}
	dists := make([]float32, n)
	distance.BatchDistance(query, batchVecs, n, dim, h.metric, dists)

	// Build results — no deleted check here (caller handles it)
	results := make([]index.SearchResult, 0, n)
	for i, d := range dirty {
		results = append(results, index.SearchResult{
			ID:       d.id,
			Distance: dists[i],
		})
	}

	// Sort by distance and take top-k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if len(results) > k {
		results = results[:k]
	}

	for i := range results {
		results[i].Score = distance.ScoreFromDistance(results[i].Distance, h.metric)
	}

	return results
}

// mergeResults merges results from main graph and dirty buffer, returning top-k.
func mergeResults(main, dirty []index.SearchResult, k int) []index.SearchResult {
	merged := append(main, dirty...)
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Distance < merged[j].Distance
	})
	if len(merged) > k {
		merged = merged[:k]
	}
	return merged
}

// drainDirtyBuffer applies all pending inserts to the main graph.
// Returns a mapping of external IDs to their newly assigned node indices.
func (h *HNSWIndex) drainDirtyBuffer() map[uint64]int {
	h.dirtyMu.Lock()
	batch := make([]pendingInsert, len(h.dirty))
	copy(batch, h.dirty)
	h.dirty = h.dirty[:0]
	h.dirtyMu.Unlock()

	if len(batch) == 0 {
		return nil
	}

	drained := make(map[uint64]int, len(batch))

	h.mu.Lock()
	for _, p := range batch {
		nodeIdxBefore := len(h.nodes)
		if err := h.insertLocked(p.id, p.vector); err != nil {
			continue
		}
		drained[p.id] = nodeIdxBefore
	}
	h.mu.Unlock()

	// Notify callback AFTER releasing h.mu to avoid ABBA deadlock
	// with bitmapIndex.mu (search holds bitmap.mu → h.mu.RLock)
	if len(drained) > 0 && h.OnDrain != nil {
		h.OnDrain(drained)
	}

	return drained
}

// backgroundApplier periodically drains the dirty buffer into the main graph.
func (h *HNSWIndex) backgroundApplier(ctx context.Context) {
	defer close(h.done)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.dirtyMu.Lock()
			dirtyLen := len(h.dirty)
			h.dirtyMu.Unlock()

			if dirtyLen > 0 {
				h.drainDirtyBuffer()
			}
		}
	}
}

// SetSnapshotPath configures the file path used by Flush for snapshot persistence.
func (h *HNSWIndex) SetSnapshotPath(path string) {
	h.snapshotPath = path
}

// SetSnapshotSeqID records the WAL sequence ID at snapshot time.
func (h *HNSWIndex) SetSnapshotSeqID(seqID uint64) {
	h.snapshotSeqID = seqID
}

// SetDistanceFunc configures the distance function for a loaded snapshot.
// Required after loading from snapshot since the distance function is not serialized.
func (h *HNSWIndex) SetDistanceFunc(metric string) error {
	distFn, err := distance.GetDistanceFunc(metric)
	if err != nil {
		return err
	}
	h.metric = metric
	h.distFn = distFn
	return nil
}

// ── Graph Internals ─────────────────────────────────────────────────────────

// candidate holds a node index and its distance to the query during search.
type candidate struct {
	nodeIdx int
	dist    float32
}

// greedyClosest performs a greedy walk from ep at the given layer,
// returning the node closest to the query.
func (h *HNSWIndex) greedyClosest(query []float32, ep int, layer int) int {
	epDist := h.distFn(query, h.nodeVector(ep))
	changed := true
	for changed {
		changed = false
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				if h.deleted[friendIdx] {
					continue
				}
				d := h.distFn(query, h.nodeVector(friendIdx))
				if d < epDist {
					ep = friendIdx
					epDist = d
					changed = true
				}
			}
		}
	}
	return ep
}

// searchLayer performs beam search at a single layer starting from ep.
// Returns candidates sorted by distance ascending (closest first).
// validIDs is optional — when non-nil, only nodes whose external ID is in the set are explored.
func (h *HNSWIndex) searchLayer(ctx context.Context, query []float32, ep int, ef int, layer int, vt *visitedTracker, validIDs map[uint64]bool) []candidate {
	// Ensure tracker is large enough
	if ep >= len(vt.generations) {
		newGen := make([]uint32, ep+1024)
		copy(newGen, vt.generations)
		vt.generations = newGen
	}
	vt.generations[ep] = vt.currentGen

	epDist := h.distFn(query, h.nodeVector(ep))

	// candidateHeap: min-heap of unexplored candidates
	cands := &minCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(cands)

	// results: max-heap of best results so far (worst distance at root)
	results := &maxCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(results)

	// Pre-allocate reusable buffers for batch distance computation
	maxFriends := h.mMax0 // max connections on layer 0 (worst case)
	unvisited := make([]int, 0, maxFriends)
	batchVecs := make([]float32, 0, maxFriends*h.dim)
	dists := make([]float32, 0, maxFriends)

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)

		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}

		if layer < len(h.nodes[c.nodeIdx].friends) {
			friends := h.nodes[c.nodeIdx].friends[layer]

			// Collect unvisited friends for batch distance computation
			unvisited = unvisited[:0]
			for _, friendIdx := range friends {
				// Grow tracker if needed
				if friendIdx >= len(vt.generations) {
					newGen := make([]uint32, friendIdx+1024)
					copy(newGen, vt.generations)
					vt.generations = newGen
				}
				if vt.generations[friendIdx] != vt.currentGen {
					vt.generations[friendIdx] = vt.currentGen
					// Pre-filter: skip nodes not in valid set
					if validIDs != nil && !validIDs[h.nodes[friendIdx].id] {
						continue
					}
					unvisited = append(unvisited, friendIdx)
				}
			}

			// Batch compute distances for all unvisited neighbors
			if len(unvisited) > 0 {
				// Build contiguous vector buffer
				batchVecs = batchVecs[:len(unvisited)*h.dim]
				for i, fi := range unvisited {
					copy(batchVecs[i*h.dim:], h.nodeVector(fi))
				}
				dists = dists[:len(unvisited)]
				distance.BatchDistance(query, batchVecs, len(unvisited), h.dim, h.metric, dists)

				// Process results
				for i, fi := range unvisited {
					if i%100 == 0 {
						select {
						case <-ctx.Done():
							// Return partial results instead of nil
							sorted := make([]candidate, results.Len())
							for j := len(sorted) - 1; j >= 0; j-- {
								sorted[j] = heap.Pop(results).(candidate)
							}
							return sorted
						default:
						}
					}
					d := dists[i]
					friend := candidate{nodeIdx: fi, dist: d}
					if results.Len() < ef || d < (*results)[0].dist {
						heap.Push(cands, friend)
						heap.Push(results, friend)
						if results.Len() > ef {
							heap.Pop(results)
						}
					}
				}
			}
		}
	}

	sorted := make([]candidate, results.Len())
	for i := len(sorted) - 1; i >= 0; i-- {
		sorted[i] = heap.Pop(results).(candidate)
	}
	return sorted
}

// selectNeighborsHeuristic selects up to M neighbors from candidates
// using the heuristic from the paper (Algorithm 4, Malkov & Yashunin).
// This prevents degenerate graphs where all neighbors are in one direction.
func (h *HNSWIndex) selectNeighborsHeuristic(query []float32, candidates []candidate, m int) []int {
	if len(candidates) <= m {
		result := make([]int, len(candidates))
		for i, c := range candidates {
			result[i] = c.nodeIdx
		}
		return result
	}

	result := make([]int, 0, m)
	for _, c := range candidates {
		if len(result) >= m {
			break
		}
		// Accept this candidate if it is closer to the query than to any already-selected neighbor.
		cv := h.nodeVector(c.nodeIdx)
		distToCandQuery := c.dist

		closer := false
		for _, rIdx := range result {
			rv := h.nodeVector(rIdx)
			if h.distFn(cv, rv) < distToCandQuery {
				closer = true
				break
			}
		}
		if !closer {
			result = append(result, c.nodeIdx)
		}
	}
	return result
}

// pruneConnections keeps the M best connections from a neighbor list using the heuristic.
func (h *HNSWIndex) pruneConnections(nodeVec []float32, friends []int, maxConn int) []int {
	if len(friends) <= maxConn {
		return friends
	}

	// Create candidates from friends
	candidates := make([]candidate, len(friends))
	for i, f := range friends {
		candidates[i] = candidate{nodeIdx: f, dist: h.distFn(nodeVec, h.nodeVector(f))}
	}

	// Sort candidates by distance (closest first)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	return h.selectNeighborsHeuristic(nodeVec, candidates, maxConn)
}

// ── Heap implementations for beam search ────────────────────────────────────

// minCandHeap is a min-heap for candidate exploration (closest first).
type minCandHeap []candidate

func (h minCandHeap) Len() int           { return len(h) }
func (h minCandHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minCandHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minCandHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minCandHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// maxCandHeap is a max-heap for result tracking (worst distance at root).
type maxCandHeap []candidate

func (h maxCandHeap) Len() int           { return len(h) }
func (h maxCandHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxCandHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxCandHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxCandHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Compile-time interface check
var _ index.Index = (*HNSWIndex)(nil)
var _ index.VectorExporter = (*HNSWIndex)(nil)

// ExportVectors returns all live (non-deleted) vectors in the HNSW graph for migration.
func (h *HNSWIndex) ExportVectors() []index.ExportedVector {
	h.mu.RLock()
	defer h.mu.RUnlock()

	result := make([]index.ExportedVector, 0, len(h.idToNode))
	for extID, nodeIdx := range h.idToNode {
		if h.deleted[nodeIdx] {
			continue
		}
		vec := make([]float32, h.dim)
		copy(vec, h.nodeVector(nodeIdx))
		result = append(result, index.ExportedVector{ID: extID, Vector: vec})
	}
	return result
}
