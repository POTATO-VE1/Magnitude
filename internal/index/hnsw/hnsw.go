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
	vector  []float32
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
	rng            *rand.Rand
	deleted        map[int]bool // soft-delete tombstones by internal nodeIdx
	snapshotPath   string       // file path for snapshot persistence
	snapshotSeqID  uint64       // WAL seqID at last snapshot time
	searchGen      uint64       // monotonically increasing search generation
	visited        []uint64     // visited[nodeIdx] == searchGen means visited
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
		efConstruction = 200
	}
	if efSearch <= 0 {
		efSearch = 50
	}

	return &HNSWIndex{
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
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
		deleted:        make(map[int]bool),
		visited:        make([]uint64, 1024),
	}, nil
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

// ensureVisited grows the visited slice if needed to cover all nodes.
// Must be called with h.mu held (at least RLock).
func (h *HNSWIndex) ensureVisited() {
	if len(h.visited) < len(h.nodes) {
		h.visited = make([]uint64, len(h.nodes))
	}
}

// Insert adds a vector with the given ID into the HNSW graph.
func (h *HNSWIndex) Insert(id uint64, vector []float32) error {
	if len(vector) != h.dim {
		return vdberrors.Newf(vdberrors.ErrDimensionMismatch,
			"expected dimension %d, got %d", h.dim, len(vector))
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	return h.insertLocked(id, vector)
}

// insertLocked performs the actual insert. Must be called with h.mu held.
func (h *HNSWIndex) insertLocked(id uint64, vector []float32) error {
	if nodeIdx, exists := h.idToNode[id]; exists {
		if h.deleted[nodeIdx] {
			delete(h.idToNode, id)
		} else {
			return vdberrors.Newf(vdberrors.ErrDuplicateID, "vector ID %d already exists", id)
		}
	}

	h.ensureVisited()

	// Assign level
	level := h.randomLevel()
	nodeIdx := len(h.nodes)

	// Create node with friend lists for each layer
	n := node{
		id:      id,
		vector:  make([]float32, h.dim),
		level:   level,
		friends: make([][]int, level+1),
	}
	copy(n.vector, vector)
	for i := range n.friends {
		n.friends[i] = make([]int, 0, h.m)
	}

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
	for lc := topLayer; lc >= 0; lc-- {
		candidates := h.searchLayer(context.Background(), vector, ep, h.efConstruction, lc)
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
					h.nodes[neighborIdx].vector, h.nodes[neighborIdx].friends[lc], maxConn)
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

	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 {
		return nil, nil
	}

	ef := h.efSearch
	if nprobe > 0 {
		ef = nprobe
	}
	if ef < k {
		ef = k
	}

	// Ensure visited slice is large enough for this search
	h.ensureVisited()

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

	// Phase 2: Beam search at layer 0
	candidates := h.searchLayer(ctx, query, ep, ef, 0)

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

	return results, nil
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

// Len returns the number of live (non-deleted) vectors.
func (h *HNSWIndex) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes) - len(h.deleted)
}

// Rebuild reconstructs the HNSW graph from scratch, excluding deleted nodes.
func (h *HNSWIndex) Rebuild() error {
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
			}{n.id, n.vector})
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
	if h.snapshotPath == "" {
		return nil
	}
	return h.SnapshotToFile(h.snapshotPath, h.snapshotSeqID)
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
	epDist := h.distFn(query, h.nodes[ep].vector)
	changed := true
	for changed {
		changed = false
		if layer < len(h.nodes[ep].friends) {
			for _, friendIdx := range h.nodes[ep].friends[layer] {
				d := h.distFn(query, h.nodes[friendIdx].vector)
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
func (h *HNSWIndex) searchLayer(ctx context.Context, query []float32, ep int, ef int, layer int) []candidate {
	h.searchGen++
	gen := h.searchGen
	h.visited[ep] = gen

	epDist := h.distFn(query, h.nodes[ep].vector)

	// candidateHeap: min-heap of unexplored candidates
	cands := &minCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(cands)

	// results: max-heap of best results so far (worst distance at root)
	results := &maxCandHeap{{nodeIdx: ep, dist: epDist}}
	heap.Init(results)

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)

		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}

		if layer < len(h.nodes[c.nodeIdx].friends) {
			for i, friendIdx := range h.nodes[c.nodeIdx].friends[layer] {
				if i%100 == 0 {
					select {
					case <-ctx.Done():
						return nil
					default:
					}
				}
				if h.visited[friendIdx] == gen {
					continue
				}
				h.visited[friendIdx] = gen

				d := h.distFn(query, h.nodes[friendIdx].vector)
				friend := candidate{nodeIdx: friendIdx, dist: d}

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
		cv := h.nodes[c.nodeIdx].vector
		distToCandQuery := c.dist

		closer := false
		for _, rIdx := range result {
			rv := h.nodes[rIdx].vector
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
		candidates[i] = candidate{nodeIdx: f, dist: h.distFn(nodeVec, h.nodes[f].vector)}
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
		n := h.nodes[nodeIdx]
		vec := make([]float32, len(n.vector))
		copy(vec, n.vector)
		result = append(result, index.ExportedVector{ID: extID, Vector: vec})
	}
	return result
}
