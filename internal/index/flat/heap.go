// Package flat implements a brute-force flat index for exact nearest-neighbor search.
// Every query computes the distance to every vector in the index — O(N*D) time.
// This is the correctness oracle for all other index types.
package flat

import (
	"container/heap"
	"sync"

	"github.com/POTATO-VE1/Magnitude/internal/index"
)

// distHeap is a max-heap of SearchResults ordered by distance (worst first).
// A max-heap lets us maintain a Top-K set efficiently: when the heap has K elements
// and a new candidate has a smaller distance than the worst (root), we replace the root.
// This avoids sorting the entire result set and runs in O(N log K) time.
type distHeap []index.SearchResult

func (h distHeap) Len() int            { return len(h) }
func (h distHeap) Less(i, j int) bool  { return h[i].Distance > h[j].Distance } // max-heap: worst first
func (h distHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *distHeap) Push(x any)         { *h = append(*h, x.(index.SearchResult)) }
func (h *distHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// heapPool recycles distHeap slices to avoid per-query heap allocations.
// At high throughput (10K+ queries/sec), each search would allocate a new heap
// slice on the Go heap, creating GC pressure. sync.Pool amortizes this cost.
var heapPool = sync.Pool{
	New: func() any {
		h := make(distHeap, 0, 64) // pre-allocate for typical K values
		return &h
	},
}

// TopK selects the K nearest results from a stream of (id, distance) pairs.
// Uses a max-heap bounded at size K: only candidates closer than the current
// worst result survive. Returns results sorted by distance ascending.
//
// distances and ids must have the same length. K must be > 0.
// metric is the distance metric name for score computation.
func TopK(distances []float32, ids []uint64, k int, metric string) []index.SearchResult {
	if k <= 0 {
		return nil
	}

	hp := heapPool.Get().(*distHeap)
	*hp = (*hp)[:0] // reset length without deallocating backing array
	heap.Init(hp)
	defer heapPool.Put(hp)

	for i, d := range distances {
		if hp.Len() < k {
			heap.Push(hp, index.SearchResult{ID: ids[i], Distance: d})
		} else if d < (*hp)[0].Distance {
			// New candidate is closer than the worst in heap — replace
			(*hp)[0] = index.SearchResult{ID: ids[i], Distance: d}
			heap.Fix(hp, 0) // more efficient than Pop+Push
		}
	}

	// Extract in ascending distance order by popping from the back
	n := hp.Len()
	result := make([]index.SearchResult, n)
	for i := n - 1; i >= 0; i-- {
		result[i] = heap.Pop(hp).(index.SearchResult)
	}

	return result
}
