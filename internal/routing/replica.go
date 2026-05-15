package routing

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/POTATO-VE1/Magnitude/internal/cluster"
	"github.com/POTATO-VE1/Magnitude/internal/index"
)

// WriteWithConsistency sends an insert to replicas and waits for the required
// number of acknowledgments based on the consistency level.
//
// Flow:
//  1. Determine replica nodes via hash ring
//  2. Send insert to all replicas in parallel
//  3. Wait for `required` acks (determined by consistency level)
//  4. Return success if enough acks received, error otherwise
func (r *Router) WriteWithConsistency(
	ctx context.Context,
	collectionID string,
	ids []uint64,
	vectors [][]float32,
	meta []map[string]any,
	level cluster.ConsistencyLevel,
	forwarder *Forwarder,
	authHeader string,
) error {
	if forwarder == nil {
		return fmt.Errorf("routing: forwarder not available")
	}

	nodes := r.GetAllNodes()
	required := cluster.ResolveConsistency(level, len(nodes))

	if required <= 0 {
		required = 1
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	acks := 0
	var firstErr error

	for _, nodeID := range nodes {
		wg.Add(1)
		go func(nid string) {
			defer wg.Done()

			var err error
			if r.IsLocal(nid) {
				// Local insert — handled by the caller
				err = nil
			} else {
				addr := r.GetAddress(nid)
				if addr == "" {
					err = fmt.Errorf("node %s unreachable", nid)
				} else {
					req := InsertRequest{
						IDs:      ids,
						Vectors:  vectors,
						Metadata: meta,
					}
					err = forwarder.ForwardInsertBatch(ctx, addr, "", "", collectionID, req, authHeader)
				}
			}

			mu.Lock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
			} else {
				acks++
			}
			mu.Unlock()
		}(nodeID)
	}

	wg.Wait()

	if acks < required {
		return fmt.Errorf("routing: insufficient acks: got %d, need %d: %w", acks, required, firstErr)
	}

	return nil
}

// ReadWithConsistency sends a search to replicas and merges results based on
// the consistency level.
//
// Flow:
//  1. Determine replica nodes via hash ring
//  2. Send search to `required` replicas in parallel
//  3. Merge results by distance
//  4. Return top-K
func (r *Router) ReadWithConsistency(
	ctx context.Context,
	collectionID string,
	query []float32,
	k int,
	nprobe int,
	filter map[string]any,
	level cluster.ConsistencyLevel,
	forwarder *Forwarder,
	authHeader string,
) ([]index.SearchResult, error) {
	if forwarder == nil {
		return nil, fmt.Errorf("routing: forwarder not available")
	}

	nodes := r.GetAllNodes()
	required := cluster.ResolveConsistency(level, len(nodes))

	if required <= 0 {
		required = 1
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var allResults []index.SearchResult
	responses := 0
	var firstErr error

	for _, nodeID := range nodes {
		wg.Add(1)
		go func(nid string) {
			defer wg.Done()

			var results []index.SearchResult
			var err error

			if r.IsLocal(nid) {
				// Local search — handled by the caller via manager
				// Return empty here; caller will add local results
				results = nil
			} else {
				addr := r.GetAddress(nid)
				if addr == "" {
					err = fmt.Errorf("node %s unreachable", nid)
				} else {
					req := SearchRequest{
						Query:  query,
						K:      k,
						Nprobe: nprobe,
						Filter: filter,
					}
					results, err = forwarder.ForwardSearch(ctx, addr, "", "", collectionID, req, authHeader)
				}
			}

			mu.Lock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
			} else {
				allResults = append(allResults, results...)
				responses++
			}
			mu.Unlock()
		}(nodeID)
	}

	wg.Wait()

	if responses < required {
		return nil, fmt.Errorf("routing: insufficient responses: got %d, need %d: %w", responses, required, firstErr)
	}

	// Sort by distance and take top-K
	if len(allResults) > k {
		sortByDistance(allResults)
		allResults = allResults[:k]
	}

	return allResults, nil
}

// sortByDistance sorts results by distance ascending (closest first).
func sortByDistance(results []index.SearchResult) {
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].Distance < results[j-1].Distance; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}
}

// LogReplicaWrite logs a replica write operation for observability.
func LogReplicaWrite(collectionID string, nodeID string, acks int, required int) {
	slog.Debug("replica write",
		"collection", collectionID,
		"node", nodeID,
		"acks", acks,
		"required", required,
	)
}
