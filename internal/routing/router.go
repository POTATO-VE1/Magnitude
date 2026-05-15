package routing

import (
	"fmt"
	"github.com/POTATO-VE1/Magnitude/internal/cluster"
)

type NodeResolver interface {
	GetAddress(nodeID string) string
}

// Router maps vector IDs to physical node IDs based on the consistent hash ring.
type Router struct {
	ring        *cluster.HashRing
	localNodeID string
	localAddr   string
	resolver    NodeResolver
}

// NewRouter creates a new router using the provided hash ring.
func NewRouter(ring *cluster.HashRing, localNodeID, localAddr string, resolver NodeResolver) *Router {
	return &Router{
		ring:        ring,
		localNodeID: localNodeID,
		localAddr:   localAddr,
		resolver:    resolver,
	}
}

// GetNodeForVector returns the single primary owner node ID for a given vector.
func (r *Router) GetNodeForVector(collectionID string, vectorID uint64) string {
	// Format: {collection_id}#{vector_id}
	// We include the collection ID so vectors with the same ID in different
	// collections don't always land on the same node (better distribution).
	key := fmt.Sprintf("%s#%d", collectionID, vectorID)
	nodeID := r.ring.GetNode(key)
	
	// Fallback to local node if ring is empty (standalone mode)
	if nodeID == "" {
		return r.localNodeID
	}
	return nodeID
}

// IsLocal checks if the target node is the local node.
func (r *Router) IsLocal(nodeID string) bool {
	return nodeID == r.localNodeID
}

// GetAddress returns the HTTP API address for a given Node ID.
func (r *Router) GetAddress(nodeID string) string {
	if r.IsLocal(nodeID) {
		return r.localAddr
	}
	if r.resolver != nil {
		return r.resolver.GetAddress(nodeID)
	}
	return ""
}

// GetAllNodes returns a list of all active node IDs in the cluster.
// Useful for scatter-gather search operations.
func (r *Router) GetAllNodes() []string {
	nodes := r.ring.Nodes()
	if len(nodes) == 0 {
		return []string{r.localNodeID} // Standalone mode
	}
	return nodes
}

// BucketVectors takes a batch of vectors and groups them by their target node ID.
// Returns a map where key = targetNodeID, value = indices of the vectors in the original slice.
func (r *Router) BucketVectors(collectionID string, vectorIDs []uint64) map[string][]int {
	buckets := make(map[string][]int)
	for i, vid := range vectorIDs {
		target := r.GetNodeForVector(collectionID, vid)
		buckets[target] = append(buckets[target], i)
	}
	return buckets
}
