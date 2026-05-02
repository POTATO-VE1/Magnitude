// Package cluster — Rendezvous (highest random weight) hashing router.
//
// Rendezvous hashing routes collection operations to specific nodes.
// Unlike consistent hashing, rendezvous hashing:
//   - Requires no virtual nodes — every physical node participates equally
//   - When a node is added/removed, only 1/N collections re-route (same as consistent)
//   - Maximizes SSD cache coherence: all queries for collection X hit the same node
//   - Hash is deterministic: any client can compute the routing without coordination
//
// Algorithm:
//
//	For each node, compute hash(node_id ⊕ collection_id).
//	The node with the highest hash value "wins" the collection.
package cluster

import (
	"crypto/sha256"
	"encoding/binary"
	"sort"
	"sync"
)

// RendezvousRouter routes operations to nodes using rendezvous hashing.
type RendezvousRouter struct {
	mu    sync.RWMutex
	nodes []string // list of active node IDs
}

// NewRendezvousRouter creates a router with the given initial node set.
func NewRendezvousRouter(nodes []string) *RendezvousRouter {
	nodeCopy := make([]string, len(nodes))
	copy(nodeCopy, nodes)
	sort.Strings(nodeCopy)
	return &RendezvousRouter{nodes: nodeCopy}
}

// AddNode adds a node to the routing pool.
func (r *RendezvousRouter) AddNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	// Check for duplicates
	for _, n := range r.nodes {
		if n == nodeID {
			return
		}
	}
	r.nodes = append(r.nodes, nodeID)
	sort.Strings(r.nodes)
}

// RemoveNode removes a node from the routing pool.
func (r *RendezvousRouter) RemoveNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, n := range r.nodes {
		if n == nodeID {
			r.nodes = append(r.nodes[:i], r.nodes[i+1:]...)
			return
		}
	}
}

// Route returns the node ID that should own the given key (e.g., collection ID).
// Returns "" if no nodes are available.
func (r *RendezvousRouter) Route(key string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.nodes) == 0 {
		return ""
	}

	var maxHash uint64
	var winner string
	for _, node := range r.nodes {
		h := rendezvousHash(node, key)
		if h > maxHash || winner == "" {
			maxHash = h
			winner = node
		}
	}
	return winner
}

// RouteN returns the top-N nodes for a key, ordered by decreasing hash weight.
// Used for N-way replication: the first node is the primary owner,
// subsequent nodes are replicas.
func (r *RendezvousRouter) RouteN(key string, n int) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.nodes) == 0 {
		return nil
	}
	if n > len(r.nodes) {
		n = len(r.nodes)
	}

	type scored struct {
		node string
		hash uint64
	}
	ranked := make([]scored, len(r.nodes))
	for i, node := range r.nodes {
		ranked[i] = scored{node: node, hash: rendezvousHash(node, key)}
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].hash > ranked[j].hash
	})

	result := make([]string, n)
	for i := 0; i < n; i++ {
		result[i] = ranked[i].node
	}
	return result
}

// NodeCount returns the number of active nodes.
func (r *RendezvousRouter) NodeCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.nodes)
}

// Nodes returns a copy of the current node list.
func (r *RendezvousRouter) Nodes() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]string, len(r.nodes))
	copy(result, r.nodes)
	return result
}

// rendezvousHash computes a deterministic hash of (nodeID, key).
// Uses SHA-256 and takes the first 8 bytes as a uint64.
func rendezvousHash(nodeID, key string) uint64 {
	h := sha256.New()
	h.Write([]byte(nodeID))
	h.Write([]byte{0}) // separator to avoid collisions
	h.Write([]byte(key))
	sum := h.Sum(nil)
	return binary.BigEndian.Uint64(sum[:8])
}
