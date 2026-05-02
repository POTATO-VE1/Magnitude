// Package cluster provides distributed system primitives for the VectorDB.
//
// Consistent Hashing (hash.go):
//   The hash ring distributes collections across nodes in a cluster.
//   Each node owns a set of virtual nodes (vnodes) on the ring.
//   Collection assignment: hash(collection_id) → walk ring clockwise → first vnode owner.
//
// Virtual nodes (vnodes) solve the imbalance problem: without vnodes,
// nodes with unlucky hash positions get disproportionate load.
// With V vnodes per node, the standard deviation of load drops to O(1/sqrt(V)).
//
// ChromaDB Cloud uses consistent hashing for collection placement across
// "chroma-shard-server" pods. Rendezvous hashing is an alternative but
// consistent hashing has better Go ecosystem support.
package cluster

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
)

// HashRing implements consistent hashing with virtual nodes for collection sharding.
type HashRing struct {
	mu       sync.RWMutex
	vnodes   int              // virtual nodes per physical node
	ring     []uint32         // sorted hash positions
	nodeMap  map[uint32]string // hash position → node ID
	nodes    map[string]bool   // set of active node IDs
}

// NewHashRing creates a hash ring with the specified number of virtual nodes per node.
// Higher vnodes = better load balance but more memory. Typical: 150–256.
func NewHashRing(vnodes int) *HashRing {
	if vnodes <= 0 {
		vnodes = 150
	}
	return &HashRing{
		vnodes:  vnodes,
		ring:    make([]uint32, 0),
		nodeMap: make(map[uint32]string),
		nodes:   make(map[string]bool),
	}
}

// AddNode adds a node to the ring with vnodes virtual nodes.
func (h *HashRing) AddNode(nodeID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.nodes[nodeID] {
		return // already present
	}
	h.nodes[nodeID] = true

	for i := 0; i < h.vnodes; i++ {
		hash := hashKey(fmt.Sprintf("%s#%d", nodeID, i))
		h.ring = append(h.ring, hash)
		h.nodeMap[hash] = nodeID
	}

	sort.Slice(h.ring, func(i, j int) bool {
		return h.ring[i] < h.ring[j]
	})
}

// RemoveNode removes a node and its virtual nodes from the ring.
func (h *HashRing) RemoveNode(nodeID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if !h.nodes[nodeID] {
		return
	}
	delete(h.nodes, nodeID)

	// Remove all vnodes for this node
	newRing := make([]uint32, 0, len(h.ring)-h.vnodes)
	for _, hash := range h.ring {
		if h.nodeMap[hash] != nodeID {
			newRing = append(newRing, hash)
		} else {
			delete(h.nodeMap, hash)
		}
	}
	h.ring = newRing
}

// GetNode returns the node responsible for the given key (e.g., collection ID).
// Returns empty string if the ring is empty.
func (h *HashRing) GetNode(key string) string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.ring) == 0 {
		return ""
	}

	hash := hashKey(key)

	// Binary search for the first vnode with hash >= key hash
	idx := sort.Search(len(h.ring), func(i int) bool {
		return h.ring[i] >= hash
	})

	// Wrap around to first vnode if past the end
	if idx >= len(h.ring) {
		idx = 0
	}

	return h.nodeMap[h.ring[idx]]
}

// GetNodes returns the N distinct nodes responsible for the given key.
// Used for replication: get N nodes for N-way replication.
func (h *HashRing) GetNodes(key string, n int) []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.ring) == 0 || n <= 0 {
		return nil
	}

	hash := hashKey(key)
	idx := sort.Search(len(h.ring), func(i int) bool {
		return h.ring[i] >= hash
	})
	if idx >= len(h.ring) {
		idx = 0
	}

	seen := make(map[string]bool)
	var result []string

	for i := 0; i < len(h.ring) && len(result) < n; i++ {
		pos := (idx + i) % len(h.ring)
		nodeID := h.nodeMap[h.ring[pos]]
		if !seen[nodeID] {
			seen[nodeID] = true
			result = append(result, nodeID)
		}
	}

	return result
}

// NodeCount returns the number of physical nodes in the ring.
func (h *HashRing) NodeCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

// Nodes returns a copy of all node IDs.
func (h *HashRing) Nodes() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	result := make([]string, 0, len(h.nodes))
	for nodeID := range h.nodes {
		result = append(result, nodeID)
	}
	sort.Strings(result)
	return result
}

// hashKey computes a 32-bit hash from a string key using SHA-256.
// We use the first 4 bytes of SHA-256 for the ring position.
func hashKey(key string) uint32 {
	h := sha256.Sum256([]byte(key))
	return binary.BigEndian.Uint32(h[:4])
}
