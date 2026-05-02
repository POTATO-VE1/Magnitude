// Package cluster — Coordinator manages cluster membership and shard routing.
//
// The Coordinator is the single source of truth for:
//   - Which nodes are alive (heartbeat-based health tracking)
//   - Which node owns each collection (via consistent hash ring)
//   - Replication factor (how many copies of each collection exist)
//
// Architecture (ChromaDB Cloud mapping):
//   Coordinator ≈ ChromaDB's "chroma-coordinator" service
//   It uses a consistent hash ring for shard assignment, with automatic
//   rebalancing when nodes join or leave the cluster.
//
// In single-node mode (Phase 1-9), the coordinator is unused.
// In distributed mode (Phase 10+), it coordinates the cluster.
package cluster

import (
	"fmt"
	"sync"
	"time"
)

// NodeState represents the health state of a cluster node.
type NodeState int

const (
	// NodeAlive indicates the node is healthy and accepting traffic.
	NodeAlive NodeState = iota
	// NodeSuspect indicates the node missed recent heartbeats.
	NodeSuspect
	// NodeDead indicates the node has been marked dead after multiple missed heartbeats.
	NodeDead
)

func (s NodeState) String() string {
	switch s {
	case NodeAlive:
		return "alive"
	case NodeSuspect:
		return "suspect"
	case NodeDead:
		return "dead"
	default:
		return "unknown"
	}
}

// NodeInfo holds metadata about a cluster node.
type NodeInfo struct {
	ID            string    `json:"id"`
	Address       string    `json:"address"`        // host:port for inter-node RPC
	APIAddress    string    `json:"api_address"`     // host:port for client API
	State         NodeState `json:"state"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	JoinedAt      time.Time `json:"joined_at"`
	Collections   int       `json:"collections"`     // number of collections owned
}

// Coordinator manages cluster membership and collection-to-node mapping.
type Coordinator struct {
	mu              sync.RWMutex
	nodeID          string              // this node's ID
	nodes           map[string]*NodeInfo // all known nodes
	ring            *HashRing           // consistent hash ring
	replicationFactor int
	heartbeatTimeout  time.Duration     // duration before marking a node suspect
	deadTimeout       time.Duration     // duration before marking a node dead
}

// CoordinatorConfig configures the cluster coordinator.
type CoordinatorConfig struct {
	NodeID            string
	ReplicationFactor int
	VirtualNodes      int
	HeartbeatTimeout  time.Duration
	DeadTimeout       time.Duration
}

// DefaultCoordinatorConfig returns sensible defaults for a cluster coordinator.
func DefaultCoordinatorConfig() CoordinatorConfig {
	return CoordinatorConfig{
		NodeID:            "node-1",
		ReplicationFactor: 3,
		VirtualNodes:      150,
		HeartbeatTimeout:  5 * time.Second,
		DeadTimeout:       15 * time.Second,
	}
}

// NewCoordinator creates a new cluster coordinator.
func NewCoordinator(cfg CoordinatorConfig) *Coordinator {
	if cfg.ReplicationFactor <= 0 {
		cfg.ReplicationFactor = 3
	}
	if cfg.VirtualNodes <= 0 {
		cfg.VirtualNodes = 150
	}
	if cfg.HeartbeatTimeout <= 0 {
		cfg.HeartbeatTimeout = 5 * time.Second
	}
	if cfg.DeadTimeout <= 0 {
		cfg.DeadTimeout = 15 * time.Second
	}

	return &Coordinator{
		nodeID:            cfg.NodeID,
		nodes:             make(map[string]*NodeInfo),
		ring:              NewHashRing(cfg.VirtualNodes),
		replicationFactor: cfg.ReplicationFactor,
		heartbeatTimeout:  cfg.HeartbeatTimeout,
		deadTimeout:       cfg.DeadTimeout,
	}
}

// JoinNode registers a new node (or re-registers an existing one) in the cluster.
func (c *Coordinator) JoinNode(nodeID, address, apiAddress string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if nodeID == "" {
		return fmt.Errorf("cluster: node ID must not be empty")
	}

	existing, exists := c.nodes[nodeID]
	if exists {
		// Re-join: update address and mark alive
		existing.Address = address
		existing.APIAddress = apiAddress
		existing.State = NodeAlive
		existing.LastHeartbeat = time.Now()
		return nil
	}

	now := time.Now()
	c.nodes[nodeID] = &NodeInfo{
		ID:            nodeID,
		Address:       address,
		APIAddress:    apiAddress,
		State:         NodeAlive,
		LastHeartbeat: now,
		JoinedAt:      now,
	}

	c.ring.AddNode(nodeID)
	return nil
}

// LeaveNode removes a node from the cluster.
func (c *Coordinator) LeaveNode(nodeID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.nodes[nodeID]; !exists {
		return fmt.Errorf("cluster: node %q not found", nodeID)
	}

	delete(c.nodes, nodeID)
	c.ring.RemoveNode(nodeID)
	return nil
}

// Heartbeat records a heartbeat from a node, marking it alive.
func (c *Coordinator) Heartbeat(nodeID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	node, exists := c.nodes[nodeID]
	if !exists {
		return fmt.Errorf("cluster: node %q not found", nodeID)
	}

	node.LastHeartbeat = time.Now()
	node.State = NodeAlive
	return nil
}

// GetNode returns info about a specific node.
func (c *Coordinator) GetNode(nodeID string) (*NodeInfo, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	node, exists := c.nodes[nodeID]
	if !exists {
		return nil, false
	}

	// Return a copy
	info := *node
	return &info, true
}

// GetAliveNodes returns all nodes in the Alive state.
func (c *Coordinator) GetAliveNodes() []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []*NodeInfo
	for _, node := range c.nodes {
		if node.State == NodeAlive {
			info := *node
			result = append(result, &info)
		}
	}
	return result
}

// GetAllNodes returns all nodes regardless of state.
func (c *Coordinator) GetAllNodes() []*NodeInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]*NodeInfo, 0, len(c.nodes))
	for _, node := range c.nodes {
		info := *node
		result = append(result, &info)
	}
	return result
}

// RouteCollection returns the node ID that should own the given collection.
func (c *Coordinator) RouteCollection(collectionID string) string {
	return c.ring.GetNode(collectionID)
}

// RouteCollectionReplicas returns the N node IDs that should hold replicas
// of the given collection (N = replication factor).
func (c *Coordinator) RouteCollectionReplicas(collectionID string) []string {
	return c.ring.GetNodes(collectionID, c.replicationFactor)
}

// CheckHealth evaluates all nodes and updates their state based on heartbeat timestamps.
// Should be called periodically (e.g., every second).
func (c *Coordinator) CheckHealth() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for _, node := range c.nodes {
		elapsed := now.Sub(node.LastHeartbeat)
		switch {
		case elapsed >= c.deadTimeout:
			if node.State != NodeDead {
				node.State = NodeDead
				c.ring.RemoveNode(node.ID) // remove from routing
			}
		case elapsed >= c.heartbeatTimeout:
			node.State = NodeSuspect
		default:
			node.State = NodeAlive
		}
	}
}

// NodeCount returns the number of registered nodes (all states).
func (c *Coordinator) NodeCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.nodes)
}

// ReplicationFactor returns the configured replication factor.
func (c *Coordinator) ReplicationFactor() int {
	return c.replicationFactor
}

// ShardMap returns a map of collection ID → owner node ID for the given collection IDs.
func (c *Coordinator) ShardMap(collectionIDs []string) map[string]string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make(map[string]string, len(collectionIDs))
	for _, id := range collectionIDs {
		result[id] = c.ring.GetNode(id)
	}
	return result
}
