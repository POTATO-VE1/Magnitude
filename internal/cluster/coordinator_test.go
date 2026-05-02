package cluster

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestCoordinator() *Coordinator {
	return NewCoordinator(CoordinatorConfig{
		NodeID:            "coordinator-1",
		ReplicationFactor: 3,
		VirtualNodes:      100,
		HeartbeatTimeout:  100 * time.Millisecond,
		DeadTimeout:       300 * time.Millisecond,
	})
}

func TestNewCoordinator(t *testing.T) {
	coord := newTestCoordinator()
	assert.Equal(t, 0, coord.NodeCount())
	assert.Equal(t, 3, coord.ReplicationFactor())
}

func TestJoinNode(t *testing.T) {
	coord := newTestCoordinator()
	require.NoError(t, coord.JoinNode("node-1", "10.0.0.1:7000", "10.0.0.1:8443"))
	assert.Equal(t, 1, coord.NodeCount())

	node, ok := coord.GetNode("node-1")
	require.True(t, ok)
	assert.Equal(t, "10.0.0.1:7000", node.Address)
	assert.Equal(t, NodeAlive, node.State)
}

func TestJoinNode_EmptyID(t *testing.T) {
	coord := newTestCoordinator()
	err := coord.JoinNode("", "addr", "api")
	require.Error(t, err)
}

func TestJoinNode_Rejoin(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "10.0.0.1:7000", "10.0.0.1:8443")
	coord.JoinNode("node-1", "10.0.0.2:7000", "10.0.0.2:8443") // re-join with new addr

	node, _ := coord.GetNode("node-1")
	assert.Equal(t, "10.0.0.2:7000", node.Address) // updated
	assert.Equal(t, 1, coord.NodeCount())           // no duplicate
}

func TestLeaveNode(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")

	require.NoError(t, coord.LeaveNode("node-1"))
	assert.Equal(t, 1, coord.NodeCount())

	_, ok := coord.GetNode("node-1")
	assert.False(t, ok)
}

func TestLeaveNode_NotFound(t *testing.T) {
	coord := newTestCoordinator()
	err := coord.LeaveNode("nonexistent")
	require.Error(t, err)
}

func TestHeartbeat(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr", "api")

	// Advance time by making node suspect, then heartbeat
	time.Sleep(150 * time.Millisecond)
	coord.CheckHealth()

	node, _ := coord.GetNode("node-1")
	assert.Equal(t, NodeSuspect, node.State)

	// Heartbeat should restore to alive
	require.NoError(t, coord.Heartbeat("node-1"))
	node, _ = coord.GetNode("node-1")
	assert.Equal(t, NodeAlive, node.State)
}

func TestHeartbeat_NotFound(t *testing.T) {
	coord := newTestCoordinator()
	err := coord.Heartbeat("nonexistent")
	require.Error(t, err)
}

func TestCheckHealth_SuspectAndDead(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr", "api")

	// Wait for suspect timeout
	time.Sleep(150 * time.Millisecond)
	coord.CheckHealth()

	node, _ := coord.GetNode("node-1")
	assert.Equal(t, NodeSuspect, node.State)

	// Wait for dead timeout
	time.Sleep(200 * time.Millisecond)
	coord.CheckHealth()

	node, _ = coord.GetNode("node-1")
	assert.Equal(t, NodeDead, node.State)
}

func TestGetAliveNodes(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")

	alive := coord.GetAliveNodes()
	assert.Len(t, alive, 2)
}

func TestRouteCollection(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")
	coord.JoinNode("node-3", "addr3", "api3")

	// Should deterministically route
	node := coord.RouteCollection("collection-abc")
	assert.NotEmpty(t, node)

	// Same key → same node
	for i := 0; i < 10; i++ {
		assert.Equal(t, node, coord.RouteCollection("collection-abc"))
	}
}

func TestRouteCollectionReplicas(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")
	coord.JoinNode("node-3", "addr3", "api3")

	replicas := coord.RouteCollectionReplicas("collection-xyz")
	require.Len(t, replicas, 3) // replication factor = 3

	// All replicas should be distinct
	seen := make(map[string]bool)
	for _, r := range replicas {
		assert.False(t, seen[r], "replicas should be distinct")
		seen[r] = true
	}
}

func TestShardMap(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")

	collections := []string{"col-1", "col-2", "col-3"}
	shardMap := coord.ShardMap(collections)

	assert.Len(t, shardMap, 3)
	for _, colID := range collections {
		assert.NotEmpty(t, shardMap[colID])
	}
}

func TestCheckHealth_RemovesDeadFromRouting(t *testing.T) {
	coord := newTestCoordinator()
	coord.JoinNode("node-1", "addr1", "api1")
	coord.JoinNode("node-2", "addr2", "api2")

	// Let node-1 die
	time.Sleep(350 * time.Millisecond)
	// Keep node-2 alive via heartbeat
	coord.Heartbeat("node-2")
	coord.CheckHealth()

	// All routing should go to node-2 now
	for i := 0; i < 20; i++ {
		node := coord.RouteCollection(string(rune('a' + i)))
		assert.Equal(t, "node-2", node, "dead node-1 should be removed from routing")
	}
}

func TestDefaultCoordinatorConfig(t *testing.T) {
	cfg := DefaultCoordinatorConfig()
	assert.Equal(t, "node-1", cfg.NodeID)
	assert.Equal(t, 3, cfg.ReplicationFactor)
	assert.Equal(t, 150, cfg.VirtualNodes)
	assert.Equal(t, 5*time.Second, cfg.HeartbeatTimeout)
	assert.Equal(t, 15*time.Second, cfg.DeadTimeout)
}
