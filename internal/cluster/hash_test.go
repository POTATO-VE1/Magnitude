package cluster

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewHashRing(t *testing.T) {
	ring := NewHashRing(100)
	assert.Equal(t, 0, ring.NodeCount())
}

func TestAddNode(t *testing.T) {
	ring := NewHashRing(100)
	ring.AddNode("node-1")
	assert.Equal(t, 1, ring.NodeCount())
	assert.Len(t, ring.ring, 100) // 100 vnodes
}

func TestAddNode_Duplicate(t *testing.T) {
	ring := NewHashRing(100)
	ring.AddNode("node-1")
	ring.AddNode("node-1") // should be idempotent
	assert.Equal(t, 1, ring.NodeCount())
	assert.Len(t, ring.ring, 100)
}

func TestRemoveNode(t *testing.T) {
	ring := NewHashRing(100)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.RemoveNode("node-1")
	assert.Equal(t, 1, ring.NodeCount())
	assert.Len(t, ring.ring, 100) // only node-2's vnodes

	// All keys should now route to node-2
	for i := 0; i < 10; i++ {
		assert.Equal(t, "node-2", ring.GetNode(fmt.Sprintf("key-%d", i)))
	}
}

func TestRemoveNode_NonExistent(t *testing.T) {
	ring := NewHashRing(100)
	ring.RemoveNode("nonexistent") // should not panic
	assert.Equal(t, 0, ring.NodeCount())
}

func TestGetNode_EmptyRing(t *testing.T) {
	ring := NewHashRing(100)
	assert.Equal(t, "", ring.GetNode("any-key"))
}

func TestGetNode_SingleNode(t *testing.T) {
	ring := NewHashRing(100)
	ring.AddNode("node-1")

	// All keys should route to the only node
	for i := 0; i < 100; i++ {
		assert.Equal(t, "node-1", ring.GetNode(fmt.Sprintf("key-%d", i)))
	}
}

func TestGetNode_Deterministic(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	// Same key should always return the same node
	node1 := ring.GetNode("collection-alpha")
	for i := 0; i < 100; i++ {
		assert.Equal(t, node1, ring.GetNode("collection-alpha"))
	}
}

func TestGetNode_Distribution(t *testing.T) {
	ring := NewHashRing(150)
	nodes := []string{"node-1", "node-2", "node-3", "node-4"}
	for _, n := range nodes {
		ring.AddNode(n)
	}

	// Generate many keys and check distribution
	counts := make(map[string]int)
	numKeys := 10000
	for i := 0; i < numKeys; i++ {
		node := ring.GetNode(fmt.Sprintf("collection-%d", i))
		counts[node]++
	}

	// Each node should get roughly 25% of keys (allow 10% deviation)
	expected := float64(numKeys) / float64(len(nodes))
	for _, node := range nodes {
		count := float64(counts[node])
		deviation := (count - expected) / expected
		assert.InDelta(t, 0, deviation, 0.20,
			"node %s got %d/%d keys (%.1f%% deviation)",
			node, counts[node], numKeys, deviation*100)
	}

	t.Logf("Distribution: %v", counts)
}

func TestGetNode_MinimalRebalance(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	// Record initial assignments
	numKeys := 1000
	initial := make(map[string]string)
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("coll-%d", i)
		initial[key] = ring.GetNode(key)
	}

	// Add a third node
	ring.AddNode("node-3")

	// Count how many keys moved
	moved := 0
	for key, oldNode := range initial {
		newNode := ring.GetNode(key)
		if newNode != oldNode {
			moved++
		}
	}

	// With consistent hashing, roughly 1/3 of keys should move (±10%)
	expectedMoved := float64(numKeys) / 3.0
	assert.InDelta(t, expectedMoved, float64(moved), float64(numKeys)*0.15,
		"expected ~%.0f keys to move, got %d", expectedMoved, moved)
	t.Logf("Keys moved: %d/%d (%.1f%%)", moved, numKeys, float64(moved)/float64(numKeys)*100)
}

func TestGetNodes_Replication(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	// GetNodes(key, 2) should return 2 distinct nodes
	nodes := ring.GetNodes("collection-xyz", 2)
	require.Len(t, nodes, 2)
	assert.NotEqual(t, nodes[0], nodes[1], "replication nodes should be distinct")
}

func TestGetNodes_MoreThanAvailable(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	// Requesting 5 replicas but only 2 nodes exist → should return 2
	nodes := ring.GetNodes("key", 5)
	assert.Len(t, nodes, 2)
}

func TestGetNodes_EmptyRing(t *testing.T) {
	ring := NewHashRing(150)
	nodes := ring.GetNodes("key", 3)
	assert.Empty(t, nodes)
}

func TestNodes(t *testing.T) {
	ring := NewHashRing(50)
	ring.AddNode("node-b")
	ring.AddNode("node-a")
	ring.AddNode("node-c")

	nodes := ring.Nodes()
	assert.Equal(t, []string{"node-a", "node-b", "node-c"}, nodes)
}
