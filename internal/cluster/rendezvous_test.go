package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRendezvous_Route_Basic(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b", "node-c"})

	result := r.Route("collection-1")
	assert.NotEmpty(t, result)
	assert.Contains(t, []string{"node-a", "node-b", "node-c"}, result)
}

func TestRendezvous_Route_Deterministic(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b", "node-c"})

	// Same key always maps to the same node
	first := r.Route("my-collection")
	for i := 0; i < 100; i++ {
		assert.Equal(t, first, r.Route("my-collection"))
	}
}

func TestRendezvous_Route_Empty(t *testing.T) {
	r := NewRendezvousRouter(nil)
	assert.Empty(t, r.Route("anything"))
}

func TestRendezvous_AddNode(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b"})
	assert.Equal(t, 2, r.NodeCount())

	r.AddNode("node-c")
	assert.Equal(t, 3, r.NodeCount())

	// Duplicate add is a no-op
	r.AddNode("node-c")
	assert.Equal(t, 3, r.NodeCount())
}

func TestRendezvous_RemoveNode(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b", "node-c"})
	r.RemoveNode("node-b")
	assert.Equal(t, 2, r.NodeCount())

	nodes := r.Nodes()
	assert.NotContains(t, nodes, "node-b")

	// Remove non-existent is a no-op
	r.RemoveNode("node-z")
	assert.Equal(t, 2, r.NodeCount())
}

func TestRendezvous_RouteN(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b", "node-c", "node-d"})

	replicas := r.RouteN("my-collection", 3)
	require.Len(t, replicas, 3)

	// All unique
	seen := make(map[string]bool)
	for _, n := range replicas {
		assert.False(t, seen[n], "duplicate node in RouteN result")
		seen[n] = true
	}

	// First replica should match Route()
	assert.Equal(t, r.Route("my-collection"), replicas[0])
}

func TestRendezvous_RouteN_ExceedsNodeCount(t *testing.T) {
	r := NewRendezvousRouter([]string{"node-a", "node-b"})
	replicas := r.RouteN("key", 5)
	assert.Len(t, replicas, 2) // capped at node count
}

func TestRendezvous_MinimalRebalance(t *testing.T) {
	// When adding a node, only ~1/N of keys should reroute.
	nodes := []string{"node-a", "node-b", "node-c", "node-d"}
	r := NewRendezvousRouter(nodes)

	numKeys := 10000
	beforeRouting := make(map[string]string)
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("collection-%d", i)
		beforeRouting[key] = r.Route(key)
	}

	// Add a 5th node
	r.AddNode("node-e")

	changed := 0
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("collection-%d", i)
		if r.Route(key) != beforeRouting[key] {
			changed++
		}
	}

	// Expected: ~1/5 = 20% of keys reroute. Allow ±10%.
	expectedFraction := 1.0 / 5.0
	actualFraction := float64(changed) / float64(numKeys)
	t.Logf("Rebalanced: %d/%d keys (%.1f%%, expected ~%.1f%%)",
		changed, numKeys, actualFraction*100, expectedFraction*100)

	assert.InDelta(t, expectedFraction, actualFraction, 0.10,
		"rebalance fraction should be approximately 1/N")
}

func TestRendezvous_Distribution(t *testing.T) {
	nodes := []string{"node-a", "node-b", "node-c", "node-d"}
	r := NewRendezvousRouter(nodes)

	counts := make(map[string]int)
	numKeys := 10000
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("collection-%d", i)
		counts[r.Route(key)]++
	}

	// Expect roughly even distribution: each node ~25%
	expectedPerNode := float64(numKeys) / float64(len(nodes))
	for _, node := range nodes {
		got := float64(counts[node])
		deviation := math.Abs(got-expectedPerNode) / expectedPerNode
		t.Logf("  %s: %d keys (%.1f%% deviation)", node, counts[node], deviation*100)
		assert.Less(t, deviation, 0.20, "node %s has >20%% deviation from ideal", node)
	}
}

func TestRendezvous_Nodes(t *testing.T) {
	r := NewRendezvousRouter([]string{"c", "a", "b"})
	nodes := r.Nodes()
	// Nodes should be sorted
	assert.Equal(t, []string{"a", "b", "c"}, nodes)

	// Returned slice is a copy
	nodes[0] = "modified"
	assert.Equal(t, []string{"a", "b", "c"}, r.Nodes())
}

func TestRendezvous_CacheCoherence(t *testing.T) {
	// All queries for the same collection should hit the same node.
	// This is the key advantage over round-robin.
	r := NewRendezvousRouter([]string{"node-1", "node-2", "node-3"})

	target := r.Route("hot-collection")
	for i := 0; i < 1000; i++ {
		assert.Equal(t, target, r.Route("hot-collection"),
			"same collection should always route to same node")
	}
}

func BenchmarkRendezvous_Route_4Nodes(b *testing.B) {
	r := NewRendezvousRouter([]string{"node-a", "node-b", "node-c", "node-d"})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Route("collection-benchmark")
	}
}

func BenchmarkRendezvous_Route_100Nodes(b *testing.B) {
	nodes := make([]string, 100)
	for i := range nodes {
		nodes[i] = fmt.Sprintf("node-%d", i)
	}
	r := NewRendezvousRouter(nodes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Route("collection-benchmark")
	}
}
