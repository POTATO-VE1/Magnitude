package routing

import (
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/cluster"
)

// ── Router Tests ────────────────────────────────────────────────────────────

func TestRouter_GetNodeForVector_Standalone(t *testing.T) {
	// Empty ring → should fallback to local node
	ring := cluster.NewHashRing(150)
	r := NewRouter(ring, "local-1", "127.0.0.1:8443", nil)

	node := r.GetNodeForVector("col-1", 42)
	if node != "local-1" {
		t.Errorf("expected local-1 fallback, got %q", node)
	}
}

func TestRouter_GetNodeForVector_Deterministic(t *testing.T) {
	ring := cluster.NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	// Same input → same output (deterministic hashing)
	node1 := r.GetNodeForVector("col-1", 100)
	node2 := r.GetNodeForVector("col-1", 100)
	if node1 != node2 {
		t.Errorf("non-deterministic: got %q then %q", node1, node2)
	}
}

func TestRouter_GetNodeForVector_Distribution(t *testing.T) {
	ring := cluster.NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	// Check that vectors distribute across multiple nodes
	counts := make(map[string]int)
	for i := uint64(0); i < 300; i++ {
		node := r.GetNodeForVector("col-1", i)
		counts[node]++
	}

	// With 150 virtual nodes per physical node and 300 vectors,
	// every node should get at least some vectors
	for _, nodeID := range []string{"node-1", "node-2", "node-3"} {
		if counts[nodeID] == 0 {
			t.Errorf("node %s got zero vectors — distribution failure", nodeID)
		}
	}
}

func TestRouter_IsLocal(t *testing.T) {
	ring := cluster.NewHashRing(150)
	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	if !r.IsLocal("node-1") {
		t.Error("expected node-1 to be local")
	}
	if r.IsLocal("node-2") {
		t.Error("expected node-2 to be remote")
	}
}

func TestRouter_GetAddress_Local(t *testing.T) {
	ring := cluster.NewHashRing(150)
	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	addr := r.GetAddress("node-1")
	if addr != "10.0.0.1:8443" {
		t.Errorf("local address = %q, want 10.0.0.1:8443", addr)
	}
}

type mockResolver struct {
	addresses map[string]string
}

func (m *mockResolver) GetAddress(nodeID string) string {
	return m.addresses[nodeID]
}

func TestRouter_GetAddress_Remote(t *testing.T) {
	ring := cluster.NewHashRing(150)
	resolver := &mockResolver{addresses: map[string]string{
		"node-2": "10.0.0.2:8443",
	}}
	r := NewRouter(ring, "node-1", "10.0.0.1:8443", resolver)

	addr := r.GetAddress("node-2")
	if addr != "10.0.0.2:8443" {
		t.Errorf("remote address = %q, want 10.0.0.2:8443", addr)
	}
}

func TestRouter_GetAddress_UnknownNode(t *testing.T) {
	ring := cluster.NewHashRing(150)
	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	addr := r.GetAddress("nonexistent")
	if addr != "" {
		t.Errorf("unknown node address = %q, want empty", addr)
	}
}

func TestRouter_GetAllNodes_Standalone(t *testing.T) {
	ring := cluster.NewHashRing(150)
	r := NewRouter(ring, "local-1", "127.0.0.1:8443", nil)

	nodes := r.GetAllNodes()
	if len(nodes) != 1 || nodes[0] != "local-1" {
		t.Errorf("standalone nodes = %v, want [local-1]", nodes)
	}
}

func TestRouter_GetAllNodes_Cluster(t *testing.T) {
	ring := cluster.NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)
	nodes := r.GetAllNodes()
	if len(nodes) != 2 {
		t.Errorf("cluster nodes count = %d, want 2", len(nodes))
	}
}

func TestRouter_BucketVectors(t *testing.T) {
	ring := cluster.NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	vectorIDs := []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	buckets := r.BucketVectors("col-1", vectorIDs)

	// Verify all vector indices are accounted for
	total := 0
	for _, indices := range buckets {
		total += len(indices)
	}
	if total != len(vectorIDs) {
		t.Errorf("bucketed %d vectors, want %d", total, len(vectorIDs))
	}

	// Verify no index is duplicated
	seen := make(map[int]bool)
	for _, indices := range buckets {
		for _, idx := range indices {
			if seen[idx] {
				t.Errorf("duplicate index %d in buckets", idx)
			}
			seen[idx] = true
		}
	}
}

func TestRouter_DifferentCollections_DifferentNodes(t *testing.T) {
	ring := cluster.NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	r := NewRouter(ring, "node-1", "10.0.0.1:8443", nil)

	// Same vectorID but different collections should not always hash to same node
	// (collection ID is part of the hash key)
	sameNode := 0
	for i := uint64(0); i < 50; i++ {
		n1 := r.GetNodeForVector("col-A", i)
		n2 := r.GetNodeForVector("col-B", i)
		if n1 == n2 {
			sameNode++
		}
	}
	// With 3 nodes, if collections had no effect, 100% would match.
	// With collection in key, ~33% should match by chance.
	if sameNode == 50 {
		t.Error("all 50 vectors land on same node for both collections — collection ID not factored into hash")
	}
}
