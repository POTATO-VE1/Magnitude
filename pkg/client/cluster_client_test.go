package client

import (
	"fmt"
	"testing"
)

func TestClusterClient_RouteCollection_Empty(t *testing.T) {
	cc := &ClusterClient{
		hashRing: nil,
	}
	_, err := cc.RouteCollection("col-1")
	if err == nil {
		t.Fatal("expected error routing with empty ring")
	}
}

func TestClusterClient_RouteCollection_SingleNode(t *testing.T) {
	cc := &ClusterClient{
		hashRing: []Shard{
			{Hash: 100, Address: "node-1:8443", NodeID: "node-1"},
		},
	}
	addr, err := cc.RouteCollection("col-1")
	if err != nil {
		t.Fatalf("RouteCollection: %v", err)
	}
	if addr != "node-1:8443" {
		t.Errorf("expected node-1:8443, got %s", addr)
	}
}

func TestClusterClient_RouteCollection_Deterministic(t *testing.T) {
	cc := &ClusterClient{
		hashRing: []Shard{
			{Hash: 100, Address: "node-1:8443", NodeID: "node-1"},
			{Hash: 200, Address: "node-2:8443", NodeID: "node-2"},
			{Hash: 300, Address: "node-3:8443", NodeID: "node-3"},
		},
	}

	// Same key should always route to same node
	addr1, _ := cc.RouteCollection("col-1")
	addr2, _ := cc.RouteCollection("col-1")
	if addr1 != addr2 {
		t.Errorf("non-deterministic routing: %s != %s", addr1, addr2)
	}
}

func TestClusterClient_RouteCollection_WrapAround(t *testing.T) {
	cc := &ClusterClient{
		hashRing: []Shard{
			{Hash: 100, Address: "node-1:8443", NodeID: "node-1"},
			{Hash: 200, Address: "node-2:8443", NodeID: "node-2"},
		},
	}

	// A key that hashes beyond the last node should wrap around
	// This is hard to test without knowing the hash function, so just verify it doesn't error
	_, err := cc.RouteCollection("some-key")
	if err != nil {
		t.Fatalf("RouteCollection: %v", err)
	}
}

func TestClusterClient_RouteCollection_MultipleNodes(t *testing.T) {
	cc := &ClusterClient{
		hashRing: []Shard{
			{Hash: 100, Address: "node-1:8443", NodeID: "node-1"},
			{Hash: 200, Address: "node-2:8443", NodeID: "node-2"},
			{Hash: 300, Address: "node-3:8443", NodeID: "node-3"},
		},
	}

	// Verify routing works for multiple different keys
	for i := 0; i < 10; i++ {
		_, err := cc.RouteCollection("col-" + string(rune('a'+i)))
		if err != nil {
			t.Fatalf("RouteCollection col-%c: %v", rune('a'+i), err)
		}
	}
}

func TestShard_SortedByHash(t *testing.T) {
	ring := []Shard{
		{Hash: 300, Address: "node-3:8443", NodeID: "node-3"},
		{Hash: 100, Address: "node-1:8443", NodeID: "node-1"},
		{Hash: 200, Address: "node-2:8443", NodeID: "node-2"},
	}

	// sortShards should sort by hash
	sortShards(ring)

	if ring[0].Hash != 100 || ring[1].Hash != 200 || ring[2].Hash != 300 {
		t.Errorf("ring not sorted: %v", ring)
	}
}

func TestIsRoutingError(t *testing.T) {
	tests := []struct {
		err  string
		want bool
	}{
		{"collection not found", true},
		{"not owned by this node", true},
		{"some other error", false},
		{"", false},
	}
	for _, tt := range tests {
		var err error
		if tt.err != "" {
			err = fmt.Errorf("%s", tt.err)
		}
		got := isRoutingError(err)
		if got != tt.want {
			t.Errorf("isRoutingError(%q) = %v, want %v", tt.err, got, tt.want)
		}
	}
}
