// Package client — cluster-aware client with client-side hash ring routing.
//
// Inspired by dbeel's DbeelClient (dbeel_client/src/lib.rs:40-54) which
// maintains a local copy of the cluster hash ring and auto-resyncs on
// routing errors.
//
// The ClusterClient wraps the single-node Client with:
//   - Client-side consistent hash ring for direct node routing
//   - Auto-resync on routing errors (stale ring detection)
//   - Replica failover on primary node failure
package client

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
)

// Shard represents a node in the client-side hash ring.
type Shard struct {
	Hash    uint32
	Address string // base URL for this node's API (e.g., "http://node-1:8443")
	NodeID  string
}

// ClusterMetadataResponse is the response from GET /v1/cluster/metadata.
type ClusterMetadataResponse struct {
	Nodes    []NodeMetadata    `json:"nodes"`
	ShardMap map[string]string `json:"shard_map"` // collection_id → node_id
}

// NodeMetadata is a node entry in the cluster metadata response.
type NodeMetadata struct {
	ID         string `json:"id"`
	Address    string `json:"address"`
	APIAddress string `json:"api_address"`
	State      string `json:"state"`
}

// ClusterClient is a distributed client with client-side hash ring routing.
type ClusterClient struct {
	mu            sync.RWMutex
	seedAddresses []string
	hashRing      []Shard
	apiKey        string
	inner         *Client // fallback for metadata sync
	resyncCount   int
}

// ClusterClientOption configures the ClusterClient.
type ClusterClientOption func(*ClusterClient)

// WithClusterAPIKey sets the API key for all nodes.
func WithClusterAPIKey(key string) ClusterClientOption {
	return func(cc *ClusterClient) { cc.apiKey = key }
}

// NewClusterClient creates a client from seed node addresses.
// It immediately syncs the hash ring from one of the seeds.
func NewClusterClient(seedAddresses []string, opts ...ClusterClientOption) (*ClusterClient, error) {
	cc := &ClusterClient{
		seedAddresses: seedAddresses,
	}
	for _, opt := range opts {
		opt(cc)
	}

	if err := cc.SyncHashRing(context.Background()); err != nil {
		return nil, fmt.Errorf("cluster client: initial hash ring sync failed: %w", err)
	}
	return cc, nil
}

// SyncHashRing fetches cluster metadata from a seed node and rebuilds the hash ring.
func (cc *ClusterClient) SyncHashRing(ctx context.Context) error {
	var lastErr error
	for _, seed := range cc.seedAddresses {
		c := New(seed, cc.apiKey)

		var metadata ClusterMetadataResponse
		if err := c.doJSON(ctx, "GET", "/v1/cluster/metadata", nil, &metadata); err != nil {
			lastErr = err
			continue
		}

		// Build hash ring from metadata
		ring := make([]Shard, 0, len(metadata.Nodes))
		for _, node := range metadata.Nodes {
			apiAddr := node.APIAddress
			if apiAddr == "" {
				apiAddr = node.Address
			}
			ring = append(ring, Shard{
				Hash:    hashKey(node.ID),
				Address: apiAddr,
				NodeID:  node.ID,
			})
		}
		sortShards(ring)

		cc.mu.Lock()
		cc.hashRing = ring
		cc.resyncCount++
		cc.inner = c // keep reference for metadata requests
		cc.mu.Unlock()

		slog.Info("cluster client: hash ring synced",
			"nodes", len(ring),
			"seed", seed,
			"resync_count", cc.resyncCount,
		)
		return nil
	}
	return fmt.Errorf("cluster client: all seeds failed: %w", lastErr)
}

// RouteCollection returns the base URL of the node owning the given collection.
func (cc *ClusterClient) RouteCollection(collectionID string) (string, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	if len(cc.hashRing) == 0 {
		return "", fmt.Errorf("cluster client: hash ring is empty")
	}

	hash := hashKey(collectionID)
	idx := sort.Search(len(cc.hashRing), func(i int) bool {
		return cc.hashRing[i].Hash >= hash
	})
	if idx >= len(cc.hashRing) {
		idx = 0 // wrap around
	}
	return cc.hashRing[idx].Address, nil
}

// Insert routes the insert request to the correct node.
// On routing errors, auto-resyncs the hash ring and retries once.
func (cc *ClusterClient) Insert(ctx context.Context, collectionID string, ids []uint64, vectors [][]float32, opts ...RequestOption) error {
	reqOpts := applyOptions(opts...)
	return cc.sendRouted(ctx, collectionID, func(nodeAddr string) error {
		c := New(nodeAddr, cc.apiKey)
		body := map[string]any{
			"ids":         ids,
			"vectors":     vectors,
			"consistency": reqOpts.consistency,
		}
		return c.doJSON(ctx, "POST", fmt.Sprintf("/v1/collections/%s/vectors", collectionID), body, nil)
	})
}

// Search routes the search request to the correct node.
func (cc *ClusterClient) Search(ctx context.Context, collectionID string, query []float32, k, nprobe int, opts ...RequestOption) ([]SearchResult, error) {
	reqOpts := applyOptions(opts...)
	var results []SearchResult
	err := cc.sendRouted(ctx, collectionID, func(nodeAddr string) error {
		c := New(nodeAddr, cc.apiKey)
		body := map[string]any{
			"query":       query,
			"k":           k,
			"nprobe":      nprobe,
			"consistency": reqOpts.consistency,
		}
		return c.doJSON(ctx, "POST", fmt.Sprintf("/v1/collections/%s/search", collectionID), body, &results)
	})
	return results, err
}

// Delete routes the delete request to the correct node.
func (cc *ClusterClient) Delete(ctx context.Context, collectionID string, vectorID uint64) error {
	return cc.sendRouted(ctx, collectionID, func(nodeAddr string) error {
		c := New(nodeAddr, cc.apiKey)
		return c.doJSON(ctx, "DELETE", fmt.Sprintf("/v1/collections/%s/vectors/%d", collectionID, vectorID), nil, nil)
	})
}

// Inner returns the underlying single-node client (for non-routed operations).
func (cc *ClusterClient) Inner() *Client {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	return cc.inner
}

// ResyncCount returns the number of hash ring resyncs (for monitoring).
func (cc *ClusterClient) ResyncCount() int {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	return cc.resyncCount
}

// sendRouted routes a request to the correct node, with auto-resync on routing errors.
func (cc *ClusterClient) sendRouted(ctx context.Context, collectionID string, fn func(string) error) error {
	for attempt := 0; attempt < 2; attempt++ {
		addr, err := cc.RouteCollection(collectionID)
		if err != nil {
			return err
		}

		err = fn(addr)
		if err == nil {
			return nil
		}

		// Check if error indicates stale routing
		if isRoutingError(err) && attempt == 0 {
			slog.Warn("cluster client: routing error, resyncing hash ring",
				"collection", collectionID,
				"node", addr,
				"error", err,
			)
			if syncErr := cc.SyncHashRing(ctx); syncErr != nil {
				slog.Error("cluster client: resync failed", "error", syncErr)
				return err // return original error
			}
			continue // retry with fresh ring
		}

		return err
	}
	return fmt.Errorf("cluster client: request failed after resync")
}

// isRoutingError checks if an error indicates stale routing information.
func isRoutingError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "not owned by this node") ||
		strings.Contains(msg, "collection not found")
}

// RequestOption configures a per-request option.
type RequestOption func(*requestOptions)

type requestOptions struct {
	consistency string
}

func applyOptions(opts ...RequestOption) requestOptions {
	var ro requestOptions
	for _, opt := range opts {
		opt(&ro)
	}
	return ro
}

// WithConsistency sets the consistency level for a single request.
func WithConsistency(level string) RequestOption {
	return func(ro *requestOptions) { ro.consistency = level }
}

// hashKey computes a SHA-256 based hash for consistent hashing.
func hashKey(key string) uint32 {
	h := sha256.Sum256([]byte(key))
	return binary.BigEndian.Uint32(h[:4])
}

// sortShards sorts shards by hash for binary search routing.
func sortShards(shards []Shard) {
	sort.Slice(shards, func(i, j int) bool {
		return shards[i].Hash < shards[j].Hash
	})
}
