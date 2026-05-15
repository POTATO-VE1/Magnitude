package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// ReplicaClient sends operations to peer nodes for replication.
// Used by the API handlers when consistency level requires multiple acks.
type ReplicaClient struct {
	httpClient *http.Client
	apiKey     string
}

// ReplicaClientOption configures the ReplicaClient.
type ReplicaClientOption func(*ReplicaClient)

// WithReplicaTimeout sets the HTTP timeout for replica requests.
func WithReplicaTimeout(d time.Duration) ReplicaClientOption {
	return func(rc *ReplicaClient) { rc.httpClient.Timeout = d }
}

// WithReplicaAPIKey sets the API key for authenticating with peer nodes.
func WithReplicaAPIKey(key string) ReplicaClientOption {
	return func(rc *ReplicaClient) { rc.apiKey = key }
}

// NewReplicaClient creates a new inter-node RPC client.
func NewReplicaClient(opts ...ReplicaClientOption) *ReplicaClient {
	rc := &ReplicaClient{
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
	for _, opt := range opts {
		opt(rc)
	}
	return rc
}

// ReplicaInsertRequest is the payload for inter-node insert replication.
type ReplicaInsertRequest struct {
	CollectionID string           `json:"collection_id"`
	IDs          []uint64         `json:"ids"`
	Vectors      [][]float32      `json:"vectors"`
	Metadata     []map[string]any `json:"metadata,omitempty"`
}

// ReplicaSearchRequest is the payload for inter-node search replication.
type ReplicaSearchRequest struct {
	CollectionID string         `json:"collection_id"`
	Query        []float32      `json:"query"`
	K            int            `json:"k"`
	Nprobe       int            `json:"nprobe"`
	Filter       map[string]any `json:"filter,omitempty"`
}

// ReplicaDeleteRequest is the payload for inter-node delete replication.
type ReplicaDeleteRequest struct {
	CollectionID string `json:"collection_id"`
	VectorID     uint64 `json:"vector_id"`
}

// ReplicaSearchResponse is the response from a replica search.
type ReplicaSearchResponse struct {
	Results []ReplicaSearchResult `json:"results"`
}

// ReplicaSearchResult is a single search result from a replica.
type ReplicaSearchResult struct {
	ID       uint64         `json:"id"`
	Distance float32        `json:"distance"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Insert sends an insert operation to a peer node.
func (rc *ReplicaClient) Insert(ctx context.Context, nodeAddr string, req ReplicaInsertRequest) error {
	url := fmt.Sprintf("http://%s/internal/v1/replicate/insert", nodeAddr)
	return rc.doJSON(ctx, "POST", url, req, nil)
}

// Search sends a search operation to a peer node and returns results.
func (rc *ReplicaClient) Search(ctx context.Context, nodeAddr string, req ReplicaSearchRequest) ([]ReplicaSearchResult, error) {
	url := fmt.Sprintf("http://%s/internal/v1/replicate/search", nodeAddr)
	var resp ReplicaSearchResponse
	if err := rc.doJSON(ctx, "POST", url, req, &resp); err != nil {
		return nil, err
	}
	return resp.Results, nil
}

// Delete sends a delete operation to a peer node.
func (rc *ReplicaClient) Delete(ctx context.Context, nodeAddr string, req ReplicaDeleteRequest) error {
	url := fmt.Sprintf("http://%s/internal/v1/replicate/delete", nodeAddr)
	return rc.doJSON(ctx, "POST", url, req, nil)
}

// doJSON performs an HTTP request with JSON serialization/deserialization.
func (rc *ReplicaClient) doJSON(ctx context.Context, method, url string, body any, result any) error {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("replica client: marshal: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return fmt.Errorf("replica client: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if rc.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+rc.apiKey)
	}

	resp, err := rc.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("replica client: %s %s: %w", method, url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("replica client: %s %s: status %d: %s", method, url, resp.StatusCode, string(bodyBytes))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("replica client: decode response: %w", err)
		}
	}

	return nil
}

// CollectInsertAcks sends insert to replicas and collects acks until the
// required consistency level is met. Returns the number of successful acks.
func (rc *ReplicaClient) CollectInsertAcks(ctx context.Context, replicas []string, skipNode string, req ReplicaInsertRequest, required int) (int, error) {
	acks := 0
	var lastErr error

	for _, addr := range replicas {
		if addr == skipNode {
			continue
		}
		if err := rc.Insert(ctx, addr, req); err != nil {
			slog.Warn("replica insert failed", "node", addr, "error", err)
			lastErr = err
			continue
		}
		acks++
		if acks >= required {
			return acks, nil
		}
	}

	if acks < required {
		return acks, fmt.Errorf("replica client: insufficient acks: got %d, need %d: %w", acks, required, lastErr)
	}
	return acks, nil
}

// CollectSearchResults sends search to replicas and collects results until the
// required consistency level is met. Returns all results from responding replicas.
func (rc *ReplicaClient) CollectSearchResults(ctx context.Context, replicas []string, skipNode string, req ReplicaSearchRequest, required int) (int, [][]ReplicaSearchResult, error) {
	acks := 0
	var results [][]ReplicaSearchResult
	var lastErr error

	for _, addr := range replicas {
		if addr == skipNode {
			continue
		}
		searchResults, err := rc.Search(ctx, addr, req)
		if err != nil {
			slog.Warn("replica search failed", "node", addr, "error", err)
			lastErr = err
			continue
		}
		results = append(results, searchResults)
		acks++
		if acks >= required {
			return acks, results, nil
		}
	}

	if acks < required {
		return acks, results, fmt.Errorf("replica client: insufficient responses: got %d, need %d: %w", acks, required, lastErr)
	}
	return acks, results, nil
}
