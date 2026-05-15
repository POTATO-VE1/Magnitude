// Package client provides a typed Go client library for the VectorDB HTTP API.
//
// The client handles:
//   - Connection pooling and keep-alive
//   - TLS configuration
//   - Request serialization / response deserialization
//   - Context propagation for cancellation and deadlines
//
// Usage:
//
//	c := client.New("http://localhost:8443", "my-api-key")
//	col, err := c.CreateCollection(ctx, "my-collection", 128, "l2", "flat")
//	err = c.Insert(ctx, col.ID, ids, vectors)
//	results, err := c.Search(ctx, col.ID, query, 10, 0)
package client

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is a typed Go client for the VectorDB HTTP API.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// Option configures the Client.
type Option func(*Client)

// WithTimeout sets the HTTP client timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		c.httpClient.Timeout = d
	}
}

// WithTLSSkipVerify disables TLS certificate verification (for development only).
func WithTLSSkipVerify() Option {
	return func(c *Client) {
		c.httpClient.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
	}
}

// New creates a new VectorDB client.
func New(baseURL, apiKey string, opts ...Option) *Client {
	c := &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ── Response types ──────────────────────────────────────────────────────────

// Envelope mirrors the server's JSON response envelope.
type Envelope struct {
	Data  json.RawMessage `json:"data,omitempty"`
	Error string          `json:"error,omitempty"`
}

// Collection represents a collection returned by the API.
type Collection struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Dimension   int    `json:"dimension"`
	Metric      string `json:"metric"`
	IndexType   string `json:"index_type"`
	VectorCount int    `json:"vector_count"`
	CreatedAt   int64  `json:"created_at"`
}

// SearchResult represents a single search result.
type SearchResult struct {
	ID       uint64  `json:"ID"`
	Distance float32 `json:"Distance"`
	Score    float32 `json:"Score"`
}

// ── Collection operations ───────────────────────────────────────────────────

// CreateCollection creates a new collection.
func (c *Client) CreateCollection(ctx context.Context, name string, dim int, metric, indexType string) (*Collection, error) {
	body := map[string]any{
		"name":       name,
		"dimension":  dim,
		"metric":     metric,
		"index_type": indexType,
	}

	var col Collection
	if err := c.doJSON(ctx, http.MethodPost, "/v1/collections", body, &col); err != nil {
		return nil, err
	}
	return &col, nil
}

// ListCollections returns all collections.
func (c *Client) ListCollections(ctx context.Context) ([]*Collection, error) {
	var cols []*Collection
	if err := c.doJSON(ctx, http.MethodGet, "/v1/collections", nil, &cols); err != nil {
		return nil, err
	}
	return cols, nil
}

// GetCollection retrieves a collection by ID.
func (c *Client) GetCollection(ctx context.Context, id string) (*Collection, error) {
	var col Collection
	if err := c.doJSON(ctx, http.MethodGet, fmt.Sprintf("/v1/collections/%s", id), nil, &col); err != nil {
		return nil, err
	}
	return &col, nil
}

// DeleteCollection removes a collection by ID.
func (c *Client) DeleteCollection(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodDelete, fmt.Sprintf("/v1/collections/%s", id), nil, nil)
}

// ── Vector operations ───────────────────────────────────────────────────────

// Insert inserts a batch of vectors into a collection without metadata.
func (c *Client) Insert(ctx context.Context, collectionID string, ids []uint64, vectors [][]float32) error {
	return c.InsertWithMetadata(ctx, collectionID, ids, vectors, nil)
}

// InsertWithMetadata inserts a batch of vectors into a collection with optional metadata.
func (c *Client) InsertWithMetadata(ctx context.Context, collectionID string, ids []uint64, vectors [][]float32, meta []map[string]any) error {
	body := map[string]any{
		"ids":     ids,
		"vectors": vectors,
	}
	if meta != nil {
		body["metadata"] = meta
	}
	return c.doJSON(ctx, http.MethodPost, fmt.Sprintf("/v1/collections/%s/vectors", collectionID), body, nil)
}

// Search performs a nearest-neighbor search.
func (c *Client) Search(ctx context.Context, collectionID string, query []float32, k, nprobe int) ([]SearchResult, error) {
	body := map[string]any{
		"query":  query,
		"k":      k,
		"nprobe": nprobe,
	}

	var results []SearchResult
	if err := c.doJSON(ctx, http.MethodPost, fmt.Sprintf("/v1/collections/%s/search", collectionID), body, &results); err != nil {
		return nil, err
	}
	return results, nil
}

// Delete removes a vector from a collection.
func (c *Client) Delete(ctx context.Context, collectionID string, vectorID uint64) error {
	return c.doJSON(ctx, http.MethodDelete, fmt.Sprintf("/v1/collections/%s/vectors/%d", collectionID, vectorID), nil, nil)
}

// Health checks server health.
func (c *Client) Health(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodGet, "/v1/health", nil, nil)
}

// ── Internal helpers ────────────────────────────────────────────────────────

// doJSON performs an HTTP request with JSON body and decodes the response envelope.
func (c *Client) doJSON(ctx context.Context, method, path string, body any, result any) error {
	url := c.baseURL + path

	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("client: marshaling request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return fmt.Errorf("client: creating request: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("client: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10MB max response
	if err != nil {
		return fmt.Errorf("client: reading response: %w", err)
	}

	var env Envelope
	if err := json.Unmarshal(respBody, &env); err != nil {
		return fmt.Errorf("client: decoding response (status %d): %w", resp.StatusCode, err)
	}

	if env.Error != "" {
		return fmt.Errorf("client: server error (status %d): %s", resp.StatusCode, env.Error)
	}

	if result != nil && len(env.Data) > 0 {
		if err := json.Unmarshal(env.Data, result); err != nil {
			return fmt.Errorf("client: decoding data: %w", err)
		}
	}

	return nil
}
