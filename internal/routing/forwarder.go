package routing

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/POTATO-VE1/Magnitude/internal/config"
	"github.com/POTATO-VE1/Magnitude/internal/index"
)

// Forwarder proxies HTTP requests to remote nodes in the cluster.
type Forwarder struct {
	client *http.Client
	scheme string // "http" or "https" depending on server TLS config
}

// NewForwarder creates a new Forwarder with connection pooling and a
// configurable timeout from the server config.
// If cfg is nil, a safe default of 30s is used.
func NewForwarder(cfg *config.Config) *Forwarder {
	timeout := 30 * time.Second // safe default
	if cfg != nil && cfg.Cluster.RequestTimeoutMS > 0 {
		timeout = time.Duration(cfg.Cluster.RequestTimeoutMS) * time.Millisecond
	}

	scheme := "http"
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
	}

	// Use HTTPS for inter-node traffic if TLS is configured
	if cfg != nil && cfg.Server.CertFile != "" {
		scheme = "https"
		transport.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: true, // cluster nodes use self-signed certs
		}
	}

	return &Forwarder{
		client: &http.Client{
			Transport: transport,
			Timeout:   timeout,
		},
		scheme: scheme,
	}
}

// InsertRequest matches the expected JSON schema for inserting vectors.
type InsertRequest struct {
	IDs      []uint64                 `json:"ids"`
	Vectors  [][]float32              `json:"vectors"`
	Metadata []map[string]interface{} `json:"metadata,omitempty"`
}

// ForwardInsertBatch proxies a batch of vectors to the target node's API.
func (f *Forwarder) ForwardInsertBatch(ctx context.Context, targetAddress string, tenant, db, colID string, req InsertRequest, authHeader string) error {
	// Re-encode payload
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("%s://%s/api/v2/tenants/%s/databases/%s/collections/%s/add", f.scheme, targetAddress, tenant, db, colID)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if authHeader != "" {
		httpReq.Header.Set("Authorization", authHeader)
	}
	// Prevent infinite loops in case of misconfiguration
	httpReq.Header.Set("X-Forwarded-For", "cluster-proxy")

	resp, err := f.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("proxy error to %s: %w", targetAddress, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("remote node %s returned %d: %s", targetAddress, resp.StatusCode, string(b))
	}
	return nil
}

// SearchRequest matches the expected JSON schema for searching.
type SearchRequest struct {
	Query  []float32      `json:"query"`
	K      int            `json:"k"`
	Nprobe int            `json:"nprobe,omitempty"`
	Filter map[string]any `json:"filter,omitempty"`
}

// ForwardSearch proxies a search request to a target node.
func (f *Forwarder) ForwardSearch(ctx context.Context, targetAddress string, tenant, db, colID string, req SearchRequest, authHeader string) ([]index.SearchResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s://%s/api/v2/tenants/%s/databases/%s/collections/%s/query", f.scheme, targetAddress, tenant, db, colID)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if authHeader != "" {
		httpReq.Header.Set("Authorization", authHeader)
	}
	httpReq.Header.Set("X-Forwarded-For", "cluster-proxy")

	resp, err := f.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("proxy error to %s: %w", targetAddress, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("remote node %s returned %d: %s", targetAddress, resp.StatusCode, string(b))
	}

	var envelope struct {
		Data []index.SearchResult `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return nil, err
	}
	return envelope.Data, nil
}

// ForwardDelete proxies a vector deletion to a target node.
func (f *Forwarder) ForwardDelete(ctx context.Context, targetAddress string, tenant, db, colID string, vectorID uint64, authHeader string) error {
	url := fmt.Sprintf("%s://%s/api/v2/tenants/%s/databases/%s/collections/%s/delete", f.scheme, targetAddress, tenant, db, colID)

	body, _ := json.Marshal(map[string]uint64{"vector_id": vectorID})
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if authHeader != "" {
		httpReq.Header.Set("Authorization", authHeader)
	}
	httpReq.Header.Set("X-Forwarded-For", "cluster-proxy")

	resp, err := f.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("proxy error to %s: %w", targetAddress, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("remote node %s returned %d: %s", targetAddress, resp.StatusCode, string(b))
	}
	return nil
}
