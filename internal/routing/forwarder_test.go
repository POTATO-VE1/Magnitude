package routing

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/POTATO-VE1/Magnitude/internal/index"
)

// ── Forwarder Tests ─────────────────────────────────────────────────────────

func TestNewForwarder(t *testing.T) {
	f := NewForwarder(nil)
	if f == nil {
		t.Fatal("NewForwarder returned nil")
	}
	if f.client == nil {
		t.Fatal("client is nil")
	}
}

func TestForwardInsertBatch_Success(t *testing.T) {
	// Mock remote node
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify method and path
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %s, want application/json", r.Header.Get("Content-Type"))
		}
		// Verify loop-detection header
		if r.Header.Get("X-Forwarded-For") != "cluster-proxy" {
			t.Error("missing X-Forwarded-For header")
		}

		// Decode body
		var req InsertRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode error: %v", err)
		}
		if len(req.IDs) != 2 {
			t.Errorf("IDs count = %d, want 2", len(req.IDs))
		}

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	f := NewForwarder(nil)
	// Extract host:port from the test server URL
	addr := server.Listener.Addr().String()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := InsertRequest{
		IDs:     []uint64{1, 2},
		Vectors: [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
	}

	err := f.ForwardInsertBatch(ctx, addr, "default", "default_db", "col-1", req, "")
	if err != nil {
		t.Fatalf("ForwardInsertBatch: %v", err)
	}
}

func TestForwardInsertBatch_AuthHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer test-token" {
			t.Errorf("Authorization = %q, want 'Bearer test-token'", auth)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()

	ctx := context.Background()
	req := InsertRequest{
		IDs:     []uint64{1},
		Vectors: [][]float32{{1.0}},
	}

	err := f.ForwardInsertBatch(ctx, addr, "default", "default_db", "col-1", req, "Bearer test-token")
	if err != nil {
		t.Fatalf("ForwardInsertBatch: %v", err)
	}
}

func TestForwardInsertBatch_RemoteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal failure"}`))
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()

	ctx := context.Background()
	req := InsertRequest{
		IDs:     []uint64{1},
		Vectors: [][]float32{{1.0}},
	}

	err := f.ForwardInsertBatch(ctx, addr, "default", "default_db", "col-1", req, "")
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestForwardSearch_Success(t *testing.T) {
	expected := []index.SearchResult{
		{ID: 1, Distance: 0.1, Score: 0.9},
		{ID: 2, Distance: 0.2, Score: 0.8},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}

		// Decode search request
		var req SearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode error: %v", err)
		}
		if req.K != 5 {
			t.Errorf("K = %d, want 5", req.K)
		}

		envelope := struct {
			Data []index.SearchResult `json:"data"`
		}{Data: expected}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(envelope)
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()

	ctx := context.Background()
	req := SearchRequest{
		Query: []float32{1.0, 2.0, 3.0},
		K:     5,
	}

	results, err := f.ForwardSearch(ctx, addr, "default", "default_db", "col-1", req, "")
	if err != nil {
		t.Fatalf("ForwardSearch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results count = %d, want 2", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("results[0].ID = %d, want 1", results[0].ID)
	}
}

func TestForwardSearch_RemoteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`bad request`))
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()
	ctx := context.Background()
	req := SearchRequest{Query: []float32{1.0}, K: 5}

	_, err := f.ForwardSearch(ctx, addr, "default", "default_db", "col-1", req, "")
	if err == nil {
		t.Fatal("expected error for 400 response")
	}
}

func TestForwardDelete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.Header.Get("X-Forwarded-For") != "cluster-proxy" {
			t.Error("missing X-Forwarded-For header")
		}
		// Verify the body contains vector_id
		var body map[string]uint64
		json.NewDecoder(r.Body).Decode(&body)
		if body["vector_id"] != 42 {
			t.Errorf("vector_id = %d, want 42", body["vector_id"])
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()
	ctx := context.Background()

	err := f.ForwardDelete(ctx, addr, "default", "default_db", "col-1", 42, "")
	if err != nil {
		t.Fatalf("ForwardDelete: %v", err)
	}
}

func TestForwardDelete_RemoteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`not found`))
	}))
	defer server.Close()

	f := NewForwarder(nil)
	addr := server.Listener.Addr().String()
	ctx := context.Background()

	err := f.ForwardDelete(ctx, addr, "default", "default_db", "col-1", 42, "")
	if err == nil {
		t.Fatal("expected error for 404 response")
	}
}

func TestForwardInsertBatch_ConnectionRefused(t *testing.T) {
	f := NewForwarder(nil)
	ctx := context.Background()
	req := InsertRequest{
		IDs:     []uint64{1},
		Vectors: [][]float32{{1.0}},
	}

	// Connect to a port that's definitely not listening
	err := f.ForwardInsertBatch(ctx, "127.0.0.1:1", "default", "default_db", "col-1", req, "")
	if err == nil {
		t.Fatal("expected connection error")
	}
}
