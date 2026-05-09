// Package api — HTTP request handlers.
//
// Handler responsibilities:
//   - Parse and validate request body (JSON)
//   - Call the Collection layer (never the Index layer directly)
//   - Map VDBErrors to HTTP status codes via VDBError.HTTPStatusCode()
//   - Write JSON responses with consistent envelope: {data, error, meta}
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"

	"github.com/go-chi/chi/v5"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/POTATO-VE1/Magnitude/internal/collection"
	"github.com/POTATO-VE1/Magnitude/internal/config"
	"github.com/POTATO-VE1/Magnitude/internal/security"
)

// Envelope is the standard JSON response wrapper.
type Envelope struct {
	Data  any    `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
}

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	manager *collection.Manager
	config  *config.Config
}

// NewRouter creates and configures the chi router with all routes and middleware.
func NewRouter(cfg *config.Config, mgr *collection.Manager) *chi.Mux {
	r := chi.NewRouter()

	// Build middleware dependencies
	sysdb := mgr.SysDB()
	tenantAuth := security.TenantAuthMiddleware(sysdb, len(cfg.Auth.KeyHashes) > 0)
	tenantRateLimiter := security.NewTenantRateLimiter(sysdb)

	// Middleware stack (outermost first)
	r.Use(Recovery)
	r.Use(RequestID)
	r.Use(StructuredLog)
	r.Use(Metrics) // Prometheus instrumentation
	r.Use(tenantAuth)
	r.Use(tenantRateLimiter.Middleware)
	r.Use(MaxBodySize(4 * 1024 * 1024)) // 4 MiB max body

	h := &Handler{manager: mgr, config: cfg}

	// Health check (no auth required — move before auth middleware if needed)
	r.Get("/v1/health", h.HealthCheck)

	// Collection routes
	r.Post("/v1/collections", h.CreateCollection)
	r.Get("/v1/collections", h.ListCollections)
	r.Get("/v1/collections/{id}", h.GetCollection)
	r.Delete("/v1/collections/{id}", h.DeleteCollection)

	// Vector routes
	r.Post("/v1/collections/{id}/vectors", h.InsertVectors)
	r.Post("/v1/collections/{id}/search", h.SearchVectors)
	r.Delete("/v1/collections/{id}/vectors/{vectorId}", h.DeleteVector)

	// Prometheus metrics endpoint
	r.Handle("/metrics", promhttp.Handler())

	// v2 API: tenant-scoped routes (Phase 8 multi-tenancy)
	RegisterV2PublicRoutes(r, h)

	return r
}

// NewAdminRouter creates the chi router for internal admin operations.
func NewAdminRouter(cfg *config.Config, mgr *collection.Manager) *chi.Mux {
	r := chi.NewRouter()

	r.Use(Recovery)
	r.Use(RequestID)
	r.Use(StructuredLog)
	r.Use(MaxBodySize(4 * 1024 * 1024))

	h := &Handler{manager: mgr, config: cfg}

	// Admin routes
	RegisterV2AdminRoutes(r, h)

	return r
}

// HealthCheck returns server health status.
func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]string{"status": "ok"}})
}

// ── Collection Handlers ──────────────────────────────────────────────────────

// CreateCollectionRequest is the JSON body for POST /v1/collections.
type CreateCollectionRequest struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    string `json:"metric"`
	IndexType string `json:"index_type"`
}

// CreateCollection handles POST /v1/collections.
func (h *Handler) CreateCollection(w http.ResponseWriter, r *http.Request) {
	var req CreateCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON: " + err.Error()})
		return
	}

	if req.Name == "" {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "name is required"})
		return
	}
	if req.Dimension <= 0 {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "dimension must be > 0"})
		return
	}
	if req.Metric == "" {
		req.Metric = "l2"
	}
	if req.IndexType == "" {
		req.IndexType = h.config.Index.Type
	}

	col, err := h.manager.CreateCollection(req.Name, req.Dimension, req.Metric, req.IndexType)
	if err != nil {
		writeJSON(w, http.StatusConflict, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusCreated, Envelope{Data: col})
}

// ListCollections handles GET /v1/collections.
func (h *Handler) ListCollections(w http.ResponseWriter, r *http.Request) {
	cols, err := h.manager.ListCollections()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: cols})
}

// GetCollection handles GET /v1/collections/{id}.
func (h *Handler) GetCollection(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	col, err := h.manager.GetCollection(id)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	if col == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: fmt.Sprintf("collection %q not found", id)})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: col})
}

// DeleteCollection handles DELETE /v1/collections/{id}.
func (h *Handler) DeleteCollection(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	if err := h.manager.DeleteCollection(id); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]string{"deleted": id}})
}

// ── Vector Handlers ──────────────────────────────────────────────────────────

// InsertVectorsRequest is the JSON body for POST /v1/collections/{id}/vectors.
type InsertVectorsRequest struct {
	IDs      []uint64         `json:"ids"`
	Vectors  [][]float32      `json:"vectors"`
	Metadata []map[string]any `json:"metadata,omitempty"` // per-vector metadata (optional)
}

// InsertVectors handles POST /v1/collections/{id}/vectors.
func (h *Handler) InsertVectors(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var req InsertVectorsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON: " + err.Error()})
		return
	}

	if len(req.IDs) == 0 {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "ids is required"})
		return
	}
	if len(req.IDs) != len(req.Vectors) {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "ids and vectors must have the same length"})
		return
	}

	for _, meta := range req.Metadata {
		if err := validateMetadata(meta); err != nil {
			writeJSON(w, http.StatusBadRequest, Envelope{Error: err.Error()})
			return
		}
	}

	if err := h.manager.InsertVectors(r.Context(), id, req.IDs, req.Vectors, req.Metadata); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusCreated, Envelope{Data: map[string]int{"inserted": len(req.IDs)}})
}

// SearchVectorsRequest is the JSON body for POST /v1/collections/{id}/search.
type SearchVectorsRequest struct {
	Query  []float32      `json:"query"`
	K      int            `json:"k"`
	Nprobe int            `json:"nprobe"`
	Filter map[string]any `json:"filter,omitempty"` // metadata filter (optional)
}

// SearchVectors handles POST /v1/collections/{id}/search.
func (h *Handler) SearchVectors(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	var req SearchVectorsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON: " + err.Error()})
		return
	}

	if len(req.Query) == 0 {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "query vector is required"})
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	results, err := h.manager.SearchVectors(r.Context(), id, req.Query, req.K, req.Nprobe, req.Filter)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, Envelope{Data: results})
}

// DeleteVector handles DELETE /v1/collections/{id}/vectors/{vectorId}.
func (h *Handler) DeleteVector(w http.ResponseWriter, r *http.Request) {
	colID := chi.URLParam(r, "id")
	vecIDStr := chi.URLParam(r, "vectorId")

	vecID, err := strconv.ParseUint(vecIDStr, 10, 64)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid vector ID"})
		return
	}

	if err := h.manager.DeleteVector(r.Context(), colID, vecID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, Envelope{Data: map[string]uint64{"deleted": vecID}})
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

var metadataKeyRegex = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// validateMetadata enforces key format and value length limits to prevent abuse.
func validateMetadata(meta map[string]any) error {
	if meta == nil {
		return nil
	}
	if len(meta) > 50 {
		return fmt.Errorf("metadata cannot exceed 50 keys")
	}
	for k, v := range meta {
		if !metadataKeyRegex.MatchString(k) {
			return fmt.Errorf("invalid metadata key format: %q", k)
		}
		if str, ok := v.(string); ok {
			if len(str) > 1024 {
				return fmt.Errorf("metadata string value for key %q exceeds 1024 bytes", k)
			}
		}
	}
	return nil
}
