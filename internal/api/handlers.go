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
	"github.com/POTATO-VE1/Magnitude/internal/metadata"
	"github.com/POTATO-VE1/Magnitude/internal/routing"
	"github.com/POTATO-VE1/Magnitude/internal/security"
)

// Envelope is the standard JSON response wrapper.
type Envelope struct {
	Data  any    `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
}

// Handler holds dependencies for HTTP handlers.
type Handler struct {
	manager   *collection.Manager
	config    *config.Config
	router    *routing.Router
	forwarder *routing.Forwarder
}

// NewRouter creates and configures the chi router with all routes and middleware.
func NewRouter(cfg *config.Config, mgr *collection.Manager, rt *routing.Router, fwd *routing.Forwarder) *chi.Mux {
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

	h := &Handler{
		manager:   mgr,
		config:    cfg,
		router:    rt,
		forwarder: fwd,
	}

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

	// Cluster metadata endpoint (for client-side hash ring sync)
	r.Get("/v1/cluster/metadata", h.ClusterMetadata)

	// Inter-node replicate endpoints (internal RPC)
	r.Post("/internal/v1/replicate/insert", h.ReplicateInsert)
	r.Post("/internal/v1/replicate/search", h.ReplicateSearch)
	r.Post("/internal/v1/replicate/delete", h.ReplicateDelete)

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

	// Tenant isolation: if auth is active, scope to the authenticated tenant
	tenantID := security.TenantID(r.Context())
	var col *metadata.Collection
	var err error
	if tenantID != "" {
		col, err = h.manager.CreateCollectionScoped(tenantID, "", req.Name, req.Dimension, req.Metric, req.IndexType)
	} else {
		col, err = h.manager.CreateCollection(req.Name, req.Dimension, req.Metric, req.IndexType)
	}
	if err != nil {
		writeJSON(w, http.StatusConflict, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusCreated, Envelope{Data: col})
}

// ListCollections handles GET /v1/collections.
func (h *Handler) ListCollections(w http.ResponseWriter, r *http.Request) {
	tenantID := security.TenantID(r.Context())
	var cols []*metadata.Collection
	var err error
	if tenantID != "" {
		cols, err = h.manager.ListCollectionsForTenant(tenantID)
	} else {
		cols, err = h.manager.ListCollections()
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: cols})
}

// GetCollection handles GET /v1/collections/{id}.
func (h *Handler) GetCollection(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	tenantID := security.TenantID(r.Context())

	var col *metadata.Collection
	var err error
	if tenantID != "" {
		col, err = h.manager.GetCollectionScoped(tenantID, id)
	} else {
		col, err = h.manager.GetCollection(id)
	}
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
	tenantID := security.TenantID(r.Context())

	var err error
	if tenantID != "" {
		err = h.manager.DeleteCollectionScoped(tenantID, id)
	} else {
		err = h.manager.DeleteCollection(id)
	}
	if err != nil {
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
	if len(req.Metadata) > 0 && len(req.Metadata) != len(req.IDs) {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "metadata array must match ids length if provided"})
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

// APISearchResult extends the internal index.SearchResult with metadata.
type APISearchResult struct {
	ID       uint64         `json:"id"`
	Distance float32        `json:"distance"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
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

	var vIDs []uint64
	for _, r := range results {
		vIDs = append(vIDs, r.ID)
	}
	batchMeta, _ := h.manager.SysDB().LoadVectorMetadataBatch(id, vIDs)

	apiResults := make([]APISearchResult, len(results))
	for i, r := range results {
		apiResults[i] = APISearchResult{
			ID:       r.ID,
			Distance: r.Distance,
			Score:    r.Score,
			Metadata: batchMeta[r.ID],
		}
	}

	writeJSON(w, http.StatusOK, Envelope{Data: apiResults})
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

// ── Cluster Metadata Handler ─────────────────────────────────────────────────

// ClusterMetadataResponse is the response for GET /v1/cluster/metadata.
type ClusterMetadataResponse struct {
	Nodes    []NodeMetadata    `json:"nodes"`
	ShardMap map[string]string `json:"shard_map"`
}

// NodeMetadata describes a cluster node for client discovery.
type NodeMetadata struct {
	ID         string `json:"id"`
	Address    string `json:"address"`
	APIAddress string `json:"api_address"`
	State      string `json:"state"`
}

// ClusterMetadata handles GET /v1/cluster/metadata.
// Returns the list of known nodes and the shard map for client-side routing.
func (h *Handler) ClusterMetadata(w http.ResponseWriter, r *http.Request) {
	if h.router == nil {
		writeJSON(w, http.StatusOK, Envelope{Data: ClusterMetadataResponse{
			Nodes:    []NodeMetadata{},
			ShardMap: map[string]string{},
		}})
		return
	}

	nodes := h.router.GetAllNodes()
	nodeMeta := make([]NodeMetadata, 0, len(nodes))
	for _, nodeID := range nodes {
		addr := h.router.GetAddress(nodeID)
		nodeMeta = append(nodeMeta, NodeMetadata{
			ID:         nodeID,
			APIAddress: addr,
			State:      "alive",
		})
	}

	// Build shard map from all collections
	cols, _ := h.manager.ListCollections()
	shardMap := make(map[string]string, len(cols))
	if cols != nil {
		for _, col := range cols {
			shardMap[col.ID] = h.router.GetNodeForVector(col.ID, 0)
		}
	}

	writeJSON(w, http.StatusOK, Envelope{Data: ClusterMetadataResponse{
		Nodes:    nodeMeta,
		ShardMap: shardMap,
	}})
}

// ── Replicate Handlers (inter-node RPC) ─────────────────────────────────────

// ReplicateInsert handles POST /internal/v1/replicate/insert.
// Receives an insert request from a peer node and applies it locally.
func (h *Handler) ReplicateInsert(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CollectionID string           `json:"collection_id"`
		IDs          []uint64         `json:"ids"`
		Vectors      [][]float32      `json:"vectors"`
		Metadata     []map[string]any `json:"metadata,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}

	if err := h.manager.InsertVectors(r.Context(), req.CollectionID, req.IDs, req.Vectors, req.Metadata); err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, Envelope{Data: map[string]int{"inserted": len(req.IDs)}})
}

// ReplicateSearch handles POST /internal/v1/replicate/search.
// Receives a search request from a peer node and searches locally.
func (h *Handler) ReplicateSearch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CollectionID string         `json:"collection_id"`
		Query        []float32      `json:"query"`
		K            int            `json:"k"`
		Nprobe       int            `json:"nprobe"`
		Filter       map[string]any `json:"filter,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}

	results, err := h.manager.SearchVectors(r.Context(), req.CollectionID, req.Query, req.K, req.Nprobe, req.Filter)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, Envelope{Data: results})
}

// ReplicateDelete handles POST /internal/v1/replicate/delete.
// Receives a delete request from a peer node and applies it locally.
func (h *Handler) ReplicateDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		CollectionID string `json:"collection_id"`
		VectorID     uint64 `json:"vector_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}

	if err := h.manager.DeleteVector(r.Context(), req.CollectionID, req.VectorID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, Envelope{Data: map[string]uint64{"deleted": req.VectorID}})
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
