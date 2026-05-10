// Package api — v2 API handlers with full Tenant → Database → Collection hierarchy.
//
// The v2 API enforces multi-tenancy at the route level:
//
//	/api/v2/tenants
//	/api/v2/tenants/{tenant}/databases
//	/api/v2/tenants/{tenant}/databases/{db}/collections
//	/api/v2/tenants/{tenant}/databases/{db}/collections/{id}/add
//	/api/v2/tenants/{tenant}/databases/{db}/collections/{id}/query
//	/api/v2/tenants/{tenant}/databases/{db}/collections/{id}/delete
//
// Cross-tenant isolation: querying collection A with credentials for tenant B
// returns 404 (not 403) to prevent existence leakage.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

// RegisterV2AdminRoutes adds the tenant and database management routes.
// These should be bound to the internal admin port.
func RegisterV2AdminRoutes(r chi.Router, h *Handler) {
	r.Route("/api/v2", func(r chi.Router) {
		// ── Tenant routes ───────────────────────────────────────────────
		r.Get("/tenants", h.ListTenants)
		r.Post("/tenants", h.CreateTenant)
		r.Get("/tenants/{tenant}", h.GetTenant)
		r.Delete("/tenants/{tenant}", h.DeleteTenantEndpoint)

		// ── Database routes (scoped to tenant) ──────────────────────────
		r.Get("/tenants/{tenant}/databases", h.ListDatabases)
		r.Post("/tenants/{tenant}/databases", h.CreateDatabaseEndpoint)
		r.Delete("/tenants/{tenant}/databases/{db}", h.DeleteDatabaseEndpoint)
	})
}

// RegisterV2PublicRoutes adds the collection and vector operations routes.
// These are safe to expose on the public internet port.
func RegisterV2PublicRoutes(r chi.Router, h *Handler) {
	r.Route("/api/v2", func(r chi.Router) {
		// ── Collection routes (scoped to tenant + database) ─────────────
		r.Get("/tenants/{tenant}/databases/{db}/collections", h.ListCollectionsScoped)
		r.Post("/tenants/{tenant}/databases/{db}/collections", h.CreateCollectionScoped)
		r.Get("/tenants/{tenant}/databases/{db}/collections/{id}", h.GetCollectionScoped)
		r.Delete("/tenants/{tenant}/databases/{db}/collections/{id}", h.DeleteCollectionScoped)

		// ── Vector operations (scoped to tenant + database + collection) ─
		r.Post("/tenants/{tenant}/databases/{db}/collections/{id}/add", h.InsertVectorsScoped)
		r.Post("/tenants/{tenant}/databases/{db}/collections/{id}/query", h.SearchVectorsScoped)
		r.Post("/tenants/{tenant}/databases/{db}/collections/{id}/delete", h.DeleteVectorScoped)
		r.Post("/tenants/{tenant}/databases/{db}/collections/{id}/hybrid", h.HybridSearchScoped)
	})
}

// ── Tenant Handlers ─────────────────────────────────────────────────────────

// CreateTenantRequest is the JSON body for POST /api/v2/tenants.
type CreateTenantRequest struct {
	Name     string `json:"name"`
	MaxDBs   int    `json:"max_databases"`
	MaxColls int    `json:"max_collections"`
}

// CreateTenant handles POST /api/v2/tenants.
func (h *Handler) CreateTenant(w http.ResponseWriter, r *http.Request) {
	var req CreateTenantRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}
	if req.Name == "" {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "name is required"})
		return
	}

	tenant, err := h.manager.SysDB().CreateTenant(req.Name, req.MaxDBs, req.MaxColls)
	if err != nil {
		writeJSON(w, http.StatusConflict, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, Envelope{Data: tenant})
}

// ListTenants handles GET /api/v2/tenants.
func (h *Handler) ListTenants(w http.ResponseWriter, r *http.Request) {
	tenants, err := h.manager.SysDB().ListTenants()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: tenants})
}

// GetTenant handles GET /api/v2/tenants/{tenant}.
func (h *Handler) GetTenant(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	tenant, err := h.manager.SysDB().GetTenant(tenantID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	if tenant == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "tenant not found"})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: tenant})
}

// DeleteTenantEndpoint handles DELETE /api/v2/tenants/{tenant}.
func (h *Handler) DeleteTenantEndpoint(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	if err := h.manager.SysDB().DeleteTenant(tenantID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]string{"deleted": tenantID}})
}

// ── Database Handlers ───────────────────────────────────────────────────────

// CreateDatabaseRequest is the JSON body for POST /api/v2/tenants/{tenant}/databases.
type CreateDatabaseRequest struct {
	Name string `json:"name"`
}

// CreateDatabaseEndpoint handles POST /api/v2/tenants/{tenant}/databases.
func (h *Handler) CreateDatabaseEndpoint(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")

	var req CreateDatabaseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}
	if req.Name == "" {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "name is required"})
		return
	}

	database, err := h.manager.SysDB().CreateDatabase(tenantID, req.Name)
	if err != nil {
		writeJSON(w, http.StatusConflict, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, Envelope{Data: database})
}

// ListDatabases handles GET /api/v2/tenants/{tenant}/databases.
func (h *Handler) ListDatabases(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	dbs, err := h.manager.SysDB().ListDatabases(tenantID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: dbs})
}

// DeleteDatabaseEndpoint handles DELETE /api/v2/tenants/{tenant}/databases/{db}.
func (h *Handler) DeleteDatabaseEndpoint(w http.ResponseWriter, r *http.Request) {
	dbID := chi.URLParam(r, "db")
	if err := h.manager.SysDB().DeleteDatabase(dbID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]string{"deleted": dbID}})
}

// ── Scoped Collection Handlers ──────────────────────────────────────────────

// resolveTenantDB extracts and validates tenant + database from URL params.
// Returns the tenant and database IDs, or writes an error and returns false.
func (h *Handler) resolveTenantDB(w http.ResponseWriter, r *http.Request) (tenantID, dbID string, ok bool) {
	tenantID = chi.URLParam(r, "tenant")
	dbID = chi.URLParam(r, "db")

	tenant, err := h.manager.SysDB().GetTenant(tenantID)
	if err != nil || tenant == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "tenant not found"})
		return "", "", false
	}

	db, err := h.manager.SysDB().GetDatabase(dbID)
	if err != nil || db == nil || db.TenantID != tenantID {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "database not found"})
		return "", "", false
	}

	return tenantID, dbID, true
}

// CreateCollectionScoped handles POST /api/v2/tenants/{tenant}/databases/{db}/collections.
func (h *Handler) CreateCollectionScoped(w http.ResponseWriter, r *http.Request) {
	tenantID, dbID, ok := h.resolveTenantDB(w, r)
	if !ok {
		return
	}

	var req CreateCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
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

	// Quota check: max collections per tenant
	tenant, _ := h.manager.SysDB().GetTenant(tenantID)
	if tenant != nil && tenant.MaxColls > 0 {
		count, err := h.manager.SysDB().CountCollectionsForTenant(tenantID)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, Envelope{Error: "quota check failed"})
			return
		}
		if count >= tenant.MaxColls {
			writeJSON(w, http.StatusPaymentRequired,
				Envelope{Error: fmt.Sprintf("tenant collection quota exceeded: %d/%d", count, tenant.MaxColls)})
			return
		}
	}

	col, err := h.manager.CreateCollectionScoped(tenantID, dbID, req.Name, req.Dimension, req.Metric, req.IndexType)
	if err != nil {
		writeJSON(w, http.StatusConflict, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, Envelope{Data: col})
}

// ListCollectionsScoped handles GET /api/v2/tenants/{tenant}/databases/{db}/collections.
func (h *Handler) ListCollectionsScoped(w http.ResponseWriter, r *http.Request) {
	tenantID, dbID, ok := h.resolveTenantDB(w, r)
	if !ok {
		return
	}

	cols, err := h.manager.ListCollectionsScoped(tenantID, dbID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: cols})
}

// GetCollectionScoped handles GET /api/v2/tenants/{tenant}/databases/{db}/collections/{id}.
func (h *Handler) GetCollectionScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	col, err := h.manager.GetCollectionScoped(tenantID, colID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}
	if col == nil {
		// 404 — do not leak existence to other tenants
		writeJSON(w, http.StatusNotFound, Envelope{Error: "collection not found"})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: col})
}

// DeleteCollectionScoped handles DELETE /api/v2/tenants/{tenant}/databases/{db}/collections/{id}.
func (h *Handler) DeleteCollectionScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	if err := h.manager.DeleteCollectionScoped(tenantID, colID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]string{"deleted": colID}})
}

// ── Scoped Vector Handlers ──────────────────────────────────────────────────

// InsertVectorsScoped handles POST /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/add.
func (h *Handler) InsertVectorsScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	// Cross-tenant isolation: verify collection ownership
	col, err := h.manager.GetCollectionScoped(tenantID, colID)
	if err != nil || col == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "collection not found"})
		return
	}

	var req InsertVectorsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
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

	if err := h.manager.InsertVectors(r.Context(), colID, req.IDs, req.Vectors, req.Metadata); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, Envelope{Data: map[string]int{"inserted": len(req.IDs)}})
}

// SearchVectorsScoped handles POST /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/query.
func (h *Handler) SearchVectorsScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	col, err := h.manager.GetCollectionScoped(tenantID, colID)
	if err != nil || col == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "collection not found"})
		return
	}

	var req SearchVectorsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}
	if len(req.Query) == 0 {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "query vector is required"})
		return
	}
	if req.K <= 0 {
		req.K = 10
	}

	results, err := h.manager.SearchVectors(r.Context(), colID, req.Query, req.K, req.Nprobe, req.Filter)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: err.Error()})
		return
	}

	ids := make([]uint64, len(results))
	for i, res := range results {
		ids[i] = res.ID
	}

	metaMap, _ := h.manager.GetVectorsMetadata(colID, ids)

	finalResults := make([]SearchResultItem, len(results))
	for i, res := range results {
		finalResults[i] = SearchResultItem{
			ID:       res.ID,
			Distance: res.Distance,
			Score:    res.Score,
			Metadata: metaMap[res.ID],
		}
	}

	writeJSON(w, http.StatusOK, Envelope{Data: finalResults})
}

// DeleteVectorScopedRequest is the JSON body for POST .../collections/{id}/delete.
type DeleteVectorScopedRequest struct {
	VectorID uint64 `json:"vector_id"`
}

// DeleteVectorScoped handles POST /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/delete.
func (h *Handler) DeleteVectorScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	col, err := h.manager.GetCollectionScoped(tenantID, colID)
	if err != nil || col == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "collection not found"})
		return
	}

	var req DeleteVectorScopedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}

	vecID := req.VectorID
	if vecIDStr := chi.URLParam(r, "vectorId"); vecIDStr != "" {
		parsed, err := strconv.ParseUint(vecIDStr, 10, 64)
		if err == nil {
			vecID = parsed
		}
	}

	if err := h.manager.DeleteVector(r.Context(), colID, vecID); err != nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, Envelope{Data: map[string]int{"deleted": int(req.VectorID)}})
}

// HybridSearchRequest is the JSON body for POST /.../hybrid
type HybridSearchRequest struct {
	QueryText      string         `json:"query_text"`
	QueryEmbedding []float32      `json:"query_embedding"`
	TopK           int            `json:"top_k"`
	Ef             int            `json:"ef,omitempty"` // mapped to nprobe/ef
	Filter         map[string]any `json:"filter,omitempty"`
}

// HybridSearchScoped handles POST /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/hybrid.
func (h *Handler) HybridSearchScoped(w http.ResponseWriter, r *http.Request) {
	tenantID := chi.URLParam(r, "tenant")
	colID := chi.URLParam(r, "id")

	// Cross-tenant isolation
	col, err := h.manager.GetCollectionScoped(tenantID, colID)
	if err != nil || col == nil {
		writeJSON(w, http.StatusNotFound, Envelope{Error: "collection not found"})
		return
	}

	var req HybridSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, Envelope{Error: "invalid JSON"})
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}
	nprobe := req.Ef
	if nprobe <= 0 {
		nprobe = 10
	}

	results, err := h.manager.HybridSearch(r.Context(), colID, req.QueryEmbedding, req.QueryText, req.TopK, nprobe, req.Filter)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, Envelope{Error: err.Error()})
		return
	}

	ids := make([]uint64, len(results))
	for i, res := range results {
		ids[i] = res.ID
	}

	metaMap, _ := h.manager.GetVectorsMetadata(colID, ids)

	finalResults := make([]SearchResultItem, len(results))
	for i, res := range results {
		finalResults[i] = SearchResultItem{
			ID:       res.ID,
			Distance: res.Distance,
			Score:    res.Score,
			Metadata: metaMap[res.ID],
		}
	}

	writeJSON(w, http.StatusOK, Envelope{Data: finalResults})
}
