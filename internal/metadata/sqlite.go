// Package metadata implements the System Database (SysDB) — the catalog layer
// that tracks tenants, databases, collections, and segments.
//
// ChromaDB Lesson 1: "SQLite is not a crutch; it is the right foundation."
// SQLite with WAL mode, PRAGMA synchronous=NORMAL, and a 64MB page cache handles
// millions of metadata records with sub-millisecond reads.
//
// ChromaDB Lesson 10: "Tenant → Database → Collection is the right hierarchy."
// Every API call is scoped to: Tenant (billing/quota) → Database (namespace) → Collection (index).
package metadata

import (
	"database/sql"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

// Collection represents a vector collection in the system database.
type Collection struct {
	ID          string `json:"id"`
	TenantID    string `json:"tenant_id"`
	DatabaseID  string `json:"database_id"`
	Name        string `json:"name"`
	Dimension   int    `json:"dimension"`
	Metric      string `json:"metric"`
	IndexType   string `json:"index_type"`
	CreatedAt   int64  `json:"created_at"`
	VectorCount int    `json:"vector_count"`
}

// Segment tracks an index or metadata file version for a collection.
type Segment struct {
	ID           string `json:"id"`
	CollectionID string `json:"collection_id"`
	Type         string `json:"type"`
	Scope        string `json:"scope"`
	FilePath     string `json:"file_path,omitempty"`
	MaxSeqID     int64  `json:"max_seq_id"`
	CreatedAt    int64  `json:"created_at"`
}

// TenantQuotas defines limits for a tenant.
type TenantQuotas struct {
	MaxVectors     int `json:"max_vectors"`
	MaxCollections int `json:"max_collections"`
	MaxQPS         int `json:"max_qps"`
	MaxBytes       int `json:"max_bytes"`
}

// TenantUsage tracks current usage for a tenant.
type TenantUsage struct {
	VectorCount     int   `json:"vector_count"`
	CollectionCount int   `json:"collection_count"`
	BytesUsed       int   `json:"bytes_used"`
	UpdatedAt       int64 `json:"updated_at"`
}

const (
	// DefaultTenantName is seeded on first startup.
	DefaultTenantName = "default_tenant"
	// DefaultDatabaseName is seeded under the default tenant.
	DefaultDatabaseName = "default"
)

// SysDB is the system database that stores collection metadata.
// It wraps a SQLite database with pragmas tuned for the vector DB workload.
type SysDB struct {
	mu sync.RWMutex
	db *sql.DB

	// tombstoneSet is an in-memory cache of deleted vector IDs per collection.
	// Consulted during search to filter out deleted vectors without hitting SQLite.
	tombstones     map[string]map[uint64]struct{} // collection_id → set of deleted vector IDs
	tombstoneMu    sync.RWMutex
}

// NewSysDB opens or creates a SysDB at the given path.
func NewSysDB(path string) (*SysDB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("metadata: opening sqlite at %q: %w", path, err)
	}

	// Performance pragmas
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA cache_size=-65536", // 64MB
		"PRAGMA busy_timeout=5000",
		"PRAGMA foreign_keys=ON",
	}
	for _, p := range pragmas {
		if _, err := db.Exec(p); err != nil {
			db.Close()
			return nil, fmt.Errorf("metadata: setting pragma %q: %w", p, err)
		}
	}

	// Create schema
	schema := `
		CREATE TABLE IF NOT EXISTS collections (
			id           TEXT    PRIMARY KEY,
			tenant_id    TEXT    NOT NULL DEFAULT '',
			database_id  TEXT    NOT NULL DEFAULT '',
			name         TEXT    NOT NULL,
			dimension    INTEGER NOT NULL,
			metric       TEXT    NOT NULL DEFAULT 'l2',
			index_type   TEXT    NOT NULL DEFAULT 'flat',
			vector_count INTEGER NOT NULL DEFAULT 0,
			created_at   INTEGER NOT NULL DEFAULT (unixepoch()),
			UNIQUE(database_id, name)
		);

		CREATE INDEX IF NOT EXISTS idx_collections_tenant ON collections(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_collections_database ON collections(database_id);

		CREATE TABLE IF NOT EXISTS vectors (
			id            INTEGER NOT NULL,
			collection_id TEXT    NOT NULL,
			file_offset   INTEGER NOT NULL DEFAULT 0,
			cluster_id    INTEGER NOT NULL DEFAULT -1,
			is_deleted    INTEGER NOT NULL DEFAULT 0,
			created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
			PRIMARY KEY (collection_id, id),
			FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection_id) WHERE is_deleted = 0;

		CREATE TABLE IF NOT EXISTS tombstones (
			collection_id TEXT    NOT NULL,
			vector_id     INTEGER NOT NULL,
			deleted_at    INTEGER NOT NULL DEFAULT (unixepoch()),
			PRIMARY KEY (collection_id, vector_id),
			FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
		);

		CREATE TABLE IF NOT EXISTS segments (
			id            TEXT PRIMARY KEY,
			collection_id TEXT NOT NULL,
			type          TEXT NOT NULL,
			scope         TEXT NOT NULL,
			file_path     TEXT,
			max_seq_id    INTEGER NOT NULL DEFAULT 0,
			created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
			FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_segments_collection ON segments(collection_id);

		CREATE TABLE IF NOT EXISTS tenant_quotas (
			tenant_id        TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
			max_vectors      INTEGER NOT NULL DEFAULT 10000000,
			max_collections  INTEGER NOT NULL DEFAULT 100,
			max_qps          INTEGER NOT NULL DEFAULT 100,
			max_bytes        INTEGER NOT NULL DEFAULT 10737418240
		);

		CREATE TABLE IF NOT EXISTS tenant_usage (
			tenant_id       TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
			vector_count    INTEGER NOT NULL DEFAULT 0,
			collection_count INTEGER NOT NULL DEFAULT 0,
			bytes_used      INTEGER NOT NULL DEFAULT 0,
			updated_at      INTEGER NOT NULL DEFAULT (unixepoch())
		);

		CREATE TABLE IF NOT EXISTS api_keys (
			key_hash        TEXT PRIMARY KEY,
			tenant_id       TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
			role            TEXT NOT NULL,
			created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
			expires_at      INTEGER
		);

		CREATE TABLE IF NOT EXISTS vector_metadata (
			collection_id TEXT    NOT NULL,
			vector_id     INTEGER NOT NULL,
			meta_key      TEXT    NOT NULL,
			meta_value    TEXT    NOT NULL,
			PRIMARY KEY (collection_id, vector_id, meta_key),
			FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_vector_metadata_lookup ON vector_metadata(collection_id, vector_id);
	`
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("metadata: creating schema: %w", err)
	}

	sysdb := &SysDB{
		db:         db,
		tombstones: make(map[string]map[uint64]struct{}),
	}

	// Load existing tombstones into memory
	if err := sysdb.loadTombstones(); err != nil {
		db.Close()
		return nil, fmt.Errorf("metadata: loading tombstones: %w", err)
	}

	slog.Info("SysDB initialized", "path", path)
	return sysdb, nil
}

// CreateCollection creates a new collection (backward-compatible, uses empty tenant/db).
func (s *SysDB) CreateCollection(name string, dim int, metric, indexType string) (*Collection, error) {
	return s.CreateCollectionScoped("", "", name, dim, metric, indexType)
}

// CreateCollectionScoped creates a new collection scoped to a tenant and database.
// Collection names are unique within a database (not globally).
func (s *SysDB) CreateCollectionScoped(tenantID, databaseID, name string, dim int, metric, indexType string) (*Collection, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := uuid.New().String()
	now := time.Now().Unix()

	_, err := s.db.Exec(
		`INSERT INTO collections (id, tenant_id, database_id, name, dimension, metric, index_type, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		id, tenantID, databaseID, name, dim, metric, indexType, now,
	)
	if err != nil {
		return nil, fmt.Errorf("metadata: creating collection %q: %w", name, err)
	}

	return &Collection{
		ID:         id,
		TenantID:   tenantID,
		DatabaseID: databaseID,
		Name:       name,
		Dimension:  dim,
		Metric:     metric,
		IndexType:  indexType,
		CreatedAt:  now,
	}, nil
}

// GetCollection retrieves a collection by ID.
func (s *SysDB) GetCollection(id string) (*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var c Collection
	err := s.db.QueryRow(
		`SELECT id, tenant_id, database_id, name, dimension, metric, index_type, vector_count, created_at
		 FROM collections WHERE id = ?`, id,
	).Scan(&c.ID, &c.TenantID, &c.DatabaseID, &c.Name, &c.Dimension, &c.Metric, &c.IndexType, &c.VectorCount, &c.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting collection %q: %w", id, err)
	}
	return &c, nil
}

// GetCollectionScoped retrieves a collection by ID, but only if it belongs to the given tenant.
// Returns nil (not error) if the collection exists but belongs to another tenant — this
// prevents cross-tenant existence leakage (returns 404 instead of 403).
func (s *SysDB) GetCollectionScoped(tenantID, collectionID string) (*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var c Collection
	err := s.db.QueryRow(
		`SELECT id, tenant_id, database_id, name, dimension, metric, index_type, vector_count, created_at
		 FROM collections WHERE id = ? AND tenant_id = ?`, collectionID, tenantID,
	).Scan(&c.ID, &c.TenantID, &c.DatabaseID, &c.Name, &c.Dimension, &c.Metric, &c.IndexType, &c.VectorCount, &c.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting scoped collection: %w", err)
	}
	return &c, nil
}

// GetCollectionByName retrieves a collection by name (searches globally).
func (s *SysDB) GetCollectionByName(name string) (*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var c Collection
	err := s.db.QueryRow(
		`SELECT id, tenant_id, database_id, name, dimension, metric, index_type, vector_count, created_at
		 FROM collections WHERE name = ?`, name,
	).Scan(&c.ID, &c.TenantID, &c.DatabaseID, &c.Name, &c.Dimension, &c.Metric, &c.IndexType, &c.VectorCount, &c.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting collection by name %q: %w", name, err)
	}
	return &c, nil
}

// ListCollections returns all collections (no tenant filter).
func (s *SysDB) ListCollections() ([]*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.scanCollections(
		`SELECT id, tenant_id, database_id, name, dimension, metric, index_type, vector_count, created_at
		 FROM collections ORDER BY created_at`,
	)
}

// ListCollectionsScoped returns collections for a specific tenant and database.
func (s *SysDB) ListCollectionsScoped(tenantID, databaseID string) ([]*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.scanCollections(
		`SELECT id, tenant_id, database_id, name, dimension, metric, index_type, vector_count, created_at
		 FROM collections WHERE tenant_id = ? AND database_id = ? ORDER BY created_at`,
		tenantID, databaseID,
	)
}

// scanCollections is a shared helper for scanning collection query results.
func (s *SysDB) scanCollections(query string, args ...any) ([]*Collection, error) {
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("metadata: listing collections: %w", err)
	}
	defer rows.Close()

	var collections []*Collection
	for rows.Next() {
		var c Collection
		if err := rows.Scan(&c.ID, &c.TenantID, &c.DatabaseID, &c.Name, &c.Dimension, &c.Metric, &c.IndexType, &c.VectorCount, &c.CreatedAt); err != nil {
			return nil, fmt.Errorf("metadata: scanning collection: %w", err)
		}
		collections = append(collections, &c)
	}
	return collections, rows.Err()
}

// DeleteCollection removes a collection by ID.
func (s *SysDB) DeleteCollection(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.Exec("DELETE FROM collections WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("metadata: deleting collection %q: %w", id, err)
	}
	affected, _ := result.RowsAffected()
	if affected == 0 {
		return fmt.Errorf("metadata: collection %q not found", id)
	}

	// Clean up tombstone cache
	s.tombstoneMu.Lock()
	delete(s.tombstones, id)
	s.tombstoneMu.Unlock()

	return nil
}

// UpdateVectorCount updates the vector count for a collection.
func (s *SysDB) UpdateVectorCount(collectionID string, count int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec("UPDATE collections SET vector_count = ? WHERE id = ?", count, collectionID)
	if err != nil {
		return fmt.Errorf("metadata: updating vector count: %w", err)
	}
	return nil
}

// AddTombstone records a vector deletion.
func (s *SysDB) AddTombstone(collectionID string, vectorID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		"INSERT OR REPLACE INTO tombstones (collection_id, vector_id) VALUES (?, ?)",
		collectionID, vectorID,
	)
	if err != nil {
		return fmt.Errorf("metadata: adding tombstone: %w", err)
	}

	// Update in-memory set
	s.tombstoneMu.Lock()
	if s.tombstones[collectionID] == nil {
		s.tombstones[collectionID] = make(map[uint64]struct{})
	}
	s.tombstones[collectionID][vectorID] = struct{}{}
	s.tombstoneMu.Unlock()

	return nil
}

// IsTombstoned checks if a vector has been deleted (hot path, in-memory only).
func (s *SysDB) IsTombstoned(collectionID string, vectorID uint64) bool {
	s.tombstoneMu.RLock()
	defer s.tombstoneMu.RUnlock()
	if set, ok := s.tombstones[collectionID]; ok {
		_, deleted := set[vectorID]
		return deleted
	}
	return false
}

// loadTombstones populates the in-memory tombstone set from SQLite.
func (s *SysDB) loadTombstones() error {
	rows, err := s.db.Query("SELECT collection_id, vector_id FROM tombstones")
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var collID string
		var vecID uint64
		if err := rows.Scan(&collID, &vecID); err != nil {
			return err
		}
		if s.tombstones[collID] == nil {
			s.tombstones[collID] = make(map[uint64]struct{})
		}
		s.tombstones[collID][vecID] = struct{}{}
	}
	return rows.Err()
}

// Close closes the underlying SQLite database.
func (s *SysDB) Close() error {
	return s.db.Close()
}

// SaveVectorMetadata persists per-vector metadata to SQLite.
// Called during vector insertion to make metadata durable across restarts.
func (s *SysDB) SaveVectorMetadata(collectionID string, vectorID uint64, meta map[string]any) error {
	if len(meta) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for k, v := range meta {
		valStr := fmt.Sprintf("%v", v)
		_, err := s.db.Exec(
			`INSERT OR REPLACE INTO vector_metadata (collection_id, vector_id, meta_key, meta_value)
			 VALUES (?, ?, ?, ?)`,
			collectionID, vectorID, k, valStr,
		)
		if err != nil {
			return fmt.Errorf("metadata: saving vector metadata: %w", err)
		}
	}
	return nil
}

// LoadAllVectorMetadata loads all metadata for all vectors in a collection.
// Called during startup to repopulate the in-memory vectorMeta map.
func (s *SysDB) LoadAllVectorMetadata(collectionID string) (map[uint64]map[string]any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT vector_id, meta_key, meta_value FROM vector_metadata WHERE collection_id = ?",
		collectionID,
	)
	if err != nil {
		return nil, fmt.Errorf("metadata: loading vector metadata: %w", err)
	}
	defer rows.Close()

	result := make(map[uint64]map[string]any)
	for rows.Next() {
		var vecID uint64
		var key, value string
		if err := rows.Scan(&vecID, &key, &value); err != nil {
			return nil, fmt.Errorf("metadata: scanning vector metadata: %w", err)
		}
		if result[vecID] == nil {
			result[vecID] = make(map[string]any)
		}
		result[vecID][key] = value
	}
	return result, rows.Err()
}

// DeleteVectorMetadata removes all metadata for a specific vector.
func (s *SysDB) DeleteVectorMetadata(collectionID string, vectorID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		"DELETE FROM vector_metadata WHERE collection_id = ? AND vector_id = ?",
		collectionID, vectorID,
	)
	return err
}

// EnsureDefaults seeds the default tenant and database on first startup.
// Idempotent — safe to call on every startup.
func (s *SysDB) EnsureDefaults() (tenantID, databaseID string, err error) {
	// Check if default tenant already exists
	existing, err := s.GetTenantByName(DefaultTenantName)
	if err != nil {
		return "", "", fmt.Errorf("metadata: checking default tenant: %w", err)
	}
	if existing != nil {
		// Tenant exists — find the default database
		dbs, err := s.ListDatabases(existing.ID)
		if err != nil {
			return "", "", err
		}
		for _, db := range dbs {
			if db.Name == DefaultDatabaseName {
				return existing.ID, db.ID, nil
			}
		}
		// Tenant exists but no default database — create it
		db, err := s.CreateDatabase(existing.ID, DefaultDatabaseName)
		if err != nil {
			return "", "", err
		}
		return existing.ID, db.ID, nil
	}

	// Create default tenant (unlimited quotas)
	tenant, err := s.CreateTenant(DefaultTenantName, 0, 0)
	if err != nil {
		return "", "", fmt.Errorf("metadata: creating default tenant: %w", err)
	}

	// Create default database
	db, err := s.CreateDatabase(tenant.ID, DefaultDatabaseName)
	if err != nil {
		return "", "", fmt.Errorf("metadata: creating default database: %w", err)
	}

	slog.Info("seeded default tenant and database",
		"tenant_id", tenant.ID,
		"database_id", db.ID,
	)

	return tenant.ID, db.ID, nil
}

// DeleteCollectionScoped removes a collection only if it belongs to the given tenant.
func (s *SysDB) DeleteCollectionScoped(tenantID, collectionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.Exec("DELETE FROM collections WHERE id = ? AND tenant_id = ?", collectionID, tenantID)
	if err != nil {
		return fmt.Errorf("metadata: deleting collection %q: %w", collectionID, err)
	}
	affected, _ := result.RowsAffected()
	if affected == 0 {
		return fmt.Errorf("metadata: collection %q not found", collectionID)
	}

	// Clean up tombstone cache
	s.tombstoneMu.Lock()
	delete(s.tombstones, collectionID)
	s.tombstoneMu.Unlock()

	return nil
}

// CountCollectionsForTenant returns the total number of collections owned by a tenant.
func (s *SysDB) CountCollectionsForTenant(tenantID string) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM collections WHERE tenant_id = ?", tenantID).Scan(&count)
	return count, err
}
