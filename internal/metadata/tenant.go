// Package metadata — Multi-tenancy: Tenant → Database → Collection hierarchy.
//
// ChromaDB Lesson 10: "Tenant → Database → Collection is the right hierarchy."
//
// Every API call is scoped to:
//   Tenant (billing/quota boundary) →
//     Database (logical namespace) →
//       Collection (vector index)
//
// This module provides CRUD operations for tenants and databases,
// stored in the same SQLite SysDB alongside collections.
package metadata

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Tenant represents a top-level tenant (billing/isolation unit).
type Tenant struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	MaxDBs    int    `json:"max_databases"`     // 0 = unlimited
	MaxColls  int    `json:"max_collections"`   // 0 = unlimited per DB
	CreatedAt int64  `json:"created_at"`
}

// Database represents a logical namespace under a tenant.
type Database struct {
	ID        string `json:"id"`
	TenantID  string `json:"tenant_id"`
	Name      string `json:"name"`
	CreatedAt int64  `json:"created_at"`
}

// InitMultiTenancySchema creates the tenants and databases tables.
// Called once during SysDB initialization.
func (s *SysDB) InitMultiTenancySchema() error {
	schema := `
		CREATE TABLE IF NOT EXISTS tenants (
			id         TEXT PRIMARY KEY,
			name       TEXT NOT NULL UNIQUE,
			max_dbs    INTEGER NOT NULL DEFAULT 0,
			max_colls  INTEGER NOT NULL DEFAULT 0,
			created_at INTEGER NOT NULL DEFAULT (unixepoch())
		);

		CREATE TABLE IF NOT EXISTS databases (
			id         TEXT PRIMARY KEY,
			tenant_id  TEXT NOT NULL,
			name       TEXT NOT NULL,
			created_at INTEGER NOT NULL DEFAULT (unixepoch()),
			UNIQUE(tenant_id, name),
			FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
		);

		-- Add tenant_id and database_id columns to collections if not present.
		-- SQLite doesn't support ADD COLUMN IF NOT EXISTS,
		-- so we use a pragmatic approach: try adding, ignore error if already exists.
	`
	_, err := s.db.Exec(schema)
	return err
}

// ── Tenant CRUD ─────────────────────────────────────────────────────────────

// CreateTenant creates a new tenant.
func (s *SysDB) CreateTenant(name string, maxDBs, maxColls int) (*Tenant, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := uuid.New().String()
	now := time.Now().Unix()

	_, err := s.db.Exec(
		"INSERT INTO tenants (id, name, max_dbs, max_colls, created_at) VALUES (?, ?, ?, ?, ?)",
		id, name, maxDBs, maxColls, now,
	)
	if err != nil {
		return nil, fmt.Errorf("metadata: creating tenant %q: %w", name, err)
	}

	_, err = s.db.Exec("INSERT INTO tenant_quotas (tenant_id) VALUES (?)", id)
	if err != nil {
		return nil, fmt.Errorf("metadata: creating quotas for tenant %q: %w", name, err)
	}
	_, err = s.db.Exec("INSERT INTO tenant_usage (tenant_id) VALUES (?)", id)
	if err != nil {
		return nil, fmt.Errorf("metadata: creating usage for tenant %q: %w", name, err)
	}

	return &Tenant{
		ID:        id,
		Name:      name,
		MaxDBs:    maxDBs,
		MaxColls:  maxColls,
		CreatedAt: now,
	}, nil
}

// GetTenant retrieves a tenant by ID.
func (s *SysDB) GetTenant(id string) (*Tenant, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var t Tenant
	err := s.db.QueryRow(
		"SELECT id, name, max_dbs, max_colls, created_at FROM tenants WHERE id = ?", id,
	).Scan(&t.ID, &t.Name, &t.MaxDBs, &t.MaxColls, &t.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting tenant %q: %w", id, err)
	}
	return &t, nil
}

// GetTenantByName retrieves a tenant by name.
func (s *SysDB) GetTenantByName(name string) (*Tenant, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var t Tenant
	err := s.db.QueryRow(
		"SELECT id, name, max_dbs, max_colls, created_at FROM tenants WHERE name = ?", name,
	).Scan(&t.ID, &t.Name, &t.MaxDBs, &t.MaxColls, &t.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting tenant by name %q: %w", name, err)
	}
	return &t, nil
}

// ListTenants returns all tenants.
func (s *SysDB) ListTenants() ([]*Tenant, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query("SELECT id, name, max_dbs, max_colls, created_at FROM tenants ORDER BY created_at")
	if err != nil {
		return nil, fmt.Errorf("metadata: listing tenants: %w", err)
	}
	defer rows.Close()

	var tenants []*Tenant
	for rows.Next() {
		var t Tenant
		if err := rows.Scan(&t.ID, &t.Name, &t.MaxDBs, &t.MaxColls, &t.CreatedAt); err != nil {
			return nil, err
		}
		tenants = append(tenants, &t)
	}
	return tenants, rows.Err()
}

// DeleteTenant removes a tenant and cascades to databases and collections.
func (s *SysDB) DeleteTenant(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.Exec("DELETE FROM tenants WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("metadata: deleting tenant %q: %w", id, err)
	}
	affected, _ := result.RowsAffected()
	if affected == 0 {
		return fmt.Errorf("metadata: tenant %q not found", id)
	}
	return nil
}

// ── Database CRUD ───────────────────────────────────────────────────────────

// CreateDatabase creates a new database under a tenant.
func (s *SysDB) CreateDatabase(tenantID, name string) (*Database, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check tenant exists
	var exists bool
	err := s.db.QueryRow("SELECT 1 FROM tenants WHERE id = ?", tenantID).Scan(&exists)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("metadata: tenant %q not found", tenantID)
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: checking tenant: %w", err)
	}

	// Check quota
	var maxDBs int
	s.db.QueryRow("SELECT max_dbs FROM tenants WHERE id = ?", tenantID).Scan(&maxDBs)
	if maxDBs > 0 {
		var count int
		s.db.QueryRow("SELECT COUNT(*) FROM databases WHERE tenant_id = ?", tenantID).Scan(&count)
		if count >= maxDBs {
			return nil, fmt.Errorf("metadata: tenant %q has reached max databases limit (%d)", tenantID, maxDBs)
		}
	}

	id := uuid.New().String()
	now := time.Now().Unix()

	_, err = s.db.Exec(
		"INSERT INTO databases (id, tenant_id, name, created_at) VALUES (?, ?, ?, ?)",
		id, tenantID, name, now,
	)
	if err != nil {
		return nil, fmt.Errorf("metadata: creating database %q: %w", name, err)
	}

	return &Database{
		ID:        id,
		TenantID:  tenantID,
		Name:      name,
		CreatedAt: now,
	}, nil
}

// GetDatabase retrieves a database by ID.
func (s *SysDB) GetDatabase(id string) (*Database, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var d Database
	err := s.db.QueryRow(
		"SELECT id, tenant_id, name, created_at FROM databases WHERE id = ?", id,
	).Scan(&d.ID, &d.TenantID, &d.Name, &d.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting database %q: %w", id, err)
	}
	return &d, nil
}

// ListDatabases returns all databases for a tenant.
func (s *SysDB) ListDatabases(tenantID string) ([]*Database, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT id, tenant_id, name, created_at FROM databases WHERE tenant_id = ? ORDER BY created_at",
		tenantID,
	)
	if err != nil {
		return nil, fmt.Errorf("metadata: listing databases: %w", err)
	}
	defer rows.Close()

	var databases []*Database
	for rows.Next() {
		var d Database
		if err := rows.Scan(&d.ID, &d.TenantID, &d.Name, &d.CreatedAt); err != nil {
			return nil, err
		}
		databases = append(databases, &d)
	}
	return databases, rows.Err()
}

// DeleteDatabase removes a database.
func (s *SysDB) DeleteDatabase(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.Exec("DELETE FROM databases WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("metadata: deleting database %q: %w", id, err)
	}
	affected, _ := result.RowsAffected()
	if affected == 0 {
		return fmt.Errorf("metadata: database %q not found", id)
	}
	return nil
}

// ── Quotas and Usage ────────────────────────────────────────────────────────

// GetTenantQuotas retrieves the quotas for a tenant.
func (s *SysDB) GetTenantQuotas(tenantID string) (*TenantQuotas, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var q TenantQuotas
	err := s.db.QueryRow(
		"SELECT max_vectors, max_collections, max_qps, max_bytes FROM tenant_quotas WHERE tenant_id = ?",
		tenantID,
	).Scan(&q.MaxVectors, &q.MaxCollections, &q.MaxQPS, &q.MaxBytes)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting tenant quotas: %w", err)
	}
	return &q, nil
}

// GetTenantUsage retrieves the current usage for a tenant.
func (s *SysDB) GetTenantUsage(tenantID string) (*TenantUsage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var u TenantUsage
	err := s.db.QueryRow(
		"SELECT vector_count, collection_count, bytes_used, updated_at FROM tenant_usage WHERE tenant_id = ?",
		tenantID,
	).Scan(&u.VectorCount, &u.CollectionCount, &u.BytesUsed, &u.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("metadata: getting tenant usage: %w", err)
	}
	return &u, nil
}

// IncrementTenantVectorCount increments the vector count for a tenant.
func (s *SysDB) IncrementTenantVectorCount(tenantID string, delta int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		"UPDATE tenant_usage SET vector_count = vector_count + ?, updated_at = ? WHERE tenant_id = ?",
		delta, time.Now().Unix(), tenantID,
	)
	return err
}



// ── API Keys ────────────────────────────────────────────────────────────────

// ResolveAPIKey looks up an API key by hash and returns its tenant ID and role.
func (s *SysDB) ResolveAPIKey(keyHash string) (string, string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var tenantID, role string
	err := s.db.QueryRow(
		"SELECT tenant_id, role FROM api_keys WHERE key_hash = ? AND (expires_at IS NULL OR expires_at > ?)",
		keyHash, time.Now().Unix(),
	).Scan(&tenantID, &role)
	
	if err == sql.ErrNoRows {
		return "", "", fmt.Errorf("unknown or expired API key")
	}
	if err != nil {
		return "", "", fmt.Errorf("metadata: resolving API key: %w", err)
	}
	return tenantID, role, nil
}

// CreateAPIKey creates a new API key for a tenant.
func (s *SysDB) CreateAPIKey(keyHash, tenantID, role string, expiresAt *int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var exp sql.NullInt64
	if expiresAt != nil {
		exp.Int64 = *expiresAt
		exp.Valid = true
	}

	_, err := s.db.Exec(
		"INSERT INTO api_keys (key_hash, tenant_id, role, expires_at) VALUES (?, ?, ?, ?)",
		keyHash, tenantID, role, exp,
	)
	return err
}
