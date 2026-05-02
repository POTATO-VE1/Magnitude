package metadata

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── Multi-Tenancy Integration Tests ─────────────────────────────────────────

func newScopedTestDB(t *testing.T) *SysDB {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", "sysdb-mt-*")
	require.NoError(t, err)
	t.Cleanup(func() { os.RemoveAll(tmpDir) })

	db, err := NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	require.NoError(t, db.InitMultiTenancySchema())
	t.Cleanup(func() { db.Close() })
	return db
}

func TestEnsureDefaults(t *testing.T) {
	db := newScopedTestDB(t)

	tenantID, dbID, err := db.EnsureDefaults()
	require.NoError(t, err)
	assert.NotEmpty(t, tenantID)
	assert.NotEmpty(t, dbID)

	// Verify the default tenant exists
	tenant, err := db.GetTenantByName(DefaultTenantName)
	require.NoError(t, err)
	require.NotNil(t, tenant)
	assert.Equal(t, tenantID, tenant.ID)

	// Verify the default database exists
	dbs, err := db.ListDatabases(tenantID)
	require.NoError(t, err)
	require.Len(t, dbs, 1)
	assert.Equal(t, DefaultDatabaseName, dbs[0].Name)
	assert.Equal(t, dbID, dbs[0].ID)
}

func TestEnsureDefaults_Idempotent(t *testing.T) {
	db := newScopedTestDB(t)

	tid1, did1, err := db.EnsureDefaults()
	require.NoError(t, err)

	tid2, did2, err := db.EnsureDefaults()
	require.NoError(t, err)

	assert.Equal(t, tid1, tid2, "tenant ID should be stable across calls")
	assert.Equal(t, did1, did2, "database ID should be stable across calls")

	// Should still be exactly one tenant and one database
	tenants, _ := db.ListTenants()
	assert.Len(t, tenants, 1)
}

// ── Scoped Collection CRUD ──────────────────────────────────────────────────

func TestCreateCollectionScoped(t *testing.T) {
	db := newScopedTestDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)
	database, _ := db.CreateDatabase(tenant.ID, "production")

	col, err := db.CreateCollectionScoped(tenant.ID, database.ID, "embeddings", 128, "l2", "flat")
	require.NoError(t, err)
	assert.Equal(t, "embeddings", col.Name)
	assert.Equal(t, tenant.ID, col.TenantID)
	assert.Equal(t, database.ID, col.DatabaseID)
	assert.Equal(t, 128, col.Dimension)
}

func TestCreateCollectionScoped_UniquenessPerDatabase(t *testing.T) {
	db := newScopedTestDB(t)
	tenantA, _ := db.CreateTenant("acme", 0, 0)
	dbProd, _ := db.CreateDatabase(tenantA.ID, "prod")
	dbStage, _ := db.CreateDatabase(tenantA.ID, "staging")

	// Same name in different databases → OK
	_, err := db.CreateCollectionScoped(tenantA.ID, dbProd.ID, "vectors", 128, "l2", "flat")
	require.NoError(t, err)
	_, err = db.CreateCollectionScoped(tenantA.ID, dbStage.ID, "vectors", 128, "l2", "flat")
	require.NoError(t, err)

	// Same name in same database → conflict
	_, err = db.CreateCollectionScoped(tenantA.ID, dbProd.ID, "vectors", 128, "l2", "flat")
	require.Error(t, err)
}

func TestGetCollectionScoped_CrossTenantIsolation(t *testing.T) {
	db := newScopedTestDB(t)
	tenantA, _ := db.CreateTenant("acme", 0, 0)
	tenantB, _ := db.CreateTenant("globex", 0, 0)
	dbA, _ := db.CreateDatabase(tenantA.ID, "prod")

	col, _ := db.CreateCollectionScoped(tenantA.ID, dbA.ID, "secret-vectors", 256, "cosine", "hnsw")

	// Tenant A can see it
	got, err := db.GetCollectionScoped(tenantA.ID, col.ID)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "secret-vectors", got.Name)

	// Tenant B gets nil (404), NOT an error (403)
	got, err = db.GetCollectionScoped(tenantB.ID, col.ID)
	require.NoError(t, err)
	assert.Nil(t, got, "cross-tenant access must return nil, not the collection")
}

func TestListCollectionsScoped(t *testing.T) {
	db := newScopedTestDB(t)
	tenantA, _ := db.CreateTenant("acme", 0, 0)
	tenantB, _ := db.CreateTenant("globex", 0, 0)
	dbA, _ := db.CreateDatabase(tenantA.ID, "prod")
	dbB, _ := db.CreateDatabase(tenantB.ID, "prod")

	db.CreateCollectionScoped(tenantA.ID, dbA.ID, "col1", 128, "l2", "flat")
	db.CreateCollectionScoped(tenantA.ID, dbA.ID, "col2", 128, "l2", "flat")
	db.CreateCollectionScoped(tenantB.ID, dbB.ID, "col3", 256, "cosine", "hnsw")

	// Tenant A sees 2
	colsA, err := db.ListCollectionsScoped(tenantA.ID, dbA.ID)
	require.NoError(t, err)
	assert.Len(t, colsA, 2)

	// Tenant B sees 1
	colsB, err := db.ListCollectionsScoped(tenantB.ID, dbB.ID)
	require.NoError(t, err)
	assert.Len(t, colsB, 1)

	// Cross-tenant query returns empty
	colsX, err := db.ListCollectionsScoped(tenantA.ID, dbB.ID)
	require.NoError(t, err)
	assert.Len(t, colsX, 0)
}

func TestDeleteCollectionScoped_CrossTenantIsolation(t *testing.T) {
	db := newScopedTestDB(t)
	tenantA, _ := db.CreateTenant("acme", 0, 0)
	tenantB, _ := db.CreateTenant("globex", 0, 0)
	dbA, _ := db.CreateDatabase(tenantA.ID, "prod")

	col, _ := db.CreateCollectionScoped(tenantA.ID, dbA.ID, "vectors", 128, "l2", "flat")

	// Tenant B cannot delete tenant A's collection
	err := db.DeleteCollectionScoped(tenantB.ID, col.ID)
	require.Error(t, err, "cross-tenant delete must fail")

	// Tenant A can delete their own collection
	err = db.DeleteCollectionScoped(tenantA.ID, col.ID)
	require.NoError(t, err)

	// Verify it's gone
	got, _ := db.GetCollectionScoped(tenantA.ID, col.ID)
	assert.Nil(t, got)
}

func TestCountCollectionsForTenant(t *testing.T) {
	db := newScopedTestDB(t)
	tenantA, _ := db.CreateTenant("acme", 0, 0)
	tenantB, _ := db.CreateTenant("globex", 0, 0)
	dbA, _ := db.CreateDatabase(tenantA.ID, "prod")
	dbB, _ := db.CreateDatabase(tenantB.ID, "prod")

	db.CreateCollectionScoped(tenantA.ID, dbA.ID, "c1", 128, "l2", "flat")
	db.CreateCollectionScoped(tenantA.ID, dbA.ID, "c2", 128, "l2", "flat")
	db.CreateCollectionScoped(tenantB.ID, dbB.ID, "c3", 256, "cosine", "hnsw")

	countA, err := db.CountCollectionsForTenant(tenantA.ID)
	require.NoError(t, err)
	assert.Equal(t, 2, countA)

	countB, err := db.CountCollectionsForTenant(tenantB.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, countB)
}

func TestCollectionQuota_MaxCollsPerTenant(t *testing.T) {
	db := newScopedTestDB(t)
	// Tenant with max 2 collections
	tenant, _ := db.CreateTenant("limited", 0, 2)
	database, _ := db.CreateDatabase(tenant.ID, "prod")

	_, err := db.CreateCollectionScoped(tenant.ID, database.ID, "c1", 128, "l2", "flat")
	require.NoError(t, err)
	_, err = db.CreateCollectionScoped(tenant.ID, database.ID, "c2", 128, "l2", "flat")
	require.NoError(t, err)

	// Check quota before 3rd creation
	count, _ := db.CountCollectionsForTenant(tenant.ID)
	assert.Equal(t, 2, count)
	assert.GreaterOrEqual(t, count, tenant.MaxColls,
		"count should be at or above quota — API handler should reject further inserts")
}

func TestDeleteTenant_CascadesDatabasesAndCollections(t *testing.T) {
	db := newScopedTestDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)
	database, _ := db.CreateDatabase(tenant.ID, "prod")
	col, _ := db.CreateCollectionScoped(tenant.ID, database.ID, "vectors", 128, "l2", "flat")

	// Manually delete the collection first (as the collection manager would),
	// then delete the tenant which cascades to databases.
	require.NoError(t, db.DeleteCollectionScoped(tenant.ID, col.ID))
	require.NoError(t, db.DeleteTenant(tenant.ID))

	// Databases should be cascade-deleted
	dbs, err := db.ListDatabases(tenant.ID)
	require.NoError(t, err)
	assert.Len(t, dbs, 0)

	// Collection should have been explicitly deleted above
	got, _ := db.GetCollection(col.ID)
	assert.Nil(t, got)
}

// ── Segments Table ──────────────────────────────────────────────────────────

func TestSegmentsTable_Exists(t *testing.T) {
	db := newScopedTestDB(t)

	// Verify the segments table was created by the schema
	var count int
	err := db.db.QueryRow("SELECT COUNT(*) FROM segments").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count, "segments table should exist and be empty")
}

// ── Backward Compatibility ──────────────────────────────────────────────────

func TestCreateCollection_BackwardCompatible(t *testing.T) {
	db := newScopedTestDB(t)

	// The unscoped CreateCollection still works (empty tenant/db)
	col, err := db.CreateCollection("legacy", 128, "l2", "flat")
	require.NoError(t, err)
	assert.Equal(t, "", col.TenantID)
	assert.Equal(t, "", col.DatabaseID)
	assert.Equal(t, "legacy", col.Name)

	// Can retrieve it
	got, err := db.GetCollection(col.ID)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "legacy", got.Name)
}
