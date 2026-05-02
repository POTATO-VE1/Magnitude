package metadata

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestSysDB(t *testing.T) *SysDB {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", "sysdb-test-*")
	require.NoError(t, err)
	t.Cleanup(func() { os.RemoveAll(tmpDir) })

	db, err := NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	require.NoError(t, db.InitMultiTenancySchema())
	t.Cleanup(func() { db.Close() })
	return db
}

// ── Tenant tests ────────────────────────────────────────────────────────────

func TestCreateTenant(t *testing.T) {
	db := newTestSysDB(t)
	tenant, err := db.CreateTenant("acme", 10, 50)
	require.NoError(t, err)
	assert.Equal(t, "acme", tenant.Name)
	assert.Equal(t, 10, tenant.MaxDBs)
	assert.Equal(t, 50, tenant.MaxColls)
	assert.NotEmpty(t, tenant.ID)
}

func TestCreateTenant_Duplicate(t *testing.T) {
	db := newTestSysDB(t)
	_, err := db.CreateTenant("acme", 0, 0)
	require.NoError(t, err)
	_, err = db.CreateTenant("acme", 0, 0)
	require.Error(t, err)
}

func TestGetTenant(t *testing.T) {
	db := newTestSysDB(t)
	created, _ := db.CreateTenant("acme", 5, 20)

	got, err := db.GetTenant(created.ID)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "acme", got.Name)
}

func TestGetTenant_NotFound(t *testing.T) {
	db := newTestSysDB(t)
	got, err := db.GetTenant("nonexistent")
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestGetTenantByName(t *testing.T) {
	db := newTestSysDB(t)
	db.CreateTenant("acme", 0, 0)

	got, err := db.GetTenantByName("acme")
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "acme", got.Name)
}

func TestListTenants(t *testing.T) {
	db := newTestSysDB(t)
	db.CreateTenant("acme", 0, 0)
	db.CreateTenant("globex", 0, 0)

	tenants, err := db.ListTenants()
	require.NoError(t, err)
	assert.Len(t, tenants, 2)
}

func TestDeleteTenant(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)

	require.NoError(t, db.DeleteTenant(tenant.ID))

	got, err := db.GetTenant(tenant.ID)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestDeleteTenant_NotFound(t *testing.T) {
	db := newTestSysDB(t)
	err := db.DeleteTenant("nonexistent")
	require.Error(t, err)
}

// ── Database tests ──────────────────────────────────────────────────────────

func TestCreateDatabase(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)

	database, err := db.CreateDatabase(tenant.ID, "production")
	require.NoError(t, err)
	assert.Equal(t, "production", database.Name)
	assert.Equal(t, tenant.ID, database.TenantID)
}

func TestCreateDatabase_TenantNotFound(t *testing.T) {
	db := newTestSysDB(t)
	_, err := db.CreateDatabase("nonexistent", "production")
	require.Error(t, err)
}

func TestCreateDatabase_DuplicateName(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)

	_, err := db.CreateDatabase(tenant.ID, "production")
	require.NoError(t, err)
	_, err = db.CreateDatabase(tenant.ID, "production")
	require.Error(t, err)
}

func TestCreateDatabase_QuotaEnforcement(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("limited", 2, 0)

	_, err := db.CreateDatabase(tenant.ID, "db1")
	require.NoError(t, err)
	_, err = db.CreateDatabase(tenant.ID, "db2")
	require.NoError(t, err)
	_, err = db.CreateDatabase(tenant.ID, "db3")
	require.Error(t, err, "should fail: quota exceeded")
}

func TestListDatabases(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)
	db.CreateDatabase(tenant.ID, "staging")
	db.CreateDatabase(tenant.ID, "production")

	dbs, err := db.ListDatabases(tenant.ID)
	require.NoError(t, err)
	assert.Len(t, dbs, 2)
}

func TestDeleteDatabase(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)
	database, _ := db.CreateDatabase(tenant.ID, "staging")

	require.NoError(t, db.DeleteDatabase(database.ID))

	dbs, _ := db.ListDatabases(tenant.ID)
	assert.Len(t, dbs, 0)
}

func TestDeleteTenant_CascadesDatabase(t *testing.T) {
	db := newTestSysDB(t)
	tenant, _ := db.CreateTenant("acme", 0, 0)
	db.CreateDatabase(tenant.ID, "production")

	require.NoError(t, db.DeleteTenant(tenant.ID))

	// Databases should be cascade-deleted
	dbs, err := db.ListDatabases(tenant.ID)
	require.NoError(t, err)
	assert.Len(t, dbs, 0)
}

// ── Collection metadata tests ───────────────────────────────────────────────

func TestSysDB_CollectionCRUD(t *testing.T) {
	db := newTestSysDB(t)

	col, err := db.CreateCollection("test", 128, "l2", "flat")
	require.NoError(t, err)
	assert.Equal(t, "test", col.Name)
	assert.Equal(t, 128, col.Dimension)

	got, err := db.GetCollection(col.ID)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "test", got.Name)

	byName, err := db.GetCollectionByName("test")
	require.NoError(t, err)
	require.NotNil(t, byName)

	cols, err := db.ListCollections()
	require.NoError(t, err)
	assert.Len(t, cols, 1)

	require.NoError(t, db.DeleteCollection(col.ID))
	got, _ = db.GetCollection(col.ID)
	assert.Nil(t, got)
}

func TestSysDB_Tombstones(t *testing.T) {
	db := newTestSysDB(t)
	col, _ := db.CreateCollection("test", 128, "l2", "flat")

	require.NoError(t, db.AddTombstone(col.ID, 42))
	assert.True(t, db.IsTombstoned(col.ID, 42))
	assert.False(t, db.IsTombstoned(col.ID, 43))
}
