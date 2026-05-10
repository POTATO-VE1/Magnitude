package collection

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/metadata"
	"github.com/POTATO-VE1/Magnitude/internal/storage"
)

func setupTestManager(t *testing.T) *Manager {
	t.Helper()

	dir := t.TempDir()
	sysdb, err := metadata.NewSysDB(filepath.Join(dir, "sysdb.sqlite"))
	if err != nil {
		t.Fatalf("NewSysDB: %v", err)
	}
	t.Cleanup(func() { sysdb.Close() })

	if err := sysdb.InitMultiTenancySchema(); err != nil {
		t.Fatalf("InitMultiTenancySchema: %v", err)
	}

	if _, _, err := sysdb.EnsureDefaults(); err != nil {
		t.Fatalf("EnsureDefaults: %v", err)
	}

	wal, err := storage.NewSQLiteWAL(filepath.Join(dir, "wal.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	t.Cleanup(func() { wal.Close() })

	mgr, err := NewManager(sysdb, wal)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	return mgr
}

func TestTenantIsolation_CrossTenantAccess(t *testing.T) {
	mgr := setupTestManager(t)
	ctx := context.Background()

	const (
		tenantA = "tenant-a"
		tenantB = "tenant-b"
		dbA     = "db-a"
		dbB     = "db-b"
		dim     = 8
	)

	// Create tenants and databases via SysDB
	sysdb := mgr.SysDB()
	ta, err := sysdb.CreateTenant(tenantA, 0, 0)
	if err != nil {
		t.Fatalf("CreateTenant A: %v", err)
	}
	tb, err := sysdb.CreateTenant(tenantB, 0, 0)
	if err != nil {
		t.Fatalf("CreateTenant B: %v", err)
	}
	dba, err := sysdb.CreateDatabase(ta.ID, dbA)
	if err != nil {
		t.Fatalf("CreateDatabase A: %v", err)
	}
	dbb, err := sysdb.CreateDatabase(tb.ID, dbB)
	if err != nil {
		t.Fatalf("CreateDatabase B: %v", err)
	}

	// Tenant A creates a collection
	colA, err := mgr.CreateCollectionScoped(ta.ID, dba.ID, "secret-data", dim, "l2", "flat")
	if err != nil {
		t.Fatalf("CreateCollectionScoped: %v", err)
	}

	// Insert vectors into tenant A's collection
	ids := []uint64{1, 2, 3}
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	}
	if err := mgr.InsertVectors(ctx, colA.ID, ids, vectors, nil); err != nil {
		t.Fatalf("InsertVectors: %v", err)
	}

	// Tenant B should NOT see tenant A's collection via GetCollectionScoped
	got, err := mgr.GetCollectionScoped(tb.ID, colA.ID)
	if err != nil {
		t.Fatalf("GetCollectionScoped: %v", err)
	}
	if got != nil {
		t.Errorf("tenant B got tenant A's collection (expected nil): %+v", got)
	}

	// Tenant B should NOT see tenant A's collection via ListCollectionsScoped
	listB, err := mgr.ListCollectionsScoped(tb.ID, dbb.ID)
	if err != nil {
		t.Fatalf("ListCollectionsScoped: %v", err)
	}
	if len(listB) != 0 {
		t.Errorf("tenant B sees %d collections (expected 0)", len(listB))
	}

	// Tenant A should see its own collection
	listA, err := mgr.ListCollectionsScoped(ta.ID, dba.ID)
	if err != nil {
		t.Fatalf("ListCollectionsScoped A: %v", err)
	}
	if len(listA) != 1 {
		t.Errorf("tenant A sees %d collections (expected 1)", len(listA))
	}
}

func TestTenantIsolation_CrossTenantInsert(t *testing.T) {
	mgr := setupTestManager(t)
	ctx := context.Background()

	sysdb := mgr.SysDB()
	ta, err := sysdb.CreateTenant("tenant-a", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	tb, err := sysdb.CreateTenant("tenant-b", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	dba, err := sysdb.CreateDatabase(ta.ID, "db-a")
	if err != nil {
		t.Fatal(err)
	}

	colA, err := mgr.CreateCollectionScoped(ta.ID, dba.ID, "a-collection", 4, "l2", "flat")
	if err != nil {
		t.Fatal(err)
	}

	// Tenant B tries to insert into tenant A's collection.
	// InsertVectors uses collectionID directly (not tenant-scoped), but the
	// collection belongs to tenant A. The Manager allows this at the raw level
	// because InsertVectors doesn't check tenant ownership — the isolation is
	// enforced at the API layer via GetCollectionScoped. Verify the raw insert
	// works (proving isolation must be enforced above the Manager).
	err = mgr.InsertVectors(ctx, colA.ID, []uint64{99}, [][]float32{{1, 2, 3, 4}}, nil)
	if err != nil {
		t.Fatalf("InsertVectors (raw): %v", err)
	}

	// But tenant B cannot discover the collection via scoped access
	got, err := mgr.GetCollectionScoped(tb.ID, colA.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Error("tenant B discovered tenant A's collection")
	}
}

func TestTenantIsolation_CrossTenantSearch(t *testing.T) {
	mgr := setupTestManager(t)
	ctx := context.Background()

	sysdb := mgr.SysDB()
	ta, err := sysdb.CreateTenant("tenant-a", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	tb, err := sysdb.CreateTenant("tenant-b", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	dba, err := sysdb.CreateDatabase(ta.ID, "db-a")
	if err != nil {
		t.Fatal(err)
	}

	colA, err := mgr.CreateCollectionScoped(ta.ID, dba.ID, "searchable", 4, "l2", "flat")
	if err != nil {
		t.Fatal(err)
	}

	// Insert vectors into tenant A's collection
	err = mgr.InsertVectors(ctx, colA.ID,
		[]uint64{1, 2, 3},
		[][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	// Tenant B cannot discover the collection, so cannot search it
	got, _ := mgr.GetCollectionScoped(tb.ID, colA.ID)
	if got != nil {
		t.Fatal("tenant B should not see tenant A's collection")
	}

	// Tenant A can search its own collection
	results, err := mgr.SearchVectors(ctx, colA.ID, []float32{1, 0, 0, 0}, 3, 0, nil)
	if err != nil {
		t.Fatalf("SearchVectors (tenant A): %v", err)
	}
	if len(results) == 0 {
		t.Error("tenant A got zero search results")
	}

	// The raw SearchVectors doesn't enforce tenant scoping (it uses collectionID).
	// But since tenant B can't discover colA.ID, it can never reach SearchVectors.
	// This is the isolation guarantee: no collection ID leakage across tenants.
}
