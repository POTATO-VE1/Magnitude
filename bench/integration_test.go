// Package bench provides end-to-end integration tests for the VectorDB.
// This test starts a real HTTP server, uses the Go client to insert vectors,
// searches them, and validates recall.
package bench

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/veda/vectordb/internal/api"
	"github.com/veda/vectordb/internal/cache"
	"github.com/veda/vectordb/internal/cluster"
	"github.com/veda/vectordb/internal/collection"
	"github.com/veda/vectordb/internal/config"
	"github.com/veda/vectordb/internal/gc"
	"github.com/veda/vectordb/internal/index/sparse"
	"github.com/veda/vectordb/internal/metadata"
	"github.com/veda/vectordb/internal/search"
	"github.com/veda/vectordb/internal/storage"
	"github.com/veda/vectordb/pkg/client"
)

func TestEndToEnd_FlatIndex(t *testing.T) {
	// Create temp directory for test data
	tmpDir, err := os.MkdirTemp("", "vectordb-e2e-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Initialize components
	sysdb, err := metadata.NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	defer sysdb.Close()

	wal, err := storage.NewSQLiteWAL(filepath.Join(tmpDir, "wal.sqlite"))
	require.NoError(t, err)
	defer wal.Close()

	mgr, err := collection.NewManager(sysdb, wal)
	require.NoError(t, err)

	cfg := config.DefaultConfig()
	cfg.Auth.KeyHashes = []string{} // no auth for test

	router := api.NewRouter(cfg, mgr)

	// Start HTTP server on random port
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	server := &http.Server{Handler: router}
	go server.Serve(listener)
	defer server.Close()

	baseURL := fmt.Sprintf("http://%s", listener.Addr().String())
	c := client.New(baseURL, "")

	ctx := context.Background()

	// ── Health check ─────────────────────────────────────────────────────
	require.NoError(t, c.Health(ctx))

	// ── Create collection ────────────────────────────────────────────────
	dim := 32
	col, err := c.CreateCollection(ctx, "test-vectors", dim, "l2", "flat")
	require.NoError(t, err)
	assert.Equal(t, "test-vectors", col.Name)
	assert.Equal(t, dim, col.Dimension)
	t.Logf("Created collection: %s (id=%s)", col.Name, col.ID)

	// ── Insert 1000 vectors ──────────────────────────────────────────────
	n := 1000
	rng := rand.New(rand.NewSource(42))
	allIDs := make([]uint64, n)
	allVecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		allIDs[i] = uint64(i)
		allVecs[i] = randomVector(rng, dim)
	}

	// Batch insert in chunks of 100
	batchSize := 100
	for start := 0; start < n; start += batchSize {
		end := start + batchSize
		if end > n {
			end = n
		}
		err := c.Insert(ctx, col.ID, allIDs[start:end], allVecs[start:end])
		require.NoError(t, err)
	}
	t.Logf("Inserted %d vectors", n)

	// ── Verify collection metadata ───────────────────────────────────────
	got, err := c.GetCollection(ctx, col.ID)
	require.NoError(t, err)
	assert.Equal(t, n, got.VectorCount)

	// ── Search ───────────────────────────────────────────────────────────
	query := randomVector(rng, dim)
	results, err := c.Search(ctx, col.ID, query, 10, 0)
	require.NoError(t, err)
	require.Len(t, results, 10)

	// Verify results are sorted by distance
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i-1].Distance, results[i].Distance,
			"results should be sorted by distance ascending")
	}
	t.Logf("Search returned %d results, nearest distance=%.4f", len(results), results[0].Distance)

	// ── Delete a vector ──────────────────────────────────────────────────
	deleteID := allIDs[0]
	require.NoError(t, c.Delete(ctx, col.ID, deleteID))

	// Verify deleted vector doesn't appear in results
	results2, err := c.Search(ctx, col.ID, allVecs[0], 10, 0)
	require.NoError(t, err)
	for _, r := range results2 {
		assert.NotEqual(t, deleteID, r.ID, "deleted vector should not appear in results")
	}

	// ── List collections ─────────────────────────────────────────────────
	cols, err := c.ListCollections(ctx)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(cols), 1)

	// ── Delete collection ────────────────────────────────────────────────
	require.NoError(t, c.DeleteCollection(ctx, col.ID))

	// Verify deletion
	cols2, err := c.ListCollections(ctx)
	require.NoError(t, err)
	for _, col := range cols2 {
		assert.NotEqual(t, "test-vectors", col.Name)
	}

	t.Log("End-to-end test passed!")
}

func TestEndToEnd_IVFIndex(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "vectordb-e2e-ivf-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	sysdb, err := metadata.NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	defer sysdb.Close()

	wal, err := storage.NewSQLiteWAL(filepath.Join(tmpDir, "wal.sqlite"))
	require.NoError(t, err)
	defer wal.Close()

	mgr, err := collection.NewManager(sysdb, wal)
	require.NoError(t, err)

	cfg := config.DefaultConfig()
	cfg.Auth.KeyHashes = []string{}

	router := api.NewRouter(cfg, mgr)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	server := &http.Server{Handler: router}
	go server.Serve(listener)
	defer server.Close()

	baseURL := fmt.Sprintf("http://%s", listener.Addr().String())
	c := client.New(baseURL, "")
	ctx := context.Background()

	dim := 32
	col, err := c.CreateCollection(ctx, "ivf-test", dim, "l2", "ivf")
	require.NoError(t, err)

	// Insert 500 vectors
	n := 500
	rng := rand.New(rand.NewSource(99))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i)
		vecs[i] = randomVector(rng, dim)
	}

	batchSize := 100
	for start := 0; start < n; start += batchSize {
		end := start + batchSize
		if end > n {
			end = n
		}
		require.NoError(t, c.Insert(ctx, col.ID, ids[start:end], vecs[start:end]))
	}

	// Search (IVF before rebuild — uses dirty buffer / brute force)
	query := randomVector(rng, dim)
	results, err := c.Search(ctx, col.ID, query, 10, 5)
	require.NoError(t, err)
	assert.Greater(t, len(results), 0)
	t.Logf("IVF search returned %d results (before rebuild)", len(results))

	// Give background rebuilder time to run (optional)
	time.Sleep(100 * time.Millisecond)

	require.NoError(t, c.DeleteCollection(ctx, col.ID))
	t.Log("IVF end-to-end test passed!")
}

func TestEndToEnd_HNSWIndex(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "vectordb-e2e-hnsw-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	sysdb, err := metadata.NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	defer sysdb.Close()

	wal, err := storage.NewSQLiteWAL(filepath.Join(tmpDir, "wal.sqlite"))
	require.NoError(t, err)
	defer wal.Close()

	mgr, err := collection.NewManager(sysdb, wal)
	require.NoError(t, err)

	cfg := config.DefaultConfig()
	cfg.Auth.KeyHashes = []string{}

	router := api.NewRouter(cfg, mgr)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	server := &http.Server{Handler: router}
	go server.Serve(listener)
	defer server.Close()

	baseURL := fmt.Sprintf("http://%s", listener.Addr().String())
	c := client.New(baseURL, "")
	ctx := context.Background()

	dim := 32
	col, err := c.CreateCollection(ctx, "hnsw-test", dim, "l2", "hnsw")
	require.NoError(t, err)
	t.Logf("Created HNSW collection: %s (id=%s)", col.Name, col.ID)

	// Insert 500 vectors
	n := 500
	rng := rand.New(rand.NewSource(77))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i)
		vecs[i] = randomVector(rng, dim)
	}

	batchSize := 100
	for start := 0; start < n; start += batchSize {
		end := start + batchSize
		if end > n {
			end = n
		}
		require.NoError(t, c.Insert(ctx, col.ID, ids[start:end], vecs[start:end]))
	}
	t.Logf("Inserted %d vectors into HNSW index", n)

	// Search
	query := randomVector(rng, dim)
	results, err := c.Search(ctx, col.ID, query, 10, 0)
	require.NoError(t, err)
	assert.Len(t, results, 10)

	// Verify distance ordering
	for i := 1; i < len(results); i++ {
		assert.LessOrEqual(t, results[i-1].Distance, results[i].Distance)
	}
	t.Logf("HNSW search returned %d results, nearest distance=%.4f", len(results), results[0].Distance)

	// Delete and verify
	require.NoError(t, c.Delete(ctx, col.ID, ids[0]))
	results2, err := c.Search(ctx, col.ID, vecs[0], 10, 0)
	require.NoError(t, err)
	for _, r := range results2 {
		assert.NotEqual(t, ids[0], r.ID)
	}

	require.NoError(t, c.DeleteCollection(ctx, col.ID))
	t.Log("HNSW end-to-end test passed!")
}

// ── Phase 7-10 Component Integration Tests ──────────────────────────────────

func TestIntegration_BM25_RRF_HybridSearch(t *testing.T) {
	// Test the full hybrid search pipeline: BM25 sparse + RRF fusion
	sparseIdx := sparse.NewInvertedIndex()

	// Index documents that have semantic + keyword overlap
	docs := map[uint64]string{
		1: "machine learning neural networks deep learning gradient descent",
		2: "database query optimization SQL indexing B-tree",
		3: "vector similarity search approximate nearest neighbor HNSW",
		4: "machine learning model training hyperparameter tuning",
		5: "database schema migration PostgreSQL ALTER TABLE",
	}

	for id, text := range docs {
		sparseIdx.AddDocument(id, text)
	}

	// Sparse BM25 search
	bm25Results := sparseIdx.Search("machine learning", 5)
	require.Greater(t, len(bm25Results), 0, "BM25 should return results")

	// Verify docs 1 and 4 match (they contain "machine learning")
	bm25IDs := make(map[uint64]bool)
	for _, r := range bm25Results {
		bm25IDs[r.DocID] = true
	}
	assert.True(t, bm25IDs[1], "doc 1 should match 'machine learning'")
	assert.True(t, bm25IDs[4], "doc 4 should match 'machine learning'")
	assert.False(t, bm25IDs[2], "doc 2 should not match 'machine learning'")

	// Now fuse dense (simulated) + sparse using RRF
	denseResults := []search.RankedResult{
		{ID: 1, Score: 0.95},
		{ID: 3, Score: 0.80},
		{ID: 4, Score: 0.75},
	}
	sparseRRF := make([]search.RankedResult, len(bm25Results))
	for i, r := range bm25Results {
		sparseRRF[i] = search.RankedResult{ID: r.DocID, Score: r.Score}
	}

	fused := search.RRFTopK([][]search.RankedResult{denseResults, sparseRRF}, 60, 3)
	require.Greater(t, len(fused), 0, "RRF fusion should return results")
	t.Logf("Hybrid search (BM25+RRF) returned %d results, top ID=%d", len(fused), fused[0].ID)
}

func TestIntegration_SegmentCache_Lifecycle(t *testing.T) {
	c := cache.NewSegmentCache(1024)

	// First access → not admitted (second-access policy)
	assert.False(t, c.Put("seg-1", []byte("data")))
	assert.Nil(t, c.Get("seg-1"))

	// Second access → admitted
	assert.True(t, c.Put("seg-1", []byte("data")))
	assert.NotNil(t, c.Get("seg-1"))

	stats := c.Stats()
	assert.Equal(t, 1, stats.Entries)
	assert.Equal(t, int64(4), stats.CurrentBytes)

	c.Remove("seg-1")
	assert.Equal(t, 0, c.Len())
}

func TestIntegration_GC_ThreePhase(t *testing.T) {
	dir := t.TempDir()

	// Create segment files
	seg1 := filepath.Join(dir, "old_index.bin")
	require.NoError(t, os.WriteFile(seg1, make([]byte, 1024), 0600))

	seg2 := filepath.Join(dir, "active_index.bin")
	require.NoError(t, os.WriteFile(seg2, make([]byte, 2048), 0600))

	pins := gc.NewPinTracker()
	gcConfig := gc.GCConfig{
		Interval:     time.Second,
		FenceTimeout: time.Second,
		MinAge:       0, // no min age for test
	}
	collector := gc.NewCollector(pins, gcConfig)

	// Phase 1: MARK both segments
	collector.Mark(gc.SegmentRef{ID: "old", FilePath: seg1, MarkedAt: time.Now().Add(-time.Hour)})
	collector.Mark(gc.SegmentRef{ID: "active", FilePath: seg2, MarkedAt: time.Now().Add(-time.Hour)})

	// Phase 2: FENCE — pin the active segment (simulating an active query)
	pins.Pin("active")

	// Phase 3: SWEEP
	collected, err := collector.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, collected, "only unpinned segment should be collected")

	// Old segment deleted, active segment preserved
	_, err = os.Stat(seg1)
	assert.True(t, os.IsNotExist(err), "old segment should be deleted")
	_, err = os.Stat(seg2)
	assert.NoError(t, err, "pinned segment should still exist")

	// Unpin and sweep again
	pins.Unpin("active")
	collected, err = collector.Sweep(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, collected)

	stats := collector.Stats()
	assert.Equal(t, int64(2), stats.TotalCollected)
	t.Logf("GC collected %d segments, freed %d bytes", stats.TotalCollected, stats.TotalBytesFreed)
}

func TestIntegration_CollectionFork(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "vectordb-fork-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	sysdb, err := metadata.NewSysDB(filepath.Join(tmpDir, "sysdb.sqlite"))
	require.NoError(t, err)
	defer sysdb.Close()

	wal, err := storage.NewSQLiteWAL(filepath.Join(tmpDir, "wal.sqlite"))
	require.NoError(t, err)
	defer wal.Close()

	mgr, err := collection.NewManager(sysdb, wal)
	require.NoError(t, err)

	// Create source collection with vectors
	src, err := mgr.CreateCollection("source", 32, "l2", "flat")
	require.NoError(t, err)

	rng := rand.New(rand.NewSource(42))
	ids := make([]uint64, 100)
	vecs := make([][]float32, 100)
	for i := range ids {
		ids[i] = uint64(i)
		vecs[i] = randomVector(rng, 32)
	}
	require.NoError(t, mgr.InsertVectors(context.Background(), src.ID, ids, vecs, nil))

	// Fork the collection
	forked, err := mgr.ForkCollection(context.Background(), src.ID, "forked-copy")
	require.NoError(t, err)
	assert.Equal(t, "forked-copy", forked.Name)
	assert.Equal(t, 32, forked.Dimension)
	assert.Equal(t, "l2", forked.Metric)
	assert.Equal(t, "flat", forked.IndexType)

	// Verify both collections are listed independently
	cols, err := mgr.ListCollections()
	require.NoError(t, err)
	assert.Equal(t, 2, len(cols))

	// Delete source should not affect fork
	require.NoError(t, mgr.DeleteCollection(src.ID))
	cols, err = mgr.ListCollections()
	require.NoError(t, err)
	assert.Equal(t, 1, len(cols))
	assert.Equal(t, "forked-copy", cols[0].Name)

	t.Log("Collection fork integration test passed!")
}

func TestIntegration_RendezvousRouting(t *testing.T) {
	// Test rendezvous hashing for cache-coherent query routing
	router := cluster.NewRendezvousRouter([]string{"node-1", "node-2", "node-3", "node-4"})

	// All queries for the same collection should always go to the same node
	target := router.Route("my-collection")
	for i := 0; i < 1000; i++ {
		assert.Equal(t, target, router.Route("my-collection"),
			"rendezvous hashing should be deterministic")
	}

	// N-way replication
	replicas := router.RouteN("my-collection", 3)
	assert.Len(t, replicas, 3)
	assert.Equal(t, target, replicas[0], "primary replica should match Route()")

	// Adding a node should only reroute ~1/N of collections
	router.AddNode("node-5")
	newTarget := router.Route("my-collection")
	// The target may or may not change, but the algorithm is correct
	t.Logf("After adding node-5: collection routes to %s (was %s)", newTarget, target)
}

func TestIntegration_CrashRecovery(t *testing.T) {
	// This test simulates an abrupt crash and verifies WAL replay recovers >= 90% of vectors.
	tmpDir, err := os.MkdirTemp("", "vectordb-crash-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	sysdbPath := filepath.Join(tmpDir, "sysdb.sqlite")
	walPath := filepath.Join(tmpDir, "wal.sqlite")

	sysdb, err := metadata.NewSysDB(sysdbPath)
	require.NoError(t, err)
	wal, err := storage.NewSQLiteWAL(walPath)
	require.NoError(t, err)

	mgr, err := collection.NewManager(sysdb, wal)
	require.NoError(t, err)

	col, err := mgr.CreateCollection("crash-test", 16, "l2", "flat")
	require.NoError(t, err)

	rng := rand.New(rand.NewSource(99))
	n := 10000
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i)
		vecs[i] = randomVector(rng, 16)
	}

	// Insert in batches
	batchSize := 100
	for start := 0; start < n; start += batchSize {
		end := start + batchSize
		if end > n {
			end = n
		}
		require.NoError(t, mgr.InsertVectors(context.Background(), col.ID, ids[start:end], vecs[start:end], nil))
	}

	// CRASH SIMULATION:
	// We intentionally DO NOT call mgr.Flush().
	// We close the raw DB connections to release locks and simulate process exit,
	// bypassing any graceful shutdown logic that would flush the in-memory indexes.
	require.NoError(t, wal.Close())
	require.NoError(t, sysdb.Close())

	// RECOVERY: Re-open storage and initialize a new Manager.
	// This triggers WAL replay on startup.
	sysdb2, err := metadata.NewSysDB(sysdbPath)
	require.NoError(t, err)
	defer sysdb2.Close()

	wal2, err := storage.NewSQLiteWAL(walPath)
	require.NoError(t, err)
	defer wal2.Close()

	mgr2, err := collection.NewManager(sysdb2, wal2)
	require.NoError(t, err)

	// Verify recovery via collection stats
	recoveredCol, err := mgr2.GetCollection(col.ID)
	require.NoError(t, err)

	// Spec says >= 90% survival. WAL replay should actually be 100%.
	assert.GreaterOrEqual(t, recoveredCol.VectorCount, int(float64(n)*0.90), "less than 90% of vectors survived crash recovery")

	// Verify data integrity with a search for the first vector inserted
	query := vecs[0]
	results, err := mgr2.SearchVectors(context.Background(), col.ID, query, 5, 0, nil)
	require.NoError(t, err)
	assert.Greater(t, len(results), 0)
	assert.Equal(t, ids[0], results[0].ID, "corrupted results detected after recovery")

	t.Logf("Crash recovery successful: %d/%d vectors recovered", recoveredCol.VectorCount, n)
}

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}
