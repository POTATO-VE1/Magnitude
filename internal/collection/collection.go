// Package collection orchestrates the full lifecycle of a vector collection:
// index operations, WAL writes, metadata updates, and compaction triggers.
//
// A Collection binds together:
//   - An Index (Flat/IVF/HNSW/SPANN) for vector search
//   - A WAL for durable write ordering
//   - A SysDB entry for metadata and segment tracking
//   - A Compactor for async index materialization
//
// ChromaDB Architecture mapping:
//
//	collection.Collection ≈ ChromaDB's Segment Manager + local HNSW segment.
//
// The Collection is the unit of concurrency: multiple goroutines may read
// concurrently; writes are serialized via a per-collection RWMutex.
package collection

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"sync"

	"github.com/POTATO-VE1/Magnitude/internal/events"
	"github.com/POTATO-VE1/Magnitude/internal/index"
	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
	"github.com/POTATO-VE1/Magnitude/internal/index/hnsw"
	"github.com/POTATO-VE1/Magnitude/internal/index/ivf"
	"github.com/POTATO-VE1/Magnitude/internal/index/spann"
	"github.com/POTATO-VE1/Magnitude/internal/index/sparse"
	"github.com/POTATO-VE1/Magnitude/internal/metadata"
	"github.com/POTATO-VE1/Magnitude/internal/search"
	"github.com/POTATO-VE1/Magnitude/internal/storage"
)

// Collection wraps an Index with WAL and metadata management.
type Collection struct {
	mu            sync.RWMutex
	meta          *metadata.Collection
	idx           index.Index
	invertedIndex *sparse.InvertedIndex
	wal           storage.WAL
	sysdb         *metadata.SysDB
	vectorMeta    map[uint64]metadata.VectorMetadata // per-vector metadata for hybrid search
}

// Manager manages all collections and routes API operations.
type Manager struct {
	mu             sync.RWMutex
	collections    map[string]*Collection // collection ID → Collection
	sysdb          *metadata.SysDB
	wal            storage.WAL
	flowBus        *events.FlowBus   // optional — nil disables event notifications
	dataDir        string            // root data directory for snapshot storage
	snapshotSeqIDs map[string]uint64 // collection ID → snapshot seqID for partial WAL replay
}

// SysDB returns the underlying system database for direct access by API handlers.
func (m *Manager) SysDB() *metadata.SysDB {
	return m.sysdb
}

// ManagerOption configures the Manager.
type ManagerOption func(*Manager)

// WithFlowBus sets the event bus for the Manager.
func WithFlowBus(bus *events.FlowBus) ManagerOption {
	return func(m *Manager) { m.flowBus = bus }
}

// WithDataDir sets the root data directory for snapshot storage.
func WithDataDir(dir string) ManagerOption {
	return func(m *Manager) { m.dataDir = dir }
}

// NewManager creates a new collection manager.
func NewManager(sysdb *metadata.SysDB, wal storage.WAL, opts ...ManagerOption) (*Manager, error) {
	mgr := &Manager{
		collections:    make(map[string]*Collection),
		sysdb:          sysdb,
		wal:            wal,
		snapshotSeqIDs: make(map[string]uint64),
	}
	for _, opt := range opts {
		opt(mgr)
	}

	// Load existing collections from SysDB
	cols, err := sysdb.ListCollections()
	if err != nil {
		return nil, fmt.Errorf("collection: loading existing collections: %w", err)
	}

	for _, meta := range cols {
		idx, err := createIndex(meta.Dimension, meta.Metric, meta.IndexType)
		if err != nil {
			slog.Error("failed to restore collection index",
				"collection", meta.Name,
				"error", err,
			)
			continue
		}

		// Try to load HNSW snapshot if available
		if meta.IndexType == "hnsw" && mgr.dataDir != "" {
			snapPath := mgr.snapshotPath(meta.ID)
			if hnswIdx, ok := idx.(*hnsw.HNSWIndex); ok {
				hnswIdx.SetSnapshotPath(snapPath)
				if loaded, seqID, err := hnsw.LoadHNSWFromSnapshot(snapPath); err == nil {
					if setErr := loaded.SetDistanceFunc(meta.Metric); setErr != nil {
						slog.Warn("failed to set distance func on loaded snapshot, using empty index",
							"collection", meta.Name,
							"error", setErr,
						)
					} else {
						loaded.SetSnapshotPath(snapPath)
						idx = loaded
						mgr.snapshotSeqIDs[meta.ID] = seqID
						slog.Info("loaded HNSW snapshot",
							"collection", meta.Name,
							"nodes", loaded.Len(),
							"seq_id", seqID,
						)
					}
				} else {
					slog.Debug("no HNSW snapshot found, will replay full WAL",
						"collection", meta.Name,
						"path", snapPath,
						"error", err,
					)
				}
			}
		}

		mgr.collections[meta.ID] = &Collection{
			meta:          meta,
			idx:           idx,
			invertedIndex: sparse.NewInvertedIndex(),
			wal:           wal,
			sysdb:         sysdb,
			vectorMeta:    make(map[uint64]metadata.VectorMetadata),
		}

		// Load persisted vector metadata from SQLite
		persistedMeta, err := sysdb.LoadAllVectorMetadata(meta.ID)
		if err != nil {
			slog.Warn("failed to load vector metadata",
				"collection", meta.Name,
				"error", err,
			)
		} else if len(persistedMeta) > 0 {
			for id, m := range persistedMeta {
				mgr.collections[meta.ID].vectorMeta[id] = metadata.VectorMetadata(m)
			}
			slog.Info("loaded vector metadata",
				"collection", meta.Name,
				"vectors_with_metadata", len(persistedMeta),
			)
		}

		slog.Info("restored collection",
			"id", meta.ID,
			"name", meta.Name,
			"dim", meta.Dimension,
			"metric", meta.Metric,
			"index_type", meta.IndexType,
		)
	}

	// Replay WAL to reconstruct in-memory index state after restart.
	// If a snapshot was loaded, only replay entries AFTER the snapshot seqID.
	if err := mgr.replayWAL(); err != nil {
		return nil, fmt.Errorf("collection: WAL replay failed: %w", err)
	}

	return mgr, nil
}

// snapshotPath returns the file path for the HNSW snapshot of a collection.
func (m *Manager) snapshotPath(collectionID string) string {
	return filepath.Join(m.dataDir, "snapshots", collectionID+".hsnp")
}

// CreateCollection creates a new collection.
func (m *Manager) CreateCollection(name string, dim int, metric, indexType string) (*metadata.Collection, error) {
	// Create metadata entry
	meta, err := m.sysdb.CreateCollection(name, dim, metric, indexType)
	if err != nil {
		return nil, fmt.Errorf("collection: creating %q: %w", name, err)
	}

	// Create index
	idx, err := createIndex(dim, metric, indexType)
	if err != nil {
		// Rollback metadata
		m.sysdb.DeleteCollection(meta.ID)
		return nil, fmt.Errorf("collection: creating index for %q: %w", name, err)
	}

	m.mu.Lock()
	m.collections[meta.ID] = &Collection{
		meta:          meta,
		idx:           idx,
		invertedIndex: sparse.NewInvertedIndex(),
		wal:           m.wal,
		sysdb:         m.sysdb,
		vectorMeta:    make(map[uint64]metadata.VectorMetadata),
	}
	m.mu.Unlock()

	slog.Info("collection created",
		"id", meta.ID,
		"name", meta.Name,
		"dim", dim,
		"metric", metric,
		"index_type", indexType,
	)

	if m.flowBus != nil {
		m.flowBus.Notify(events.EventCollectionCreated)
	}

	return meta, nil
}

// CreateCollectionScoped creates a new collection scoped to a tenant and database.
func (m *Manager) CreateCollectionScoped(tenantID, databaseID, name string, dim int, metric, indexType string) (*metadata.Collection, error) {
	meta, err := m.sysdb.CreateCollectionScoped(tenantID, databaseID, name, dim, metric, indexType)
	if err != nil {
		return nil, fmt.Errorf("collection: creating scoped %q: %w", name, err)
	}

	idx, err := createIndex(dim, metric, indexType)
	if err != nil {
		m.sysdb.DeleteCollectionScoped(tenantID, meta.ID)
		return nil, fmt.Errorf("collection: creating index for %q: %w", name, err)
	}

	m.mu.Lock()
	m.collections[meta.ID] = &Collection{
		meta:          meta,
		idx:           idx,
		invertedIndex: sparse.NewInvertedIndex(),
		wal:           m.wal,
		sysdb:         m.sysdb,
		vectorMeta:    make(map[uint64]metadata.VectorMetadata),
	}
	m.mu.Unlock()

	slog.Info("collection created (scoped)",
		"id", meta.ID,
		"tenant_id", tenantID,
		"database_id", databaseID,
		"name", meta.Name,
	)

	return meta, nil
}

// GetCollection returns a collection by ID.
func (m *Manager) GetCollection(id string) (*metadata.Collection, error) {
	m.mu.RLock()
	col, exists := m.collections[id]
	m.mu.RUnlock()

	if !exists {
		return nil, nil
	}

	col.mu.RLock()
	defer col.mu.RUnlock()
	// Return a copy to avoid data races
	metaCopy := *col.meta
	metaCopy.VectorCount = col.idx.Len()
	return &metaCopy, nil
}

// GetCollectionScoped returns a collection by ID, only if it belongs to the given tenant.
// Returns nil if not found or belongs to another tenant (cross-tenant isolation via 404).
func (m *Manager) GetCollectionScoped(tenantID, collectionID string) (*metadata.Collection, error) {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return nil, nil
	}

	col.mu.RLock()
	defer col.mu.RUnlock()

	if col.meta.TenantID != tenantID {
		return nil, nil // cross-tenant isolation: 404, not 403
	}

	metaCopy := *col.meta
	metaCopy.VectorCount = col.idx.Len()
	return &metaCopy, nil
}

// ListCollections returns all collections.
func (m *Manager) ListCollections() ([]*metadata.Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*metadata.Collection, 0, len(m.collections))
	for _, col := range m.collections {
		col.mu.RLock()
		metaCopy := *col.meta
		metaCopy.VectorCount = col.idx.Len()
		result = append(result, &metaCopy)
		col.mu.RUnlock()
	}
	return result, nil
}

// ListCollectionsScoped returns collections for a specific tenant and database.
func (m *Manager) ListCollectionsScoped(tenantID, databaseID string) ([]*metadata.Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []*metadata.Collection
	for _, col := range m.collections {
		col.mu.RLock()
		if col.meta.TenantID == tenantID && col.meta.DatabaseID == databaseID {
			metaCopy := *col.meta
			metaCopy.VectorCount = col.idx.Len()
			result = append(result, &metaCopy)
		}
		col.mu.RUnlock()
	}
	return result, nil
}

// DeleteCollection removes a collection.
func (m *Manager) DeleteCollection(id string) error {
	m.mu.Lock()
	col, exists := m.collections[id]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("collection %q not found", id)
	}
	delete(m.collections, id)
	m.mu.Unlock()

	// Close IVF background goroutine if applicable
	if ivfIdx, ok := col.idx.(*ivf.IVFIndex); ok {
		ivfIdx.Close()
	}

	return m.sysdb.DeleteCollection(id)
}

// DeleteCollectionScoped removes a collection only if it belongs to the given tenant.
func (m *Manager) DeleteCollectionScoped(tenantID, collectionID string) error {
	m.mu.Lock()
	col, exists := m.collections[collectionID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("collection %q not found", collectionID)
	}
	if col.meta.TenantID != tenantID {
		m.mu.Unlock()
		return fmt.Errorf("collection %q not found", collectionID) // 404, not 403
	}
	delete(m.collections, collectionID)
	m.mu.Unlock()

	if ivfIdx, ok := col.idx.(*ivf.IVFIndex); ok {
		ivfIdx.Close()
	}

	return m.sysdb.DeleteCollectionScoped(tenantID, collectionID)
}

// InsertVectors inserts a batch of vectors into a collection.
// Each vector is first written to the WAL, then inserted into the index.
// metadata is optional — pass nil for no per-vector metadata.
func (m *Manager) InsertVectors(ctx context.Context, collectionID string, ids []uint64, vectors [][]float32, meta []map[string]any) error {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("collection %q not found", collectionID)
	}

	col.mu.Lock()
	defer col.mu.Unlock()

	ops := make([]storage.WALOp, len(ids))
	docs := make([]string, len(ids))

	for i, id := range ids {
		var doc string
		if meta != nil && i < len(meta) && meta[i] != nil {
			if d, ok := meta[i]["document"].(string); ok {
				doc = d
			}
		}
		docs[i] = doc

		ops[i] = storage.WALOp{
			Type:         storage.WALOpInsert,
			CollectionID: collectionID,
			ID:           id,
			Vector:       vectors[i],
			Document:     doc,
		}
	}

	// WAL first (durability guarantee)
	_, err := col.wal.AppendBatch(ops)
	if err != nil {
		return fmt.Errorf("collection: WAL append batch failed: %w", err)
	}

	// Then index (in-memory, fast)
	for i, id := range ids {
		// Clear any existing tombstone if this is a re-insertion
		if err := m.sysdb.RemoveTombstone(collectionID, id); err != nil {
			slog.Warn("failed to clear tombstone", "vector_id", id, "error", err)
		}

		if err := col.idx.Insert(id, vectors[i]); err != nil {
			return fmt.Errorf("collection: index insert failed for vector %d: %w", id, err)
		}

		// Store metadata if provided
		if meta != nil && i < len(meta) && meta[i] != nil {
			col.vectorMeta[id] = metadata.VectorMetadata(meta[i])
			// Persist to SQLite for durability across restarts
			if err := col.sysdb.SaveVectorMetadata(collectionID, id, meta[i]); err != nil {
				slog.Warn("failed to persist vector metadata",
					"vector_id", id,
					"error", err,
				)
			}
		}

		if docs[i] != "" {
			col.invertedIndex.AddDocument(id, docs[i])
		}
	}

	// Update metadata
	col.meta.VectorCount = col.idx.Len()
	m.sysdb.UpdateVectorCount(collectionID, col.idx.Len())

	if m.flowBus != nil {
		m.flowBus.Notify(events.EventVectorInserted)
	}

	return nil
}

// SearchVectors performs a nearest-neighbor search on a collection.
// If filterMap is non-nil, results are post-filtered against per-vector metadata.
func (m *Manager) SearchVectors(ctx context.Context, collectionID string, query []float32, k, nprobe int, filterMap map[string]any) ([]index.SearchResult, error) {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %q not found", collectionID)
	}

	col.mu.RLock()
	defer col.mu.RUnlock()

	// Parse filter if provided
	var filter *metadata.Filter
	if len(filterMap) > 0 {
		var err error
		filter, err = metadata.ParseFilter(filterMap)
		if err != nil {
			return nil, fmt.Errorf("collection: invalid filter: %w", err)
		}
	}

	var results []index.SearchResult
	if filter == nil {
		var err error
		results, err = col.idx.Search(ctx, query, k, nprobe)
		if err != nil {
			return nil, err
		}
	} else {
		searchK := k * 4
		if searchK < 50 {
			searchK = 50
		}
		maxK := col.idx.Len()

		for {
			res, err := col.idx.Search(ctx, query, searchK, nprobe)
			if err != nil {
				return nil, err
			}

			filtered := res[:0]
			for _, r := range res {
				if filter.Match(col.vectorMeta[r.ID]) {
					filtered = append(filtered, r)
				}
			}

			if len(filtered) >= k || searchK >= maxK || len(res) < searchK {
				if len(filtered) > k {
					filtered = filtered[:k]
				}
				results = filtered
				break
			}

			searchK *= 2
			if searchK > maxK {
				searchK = maxK
			}
		}
	}

	return results, nil
}

// HybridSearch executes dense + sparse search in parallel, fuses results with RRF.
func (m *Manager) HybridSearch(
	ctx context.Context,
	collectionID string,
	queryVector []float32,
	queryText string,
	topK int,
	nprobe int,
	filterMap map[string]any,
) ([]index.SearchResult, error) {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %q not found", collectionID)
	}

	col.mu.RLock()
	defer col.mu.RUnlock()

	var filter *metadata.Filter
	if len(filterMap) > 0 {
		var err error
		filter, err = metadata.ParseFilter(filterMap)
		if err != nil {
			return nil, fmt.Errorf("collection: invalid filter: %w", err)
		}
	}

	var filteredDense []search.RankedResult
	var filteredSparse []search.RankedResult

	if filter == nil {
		var (
			denseResults  []index.SearchResult
			sparseResults []sparse.SearchResult
			denseErr      error
			wg            sync.WaitGroup
		)
		wg.Add(2)
		go func() {
			defer wg.Done()
			denseResults, denseErr = col.idx.Search(ctx, queryVector, topK, nprobe)
		}()
		go func() {
			defer wg.Done()
			sparseResults = col.invertedIndex.Search(queryText, topK)
		}()
		wg.Wait()
		if denseErr != nil {
			return nil, denseErr
		}
		for _, r := range denseResults {
			filteredDense = append(filteredDense, search.RankedResult{ID: r.ID, Score: r.Score})
		}
		for _, r := range sparseResults {
			filteredSparse = append(filteredSparse, search.RankedResult{ID: r.DocID, Score: r.Score})
		}
	} else {
		searchK := topK * 4
		if searchK < 50 {
			searchK = 50
		}
		maxK := col.idx.Len()

		for {
			var (
				denseResults  []index.SearchResult
				sparseResults []sparse.SearchResult
				denseErr      error
				wg            sync.WaitGroup
			)
			wg.Add(2)
			go func() {
				defer wg.Done()
				denseResults, denseErr = col.idx.Search(ctx, queryVector, searchK, nprobe)
			}()
			go func() {
				defer wg.Done()
				sparseResults = col.invertedIndex.Search(queryText, searchK)
			}()
			wg.Wait()
			if denseErr != nil {
				return nil, denseErr
			}

			filteredDense = nil
			for _, r := range denseResults {
				if filter.Match(col.vectorMeta[r.ID]) {
					filteredDense = append(filteredDense, search.RankedResult{ID: r.ID, Score: r.Score})
				}
			}

			filteredSparse = nil
			for _, r := range sparseResults {
				if filter.Match(col.vectorMeta[r.DocID]) {
					filteredSparse = append(filteredSparse, search.RankedResult{ID: r.DocID, Score: r.Score})
				}
			}

			if (len(filteredDense) >= topK && len(filteredSparse) >= topK) || searchK >= maxK || (len(denseResults) < searchK && len(sparseResults) < searchK) {
				break
			}
			searchK *= 2
			if searchK > maxK {
				searchK = maxK
			}
		}
	}

	// Fuse using RRF
	fused := search.RRF([][]search.RankedResult{filteredDense, filteredSparse}, 60)

	results := make([]index.SearchResult, 0, topK)
	for i, r := range fused {
		if i >= topK {
			break
		}
		results = append(results, index.SearchResult{ID: r.ID, Score: r.Score})
	}

	return results, nil
}

// DeleteVector deletes a single vector from a collection.
func (m *Manager) DeleteVector(ctx context.Context, collectionID string, vectorID uint64) error {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("collection %q not found", collectionID)
	}

	col.mu.Lock()
	defer col.mu.Unlock()

	// WAL first
	_, err := col.wal.Append(storage.WALOp{
		Type:         storage.WALOpDelete,
		CollectionID: collectionID,
		ID:           vectorID,
	})
	if err != nil {
		return fmt.Errorf("collection: WAL append failed for delete %d: %w", vectorID, err)
	}

	// Then index
	if err := col.idx.Delete(vectorID); err != nil {
		return err
	}
	col.invertedIndex.RemoveDocument(vectorID)

	// Remove persisted metadata
	col.sysdb.DeleteVectorMetadata(collectionID, vectorID)

	// Tombstone in metadata
	m.sysdb.AddTombstone(collectionID, vectorID)

	col.meta.VectorCount = col.idx.Len()
	m.sysdb.UpdateVectorCount(collectionID, col.idx.Len())

	return nil
}

// Flush flushes all collections' indexes.
// For HNSW collections, sets the current WAL seqID before snapshotting.
func (m *Manager) Flush() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Get current max WAL seqID for snapshot coordination
	var maxSeqID uint64
	if m.dataDir != "" {
		if entries, err := m.wal.ReadFrom(0); err == nil && len(entries) > 0 {
			maxSeqID = entries[len(entries)-1].SeqID
		}
	}

	for id, col := range m.collections {
		// Set snapshot seqID for HNSW indexes
		if hnswIdx, ok := col.idx.(*hnsw.HNSWIndex); ok {
			hnswIdx.SetSnapshotSeqID(maxSeqID)
		}
		if err := col.idx.Flush(); err != nil {
			slog.Error("flush failed", "collection", id, "error", err)
		}
	}
	return nil
}

// replayWAL replays WAL entries into the in-memory indexes.
// Called once during startup to recover state after a crash or restart.
// For collections with a loaded HNSW snapshot, only entries AFTER the
// snapshot's seqID are replayed. For all others, the full WAL is replayed.
func (m *Manager) replayWAL() error {
	// Find the minimum snapshot seqID across all collections.
	// Collections with snapshots only need entries after their snapshot seqID.
	// Collections without snapshots need all entries (from seq 0).
	var afterSeq uint64 = 0 // default: replay everything

	hasSnapshot := false
	for _, seqID := range m.snapshotSeqIDs {
		if !hasSnapshot || seqID < afterSeq {
			afterSeq = seqID
			hasSnapshot = true
		}
	}

	entries, err := m.wal.ReadFrom(afterSeq)
	if err != nil {
		return fmt.Errorf("reading WAL entries: %w", err)
	}

	if len(entries) == 0 {
		slog.Info("WAL replay: no entries to replay")
		return nil
	}

	slog.Info("WAL replay starting",
		"entries", len(entries),
		"after_seq", afterSeq,
		"has_snapshot", hasSnapshot,
	)

	var replayed, skipped, errors int
	for _, entry := range entries {
		col, exists := m.collections[entry.Op.CollectionID]
		if !exists {
			// Collection was deleted after this WAL entry was written — skip
			skipped++
			continue
		}

		// If this collection has a snapshot, skip entries already covered by it
		if snapSeq, ok := m.snapshotSeqIDs[entry.Op.CollectionID]; ok && entry.SeqID <= snapSeq {
			skipped++
			continue
		}

		switch entry.Op.Type {
		case storage.WALOpInsert:
			if err := col.idx.Insert(entry.Op.ID, entry.Op.Vector); err != nil {
				// Duplicate ID is expected if the insert was already compacted
				slog.Debug("WAL replay: insert skipped (likely already compacted)",
					"vector_id", entry.Op.ID,
					"error", err,
				)
				skipped++
				continue
			}
			if entry.Op.Document != "" {
				col.invertedIndex.AddDocument(entry.Op.ID, entry.Op.Document)
			}
			if entry.Op.CollectionID != "" {
				col.meta.VectorCount = col.idx.Len()
			}

		case storage.WALOpDelete:
			if err := col.idx.Delete(entry.Op.ID); err != nil {
				// Vector may have already been deleted — not an error
				slog.Debug("WAL replay: delete skipped",
					"vector_id", entry.Op.ID,
					"error", err,
				)
				skipped++
				continue
			}
			col.invertedIndex.RemoveDocument(entry.Op.ID)

		default:
			slog.Warn("WAL replay: unknown op type", "type", entry.Op.Type)
			errors++
			continue
		}
		replayed++
	}

	slog.Info("WAL replay complete",
		"replayed", replayed,
		"skipped", skipped,
		"errors", errors,
	)

	if m.flowBus != nil {
		m.flowBus.Notify(events.EventWALReplayComplete)
	}

	return nil
}

// GetVectorsMetadata returns the metadata for a list of vector IDs in a collection.
func (m *Manager) GetVectorsMetadata(collectionID string, ids []uint64) (map[uint64]map[string]any, error) {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %q not found", collectionID)
	}

	col.mu.RLock()
	defer col.mu.RUnlock()

	result := make(map[uint64]map[string]any, len(ids))
	for _, id := range ids {
		if meta, ok := col.vectorMeta[id]; ok && meta != nil {
			result[id] = meta
		}
	}
	return result, nil
}

// SnapshotCollection triggers an HNSW snapshot for a specific collection.
// Called after compaction to create a new recovery checkpoint.
func (m *Manager) SnapshotCollection(collectionID string) error {
	m.mu.RLock()
	col, exists := m.collections[collectionID]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("collection %q not found", collectionID)
	}

	hnswIdx, ok := col.idx.(*hnsw.HNSWIndex)
	if !ok {
		return nil // not an HNSW index, nothing to snapshot
	}

	// Get current max WAL seqID
	var maxSeqID uint64
	if entries, err := m.wal.ReadFrom(0); err == nil && len(entries) > 0 {
		maxSeqID = entries[len(entries)-1].SeqID
	}
	hnswIdx.SetSnapshotSeqID(maxSeqID)

	return hnswIdx.Flush()
}

// createIndex creates an index of the specified type.
func createIndex(dim int, metric, indexType string) (index.Index, error) {
	switch indexType {
	case "flat":
		return flat.NewFlatIndex(dim, metric)
	case "ivf":
		return ivf.NewIVFIndex(dim, 256, 5, metric, 0.10)
	case "hnsw":
		return hnsw.NewHNSWIndex(dim, 16, 200, 50, metric)
	case "spann":
		return spann.NewSPANNIndex(dim, 256, 5, metric)
	default:
		return nil, fmt.Errorf("unsupported index type: %q", indexType)
	}
}
