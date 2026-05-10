package storage

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/POTATO-VE1/Magnitude/internal/index/flat"
)

func TestWALRecovery_EntriesSurviveRestart(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "recovery-wal.sqlite")

	const (
		numVectors = 100
		dim        = 16
	)

	// Phase 1: Write 100 vectors with known deterministic values
	wal1, err := NewSQLiteWAL(path, WithSyncMode("per-write"))
	if err != nil {
		t.Fatalf("NewSQLiteWAL (phase 1): %v", err)
	}

	originals := make(map[uint64][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		id := uint64(i + 1)
		vec := make([]float32, dim)
		for j := range vec {
			// Deterministic but varied values
			vec[j] = float32(i*dim+j) + 0.5
		}
		originals[id] = vec

		_, err := wal1.Append(WALOp{
			Type:         WALOpInsert,
			CollectionID: "test-col",
			ID:           id,
			Vector:       vec,
		})
		if err != nil {
			t.Fatalf("Append %d: %v", id, err)
		}
	}

	// Close WAL (simulating process shutdown)
	if err := wal1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Phase 2: Reopen WAL and verify all entries are durable
	wal2, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL (phase 2): %v", err)
	}
	defer wal2.Close()

	entries, err := wal2.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}

	if len(entries) != numVectors {
		t.Fatalf("expected %d entries after reopen, got %d", numVectors, len(entries))
	}

	// Verify every entry matches the original
	for _, e := range entries {
		orig, ok := originals[e.Op.ID]
		if !ok {
			t.Errorf("unexpected vector ID %d in WAL", e.Op.ID)
			continue
		}
		if len(e.Op.Vector) != dim {
			t.Errorf("ID %d: vector len = %d, want %d", e.Op.ID, len(e.Op.Vector), dim)
			continue
		}
		for j, v := range e.Op.Vector {
			if math.Float32bits(v) != math.Float32bits(orig[j]) {
				t.Errorf("ID %d, vec[%d]: got %f, want %f", e.Op.ID, j, v, orig[j])
				break
			}
		}
		if e.Op.CollectionID != "test-col" {
			t.Errorf("ID %d: collection = %q, want %q", e.Op.ID, e.Op.CollectionID, "test-col")
		}
	}
}

func TestWALRecovery_CollectionManagerReplay(t *testing.T) {
	// This test verifies that WAL entries can be replayed into a fresh
	// collection manager to reconstruct searchable state after a restart.
	dir := t.TempDir()
	walPath := filepath.Join(dir, "replay-wal.sqlite")

	const (
		numVectors = 50
		dim        = 8
	)

	// Phase 1: Write vectors to WAL
	wal1, err := NewSQLiteWAL(walPath, WithSyncMode("per-write"))
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < numVectors; i++ {
		id := uint64(i + 1)
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim+j) + 0.1
		}
		_, err := wal1.Append(WALOp{
			Type:         WALOpInsert,
			CollectionID: "replay-col",
			ID:           id,
			Vector:       vec,
		})
		if err != nil {
			t.Fatalf("Append %d: %v", id, err)
		}
	}

	if err := wal1.Close(); err != nil {
		t.Fatal(err)
	}

	// Phase 2: Reopen WAL, read entries, build a flat index from them
	// (simulating what replayWAL does in the collection manager)
	wal2, err := NewSQLiteWAL(walPath)
	if err != nil {
		t.Fatal(err)
	}
	defer wal2.Close()

	entries, err := wal2.ReadFrom(0)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != numVectors {
		t.Fatalf("expected %d entries, got %d", numVectors, len(entries))
	}

	// Reconstruct index from WAL entries (replay)
	idx, err := flat.NewFlatIndex(dim, "l2")
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range entries {
		if e.Op.Type == WALOpInsert {
			if err := idx.Insert(e.Op.ID, e.Op.Vector); err != nil {
				t.Fatalf("replay insert ID %d: %v", e.Op.ID, err)
			}
		}
	}

	if idx.Len() != numVectors {
		t.Fatalf("index len = %d, want %d", idx.Len(), numVectors)
	}

	// Phase 3: Search the reconstructed index to verify correctness
	ctx := context.Background()

	// Use the first vector as query — should find itself at distance 0
	queryVec := entries[0].Op.Vector
	results, err := idx.Search(ctx, queryVec, 5, 0)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("search returned zero results")
	}

	// The closest result should be the vector itself
	if results[0].ID != entries[0].Op.ID {
		t.Errorf("top-1 ID = %d, want %d (self-query)", results[0].ID, entries[0].Op.ID)
	}
	if results[0].Distance != 0.0 {
		t.Errorf("top-1 distance = %f, want 0.0 (self-query)", results[0].Distance)
	}
}
