package storage

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestWAL_DelayedSync_AppendAndRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal.sqlite")

	wal, err := NewSQLiteWAL(path,
		WithSyncMode("delayed"),
		WithSyncDelay(50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	// Append some entries
	for i := uint64(1); i <= 5; i++ {
		seqID, err := wal.Append(WALOp{
			Type:         WALOpInsert,
			CollectionID: "col-1",
			ID:           i,
			Vector:       []float32{float32(i), float32(i + 1)},
		})
		if err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
		if seqID != i {
			t.Errorf("seqID = %d, want %d", seqID, i)
		}
	}

	// Wait for delayed sync to flush
	time.Sleep(100 * time.Millisecond)

	// Read back
	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 5 {
		t.Fatalf("expected 5 entries, got %d", len(entries))
	}

	// Verify vector data round-trips correctly
	for i, e := range entries {
		expectedID := uint64(i + 1)
		if e.Op.ID != expectedID {
			t.Errorf("entry %d: ID = %d, want %d", i, e.Op.ID, expectedID)
		}
		if len(e.Op.Vector) != 2 {
			t.Errorf("entry %d: vector len = %d, want 2", i, len(e.Op.Vector))
		}
	}
}

func TestWAL_NoneSync_AppendAndRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-nosync.sqlite")

	wal, err := NewSQLiteWAL(path, WithSyncMode("none"))
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	_, err = wal.Append(WALOp{
		Type:         WALOpInsert,
		CollectionID: "col-1",
		ID:           1,
		Vector:       []float32{1.0, 2.0, 3.0},
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].Op.Vector[0] != 1.0 || entries[0].Op.Vector[2] != 3.0 {
		t.Errorf("vector mismatch: %v", entries[0].Op.Vector)
	}
}

func TestWAL_PerWriteSync_AppendAndRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-perwrite.sqlite")

	wal, err := NewSQLiteWAL(path, WithSyncMode("per-write"))
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	_, err = wal.Append(WALOp{
		Type:         WALOpInsert,
		CollectionID: "col-1",
		ID:           42,
		Vector:       []float32{3.14, 2.72},
		Document:     "test document",
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].Op.ID != 42 {
		t.Errorf("ID = %d, want 42", entries[0].Op.ID)
	}
	if entries[0].Op.Document != "test document" {
		t.Errorf("Document = %q, want %q", entries[0].Op.Document, "test document")
	}
}

func TestWAL_Close_FlushesPending(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-close.sqlite")

	wal, err := NewSQLiteWAL(path,
		WithSyncMode("delayed"),
		WithSyncDelay(1*time.Second), // very long delay
	)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}

	// Append and immediately close
	_, err = wal.Append(WALOp{
		Type:         WALOpInsert,
		CollectionID: "col-1",
		ID:           1,
		Vector:       []float32{1.0},
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	if err := wal.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen and verify data is durable
	wal2, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL reopen: %v", err)
	}
	defer wal2.Close()

	entries, err := wal2.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry after reopen, got %d", len(entries))
	}
}

func TestWAL_DefaultSyncMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-default.sqlite")

	// No options — should default to per-write (current behavior)
	wal, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	_, err = wal.Append(WALOp{
		Type:         WALOpInsert,
		CollectionID: "col-1",
		ID:           1,
		Vector:       []float32{1.0},
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
}

func TestWAL_BinaryVectorEncoding(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-binary.sqlite")

	wal, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	// Insert a vector with specific values
	original := []float32{1.5, -2.5, 3.14159, 0.0, 100.0}
	_, err = wal.Append(WALOp{
		Type:         WALOpInsert,
		CollectionID: "col-1",
		ID:           1,
		Vector:       original,
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	// Verify exact float32 values round-trip
	got := entries[0].Op.Vector
	if len(got) != len(original) {
		t.Fatalf("vector length = %d, want %d", len(got), len(original))
	}
	for i := range original {
		if got[i] != original[i] {
			t.Errorf("vector[%d] = %f, want %f", i, got[i], original[i])
		}
	}
}

func TestWAL_DeleteOp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-delete.sqlite")

	wal, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	// Insert then delete
	_, err = wal.Append(WALOp{Type: WALOpInsert, CollectionID: "col-1", ID: 1, Vector: []float32{1.0}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = wal.Append(WALOp{Type: WALOpDelete, CollectionID: "col-1", ID: 1})
	if err != nil {
		t.Fatal(err)
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].Op.Type != WALOpInsert {
		t.Errorf("entry 0 type = %d, want INSERT", entries[0].Op.Type)
	}
	if entries[1].Op.Type != WALOpDelete {
		t.Errorf("entry 1 type = %d, want DELETE", entries[1].Op.Type)
	}
}

func TestWAL_InvalidSyncMode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-invalid.sqlite")

	_, err := NewSQLiteWAL(path, WithSyncMode("invalid"))
	if err == nil {
		t.Fatal("expected error for invalid sync mode")
	}
}

func TestWAL_Path(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-path.sqlite")

	wal, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	if wal.Path() != path {
		t.Errorf("Path() = %q, want %q", wal.Path(), path)
	}
}

func TestWAL_AppendBatch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-wal-batch.sqlite")

	wal, err := NewSQLiteWAL(path)
	if err != nil {
		t.Fatalf("NewSQLiteWAL: %v", err)
	}
	defer wal.Close()

	ops := []WALOp{
		{Type: WALOpInsert, CollectionID: "col-1", ID: 1, Vector: []float32{1.0}},
		{Type: WALOpInsert, CollectionID: "col-1", ID: 2, Vector: []float32{2.0}},
		{Type: WALOpInsert, CollectionID: "col-1", ID: 3, Vector: []float32{3.0}},
	}

	seqIDs, err := wal.AppendBatch(ops)
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	if len(seqIDs) != 3 {
		t.Fatalf("expected 3 seqIDs, got %d", len(seqIDs))
	}

	entries, err := wal.ReadFrom(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}
}

func BenchmarkWAL_Append_Single(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench-wal.sqlite")

	wal, err := NewSQLiteWAL(path, WithSyncMode("none"))
	if err != nil {
		b.Fatal(err)
	}
	defer wal.Close()

	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := wal.Append(WALOp{
			Type:         WALOpInsert,
			CollectionID: "col-1",
			ID:           uint64(i),
			Vector:       vector,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWAL_Append_Batch100(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench-wal-batch.sqlite")

	wal, err := NewSQLiteWAL(path, WithSyncMode("none"))
	if err != nil {
		b.Fatal(err)
	}
	defer wal.Close()

	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}

	ops := make([]WALOp, 100)
	for i := range ops {
		ops[i] = WALOp{
			Type:         WALOpInsert,
			CollectionID: "col-1",
			ID:           uint64(i),
			Vector:       vector,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := wal.AppendBatch(ops)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func init() {
	// Ensure temp dir exists for tests
	os.MkdirAll(os.TempDir(), 0o755)
}
