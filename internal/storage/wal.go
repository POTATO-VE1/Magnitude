// Package storage implements write-ahead log (WAL) operations for the VectorDB.
// The WAL is the primary source of truth — writes are acknowledged only after
// they are durable in the WAL. The compaction worker materializes WAL entries
// into indexed segments asynchronously.
//
// ChromaDB Lesson 2: "The write-ahead log IS the database."
package storage

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// WAL is the write-ahead log interface for durable vector storage.
// All insert and delete operations are written to the WAL before
// being applied to any in-memory index.
type WAL interface {
	// Append writes an operation to the log and returns its sequence ID.
	Append(op WALOp) (seqID uint64, err error)

	// AppendBatch writes multiple operations in a single transaction.
	// Returns a sequence ID for each operation.
	AppendBatch(ops []WALOp) ([]uint64, error)

	// ReadFrom returns all WAL entries with seqID > afterSeq.
	// Used by the Query Executor to merge recent writes with the index.
	ReadFrom(afterSeq uint64) ([]WALEntry, error)

	// Truncate removes all entries with seqID <= upToSeq.
	// Called by the Compactor after segments are written to disk.
	Truncate(upToSeq uint64) error

	// Close flushes all pending writes and releases the WAL file.
	Close() error
}

// WALOpType identifies the type of WAL operation.
type WALOpType uint8

const (
	WALOpInsert WALOpType = iota + 1
	WALOpDelete
)

// WALOp represents a single operation to append to the WAL.
type WALOp struct {
	Type         WALOpType
	CollectionID string
	ID           uint64
	Vector       []float32
	Document     string
}

// WALEntry is a WAL operation read back from the log, with its sequence ID.
type WALEntry struct {
	SeqID uint64
	Op    WALOp
}

// WALOption configures the SQLiteWAL.
type WALOption func(*SQLiteWAL)

// WithSyncMode sets the WAL sync mode: "per-write", "delayed", or "none".
//   - "per-write": PRAGMA synchronous=FULL, every write is fsynced (safest, slowest)
//   - "delayed":   PRAGMA synchronous=NORMAL, syncs at checkpoint boundaries (balanced)
//   - "none":      PRAGMA synchronous=OFF, never syncs (fastest, risk data loss on crash)
//
// Default: "per-write" (preserves current behavior).
func WithSyncMode(mode string) WALOption {
	return func(w *SQLiteWAL) { w.syncMode = mode }
}

// WithSyncDelay sets the delay before a batched sync in "delayed" mode.
// Only effective when syncMode is "delayed". Default: 0 (immediate checkpoint).
func WithSyncDelay(d time.Duration) WALOption {
	return func(w *SQLiteWAL) { w.syncDelay = d }
}

// SQLiteWAL implements the WAL interface using SQLite as the backing store.
// SQLite in WAL mode provides:
//   - Concurrent readers don't block writers
//   - fsync guarantees durability
//   - Atomic transactions
type SQLiteWAL struct {
	mu          sync.RWMutex
	db          *sql.DB
	path        string
	syncMode    string
	syncDelay   time.Duration
	pendingSync bool
	closeCh     chan struct{}
	wg          sync.WaitGroup
}

// NewSQLiteWAL creates a new SQLite-backed WAL at the given path.
// The database is configured with pragmas optimized for WAL workloads.
func NewSQLiteWAL(path string, opts ...WALOption) (*SQLiteWAL, error) {
	w := &SQLiteWAL{
		path:     path,
		syncMode: "per-write",
		closeCh:  make(chan struct{}),
	}
	for _, opt := range opts {
		opt(w)
	}

	// Validate sync mode
	validModes := map[string]bool{"per-write": true, "delayed": true, "none": true}
	if !validModes[w.syncMode] {
		return nil, fmt.Errorf("wal: invalid sync mode %q, must be one of [per-write, delayed, none]", w.syncMode)
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("wal: opening sqlite at %q: %w", path, err)
	}

	// Set synchronous pragma based on sync mode
	syncPragma := "PRAGMA synchronous=FULL"
	switch w.syncMode {
	case "per-write":
		syncPragma = "PRAGMA synchronous=FULL"
	case "delayed":
		syncPragma = "PRAGMA synchronous=NORMAL"
	case "none":
		syncPragma = "PRAGMA synchronous=OFF"
	}

	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		syncPragma,
		"PRAGMA cache_size=-65536", // 64MB cache
		"PRAGMA busy_timeout=5000",
		"PRAGMA foreign_keys=ON",
	}
	for _, p := range pragmas {
		if _, err := db.Exec(p); err != nil {
			db.Close()
			return nil, fmt.Errorf("wal: setting pragma %q: %w", p, err)
		}
	}

	// Create WAL entries table with binary vector storage
	schema := `
		CREATE TABLE IF NOT EXISTS wal_entries (
			seq_id         INTEGER PRIMARY KEY AUTOINCREMENT,
			op_type        INTEGER NOT NULL,
			collection_id  TEXT    NOT NULL,
			vector_id      INTEGER NOT NULL,
			vector_data    BLOB,
			document       TEXT,
			created_at     INTEGER NOT NULL DEFAULT (unixepoch())
		);
		CREATE INDEX IF NOT EXISTS idx_wal_seq ON wal_entries(seq_id);
		CREATE INDEX IF NOT EXISTS idx_wal_collection ON wal_entries(collection_id);
	`
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("wal: creating schema: %w", err)
	}

	w.db = db

	slog.Info("WAL initialized",
		"path", path,
		"sync_mode", w.syncMode,
		"sync_delay", w.syncDelay,
	)

	// Start delayed sync goroutine if needed
	if w.syncMode == "delayed" && w.syncDelay > 0 {
		w.wg.Add(1)
		go w.delayedSyncLoop()
	}

	return w, nil
}

// Path returns the file path of the WAL database.
func (w *SQLiteWAL) Path() string {
	return w.path
}

// encodeVectorBinary encodes a float32 slice as little-endian bytes.
// This is ~10x faster than JSON encoding for float32 arrays.
func encodeVectorBinary(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// decodeVectorBinary decodes little-endian bytes back to a float32 slice.
func decodeVectorBinary(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 4
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return v
}

// Append writes an operation to the WAL and returns its sequence ID.
func (w *SQLiteWAL) Append(op WALOp) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	var vectorData []byte
	if op.Vector != nil {
		vectorData = encodeVectorBinary(op.Vector)
	}

	result, err := w.db.Exec(
		"INSERT INTO wal_entries (op_type, collection_id, vector_id, vector_data, document) VALUES (?, ?, ?, ?, ?)",
		uint8(op.Type), op.CollectionID, op.ID, vectorData, op.Document,
	)
	if err != nil {
		return 0, fmt.Errorf("wal: appending entry: %w", err)
	}

	seqID, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("wal: getting seq id: %w", err)
	}

	return uint64(seqID), nil
}

// AppendBatch writes multiple operations in a single transaction.
// This is significantly faster than individual Appends for bulk ingestion.
func (w *SQLiteWAL) AppendBatch(ops []WALOp) ([]uint64, error) {
	if len(ops) == 0 {
		return nil, nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	tx, err := w.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("wal: beginning transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(
		"INSERT INTO wal_entries (op_type, collection_id, vector_id, vector_data, document) VALUES (?, ?, ?, ?, ?)",
	)
	if err != nil {
		return nil, fmt.Errorf("wal: preparing statement: %w", err)
	}
	defer stmt.Close()

	seqIDs := make([]uint64, len(ops))
	for i, op := range ops {
		var vectorData []byte
		if op.Vector != nil {
			vectorData = encodeVectorBinary(op.Vector)
		}

		result, err := stmt.Exec(uint8(op.Type), op.CollectionID, op.ID, vectorData, op.Document)
		if err != nil {
			return nil, fmt.Errorf("wal: batch insert %d: %w", i, err)
		}

		seqID, err := result.LastInsertId()
		if err != nil {
			return nil, fmt.Errorf("wal: getting seq id for batch %d: %w", i, err)
		}
		seqIDs[i] = uint64(seqID)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("wal: committing batch: %w", err)
	}

	return seqIDs, nil
}

// ReadFrom returns all WAL entries with seqID > afterSeq.
func (w *SQLiteWAL) ReadFrom(afterSeq uint64) ([]WALEntry, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	rows, err := w.db.Query(
		"SELECT seq_id, op_type, collection_id, vector_id, vector_data, document FROM wal_entries WHERE seq_id > ? ORDER BY seq_id",
		afterSeq,
	)
	if err != nil {
		return nil, fmt.Errorf("wal: reading from seq %d: %w", afterSeq, err)
	}
	defer rows.Close()

	var entries []WALEntry
	for rows.Next() {
		var e WALEntry
		var opType uint8
		var vectorData []byte
		var document sql.NullString

		if err := rows.Scan(&e.SeqID, &opType, &e.Op.CollectionID, &e.Op.ID, &vectorData, &document); err != nil {
			return nil, fmt.Errorf("wal: scanning entry: %w", err)
		}
		e.Op.Type = WALOpType(opType)
		if document.Valid {
			e.Op.Document = document.String
		}

		if vectorData != nil {
			e.Op.Vector = decodeVectorBinary(vectorData)
		}

		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// Truncate removes all entries with seqID <= upToSeq.
func (w *SQLiteWAL) Truncate(upToSeq uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	_, err := w.db.Exec("DELETE FROM wal_entries WHERE seq_id <= ?", upToSeq)
	if err != nil {
		return fmt.Errorf("wal: truncating to seq %d: %w", upToSeq, err)
	}
	return nil
}

// Close flushes all pending writes and closes the underlying SQLite database.
func (w *SQLiteWAL) Close() error {
	// Signal delayed sync goroutine to stop
	select {
	case <-w.closeCh:
		// already closed
	default:
		close(w.closeCh)
	}

	w.wg.Wait()

	// In delayed mode, do a final checkpoint to ensure all data is durable
	if w.syncMode == "delayed" {
		w.mu.Lock()
		_, _ = w.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
		w.mu.Unlock()
	}

	return w.db.Close()
}

// delayedSyncLoop periodically triggers WAL checkpoints in "delayed" mode.
// This batches multiple writes into a single fsync for better throughput.
func (w *SQLiteWAL) delayedSyncLoop() {
	defer w.wg.Done()
	ticker := time.NewTicker(w.syncDelay)
	defer ticker.Stop()

	for {
		select {
		case <-w.closeCh:
			return
		case <-ticker.C:
			w.mu.Lock()
			// PASSIVE checkpoint: doesn't block readers, flushes as much as possible
			_, _ = w.db.Exec("PRAGMA wal_checkpoint(PASSIVE)")
			w.mu.Unlock()
		}
	}
}

// Compile-time interface check
var _ WAL = (*SQLiteWAL)(nil)
