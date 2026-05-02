// Package storage implements write-ahead log (WAL) operations for the VectorDB.
// The WAL is the primary source of truth — writes are acknowledged only after
// they are durable in the WAL. The compaction worker materializes WAL entries
// into indexed segments asynchronously.
//
// ChromaDB Lesson 2: "The write-ahead log IS the database."
package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	_ "modernc.org/sqlite"
)

// WAL is the write-ahead log interface for durable vector storage.
// All insert and delete operations are written to the WAL before
// being applied to any in-memory index.
type WAL interface {
	// Append writes an operation to the log and returns its sequence ID.
	Append(op WALOp) (seqID uint64, err error)

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

// SQLiteWAL implements the WAL interface using SQLite as the backing store.
// SQLite in WAL mode provides:
//   - Concurrent readers don't block writers
//   - fsync guarantees durability
//   - Atomic transactions
type SQLiteWAL struct {
	mu sync.Mutex
	db *sql.DB
}

// NewSQLiteWAL creates a new SQLite-backed WAL at the given path.
// The database is configured with pragmas optimized for WAL workloads.
func NewSQLiteWAL(path string) (*SQLiteWAL, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("wal: opening sqlite at %q: %w", path, err)
	}

	// SQLite performance pragmas
	pragmas := []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
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

	// Create WAL entries table
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

	slog.Info("WAL initialized", "path", path)

	return &SQLiteWAL{db: db}, nil
}

// Append writes an operation to the WAL and returns its sequence ID.
func (w *SQLiteWAL) Append(op WALOp) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	var vectorData []byte
	if op.Vector != nil {
		var err error
		vectorData, err = json.Marshal(op.Vector)
		if err != nil {
			return 0, fmt.Errorf("wal: marshaling vector: %w", err)
		}
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

// ReadFrom returns all WAL entries with seqID > afterSeq.
func (w *SQLiteWAL) ReadFrom(afterSeq uint64) ([]WALEntry, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

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
			if err := json.Unmarshal(vectorData, &e.Op.Vector); err != nil {
				return nil, fmt.Errorf("wal: unmarshaling vector: %w", err)
			}
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

// Close closes the underlying SQLite database.
func (w *SQLiteWAL) Close() error {
	return w.db.Close()
}

// Compile-time interface check
var _ WAL = (*SQLiteWAL)(nil)
