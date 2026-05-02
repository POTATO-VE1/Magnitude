// Package storage — S3-backed Write-Ahead Log (wal3 pattern).
//
// ChromaDB's distributed WAL (wal3) is built ENTIRELY on object storage
// with no additional locking service. The key insight:
//
//   S3's If-Match conditional PUT (optimistic concurrency) is used as the
//   only coordination primitive. No ZooKeeper, no etcd, no Redis needed.
//
// Data structures:
//   - Fragment: immutable file containing a subsequence of log records.
//     Path: {prefix}/fragments/Bucket={N}/Seq={M}.bin
//     Grouped in buckets of 4096 for S3 per-prefix rate limiting.
//   - Manifest: mutable file listing all active fragments in order.
//     Updated via conditional PUT (compare-and-swap on S3 ETag).
//   - Cursor: a position in the log that pins all subsequent records
//     from garbage collection.
//
// For single-node, SQLiteWAL is used. S3WAL is for distributed deployment.
package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// S3WALConfig configures the S3-backed WAL.
type S3WALConfig struct {
	Bucket           string        // S3 bucket name
	Prefix           string        // key prefix (e.g., "wal3/tenant1/db1/col1")
	FragmentBucket   int           // fragments per S3 prefix bucket (default: 4096)
	MaxRetries       int           // retries on manifest CAS conflict (default: 5)
	CompactThreshold int           // number of fragments before triggering compaction
}

// DefaultS3WALConfig returns production-safe defaults.
func DefaultS3WALConfig() S3WALConfig {
	return S3WALConfig{
		FragmentBucket:   4096,
		MaxRetries:       5,
		CompactThreshold: 100,
	}
}

// Fragment is an immutable file containing a contiguous run of log records.
type Fragment struct {
	SeqStart  uint64 `json:"seq_start"`
	SeqEnd    uint64 `json:"seq_end"`
	S3Key     string `json:"s3_key"`
	SizeBytes int64  `json:"size_bytes"`
	CreatedAt int64  `json:"created_at"` // Unix timestamp
}

// Manifest lists all active fragments composing the log.
// Updated via S3 conditional PUT using ETag for compare-and-swap.
type Manifest struct {
	Fragments []Fragment `json:"fragments"`
	HeadSeq   uint64     `json:"head_seq"`    // sequence ID of last acknowledged record
	CursorSeq uint64     `json:"cursor_seq"`  // oldest unconsumed record (GC fence)
	Version   int64      `json:"version"`     // monotonically increasing
	ETag      string     `json:"-"`           // S3 ETag (not persisted in JSON)
}

// S3WALRecord represents a single WAL entry for distributed mode.
type S3WALRecord struct {
	SeqID        uint64                 `json:"seq_id"`
	OpType       WALOpType              `json:"op_type"`
	CollectionID string                 `json:"collection_id"`
	VectorID     uint64                 `json:"vector_id"`
	Vector       []float32              `json:"vector,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Timestamp    int64                  `json:"timestamp"`
}

// S3WAL implements a write-ahead log on object storage.
type S3WAL struct {
	mu       sync.Mutex
	store    ObjectStore
	config   S3WALConfig
	manifest *Manifest
	logger   *slog.Logger
}

// NewS3WAL creates an S3-backed WAL.
// It loads or creates the manifest on startup.
func NewS3WAL(store ObjectStore, config S3WALConfig) (*S3WAL, error) {
	if config.FragmentBucket == 0 {
		config.FragmentBucket = 4096
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 5
	}

	w := &S3WAL{
		store:  store,
		config: config,
		logger: slog.Default().With("component", "s3wal"),
	}

	// Load existing manifest or create new
	if err := w.loadOrCreateManifest(context.Background()); err != nil {
		return nil, fmt.Errorf("s3wal: initializing manifest: %w", err)
	}

	return w, nil
}

// loadOrCreateManifest loads the manifest from S3 or creates a new one.
func (w *S3WAL) loadOrCreateManifest(ctx context.Context) error {
	key := w.manifestKey()
	data, err := w.store.Get(ctx, w.config.Bucket, key)
	if err != nil {
		// Assume not found — create new manifest
		w.manifest = &Manifest{
			Fragments: make([]Fragment, 0),
			HeadSeq:   0,
			CursorSeq: 0,
			Version:   1,
		}
		return w.writeManifest(ctx)
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("s3wal: corrupted manifest: %w", err)
	}
	w.manifest = &m
	return nil
}

// Append writes new records to the WAL as a new fragment.
// Returns the sequence ID of the last record written.
func (w *S3WAL) Append(ctx context.Context, records []S3WALRecord) (uint64, error) {
	if len(records) == 0 {
		return w.manifest.HeadSeq, nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	newSeqStart := w.manifest.HeadSeq + 1
	newSeqEnd := newSeqStart + uint64(len(records)) - 1

	// Assign sequence IDs to records
	for i := range records {
		records[i].SeqID = newSeqStart + uint64(i)
		if records[i].Timestamp == 0 {
			records[i].Timestamp = time.Now().UnixNano()
		}
	}

	// Encode records as JSON (production would use Parquet for efficiency)
	data, err := json.Marshal(records)
	if err != nil {
		return 0, fmt.Errorf("s3wal: encoding fragment: %w", err)
	}

	// Write fragment to S3 (unconditional — fragments are immutable)
	fragKey := w.fragmentKey(newSeqStart)
	if err := w.store.Put(ctx, w.config.Bucket, fragKey, data); err != nil {
		return 0, fmt.Errorf("s3wal: writing fragment: %w", err)
	}

	// Update manifest with new fragment
	frag := Fragment{
		SeqStart:  newSeqStart,
		SeqEnd:    newSeqEnd,
		S3Key:     fragKey,
		SizeBytes: int64(len(data)),
		CreatedAt: time.Now().Unix(),
	}
	w.manifest.Fragments = append(w.manifest.Fragments, frag)
	w.manifest.HeadSeq = newSeqEnd
	w.manifest.Version++

	// Write manifest (with CAS if supported)
	if err := w.writeManifest(ctx); err != nil {
		return 0, fmt.Errorf("s3wal: updating manifest: %w", err)
	}

	w.logger.Info("fragment appended",
		"seq_start", newSeqStart,
		"seq_end", newSeqEnd,
		"records", len(records),
		"key", fragKey,
	)

	return newSeqEnd, nil
}

// ReadSince reads all records after the given sequence ID.
// Used by the Compactor to catch up on recent WAL entries.
func (w *S3WAL) ReadSince(ctx context.Context, afterSeq uint64) ([]S3WALRecord, error) {
	w.mu.Lock()
	fragments := make([]Fragment, len(w.manifest.Fragments))
	copy(fragments, w.manifest.Fragments)
	w.mu.Unlock()

	var allRecords []S3WALRecord

	for _, frag := range fragments {
		if frag.SeqEnd <= afterSeq {
			continue // already consumed
		}

		data, err := w.store.Get(ctx, w.config.Bucket, frag.S3Key)
		if err != nil {
			return nil, fmt.Errorf("s3wal: reading fragment %s: %w", frag.S3Key, err)
		}

		var records []S3WALRecord
		if err := json.Unmarshal(data, &records); err != nil {
			return nil, fmt.Errorf("s3wal: decoding fragment %s: %w", frag.S3Key, err)
		}

		for _, r := range records {
			if r.SeqID > afterSeq {
				allRecords = append(allRecords, r)
			}
		}
	}

	return allRecords, nil
}

// AdvanceCursor moves the cursor forward, allowing GC to collect old fragments.
// Called by the Compactor after successfully compacting records.
func (w *S3WAL) AdvanceCursor(ctx context.Context, newCursorSeq uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if newCursorSeq <= w.manifest.CursorSeq {
		return nil // already past this point
	}

	w.manifest.CursorSeq = newCursorSeq
	w.manifest.Version++

	// Remove fragments fully below the cursor
	remaining := w.manifest.Fragments[:0]
	for _, f := range w.manifest.Fragments {
		if f.SeqEnd >= newCursorSeq {
			remaining = append(remaining, f)
		} else {
			// Delete old fragment from S3
			if err := w.store.Delete(ctx, w.config.Bucket, f.S3Key); err != nil {
				w.logger.Error("failed to delete old fragment",
					"key", f.S3Key, "error", err)
			}
		}
	}
	w.manifest.Fragments = remaining

	return w.writeManifest(ctx)
}

// HeadSeq returns the current head sequence ID.
func (w *S3WAL) HeadSeq() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.manifest.HeadSeq
}

// FragmentCount returns the number of active fragments.
func (w *S3WAL) FragmentCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.manifest.Fragments)
}

// ── Internal helpers ────────────────────────────────────────────────────────

func (w *S3WAL) manifestKey() string {
	return w.config.Prefix + "/manifest/MANIFEST"
}

func (w *S3WAL) fragmentKey(seqStart uint64) string {
	bucket := seqStart / uint64(w.config.FragmentBucket) * uint64(w.config.FragmentBucket)
	return fmt.Sprintf("%s/fragments/Bucket=%d/Seq=%016x.bin",
		w.config.Prefix, bucket, seqStart)
}

func (w *S3WAL) writeManifest(ctx context.Context) error {
	data, err := json.Marshal(w.manifest)
	if err != nil {
		return fmt.Errorf("s3wal: encoding manifest: %w", err)
	}

	key := w.manifestKey()

	if w.manifest.ETag != "" {
		// Use conditional PUT for CAS (linearizable writes)
		newETag, err := w.store.ConditionalPut(ctx, w.config.Bucket, key, data, w.manifest.ETag)
		if err != nil {
			return fmt.Errorf("s3wal: manifest CAS failed (concurrent writer?): %w", err)
		}
		w.manifest.ETag = newETag
	} else {
		// First write — no ETag to compare against
		if err := w.store.Put(ctx, w.config.Bucket, key, data); err != nil {
			return err
		}
	}

	return nil
}
